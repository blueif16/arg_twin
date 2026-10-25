"""Run a brain over the playbook tools, replay a recorded run, or summarize one.

    .venv/bin/python -m harness.brains.run --brain fake   --task 'Tour the skills'
    .venv/bin/python -m harness.brains.run --brain gemini --task '...' [--thinking-level high] [--yes]
    .venv/bin/python -m harness.brains.run --replay <run-id> [--pace 1] [--replay-tools live]
    .venv/bin/python -m harness.brains.run --summary <run-id>
    .venv/bin/python -m harness.brains.run --print-system [--brain claude]   # prompt for a stand-in
    .venv/bin/python -m harness.brains.run --import-calls claude:<run-id>    # record a CLI stand-in run

Records go to runs/loop/<run-id>/ (see harness/brains/record.py). --brain gemini needs
GEMINI_API_KEY or GOOGLE_API_KEY, and asks for confirmation (it spends money) unless --yes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

from harness.brains.fake import FakeInteractionsServer
from harness.brains.gemini import (ARGON, BrainConfig, Budget, GeminiBrain, files_uploader,
                                   live_tools, make_client)
from harness.brains.prompt import build_system_instruction
from harness.brains.record import (LOOP_DIR, RecordedTools, RecordingTransport, ReplayTransport,
                                   RunRecorder, import_cli_calls, load_run, new_run_id, summarize)

OFFLINE_KEY = "offline-no-key"  # placeholder for fake/replay clients; never reaches a server


def print_turn(turn: dict) -> None:
    u = turn.get("usage") or {}
    calls = turn.get("tool_calls") or []
    head = (f"[t{turn['turn']}] {turn.get('latency_s', 0):.2f}s  in {u.get('input', 0):,} "
            f"(cached {u.get('cached', 0):,})  out {u.get('output', 0):,}  think {u.get('thought', 0):,}")
    print(head + (f" | {', '.join(c['name'] for c in calls)}" if calls else ""), flush=True)
    for th in turn.get("thoughts") or []:
        print(f"     thought: {th[:200]}")
    for c in calls:
        if "skipped" in c:
            print(f"     -> {c['name']}: skipped ({c['skipped']})")
            continue
        res = json.dumps(c.get("result"), default=str)
        media = [m["path"] + ("" if m.get("mode") != "skipped" else " (skipped)") for m in c.get("media") or []]
        print(f"     -> {c['name']}({json.dumps(c['args'])[:120]}): {'ERROR ' if c['is_error'] else ''}"
              f"{res[:160]}{'...' if len(res) > 160 else ''}  [{c['dt_s']:.2f}s]"
              + (f"  media: {media}" if media else ""))
    if turn.get("error"):
        print(f"     error: {turn['error'][:500]}")
    if turn.get("text") and not calls:
        print(f"     text: {turn['text'][:500]}")


CORE_TOOLS = ["list_skills", "read_skill", "read_playbook_file", "log_note", "log_decision", "record_event",
              "status_summary"]


def skill_tools(skills: list[str]) -> list[str]:
    """Tool allowlist for a run: the registered tools named in each skill's "## Tools" section, plus CORE_TOOLS.
    The playbook stays the source of truth; the docs advise 10-20 active tools per call."""
    import re
    from harness import registry
    tools = registry.load_all()
    names = [t for t in CORE_TOOLS if t in tools]
    for skill in skills:
        text = (registry.REPO / "playbook" / "skills" / skill / "SKILL.md").read_text()
        section = re.search(r"^## Tools\n(.*?)(?=^## )", text, re.S | re.M)
        for n in re.findall(r"`([a-z_0-9]+)`", section.group(1) if section else ""):
            if n in tools and n not in names:
                names.append(n)
    return names


def config_from_args(a: argparse.Namespace) -> BrainConfig:
    budget = Budget(max_model_calls=a.max_model_calls, max_tool_calls=a.max_tool_calls,
                    max_tokens=a.max_tokens, max_usd=a.max_usd, max_wall_s=a.max_wall_s)
    return BrainConfig(model=a.model, thinking_level=a.thinking_level,
                       thinking_summaries=None if a.thinking_summaries == "none" else a.thinking_summaries,
                       max_output_tokens=a.max_output_tokens, background=True if a.background else None,
                       http_timeout_s=a.http_timeout_s,
                       tool_names=([t for t in a.tools.split(",") if t] if a.tools else
                                   skill_tools([s for s in a.skills.split(",") if s]) if a.skills else None),
                       caller=a.caller or ("argon" if a.brain == "gemini" else a.brain),
                       media_resolution=a.media_resolution, video_placement=a.video_placement,
                       video_fps=a.video_fps,
                       budget=budget)


def run_brain(brain: str, task: str, cfg: BrainConfig, run_id: str | None = None,
              root: Path = LOOP_DIR, record: bool = True, server: FakeInteractionsServer | None = None,
              on_turn=None, sleep=time.sleep) -> dict:
    """brain 'gemini' talks to the live API; 'fake' to the scripted offline server."""
    run_id = run_id or new_run_id(brain)
    if brain == "gemini":
        inner: httpx.BaseTransport = httpx.HTTPTransport()  # the SDK does its own retries
        api_key = None  # the SDK reads GEMINI_API_KEY / GOOGLE_API_KEY
    elif brain == "fake":
        server = server or FakeInteractionsServer()
        inner, api_key = server.transport(), OFFLINE_KEY
        sleep = (lambda s: None) if sleep is time.sleep else sleep
    else:
        raise ValueError(f"unknown brain {brain!r}")
    transport = RecordingTransport(inner)
    client = make_client(transport, api_key, cfg.http_timeout_s)
    recorder = RunRecorder(run_id, root, enabled=record)
    uploader = files_uploader(client, sleep) if brain == "gemini" else None
    gb = GeminiBrain(cfg, client, recorder, transport, brain_name="argon" if brain == "gemini" else "fake",
                     tool_executor=live_tools(cfg.caller), uploader=uploader, sleep=sleep, on_turn=on_turn)
    return gb.run(task)


def replay(run_id: str, root: Path = LOOP_DIR, replay_tools: str = "recorded", pace: float = 0.0,
           record: bool = False, on_turn=None, sleep=time.sleep) -> dict:
    """Re-run a recorded run with no network: recorded model responses, recorded (or live) tools."""
    rec = load_run(run_id, root)
    meta = rec["meta"]
    cfg = BrainConfig.from_dict(meta["config"])
    cfg.attach_media = replay_tools == "live" and cfg.attach_media
    cfg.budget.max_wall_s = 1e9  # timing differs on replay; every other budget replays exactly
    exchanges = [h for t in rec["turns"] for h in t.get("http", [])]
    rt = ReplayTransport(exchanges, pace=pace, sleep=sleep)
    transport = RecordingTransport(rt)
    client = make_client(transport, OFFLINE_KEY, cfg.http_timeout_s)
    tools = RecordedTools(rec["turns"]) if replay_tools == "recorded" else live_tools(f"replay:{run_id}")
    recorder = RunRecorder(new_run_id("replay"), root, enabled=record)
    gb = GeminiBrain(cfg, client, recorder, transport, brain_name=meta.get("brain", "argon"),
                     system_instruction=rec["system_instruction"], tool_declarations=rec["tools"],
                     tool_executor=tools, sleep=lambda s: None, on_turn=on_turn,
                     meta_extra={"replay_of": run_id, "replay_tools": replay_tools})
    s = gb.run(meta["task"])
    orig = rec["summary"] or summarize(rec["turns"], meta)
    keys = ("model_calls", "tool_calls", "tool_calls_by_name", "tokens", "final_text")
    s["replay"] = {
        "of": run_id, "tools": replay_tools,
        "exchanges_served": rt.i, "exchanges_recorded": len(rt.queue),
        "divergences": rt.divergences + getattr(tools, "divergences", []),
        "matches_original": {k: s.get(k) == orig.get(k) for k in keys},
        "original_stop_reason": orig.get("stop_reason"),
    }
    return s


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--brain", choices=["gemini", "fake", "claude"], default="fake")
    p.add_argument("--task")
    p.add_argument("--replay", metavar="RUN_ID")
    p.add_argument("--replay-tools", choices=["recorded", "live"], default="recorded")
    p.add_argument("--pace", type=float, default=0.0, help="replay: sleep recorded latency x PACE")
    p.add_argument("--record-replay", action="store_true", help="replay: also write a new run record")
    p.add_argument("--summary", metavar="RUN_ID", help="print a run's summary (recomputed if missing)")
    p.add_argument("--print-system", action="store_true", help="print the system instruction and exit")
    p.add_argument("--import-calls", metavar="CALLER", help="record runs/calls entries of a CLI caller")
    p.add_argument("--run-id")
    p.add_argument("--root", type=Path, default=LOOP_DIR)
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--yes", action="store_true", help="skip the spend confirmation for --brain gemini")
    p.add_argument("--quiet", action="store_true")
    g = p.add_argument_group("model")
    g.add_argument("--model", default=ARGON)
    g.add_argument("--thinking-level", default="medium", choices=["minimal", "low", "medium", "high"])
    g.add_argument("--thinking-summaries", default="auto", choices=["auto", "none"])
    g.add_argument("--max-output-tokens", type=int, default=32_768)
    g.add_argument("--background", action="store_true", help="force background mode (auto above 65,536)")
    g.add_argument("--http-timeout-s", type=float, default=1200.0)
    g.add_argument("--tools", help="comma-separated tool allowlist (default: all registered)")
    g.add_argument("--skills", help="comma-separated skills; the allowlist becomes their Tools sections plus core tools")
    g.add_argument("--caller", help="caller name for runs/calls (default argon / fake)")
    g.add_argument("--media-resolution", choices=["low", "medium", "high", "ultra_high"])
    g.add_argument("--video-placement", choices=["function_result", "user_input"], default="function_result")
    g.add_argument("--video-fps", type=float, help="sampling fps for attached clips (default: server's 1 fps)")
    b = p.add_argument_group("budget")
    d = Budget()
    b.add_argument("--max-model-calls", type=int, default=d.max_model_calls)
    b.add_argument("--max-tool-calls", type=int, default=d.max_tool_calls)
    b.add_argument("--max-tokens", type=int, default=d.max_tokens)
    b.add_argument("--max-usd", type=float, default=d.max_usd)
    b.add_argument("--max-wall-s", type=float, default=d.max_wall_s)
    a = p.parse_args(argv)
    on_turn = None if a.quiet else print_turn

    if a.print_system:
        print(build_system_instruction(a.brain if a.brain != "gemini" else "argon",
                                       a.run_id or "<run-id>", budget=vars(config_from_args(a).budget)))
        return 0
    if a.summary:
        r = load_run(a.summary, a.root)
        print(json.dumps(r["summary"] or summarize(r["turns"], r["meta"]), indent=1, default=str))
        return 0
    if a.import_calls:
        s = import_cli_calls(a.import_calls, a.run_id or new_run_id("claude"), a.root, task=a.task)
        print(json.dumps(s, indent=1, default=str))
        return 0
    if a.replay:
        s = replay(a.replay, a.root, a.replay_tools, a.pace, record=a.record_replay, on_turn=on_turn)
    else:
        if not a.task:
            p.error("--task is required (or use --replay / --summary / --print-system)")
        if a.brain == "claude":
            p.error("the Claude stand-in runs as a subagent through harness.cli; see "
                    "playbook/skills/brains/SKILL.md (--print-system gives its prompt)")
        cfg = config_from_args(a)
        cfg.validate()
        if a.brain == "gemini":
            if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
                print("No GEMINI_API_KEY or GOOGLE_API_KEY set. Use --brain fake or --replay.", file=sys.stderr)
                return 2
            ask = (f"Run {cfg.model} (thinking {cfg.thinking_level}) with budget "
                   f"{json.dumps(vars(cfg.budget))}? This spends API credit. [y/N] ")
            if not a.yes:
                if not sys.stdin.isatty():
                    print("Refusing to spend API credit without confirmation: pass --yes.", file=sys.stderr)
                    return 2
                if input(ask).strip().lower() not in ("y", "yes"):
                    return 1
        s = run_brain(a.brain, a.task, cfg, a.run_id, a.root, record=not a.no_record, on_turn=on_turn)
    final = s.pop("final_text", None)
    print(json.dumps(s, indent=1, default=str))
    if final:
        print("\nfinal text:\n" + final)
    return 0 if s["stop_reason"] in ("done", "replay_end") or str(s["stop_reason"]).startswith("budget:") else 1


if __name__ == "__main__":
    sys.exit(main())
