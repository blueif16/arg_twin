"""Offline tests for the brain runtime: playbook tools, the Gemini loop over the real SDK with a
scripted server, run records, replay, media attachment, budgets and API-contract guards.
No network and no API key: any attempt to open a real HTTP connection fails the test.

    .venv/bin/python -m harness.brains.test_offline        (pytest also collects it)
"""

from __future__ import annotations

import base64
import contextlib
import json
import shutil
import struct
import sys
import tempfile
import traceback
import zlib
from pathlib import Path

import httpx

from harness import registry
from harness.brains import record as R
from harness.brains.fake import FakeInteractionsServer, call, say, skills_tour, thought
from harness.brains.gemini import BrainConfig, Budget, GeminiBrain, live_tools, make_client
from harness.brains.run import OFFLINE_KEY, replay, run_brain
from harness.registry import REPO
from harness.tools import playbook

DEMO_SKILL = """---
name: demo-skill
description: Use when testing: a skill with a helper file
stage: practice
---
## Goal
Pass the offline test.

## Lessons

- **2026-10-01** (test): first lesson. Evidence: `playbook/rules.md`.

## Notes after lessons
kept in place
"""


# ---------------------------------------------------------------- fixtures

@contextlib.contextmanager
def temp_playbook():
    """A copy of playbook/ plus a demo skill, so tests never write the real skills, and a temp call
    log, so test tool calls stay out of runs/calls/."""
    with tempfile.TemporaryDirectory() as d:
        old_log, registry.CALL_LOG_DIR = registry.CALL_LOG_DIR, Path(d) / "calls"
        pb = Path(d) / "playbook"
        shutil.copytree(REPO / "playbook", pb)
        (pb / "skills" / "demo-skill" / "notes").mkdir(parents=True)
        (pb / "skills" / "demo-skill" / "SKILL.md").write_text(DEMO_SKILL)
        (pb / "skills" / "demo-skill" / "notes" / "how.md").write_text("helper text\n")
        old, playbook.PLAYBOOK = playbook.PLAYBOOK, pb
        try:
            yield pb
        finally:
            playbook.PLAYBOOK = old
            registry.CALL_LOG_DIR = old_log


@contextlib.contextmanager
def no_network():
    def refuse(self, request):
        raise AssertionError(f"network access attempted: {request.method} {request.url}")
    old = httpx.HTTPTransport.handle_request
    httpx.HTTPTransport.handle_request = refuse  # type: ignore[method-assign]
    try:
        yield
    finally:
        httpx.HTTPTransport.handle_request = old  # type: ignore[method-assign]


@contextlib.contextmanager
def no_tools():
    """Fail if any registry tool runs (replay must re-serve recorded results)."""
    old = registry.call

    def refuse(*a, **k):
        raise AssertionError(f"tool executed during recorded replay: {a}")
    registry.call = refuse  # type: ignore[assignment]  (gemini.live_tools looks it up at call time)
    try:
        yield
    finally:
        registry.call = old  # type: ignore[assignment]


@contextlib.contextmanager
def temp_tool(name: str, fn, params: dict | None = None):
    registry.load_all()
    registry.tool(name, f"test-only tool {name}", params)(fn)
    try:
        yield
    finally:
        registry.TOOLS.pop(name, None)


def tiny_png(w: int = 4, h: int = 3) -> bytes:
    raw = b"".join(b"\x00" + bytes([200, 30, 30]) * w for _ in range(h))

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def cfg(**kw) -> BrainConfig:
    budget = Budget(**kw.pop("budget", {}))
    return BrainConfig(caller="test", budget=budget, **kw)


# ---------------------------------------------------------------- playbook tools

def test_playbook_tools():
    with temp_playbook() as pb:
        idx = playbook.list_skills()
        names = {s["name"]: s for s in idx["skills"]}
        assert names["demo-skill"]["stage"] == "practice"
        assert names["demo-skill"]["description"] == "Use when testing: a skill with a helper file"
        assert names["demo-skill"]["lessons"] == 1
        assert playbook.list_skills(stage="practice")["count"] >= 1
        sk = playbook.read_skill("demo-skill")
        assert sk["text"].startswith("---") and sk["helpers"] == [{"file": "notes/how.md", "bytes": 12}]
        assert playbook.read_skill("demo-skill", file="notes/how.md")["text"] == "helper text\n"
        for bad in [lambda: playbook.read_skill("demo-skill", file="../../rules.md"),
                    lambda: playbook.read_skill("../rules"),
                    lambda: playbook.read_skill("nope"),
                    lambda: playbook.read_playbook_file("secrets")]:
            try:
                bad()
                raise AssertionError("expected an error")
            except (ValueError, FileNotFoundError):
                pass
        assert playbook.read_playbook_file("Orchestrator.md")["text"].startswith("# Orchestrator")

        # append_lesson: evidence is required and must resolve
        for ev in ([], ["runs/does/not/exist.json"]):
            try:
                playbook.append_lesson("demo-skill", "x", ev)
                raise AssertionError("expected evidence error")
            except ValueError:
                pass
        try:
            playbook.append_lesson("demo-skill", "key AIzaSyA1234567890abcdefghijklmnop", ["playbook/rules.md"])
            raise AssertionError("expected secret error")
        except ValueError:
            pass
        r = playbook.append_lesson("demo-skill", "second\nlesson, 3 of 4 passed.",
                                   ["playbook/rules.md:12", "https://ai.google.dev/x"], author="argon")
        assert r["ok"] and "second lesson, 3 of 4 passed." in r["entry"] and r["missing_evidence"] == []
        playbook.append_lesson("demo-skill", "third", ["playbook/README.md"], author="claude:t")
        text = (pb / "skills" / "demo-skill" / "SKILL.md").read_text()
        body = text.split("## Lessons\n", 1)[1]
        lessons, rest = body.split("## Notes after lessons", 1)
        assert [ln.split("(")[1].split(")")[0] for ln in lessons.strip().splitlines()] == ["test", "argon", "claude:t"]
        assert lessons.endswith("Evidence: `playbook/README.md`.\n\n") and "kept in place" in rest
        # a skill without a Lessons section gets one
        (pb / "skills" / "bare").mkdir()
        (pb / "skills" / "bare" / "SKILL.md").write_text("---\nname: bare\nstage: infra\n---\n## Goal\nx\n")
        playbook.append_lesson("bare", "first", ["playbook/rules.md"])
        lines = (pb / "skills" / "bare" / "SKILL.md").read_text().splitlines()
        assert lines[-4:-1] == ["x", "", "## Lessons"] or lines[-3:-1] == ["## Lessons", ""], lines
        assert lines[-1].endswith("(unknown): first Evidence: `playbook/rules.md`.")


# ---------------------------------------------------------------- loop, record, replay

def test_fake_loop_records_and_replays():
    with temp_playbook(), no_network(), tempfile.TemporaryDirectory() as root:
        root = Path(root)
        server = FakeInteractionsServer()
        c = cfg(thinking_level="high", http_timeout_s=900)
        s = run_brain("fake", "Tour the skills", c, run_id="t-fake", root=root, server=server)
        assert s["stop_reason"] == "done", s
        assert s["model_calls"] == 3 and s["tool_calls"] == 4 and s["tool_errors"] == 0, s
        assert s["tool_calls_by_name"] == {"log_note": 1, "list_skills": 1, "read_skill": 1,
                                           "read_playbook_file": 1}
        assert "read skill 'demo-skill'" in s["final_text"] or "read skill 'brains'" in s["final_text"]
        assert s["tokens"]["cached"] > 0 and 0 < s["cache_hit_ratio"] < 1

        # what was actually sent, as seen by the server
        for i, body in enumerate(server.requests):
            gc = body["generation_config"]
            assert gc["thinking_level"] == "high" and gc["thinking_summaries"] == "auto"
            assert not {"temperature", "top_p", "top_k", "thinking_budget"} & (set(gc) | set(body))
            assert body["tools"] and body["system_instruction"], "interaction-scoped fields re-sent"
            assert body.get("previous_interaction_id") == (f"fake_ix_{i}" if i else None)
        assert {t["read"] for t in server.timeouts} == {900.0}

        # the record on disk
        run = R.load_run("t-fake", root)
        assert len(run["turns"]) == 3 and run["summary"]["tokens"] == s["tokens"]
        t2 = run["turns"][1]
        assert t2["usage"]["cached"] == run["turns"][0]["usage"]["total"]
        assert t2["request"]["previous_interaction_id"] == "fake_ix_1"
        assert [(x["type"], x["name"]) for x in t2["request"]["input"]] == [
            ("function_result", "log_note"), ("function_result", "list_skills")]
        assert t2["request"]["tools"]["n"] == len(run["tools"]) and t2["request"]["system_instruction_sha"]
        assert [c["name"] for c in t2["tool_calls"]] == ["read_skill", "read_playbook_file"]
        assert run["turns"][0]["thoughts"] == ["Start from the skill index, as the preamble says."]
        assert "# playbook/orchestrator.md" in run["system_instruction"]
        assert "demo-skill" in run["system_instruction"]
        blob = "".join(p.read_text() for p in (root / "t-fake").iterdir())
        assert OFFLINE_KEY not in blob and "x-goog-api-key" not in blob, "no secrets in records"

        # replay: recorded responses and recorded tool results, no tools, no network
        with no_tools():
            r = replay("t-fake", root)
        assert all(r["replay"]["matches_original"].values()), r["replay"]
        assert r["replay"]["divergences"] == [] and r["stop_reason"] == "done"
        assert r["replay"]["exchanges_served"] == r["replay"]["exchanges_recorded"] == 3

        # replay with live tools gives the same calls and final text (tools are deterministic here)
        r2 = replay("t-fake", root, replay_tools="live", record=True)
        assert r2["replay"]["matches_original"]["final_text"] and r2["replay"]["divergences"] == []
        assert R.load_run(r2["run_id"], root)["meta"]["replay_of"] == "t-fake"


def test_background_polling_and_replay():
    with temp_playbook(), no_network(), tempfile.TemporaryDirectory() as root:
        root = Path(root)
        server = FakeInteractionsServer(background_polls=2)
        s = run_brain("fake", "bg", cfg(max_output_tokens=131072), run_id="t-bg", root=root, server=server)
        assert s["stop_reason"] == "done" and s["model_calls"] == 3
        assert all(b.get("background") is True for b in server.requests)
        http = R.load_run("t-bg", root)["turns"][0]["http"]
        assert [h["method"] for h in http] == ["POST", "GET", "GET", "GET"], http
        with no_tools():
            r = replay("t-bg", root)
        assert all(r["replay"]["matches_original"].values()) and r["replay"]["exchanges_served"] == 12


def test_media_attached_to_function_results():
    media_dir = REPO / "runs" / "loop" / "_selftest_media"
    media_dir.mkdir(parents=True, exist_ok=True)
    png, mp4, big = media_dir / "grid.png", media_dir / "clip.mp4", media_dir / "big.png"
    png.write_bytes(tiny_png())
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    big.write_bytes(tiny_png(64, 64) + b"\x00" * 5000)
    rel = lambda p: str(p.relative_to(REPO))
    seen: dict = {}

    def script(turn, ctx):
        if turn == 1:
            return [call("_test_render", {})]
        seen["results"] = ctx["results"]
        seen["user_media"] = ctx.get("user_media", [])
        return [say("looked")]

    def render():
        return {"grid": rel(png), "clip": rel(mp4), "nested": {"big": rel(big)},
                "outside": "/etc/hostname.png", "missing": "runs/nope.png"}
    try:
        with temp_playbook(), temp_tool("_test_render", render), no_network(), \
                tempfile.TemporaryDirectory() as root:
            server = FakeInteractionsServer(script=script)
            c = cfg(tool_names=["_test_render"], max_inline_image_bytes=2000, media_resolution="high",
                    video_fps=4)
            s = run_brain("fake", "look", c, run_id="t-media", root=Path(root), server=server)
            assert s["stop_reason"] == "done" and s["media_attached"] == 2, s
            res = seen["results"][0]
            kinds = [(m["type"], m["mime_type"]) for m in res["media"]]
            assert kinds == [("image", "image/png"), ("video", "video/mp4")], kinds
            assert base64.b64decode(res["media"][0]["data"]) == png.read_bytes()
            assert res["media"][0]["resolution"] == "high"
            assert res["media"][1]["processing"] == {"type": "static", "fps": 4.0}
            assert f"[not attached: {rel(big)} (image too large to send inline)]" in res["text"]
            media = R.load_run("t-media", Path(root))["turns"][0]["tool_calls"][0]["media"]
            assert [m["mode"] for m in media] == ["inline", "inline", "skipped"]
            t1 = R.load_run("t-media", Path(root))["turns"][1]
            assert t1["request"]["input"][0]["media"] == ["image", "video"]
            sent = t1["http"][0]["request"]["input"][0]["result"][2]
            assert sent["data"].startswith("<base64 ") and sent["type"] == "image", "no raw media in records"

            # videos above the inline cap go through the uploader; user_input placement option
            server = FakeInteractionsServer(script=script)
            transport = R.RecordingTransport(server.transport())
            c = cfg(tool_names=["_test_render"], max_inline_video_bytes=10, video_placement="user_input")
            gb = GeminiBrain(c, make_client(transport, OFFLINE_KEY, 60), R.RunRecorder("t-up", Path(root)),
                             transport, uploader=lambda p: ("https://files.example/abc", "video/mp4"),
                             tool_executor=live_tools("test"))
            s = gb.run("look again")
            assert s["stop_reason"] == "done"
            assert [m["type"] for m in seen["results"][0]["media"]] == ["image", "image"]
            assert seen["user_media"] == [{"type": "video", "uri": "https://files.example/abc",
                                           "mime_type": "video/mp4"}]
    finally:
        shutil.rmtree(media_dir, ignore_errors=True)


def test_budgets_and_errors():
    with temp_playbook(), no_network(), tempfile.TemporaryDirectory() as root:
        root = Path(root)
        s = run_brain("fake", "x", cfg(budget={"max_model_calls": 1}), run_id="b1", root=root)
        assert s["stop_reason"] == "budget:model_calls" and s["model_calls"] == 1 and s["tool_calls"] == 2
        s = run_brain("fake", "x", cfg(budget={"max_tool_calls": 1}), run_id="b2", root=root)
        assert s["stop_reason"] == "budget:tool_calls" and s["tool_calls"] == 1
        assert R.load_run("b2", root)["turns"][0]["tool_calls"][1]["skipped"] == "budget:tool_calls"
        s = run_brain("fake", "x", cfg(budget={"max_tokens": 1}), run_id="b3", root=root)
        assert s["stop_reason"] == "budget:tokens" and s["model_calls"] == 1
        s = run_brain("fake", "x", cfg(budget={"max_usd": 0.0001}), run_id="b4", root=root)
        assert s["stop_reason"] == "budget:usd" and s["model_calls"] == 1
        s = run_brain("fake", "x", cfg(budget={"max_wall_s": 0}), run_id="b5", root=root)
        assert s["stop_reason"] == "budget:wall_s" and s["model_calls"] == 0

        # replay of a budget-stopped run stops at the same place; with a raised budget it ends cleanly
        with no_tools():
            r = replay("b1", root)
        assert r["stop_reason"] == "budget:model_calls" and all(r["replay"]["matches_original"].values())
        meta = json.loads((root / "b1" / "meta.json").read_text())
        meta["config"]["budget"]["max_model_calls"] = 5
        (root / "b1" / "meta.json").write_text(json.dumps(meta))
        with no_tools():
            r = replay("b1", root)
        assert r["stop_reason"] == "replay_end" and r["model_calls"] == 1

        # unknown tool -> is_error result goes back to the model
        flags = {}

        def script(turn, ctx):
            if turn == 1:
                return [thought("try a tool that does not exist"), call("no_such_tool", {"a": 1})]
            flags["r"] = ctx["results"][0]
            return [say("ok")]
        s = run_brain("fake", "x", cfg(), run_id="e1", root=root, server=FakeInteractionsServer(script=script))
        assert s["stop_reason"] == "done" and s["tool_errors"] == 1
        assert flags["r"]["is_error"] and "unknown tool" in flags["r"]["data"]["error"]

        # an API error ends the run cleanly and is recorded
        class Broken(FakeInteractionsServer):
            def create(self, body):
                raise ValueError("fake 400 for test")
        s = run_brain("fake", "x", cfg(), run_id="e2", root=root, server=Broken())
        assert s["stop_reason"] == "api_error" and "fake 400 for test" in R.load_run("e2", root)["turns"][0]["error"]


def test_config_guards():
    for bad in [dict(extra_generation_config={"temperature": 0.2}),
                dict(extra_generation_config={"thinking_budget": 1024}),
                dict(thinking_level="minimal"),
                dict(max_output_tokens=131072, background=False),
                dict(video_placement="sideways")]:
        try:
            BrainConfig(**bad).validate()
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass
    c = BrainConfig(max_output_tokens=131072)
    c.validate()
    assert c.use_background() and not BrainConfig().use_background()
    assert set(BrainConfig().generation_config()) == {"thinking_level", "max_output_tokens", "thinking_summaries"}
    assert BrainConfig.from_dict(BrainConfig(thinking_level="low").to_dict()).thinking_level == "low"


def test_import_cli_calls():
    with tempfile.TemporaryDirectory() as d:
        calls = Path(d) / "calls"
        calls.mkdir()
        rows = [{"t": 100.0, "dt": 0.5, "caller": "claude:r1", "tool": "list_skills", "args": {},
                 "ok": True, "result": {"count": 0}},
                {"t": 101.0, "dt": 0.1, "caller": "other", "tool": "x", "args": {}, "ok": True, "result": {}},
                {"t": 103.0, "dt": 1.0, "caller": "claude:r1", "tool": "read_skill", "args": {"name": "a"},
                 "ok": False, "result": {"error": "nope"}}]
        (calls / "20261024.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        old, R.CALL_LOG_DIR = R.CALL_LOG_DIR, calls
        try:
            s = R.import_cli_calls("claude:r1", "r1", Path(d) / "loop", task="t")
        finally:
            R.CALL_LOG_DIR = old
        assert s["tool_calls"] == 2 and s["tool_errors"] == 1 and s["wall_s"] == 4.0 and s["brain"] == "claude"


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}\n{traceback.format_exc()}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
