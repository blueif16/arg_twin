"""Practice-stage tools (stage D; skill playbook/skills/practice).

  env_observe_image   the actor's eyes: calibrated hama1 / hama2 / wrist views of the current episode with the visual
                      coordinate harness drawn from BELIEVED geometry (twin/practice_view.py)
  practice_manifest   freeze a practice run before attempt 1: budget, start ids, brain and model, evaluator and code
                      hashes, skill versions, tool allowlist; `verify` re-checks every hash later
  practice_ledger     watcher / reporting: one row per episode (rubric stage, outcome, claim, limit hits, calls, time,
                      tokens), the code monitor's wake events, D aggregates, a watcher packet per episode (no actor
                      reasoning), and an exact McNemar test against a paired run
  practice_clip       an H.264 clip of a finished episode, replayed in a shadow env with the same overlays

Information honesty: everything drawn for the actor comes from what the env's task() told it (nominal board, nominal
grasp) plus its own observations. Sim truth stays with the env and the evaluator; held-out outcomes never come back.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from harness.registry import REPO, TOOLS, tool

NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_RUN = {"brain": {"type": "string", "default": "claude", "description": "claude (stand-in) or argon"},
        "run_id": {"type": "string", "default": "run-standin-1", "description": "runs/<brain>/<run_id>/"}}

# The frozen per-episode budget of the attempt-1 baseline (playbook/skills/practice "Budget"). Evidence:
# - LIBERO-10, end-effector pose tool calls: Argon (high) hit the 80-call / spend / token limit in 41 of 100 episodes,
#   averaging ~50 calls on two-subtask tasks [llm-libero-2026]. One in-hand insertion is one subtask, so the same
#   80-call cap gives it twice LIBERO's per-subtask allowance: a limit hit here means no convergence, not a tight cap.
# - Google's arm prompt sizes fine visual servoing at 8-20 calls [google-argon-robotics-2026]; Robocurve's puzzle-in-
#   groove ran on 20 calls [robocurve-astra-2026]. 80 is 4x both.
# - Real teleop insert segments: median 11.5-19.6 s, slowest successful insertion 27.0 s (twin/envelope.py, train).
#   120 sim s is 4.4x the slowest and the env default.
# - Tokens: a chained run re-reads its context each turn. With a ~25k-token prefix (system, tools, skill) and ~2.5k
#   per turn (JSON result + one composite image), 80 turns cost about 25k*80 + 2.5k*80^2/2 = 10.1M tokens; 12M keeps
#   tokens from binding before calls. Most of it is cached input (5% price).
# - Wall: physics is paused while the brain thinks. 80 calls x ~18.6 s (Astra direct per-call latency
#   [astranomy-2026]) + ~2 s sim and render per call = ~1,650 s; 3,600 s guards against hangs only.
BASELINE_BUDGET = dict(brain_calls=80, env_commands=160, sim_s=120.0, wall_s=3600.0, tokens=12_000_000)
WATCHER = dict(wake_on=["episode_end", "anomaly", "stuck", "near_promotion"],
               stuck=dict(sim_s=20.0, commands=12), anomaly_force_frac=0.8, near_promotion_streak=3,
               sees=["tool calls and results (no log_note / log_decision text)", "observation images", "episode records"],
               never_sees=["actor reasoning or thoughts", "held-out outcomes before the run ends"],
               may=["end or invalidate an episode (with reason)", "pick the next practice start",
                    "raise a physics question (F)", "write the post-episode critique"],
               never=["record D checks for its own actor", "score episodes (the evaluator does)"],
               tokens="counted in D.tokens_per_success")


def _name(v, what):
    if not isinstance(v, str) or not NAME_RE.match(v) or v.startswith("."):
        raise ValueError(f"bad {what}: {v!r}")
    return v


def _run_dir(brain, run_id) -> Path:
    return REPO / "runs" / _name(brain, "brain") / _name(run_id, "run_id")


def _rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ================================================================================================ observation image
@tool("env_observe_image",
      "Look at the current episode: calibrated views (wrist camera 'hand' at native 640x480; fixed cameras 'hama1', "
      "'hama2' cropped around the believed hole and resampled by `zoom`) composited into one PNG under a text header. "
      "Pixels are the cameras. Overlays are BELIEFS from your task (nominal board and grasp), never sim truth: "
      "yellow 1 cm world grid at the plate top (border labels X / Y in world cm), green believed hole outline + "
      "centre + hole axis drawn 60 mm out of the hole, magenta believed part axis + tip + the tip's foot on the grid "
      "plane, cyan pinch centre (+) and jaw line [ ]. The header gives the believed tip in the believed hole frame "
      "(mm), axis tilt, TCP, force and gripper width. Overlays carry the real calibration error (about 1-2 px fixed "
      "cameras, about 5 px wrist). Does not advance time or count as a command.",
      {"type": "object", "required": ["session"], "properties": {
          **_RUN, "session": {"type": "string", "description": "session id from env_open"},
          "views": {"type": "array", "items": {"type": "string", "enum": ["hand", "hama1", "hama2"]},
                    "description": "panels left to right (default hand, hama1, hama2)"},
          "zoom": {"type": "number", "description": "crop factor for the fixed cameras, 1-6 (default 3)"},
          "overlays": {"type": "boolean", "description": "false: pixels only, to see the part itself (default true)"}}})
def env_observe_image(session: str, views: list | None = None, zoom: float = 3.0, brain: str = "claude",
                      run_id: str = "run-standin-1", overlays: bool = True) -> dict:
    from twin import practice_view as PV
    t0 = time.time()
    views = list(views or PV.VIEWS)
    if not views or any(v not in PV.VIEWS for v in views):
        raise ValueError(f"views must be a non-empty subset of {list(PV.VIEWS)}")
    zoom = float(zoom)
    if not 1.0 <= zoom <= 6.0:
        raise ValueError("zoom must be between 1 and 6")
    sdir = _run_dir(brain, run_id) / "env" / "sessions" / _name(session, "session")
    if not (sdir / "config.json").exists():
        raise FileNotFoundError(f"no session {session!r} in {brain}/{run_id}")
    cfg = json.loads((sdir / "config.json").read_text())
    ep = PV.episode_log(session, brain, run_id)
    if not ep.get("task"):
        raise ValueError("the episode's env_reset result has no task in the call log")
    # The shadow can only replay commands that went through env_move / env_gripper / env_wait (the call log).
    # Check against the LIVE env: its command count and current observation. Tools that drive the env socket
    # directly (e.g. task_* tools) leave commands the shadow cannot see; never render a stale state as current.
    if not ep.get("finished"):
        from harness.tools.env import _call
        try:
            st = _call(brain, run_id, session, {"op": "status"})
        except FileNotFoundError:
            st = {}
        n_logged = sum(1 for c in ep["commands"] if c["op"] in ("move", "gripper", "wait"))
        if st.get("start_id") == ep["start_id"] and st.get("commands_used") is not None:
            if st["commands_used"] != n_logged:
                raise RuntimeError(
                    f"no image: the env has run {st['commands_used']} commands this episode but the call log shows "
                    f"{n_logged} env_move/env_gripper/env_wait calls. Commands sent around the logged env_* tools "
                    "(e.g. by task_* tools over the session socket) cannot be replayed by the image shadow")
            if not st.get("episode_over"):
                ep["obs"] = _call(brain, run_id, session, {"op": "observe"})
    sock = sdir / "view.sock"
    if not sock.exists():
        vcfg = dict(brain=brain, run_id=run_id, start_set=cfg["start_set"], cousin=cfg.get("cousin", "c000"),
                    session=session)
        (sdir / "view_config.json").write_text(json.dumps(vcfg, indent=1))
        log = open(sdir / "view_daemon.log", "ab")
        subprocess.Popen([sys.executable, "-m", "twin.practice_view", "serve", "--socket", str(sock), "--config",
                          str(sdir / "view_config.json")], cwd=str(REPO), stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, start_new_session=True,
                         env={**{k: v for k, v in os.environ.items() if not re.search(r"KEY|TOKEN|SECRET", k)},
                              "PYTHONPATH": str(REPO), "MUJOCO_GL": os.environ.get("MUJOCO_GL", "glfw")})
        for _ in range(600):
            if sock.exists():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"view daemon did not start; see {_rel(sdir / 'view_daemon.log')}")
    img_dir = _run_dir(brain, run_id) / "practice" / "images" / session
    n = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
    t_obs = (ep.get("obs") or {}).get("t", 0.0)
    out = img_dir / f"{n:03d}_{ep['start_id']}_t{t_obs:07.2f}.png"
    rep = PV.call_socket(str(sock), dict(op="render", key=ep["key"], start_id=ep["start_id"], commands=ep["commands"],
                                         expect=ep.get("obs"), task=ep["task"], views=views, zoom=zoom, out=str(out),
                                         overlays=bool(overlays)))
    if "error" in rep:
        raise RuntimeError(f"render failed: {rep['error']}")
    sync = rep["sync"]
    res = dict(image=_rel(out), views=views, t=rep["belief"]["t"],
               believed=dict(tip_vs_hole_mm=rep["belief"]["tip_vs_hole_mm"], tilt_deg=rep["belief"]["tilt_deg"],
                             note="tip in the believed hole frame: x, y lateral; depth + = into the hole"),
               wall_s=round(time.time() - t0, 2))
    if sync.get("match") is False:
        res["warning"] = (f"the image may not show the latest state (shadow replay off by {sync.get('dt_s')} s / "
                          f"{sync.get('dpos_m')} m)")
    if ep.get("finished"):
        res["note"] = "the episode is finished; this is its final state"
    return res


# ================================================================================================ manifest
FROZEN_SETS = {"starts": ["manifest.json", "manifest.sha256"], "cousins": ["manifest.json", "manifest.sha256"],
               "envelope": ["envelope.json", "envelope.sha256"]}
CODE_FILES = ["twin/env.py", "twin/evaluator.py", "twin/monitors.py", "twin/starts.py", "twin/cousins.py",
              "twin/envelope.py", "twin/replay.py", "twin/control.py", "twin/practice_view.py", "harness/tools/env.py",
              "harness/tools/practice.py", "harness/registry.py", "playbook/rules.md", "playbook/orchestrator.md"]


def _copy_frozen(src: Path, dst: Path) -> dict:
    """Copy frozen start sets, cousins and the envelope byte for byte (paired starts across runs), then load them
    through their own hash checks."""
    from twin import cousins as C
    from twin import starts as S
    out = {}
    for sub in FROZEN_SETS:
        s, d = src / sub, dst / sub
        if not s.exists():
            raise FileNotFoundError(f"{_rel(s)} missing: the source run has no frozen {sub}")
        if not d.exists():
            shutil.copytree(s, d)
        for f in sorted(s.iterdir()):
            if f.is_file() and _sha(f) != _sha(d / f.name):
                raise RuntimeError(f"{_rel(d / f.name)} differs from {_rel(f)}: frozen sets must be byte-identical")
        out[sub] = {f.name: _sha(f) for f in sorted(d.iterdir()) if f.is_file()}
    for name in ("practice", "heldout"):
        S.load(name, dst / "starts")
    C.load(dst / "cousins")
    return out


def _skill_versions(skills: list[str]) -> dict:
    out = {}
    for s in skills:
        f = REPO / "playbook" / "skills" / s / "SKILL.md"
        if not f.exists():
            raise FileNotFoundError(f"no skill {s!r}")
        txt = f.read_text()
        lessons = txt.split("## Lessons", 1)[1] if "## Lessons" in txt else ""
        out[s] = dict(sha256=_sha(f), lessons=len(re.findall(r"^\s*[-*]?\s*\*\*20\d\d-\d\d-\d\d", lessons, re.M)),
                      mtime=time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)))
    return out


def _tool_hash(names: list[str]) -> dict:
    from harness.registry import load_all
    tools = load_all()
    missing = [n for n in names if n not in tools]
    if missing:
        raise ValueError(f"unknown tools {missing}")
    decl = json.dumps([tools[n].declaration() for n in names], sort_keys=True)
    return dict(names=names, sha256=hashlib.sha256(decl.encode()).hexdigest())


def _manifest_state(m: dict, run_dir: Path) -> dict:
    """Recompute every hash a manifest froze."""
    from twin.evaluator import EVALUATOR_VERSION, code_hash
    import yaml
    val = REPO / "playbook" / "validation.yaml"
    return dict(
        evaluator=dict(version=EVALUATOR_VERSION, code_hash=code_hash()),
        code={f: _sha(REPO / f) for f in m["code"]},
        frozen_sets={sub: {f.name: _sha(f) for f in sorted((run_dir / sub).iterdir()) if f.is_file()}
                     for sub in FROZEN_SETS},
        skills={k: v["sha256"] for k, v in _skill_versions(list(m["skills"])).items()},
        tools=_tool_hash(m["tools"]["names"])["sha256"],
        validation=dict(version=yaml.safe_load(val.read_text())["version"], sha256=_sha(val)),
        budget_file=_sha(run_dir / "env" / "budget.json"))


def _verify(rd: Path) -> dict:
    mf = rd / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"{_rel(mf)} missing")
    m = json.loads(mf.read_text())
    if _sha(mf) != (rd / "manifest.sha256").read_text().split()[0]:
        return dict(ok=False, changed=["manifest.json itself (does not match manifest.sha256)"], manifest=m)
    now = _manifest_state(m, rd)
    changed = []
    for k, v in now.items():
        if v != m["frozen"][k]:
            if isinstance(v, dict):
                changed += [f"{k}.{kk}" for kk in v if v.get(kk) != m["frozen"][k].get(kk)]
            else:
                changed.append(k)
    return dict(ok=not changed, changed=changed, manifest=m)


@tool("practice_manifest",
      "Freeze a practice run BEFORE its first episode (skill practice): per-episode budget (brain calls, env commands, "
      "sim s, wall s, tokens), the start ids, brain, model and thinking level, the tool allowlist, skill versions, "
      "the evaluator and code hashes, and the watcher's wake rules. Copies the frozen start sets, cousins and envelope "
      "byte-identical from source_run so paired runs share starts, writes the env's frozen budget, and refuses to "
      "overwrite. verify=true re-hashes everything and lists what changed since the freeze.",
      {"type": "object", "required": ["run_id"], "properties": {
          **_RUN,
          "verify": {"type": "boolean", "description": "re-check an existing manifest instead of writing one"},
          "require_fresh": {"type": "boolean", "description": "runner, before attempt 1 and each session: refuse "
                            "(error) if anything frozen changed, unless the run is a dry run"},
          "dry_run": {"type": "boolean", "description": "new manifest: a harness test, never scored (default false)"},
          "mark_dry_run": {"type": "boolean", "description": "amend an existing manifest: mark it a dry run and record "
                           "what changed since the freeze (one way: a dry run never becomes scored)"},
          "attempt": {"type": "string", "description": "baseline | attempt_n | dry_run (label; default baseline)"},
          "model": {"type": "string", "description": "e.g. gemini-4-argon, or claude-opus-5.5 for the stand-in"},
          "thinking_level": {"type": "string"},
          "start_set": {"type": "string", "enum": ["practice", "heldout"]},
          "start_ids": {"type": "array", "items": {"type": "string"}, "description": "explicit start ids, in order"},
          "parts": {"type": "array", "items": {"type": "string"}, "description": "or: parts, with per_part"},
          "per_part": {"type": "integer", "description": "first N starts of each part (held-out: <= 29 or 46)"},
          "budget": {"type": "object", "description": "per episode: brain_calls, env_commands, sim_s, wall_s, tokens "
                                                      "(default: the baseline budget in the skill)"},
          "skills": {"type": "array", "items": {"type": "string"}, "description": "skills given to the brain"},
          "tools": {"type": "array", "items": {"type": "string"}, "description": "the brain's tool allowlist"},
          "paired_with": {"type": "string", "description": "runs/<brain>/<run_id> this run is paired with"},
          "source_run": {"type": "string", "description": "run dir holding the frozen sets (default runs/claude/run-standin-1)"},
          "note": {"type": "string"}}})
def practice_manifest(run_id: str, brain: str = "claude", verify: bool = False, attempt: str = "baseline",
                      model: str = "", thinking_level: str = "", start_set: str = "practice",
                      start_ids: list | None = None, parts: list | None = None, per_part: int | None = None,
                      budget: dict | None = None, skills: list | None = None, tools: list | None = None,
                      paired_with: str = "", source_run: str = "runs/claude/run-standin-1", note: str = "",
                      require_fresh: bool = False, dry_run: bool = False, mark_dry_run: bool = False) -> dict:
    from twin import starts as S
    from harness.tools.env import frozen_budget
    rd = _run_dir(brain, run_id)
    mf = rd / "manifest.json"
    if verify or require_fresh:
        v = _verify(rd)
        m = v.pop("manifest")
        out = dict(ok=v["ok"], changed=v["changed"], dry_run=bool(m.get("dry_run")), manifest=_rel(mf),
                   frozen_at=m["frozen_at"], amendments=len(m.get("amendments", [])))
        if require_fresh and not v["ok"] and not m.get("dry_run"):
            raise PermissionError(f"{brain}/{run_id} is stale: {v['changed']} changed since the freeze at "
                                  f"{m['frozen_at']}. A scored run never continues on changed code or sets: freeze "
                                  "a NEW run_id with practice_manifest (it re-hashes the current code).")
        if require_fresh and not v["ok"]:
            out["warning"] = "dry run on changed code: allowed, never scored"
        return out
    if mark_dry_run:
        v = _verify(rd)
        m = v.pop("manifest")
        old_sha = (rd / "manifest.sha256").read_text().split()[0]
        m["dry_run"] = True
        m.setdefault("amendments", []).append(dict(
            at=time.strftime("%Y-%m-%d %H:%M:%S"), what="marked dry_run: true (harness test, never scored)",
            changed_since_freeze=v["changed"], manifest_sha256_before=old_sha, note=note))
        mf.write_text(json.dumps(m, indent=1))
        (rd / "manifest.sha256").write_text(_sha(mf) + "\n")
        return dict(manifest=_rel(mf), dry_run=True, changed_since_freeze=v["changed"], sha256=_sha(mf))
    if mf.exists():
        raise FileExistsError(f"{_rel(mf)} exists: a practice run is frozen once. Use a new run_id.")
    eps = rd / "env" / "episodes.jsonl"
    if eps.exists() and eps.read_text().strip():
        raise RuntimeError(f"{_rel(eps)} already has episodes: freeze the manifest before attempt 1")
    src = (REPO / source_run).resolve()
    if not src.is_relative_to(REPO / "runs"):
        raise ValueError("source_run must be a run directory under runs/")
    frozen_sets = _copy_frozen(src, rd)
    starts, _ = S.load(start_set, rd / "starts")
    if start_ids:
        bad = [s for s in start_ids if s not in starts]
        if bad:
            raise ValueError(f"not in the {start_set} set: {bad}")
        ids = list(start_ids)
    elif parts and per_part:
        cap = S.HELDOUT_SUBSET if start_set == "heldout" and per_part <= S.HELDOUT_SUBSET else per_part
        ids = []
        for p in parts:
            mine = sorted(k for k, v in starts.items() if v["part"] == p)
            if not mine:
                raise ValueError(f"no {start_set} starts for part {p!r}")
            ids += mine[:cap]
    else:
        raise ValueError("give start_ids, or parts with per_part")
    b = dict(BASELINE_BUDGET, **(budget or {}))
    unknown = set(b) - set(BASELINE_BUDGET)
    if unknown:
        raise ValueError(f"unknown budget keys {sorted(unknown)}; known {sorted(BASELINE_BUDGET)}")
    env_b = frozen_budget(brain, run_id, dict(max_commands=int(b["env_commands"]), max_sim_s=float(b["sim_s"]),
                                              max_wall_s=float(b["wall_s"]), max_tokens=int(b["tokens"])))
    skills = list(skills or ["practice"])
    tools = list(tools or [])
    if not tools:
        from harness.brains.run import skill_tools
        tools = skill_tools(skills)
    m = dict(
        run=f"{brain}/{run_id}", brain=brain, run_id=run_id, attempt=attempt, dry_run=bool(dry_run), model=model,
        thinking_level=thinking_level,
        frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"), start_set=start_set, start_ids=ids,
        parts=sorted({starts[s]["part"] for s in ids}), paired_with=paired_with or None,
        budget=dict(per_episode=b, env_budget=env_b,
                    enforcement=dict(brain_calls="the brain runner (or the stand-in itself) ends the episode with "
                                                 "env_finish and no claim; the ledger records limit_reached:brain_calls",
                                     env_commands_sim_wall_tokens="the env ends the episode limit_reached",
                                     tokens="the runner charges each turn's tokens with env_charge_tokens"),
                    why="playbook/skills/practice 'Budget'; harness/tools/practice.py BASELINE_BUDGET"),
        watcher=WATCHER, skills=_skill_versions(skills), tools=_tool_hash(tools), code=CODE_FILES,
        source_run=_rel(src), frozen_sets=frozen_sets,
        information=dict(actor_gets="env task(): nominal hole pose, nominal grasp, part geometry, clearance, limits; "
                                    "observations; env_observe_image (true pixels, belief overlays, measured camera "
                                    "error)", actor_never_gets="board_err, in-hand pose, cousin physics, evaluator "
                                    "state, held-out outcomes"),
        rules=["no start outside start_ids is attempted under this run", "held-out starts are never touched in practice",
               "limit hits are reported apart from failures", "the actor never records D checks"],
        note=note)
    m["frozen"] = _manifest_state(m, rd)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "practice").mkdir(exist_ok=True)
    mf.write_text(json.dumps(m, indent=1))
    (rd / "manifest.sha256").write_text(_sha(mf) + "\n")
    return dict(manifest=_rel(mf), sha256=_sha(mf), start_ids=ids, budget=b, tools=tools,
                skills=list(m["skills"]), evaluator=m["frozen"]["evaluator"])


# ================================================================================================ ledger
def _cp(k: int, n: int, a: float = 0.05) -> list:
    """Two-sided Clopper-Pearson interval."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return [round(lo, 3), round(hi, 3)]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return round(min(1.0, 2 * p), 4)


def _calls_for(sessions: set[str]) -> list[dict]:
    out = []
    for f in sorted((REPO / "runs" / "calls").glob("*.jsonl"))[-3:]:
        with open(f) as fh:
            for line in fh:
                if not any(s in line for s in sessions):
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (r.get("args") or {}).get("session") in sessions:
                    out.append(r)
    return sorted(out, key=lambda r: r["t"])


def _turns(calls: list[dict], gap_s: float = 1.0) -> int:
    """Brain decisions of a CLI stand-in: calls chained in one shell command (start within gap_s of the previous
    call's end) are one turn, like several function calls in one model turn."""
    n, last_end = 0, None
    for c in calls:
        if last_end is None or c["t"] - last_end > gap_s:
            n += 1
        last_end = c["t"] + c.get("dt", 0.0)
    return n


def _episode_row(rec_line: dict, run_dir: Path, calls: list[dict], tokens: dict, budget: dict, force_cap: float) -> dict:
    rec = json.loads((REPO / rec_line["file"]).read_text())
    ev = rec["evaluation"]
    # this episode's calls: from its env_reset to its env_finish (or the last call before the next reset)
    sess = [c for c in calls if (c.get("args") or {}).get("session") == rec["session"]]
    # every reset that started an episode (a grasp that fails in the pre-roll still records an 'error' episode)
    resets = [i for i, c in enumerate(sess) if c["tool"] == "env_reset" and c.get("ok")
              and ("error" not in (c.get("result") or {}) or "grasp failed" in str(c["result"]["error"]))]
    k = rec["episode"] - 1
    i0 = resets[k] if k < len(resets) else None
    win = []
    if i0 is not None:
        i1 = resets[k + 1] if k + 1 < len(resets) else len(sess)
        win = [c for c in sess[i0:i1] if c["tool"] not in ("log_note", "log_decision")]
        fin = [j for j, c in enumerate(win) if c["tool"] == "env_finish"]
        if fin:
            win = win[:fin[0] + 1]
    by_tool: dict[str, int] = {}
    for c in win:
        by_tool[c["tool"]] = by_tool.get(c["tool"], 0) + 1
    tool_s = sum(c.get("dt", 0.0) for c in win)
    span = (win[-1]["t"] + win[-1].get("dt", 0.0) - win[0]["t"]) if win else rec["wall_s"]
    decisions = tokens.get(rec["start_id"], {}).get("decisions") or _turns(win[1:]) if win else None
    peak = max([abs(c["result"]["result"].get("max_force_N", 0.0)) for c in win
                if isinstance((c.get("result") or {}).get("result"), dict)] or [0.0])
    flags = rec["monitors"]["flags"]
    limit = rec["ended_by"] == "limit_reached"
    if decisions is not None and decisions > budget.get("brain_calls", 1e9):
        limit = True
    tl = ev.get("stage_timeline") or []
    stage_t = [t for t, _ in tl]
    cmd_t = [c["t"] for c in rec["commands"]]
    # code-monitor wake events (when the separate LLM watcher would have been woken)
    events = []
    t_prev = 0.0
    n_since = 0
    for ct in cmd_t:
        n_since += 1
        if any(t_prev < st <= ct for st in stage_t):
            n_since, t_prev = 0, ct
        elif ct - t_prev >= WATCHER["stuck"]["sim_s"] or n_since >= WATCHER["stuck"]["commands"]:
            events.append(dict(kind="stuck", t=ct, why=f"no rubric progress for {ct - t_prev:.1f} sim s / {n_since} commands"))
            n_since, t_prev = 0, ct
    if flags:
        events.append(dict(kind="anomaly", why="; ".join(f"{f['monitor']}" for f in flags)))
    if peak >= WATCHER["anomaly_force_frac"] * force_cap:
        events.append(dict(kind="anomaly", why=f"peak force {peak:.1f} N >= {WATCHER['anomaly_force_frac']} x cap"))
    errs = [c for c in win if not c.get("ok") or "error" in (c.get("result") or {})]
    if errs:
        events.append(dict(kind="anomaly", why=f"{len(errs)} tool errors: " + "; ".join(
            str((c.get("result") or {}).get("error", ""))[:80] for c in errs[:3])))
    events.append(dict(kind="episode_end", why=rec["outcome"]))
    tok = tokens.get(rec["start_id"], {})
    think_s = max(0.0, span - tool_s)
    return dict(
        session=rec["session"], episode=rec["episode"], start_id=rec["start_id"], part=rec["part"], cousin=rec["cousin"],
        outcome=rec["outcome"], counted=rec["counted"], max_stage=ev["max_stage"], stage_timeline=tl,
        claim=rec["claim"], false_claim=rec["false_claim"], ended_by=rec["ended_by"],
        limit_hit=limit, limit_why=("brain_calls" if decisions and decisions > budget.get("brain_calls", 1e9)
                                    else rec["ended_by"] if limit else None),
        final=ev["final"], max_depth_mm=ev["max_depth_mm"], env_commands=rec["n_commands"], tool_calls=by_tool,
        decisions=decisions, decisions_source="runner" if tok.get("decisions") else "call-log turns (gap < 1 s)",
        images=by_tool.get("env_observe_image", 0), peak_force_N=round(peak, 2),
        flags=[f"{f['monitor']}: {f['reason']}" for f in flags],
        envelope_violations=[v["reason"] for v in (rec["envelope"]["violations"] or [])],
        sim_s=rec["sim_s"], wall_s=round(span, 1), env_wall_s=rec["wall_s"], tool_s=round(tool_s, 1),
        think_s=round(think_s, 1), realtime_equiv_s=round(rec["sim_s"] + think_s, 1),
        tokens=tok.get("tokens"), tokens_source=tok.get("source"), watcher_tokens=tok.get("watcher_tokens"),
        wake_events=events, file=rec_line["file"], _calls=win)


@tool("practice_ledger",
      "Watcher and reporting tool for a practice run (never for the actor). One row per episode: rubric stage reached, "
      "outcome, counted, claim and false claim, limit hit (apart from failures), env commands, brain decisions, tool "
      "calls, images, peak force, sim s, wall s (physics paused while the brain thinks), think s and the real-time "
      "equivalent (sim + think), tokens; the code monitor's wake events (stuck, anomaly, near_promotion, episode_end); "
      "D aggregates with Clopper-Pearson intervals; and, with paired_with, an exact McNemar test on shared starts. "
      "episode=<start_id> also writes that episode's watcher packet (tool calls and results, images; no actor reasoning).",
      {"type": "object", "required": ["run_id"], "properties": {
          **_RUN,
          "episode": {"type": "string", "description": "start id: also build this episode's watcher packet"},
          "paired_with": {"type": "string", "description": "<brain>/<run_id> of the paired run (same starts)"},
          "actor": {"type": "string", "description": "only episodes of this env actor (default all but shadows)"}}})
def practice_ledger(run_id: str, brain: str = "claude", episode: str | None = None, paired_with: str | None = None,
                    actor: str | None = None) -> dict:
    rd = _run_dir(brain, run_id)
    m = json.loads((rd / "manifest.json").read_text()) if (rd / "manifest.json").exists() else {}
    budget = (m.get("budget") or {}).get("per_episode", BASELINE_BUDGET)
    lines = []
    f = rd / "env" / "episodes.jsonl"
    if f.exists():
        for ln in f.read_text().splitlines():
            r = json.loads(ln)
            if r.get("privileged") or str(r.get("actor", "")).startswith(("selftest", "test-", "obs-")):
                continue
            if actor and r.get("actor") != actor:
                continue
            lines.append(r)
    if m.get("start_set") == "heldout":
        done = {r["start_id"] for r in lines}
        left = [s for s in m.get("start_ids", []) if s not in done]
        if left:
            raise PermissionError(f"held-out run {brain}/{run_id} still has {len(left)} unattempted starts: its outcomes "
                                  "stay with the evaluator log until every held-out start has been attempted")
    tf = rd / "practice" / "tokens.json"
    tokens = json.loads(tf.read_text()) if tf.exists() else {}
    calls = _calls_for({r["session"] for r in lines})
    from twin.env import FORCE_CAP
    rows = [_episode_row(r, rd, calls, tokens, budget, FORCE_CAP) for r in lines]
    # near-promotion: a streak of counted successes on one part
    streak: dict[str, int] = {}
    for r in rows:
        streak[r["part"]] = streak.get(r["part"], 0) + 1 if r["counted"] else 0
        if streak[r["part"]] >= WATCHER["near_promotion_streak"]:
            r["wake_events"].insert(-1, dict(kind="near_promotion", why=f"{streak[r['part']]} counted successes in a row on {r['part']}"))
    scored = [r for r in rows if r["outcome"] not in ("abandoned", "error")]
    n = len(scored)
    succ = sum(r["counted"] for r in scored)
    limit = sum(r["limit_hit"] for r in scored)
    done_claims = [r for r in scored if r["claim"] == "success"]
    tok = [r["tokens"] for r in scored if r["tokens"] is not None]
    agg = dict(
        episodes=n, counted_successes=succ, success_rate=round(succ / n, 3) if n else None,
        success_ci95=_cp(succ, n) if n else None, limit_reached=limit,
        failures_by_outcome={o: sum(1 for r in scored if r["outcome"] == o and not r["limit_hit"])
                             for o in sorted({r["outcome"] for r in scored}) if not o.startswith("success")},
        stage_reached={s: sum(1 for r in scored if r["max_stage"] == s)
                       for s in ("no_approach", "contact", "lifted", "above_target", "seated")},
        false_done=dict(false=sum(r["false_claim"] for r in done_claims), claimed_done=len(done_claims),
                        rate=round(sum(r["false_claim"] for r in done_claims) / len(done_claims), 3) if done_claims else None),
        decisions_per_success=round(sum(r["decisions"] or 0 for r in scored) / succ, 1) if succ else None,
        tokens_per_success=round(sum(tok) / succ) if succ and len(tok) == n else None,
        tokens_total=sum(tok) if tok else None,
        time_per_success_s=dict(paused_physics_wall=round(sum(r["wall_s"] for r in scored) / succ, 1) if succ else None,
                                realtime_equiv=round(sum(r["realtime_equiv_s"] for r in scored) / succ, 1) if succ else None,
                                sim=round(sum(r["sim_s"] for r in scored) / succ, 1) if succ else None),
        brain_share=1.0 if not any(r["tool_calls"].keys() - {"env_reset", "env_move", "env_gripper", "env_wait",
                                                             "env_observe", "env_observe_image", "env_status",
                                                             "env_finish", "env_charge_tokens"} for r in scored)
        else "mixed: routine calls present; compute per call from the call log",
        note="tokens per success need every episode's tokens (runner record or practice/tokens.json); "
             "brain_share = 1.0 when every command came from a brain decision (no routine)")
    chk = None
    if m:
        v = _verify(rd)
        chk = dict(ok=v["ok"], changed=v["changed"], dry_run=bool(v["manifest"].get("dry_run")))
    agg["scorable"] = bool(chk and chk["ok"] and not chk["dry_run"])
    agg["scorable_why"] = ("no manifest" if not chk else "dry run: harness test, never scored" if chk["dry_run"]
                           else f"stale manifest: {chk['changed']}" if not chk["ok"] else "frozen and unchanged")
    out = dict(run=f"{brain}/{run_id}", manifest=_rel(rd / "manifest.json") if m else None, manifest_check=chk,
               aggregates=agg, episodes=rows)
    if paired_with:
        pb, pr = paired_with.split("/", 1)
        other = {r["start_id"]: r for r in practice_ledger(pr, pb)["episodes"]}
        mine = {r["start_id"]: r for r in scored}
        shared = sorted(set(mine) & set(other))
        b = sum(1 for s in shared if mine[s]["counted"] and not other[s]["counted"])
        c = sum(1 for s in shared if not mine[s]["counted"] and other[s]["counted"])
        out["paired"] = dict(with_run=paired_with, shared_starts=len(shared), this_only=b, other_only=c,
                             mcnemar_exact_p=mcnemar_exact(b, c))
    win_by_start = {r["start_id"]: r.pop("_calls") for r in rows}
    (rd / "practice").mkdir(parents=True, exist_ok=True)
    (rd / "practice" / "ledger.json").write_text(json.dumps(out, indent=1, default=str))
    out["ledger"] = _rel(rd / "practice" / "ledger.json")
    if episode:
        row = next((r for r in rows if r["start_id"] == episode), None)
        if row is None:
            raise ValueError(f"no episode at start {episode!r} in {brain}/{run_id}")
        pk = [dict(t=round(c["t"], 2), tool=c["tool"], args={k: v for k, v in (c.get("args") or {}).items()
                                                              if k not in ("brain", "run_id", "session")},
                   result=c.get("result")) for c in win_by_start[episode]]
        pf = rd / "practice" / "watch" / f"{episode}.json"
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text(json.dumps(dict(episode=row, calls=pk,
                                      rules="watcher packet: tool calls, results, images and the episode record; "
                                            "no actor reasoning"), indent=1, default=str))
        out["packet"] = _rel(pf)
    out["episodes"] = [{k: r[k] for k in ("start_id", "part", "outcome", "counted", "max_stage", "claim", "false_claim",
                                          "limit_hit", "env_commands", "decisions", "images", "sim_s", "wall_s",
                                          "realtime_equiv_s", "peak_force_N", "tokens")}
                       | dict(wake=[e["kind"] for e in r["wake_events"]]) for r in rows]
    return out


# ================================================================================================ clip
@tool("practice_clip",
      "For people and watchers (not the actor): an H.264 clip of a finished practice episode, replayed from its frozen "
      "start and command log in a shadow env, with the same belief overlays as env_observe_image.",
      {"type": "object", "required": ["run_id", "start_id"], "properties": {
          **_RUN, "start_id": {"type": "string"},
          "fps": {"type": "number", "description": "frames per sim second (default 10)"},
          "speed": {"type": "number", "description": "playback speed vs sim time (default 1)"},
          "views": {"type": "array", "items": {"type": "string", "enum": ["hand", "hama1", "hama2"]}},
          "title": {"type": "string"}}})
def practice_clip(run_id: str, start_id: str, brain: str = "claude", fps: float = 10.0, speed: float = 1.0,
                  views: list | None = None, title: str = "") -> dict:
    from twin import practice_view as PV
    rd = _run_dir(brain, run_id)
    lines = [json.loads(x) for x in (rd / "env" / "episodes.jsonl").read_text().splitlines()]
    hit = [r for r in lines if r["start_id"] == start_id and not r.get("privileged")]
    if not hit:
        raise ValueError(f"no finished episode at {start_id} in {brain}/{run_id}")
    rec = json.loads((REPO / hit[-1]["file"]).read_text())
    if rec["start_set"] == "heldout":
        raise PermissionError("held-out episodes are not replayed into clips during the run")
    out = rd / "practice" / "clips" / f"{start_id}.mp4"
    res = PV.episode_clip(rec, rd, out, fps=float(fps), views=tuple(views or ("hand", "hama1")), speed=float(speed),
                          title=title)
    return res
