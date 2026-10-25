"""Practice-env tools: the one door from a brain to the simulator (skill playbook/skills/sim-integrity).

Every brain uses the same narrow API. The simulator lives in a session daemon (twin/env.py `serve`), one per session,
reached over a Unix socket; tool calls never hold MjModel/MjData. Commands: absolute TCP target with a speed and a
force ceiling enforced in env code, gripper, wait; observations: robot state, F/T in the base frame, gripper width,
time. Episodes start only from frozen starts (runs/<brain>/<run_id>/starts/); held-out starts are attempted once per
run and their outcomes go to the evaluator log, not back to the actor.

  env_open      start a session (start set practice | heldout, cousin, budget frozen per run)
  env_starts    ids of the frozen starts of a set (ids and parts only, never their offsets)
  env_reset     begin an episode at a start: task (nominal hole, nominal grasp, limits) + first observation
  env_move      absolute TCP target: pos [m], quat [w,x,y,z], speed [m/s], force_max [N], stop_force [N]
  env_gripper   open | close | width
  env_wait      advance time
  env_observe   robot state, F/T, gripper width, time
  env_status    budget used, episode state
  env_finish    end the episode (optional claim); practice: the outcome; held-out: recorded only
  env_close     end the session
  env_charge_tokens  the brain runner (not the actor) reports tokens spent, for the frozen token budget
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from harness.registry import REPO, tool

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_RUN = {"brain": {"type": "string", "default": "claude", "description": "claude (stand-in) or argon"},
        "run_id": {"type": "string", "default": "run-standin-1", "description": "runs/<brain>/<run_id>/"}}
_SES = {"session": {"type": "string", "description": "session id returned by env_open"},
        **_RUN}
DEFAULT_BUDGET = dict(max_commands=200, max_sim_s=120.0, max_wall_s=1800.0, max_tokens=None)


def _name(v, what):
    if not isinstance(v, str) or not NAME_RE.match(v) or v.startswith("."):
        raise ValueError(f"bad {what}: {v!r}")
    return v


def _env_dir(brain, run_id) -> Path:
    return REPO / "runs" / _name(brain, "brain") / _name(run_id, "run_id") / "env"


def _sock(brain, run_id, session) -> Path:
    p = _env_dir(brain, run_id) / "sessions" / _name(session, "session") / "env.sock"
    if not p.exists():
        raise FileNotFoundError(f"session {session!r} is not running (closed, idle-timed-out or never opened)")
    return p


def _call(brain, run_id, session, req: dict) -> dict:
    from twin.env import call_socket
    return call_socket(str(_sock(brain, run_id, session)), req)


def frozen_budget(brain: str, run_id: str, budget: dict | None) -> dict:
    """Budget is frozen per run: the first env_open writes it, later opens must match or omit it."""
    f = _env_dir(brain, run_id) / "budget.json"
    if f.exists():
        frozen = json.loads(f.read_text())["budget"]
        if budget and {k: budget.get(k, frozen.get(k)) for k in frozen} != frozen:
            raise ValueError(f"the budget of {brain}/{run_id} is frozen at {frozen} ({f.relative_to(REPO)}); "
                             "a different budget needs a new run")
        return frozen
    b = dict(DEFAULT_BUDGET, **(budget or {}))
    unknown = set(b) - set(DEFAULT_BUDGET)
    if unknown:
        raise ValueError(f"unknown budget keys {sorted(unknown)}; known: {sorted(DEFAULT_BUDGET)}")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(dict(budget=b, frozen_at=time.strftime("%Y-%m-%d %H:%M"),
                                 note="per episode; frozen before the first episode of this run"), indent=1))
    return b


@tool("env_open",
      "Open a practice-env session (a simulator process you reach only through env_* tools). start_set: practice "
      "(outcomes shown after each episode) or heldout (each start once per run; outcomes go to the evaluator log only). "
      "cousin: physics variant id (c000 = nominal twin). The per-episode budget (commands, sim s, wall s, tokens) is "
      "frozen per run at the first open. Returns the session id, conventions and limits.",
      {"type": "object", "properties": {
          **_RUN,
          "start_set": {"type": "string", "enum": ["practice", "heldout"], "default": "practice"},
          "cousin": {"type": "string", "default": "c000"},
          "budget": {"type": "object", "description": "max_commands, max_sim_s, max_wall_s, max_tokens (first open only)"},
          "actor": {"type": "string", "default": "brain", "description": "who acts (logged; held-out once-only key)"}}})
def env_open(brain: str = "claude", run_id: str = "run-standin-1", start_set: str = "practice", cousin: str = "c000",
             budget: dict | None = None, actor: str = "brain") -> dict:
    if start_set not in ("practice", "heldout"):
        raise ValueError("start_set must be practice or heldout (red-team starts: run_actor)")
    b = frozen_budget(brain, run_id, budget)
    sid = time.strftime("s%Y%m%d-%H%M%S") + f"-{os.getpid() % 100000}"
    d = _env_dir(brain, run_id) / "sessions" / sid
    d.mkdir(parents=True, exist_ok=True)
    cfg = dict(brain=brain, run_id=run_id, start_set=start_set, cousin=_name(cousin, "cousin"), budget=b,
               actor=_name(actor, "actor"), session=sid)
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    sock = d / "env.sock"
    log = open(d / "daemon.log", "ab")
    subprocess.Popen([sys.executable, "-m", "twin.env", "serve", "--socket", str(sock), "--config", str(d / "config.json")],
                     cwd=str(REPO), stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True,
                     env={**{k: v for k, v in os.environ.items() if not re.search(r"KEY|TOKEN|SECRET", k)},
                          "PYTHONPATH": str(REPO), "MUJOCO_GL": os.environ.get("MUJOCO_GL", "glfw")})
    for _ in range(600):
        if sock.exists():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError(f"session daemon did not start; see {(d / 'daemon.log').relative_to(REPO)}")
    rep = _call(brain, run_id, sid, {"op": "task"})
    return dict(session=sid, start_set=start_set, cousin=cousin, budget=b, conventions=rep.get("conventions"),
                limits=rep.get("limits"), next="env_starts to list start ids, then env_reset {session, start_id}")


@tool("env_starts",
      "Ids and parts of the frozen starts in a set (practice | heldout). Never returns the starts' offsets.",
      {"type": "object", "properties": {**_RUN, "start_set": {"type": "string", "default": "practice"},
                                        "part": {"type": "string", "description": "filter, e.g. 'round peg 2'"}}})
def env_starts(brain: str = "claude", run_id: str = "run-standin-1", start_set: str = "practice",
               part: str | None = None) -> dict:
    from twin import starts as S
    if start_set not in ("practice", "heldout"):
        raise ValueError("start_set must be practice or heldout")
    st, _ = S.load(start_set, REPO / "runs" / _name(brain, "brain") / _name(run_id, "run_id") / "starts")
    by: dict[str, list[str]] = {}
    for sid, s in sorted(st.items()):
        if part and s["part"] != part:
            continue
        by.setdefault(s["part"], []).append(sid)
    out = dict(start_set=start_set, n=sum(len(v) for v in by.values()), by_part=by)
    if start_set == "heldout":
        out["note"] = (f"each held-out start is attempted once per run; the first {S.HELDOUT_SUBSET} per part are the "
                       "pre-declared smaller set")
    return out


@tool("env_reset",
      "Begin an episode at a frozen start (the part is already in the hand above its hole). Returns the task (part, "
      "nominal hole entry pose, nominal grasp, limits, budget) and the first observation. Resetting mid-episode "
      "abandons the current episode (recorded as abandoned).",
      {"type": "object", "required": ["session", "start_id"], "properties": {**_SES, "start_id": {"type": "string"}}})
def env_reset(session: str, start_id: str, brain: str = "claude", run_id: str = "run-standin-1") -> dict:
    return _call(brain, run_id, session, {"op": "reset", "start_id": start_id})


@tool("env_move",
      "Move the TCP to an absolute target pose (world = FR3 base frame, m; quat w,x,y,z; omitted quat keeps the "
      "orientation). The env interpolates at `speed` (capped at 0.25 m/s), filters like the real controller (0.2 s) and "
      "clamps the commanded spring force to force_max (capped at 15 N). stop_force ends the move at the first 1 ms "
      "control tick where |F| reaches it (a guarded move); a guarded approach is slowed to the guard speed for that "
      "threshold (task limits) so the peak stays within max(0.5 N, 20%) of it. Returns why it stopped (reached | "
      "settled | stop_force | timeout), the peak force, any overshoot and the observation.",
      {"type": "object", "required": ["session", "pos"], "properties": {
          **_SES, "pos": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
          "quat": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
          "speed": {"type": "number", "description": "m/s, default 0.02, cap 0.25"},
          "force_max": {"type": "number", "description": "N, default 10, cap 15"},
          "stop_force": {"type": "number", "description": "N: stop when |F| reaches it"},
          "timeout": {"type": "number", "description": "s, default distance/speed + 2, max 20"}}})
def env_move(session: str, pos: list, quat: list | None = None, speed: float | None = None,
             force_max: float | None = None, stop_force: float | None = None, timeout: float | None = None,
             brain: str = "claude", run_id: str = "run-standin-1") -> dict:
    req = {"op": "move", "pos": pos, "quat": quat, "speed": speed, "force_max": force_max, "stop_force": stop_force,
           "timeout": timeout}
    return _call(brain, run_id, session, {k: v for k, v in req.items() if v is not None})


@tool("env_gripper",
      "Gripper: open, close (60 N per finger, fixed) or width (finger opening in m).",
      {"type": "object", "required": ["session", "action"], "properties": {
          **_SES, "action": {"type": "string", "enum": ["open", "close", "width"]}, "width": {"type": "number"}}})
def env_gripper(session: str, action: str, width: float | None = None, brain: str = "claude",
                run_id: str = "run-standin-1") -> dict:
    req = {"op": "gripper", "action": action}
    if width is not None:
        req["width"] = width
    return _call(brain, run_id, session, req)


@tool("env_wait", "Advance time with the current targets held (max 5 s per call).",
      {"type": "object", "required": ["session", "seconds"], "properties": {**_SES, "seconds": {"type": "number"}}})
def env_wait(session: str, seconds: float, brain: str = "claude", run_id: str = "run-standin-1") -> dict:
    return _call(brain, run_id, session, {"op": "wait", "seconds": seconds})


@tool("env_observe", "Current observation: TCP pose and twist, joints, wrench on the tool (base frame), gripper width, "
                     "time. Does not advance time.",
      {"type": "object", "required": ["session"], "properties": _SES})
def env_observe(session: str, brain: str = "claude", run_id: str = "run-standin-1") -> dict:
    return _call(brain, run_id, session, {"op": "observe"})


@tool("env_status", "Session and episode state: start, commands and tokens used against the frozen budget.",
      {"type": "object", "required": ["session"], "properties": _SES})
def env_status(session: str, brain: str = "claude", run_id: str = "run-standin-1") -> dict:
    return _call(brain, run_id, session, {"op": "status"})


@tool("env_finish",
      "End the episode. claim: success | failure (recorded next to the evaluator's verdict; false claims are counted). "
      "The env holds the last command 0.5 s, then scores from simulator state. Practice: returns the outcome, stage "
      "reached, final depth/lateral and any monitor flags. Held-out: returns only that it was recorded.",
      {"type": "object", "required": ["session"], "properties": {
          **_SES, "claim": {"type": "string", "enum": ["success", "failure"]}}})
def env_finish(session: str, claim: str | None = None, brain: str = "claude", run_id: str = "run-standin-1") -> dict:
    req = {"op": "finish"}
    if claim is not None:
        req["claim"] = claim
    return _call(brain, run_id, session, req)


@tool("env_close", "Close the session (an unfinished episode is recorded as abandoned).",
      {"type": "object", "required": ["session"], "properties": _SES})
def env_close(session: str, brain: str = "claude", run_id: str = "run-standin-1") -> dict:
    return _call(brain, run_id, session, {"op": "close"})


@tool("env_charge_tokens",
      "For the brain runner (not the actor): report model tokens spent in the current episode; the episode ends "
      "limit_reached when the frozen token budget is used up.",
      {"type": "object", "required": ["session", "tokens"], "properties": {**_SES, "tokens": {"type": "integer"}}})
def env_charge_tokens(session: str, tokens: int, brain: str = "claude", run_id: str = "run-standin-1") -> dict:
    return _call(brain, run_id, session, {"op": "charge_tokens", "tokens": tokens})
