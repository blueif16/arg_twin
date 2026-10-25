"""Harness tools for the LIBERO playground (installed 2026-10-25 by runs/orchestrator/libero-proof/install.py from playgrounds/libero/harness_tools_staged.py).

Move to harness/tools/libero.py after the Claude pilot runs end (their manifests hash harness/; nothing in harness/,
twin/ or playbook/ may change while they run). Same shape as harness/tools/env.py: the simulator lives in a session
daemon (playgrounds/libero/server.py `serve`, under .venv-libero), one per session, reached over a Unix socket; the
tools never import LIBERO or hold sim state. Schemas use only the gemini_schema subset (type, description, enum,
properties, required, items, default -> description), so Tool.declaration() passes them to Argon unchanged.

  libero_open      start a session on a split (practice | feasibility | scored); budget frozen per run
  libero_episodes  the split's episode ids (task id, init state index, instruction); never object poses
  libero_reset     begin an episode: instruction, conventions, budget and the first observation
  libero_move_to   absolute EEF target (x y z, rot6d r00..r21, gripper) + note; returns the step count and the next
                   observation (agentview + wrist PNG paths are attached as images by the brain runner)
  libero_observe   the current observation again
  libero_done      claim the task is finished (summary, hindsight); scored/feasibility: outcome not returned
  libero_give_up   stop (reason, hindsight)
  libero_status    budget used / left
  libero_close     end the session
  libero_charge    brain runner only: one model call's tokens and cost against the frozen per-episode budget

Proposal for registration: brains get open/episodes/reset/move_to/observe/done/give_up/status/close; libero_charge
is called by harness/brains (like env_charge_tokens) and should be hidden from the brain's tool list.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from harness.registry import REPO, tool

LIBERO_PY = REPO / ".venv-libero" / "bin" / "python"
SERVER = REPO / "playgrounds" / "libero" / "server.py"
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
SPLITS = ("practice", "feasibility", "scored")
DEFAULT_BUDGET = dict(max_llm_calls=80, max_steps=900, max_usd=30.0, max_output_tokens_per_call=16000)
LABELS = ("x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper")
_RUN = {"brain": {"type": "string", "default": "argon", "description": "argon or claude (stand-in)"},
        "run_id": {"type": "string", "description": "runs/libero/<run_id>/"}}
_SES = {"session": {"type": "string", "description": "session id returned by libero_open"}, **_RUN}
_HINDSIGHT = {"type": "string", "description": "What you know now that you wish you had known at the start: concrete, "
                                               "transferable facts about this rig, task or embodiment; 'none' if "
                                               "nothing qualifies."}


def _name(v, what):
    if not isinstance(v, str) or not NAME_RE.match(v) or v.startswith("."):
        raise ValueError(f"bad {what}: {v!r}")
    return v


def _run_dir(run_id) -> Path:
    return REPO / "runs" / "libero" / _name(run_id, "run_id")


def _sock(run_id, session) -> Path:
    p = _run_dir(run_id) / "sessions" / _name(session, "session") / "env.sock"
    if not p.exists():
        raise FileNotFoundError(f"session {session!r} is not running (closed, idle-timed-out or never opened)")
    return p


def _call(run_id, session, req: dict) -> dict:
    from playgrounds.libero.client import call_socket
    return call_socket(str(_sock(run_id, session)), req)


def frozen_budget(run_id: str, budget: dict | None) -> dict:
    f = _run_dir(run_id) / "budget.json"
    if f.exists():
        frozen = json.loads(f.read_text())["budget"]
        if budget and {k: budget.get(k, frozen.get(k)) for k in frozen} != frozen:
            raise ValueError(f"the budget of run {run_id} is frozen at {frozen}; a different budget needs a new run")
        return frozen
    b = dict(DEFAULT_BUDGET, **(budget or {}))
    if set(b) - set(DEFAULT_BUDGET):
        raise ValueError(f"unknown budget keys {sorted(set(b) - set(DEFAULT_BUDGET))}")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(dict(budget=b, frozen_at=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    return b


@tool("libero_open",
      "Open a LIBERO session (a simulator process you reach only through libero_* tools). split: practice "
      "(LIBERO-90; outcomes shown after each episode), feasibility or scored (LIBERO-10; outcomes go to the evaluator "
      "log only; a scored episode is attempted once per run). The per-episode budget (LLM calls, control steps, USD) "
      "is frozen per run at the first open. Returns the session id and the action conventions.",
      {"type": "object", "required": ["run_id"], "properties": {
          **_RUN, "split": {"type": "string", "enum": list(SPLITS), "default": "practice"},
          "budget": {"type": "object", "description": "max_llm_calls, max_steps, max_usd, max_output_tokens_per_call "
                                                      "(first open of a run only)"},
          "actor": {"type": "string", "default": "brain", "description": "who acts (logged; scored once-only key)"}}})
def libero_open(run_id: str, brain: str = "argon", split: str = "practice", budget: dict | None = None,
                actor: str = "brain") -> dict:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    b = frozen_budget(run_id, budget)
    sid = time.strftime("s%Y%m%d-%H%M%S") + f"-{os.getpid() % 100000}"
    d = _run_dir(run_id) / "sessions" / sid
    d.mkdir(parents=True, exist_ok=True)
    cfg = dict(run_id=run_id, actor=_name(actor, "actor"), brain=_name(brain, "brain"), split=split, budget=b)
    sock = d / "env.sock"
    log = open(d / "daemon.log", "ab")
    env = {k: v for k, v in os.environ.items() if not re.search(r"KEY|TOKEN|SECRET", k, re.I)}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.Popen([str(LIBERO_PY), str(SERVER), "serve", "--socket", str(sock), "--config", json.dumps(cfg)],
                     cwd=str(REPO), stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    for _ in range(1200):
        if sock.exists():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError(f"session daemon did not start; see {(d / 'daemon.log').relative_to(REPO)}")
    rep = _call(run_id, sid, {"op": "task"})
    return dict(session=sid, split=split, budget=b, conventions=rep.get("conventions"),
                next="libero_episodes to list episode ids, then libero_reset {session, task_id, init_state}")


@tool("libero_episodes", "Episode ids of a split: task id, init-state index and instruction. Never object poses.",
      {"type": "object", "properties": {"split": {"type": "string", "enum": list(SPLITS), "default": "practice"}}})
def libero_episodes(split: str = "practice") -> dict:
    sp = json.loads((REPO / "playgrounds" / "libero" / "splits.json").read_text())["splits"]
    if split not in sp:
        raise ValueError(f"split {split!r} is not declared in playgrounds/libero/splits.json yet")
    return dict(split=split, suite=sp[split].get("suite"), status=sp[split].get("status"),
                episodes=sp[split]["episodes"])


@tool("libero_reset",
      "Begin an episode of the session's split at (task_id, init_state). Returns the instruction, conventions, budget "
      "and the first observation (state text and agentview + wrist images). Resetting mid-episode abandons it.",
      {"type": "object", "required": ["session", "run_id", "task_id", "init_state"],
       "properties": {**_SES, "task_id": {"type": "integer"}, "init_state": {"type": "integer"}}})
def libero_reset(session: str, run_id: str, task_id: int, init_state: int, brain: str = "argon") -> dict:
    return _call(run_id, session, {"op": "reset", "task_id": int(task_id), "init_state": int(init_state)})


@tool("libero_move_to",
      "Move to absolute Cartesian end-effector targets in the world frame (m; rot6d = first two rotation-matrix "
      "columns; gripper 0 open, 1 close). Unnamed dimensions hold their current value. The motion is a straight line "
      "at the declared per-step limit (0.01 m, 0.05 rot6d, 0.1 gripper per 20 Hz step), at most 10 s; approvers clamp "
      "out-of-bounds and too-fast steps. Bounds: x, y [-0.6, 0.6], z [0.42, 1.6], rot6d [-1, 1], gripper [0, 1]. "
      "Returns the step count and the next observation; the episode may end during the move.",
      {"type": "object", "required": ["session", "run_id", "targets", "note"], "properties": {
          **_SES,
          "targets": {"type": "object", "description": "Map of dimension name to value. Valid names: "
                                                       + ", ".join(LABELS)},
          "note": {"type": "string", "description": "What you observe now and why you chose this motion (1-2 "
                                                    "sentences)."}}})
def libero_move_to(session: str, run_id: str, targets: dict, note: str, brain: str = "argon") -> dict:
    return _call(run_id, session, {"op": "move_to", "targets": targets, "note": note})


@tool("libero_observe", "The current observation: eef_state (10 labelled values), gripper_qpos and the agentview and "
                        "wrist images.",
      {"type": "object", "required": ["session", "run_id"], "properties": {**_SES}})
def libero_observe(session: str, run_id: str, brain: str = "argon") -> dict:
    return _call(run_id, session, {"op": "observe"})


@tool("libero_done", "Declare the task finished. The episode ends; the frozen LIBERO predicate judges success "
                     "(practice: the outcome is returned; feasibility/scored: recorded only).",
      {"type": "object", "required": ["session", "run_id", "summary", "hindsight"],
       "properties": {**_SES, "summary": {"type": "string"}, "hindsight": _HINDSIGHT}})
def libero_done(session: str, run_id: str, summary: str, hindsight: str, brain: str = "argon") -> dict:
    return _call(run_id, session, {"op": "done", "summary": summary, "hindsight": hindsight})


@tool("libero_give_up", "Stop trying; the task cannot be completed. The episode ends.",
      {"type": "object", "required": ["session", "run_id", "reason", "hindsight"],
       "properties": {**_SES, "reason": {"type": "string"}, "hindsight": _HINDSIGHT}})
def libero_give_up(session: str, run_id: str, reason: str, hindsight: str, brain: str = "argon") -> dict:
    return _call(run_id, session, {"op": "give_up", "reason": reason, "hindsight": hindsight})


@tool("libero_status", "Session and episode state: calls, control steps, tokens and USD used against the frozen budget.",
      {"type": "object", "required": ["session", "run_id"], "properties": {**_SES}})
def libero_status(session: str, run_id: str, brain: str = "argon") -> dict:
    return _call(run_id, session, {"op": "status"})


@tool("libero_close", "Close the session (an unfinished episode is recorded as abandoned).",
      {"type": "object", "required": ["session", "run_id"], "properties": {**_SES}})
def libero_close(session: str, run_id: str, brain: str = "argon") -> dict:
    return _call(run_id, session, {"op": "close"})


@tool("libero_charge",
      "Brain runner only (not the actor): account one model call (tokens, USD) against the frozen per-episode budget. "
      "Call with check_only=true before a model call to learn whether the call cap or spend cap allows it; a refusal "
      "ends the episode as limit_reached.",
      {"type": "object", "required": ["session", "run_id"], "properties": {
          **_SES, "check_only": {"type": "boolean", "default": False},
          "reserve_usd": {"type": "number", "default": 0.0},
          "usage": {"type": "object", "description": "input, cached, output, thought, total tokens"},
          "usd": {"type": "number", "default": 0.0}}})
def libero_charge(session: str, run_id: str, check_only: bool = False, reserve_usd: float = 0.0,
                  usage: dict | None = None, usd: float = 0.0, brain: str = "argon") -> dict:
    if check_only:
        return _call(run_id, session, {"op": "begin_call", "reserve_usd": float(reserve_usd)})
    return _call(run_id, session, {"op": "charge", "usage": usage or {}, "usd": float(usd)})
