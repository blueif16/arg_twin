"""Same-start trials with observer feedback: a practice brain that gets a second (third, fourth) try.

Design (orchestrator brief 2026-10-25, run-argon-trials-1; scripts in runs/argon/trials/):

  trial k on a start  = one GeminiBrain-style run with a short budget (30 brain calls, 120 s sim).
  between trials      = `reflect()`: Argon (high thinking) reads the failed trial's trace -- the brain's commands and
                        tool results, its own images, the outcome report, the rubric-stage timeline and monitor flags
                        the env reports -- and writes at most 6 lines (status, cause, fix, guide). The notes of every
                        earlier trial on that start go at the top of the next trial's task (a Reflexion-style memory).
  during a trial      = `Checkpointer`: gemini-3.8-flash is a fast observer. It wakes when the brain-visible progress
                        stage changes (first plate contact, travel past it) or after 8 brain calls without stage or
                        depth progress, sees the last few steps as text, and gives a 1-2 line nudge that `TrialBrain`
                        injects as a user step before the brain's next turn.

Information honesty: every observer input is built from what the brain itself received (tool results, its images,
its text) plus the end-of-episode report the env gives the actor on practice starts, and the stage timeline. No
observer ever reads the episode record's start, episode_geometry, physics or any other sim truth. The env does not
report the rubric stage mid-episode, so the in-trial trigger uses a brain-visible proxy (`Progress`): first plate
contact from the wrench, then TCP travel past that contact height.

`TrialBrain` is a new class. GeminiBrain and every existing runner are untouched; with `checkpoint=None` the loop
behaves like runs/argon/pilot/pilot_run.py's RetryBrain (client-side timeouts retried on the same chain).
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from harness.brains.gemini import GeminiBrain, output_text, step_summary
from harness.brains.prompt import build_system_instruction
from harness.brains.record import cost_usd, request_digest, usage_dict
from harness.registry import REPO

FLASH = "gemini-3.8-flash"
ARGON = "gemini-4-argon"
# introductory list prices per 1M tokens; Flash is an estimate (not in record.PRICES), used only for reporting
FLASH_PRICE = {"input": 0.30, "cached": 0.03, "output": 2.50}

# The guide: method lessons from the Claude pilot arms' trace audit, the Argon practice probe and the practice skill.
# Method knowledge only, never values of any start's truth. Observers quote one line as "the guide".
GUIDE = [
    "G1 Measure tip against hole before descending: rest the tip on the plate short of the hole (13-27 mm along -hand x) and fit both in ONE wrist image; aim from hole-minus-tip in that image (17/24 first-descent seats).",
    "G2 Correct in-hand tilt first: a tilt above ~3 deg jams round pegs at 4-6 mm; read it from the fixed cameras at zoom 6 (part vs magenta axis) and the pressure centre of a light press, then turn the hand and wait 2-5 s for the rotation to finish.",
    "G3 Square pegs: at a chamfer stop, read the yaw-torque sign from a light press; a ~2 deg hand yaw in that direction drops it in.",
    "G4 Pull a jammed part gently (a hard lift flags force). On a jam deeper than 3 mm, opening the gripper let the part slide home (3/3).",
    "G5 Stalls end the episode exactly 40 commands after the last progress: a recovery must fit inside that window; once seated at a repeatable hard stop, release and finish.",
    "G6 Read the hole rim, not the dark blob: visual offsets run +-2 mm and often have the wrong sign; treat one visual read as a probe start, not the aim.",
    "G7 After a missed probe, look and re-aim from a new image instead of spiralling blind; torque sign flips between probes localise the hole in 3-4 probes.",
    "G8 Guard only the last 3-5 mm (stop_force 3-5 N caps speed at 2-3 mm/s); move unguarded to just above the expected contact. Depth from travel: ~20 mm past the plate touch is seated.",
    "G9 Use force_max >= 12 for free-space moves that turn the hand; force_max below stop_force makes a guarded move a sustained press with a readable torque.",
    "G10 Claim success only on named evidence (travel ~20 mm past the plate touch, F back to ~0 after release).",
]
GUIDE_TEXT = "\n".join(GUIDE)

_SKIP_ARGS = {"brain", "run_id", "session"}


def _is_timeout(e: Exception) -> bool:
    return type(e).__name__ in ("APITimeoutError", "ReadTimeout", "TimeoutException", "WriteTimeout",
                                "ConnectTimeout") or "timed out" in str(e).lower()


# ------------------------------------------------------------------------------------------------ trace as text

def _r(v, n=1):
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return v


def _obs_line(obs: dict) -> str:
    if not isinstance(obs, dict) or "tcp" not in obs:
        return ""
    p = [round(1000 * float(x), 1) for x in obs["tcp"]["pos"]]
    q = [round(float(x), 4) for x in obs["tcp"]["quat"]]
    f = [round(float(x), 2) for x in (obs.get("wrench") or {}).get("force_N", [])]
    tq = [round(float(x), 4) for x in (obs.get("wrench") or {}).get("torque_Nm", [])]
    s = f"t={_r(obs.get('t'), 2)} tcp_mm={p} quat={q} F={f} T={tq} grip={_r(1000 * float(obs.get('gripper_width_m', 0)), 2)}mm"
    if obs.get("episode_over"):
        s += " EPISODE_OVER"
    return s


def compact_result(name: str, res: Any) -> str:
    """One tool result as the brain saw it, shortened (no truth: this is the function_result the brain got)."""
    if not isinstance(res, dict):
        return json.dumps(res, default=str)[:400]
    if "error" in res and len(res) <= 2:
        return "ERROR " + str(res["error"])[:300]
    if name in ("env_move", "env_gripper", "env_wait"):
        r = res.get("result") or {}
        keep = {k: r[k] for k in ("stopped", "duration_s", "max_force_N", "target_error_mm", "stop_force",
                                  "speed_used", "force_max_used") if k in r}
        s = f"{json.dumps(keep)} | {_obs_line(res.get('obs') or res)}"
        if res.get("episode_over"):
            s += f" | EPISODE_OVER ended_by={res.get('ended_by')} report={json.dumps(res.get('report'), default=str)[:600]}"
        return s
    if name == "env_observe":
        return _obs_line(res)
    if name == "env_observe_image":
        return f"image={Path(str(res.get('image', ''))).name} t={res.get('t')} believed={json.dumps(res.get('believed'))}"
    if name == "env_reset":
        task = res.get("task") or {}
        keep = {k: task[k] for k in ("part", "hole", "grasp", "nominal", "tip_in_tcp_frame", "budget")
                if k in task}
        return f"task={json.dumps(keep, default=str)[:1500]} | first obs: {_obs_line(res.get('obs') or {})}"
    if name == "env_finish":
        return f"report={json.dumps(res.get('report'), default=str)[:800]}"
    if name in ("read_skill", "read_playbook_file"):
        return "(skill text)"
    if name == "env_open":
        return f"session={res.get('session')} budget={json.dumps(res.get('budget'))}"
    return json.dumps(res, default=str)[:300]


def compact_turn(t: dict, think_chars: int = 400, text_chars: int = 400) -> str:
    lines = [f"[call {t['turn']}]"]
    th = " ".join(t.get("thoughts") or [])
    if th:
        lines.append(f"  thought: {th[:think_chars]}")
    if t.get("text"):
        lines.append(f"  text: {t['text'][:text_chars]}")
    if t.get("nudge"):
        lines.append(f"  (observer nudge injected after this call: {t['nudge'][:300]})")
    for c in t.get("tool_calls") or []:
        if "skipped" in c:
            lines.append(f"  {c['name']} SKIPPED ({c['skipped']})")
            continue
        args = {k: v for k, v in (c.get("args") or {}).items() if k not in _SKIP_ARGS}
        if c["name"] == "log_note":
            lines.append(f"  log_note: {json.dumps(args)[:500]}")
            continue
        lines.append(f"  {c['name']}({json.dumps(args)[:300]}) -> {compact_result(c['name'], c.get('result'))}")
    if t.get("error"):
        lines.append(f"  RUNNER ERROR: {str(t['error'])[:200]}")
    return "\n".join(lines)


def images_of(turns: list[dict]) -> list[Path]:
    out = []
    for t in turns:
        for c in t.get("tool_calls") or []:
            if c.get("name") == "env_observe_image" and isinstance(c.get("result"), dict) and c["result"].get("image"):
                p = REPO / c["result"]["image"]
                if p.is_file():
                    out.append(p)
    return out


# ------------------------------------------------------------------------------------------------ progress proxy

PROXY_STAGES = ("free_air", "plate_contact", "entering(>3mm past contact)", "deep(>12mm past contact)",
                "released_deep")


class Progress:
    """Brain-visible progress: first plate contact (F_z > 1 N pressing down) and TCP travel past that height."""

    def __init__(self):
        self.contact_z = None
        self.best_travel_mm = 0.0
        self.stage = 0
        self.over = False
        self.last_progress_call = 0
        self.n_commands = 0
        self.sim_t = 0.0

    def update(self, turn: dict) -> bool:
        """Returns True when the proxy stage went up on this call."""
        before = self.stage
        for c in turn.get("tool_calls") or []:
            res = c.get("result")
            if not isinstance(res, dict) or c.get("is_error"):
                continue
            name = c["name"]
            if name in ("env_move", "env_gripper", "env_wait"):
                self.n_commands += 1
            obs = res.get("obs") if name in ("env_move", "env_gripper", "env_wait", "env_reset") else (
                res if name == "env_observe" else None)
            if res.get("episode_over") or (isinstance(obs, dict) and obs.get("episode_over")) or name == "env_finish":
                self.over = True
            if not isinstance(obs, dict) or "tcp" not in obs:
                continue
            self.sim_t = float(obs.get("t", self.sim_t))
            z = float(obs["tcp"]["pos"][2])
            fz = float((obs.get("wrench") or {}).get("force_N", [0, 0, 0])[2])
            if name == "env_reset":
                continue
            r = res.get("result") if isinstance(res.get("result"), dict) else {}
            touched = fz > 1.0 or r.get("stopped") == "stop_force" or (
                name == "env_move" and float(r.get("max_force_N") or 0) > 1.2 and float(r.get("target_error_mm") or 0) > 2.0)
            if self.contact_z is None and touched:
                self.contact_z = z
                self.stage = max(self.stage, 1)
                self.last_progress_call = turn["turn"]
            if self.contact_z is not None:
                travel = 1000 * (self.contact_z - z)
                if travel > self.best_travel_mm + 1.0:
                    self.best_travel_mm = travel
                    self.last_progress_call = turn["turn"]
                if travel > 12:
                    self.stage = max(self.stage, 3)
                elif travel > 3:
                    self.stage = max(self.stage, 2)
                if name == "env_gripper" and (c.get("args") or {}).get("action") == "open" and self.stage >= 3:
                    self.stage = 4
        if self.stage > before:
            self.last_progress_call = turn["turn"]
        return self.stage > before

    def summary(self) -> str:
        cz = "none yet" if self.contact_z is None else f"{1000 * self.contact_z:.1f} mm TCP z"
        return (f"proxy stage {PROXY_STAGES[self.stage]}; first plate contact at {cz}; best TCP travel past it "
                f"{self.best_travel_mm:.1f} mm; env commands {self.n_commands}; sim t {self.sim_t:.1f} s")


# ------------------------------------------------------------------------------------------------ the brain

class TrialBrain(GeminiBrain):
    """GeminiBrain loop with client-timeout retries and an optional checkpoint hook (default off).

    checkpoint(turn, brain) -> str | None is called after each turn whose tool calls ran; a returned string is
    appended to the next input as a user_input step (the brain reads it before its next turn) and stored in the
    turn record as `nudge`.
    """
    retries = 3
    checkpoint: Callable[[dict, "TrialBrain"], str | None] | None = None
    runner_note = "harness/brains/trials.py TrialBrain"

    def create(self, inp, prev, system, decls):
        self._retry_log = []
        for attempt in range(self.retries + 1):
            t = time.monotonic()
            try:
                return super().create(inp, prev, system, decls)
            except Exception as e:  # noqa: BLE001
                if not _is_timeout(e) or attempt == self.retries:
                    raise
                self._retry_log.append(dict(attempt=attempt + 1, after_s=round(time.monotonic() - t, 1),
                                            error=f"{type(e).__name__}: {e}"[:300]))
                if self.transport:
                    self.transport.drain()

    def run(self, task: str) -> dict:
        cfg = self.cfg
        decls = self.tool_declarations if self.tool_declarations is not None else self.declarations()
        allowed = {d["name"] for d in decls if d.get("type") == "function"}
        system = self.system_instruction or build_system_instruction(
            self.brain_name, self.rec.run_id, budget=asdict(cfg.budget))
        meta = {"brain": self.brain_name, "model": cfg.model, "task": task, "caller": cfg.caller,
                "config": cfg.to_dict(), "n_tools": len(decls), "runner": self.runner_note, **self.meta_extra}
        self.rec.start(meta, system, decls)
        self.t0 = time.monotonic()
        inp: Any = [{"type": "text", "text": task}]
        prev = None
        stop, final_text, n = None, "", 0
        self.turns: list[dict] = []
        while stop is None:
            stop = self.budget_stop("model")
            if stop:
                break
            n += 1
            turn: dict[str, Any] = {"turn": n, "kind": "model", "t_start": time.time()}
            t = time.monotonic()
            ix = None
            self._retry_log = []
            try:
                ix = self.create(inp, prev, system, decls)
            except Exception as e:  # noqa: BLE001
                stop = "api_error" if type(e).__name__ != "BudgetStop" else str(e)
                turn["error"] = f"{type(e).__name__}: {e}"[:4000]
            turn["latency_s"] = round(time.monotonic() - t, 3)
            if self._retry_log:
                turn["retries"] = self._retry_log
            http = self.transport.drain() if self.transport else []
            turn["request"] = request_digest(next((h["request"] for h in http if h["method"] == "POST"
                                                   and h["path"].split("?")[0].endswith("/interactions")), None))
            turn["http"] = http
            self.totals["model_calls"] += 1
            if ix is not None:
                u = usage_dict(ix.usage)
                turn["usage"] = u
                self.totals["tokens"] += u["total"]
                self.totals["usd"] += cost_usd(cfg.model, u) or 0.0
                steps = ix.steps or []
                turn["interaction"] = {"id": ix.id, "status": ix.status, "steps": [step_summary(s) for s in steps]}
                turn["text"] = output_text(ix)
                turn["thoughts"] = [x for s in turn["interaction"]["steps"] for x in s.get("summary", [])]
                calls = [s for s in steps if s.type == "function_call"]
                if ix.status in ("failed", "cancelled"):
                    stop = f"status:{ix.status}"
                    turn["error"] = str(getattr(ix, "error", None) or "")[:4000] or None
                elif not calls:
                    stop = "done" if ix.status == "completed" else f"status:{ix.status}"
                    final_text = turn["text"]
                else:
                    t_tools = time.monotonic()
                    inp, turn["tool_calls"], stop = self.run_tools(calls, allowed)
                    turn["tools_s"] = round(time.monotonic() - t_tools, 3)
                    prev = ix.id
                    if stop is None and self.checkpoint is not None:
                        try:
                            nudge = self.checkpoint(turn, self)
                        except Exception as e:  # noqa: BLE001  an observer failure never ends a trial
                            nudge = None
                            turn["nudge_error"] = f"{type(e).__name__}: {e}"[:300]
                        if nudge:
                            inp = list(inp) + [{"type": "user_input", "content": [{"type": "text", "text": nudge}]}]
                            turn["nudge"] = nudge
            else:
                turn["usage"] = usage_dict(None)
            turn["totals"] = {**self.totals, "usd": round(self.totals["usd"], 6), "wall_s": round(self.elapsed(), 2)}
            self.rec.write_turn(turn)
            self.turns.append(turn)
        return self.rec.finish(stop or "done", final_text)


# ------------------------------------------------------------------------------------------------ observers

def _create(client, model, inp, system, gen, timeout_s, retries=1):
    last = None
    for attempt in range(retries + 1):
        t = time.monotonic()
        try:
            ix = client.interactions.create(model=model, input=inp, system_instruction=system,
                                            generation_config=gen, timeout=timeout_s)
            return ix, round(time.monotonic() - t, 2), attempt
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt == retries:
                raise
    raise last  # pragma: no cover


def _flash_cost(u: dict) -> float:
    fresh = max(u["input"] - u["cached"], 0)
    return round((fresh * FLASH_PRICE["input"] + u["cached"] * FLASH_PRICE["cached"]
                  + (u["output"] + u["thought"]) * FLASH_PRICE["output"]) / 1e6, 6)


NUDGE_SYSTEM = f"""You are a fast checkpoint observer for a robot-arm brain doing in-hand peg insertion in a MuJoCo twin.
You see only what the brain itself saw: its recent tool calls and results (TCP pose in mm, wrench, the believed
tip-vs-hole numbers from its image tool), its notes, and a brain-visible progress summary. You never know the true
hole or grasp; do not invent coordinates you cannot derive from the trace.

Write ONE or TWO short lines (max 220 characters total) telling the brain the single most useful next move or
correction, grounded in the trace and, where it fits, one line of the guide (cite it as G<n>). No preamble.

The guide:
{GUIDE_TEXT}
"""


class Checkpointer:
    """In-trial nudges from gemini-3.8-flash on brain-visible progress events (see module doc)."""

    def __init__(self, client, notes: list[str], log_path: Path, stall_calls: int = 8, max_nudges: int = 4,
                 last_k: int = 4, budget_calls: int = 30, budget_sim_s: float = 120.0):
        self.client = client
        self.notes = notes
        self.log_path = log_path
        self.stall_calls = stall_calls
        self.max_nudges = max_nudges
        self.last_k = last_k
        self.budget_calls = budget_calls
        self.budget_sim_s = budget_sim_s
        self.progress = Progress()
        self.history: list[dict] = []
        self.nudges: list[dict] = []
        self.last_nudge_call = 0
        self.usd = 0.0

    def __call__(self, turn: dict, brain: TrialBrain) -> str | None:
        self.history.append(turn)
        stage_before = self.progress.stage
        changed = self.progress.update(turn)
        p = self.progress
        if p.over or len(self.nudges) >= self.max_nudges:
            return None
        n = turn["turn"]
        reason = None
        if changed and p.stage >= 1:
            reason = f"stage {PROXY_STAGES[stage_before]} -> {PROXY_STAGES[p.stage]}"
        elif n - max(p.last_progress_call, self.last_nudge_call) >= self.stall_calls and n >= 3:
            reason = f"{n - max(p.last_progress_call, self.last_nudge_call)} brain calls without stage or depth progress"
        if reason is None:
            return None
        trace = "\n".join(compact_turn(t, 250, 250) for t in self.history[-self.last_k:])
        notes = "\n".join(self.notes) if self.notes else "(none: this is the first trial with observers)"
        prompt = (f"Checkpoint because: {reason}.\nBrain calls used: {n} of {self.budget_calls + 2} (setup turn "
                  f"included); {p.summary()}; sim budget {self.budget_sim_s:.0f} s.\n\nReflection notes from earlier "
                  f"trials on this start (what the brain was told):\n{notes}\n\nLast {self.last_k} brain calls:\n"
                  f"{trace}\n\nYour 1-2 line nudge:")
        rec = {"call": n, "reason": reason, "stage": PROXY_STAGES[p.stage], "best_travel_mm": round(p.best_travel_mm, 2),
               "sim_t": p.sim_t, "commands": p.n_commands, "prompt": prompt}
        try:
            ix, lat, retr = _create(self.client, FLASH, [{"type": "text", "text": prompt}], NUDGE_SYSTEM,
                                    {"thinking_level": "low", "max_output_tokens": 4096}, 90.0, retries=1)
            text = (output_text(ix) or "").strip()
            u = usage_dict(ix.usage)
            rec.update(text=text, latency_s=lat, usage=u, usd=_flash_cost(u), retries=retr, status=ix.status)
            self.usd += rec["usd"]
        except Exception as e:  # noqa: BLE001
            rec.update(text="", error=f"{type(e).__name__}: {e}"[:300])
            text = ""
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        if not text:
            return None
        text = " ".join(text.split())[:400]
        self.last_nudge_call = n
        self.nudges.append(rec)
        return f"[Observer checkpoint ({reason})] {text}"


REFLECT_SYSTEM = f"""You are the observer between trials of a robot-arm brain doing in-hand peg insertion in a MuJoCo
twin. The same start is reset for every trial, so the hole and the grasp are the same next time. You read the failed
trial's trace: the brain's own tool calls and results, its own camera images (composite: wrist 'hand' camera and two
fixed cameras, with BELIEF overlays: yellow grid, green believed hole, magenta believed part axis/tip, cyan pinch),
the outcome report the env gave it, the coarse rubric-stage timeline and monitor flags.

You do NOT know the true hole position or the true in-hand pose; only infer them from what the trace and the images
show, call such positions "estimated" (never "true"), and say how sure you are. Be concrete: name moves, offsets in hand or world mm derived from the trace, forces,
which call went wrong.

Write AT MOST 6 lines, plain text, exactly these labels:
STATUS: where it got to (stage, travel past contact, how it ended)
CAUSE: the likely cause of the failure
FIX: the specific fix for the next trial (what to do differently, in order, within 30 brain calls and 120 s sim)
GUIDE: one relevant line from the guide below, quoted (with its G number)
(optionally up to 2 more lines: KEEP: what worked and should be reused, e.g. a measured plate height or offset)

The guide:
{GUIDE_TEXT}
"""


def reflect(client, trial: int, turns: list[dict], outcome: dict, notes: list[str], out_path: Path,
            thinking: str = "high", n_images: int = 4, timeout_s: float = 900.0) -> dict:
    """Argon reflection on one failed trial -> {text, usage, usd, latency_s, ...}; the full input is saved."""
    trace = "\n".join(compact_turn(t) for t in turns)
    if len(trace) > 60000:
        trace = trace[:15000] + "\n...[middle of the trace cut]...\n" + trace[-45000:]
    imgs = images_of(turns)
    pick = (imgs[:1] + imgs[1:][-(n_images - 1):]) if len(imgs) > n_images else imgs
    prior = "\n".join(notes) if notes else "(none)"
    text = (f"Trial {trial} on this start failed. Outcome the env reported to the brain, with the rubric-stage "
            f"timeline (sim s, stage) and flags:\n{json.dumps(outcome, default=str)}\n\nNotes the brain already had "
            f"from earlier trials:\n{prior}\n\nThe brain's trace (one block per brain call):\n{trace}\n\n"
            f"The brain's own images follow ({len(pick)} of {len(imgs)}: {[p.name for p in pick]}).\n"
            f"Write your <=6 lines now.")
    content: list[dict] = [{"type": "text", "text": text}]
    for p in pick:
        content.append({"type": "text", "text": f"[image {p.name}]"})
        content.append({"type": "image", "data": base64.b64encode(p.read_bytes()).decode(), "mime_type": "image/png",
                        "resolution": "medium"})
    rec: dict[str, Any] = {"trial": trial, "thinking": thinking, "model": ARGON, "images": [str(p.relative_to(REPO)) for p in pick],
                           "input_text": text, "system": REFLECT_SYSTEM}
    t0 = time.monotonic()
    try:
        ix, lat, retr = _create(client, ARGON, content, REFLECT_SYSTEM, {"thinking_level": thinking, "max_output_tokens": 32768,
                                                 "thinking_summaries": "auto"}, timeout_s, retries=1)
    except Exception as e:  # noqa: BLE001  fall back to medium once
        rec["error_first"] = f"{type(e).__name__}: {e}"[:300]
        rec["thinking"] = "medium"
        ix, lat, retr = _create(client, ARGON, content, REFLECT_SYSTEM,
                                {"thinking_level": "medium", "max_output_tokens": 32768}, 600.0, retries=1)
    out = (output_text(ix) or "").strip()
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()][:6]
    u = usage_dict(ix.usage)
    rec.update(raw=out, text="\n".join(lines), usage=u, usd=cost_usd(ARGON, u), latency_s=round(time.monotonic() - t0, 2),
               retries=retr, status=ix.status,
               thoughts=[x for s in (ix.steps or []) for x in step_summary(s).get("summary", [])])
    out_path.write_text(json.dumps(rec, indent=1, default=str))
    return rec


TRUTH_KEYS = re.compile(r"board_err|episode_geometry|inhand|in_hand_pose|physics_override|offset_mm|true hole|"
                        r"true_hole|start_file", re.I)
