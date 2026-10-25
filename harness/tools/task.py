"""Task-level tools for peg insertion (stage C, interface task/v1; skill playbook/skills/author-tools).

Each tool runs a whole closed-loop motion through the SAME session daemon and actor ops as env_move / env_gripper /
env_wait (twin/env.py), then returns structured feedback and hands control back to the brain, which decides the next
call (URAI: deciding between calls beats a pre-written program, 56% vs 24% [urai-2026]). The routines live in
twin/task_tools.py (numpy only, no simulator access).

Information honesty: the tools act on the BELIEVED geometry: env task() (nominal board, nominal grasp), plus any
`corrections` the brain supplies from its own perception (e.g. a camera estimate of the hole or the in-hand pose),
plus what contact teaches (kept per episode in runs/<brain>/<run_id>/task_tools/sessions/<session>.json). They never
read the start's offsets, the evaluator or MjData. "Exact information" exists only inside task_tool_selftest.

  task_state               belief, contact memory and estimated stage (no motion)
  task_approach            approach_hole + align: square the part to the hole axis, hover over the believed hole
  task_move_until_contact  guarded move; re-touch slowly, probe press: contact class, touch point, stiffness
  task_search              spiral / raster search on the surface under a light press until the tip drops in
  task_insert              composite: approach, touch, (re-aim, search), push to the seat with wiggle; strategies
  task_release             open, check, retreat
  task_tool_selftest       deterministic test runner for C.tool_success (practice starts only; exact + believed)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from harness.registry import REPO, tool
from harness.tools.env import _call, _name

INTERFACE = "task/v1"
_RUN = {"brain": {"type": "string", "default": "claude", "description": "claude (stand-in) or argon"},
        "run_id": {"type": "string", "default": "run-standin-1", "description": "runs/<brain>/<run_id>/"}}
_CORR = {"corrections": {
    "type": "object",
    "description": "OPTIONAL: your own estimates that correct the believed geometry for the rest of this episode "
                   "(kept until the next env_reset). hole_shift_mm: [x, y] or [x, y, z] mm, where you think the true "
                   "hole entry is in the believed hole frame. grip: {dx_mm (along the pads, hand x), dy_mm (across, "
                   "hand y), grasp_from_top_mm (nominal 9), pivot_deg (about hand y), rock_deg (about hand x)}. "
                   "Only a grip with grasp_from_top_mm narrows the tip-height band (~10 mm -> 1.5 mm). A grip "
                   "correction forgets the contact memory measured with the old grip (touch height, touch point, "
                   "engaged, grip check). Omit to act on the nominal belief.",
    "properties": {"hole_shift_mm": {"type": "array", "items": {"type": "number"}},
                   "grip": {"type": "object", "properties": {
                       "dx_mm": {"type": "number"}, "dy_mm": {"type": "number"},
                       "grasp_from_top_mm": {"type": "number"}, "pivot_deg": {"type": "number"},
                       "rock_deg": {"type": "number"}}}}}}
_SES = {"session": {"type": "string", "description": "session id from env_open (an episode must be running)"},
        **_RUN, **_CORR}
FEEDBACK = (
    "Returns (task/v1 feedback): ok, code (OK | NO_CONTACT | UNEXPECTED_CONTACT | NOT_FOUND | JAMMED | SHALLOW | "
    "NO_SURFACE_REF | NOT_HOLDING | FORCE_LIMIT | GRIP_SLIP | EPISODE_OVER | ENV_ERROR | BAD_ARGS), summary (text with numbers), "
    "tried (phases, commands, why each stopped), contact {state none|surface|in_hole|jam, force_N = |F - F_ref| with "
    "F_ref the free-space reading, force_hole_N, torque_Nm, touch_point_hole_mm = where the part touches from the "
    "wrench}, depth {below_touch_mm = TCP travel past the first surface touch (the tip depth if the grip did not slip), "
    "below_believed_surface_mm, seat_depth_mm}, stage_est (ESTIMATED from observations: lifted | "
    "above_target_believed | contact | engaged | seated_est | released; the evaluator's rubric comes only from "
    "env_finish), tip_hole_mm (believed tip in the hole frame: x, y lateral, z > 0 below the surface), commands, "
    "commands_since_drop (the env stalls an episode after 40 commands or 60 sim s with no progress), sim_s, memory. "
    "Units: mm, N, deg, m/s. Hole frame: origin at the believed hole centre on the surface, +z into the hole.")


LOGGED_OPS = {"move": "env_move", "gripper": "env_gripper", "wait": "env_wait", "observe": "env_observe",
              "status": "env_status"}   # the read-only `task` op has no env_* tool and is not logged


def _registry_caller(default: str = "task-tools") -> str:
    """The caller of the registry call that is running this tool (harness.registry.call's `caller`)."""
    import inspect
    f = inspect.currentframe()
    while f is not None:
        if f.f_code.co_name == "call" and f.f_globals.get("__name__") == "harness.registry":
            return str(f.f_locals.get("caller", default))
        f = f.f_back
    return default


class SocketAPI:
    """The actor ops over the session daemon's socket (the same door as env_move / env_gripper / env_wait).

    Every env command a task tool issues is appended to the call log (runs/calls/<date>.jsonl) as its own line, in
    exactly the shape harness.registry.call writes for env_move / env_gripper / env_wait / env_observe / env_status
    (same caller, session, brain and run_id), plus `parent` = the task tool call it belongs to. So the replay recorder
    (twin/practice_view.py episode_log) and integrity_audit see the full command sequence; the task tool's own line
    follows when it returns."""

    def __init__(self, brain: str, run_id: str, session: str, parent_tool: str = "task", caller: str | None = None):
        self.brain, self.run_id, self.session = brain, run_id, session
        self.caller = caller or _registry_caller()
        self.parent = dict(tool=parent_tool, call_id=f"{parent_tool}-{time.time_ns()}", caller=self.caller)
        self.n_logged = 0

    def _c(self, req: dict) -> dict:
        from harness import registry
        req = {k: v for k, v in req.items() if v is not None}
        t0 = time.time()
        r = _call(self.brain, self.run_id, self.session, req)
        name = LOGGED_OPS.get(req.get("op"))
        if name:
            args = {"session": self.session, **{k: v for k, v in req.items() if k != "op"}, "brain": self.brain,
                    "run_id": self.run_id}
            ok = "error" not in r
            line = {"t": t0, "dt": round(time.time() - t0, 3), "caller": self.caller, "tool": name, "args": args,
                    "ok": ok, "result": r if ok else {"error": r["error"]}, "parent": self.parent}
            registry.CALL_LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(registry.CALL_LOG_DIR / f"{time.strftime('%Y%m%d')}.jsonl", "a") as fh:
                fh.write(json.dumps(line, default=str) + "\n")
            self.n_logged += 1
        if "error" in r:
            raise RuntimeError(r["error"])
        return r

    def task(self):
        return self._c({"op": "task"})

    def status(self):
        return self._c({"op": "status"})

    def observe(self):
        return self._c({"op": "observe"})

    def move(self, pos, quat=None, speed=None, force_max=None, stop_force=None, timeout=None):
        return self._c({"op": "move", "pos": pos, "quat": quat, "speed": speed, "force_max": force_max,
                        "stop_force": stop_force, "timeout": timeout})

    def gripper(self, action, width=None):
        return self._c({"op": "gripper", "action": action, "width": width})

    def wait(self, seconds):
        return self._c({"op": "wait", "seconds": seconds})


# ------------------------------------------------------------------------------------------------ memory and caps
def _mem_file(brain, run_id, session) -> Path:
    return REPO / "runs" / _name(brain, "brain") / _name(run_id, "run_id") / "task_tools" / "sessions" / \
        f"{_name(session, 'session')}.json"


def _episode_key(api) -> str:
    st = api.status()
    if st.get("episode_over"):
        return f"{st.get('start_id')}#{st.get('episodes')}#over"
    return f"{st.get('start_id')}#{st.get('episodes')}"


def force_cap(part: str, run_dir: Path | None = None) -> float:
    """Real-envelope force ceiling for a part family: the 99th percentile of real successful insertion peaks
    (twin/envelope.py, train only), never above the env cap (15 N)."""
    try:
        from twin.envelope import family_of
        f = (run_dir or REPO / "runs" / "claude" / "run-standin-1") / "envelope" / "envelope.json"
        env = json.loads(f.read_text())
        return float(min(15.0, env["envelope"][family_of(part)]["features"]["peak_force_N"]["hi"]))
    except Exception:  # noqa: BLE001
        return 12.0


def _open(session, brain, run_id, corrections, tool_name="task"):
    """(TaskTools, save) for the running episode: belief from task() + corrections, memory from disk."""
    from twin import task_tools as TT
    api = SocketAPI(brain, run_id, session, parent_tool=tool_name)
    key = _episode_key(api)
    f = _mem_file(brain, run_id, session)
    state = json.loads(f.read_text()) if f.exists() else {}
    if state.get("episode") != key.replace("#over", ""):
        state = dict(episode=key.replace("#over", ""), memory={}, corrections={})
    if corrections:
        bad = set(corrections) - set(TT.CORRECTION_KEYS)
        if bad:
            raise ValueError(f"unknown correction keys {sorted(bad)}; known: {sorted(TT.CORRECTION_KEYS)}")
        state["corrections"] = {**state.get("corrections", {}), **corrections}
        state["memory"].pop("q_hand", None)       # the part axis changed: re-align at the next approach
        if corrections.get("grip"):               # the tip moved relative to the TCP: TCP-frame contact memory is stale
            TT.forget_grip_dependent(state["memory"])
    task = api.task()
    if "hole" not in task:
        raise RuntimeError("no episode is running in this session: env_reset to a start first")
    belief = TT.believed(task, state.get("corrections") or None)
    tt = TT.TaskTools(api, belief, state["memory"])

    def save(out: dict) -> dict:
        f.parent.mkdir(parents=True, exist_ok=True)
        state["memory"] = tt.mem
        state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write_text(json.dumps(state, default=float))
        out["memory_file"] = str(f.relative_to(REPO))
        out["call_log"] = dict(call_id=api.parent["call_id"], env_calls_logged=api.n_logged)
        return out
    return tt, save, task


def _run(session, brain, run_id, corrections, fn):
    try:
        tt, save, task = _open(session, brain, run_id, corrections, getattr(fn, "tool", "task"))
    except Exception as ex:  # noqa: BLE001
        return dict(interface=INTERFACE, ok=False, code="ENV_ERROR", summary=f"{type(ex).__name__}: {ex}")
    try:
        out = fn(tt, task)
    except RuntimeError as ex:      # the env refused a command
        tt.err = str(ex)
        out = tt.report(getattr(fn, "tool", "task"), False, "ENV_ERROR", f"the env refused a command: {ex}")
    return save(out)


# ------------------------------------------------------------------------------------------------ tools
@tool("task_state",
      "Task tools (task/v1): what the tools believe and have learned this episode, without moving. Belief: believed "
      "hole entry and axis (env task(): the nominal board, off the true one by the calibration error, median 1.3 mm), "
      "believed tip in the TCP frame (nominal grasp; the true in-hand pose is unknown: along-pad offset up to 6.5 mm, "
      "tilt up to ~15 deg in 90% of grasps), your corrections, and the height uncertainty. Memory: the surface touch "
      "height, the touch point, whether the tip dropped in. " + FEEDBACK,
      {"type": "object", "required": ["session"], "properties": _SES})
def task_state(session: str, brain: str = "claude", run_id: str = "run-standin-1",
               corrections: dict | None = None) -> dict:
    def fn(tt, task):
        return tt.state()
    fn.tool = "task_state"
    return _run(session, brain, run_id, corrections, fn)


@tool("task_approach",
      "Task tools (task/v1), approach_hole + align. Squares the hand so the believed part axis is on the hole axis "
      "(round: keeps the current yaw; square: the allowed yaw nearest the current one, joint 7 has +-3.05 rad), "
      "rises first if the tip is low, crosses over and hovers the believed tip standoff_mm above the believed hole "
      "entry (plus offset_mm in the hole frame), then waits until the hand converged (lateral <= 0.3 mm, rotation "
      "<= 0.7 deg; large re-orientations creep at ~2 deg/s under the env's torque ceiling, up to 12 s), then trims the "
      "steady lateral error. Legs are unguarded free-space moves (the env slows every guarded move to 1.5-3 mm/s) "
      "with a 15 N spring ceiling, checked after each leg: |F - F_ref| >= 3 N is an unexpected contact. ok: hovering "
      "with no contact. Failure: UNEXPECTED_CONTACT (board higher or part longer than "
      "believed), NOT_HOLDING. " + FEEDBACK,
      {"type": "object", "required": ["session"], "properties": {
          **_SES, "standoff_mm": {"type": "number", "default": 12.0, "description": "believed tip height above the "
                                  "believed surface; keep it above the height uncertainty (~10 mm uncorrected)"},
          "offset_mm": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                        "description": "hole-frame [x, y] aim offset, mm (default [0, 0])"},
          "speed": {"type": "number", "default": 0.03, "description": "m/s, free-space legs (descent leg <= 0.015)"}}})
def task_approach(session: str, standoff_mm: float = 12.0, offset_mm: list | None = None, speed: float = 0.03,
                  brain: str = "claude", run_id: str = "run-standin-1", corrections: dict | None = None) -> dict:
    def fn(tt, task):
        return tt.approach(standoff_mm=float(standoff_mm), offset_mm=tuple(offset_mm or (0.0, 0.0)),
                           speed=min(float(speed), 0.1))
    fn.tool = "task_approach"
    return _run(session, brain, run_id, corrections, fn)


@tool("task_move_until_contact",
      "Task tools (task/v1), guarded move (move_until_contact). Moves from where the hand is along 'down' (into the "
      "hole), 'up', '+x', '-x', '+y', '-y' (hole frame) or a hole-frame vector, until |F - F_ref| >= stop_force. "
      "Downward: unguarded at `speed` to the top of the band where contact is possible (believed surface minus the "
      "height uncertainty and 2 mm: ~12 mm uncorrected, 3.5 mm corrected), converge sideways, then guarded through "
      "the band, which ends where an untouched tip must be in the hole (the env caps guarded moves at 1.5-3 mm/s by "
      "stop force: ~13 s for the uncorrected 22 mm band, ~5 s corrected). An early contact re-touches (lift 1.5 mm "
      "back onto the commanded line, converge, touch again). Then it probes: a press at ~half probe_force, then a "
      "steady press of probe_force for 0.4 s (a tip over the hole sinks in; the static wrench gives the touch point; "
      "the arm stiffness is measured). Grip check (probe.grip_check): between the two presses a part that turns in the "
      "fingers moves its contact >= 0.5 mm in the hand frame AND lets the hand sink >= 0.15 mm (rigid grips: < 0.15 / "
      "< 0.1 mm); then the summary says GRIP CHECK, memory.grip_pivot holds a rough tilt estimate and a suggested "
      "grip correction, and task_search / task_insert refuse with GRIP_SLIP until the grip is corrected. "
      "contact.state: surface (the first downward touch sets the depth reference), in_hole (sank, or deeper than an "
      "earlier touch by 2 mm, or past the believed surface by more than the height uncertainty), none. relief_mm "
      "backs off afterwards. ok: contact made. NO_CONTACT with contact.state in_hole means the tip went down past the "
      "believed surface unopposed (engaged). " + FEEDBACK,
      {"type": "object", "required": ["session"], "properties": {
          **_SES, "direction": {"description": "down | up | +x | -x | +y | -y, or a hole-frame [x, y, z]",
                                "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "number"}}]},
          "distance_mm": {"type": "number", "default": 40.0},
          "speed": {"type": "number", "default": 0.013, "description": "m/s (faster lags sideways)"},
          "stop_force": {"type": "number", "default": 3.0, "description": "N"},
          "max_force": {"type": "number", "default": 8.0, "description": "N, spring ceiling (capped by the real "
                                                                         "envelope of the part family)"},
          "relief_mm": {"type": "number", "default": 0.0},
          "probe_force": {"type": "number", "default": 3.0, "description": "N, 0 = no probe"},
          "retouch": {"type": "boolean", "default": True},
          "learn_tip": {"type": "boolean", "default": False,
                        "description": "correct where the tip is from the first touch point (off by up to the part "
                                       "radius for a flat face; worth it when the touch is further than that)"}}})
def task_move_until_contact(session: str, direction="down", distance_mm: float = 40.0, speed: float = 0.013,
                            stop_force: float = 3.0, max_force: float = 8.0, relief_mm: float = 0.0,
                            probe_force: float = 3.0, retouch: bool = True, learn_tip: bool = False,
                            brain: str = "claude",
                            run_id: str = "run-standin-1", corrections: dict | None = None) -> dict:
    def fn(tt, task):
        cap = force_cap(task["part"])
        return tt.move_until_contact(direction=direction, distance_mm=float(distance_mm), speed=float(speed),
                                     stop_force=min(float(stop_force), cap), max_force=min(float(max_force), cap),
                                     relief_mm=float(relief_mm), probe_force=min(float(probe_force), cap),
                                     retouch=bool(retouch), learn_tip=bool(learn_tip))
    fn.tool = "task_move_until_contact"
    return _run(session, brain, run_id, corrections, fn)


@tool("task_search",
      "Task tools (task/v1), search for the hole on the surface. Needs a surface touch first (depth reference). "
      "Slides the tip from where it is along an Archimedean spiral (turns pitch_mm apart, chords 2.5 pitches) or a "
      "raster (rows pitch_mm apart) out to radius_mm (defaults from the part: pitch 1.2 x (chamfer + clearance), "
      "radius what the moves left cover: ~3 mm for the 4 mm pegs, 5-8 mm for the larger ones), pressing press_force N (a light press: the tip lags the target by about "
      "friction x press depth), one env move per chord, at most max_moves and never past 36 commands without going "
      "deeper (the env stalls an episode after 40), so control comes back with commands to spare. A sink of >= 0.5 mm "
      "(a chamfer caught on the edge) is checked with a confirming press, at most twice. Stops when the tip drops below the touch height. Keep pitch_mm under twice "
      "(chamfer + clearance): 0.9 mm for the 4 mm pegs, 1.7 mm for 8 mm. ok: dropped in (engaged). NOT_FOUND: the hole "
      "is further than radius_mm (board error p90 4 mm) or the part is tilted in the hand. GRIP_SLIP: the grip check "
      "found the part turning in the fingers (nothing moved), or the fingers closed > 1 mm further during the search. "
      + FEEDBACK,
      {"type": "object", "required": ["session"], "properties": {
          **_SES, "pattern": {"type": "string", "enum": ["spiral", "raster"], "default": "spiral"},
          "radius_mm": {"type": "number", "description": "mm; default: what max_moves chords cover, <= 8"},
          "pitch_mm": {"type": "number", "description": "mm; default 1.2 x (chamfer + clearance), 0.3-2"},
          "press_force": {"type": "number", "default": 0.8, "description": "N"},
          "speed": {"type": "number", "default": 0.004, "description": "m/s along the path"},
          "max_moves": {"type": "integer", "default": 32},
          "center": {"type": "string", "enum": ["here", "hole"], "default": "here",
                     "description": "here: around where the tip is; hole: lift, cross to the believed axis with the "
                                    "touch-corrected tip (needs a touch with learn_tip), touch down, then search"}}})
def task_search(session: str, pattern: str = "spiral", radius_mm: float | None = None, pitch_mm: float | None = None,
                press_force: float = 0.8, speed: float = 0.004, max_moves: int = 32, center: str = "here",
                brain: str = "claude",
                run_id: str = "run-standin-1", corrections: dict | None = None) -> dict:
    def fn(tt, task):
        cap = force_cap(task["part"])
        return tt.search(pattern=pattern, radius_mm=None if radius_mm is None else float(radius_mm),
                         pitch_mm=None if pitch_mm is None else max(0.2, float(pitch_mm)),
                         press_force=min(float(press_force), cap), speed=min(float(speed), 0.02),
                         max_force=cap, max_moves=int(max(1, min(int(max_moves), 60))),
                         center="hole" if center == "hole" else "here")
    fn.tool = "task_search"
    return _run(session, brain, run_id, corrections, fn)


@tool("task_insert",
      "Task tools (task/v1), insert(part, hole). Composite: approach (unless already hovering), guarded descent with "
      "re-touch and probe, then by strategy: auto (with corrected grip: a small spiral if the probe did not sink; "
      "otherwise as spiral), press (push straight in from the first contact), spiral (move the touch point onto the "
      "believed axis, re-touch, spiral search, push), wiggle (spiral with more and larger wiggles), tilt (search with "
      "the part tilted so its rim leads). The push drives the tip to the seat at <= max_force (capped by the real "
      "envelope: 13.8 N round, 12.6 N square) and wiggles +-wiggle_deg about both hole axes on a stall. params "
      "overrides: standoff_mm, descent_speed, stop_force, probe_force, reaim, radius_mm, pitch_mm, chord_mm, "
      "press_force, search_speed, max_moves, wiggle_deg, max_wiggles, push_speed, tilt_deg, retreat_mm, center "
      "(here | hole: centre the search on the believed axis with the touch-corrected tip), learn_tip (correct the tip "
      "from the first touch point: useful when the touch point is further from the believed tip than the part radius). "
      "release: "
      "open and retreat after an estimated seat. ok: estimated seated (and released). Failure: JAMMED, SHALLOW, "
      "GRIP_SLIP (the part turns in the fingers: the grip check after the first touch, the fingers closing > 1 mm, or "
      "the hand going past the seat; correct the grip or regrasp), NOT_FOUND, NO_CONTACT, UNEXPECTED_CONTACT. The "
      "episode's real outcome comes from env_finish. " + FEEDBACK,
      {"type": "object", "required": ["session"], "properties": {
          **_SES, "strategy": {"type": "string", "enum": ["auto", "press", "spiral", "wiggle", "tilt"],
                               "default": "auto"},
          "max_force": {"type": "number", "default": 10.0, "description": "N"},
          "params": {"type": "object", "description": "strategy parameter overrides (names above)"},
          "release": {"type": "boolean", "default": False}}})
def task_insert(session: str, strategy: str = "auto", max_force: float = 10.0, params: dict | None = None,
                release: bool = False, brain: str = "claude", run_id: str = "run-standin-1",
                corrections: dict | None = None) -> dict:
    def fn(tt, task):
        return tt.insert(strategy=strategy, max_force=min(float(max_force), force_cap(task["part"])),
                         params=params, release=bool(release))
    fn.tool = "task_insert"
    return _run(session, brain, run_id, corrections, fn)


@tool("task_release",
      "Task tools (task/v1), release + retreat. Opens the gripper, then retreats retreat_mm along the hole axis, "
      "guarded at 3 N. ok: the fingers opened past the part and nothing pulled during the retreat. The part's pose "
      "is not observable: call env_finish for the evaluator's verdict (practice starts). " + FEEDBACK,
      {"type": "object", "required": ["session"], "properties": {
          **_SES, "retreat_mm": {"type": "number", "default": 30.0},
          "speed": {"type": "number", "default": 0.03, "description": "m/s"}}})
def task_release(session: str, retreat_mm: float = 30.0, speed: float = 0.03, brain: str = "claude",
                 run_id: str = "run-standin-1", corrections: dict | None = None) -> dict:
    def fn(tt, task):
        return tt.release(retreat_mm=float(retreat_mm), speed=min(float(speed), 0.1))
    fn.tool = "task_release"
    return _run(session, brain, run_id, corrections, fn)


@tool("task_tool_selftest",
      "Stage C self-test of the task tools (task/v1), deterministic: fixed PRACTICE starts (never held-out or "
      "red-team), the first per_part starts of each part, each run as separate episodes per information mode. "
      "exact = the tools get the true hole and in-hand pose as corrections (C.tool_success as defined); believed = "
      "the nominal belief only (what a brain has). Per episode it chains task_approach, task_move_until_contact, "
      "task_search (when needed), task_insert (continuation: push), task_release, env finish; a second episode runs "
      "task_insert alone (release on); a third, optional, runs a single raw-pose descent as the baseline. A watcher "
      "scores every tool from the evaluator's state between calls (the tools never see it). Writes "
      "runs/<brain>/<run_id>/task_tools/selftest/<stamp>/ (results.jsonl, summary.json). Returns per-tool success with "
      "Clopper-Pearson 95% intervals per mode. It records no checks (actor != watcher).",
      {"type": "object", "properties": {
          **_RUN, "parts": {"type": "array", "items": {"type": "string"}, "description": "default: all 8 pegs"},
          "per_part": {"type": "integer", "default": 6, "description": "first N practice starts per part (max 20)"},
          "info_modes": {"type": "array", "items": {"type": "string", "enum": ["exact", "believed"]}},
          "chains": {"type": "array", "items": {"type": "string", "enum": ["tools", "insert", "raw"]},
                     "description": "default tools + insert"},
          "workers": {"type": "integer", "default": 12},
          "cousin": {"type": "string", "default": "c000"},
          "dt_half": {"type": "boolean", "default": True, "description": "confirm successes at dt/2 (counted)"}}})
def task_tool_selftest(brain: str = "claude", run_id: str = "run-standin-1", parts: list | None = None,
                       per_part: int = 6, info_modes: list | None = None, chains: list | None = None,
                       workers: int = 12, cousin: str = "c000", dt_half: bool = True) -> dict:
    from twin import task_selftest
    return task_selftest.run(brain=brain, run_id=run_id, parts=parts, per_part=int(per_part),
                             info_modes=info_modes or ["exact", "believed"], chains=chains or ["tools", "insert"],
                             workers=int(workers), cousin=cousin, dt_half=bool(dt_half))
