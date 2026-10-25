"""Task-level tools for peg insertion (stage C, interface task/v1; skill playbook/skills/author-tools).

Every tool here is a closed-loop routine built ONLY from the actor ops of the practice env (twin/env.py): task,
observe, move (absolute TCP target with speed, force ceiling and guarded stop), gripper, wait. It holds no reference
to MjModel/MjData and imports nothing but numpy and math, so the same code runs behind a harness tool (session
daemon over a socket), in-process against `PracticeEnv.command(..., allowed=ACTOR_OPS)`, or, copied into a skill
workspace, against `envapi` in the sandbox.

Information honesty
-------------------
The tools act on a BELIEF, never on simulator truth:
  hole   the env's task(): the nominal (believed) board from board-from-contacts. The true board differs by the
         start's board_err (train leave-one-out across-pad errors: |e| median 1.28 mm, p90 4.0 mm; z sigma 1.6 mm),
         which the env never exposes.
  grip   the nominal grasp: part axis = hand z through the pad centre, top 9 mm above the pad centre. The true
         in-hand pose (along-pad dx p90 5.4 mm, pivot p90 14.5 deg, grasp height 2-19 mm) is never exposed.
A brain may CORRECT the belief with its own estimates (`corrections`: hole_shift_mm, grip dx/dy/height/pivot), e.g.
from a camera image; the tools then act on the corrected belief. Contact teaches the tools what perception could not:
the surface touch height (depth reference), the touch point from the wrench's line of action (where the tip really
is) and the drop into the hole. That knowledge is kept in a per-episode `memory` dict built from observations only.
"Exact information" (C.tool_success) is the same corrections filled with the truth by the stage C self-test from the
frozen practice start file. Brains never get it.

Feedback conventions (every tool returns the same shape)
--------------------------------------------------------
  tool, interface         name and "task/v1"
  ok, code                did the tool reach its own postcondition; OK or a failure code (CODES below)
  summary                 one or two sentences for the brain, with numbers
  tried                   the phases run, in order, each with its command count and why it stopped
  contact                 state (none | surface | in_hole | jam), force_N (|F - F_ref| now; F_ref = the free-space
                          reading, i.e. the part's weight), force_hole_N (that force in the hole frame), torque_Nm,
                          touch_point_hole_mm (where the part touches, hole frame x/y, from the wrench's line of action)
  depth                   below_touch_mm (TCP travel along the hole axis past the surface touch: the tip depth if the
                          grip did not slip), below_believed_surface_mm (believed tip vs believed surface; off by the
                          grasp-height and board-z uncertainty, about +-10 mm uncorrected), seat_depth_mm
  stage_est               no_approach | lifted | above_target_believed | contact | engaged | seated_est | released.
                          ESTIMATED from robot observations and the belief. The frozen evaluator's rubric (true hole)
                          is reported only by env_finish, on practice starts only.
  tip_hole_mm             believed tip in the believed hole frame (x, y lateral; z > 0 = below the surface)
  commands, sim_s         env commands used and sim time spent by this call;
  commands_since_drop     moves since the tip last went deeper (the env ends an episode as stalled after 40 commands
                          or 60 sim s with no rubric or depth progress)
  memory                  what contact taught so far (surface_s, touch point, engaged, F_ref, corrections ...)
Arguments: mm, N, degrees; speeds in m/s. Hole frame: origin at the believed hole centre on the surface, +z INTO the
hole, +y along the hole's long side.
"""

from __future__ import annotations

import json
import math

import numpy as np

INTERFACE = "task/v1"
PAD_TO_TCP = 0.0229            # m: TCP lies this far beyond the finger-pad centre along hand z (env task conventions)
NOMINAL_GRASP_FROM_TOP_MM = 9.0
CONTACT_N = 2.0                # N: |F - F_ref| above this = touching (twin/envelope.py CONTACT_N, the real onset rule)
ENGAGE_MM = 2.0                # mm below the touch height = engaged (twin/evaluator.py `engaged`: tip > 2 mm in)
DROP_MM = 1.0                  # mm: a search stops when the tip drops this far below the touch height
SEAT_TOL_MM = 1.5              # mm: estimated seated within this of the seat depth (evaluator tol 1 mm + slip)
MAX_FORCE_DEFAULT = 10.0       # N: below the real 99th-percentile insertion peak (13.8 round, 12.6 square)
HOVER_SPEED = 0.03             # m/s: free-space moves (real approach 12-122 mm/s, median 27-37; envelope.json)
FREE_FORCE_N = 8.0             # N: spring ceiling of unguarded free-space moves (bounds an unexpected collision)
FREE_CONTACT_N = 3.0           # N: |F - F_ref| at the end of a free move that counts as an unexpected contact
DESCENT_SPEED = 0.013          # m/s: guarded descent; faster lags sideways (2.3 mm at 25 mm/s). Real approach
                               # 1st pct 10.5-12.4 mm/s, median 27-37 (envelope.json)
PROBE_MIN_N = 1.0              # N: smallest steady press the touch point is computed from
SLOW_SPEED = 0.004             # m/s: speed through the band where contact is possible (lag ~0.35 mm)
CANDIDATE_DROP_MM = 0.5        # mm: a sink this deep during a search is checked with a confirming press
SLIP_MM = 1.5                  # mm past the seat depth: the hand cannot get there unless the part moves in the grip
FIRST_PUSH_N = 6.0             # N: the first push of a seat (real successful insert peaks: median 5.8-6.6 N)
NO_PROGRESS_CMDS = 40          # the env's progress.stuck limit (twin/monitors.py STUCK_CMDS): commands with no depth
NO_PROGRESS_MARGIN = 4         # or stage progress; tools stop this many commands short and hand back
SLOW_SPEED_KNOWN = 0.003       # m/s: the same when the height is known (corrected belief: a 3.5 mm band; lag ~0.25 mm)
K_DEFAULT = 500.0              # N/m: REASSEMBLE's translational stiffness (cousins 200-800); the probe measures it
TIP_SHIFT_MIN_MM = 0.5         # touch-point corrections smaller than this are ignored
TIP_SHIFT_MAX_MM = 12.0        # and larger ones clipped (in-hand offset p90 5.4 mm + tilt lever ~8 mm)
# Grip check (the probe presses twice, ~1.5 N then probe_force, standing still): a part that turns in the pads moves
# its contact point in the HAND frame and lets the hand sink. Measured 2026-10-25 (author-tools Lessons), 1.5 -> 3 N:
# rigid grips (round peg 2) moved the contact 0.03-0.14 mm and sank 0.05-0.09 mm; round peg 1 at 12-17 deg in-hand
# tilt moved it 1.70-1.74 mm and sank 0.39-0.40 mm. The thresholds sit between; both must be exceeded (a flat face's
# centre of pressure can wander without the hand sinking).
PIVOT_SHIFT_MM = 0.5           # contact moved at least this far in the hand frame between the two presses
PIVOT_SINK_MM = 0.15           # and the hand sank at least this much: the part is turning in the fingers
GRIP_CLOSE_MM = 1.0            # the fingers closed max(this, GRIP_CLOSE_FRAC x their width at the approach) further:
GRIP_CLOSE_FRAC = 0.2          # the part is slipping out (round peg 1 slipping out: 3.2 -> 1.0 mm; a 12 mm peg held
                               # 2 mm from its top settled 1.0 mm narrower and stayed upright: not a slip)
J7_LIMIT = 3.0508              # rad, FR3 joint 7 (menagerie fr3.xml; the env flags leaving it as invalidating)
J7_MARGIN = 0.25               # rad kept free when choosing a square part's yaw
STAGES_EST = ("no_approach", "lifted", "above_target_believed", "contact", "engaged", "seated_est", "released")
CODES = {
    "OK": "the tool reached its postcondition",
    "NO_CONTACT": "a guarded move ran its full distance without touching anything",
    "UNEXPECTED_CONTACT": "touched something during a free-space move (board higher or part longer than believed)",
    "NOT_FOUND": "a search finished its pattern without the tip dropping into a hole",
    "JAMMED": "the push stalled short of the seat at the force ceiling, wiggles included",
    "SHALLOW": "the push ended short of the seat without a jam signature",
    "NO_SURFACE_REF": "the tool needs a surface touch first (task_move_until_contact down)",
    "NOT_HOLDING": "the gripper is open or empty",
    "FORCE_LIMIT": "a force above the tool's ceiling, or a pull during a retreat",
    "GRIP_SLIP": "the part is turning or sliding in the fingers: a probe press moved its contact in the hand and let "
                 "the hand sink (grip check), the fingers closed > max(1 mm, 20% of their width) further than at the approach, or the hand "
                 "travelled past the seat; thin pegs do this under a few newtons",
    "EPISODE_OVER": "the env ended the episode (budget, stall or finish): reset to a start",
    "ENV_ERROR": "the env refused a command; the message says why",
    "BAD_ARGS": "an argument is out of range or of the wrong kind",
}


# ============================================================================================== math (numpy only)
def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R) -> np.ndarray:
    R = np.asarray(R, float)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 2 * math.sqrt(t + 1)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2 * math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2 * math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def rot_axis(axis, ang) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def rot_angle_deg(Ra, Rb) -> float:
    c = (float(np.trace(np.asarray(Ra).T @ np.asarray(Rb))) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def json_dumps(x) -> str:
    return json.dumps(x)


def _r(v, k=3):
    return [round(float(x), k) for x in np.asarray(v, float).ravel()]


# ============================================================================================== belief
CORRECTION_KEYS = {"hole_shift_mm": "hole-frame [x, y] or [x, y, z] mm: where the true hole entry is relative to the "
                                    "believed one",
                   "grip": "in-hand pose: {dx_mm (along the pads, hand x), dy_mm (across, hand y), grasp_from_top_mm "
                           "(pad centre below the part's top end; nominal 9), pivot_deg (about hand y), rock_deg (about "
                           "hand x)}"}


def believed(task: dict, corrections: dict | None = None) -> dict:
    """Belief from the env's task() (what any actor may know), optionally corrected by the brain's own estimates."""
    h = task["hole"]
    c = corrections or {}
    pg = task.get("part_geometry", {})
    L = float(pg.get("length_mm", 50.0)) / 1000.0
    g = dict(dx_mm=0.0, dy_mm=0.0, grasp_from_top_mm=NOMINAL_GRASP_FROM_TOP_MM, pivot_deg=0.0, rock_deg=0.0)
    # the tip's height is known only if the grasp height is: a pivot-only correction leaves the height uncertainty
    grip_given = "grasp_from_top_mm" in (c.get("grip") or {})
    g.update({k: float(v) for k, v in (c.get("grip") or {}).items() if k in g})
    Rin = rx(math.radians(g["rock_deg"])) @ ry(math.radians(g["pivot_deg"]))
    tip = (np.array([g["dx_mm"] / 1000, g["dy_mm"] / 1000, -PAD_TO_TCP])
           + Rin @ np.array([0.0, 0.0, L - g["grasp_from_top_mm"] / 1000]))
    sh = list(map(float, c.get("hole_shift_mm") or [0.0, 0.0])) + [0.0]
    Re = quat_to_mat(h["entry_quat"])
    H = np.asarray(h["entry_pos"], float) + Re @ (np.array(sh[:3]) / 1000.0)
    return dict(
        source="corrected" if c else "believed", part=task["part"], family=task["family"],
        start_id=task.get("start_id"),
        hole=dict(pos=H.tolist(), R=Re.tolist(), yaw_symmetry_deg=h["yaw_symmetry_deg"],
                  clearance_mm=h["clearance_per_side_mm"], seat_depth_mm=h["seat_depth_mm"]),
        grip=dict(tip_tcp=tip.tolist(), R_in=Rin.tolist(), **g),
        part_geometry=pg, corrections=c,
        # how far off the tip's height can be: the grasp height (9 +- 5 mm prior, 2-19 mm) and board z (1.6 mm);
        # small only when the correction gives the grasp height
        axial_sigma_mm=1.5 if grip_given else 10.0,
        note="believed = task() nominal board + nominal grasp" + ("; corrected by the caller" if c else ""))


# Memory that is measured in TCP coordinates or about the old grip: stale once the brain corrects the grip (the tip
# then sits elsewhere relative to the TCP). The harness drops these when a `grip` correction arrives.
# deepest_s and cmds_since_drop stay: the env's no-progress count runs on regardless, and forgetting the deepest TCP
# depth would let a re-approach count as progress (a recovery run on round peg 1-001 stalled that way).
GRIP_DEPENDENT_MEMORY = ("q_hand", "surface_s", "surface_s_est", "touch_tcp", "touch_point_hole_mm", "tip_shift_mm",
                         "engaged", "found_at_hole_mm", "grip_pivot")


def forget_grip_dependent(memory: dict) -> list:
    """Drop the memory entries a grip correction makes stale; returns their names."""
    gone = [k for k in GRIP_DEPENDENT_MEMORY if k in memory]
    for k in gone:
        memory.pop(k, None)
    return gone


def _B(belief: dict):
    h, g = belief["hole"], belief["grip"]
    return (np.asarray(h["pos"], float), np.asarray(h["R"], float), np.asarray(g["tip_tcp"], float),
            np.asarray(g["R_in"], float))


def hand_rotation(belief: dict, q_now, q7=None) -> np.ndarray:
    """Hand rotation that puts the believed part axis along the hole axis, with the hole yaw allowed by the part's
    symmetry nearest the current hand yaw. Joint 7 turns the hand about its own z, so a yaw change of d rad moves it by
    d: with q7 (the observed joint 7 angle) the yaw is the nearest one that keeps joint 7 J7_MARGIN inside its +-3.05
    rad (two square-peg approaches drove it to 3.052 rad, an invalidating flag)."""
    _, Re, _, Rin = _B(belief)
    zin = Re[:, 2]
    Rnow = quat_to_mat(q_now) @ Rin                  # current part frame in the world
    xc = Rnow[:, 0] - zin * float(Rnow[:, 0] @ zin)
    if np.linalg.norm(xc) < 1e-6:
        xc = Re[:, 0]
    xc = xc / np.linalg.norm(xc)
    sym = belief["hole"]["yaw_symmetry_deg"]
    if sym == 0:
        cands = [xc]
    else:
        cands = [Re[:, 0] * math.cos(math.radians(a)) + Re[:, 1] * math.sin(math.radians(a))
                 for a in np.arange(0, 360, sym)]
    Rh_now = quat_to_mat(q_now)

    def hand_for(x):
        y = np.cross(zin, x)
        return np.column_stack([x, y, zin]) @ Rin.T   # desired part frame in the world; R_hand @ R_in = R_part

    def yaw_change(Rh):                               # signed turn about the hand's own z = joint 7's change
        Rr = Rh_now.T @ Rh
        return math.atan2(Rr[1, 0] - Rr[0, 1], Rr[0, 0] + Rr[1, 1])
    opts = [(abs(yaw_change(hand_for(x))), hand_for(x), yaw_change(hand_for(x))) for x in cands]
    if q7 is not None and len(opts) > 1:
        ok = [o for o in opts if abs(float(q7) + o[2]) <= J7_LIMIT - J7_MARGIN]
        if ok:
            return min(ok, key=lambda o: o[0])[1]
        return min(opts, key=lambda o: abs(float(q7) + o[2]))[1]
    return min(opts, key=lambda o: o[0])[1]


class _EpisodeOver(Exception):
    pass


# ============================================================================================== the tools
class TaskTools:
    """Task-level tools over an actor API object (task/observe/move/gripper/wait), a belief and a memory dict.
    One instance per tool call is fine: everything that must persist lives in `memory` (JSON-able)."""

    def __init__(self, api, belief: dict, memory: dict | None = None, perceive=None):
        self.api = api
        self.mem = memory if memory is not None else {}
        self.perceive = perceive        # optional: () -> belief, a fresh perception (camera, or the self-test oracle)
        self.set_belief(belief)
        self.n_cmd = 0
        self.t0 = None
        self.tried: list[dict] = []
        self.last = None
        self.over = False
        self.err = None

    def set_belief(self, belief: dict):
        self.b = belief
        self.H, self.Re, self.tip_tcp, self.Rin = _B(belief)
        self.zin = self.Re[:, 2]

    # ------------------------------------------------------------------------------------------ plumbing
    def _do(self, op, **kw) -> dict:
        if self.over:
            raise _EpisodeOver()
        r = getattr(self.api, op)(**kw)
        if op in ("move", "gripper", "wait"):
            self.n_cmd += 1
            self.mem["cmds_since_drop"] = int(self.mem.get("cmds_since_drop", 0)) + 1
        if "obs" in r:
            o = r["obs"]
            if self.t0 is None:
                self.t0 = float(o["t"]) - float(r.get("result", {}).get("duration_s", 0.0))
            self._note_depth(o)
            self.last = o
        if r.get("episode_over"):
            self.over = True
            self.mem["episode_over"] = dict(ended_by=r.get("ended_by"), report=r.get("report"))
            raise _EpisodeOver()
        return r

    def _note_depth(self, o):
        """Track progress as the env's stuck monitor would see it, conservatively: the TCP going 0.5 mm deeper than
        ever before; once the surface has been touched, only deeper than 2 mm below the touch counts (pressing can
        tilt a part in the pads and lower the TCP while the tip stays on the surface)."""
        s = float(np.asarray(o["tcp"]["pos"], float) @ self.zin)
        best = self.mem.get("deepest_s")
        touch = self.mem.get("surface_s")
        if best is None or s > float(best) + 0.0005:
            self.mem["deepest_s"] = round(s, 6)
            if touch is None or s > float(touch) + ENGAGE_MM / 1000.0:
                self.mem["cmds_since_drop"] = 0

    def observe(self) -> dict:
        o = self.api.observe()
        if self.t0 is None:
            self.t0 = float(o["t"])
        self.last = o
        if o.get("episode_over"):
            self.over = True
        return o

    def F(self, o=None) -> np.ndarray:
        o = o or self.last
        return np.asarray(o["wrench"]["force_N"], float) - np.asarray(self.mem.get("F_ref", [0, 0, 0]), float)

    def tau(self, o=None) -> np.ndarray:
        o = o or self.last
        return np.asarray(o["wrench"]["torque_Nm"], float) - np.asarray(self.mem.get("tau_ref", [0, 0, 0]), float)

    def fmag(self, o=None) -> float:
        return float(np.linalg.norm(self.F(o)))

    def p(self, o=None) -> np.ndarray:
        return np.asarray((o or self.last)["tcp"]["pos"], float)

    def q(self, o=None) -> np.ndarray:
        return np.asarray((o or self.last)["tcp"]["quat"], float)

    def q7(self, o=None):
        """Observed joint 7 angle (rad), or None if the observation has no joint angles."""
        qs = (o or self.last).get("q")
        return float(qs[6]) if qs is not None and len(qs) >= 7 else None

    def s(self, o=None) -> float:
        """TCP coordinate along the hole axis (m; grows going into the hole)."""
        return float(self.p(o) @ self.zin)

    def tip(self, o=None) -> np.ndarray:
        """Believed tip position (world) from the current TCP pose and the believed grip, shifted sideways by what
        the first surface touch taught (memory tip_shift_mm: where the part really touched minus where the belief
        put the tip, hole frame)."""
        sh = np.asarray(self.mem.get("tip_shift_mm") or [0.0, 0.0], float) / 1000.0
        return self.p(o) + quat_to_mat(self.q(o)) @ self.tip_tcp + self.Re[:, :2] @ sh

    def hole_point(self, xy_mm=(0.0, 0.0), z_mm=0.0) -> np.ndarray:
        x, y = np.asarray(xy_mm, float) / 1000.0
        return self.H + self.Re @ np.array([x, y, z_mm / 1000.0])

    def to_hole(self, x) -> np.ndarray:
        return self.Re.T @ (np.asarray(x, float) - self.H)

    def Rh(self) -> np.ndarray:
        if self.mem.get("q_hand") is not None:
            return quat_to_mat(self.mem["q_hand"])
        return hand_rotation(self.b, self.q())

    def tcp_for(self, tip_world, Rh=None) -> np.ndarray:
        Rh = self.Rh() if Rh is None else Rh
        return np.asarray(tip_world, float) - Rh @ self.tip_tcp

    def move(self, pos, quat=None, speed=0.02, force_max=MAX_FORCE_DEFAULT, stop_force=None, timeout=None) -> dict:
        q = mat_to_quat(self.Rh()) if quat is None else quat
        kw = dict(pos=list(map(float, pos)), quat=list(map(float, q)), speed=float(speed), force_max=float(force_max))
        if stop_force is not None:
            kw["stop_force"] = float(stop_force)
        if timeout is not None:
            kw["timeout"] = float(min(20.0, max(0.05, timeout)))
        return self._do("move", **kw)

    def free(self, pos, quat=None, speed=HOVER_SPEED, force_max=FREE_FORCE_N, timeout=None) -> dict:
        """A move the belief says is in free space: unguarded (the env slows every guarded move to 1.5-3 mm/s so its
        stop can be honoured), spring capped at force_max, and checked afterwards: |F - F_ref| >= 3 N at the end is
        reported as stopped = 'stop_force' (unexpected contact)."""
        r = self.move(pos, quat, speed=speed, force_max=force_max, timeout=timeout)
        if self.fmag() >= FREE_CONTACT_N:
            r["result"]["stopped"] = "stop_force"
        return r

    def wait(self, seconds) -> dict:
        return self._do("wait", seconds=float(seconds))

    def moves_left(self) -> int:
        """Commands left before the env's no-progress limit, estimated from the TCP's own depth (the env counts the
        true tip depth and the rubric stage, which the tools cannot see)."""
        return NO_PROGRESS_CMDS - NO_PROGRESS_MARGIN - int(self.mem.get("cmds_since_drop", 0))

    def _phase(self, name, n0, why, **extra):
        self.tried.append(dict(phase=name, commands=self.n_cmd - n0, stopped=why, **extra))

    def holding(self, o=None) -> bool:
        return float((o or self.last)["gripper_width_m"]) < 0.075

    def grip_closed_mm(self, o=None) -> float | None:
        """How much further the fingers have closed since the approach (mm; None before an approach). The fingers
        hold the part at a fixed width; closing further means the part is squeezing out or turning between them."""
        w0 = self.mem.get("grip_width_mm")
        if w0 is None:
            return None
        return round(float(w0) - 1000 * float((o or self.last)["gripper_width_m"]), 3)

    def grip_lost(self, o=None) -> bool:
        c = self.grip_closed_mm(o)
        return c is not None and c > max(GRIP_CLOSE_MM, GRIP_CLOSE_FRAC * float(self.mem["grip_width_mm"]))

    def contact_hand(self, o=None):
        """Where the wrench's line of action meets the believed-tip plane, relative to the TCP in the HAND frame
        (x along the pads, y across; mm). None when |F - F_ref| < PROBE_MIN_N or the force is too horizontal."""
        o = o or self.last
        F, T = self.F(o), self.tau(o)
        f2 = float(F @ F)
        if f2 < PROBE_MIN_N ** 2:
            return None
        n = F / math.sqrt(f2)
        if abs(float(n @ self.zin)) < 0.3:
            return None
        base = self.p(o) + np.cross(F, T) / f2
        lam = (float(self.tip(o) @ self.zin) - float(base @ self.zin)) / float(n @ self.zin)
        Rh = quat_to_mat(self.q(o))
        return (Rh.T @ (base + lam * n - self.p(o)))[:2] * 1000.0

    def pivot_block(self, tool):
        """A report refusing to press on, if a grip check found the part turning in the fingers and the brain has not
        corrected the grip since (a corrected grip is the brain's own estimate: the tools then trust it)."""
        gp = self.mem.get("grip_pivot")
        if not gp or self.b.get("corrections", {}).get("grip"):
            return None
        return self.report(tool, False, "GRIP_SLIP",
                           f"the part turns in the fingers: under the probe press its contact moved "
                           f"{gp['contact_shift_mm']:.2f} mm in the hand while the hand sank {gp['sink_mm']:.2f} mm "
                           f"(a rigid grip: < {PIVOT_SHIFT_MM} mm / < {PIVOT_SINK_MM} mm). Searching or pushing would "
                           "turn it further (round peg 1: 15 -> 64 deg) and fake a drop. Rough in-hand tilt from the "
                           f"sink/shift ratio: {gp.get('tilt_est_deg')} deg (it folds in the along-pad offset). Correct "
                           f"the grip (e.g. corrections.grip = {json_dumps(gp.get('suggested_grip'))}, or better an "
                           "estimate from the wrist view) and re-approach, or regrasp", contact_state="surface",
                           grip_pivot=gp)

    def settle(self, goal_pos, R_goal, pos_tol_mm=0.3, rot_tol_deg=0.7, max_s=12.0, step_s=2.0) -> dict:
        """Wait until the hand has converged on its target. The env's rotational spring is capped at
        force_max x 0.05 m, so large re-orientations creep at about 2 deg/s (Lessons)."""
        t = 0.0
        while True:
            e = self.p() - np.asarray(goal_pos)
            ax = float(e @ self.zin)
            pe = 1000 * float(np.linalg.norm(e - self.zin * ax))      # lateral error (the arm sags ~0.4 mm axially)
            re = rot_angle_deg(quat_to_mat(self.q()), R_goal)
            if (pe <= pos_tol_mm and abs(ax) <= 0.001 and re <= rot_tol_deg) or t >= max_s - 1e-9:
                return dict(lateral_err_mm=round(pe, 3), axial_err_mm=round(1000 * ax, 3), rot_err_deg=round(re, 3),
                            waited_s=round(t, 2))
            dt = min(step_s, max_s - t)
            self.wait(dt)
            t += dt

    def trim(self, goal, R_goal, qh, tol_mm=0.08, rounds=2) -> dict | None:
        """The impedance controller keeps a steady sideways error of ~0.25 mm here (Lessons): command the goal shifted
        by the remaining lateral error, so the hand ends on it. Returns the settle result, or None if not needed."""
        st = None
        aim = np.asarray(goal, float).copy()
        for _ in range(rounds):
            e = self.p() - np.asarray(goal)
            lat = e - self.zin * float(e @ self.zin)
            if 1000 * float(np.linalg.norm(lat)) <= tol_mm:
                break
            aim = aim - lat
            self.free(aim, qh, speed=0.005, force_max=15.0, timeout=1.0)
            st = self.settle(aim, R_goal, pos_tol_mm=tol_mm, max_s=2.0)
            e = self.p() - np.asarray(goal)
            lat = e - self.zin * float(e @ self.zin)
            st = dict(st, lateral_err_vs_goal_mm=round(1000 * float(np.linalg.norm(lat)), 3))
        return st

    def touch_point(self, o=None):
        """Where the part touches: the line of action of F (through tau about the TCP) cut with the plane of the tip.
        Returns hole-frame (x, y) in mm, or None when the force is too small or too horizontal to say."""
        F = self.F(o)
        T = self.tau(o)
        f2 = float(F @ F)
        if f2 < PROBE_MIN_N ** 2:
            return None
        n = F / math.sqrt(f2)
        if abs(float(n @ self.zin)) < 0.3:
            return None
        base = self.p(o) + np.cross(F, T) / f2
        plane_s = float(self.tip(o) @ self.zin)
        lam = (plane_s - float(base @ self.zin)) / float(n @ self.zin)
        c = base + lam * n
        return self.to_hole(c)[:2] * 1000.0

    # ------------------------------------------------------------------------------------------ state
    def depth(self, o=None) -> dict:
        o = o or self.last
        out = dict(seat_depth_mm=float(self.b["hole"]["seat_depth_mm"]))
        out["below_touch_mm"] = (round(1000 * (self.s(o) - float(self.mem["surface_s"])), 2)
                                 if self.mem.get("surface_s") is not None else None)
        out["below_believed_surface_mm"] = round(float(self.to_hole(self.tip(o))[2]) * 1000, 2)
        return out

    def stage_est(self, o=None) -> str:
        o = o or self.last
        if not self.holding(o):
            return "released" if self.mem.get("released") else "no_approach"
        d = self.depth(o)
        bt = d["below_touch_mm"]
        seat = d["seat_depth_mm"]
        if (bt is not None and bt >= seat - SEAT_TOL_MM) or (
                bt is None and self.b["axial_sigma_mm"] < 3 and d["below_believed_surface_mm"] >= seat - SEAT_TOL_MM):
            return "seated_est"
        if (bt is not None and bt >= ENGAGE_MM) or self.mem.get("engaged"):
            return "engaged"
        if self.fmag(o) >= CONTACT_N or (self.mem.get("surface_s") is not None and self.fmag(o) >= 0.5
                                         and bt is not None and abs(bt) < ENGAGE_MM):
            return "contact"
        th = self.to_hole(self.tip(o)) * 1000
        if math.hypot(th[0], th[1]) <= 2.0 and th[2] >= -20.0:
            return "above_target_believed"
        return "lifted"

    def report(self, tool: str, ok: bool, code: str, summary: str, **extra) -> dict:
        o = self.last or self.observe()
        tp = self.touch_point(o) if self.fmag(o) >= CONTACT_N else None
        cstate = extra.pop("contact_state", None)
        if cstate is None:
            cstate = "none" if self.fmag(o) < CONTACT_N else ("in_hole" if self.mem.get("engaged") else "surface")
        th = self.to_hole(self.tip(o)) * 1000
        out = dict(
            tool=tool, interface=INTERFACE, ok=bool(ok), code=code, summary=summary, tried=self.tried,
            contact=dict(state=cstate, force_N=round(self.fmag(o), 2), force_hole_N=_r(self.Re.T @ self.F(o), 2),
                         torque_Nm=_r(self.tau(o), 4), touch_point_hole_mm=None if tp is None else _r(tp, 2)),
            depth=self.depth(o), stage_est=self.stage_est(o), tip_hole_mm=_r(th, 2),
            tcp=dict(pos=_r(self.p(o), 5), quat=_r(self.q(o), 5)), gripper_width_mm=round(1000 * o["gripper_width_m"], 2),
            commands=self.n_cmd, commands_since_drop=int(self.mem.get("cmds_since_drop", 0)),
            sim_s=round(float(o["t"]) - (self.t0 if self.t0 is not None else float(o["t"])), 3),
            t=o["t"], belief_source=self.b["source"],
            stage_note="estimated from robot observations and the belief; the evaluator's rubric comes from env_finish")
        if self.err:
            out["env_error"] = self.err
        out.update(extra)
        self.mem.setdefault("calls", []).append(dict(tool=tool, ok=bool(ok), code=code, t=o["t"], commands=self.n_cmd))
        self.mem["calls"] = self.mem["calls"][-30:]
        out["memory"] = {k: v for k, v in self.mem.items() if k not in ("calls",)}
        return out

    def _ref_force(self):
        """Free-space F/T reference (the part's weight and offsets): taken when nothing touches."""
        o = self.last or self.observe()
        self.mem["F_ref"] = _r(o["wrench"]["force_N"], 4)
        self.mem["tau_ref"] = _r(o["wrench"]["torque_Nm"], 5)

    def _guard(self, tool):
        o = self.observe()
        if o.get("episode_over"):
            return self.report(tool, False, "EPISODE_OVER", "the episode is over; reset to a start")
        return None

    # ------------------------------------------------------------------------------------------ approach
    def approach(self, standoff_mm=10.0, offset_mm=(0.0, 0.0), speed=HOVER_SPEED, clear_mm=15.0) -> dict:
        """approach_hole + align. Square the believed part axis to the hole axis and hover the believed tip
        standoff_mm above the believed hole entry (+ offset_mm in the hole frame). Path: square the hand in place
        (rising first if the tip is lower than clear_mm above the goal), cross over, descend, then wait until the hand
        has converged (pos <= 0.3 mm, rotation <= 0.7 deg, at most 12 s). Every leg is guarded (stop at 3 N).
        ok: arrived and converged with no contact."""
        tool = "task_approach"
        g = self._guard(tool)
        if g:
            return g
        o = self.last
        if not self.holding(o):
            return self.report(tool, False, "NOT_HOLDING", "the gripper is open: nothing to insert")
        if self.fmag(o) < CONTACT_N and not self.mem.get("surface_s"):
            self._ref_force()
        if self.mem.get("grip_width_mm") is None:
            self.mem["grip_width_mm"] = round(1000 * float(o["gripper_width_m"]), 3)
        Rh = hand_rotation(self.b, self.q(o), self.q7(o))
        self.mem["q_hand"] = _r(mat_to_quat(Rh), 6)
        qh = mat_to_quat(Rh)
        goal_tip = self.hole_point(offset_mm, -float(standoff_mm))
        goal = self.tcp_for(goal_tip, Rh)
        tip_now = self.p(o) + Rh @ self.tip_tcp
        s_safe = float(goal_tip @ self.zin) - clear_mm / 1000.0     # tip coordinate clear_mm above the goal
        n0 = self.n_cmd
        st = {}
        try:
            lift = max(0.0, float(tip_now @ self.zin) - s_safe)
            p1 = self.p(o) - self.zin * lift
            r = self.free(p1, qh, speed=speed, force_max=15.0)
            if r["result"]["stopped"] == "stop_force":
                self._phase("square_and_rise", n0, "stop_force")
                return self.report(tool, False, "UNEXPECTED_CONTACT",
                                   f"touched something while squaring the hand ({self.fmag():.1f} N)")
            st = self.settle(p1, Rh, pos_tol_mm=2.0, rot_tol_deg=2.0, max_s=12.0)
            if self.perceive is not None:
                # a part can shift in the pads while the hand turns: look again, re-square, recompute the goal
                self.set_belief(self.perceive())
                Rh = hand_rotation(self.b, self.q(), self.q7())
                self.mem["q_hand"] = _r(mat_to_quat(Rh), 6)
                qh = mat_to_quat(Rh)
                goal_tip = self.hole_point(offset_mm, -float(standoff_mm))
                goal = self.tcp_for(goal_tip, Rh)
                self.free(self.p(), qh, speed=speed, force_max=15.0, timeout=3.0)
                st = self.settle(self.p(), Rh, pos_tol_mm=5.0, rot_tol_deg=1.0, max_s=4.0)
            p2 = goal + self.zin * float((self.p() - goal) @ self.zin)
            if float(np.linalg.norm(p2 - self.p())) > 0.001:
                r = self.free(p2, qh, speed=speed, force_max=15.0)
                if r["result"]["stopped"] == "stop_force":
                    self._phase("cross_over", n0, "stop_force")
                    return self.report(tool, False, "UNEXPECTED_CONTACT",
                                       f"touched something crossing over ({self.fmag():.1f} N)")
            r = self.free(goal, qh, speed=min(speed, 0.015), force_max=15.0)
            why = r["result"]["stopped"]
            if why != "stop_force":
                st = self.settle(goal, Rh)
                st = self.trim(goal, Rh, qh) or st
            self._phase("approach", n0, why, settle=st)
        except _EpisodeOver:
            return self.report(tool, False, "EPISODE_OVER", "the episode ended during the approach")
        if why == "stop_force":
            return self.report(tool, False, "UNEXPECTED_CONTACT",
                               f"touched something {standoff_mm} mm before the believed surface ({self.fmag():.1f} N): "
                               "the board may be higher than believed or the part longer in the hand")
        self._ref_force()
        th = self.to_hole(self.tip()) * 1000
        return self.report(tool, True, "OK",
                           f"hovering: believed tip {-th[2]:.1f} mm above the believed surface, "
                           f"{math.hypot(th[0], th[1]):.2f} mm off the believed axis; hand squared to the hole "
                           f"(converged to {st.get('lateral_err_mm')} mm / {st.get('rot_err_deg')} deg). "
                           + ("The belief is uncorrected: the true hole is off by the board error (median 1.3 mm) and "
                              "the tip by the in-hand pose, so localize by contact next."
                              if self.b["source"] == "believed" else ""))

    # ------------------------------------------------------------------------------------------ guarded move
    def move_until_contact(self, direction="down", distance_mm=40.0, speed=DESCENT_SPEED, stop_force=3.0,
                           max_force=8.0, relief_mm=0.0, probe_force=3.0, retouch=True, learn_tip=False,
                           slow_zone=True) -> dict:
        """Guarded move from where the hand is: along the hole axis ('down' = into the hole, 'up'), a hole-frame
        axis ('+x', '-x', '+y', '-y') or a hole-frame vector [x, y, z]. Stops when |F - F_ref| reaches stop_force.
        Downward moves are two-speed (slow_zone): at `speed` to the top of the band where contact is possible (the
        believed surface minus its height uncertainty and 2 mm), converge sideways, then at 4 mm/s with a light stop
        (0.7 x stop_force, >= 1.5 N). A stop above the band RE-TOUCHES (lift 1.5 mm onto the commanded line, converge,
        touch again at 3 mm/s: a fast stop freezes a sideways lag). Then it PROBES: a steady press of about probe_force N for 0.4 s, which (1) lets a tip that
        is over the hole sink in, (2) gives a static wrench for the touch point and (3) measures the arm's stiffness
        (k_est_N_m, used by later presses). Classifies the contact: surface (the first downward touch sets the depth
        reference), in_hole (the tip sank more than 1 mm during the probe, is more than 2 mm past an earlier surface
        touch, or is deeper than the belief's height uncertainty past the believed surface), none.
        With learn_tip, the first surface touch's static touch point also corrects where the tip is (memory
        tip_shift_mm, used by every later tool). Off by default: a flat tip face can touch anywhere on its face, so
        the touch point is off by up to the part radius (it misled the search on round peg 2, Lessons); use it when
        the touch point is far (> radius) from the believed tip.
        relief_mm backs off along the reverse direction afterwards (Google's `relief`). ok: contact made."""
        tool = "task_move_until_contact"
        learn_tip = learn_tip and self.b["axial_sigma_mm"] >= 3
        g = self._guard(tool)
        if g:
            return g
        o = self.last
        if self.mem.get("F_ref") is None and self.fmag(o) < CONTACT_N:
            self._ref_force()
        try:
            dvec = self._direction(direction)
        except ValueError as ex:
            return self.report(tool, False, "BAD_ARGS", str(ex))
        n0 = self.n_cmd
        p_start = self.p(o)
        target = p_start + dvec * float(distance_mm) / 1000.0
        cls, tp, F_stop, probe = "none", None, 0.0, None
        down = float(dvec @ self.zin) > 0.7
        try:
            slow = False
            if down and slow_zone:
                # fast to the top of the band where contact is possible (believed surface minus the height
                # uncertainty and 2 mm), converge sideways, then slowly through the band: a slow first touch keeps the
                # sideways lag (~0.35 mm per 4 mm/s) small, and a light stop keeps thin pegs from pivoting in the pads
                tip_s = float(self.tip() @ self.zin)
                top_s = float(self.H @ self.zin) - (self.b["axial_sigma_mm"] + 2.0) / 1000.0
                lead = top_s - tip_s
                if lead > 0.001:
                    r = self.free(p_start + dvec * lead, speed=speed, force_max=max_force)
                    if r["result"]["stopped"] != "stop_force":
                        self.settle(p_start + dvec * lead, self.Rh(), pos_tol_mm=0.15, rot_tol_deg=1.0, max_s=2.0)
                        slow = True
                else:
                    slow = True
                if slow and self.perceive is not None:
                    # look again just before the band (thin pegs creep in the pads) and re-aim the tip at the axis
                    self.set_belief(self.perceive())
                    off = self.to_hole(self.tip())[:2]
                    if float(np.linalg.norm(off)) > 0.00005:
                        aim = self.p() - self.Re[:, :2] @ off
                        self.free(aim, speed=0.005, force_max=max_force, timeout=1.0)
                        self.settle(aim, self.Rh(), pos_tol_mm=0.1, rot_tol_deg=1.0, max_s=1.0)
                    target = self.p() + dvec * (float(distance_mm) / 1000.0 - float((self.p() - p_start) @ dvec))
                if slow:
                    v = SLOW_SPEED if self.b["axial_sigma_mm"] >= 3 else SLOW_SPEED_KNOWN
                    # guard only the band where contact is possible: end it 4 mm past the depth where an untouched
                    # tip must be in the hole (height uncertainty + 2 mm past the believed surface), so a guarded leg
                    # (the env caps it at 1.5-3 mm/s by the stop force) stays under the env's 20 s move timeout. The
                    # 4 mm keep the spring pushing a tip that rubs in a tight hole (1.5 mm left one square peg
                    # settled 2.9 mm in, short of the in-hole depth)
                    end_s = float(self.H @ self.zin) + (self.b["axial_sigma_mm"] + ENGAGE_MM + 4.0) / 1000.0
                    room = end_s - float(self.tip() @ self.zin)
                    rest = float((target - self.p()) @ dvec)
                    if 0.0 < room < rest:
                        target = self.p() + dvec * room
                    # no timeout is passed: the env sets it from the capped speed
                    r = self.move(target, speed=v, force_max=max_force, stop_force=max(1.5, 0.7 * stop_force))
            else:
                r = self.move(target, speed=speed, force_max=max_force, stop_force=stop_force)
            why = r["result"]["stopped"]
            F_stop = self.fmag()
            self._phase("guarded_move", n0, why, travel_mm=round(1000 * float(np.linalg.norm(self.p() - p_start)), 2),
                        force_at_stop_N=round(F_stop, 2), slow_band=slow,
                        env_stop_overshoot_N=r["result"].get("stop_overshoot_N"),
                        env_speed_used=r["result"].get("speed_used"))
            if why == "stop_force" and down and retouch and not slow:
                # The arm lags sideways while it moves (about 0.9 mm at 10 mm/s, 2.3 mm at 25 mm/s; Lessons) and the
                # stop freezes that lag, while pressing sideways cannot undo it (friction). So: lift 1.5 mm back onto
                # the commanded line, let it converge, and touch again slowly (about 0.3 mm of lag at 3 mm/s).
                n1 = self.n_cmd
                s_hit0 = self.s()
                line = p_start + self.zin * float((self.p() - p_start) @ self.zin - 0.0015)
                self.move(line, speed=0.01, force_max=max_force)
                st = self.settle(line, self.Rh(), pos_tol_mm=0.15, rot_tol_deg=1.0, max_s=3.0)
                r = self.move(line + self.zin * 0.0060, speed=0.003, force_max=max_force,
                              stop_force=max(1.5, 0.7 * stop_force))
                why = r["result"]["stopped"]
                F_stop = self.fmag()
                self._phase("retouch", n1, why, settle=st, below_first_touch_mm=round(1000 * (self.s() - s_hit0), 2))
                if why != "stop_force":
                    # nothing stopped the slow re-touch 4.5 mm past the first touch: the tip is in the hole
                    self.mem["surface_s"] = round(s_hit0, 6)
                    self.mem["engaged"] = True
                    return self.report(tool, True, "OK", f"the slow re-touch went {1000 * (self.s() - s_hit0):.1f} mm "
                                       "past the first touch without resistance: the tip is in the hole",
                                       contact_state="in_hole")
            if why == "stop_force":
                s_hit = self.s()
                if down and probe_force:
                    probe = self.probe(probe_force, check_grip=True)
                    tp = probe.get("touch_point_hole_mm")
                ss = self.s()
                bel = self.depth()["below_believed_surface_mm"]
                if self.mem.get("surface_s") is not None:
                    cls = "in_hole" if 1000 * (ss - float(self.mem["surface_s"])) > ENGAGE_MM else "surface"
                elif probe and probe["sank_mm"] > DROP_MM:
                    cls = "in_hole"
                    self.mem["surface_s"] = round(s_hit, 6)
                else:
                    cls = "in_hole" if bel > self.b["axial_sigma_mm"] + ENGAGE_MM else "surface"
                    if cls == "surface" and down:
                        self.mem["surface_s"] = round(s_hit, 6)
                if cls == "surface" and down:
                    self.mem["touch_tcp"] = _r(self.p(), 6)
                    self.mem["touch_point_hole_mm"] = None if tp is None else _r(tp, 3)
                    if tp is not None and learn_tip and self.mem.get("tip_shift_mm") is None:
                        bt = self.to_hole(self.tip())[:2] * 1000
                        shift = np.asarray(tp, float) - bt
                        if float(np.linalg.norm(shift)) > TIP_SHIFT_MIN_MM:
                            self.mem["tip_shift_mm"] = _r(np.clip(shift, -TIP_SHIFT_MAX_MM, TIP_SHIFT_MAX_MM), 3)
                if cls == "in_hole":
                    self.mem["engaged"] = True
            if relief_mm and why == "stop_force":
                n1 = self.n_cmd
                self.move(self.p() - dvec * relief_mm / 1000.0, speed=0.01, force_max=max_force)
                self._phase("relief", n1, "done")
        except _EpisodeOver:
            return self.report(tool, False, "EPISODE_OVER", "the episode ended during the move")
        if why != "stop_force":
            bel = self.depth()["below_believed_surface_mm"]
            if down and bel > self.b["axial_sigma_mm"] + ENGAGE_MM:
                self.mem["engaged"] = True
                if self.mem.get("surface_s") is None:
                    self.mem["surface_s_est"] = round(self.s() - bel / 1000.0, 6)
                return self.report(tool, False, "NO_CONTACT",
                                   f"no contact within {distance_mm} mm ({why}), but the believed tip is {bel:.1f} mm "
                                   "past the believed surface, beyond its height uncertainty: probably in the hole "
                                   "(engaged)", contact_state="in_hole")
            return self.report(tool, False, "NO_CONTACT", f"no contact within {distance_mm} mm ({why})",
                               contact_state="none")
        d = self.depth()
        msg = (f"contact ({cls}) at {F_stop:.1f} N; believed tip {d['below_believed_surface_mm']:.1f} mm past the "
               "believed surface")
        if tp is not None:
            msg += f"; the part touches at ({tp[0]:.1f}, {tp[1]:.1f}) mm from the believed hole axis"
        if probe:
            msg += f"; probe press {probe['force_N']:.1f} N, sank {probe['sank_mm']:.2f} mm"
            gc = probe.get("grip_check") or {}
            if gc.get("pivot"):
                msg += (f". GRIP CHECK: the part turns in the fingers (contact moved {gc['contact_shift_mm']:.2f} mm in "
                        f"the hand, hand sank {gc['sink_mm']:.2f} mm between the two presses); task_search and "
                        "task_insert will refuse until the grip is corrected (corrections.grip) or the part regrasped")
        return self.report(tool, True, "OK", msg, contact_state=cls, probe=probe,
                           stop=dict(force_N=round(F_stop, 2), depth=d))

    def probe(self, force=3.0, hold_s=0.4, line_point=None, check_grip=False) -> dict:
        """Steady press along the hole axis: command a target force/k below (on line_point's lateral position if
        given) with the ceiling at force, hold, read. Returns the force, how far the TCP sank, the stiffness estimate
        and the static touch point. With check_grip it first presses at about half the force and compares the two
        presses (grip_check): a part turning in the fingers moves its contact point in the hand frame and lets the
        hand sink; a rigid grip does neither (PIVOT_SHIFT_MM, PIVOT_SINK_MM)."""
        n0 = self.n_cmd
        k = float(self.mem.get("k_est_N_m") or K_DEFAULT)
        s0 = self.s()
        base = self.p() if line_point is None else np.asarray(line_point, float)
        base = base + self.zin * (self.s() - float(base @ self.zin))
        gc = None
        if check_grip and force >= 2.5:
            f1 = max(PROBE_MIN_N + 0.5, 0.5 * force)
            tgt1 = base + self.zin * max(0.0015, 1.5 * f1 / k)
            d1 = float(np.linalg.norm(tgt1 - self.p()))
            self.move(tgt1, speed=0.01, force_max=f1, timeout=d1 / 0.01 + 0.3)
            o1 = self.last
            c1, s1, F1 = self.contact_hand(o1), self.s(o1), self.fmag(o1)
        depth_cmd = max(0.002, 1.5 * force / k)
        tgt = base + self.zin * depth_cmd
        dist = float(np.linalg.norm(tgt - self.p()))
        self.move(tgt, speed=0.01, force_max=force, timeout=dist / 0.01 + 0.2)
        self.wait(hold_s)
        o = self.last
        if check_grip and force >= 2.5:
            c2, s2, F2 = self.contact_hand(o), self.s(o), self.fmag(o)
            if c1 is not None and c2 is not None and F2 - F1 > 0.5:
                shift = float(np.linalg.norm(np.asarray(c2) - np.asarray(c1)))
                sink = 1000 * (s2 - s1)
                pivot = shift >= PIVOT_SHIFT_MM and PIVOT_SINK_MM <= sink < DROP_MM   # a deeper sink: over the hole
                gc = dict(contact_hand_mm=_r(c2, 2), contact_shift_mm=round(shift, 3), sink_mm=round(sink, 3),
                          force_step_N=[round(F1, 2), round(F2, 2)], pivot=bool(pivot))
                if pivot:
                    # pivot about the pads with the tip contact fixed: the hand sinks x_c * dth while the contact moves
                    # z_c * dth in the hand, so atan(sink / shift) is the contact's angle off the hand axis seen from
                    # the pads: a rough in-hand tilt (it also folds in the along-pad offset); sign from the direction
                    u = (np.asarray(c2) - np.asarray(c1)) / max(shift, 1e-9)
                    tilt = math.degrees(math.atan2(sink, shift))
                    self.mem["grip_pivot"] = dict(contact_shift_mm=round(shift, 3), sink_mm=round(sink, 3),
                                                  direction_hand=_r(u, 2), tilt_est_deg=round(tilt, 1),
                                                  suggested_grip=dict(pivot_deg=round(tilt * float(u[0]), 1),
                                                                      rock_deg=round(-tilt * float(u[1]), 1)))
            else:
                gc = dict(contact_hand_mm=None if c2 is None else _r(c2, 2), pivot=None,
                          note="too little force to compare the two presses")
        Fax = abs(float(self.F(o) @ self.zin))
        err = float((tgt - self.p(o)) @ self.zin)
        if err > 0.0005 and Fax > PROBE_MIN_N and Fax < 0.95 * force:
            self.mem["k_est_N_m"] = round(Fax / err, 1)
        tp = self.touch_point(o)
        out = dict(force_N=round(self.fmag(o), 2), sank_mm=round(1000 * (self.s(o) - s0), 3),
                   k_est_N_m=self.mem.get("k_est_N_m"), touch_point_hole_mm=None if tp is None else _r(tp, 2))
        if gc is not None:
            out["grip_check"] = gc
        self._phase("probe", n0, "held", **out)
        return out

    def _direction(self, d) -> np.ndarray:
        if isinstance(d, str):
            m = {"down": self.zin, "up": -self.zin, "+x": self.Re[:, 0], "-x": -self.Re[:, 0], "+y": self.Re[:, 1],
                 "-y": -self.Re[:, 1]}
            if d not in m:
                raise ValueError(f"direction must be down | up | +x | -x | +y | -y or a hole-frame [x, y, z]; got {d!r}")
            return m[d]
        v = np.asarray(d, float)
        if v.shape != (3,) or np.linalg.norm(v) < 1e-9:
            raise ValueError("direction vector must be 3 numbers, not all zero")
        return self.Re @ (v / np.linalg.norm(v))

    # ------------------------------------------------------------------------------------------ re-aim
    def reaim(self, max_shift_mm=8.0, lift_mm=1.5, speed=0.01) -> dict:
        """After a surface touch: shift the hand so the touch point (where the part really touches, from the wrench)
        moves onto the believed hole axis. Corrects the in-hand offset, not the board error."""
        tp = self.mem.get("touch_point_hole_mm")
        if tp is None:
            return dict(shift_mm=None, why="no touch point")
        sh = -np.asarray(tp, float)
        if np.linalg.norm(sh) > max_shift_mm:
            sh = sh * (max_shift_mm / np.linalg.norm(sh))
        p = self.p()
        up = p - self.zin * lift_mm / 1000.0
        dxy = self.Re[:, :2] @ (sh / 1000.0)
        n0 = self.n_cmd
        self.move(up, speed=speed)
        self.move(up + dxy, speed=speed)
        self._phase("reaim", n0, "done", shift_mm=_r(sh, 2))
        return dict(shift_mm=_r(sh, 2))

    # ------------------------------------------------------------------------------------------ search
    def search_defaults(self, max_moves=32) -> dict:
        """Pitch from the capture width: the tip drops when it is within chamfer + clearance of the axis, so turns
        must be closer than twice that; 1.2 x (chamfer + clearance) leaves margin for the sliding lag (at most 2 mm).
        Chords 2.5 pitches long (>= 2 mm)."""
        ch = float(self.b.get("part_geometry", {}).get("chamfer_mm", 0.4))
        cl = float(min(self.b["hole"]["clearance_mm"]))
        pitch = max(0.3, min(2.0, 1.2 * (ch + cl)))
        chord = max(2.0, 2.5 * pitch)
        return dict(pitch_mm=round(pitch, 2), chord_mm=round(chord, 2),
                    radius_mm=round(min(8.0, math.sqrt(max_moves * chord * pitch / math.pi)), 2))

    def search(self, pattern="spiral", radius_mm=None, pitch_mm=None, press_force=0.8, speed=0.004, max_force=8.0,
               max_moves=32, chord_mm=None, drop_mm=DROP_MM, center="here", confirm_force=2.0) -> dict:
        """Search for the hole on the surface under a light press, centred where the tip is now (center='here') or on
        the believed hole axis (center='hole': lift 1 mm, cross over with the contact-corrected tip, touch down). Needs a surface
        touch first (the depth reference). Moves are chords about chord_mm long of an Archimedean spiral (turns
        pitch_mm apart) or the legs of a raster (rows pitch_mm apart). Unset pitch/chord/radius come from the part
        (search_defaults: pitch 1.2 x (chamfer + clearance), radius what the moves left cover, at most 8 mm). The press comes from a target press_mm below
        the touch height chosen so the spring presses press_force N (stiffness from the probe; a light press keeps the
        sliding lag behind the target near friction x press depth). Stops when the tip has dropped drop_mm below the
        touch height; a smaller sink (>= 0.5 mm: a chamfer caught on the edge) stops the slide for a confirming
        press (confirm_force), at most twice per search. The pattern is cut short to leave 4 commands before the env's no-progress limit.
        Keep pitch_mm under twice (chamfer + clearance) or the tip can pass over the hole.
        ok: dropped (engaged)."""
        tool = "task_search"
        g = self._guard(tool)
        if g:
            return g
        o = self.last
        if self.mem.get("surface_s") is None:
            return self.report(tool, False, "NO_SURFACE_REF",
                               "no surface touch yet: call task_move_until_contact (down) first")
        if pattern not in ("spiral", "raster"):
            return self.report(tool, False, "BAD_ARGS", "pattern must be spiral | raster")
        pb = self.pivot_block(tool)
        if pb:
            return pb
        s0 = float(self.mem["surface_s"])
        n_eff = max(4, min(int(max_moves), self.moves_left() - (3 if center == "hole" else 0)))
        auto = self.search_defaults(n_eff)
        pitch_mm = auto["pitch_mm"] if pitch_mm is None else float(pitch_mm)
        chord_mm = auto["chord_mm"] if chord_mm is None else float(chord_mm)
        if radius_mm is None:
            radius_mm = min(8.0, math.sqrt(n_eff * chord_mm * pitch_mm / math.pi))
        press_m = min(0.006, max(0.001, press_force / float(self.mem.get("k_est_N_m") or K_DEFAULT)))
        drop_mm = min(drop_mm, 0.6 * 1000 * press_m)      # the light press can sink the tip only press_m
        pts = (_spiral(radius_mm, pitch_mm, max_moves, chord_mm) if pattern == "spiral"
               else _raster(radius_mm, pitch_mm, max_moves))
        n0 = self.n_cmd
        found = False
        capped = False
        slip = False
        candidates = 0
        path = 0.0
        prev = np.zeros(2)
        qh = mat_to_quat(self.Rh())
        tip_xy = self.to_hole(self.tip(o))[:2]
        c = self.p(o) - self.zin * (self.s(o) - s0)          # TCP at the touch height, where the tip is now
        try:
            if center == "hole" and float(np.linalg.norm(tip_xy)) > 0.0005:
                # lift, cross to the believed hole axis (corrected tip), touch down lightly
                c = c - self.Re[:, :2] @ tip_xy
                up = c - self.zin * 0.001
                self.move(self.p() - self.zin * (self.s() - float(up @ self.zin)), qh, speed=0.01)
                self.move(up, qh, speed=0.01)
                r = self.move(up + self.zin * 0.004, qh, speed=0.003, force_max=max_force, stop_force=1.5)
                if 1000 * (self.s() - s0) >= drop_mm:
                    found = True
                elif r["result"]["stopped"] == "stop_force" and abs(self.s() - s0) < 0.002:
                    c = c + self.zin * (self.s() - s0)     # local surface height at the centre
                    s0 = self.s()
            budget = self.moves_left()
            if len(pts) > budget:
                pts = pts[:max(0, budget)]
                capped = True
            for x, y in ([] if found else pts):
                if self.moves_left() <= 3:
                    capped = True
                    break
                tgt = c + self.Re[:, :2] @ (np.array([x, y]) / 1000.0) + self.zin * press_m
                seg = math.hypot(x - prev[0], y - prev[1]) / 1000.0
                self.move(tgt, qh, speed=speed, force_max=max_force, timeout=seg / speed + 0.15)
                path += seg * 1000
                prev = np.array([x, y])
                if self.grip_lost():
                    slip = True
                    break
                sank = 1000 * (self.s() - s0)
                if sank >= drop_mm:
                    found = True
                    break
                if sank >= CANDIDATE_DROP_MM and candidates < 2:
                    # the chamfer may have caught the edge (it sinks only ~the chamfer height): stop sliding and press
                    candidates += 1
                    pr = self.probe(confirm_force)
                    if 1000 * (self.s() - s0) >= drop_mm:
                        found = True
                        break

        except _EpisodeOver:
            self._phase("search", n0, "episode_over", path_mm=round(path, 1))
            return self.report(tool, False, "EPISODE_OVER", "the episode ended during the search")
        o = self.last
        drop = 1000 * (self.s(o) - s0)
        if slip:
            self._phase("search", n0, "grip_slip", path_mm=round(path, 1), grip_closed_mm=self.grip_closed_mm(o))
            return self.report(tool, False, "GRIP_SLIP",
                               f"the fingers closed {self.grip_closed_mm(o):.1f} mm further than at the approach during "
                               "the search: the part is turning or slipping out of the grip. Lift and regrasp",
                               contact_state="surface")
        self._phase("search", n0, "dropped" if found else ("budget" if capped else "pattern_done"), pattern=pattern,
                    radius_mm=round(float(radius_mm), 2), pitch_mm=round(float(pitch_mm), 2), candidates=candidates,
                    path_mm=round(path, 1),
                    drop_mm=round(drop, 2))
        if found:
            self.mem["engaged"] = True
            off = self.to_hole(self.tip(o))[:2] * 1000
            self.mem["found_at_hole_mm"] = _r(off, 2)
            return self.report(tool, True, "OK",
                               f"dropped {drop:.1f} mm below the touch height after {path:.0f} mm of {pattern} "
                               f"({self.n_cmd - n0} moves); the hole is about ({off[0]:.1f}, {off[1]:.1f}) mm from "
                               "the believed axis", contact_state="in_hole")
        if capped:
            return self.report(tool, False, "NOT_FOUND",
                               f"stopped the {pattern} after {path:.0f} mm ({self.n_cmd - n0} moves) to leave "
                               f"{self.moves_left()} commands before the env's no-progress limit (40 commands or 60 sim "
                               "s without going deeper): the hole is not where the belief puts it. Correct the belief "
                               "(hole_shift_mm / grip) or use the touch point (learn_tip, center='hole')",
                               contact_state="surface")
        return self.report(tool, False, "NOT_FOUND",
                           f"no drop over a {radius_mm:.1f} mm {pattern} ({path:.0f} mm path, {self.n_cmd - n0} moves, "
                           f"press {press_force} N, {1000 * press_m:.1f} mm); the hole may be further than {radius_mm} mm (board error p90 4 mm) "
                           "or the part is tilted in the hand", contact_state="surface")

    # ------------------------------------------------------------------------------------------ push / seat
    def push(self, max_force=MAX_FORCE_DEFAULT, speed=0.02, wiggle_deg=2.0, max_wiggles=3, extra_mm=4.0,
             align=True) -> dict:
        """From engaged: drive the tip to the seat (target seat + extra_mm along the axis; the first push capped at
        6 N, later ones at max_force). Reports GRIP_SLIP when the hand goes further than the seat allows. On a stall
        short of the seat: (1) align: rotate the hand about the tip in the direction of the
        residual torque the walls apply (torque minus the push force's own lever about the believed tip), i.e. comply
        so a peg tilted in the hand stands up (1 deg per 0.02 N m, 0.3-3 deg a step, up to 3 steps); (2) wiggle:
        tilt +-wiggle_deg about both hole axes while pressing; then press again."""
        seat = float(self.b["hole"]["seat_depth_mm"])
        s0 = self.mem.get("surface_s")
        n0 = self.n_cmd
        if s0 is None:
            s0 = self.s() - self.depth()["below_believed_surface_mm"] / 1000.0
            self.mem["surface_s_est"] = round(s0, 6)
        s0 = float(s0)
        target_s = s0 + (seat + extra_mm) / 1000.0
        jam = False
        w = 0
        aligned_deg = 0.0
        slip = False
        for w in range(max_wiggles + 1):
            qh = mat_to_quat(self.Rh())
            dist = max(0.0, target_s - self.s())
            tgt = self.p() + self.zin * dist
            # first press at the operators' level (median push 2.0-2.5 N, peak ~6 N on train), then the ceiling: a
            # hard first push turns a part that is caught on the edge in the fingers (thin pegs)
            fm = min(max_force, FIRST_PUSH_N) if w == 0 else max_force
            self.move(tgt, qh, speed=speed, force_max=fm, timeout=dist / speed + 0.8)
            d = 1000 * (self.s() - s0)
            if d > seat + SLIP_MM or self.grip_lost():
                slip = True     # past the seat floor, or the fingers closed further: the part turned or slid in them
                break
            if d >= seat - SEAT_TOL_MM:
                break
            if w == max_wiggles or self.moves_left() < 9:
                jam = True
                break
            if align:
                for _ in range(3):
                    tip = self.tip()
                    F, T = self.F(), self.tau()
                    T_res = T - np.cross(tip - self.p(), F)
                    T_lat = T_res - self.zin * float(T_res @ self.zin)
                    mag = float(np.linalg.norm(T_lat))
                    if mag < 0.006:
                        break
                    ang = math.radians(min(3.0, max(0.3, mag / 0.02)))
                    R = rot_axis(T_lat / mag, ang) @ self.Rh()
                    p_new = tip - R @ self.tip_tcp - self.Re[:, :2] @ (np.asarray(self.mem.get("tip_shift_mm") or [0, 0],
                                                                                   float) / 1000.0)
                    self.move(p_new + self.zin * 0.002, mat_to_quat(R), speed=0.005, force_max=max_force, timeout=1.2)
                    self.mem["q_hand"] = _r(mat_to_quat(R), 6)
                    aligned_deg += math.degrees(ang)
                    if 1000 * (self.s() - s0) >= seat - SEAT_TOL_MM:
                        break
            for ax, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
                R = rot_axis(self.Re[:, ax], sign * math.radians(wiggle_deg)) @ self.Rh()
                self.move(self.p() + self.zin * 0.002, mat_to_quat(R), speed=0.01, force_max=max_force, timeout=0.6)
        d = 1000 * (self.s() - s0)
        if slip:
            # back off so the part is not driven further over
            self.move(self.p() - self.zin * 0.002, mat_to_quat(self.Rh()), speed=0.01, force_max=max_force)
        self._phase("push", n0, "grip_slip" if slip else ("jam" if jam else "seated_est"),
                    depth_below_touch_mm=round(d, 2), wiggle_rounds=w, aligned_deg=round(aligned_deg, 2))
        return dict(depth_mm=round(d, 2), seated_est=(not slip) and d >= seat - SEAT_TOL_MM, jam=jam, slip=slip,
                    wiggle_rounds=w, aligned_deg=round(aligned_deg, 2), force_N=round(self.fmag(), 2),
                    grip_closed_mm=self.grip_closed_mm())

    # ------------------------------------------------------------------------------------------ insert
    def insert(self, strategy="auto", max_force=MAX_FORCE_DEFAULT, params: dict | None = None,
               release=False) -> dict:
        """Composite insertion: approach (unless already hovering or touching), guarded descent with a probe press,
        then by strategy
          auto    corrected belief (grip given): small spiral (1.5 mm, pitch 0.5) if the probe did not sink, then push;
                  uncorrected: as spiral
          press   push straight in from the first contact
          spiral  spiral search centred on the believed hole with the touch-corrected tip, push (align, wiggle)
          wiggle  as spiral, with larger and more wiggles at a stall
          tilt    as spiral, but search with the part tilted (tilt_deg) so its leading rim catches the hole edge
        release: open and retreat after an estimated seat. ok: estimated seated (and released if asked)."""
        tool = "task_insert"
        corrected = self.b["axial_sigma_mm"] < 3
        P = dict(standoff_mm=12.0, descent_speed=DESCENT_SPEED, stop_force=3.0, probe_force=3.0,
                 radius_mm=1.5 if corrected else None, pitch_mm=0.5 if corrected else None, chord_mm=None,
                 press_force=0.8, search_speed=0.004, max_moves=32, wiggle_deg=2.0, max_wiggles=3, push_speed=0.03,
                 tilt_deg=8.0, retreat_mm=30.0, center="here", learn_tip=False)
        bad = sorted(set(params or {}) - set(P))
        if bad:
            return self.report(tool, False, "BAD_ARGS", f"unknown params {bad}; known: {sorted(P)}")
        P.update(params or {})
        if strategy not in ("auto", "press", "spiral", "wiggle", "tilt"):
            return self.report(tool, False, "BAD_ARGS", "strategy must be auto | press | spiral | wiggle | tilt")
        if strategy == "auto":
            strategy = "precise" if corrected else "spiral"
        if strategy == "wiggle":
            P["wiggle_deg"] = max(P["wiggle_deg"], 3.0)
            P["max_wiggles"] = max(P["max_wiggles"], 5)
        g = self._guard(tool)
        if g:
            return g
        o = self.last
        if not self.holding(o):
            return self.report(tool, False, "NOT_HOLDING", "the gripper is open")
        try:
            if not self.mem.get("engaged"):
                if self.mem.get("surface_s") is None:
                    th = self.to_hole(self.tip(o)) * 1000
                    near = math.hypot(th[0], th[1]) <= 1.0 and -(P["standoff_mm"] + 5) <= th[2] <= 0
                    if self.mem.get("q_hand") is None or not near:
                        a = self.approach(standoff_mm=P["standoff_mm"])
                        if not a["ok"]:
                            return self.report(tool, False, a["code"], "approach: " + a["summary"])
                    g = self.move_until_contact("down", distance_mm=P["standoff_mm"] + 2 * self.b["axial_sigma_mm"] + 5,
                                                speed=P["descent_speed"], stop_force=P["stop_force"],
                                                max_force=max_force, probe_force=P["probe_force"],
                                                learn_tip=bool(P["learn_tip"]))
                    if not g["ok"] and not self.mem.get("engaged"):
                        return self.report(tool, False, g["code"], "descent: " + g["summary"])
                if not self.mem.get("engaged"):
                    pb = self.pivot_block(tool)
                    if pb:
                        return pb
                if not self.mem.get("engaged") and strategy != "press":
                    if not self.mem.get("engaged"):
                        if strategy == "tilt":
                            sr = self.tilt_search(P)
                        else:
                            sr = self.search("spiral", radius_mm=P["radius_mm"], pitch_mm=P["pitch_mm"],
                                             press_force=P["press_force"], speed=P["search_speed"], max_force=max_force,
                                             max_moves=P["max_moves"], chord_mm=P["chord_mm"],
                                             center=P["center"])
                        if not sr["ok"]:
                            return self.report(tool, False, sr["code"], "search: " + sr["summary"])
            pr = self.push(max_force=max_force, speed=P["push_speed"], wiggle_deg=P["wiggle_deg"],
                           max_wiggles=P["max_wiggles"])
        except _EpisodeOver:
            return self.report(tool, False, "EPISODE_OVER", "the episode ended during the insertion")
        if pr.get("slip"):
            why = (f"the fingers closed {pr['grip_closed_mm']:.1f} mm further than at the approach"
                   if (pr.get("grip_closed_mm") or 0) > GRIP_CLOSE_MM else
                   f"the hand went {pr['depth_mm']:.1f} mm past the first touch, deeper than the seat "
                   f"({self.b['hole']['seat_depth_mm']} mm)")
            return self.report(tool, False, "GRIP_SLIP",
                               why + ": the part is turning or sliding in the fingers. Lift, release and regrasp, or "
                               "retry with a lower max_force", contact_state="jam", push=pr)
        if not pr["seated_est"]:
            code = "JAMMED" if pr["jam"] else "SHALLOW"
            return self.report(tool, False, code,
                               f"stopped {pr['depth_mm']:.1f} mm below the touch height (seat "
                               f"{self.b['hole']['seat_depth_mm']} mm) after {pr['wiggle_rounds']} wiggle rounds at "
                               f"<= {max_force} N", contact_state="jam", push=pr)
        if release:
            rr = self.release(retreat_mm=P["retreat_mm"])
            return self.report(tool, rr["ok"], rr["code"],
                               f"seated (estimated {pr['depth_mm']:.1f} mm below the touch height); " + rr["summary"],
                               push=pr)
        return self.report(tool, True, "OK",
                           f"seated (estimated): {pr['depth_mm']:.1f} mm below the touch height, seat "
                           f"{self.b['hole']['seat_depth_mm']} mm; still holding (call task_release)",
                           contact_state="in_hole", push=pr)

    # ------------------------------------------------------------------------------------------ tilt search
    def tilt_search(self, P) -> dict:
        """Tilt-then-insert for chamfer-less holes: tilt the part tilt_deg so its rim leads, press the leading rim
        on the surface and spiral; a tilted rim drops when it is anywhere over the hole (capture range about the hole
        radius instead of chamfer + clearance). On the drop, rotate upright about the tip while pressing."""
        th = math.radians(P["tilt_deg"])
        R0 = self.Rh()
        Rt = rot_axis(self.Re[:, 1], th) @ R0
        n0 = self.n_cmd
        tipw = self.tip() - self.zin * 0.002
        self.move(tipw - Rt @ self.tip_tcp, mat_to_quat(Rt), speed=0.01, force_max=15.0)
        self.settle(tipw - Rt @ self.tip_tcp, Rt, pos_tol_mm=0.5, rot_tol_deg=1.0, max_s=6.0)
        self.move(self.p() + self.zin * 0.010, mat_to_quat(Rt), speed=0.006, force_max=6.0, stop_force=2.5)
        s_t = self.s()
        c = self.p()
        found = False
        auto = self.search_defaults(P["max_moves"])
        for x, y in _spiral(P["radius_mm"] or auto["radius_mm"], 2 * (P["pitch_mm"] or auto["pitch_mm"]),
                            P["max_moves"], P["chord_mm"] or auto["chord_mm"]):
            tgt = c + self.Re[:, :2] @ (np.array([x, y]) / 1000.0) + self.zin * 0.004
            self.move(tgt, mat_to_quat(Rt), speed=P["search_speed"], force_max=P["press_force"], timeout=0.6)
            if self.grip_lost():
                break
            if 1000 * (self.s() - s_t) >= 0.5:
                found = True
                break
        if found:
            tipw = self.p() + Rt @ self.tip_tcp
            self.move(tipw - R0 @ self.tip_tcp + self.zin * 0.001, mat_to_quat(R0), speed=0.006, force_max=8.0,
                      timeout=3.0)
            self.settle(self.p(), R0, pos_tol_mm=5.0, rot_tol_deg=1.0, max_s=6.0)
        self._phase("tilt_search", n0, "dropped" if found else "pattern_done")
        if found:
            self.mem["engaged"] = True
            return self.report("task_search", True, "OK", "tilted rim dropped into the hole; uprighted")
        self.move(self.p() - self.zin * 0.002, mat_to_quat(R0), speed=0.01)
        return self.report("task_search", False, "NOT_FOUND", "tilt search found no drop")

    # ------------------------------------------------------------------------------------------ release
    def release(self, retreat_mm=30.0, speed=0.03) -> dict:
        """Open the gripper, then retreat retreat_mm along the hole axis. Checks from observations: the fingers opened
        past the part, and nothing pulled on the hand during the retreat. The part's own pose is not observable:
        env_finish scores it. ok: opened and retreated with no pull."""
        tool = "task_release"
        g = self._guard(tool)
        if g:
            return g
        w0 = float(self.last["gripper_width_m"])
        n0 = self.n_cmd
        try:
            self._do("gripper", action="open")
            w1 = float(self.last["gripper_width_m"])
            self.mem["released"] = True
            self._phase("open", n0, "done", width_mm=round(1000 * w1, 2))
            self._ref_force()
            n1 = self.n_cmd
            r = self.free(self.p() - self.zin * retreat_mm / 1000.0, speed=speed, force_max=MAX_FORCE_DEFAULT)
            why = r["result"]["stopped"]
            self._phase("retreat", n1, why)
        except _EpisodeOver:
            return self.report(tool, False, "EPISODE_OVER", "the episode ended during the release")
        opened = w1 > w0 + 0.003
        pulled = why == "stop_force"
        ok = opened and not pulled
        msg = (f"fingers {1000 * w0:.1f} -> {1000 * w1:.1f} mm; retreated {retreat_mm} mm "
               + ("with no pull" if not pulled else f"but something pulled ({self.fmag():.1f} N)"))
        wa = self.mem.get("grip_width_mm")
        if wa is not None and float(wa) - 1000 * w0 > GRIP_CLOSE_MM:
            msg += (f". Before opening, the fingers were {float(wa) - 1000 * w0:.1f} mm narrower than at the approach: "
                    "the part had slipped in the grip, so the seat estimate is doubtful")
        return self.report(tool, ok, "OK" if ok else ("FORCE_LIMIT" if pulled else "NOT_HOLDING"), msg)

    # ------------------------------------------------------------------------------------------ state
    def state(self) -> dict:
        g = self._guard("task_state")
        if g:
            return g
        return self.report("task_state", True, "OK", f"belief {self.b['source']}; estimated stage {self.stage_est()}",
                           belief=dict(source=self.b["source"], hole_entry_pos=_r(self.H, 5),
                                       hole_axis=_r(self.zin, 4), tip_in_tcp_mm=_r(self.tip_tcp * 1000, 2),
                                       grip={k: self.b["grip"][k] for k in ("dx_mm", "dy_mm", "grasp_from_top_mm",
                                                                            "pivot_deg", "rock_deg")},
                                       axial_sigma_mm=self.b["axial_sigma_mm"],
                                       yaw_symmetry_deg=self.b["hole"]["yaw_symmetry_deg"],
                                       clearance_mm=self.b["hole"]["clearance_mm"],
                                       corrections=self.b.get("corrections")))


def _spiral(radius_mm, pitch_mm, max_moves, chord_mm=2.0):
    """Archimedean spiral r = pitch * theta / 2pi as chord end points (hole-frame mm), from the centre outwards,
    chords about chord_mm long (at least 6 per turn)."""
    pts = []
    th = 0.0
    while len(pts) < max_moves:
        r = pitch_mm * th / (2 * math.pi)
        dth = min(2 * math.pi / 6, chord_mm / max(r, 1e-6)) if r > 0 else 2 * math.pi / 6
        th += dth
        r = pitch_mm * th / (2 * math.pi)
        if r > radius_mm + 1e-9:
            break
        pts.append((r * math.cos(th), r * math.sin(th)))
    return pts


def _raster(radius_mm, pitch_mm, max_moves):
    """Boustrophedon over a square of half-width radius_mm, rows pitch_mm apart, starting at the centre row."""
    rows = [0.0]
    k = 1
    while k * pitch_mm <= radius_mm + 1e-9:
        rows += [k * pitch_mm, -k * pitch_mm]
        k += 1
    pts = []
    side = 1
    for y in rows:
        pts.append((-side * radius_mm, y))
        pts.append((side * radius_mm, y))
        side = -side
    return pts[:max_moves]
