"""Practice environment for peg insertion, behind a capability boundary (stage D/E; skill playbook/skills/sim-integrity).

One process owns the simulator: MjModel/MjData, the REASSEMBLE impedance controller (twin/control.py), the frozen
evaluator (twin/evaluator.py) and the runtime monitors (twin/monitors.py). Actors never touch it. They reach it through
a narrow command API, the same for every actor:

  task                                     part, nominal hole entry pose, nominal grasp, limits, budget
  reset {start_id}                         harness only: start an episode from a frozen start (twin/starts.py)
  move {pos, quat?, speed?, force_max?, stop_force?, timeout?}
                                           absolute TCP target; the env interpolates at `speed` (capped at SPEED_CAP),
                                           runs it through the controller's 0.2 s target filter, and clamps the springs
                                           so the commanded force never exceeds force_max (capped at FORCE_CAP) and
                                           the commanded torque force_max * 0.05 m;
                                           stop_force ends the move at the first 1 kHz tick where |F| reaches it
                                           (a guarded move); a guarded approach is capped at guard_speed(stop_force)
                                           so the peak stays within GUARD_TOL of the threshold
  gripper {action: open | close | width, width?}   fixed 60 N grip force (changing it would change the model)
  wait {seconds}
  observe / status                         robot state, F/T in the base frame, gripper width, time / budget used
  finish {claim?}                          the env holds the last command 0.5 s, then the evaluator scores the episode

Brains reach it through harness tools (harness/tools/env.py -> a session daemon on a Unix socket: `serve`); compiled
skill programs through JSON lines on stdin/stdout from a sandboxed process (`run_program`, twin/sandbox.py). Both use
`PracticeEnv.command`. Privileged red-team scripts (`run_privileged`) run in this process with raw (m, d) access; they
may use red-team starts only, are logged, and their episodes never count.

Nothing changes the model after t = 0. The start's uncertain geometry (board error, in-hand pose) is applied before
t = 0 by building the scene with it (twin/replay.py build_model); the grasp is closed in a pre-roll before t = 0.
"""

from __future__ import annotations

import json
import math
import os
import select
import socket
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from twin import REPO_DIR

SPEED_CAP = 0.25            # m/s: above the fastest real approach (160 mm/s, twin/envelope.py, train) by ~1.5x
ANG_SPEED_CAP = 1.0         # rad/s
FORCE_CAP = 15.0            # N: the largest real peak of any train peg attempt is 15.6 N (envelope.json)
TORQUE_ARM = 0.05           # m: rotational spring ceiling = force_max * TORQUE_ARM (TCP to part tip is about 18-45 mm)
DEFAULT_SPEED = 0.02
DEFAULT_ANG_SPEED = 0.5
DEFAULT_FORCE = 10.0
MOVE_MAX_S = 20.0
WAIT_MAX_S = 5.0
GRIP_S = 1.0
GRIP_FORCE = 60.0           # N per finger (teleop_franka.py GripperWrapper.close(60))
GRIP_EMPTY_W = 0.0005       # m: a close that reaches this width holds nothing; the servo then targets closed (0)
FILTER_ALPHA = 0.005        # REASSEMBLE target filter per 1 kHz tick (time constant ~0.2 s)
PREROLL_S = 0.25
TRAJ_HZ = 30.0
STOP_TICKS = 1              # |F| >= stop_force at one 1 kHz control tick ends a guarded move
STOP_SETTLE_S = 0.05        # after a guarded stop the move holds 50 ms more, so its peak force includes the overshoot
# Guarded moves approach at most GUARD_SPEED(stop_force). The first touch of a held peg on the board is a step in force
# within one 1 ms tick (the twin's contact is stiff; B2 found it over-predicts real peaks), so detection alone cannot
# honour a threshold: the approach speed sets the overshoot, as on the real arm. Calibrated over 5 pegs x 2 contact
# settings (sim-integrity Lessons): the cap keeps the peak within GUARD_TOL(stop_force) of the threshold.
# 150 calibration stops (1.5 / 3 / 5 N x 1-4 mm/s x 5 pegs x c000 / c023): worst peak 1.81 N at 1.5 mm/s for 1.5 N
# (2.74 N at 2 mm/s), 3.00 N at 2 mm/s for 3 N (4.34 N at 3 mm/s), 5.00 N at 3 mm/s for 5 N (8.08 N at 4 mm/s)
GUARD_SPEED_TABLE = ((1.5, 0.0015), (3.0, 0.002), (5.0, 0.003))   # (stop_force N, max approach m/s), interpolated
GUARD_TOL = lambda sf: max(0.5, 0.2 * sf)  # noqa: E731  stated tolerance on the peak force, N


def guard_speed(stop_force: float) -> float:
    fs, vs = zip(*GUARD_SPEED_TABLE)
    return float(np.interp(stop_force, fs, vs))
WORKSPACE = dict(x=(-0.2, 1.0), y=(-0.8, 0.8), z=(-0.05, 0.9))
# Everything else build_model reads, pinned here so later edits of ReplayParams defaults cannot move the frozen twin.
# Values = twin/replay.py ReplayParams as of 2026-10-24 22:11 (the replay-validate frozen controller and grasp model).
PINNED_REPLAY = dict(zeta=1.0, joint_friction_scale=0.0, ns_every=10, ns_damping=0.1, ns_manip=0.0, ns_jl=0.0,
                     joint_damping=None, pad_torsion=0.002, pad_solref=None, finger_rock_k=0.0, finger_pivot_k=0.0,
                     cone="elliptic", impratio=10.0, plate_thickness=None, board_offset=(0.0, 0.0, 0.0),
                     pad_offset=0.0, legacy_hand=False, registration="none", hole_widen=0.0, hole_widen_by_part=None)
JOINT_RANGE = [[-2.9007, 2.9007], [-1.8361, 1.8361], [-2.9007, 2.9007], [-3.077, -0.1169], [-2.8763, 2.8763],
               [0.4398, 4.6216], [-3.0508, 3.0508]]  # menagerie fr3.xml: leaving it is flagged (plausibility.joint_range)
ACTOR_OPS = ("task", "move", "gripper", "wait", "observe", "status", "finish")
# harness-only: the env daemon serves these to harness tools and the brain runner; skill programs (run_program) get
# ACTOR_OPS only, and no brain-facing tool forwards render or end_limit as such
HARNESS_OPS = ACTOR_OPS + ("reset", "charge_tokens", "close", "ping", "render", "end_limit")
LIMIT_WHAT = ("brain_calls", "tokens", "wall_s", "env_commands", "sim_s", "other")


class EnvError(ValueError):
    pass


def _f(x, name, lo=None, hi=None) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise EnvError(f"{name} must be a number, got {x!r}")
    if not math.isfinite(v):
        raise EnvError(f"{name} must be finite")
    if lo is not None and v < lo:
        raise EnvError(f"{name} {v} below {lo}")
    if hi is not None and v > hi:
        raise EnvError(f"{name} {v} above {hi}")
    return v


def _vec(x, n, name) -> np.ndarray:
    if not isinstance(x, (list, tuple)) or len(x) != n:
        raise EnvError(f"{name} must be a list of {n} numbers")
    return np.array([_f(v, f"{name}[{i}]") for i, v in enumerate(x)])


def _relpath(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO_DIR)) if p.is_relative_to(REPO_DIR) else str(p)


def _r(a, k=6):
    return [round(float(v), k) for v in np.asarray(a).ravel()]


class ToolWrench:
    """What the wrist F/T sensor sees (same definition as twin/replay.py ToolSensor, vectorized): contact forces on the
    hand and fingers from everything else, world frame; torque about the world origin."""

    def __init__(self, m):
        import mujoco
        self.m = m
        mask = np.zeros(m.nbody, bool)
        for n in ("gripper/hand", "gripper/left_finger", "gripper/right_finger"):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
            if b >= 0:
                mask[b] = True
        self.mask = mask
        self._f6 = np.zeros(6)

    def wrench(self, d) -> tuple[np.ndarray, np.ndarray]:
        import mujoco
        n = d.ncon
        if n == 0:
            return np.zeros(3), np.zeros(3)
        con = d.contact
        tb = self.mask[self.m.geom_bodyid[con.geom[:n]]]
        sel = np.flatnonzero((tb[:, 0] != tb[:, 1]) & (con.efc_address[:n] >= 0))
        if len(sel) == 0:
            return np.zeros(3), np.zeros(3)
        fs = np.empty((len(sel), 3))
        for k, i in enumerate(sel):
            mujoco.mj_contactForce(self.m, d, int(i), self._f6)
            fs[k] = self._f6[:3]
        frames = con.frame[sel].reshape(-1, 3, 3)
        fw = np.einsum("kji,kj->ki", frames, fs)            # frame^T f: force on geom2, world
        sign = np.where(tb[sel, 1], 1.0, -1.0)[:, None]     # we want the force ON the tool body
        fw = fw * sign
        pos = con.pos[sel]
        return fw.sum(0), np.cross(pos, fw).sum(0)


def build_episode(ep: dict, nominal: dict):
    """Scene, controller and the pre-roll grasp, all before t = 0. ep = start + physics (twin.cousins.apply)."""
    import mujoco
    from twin import starts as S
    from twin.control import CartesianImpedance
    from twin.replay import GRIP_CLOSED, ReplayParams, build_model, close_fingers_on_part, impedance
    from twin.scene.build_base import START_ARM_QPOS
    part = ep["part"]
    ph = ep["physics"]
    ih = ep["inhand"]
    w = ph["solimp_width"]
    known = set(ReplayParams.__dataclass_fields__)
    rp = replace(ReplayParams(), **{k: v for k, v in PINNED_REPLAY.items() if k in known})
    rp = replace(rp, k_trans=ph["k_trans"], k_rot=ph["k_rot"], friction=ph["friction"],
                 pad_friction=ph["pad_friction"], solref_tc=ph["solref_tc"], solimp=(0.95, 0.99, w, 0.5, 2.0),
                 timestep=ph["timestep"], grasp_dx=ih["dx"], grasp_dy=ih["dy"],
                 grasp_from_top=ih["grasp_from_top"], grasp_rock=ih["rock"], grasp_pivot=ih["pivot"],
                 grip_force=GRIP_FORCE, surface_z=nominal["surface_z"], seat_depth=nominal["seat_depth"])
    extra = ph.get("replay_extra") or {}          # self-test physics overrides (privileged envs only)
    if extra:
        rp = replace(rp, **{k: (tuple(v) if isinstance(v, list) else v) for k, v in extra.items() if k in known})
    bp = S.true_board(nominal, ep["board_err"])
    bp["hole_xy"] = list(nominal["hole_xy"][part])
    Hn, Rwn = S.entry_pose(part, nominal)
    tcp = Hn + Rwn @ np.asarray(ep["hand"]["offset"])
    q_hand = S.mat_quat(Rwn @ S.quat_mat(ep["hand"]["R_rel_quat"]))
    R_in = S.rx(ih["rock"]) @ S.ry(ih["pivot"])
    m, info = build_model(part, bp, rp, tcp_pose=(tcp, q_hand), R_inhand=R_in)
    m.opt.solver = mujoco.mjtSolver.mjSOL_CG if ph["solver"] == "cg" else mujoco.mjtSolver.mjSOL_NEWTON
    # The fingers never collide with each other (before t = 0). In menagerie's hand the closed finger meshes overlap,
    # so an empty close drove them 8 mm into each other and past their joint stops; a real empty Franka grasp stops at
    # width 0. contype 2 / conaffinity 1: finger-finger (2&1, 2&1) never collide; finger-part, finger-board and
    # finger-arm (1&1 one way) still do.
    fingers = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f) for f in ("gripper/left_finger", "gripper/right_finger")}
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in fingers and m.geom_contype[g]:
            m.geom_contype[g] = 2
            m.geom_conaffinity[g] = 1
    q7, ik = S.solve_ik(m, ep.get("q_start") or START_ARM_QPOS, tcp, q_hand)
    if not ik["ok"]:
        raise EnvError(f"start {ep['id']}: no IK solution for the start TCP pose ({ik})")
    d = mujoco.MjData(m)
    ctrl = CartesianImpedance(m, impedance(rp))
    d.qpos[ctrl.arm.qadr] = q7
    mujoco.mj_forward(m, d)
    slug = info["part"]["slug"]
    close_fingers_on_part(m, d, slug, dy=rp.grasp_dy)
    mujoco.mj_forward(m, d)
    grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper/actuator8")
    ctrl.reset(d)
    n_sub = max(1, int(round(ctrl.p.period / m.opt.timestep)))
    step = 0
    while d.time < PREROLL_S - 1e-9:
        if step % n_sub == 0:
            d.ctrl[:7] = ctrl.torque(d)
            d.ctrl[grip] = GRIP_CLOSED
        mujoco.mj_step(m, d)
        step += 1
    return m, d, ctrl, info, dict(rp=rp, grip=grip, n_sub=n_sub, slug=slug, sensor=ToolWrench(m),
                                  GRIP_CLOSED=GRIP_CLOSED, ik=ik, tcp_start=tcp, q_hand_start=q_hand,
                                  finger_qadr=tuple(int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)])
                                                    for j in ("gripper/finger_joint1", "gripper/finger_joint2")))


def check_start(start: dict, nominal: dict, hold_s: float = 0.5) -> dict:
    """Is a drawn start physically valid in the nominal twin? The grasp must survive the pre-roll and a further
    hold_s with the arm holding still: both pads on the part, tip above the true surface. Used when freezing starts."""
    import mujoco
    from twin import cousins as C
    from twin.evaluator import Evaluator
    ep = C.apply(start, C.nominal())
    try:
        m, d, ctrl, info, aux = build_episode(ep, nominal)
    except EnvError as ex:
        return dict(ok=False, why=str(ex))
    ev = Evaluator(m, aux["slug"])
    st0 = ev.state(d)
    step = 0
    while d.time < PREROLL_S + hold_s - 1e-9:
        if step % aux["n_sub"] == 0:
            d.ctrl[:7] = ctrl.torque(d)
            d.ctrl[aux["grip"]] = aux["GRIP_CLOSED"]
        mujoco.mj_step(m, d)
        step += 1
    st = ev.state(d)
    ok = bool(st0["held"] and st["held"] and st["depth"] < -0.001 and np.isfinite(d.qpos).all())
    return dict(ok=ok, why=None if ok else f"grasp not stable: held {st0['held']}->{st['held']}, tip depth "
                f"{1000 * st['depth']:.1f} mm", tip_depth_mm=round(1000 * st["depth"], 2),
                tilt_deg=round(st["tilt_deg"], 2), lateral_mm=round(1000 * st["lateral"], 2))


class PracticeEnv:
    """The simulator side of the boundary. One instance = one session (many episodes)."""

    def __init__(self, brain: str = "claude", run_id: str = "run-standin-1", start_set: str = "practice",
                 cousin: str = "c000", budget: dict | None = None, actor: str = "brain", session: str | None = None,
                 privileged: bool = False, dt_half: bool | None = None, record: bool = True,
                 run_dir: Path | None = None, timestep_scale: float = 1.0, physics_override: dict | None = None):
        from twin import starts as S
        from twin.monitors import Budget
        self.brain, self.run_id = brain, run_id
        self.run_dir = Path(run_dir) if run_dir else REPO_DIR / "runs" / brain / run_id
        self.start_set = start_set
        self.starts, self.nominal = S.load(start_set, self.run_dir / "starts")
        if privileged and start_set != "redteam":
            raise PermissionError("privileged mode is allowed on the redteam start set only")
        self.privileged = privileged
        self.cousin_id = cousin
        self.cousin = self._load_cousin(cousin)
        self.budget = Budget(**(budget or {}))
        self.actor = actor
        self.session = session or time.strftime("s%Y%m%d-%H%M%S") + f"-{os.getpid()}"
        # dt/2 rerun: on for every normal env. Privileged envs skip it unless asked (dt_half=True): their hooks are
        # arbitrary code that a command-log rerun cannot replay; self-tests that plant a physics quirk (no hooks) ask.
        self.dt_half_enabled = (not privileged) if dt_half is None else bool(dt_half)
        if physics_override and not privileged:
            raise PermissionError("physics overrides are for privileged self-tests only")
        self.physics_override = dict(physics_override or {})
        self.record = record
        self.timestep_scale = timestep_scale
        self.ep = None              # current episode state (dict) or None
        self.n_episodes = 0
        self.hooks = []             # privileged pre-step hooks fn(m, d, step)
        self.out_dir = self.run_dir / "env" / "sessions" / self.session
        self.envelope = self._load_envelope()

    # ------------------------------------------------------------------------------------------- setup
    def _load_cousin(self, cid: str) -> dict:
        from twin import cousins as C
        if cid in (None, "c000"):
            return C.nominal()
        return C.load(self.run_dir / "cousins")["by_id"][cid]

    def _load_envelope(self) -> dict | None:
        try:
            from twin.envelope import load
            return load(self.run_dir / "envelope" / "envelope.json")
        except FileNotFoundError:
            return None

    def task(self, start: dict | None = None) -> dict:
        """What the actor may know: nominal geometry (the believed board), never the start's offsets."""
        from twin import starts as S
        from twin.parts import load_layout, part_spec
        from twin.replay import PAD_TO_TCP
        start = start or (self.ep or {}).get("start")
        out = dict(
            conventions=dict(
                units="SI (m, s, N, rad); quaternions w-first", world="FR3 base frame (link0), z up, table top z = 0",
                tcp="site 'tcp' = REASSEMBLE logged O_T_EE; it lies 22.9 mm beyond the finger-pad centre along the "
                    "hand z axis (hand z points out of the fingers)",
                force="force on the tool from everything it touches, base frame (REASSEMBLE force.base sign: pressing "
                      "down on the board reads +z); torque about the TCP; includes the held part's weight",
                hole_frame="entry pose: origin at the hole centre on the plate surface, +z INTO the hole, +y along "
                           "the hole's long side (square pegs: the key's long side a)",
                grasp="the part is held between the pads, closing axis = hand y; its pose in the fingers is NOT "
                      "measured: real grasps put it several mm along the pads (hand x) and a few degrees tilted"),
            limits=dict(speed_cap_m_s=SPEED_CAP, ang_speed_cap_rad_s=ANG_SPEED_CAP, force_cap_N=FORCE_CAP,
                        torque_ceiling="force_max x 0.05 m", joint_range_rad=JOINT_RANGE,
                        guarded_move=dict(note="with stop_force the approach speed is capped so the stop is honoured; "
                                               "the first touch is a force step within one 1 ms tick",
                                          speed_cap_m_s_by_stop_force=[list(r) for r in GUARD_SPEED_TABLE],
                                          peak_tolerance_N="max(0.5, 0.2 x stop_force)"),
                        default_speed_m_s=DEFAULT_SPEED, default_force_max_N=DEFAULT_FORCE, move_max_s=MOVE_MAX_S,
                        wait_max_s=WAIT_MAX_S, grip_force_N=GRIP_FORCE, workspace=WORKSPACE),
            budget=self.budget.as_dict(), start_set=self.start_set, cousin=self.cousin_id)
        if start is not None:
            part = start["part"]
            ps = part_spec(part, load_layout())
            H, Rw = S.entry_pose(part, self.nominal)
            sx, sy = ps.section
            cx, cy = ps.radial_clearance
            g = S.NOMINAL_GRASP_FROM_TOP
            out.update(
                start_id=start["id"], part=part, family=ps.family,
                part_geometry=dict(section_mm=[round(1000 * sx, 3), round(1000 * sy, 3)],
                                   length_mm=round(1000 * ps.length, 3),
                                   chamfer_mm=round(1000 * ps.v("chamfer_frac") * max(sx, sy), 3)),
                hole=dict(entry_pos=_r(H), entry_quat=_r(S.mat_quat(Rw)),
                          yaw_symmetry_deg=0 if ps.family == "round_peg" else 180,
                          yaw_note="0 = any hand yaw about the hole axis fits (round); 180 = the key fits at the "
                                   "entry yaw or 180 deg from it. Joint 7 has a +-3.05 rad range: pick the option "
                                   "nearest the current hand yaw",
                          clearance_per_side_mm=[round(1000 * cx, 4), round(1000 * cy, 4)],
                          seat_depth_mm=round(1000 * self.nominal["seat_depth"], 2),
                          source="nominal board (contact fit of a train recording); the true board differs by the "
                                 "calibration error (about 1-4 mm)"),
                grasp_nominal=dict(grasp_from_top_mm=round(1000 * g, 2), pad_to_tcp_mm=round(1000 * PAD_TO_TCP, 2),
                                   tip_in_tcp_frame_m=_r([0.0, 0.0, ps.length - g - PAD_TO_TCP]),
                                   note="nominal: part axis = hand z through the pad centre; the true offset is "
                                        "unknown"))
        return out

    def _episode_physics(self, start: dict) -> dict:
        from twin import cousins as C
        ep = C.apply(start, self.cousin)
        ph = dict(ep["physics"])
        ph.update(self.physics_override)
        ph["timestep"] = ph["timestep"] * self.timestep_scale
        ep["physics"] = ph
        return ep

    def _build(self, ep: dict):
        return build_episode(ep, self.nominal)

    # ------------------------------------------------------------------------------------------- episodes
    def reset(self, start_id: str, start_override: dict | None = None) -> dict:
        from twin.envelope import family_of
        from twin.evaluator import Evaluator
        from twin.monitors import MonitorSuite
        if self.ep is not None and not self.ep.get("done"):
            self._end_episode("abandoned")
        if start_override is not None:
            if not self.privileged:
                raise PermissionError("ad-hoc starts are allowed in privileged (red-team / self-test) mode only")
            start = dict(start_override)
        else:
            start = self.starts.get(start_id)
            if start is None:
                raise EnvError(f"unknown start {start_id!r} in the {self.start_set} set")
        self._claim_attempt(start["id"])
        ep = self._episode_physics(start)
        t0w = time.perf_counter()
        m, d, ctrl, info, aux = self._build(ep)
        ev = Evaluator(m, aux["slug"])
        st0 = ev.update(d)
        fam = family_of(start["part"])
        flim = None
        if self.envelope and fam in self.envelope["envelope"]:
            flim = self.envelope["envelope"][fam]["peak_force_any_max_N"]
        mon = MonitorSuite(m, d, aux["slug"], force_limit_n=flim, budget=self.budget,
                           contact_tc=ep["physics"]["solref_tc"], contact_width=ep["physics"]["solimp_width"])
        x, q, _ = ctrl.arm.pose(d)
        self.n_episodes += 1
        self.ep = dict(start=start, override=start_override is not None, ep=ep, m=m, d=d, ctrl=ctrl, info=info, aux=aux, ev=ev, mon=mon,
                       x_i=x.copy(), q_i=q.copy(), x_f=x.copy(), q_f=q.copy(), force_max=DEFAULT_FORCE,
                       grip_cmd=aux["GRIP_CLOSED"], commands=[], traj=[], next_rec=0.0, done=False, nan=False,
                       t_build_s=round(time.perf_counter() - t0w, 3), start_state=st0, wall0=time.time(),
                       index=self.n_episodes, claim=None, ended_by=None, last_F=np.zeros(3), last_T=np.zeros(3))
        if not (st0["held"] and st0["depth"] < 0):
            self._end_episode("error")
            raise EnvError(f"start {start['id']}: the grasp failed in the pre-roll (held={st0['held']}, "
                           f"tip depth {1000 * st0['depth']:.1f} mm)")
        self._ctrl_tick_prepare()
        mon.arm()
        self._record(force=True)
        return dict(task=self.task(start), obs=self.observe())

    def _claim_attempt(self, start_id: str):
        """Held-out starts: one attempt per (start, cousin, actor) per run."""
        if self.start_set != "heldout" or not self.record:
            return
        f = self.run_dir / "env" / "heldout_attempts.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        key = [start_id, self.cousin_id, self.actor]
        if f.exists():
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if [r["start"], r["cousin"], r["actor"]] == key:
                    raise PermissionError(f"held-out start {start_id} (cousin {self.cousin_id}) was already attempted by "
                                          f"{self.actor} in this run (session {r['session']}); held-out starts are "
                                          "attempted once")
        with open(f, "a") as fh:
            fh.write(json.dumps(dict(start=start_id, cousin=self.cousin_id, actor=self.actor, session=self.session,
                                     t=time.time())) + "\n")

    def _need_ep(self):
        if self.ep is None:
            raise EnvError("no episode: reset to a start first")
        if self.ep["done"]:
            raise EnvError("the episode is over; reset to the next start")

    # ------------------------------------------------------------------------------------------- stepping
    def _ctrl_tick_prepare(self):
        e = self.ep
        e["mon"].expect_ctrl(e["d"].ctrl)

    def _tick(self):
        """One 1 kHz controller tick (zero-order hold over n_sub physics steps), monitors around every mj_step."""
        import mujoco
        from twin.control import quat_conj, quat_mul, slerp
        e = self.ep
        m, d, ctrl, mon = e["m"], e["d"], e["ctrl"], e["mon"]
        x, q, _ = ctrl.arm.pose(d)
        e["x_f"] = FILTER_ALPHA * e["x_i"] + (1 - FILTER_ALPHA) * e["x_f"]
        e["q_f"] = slerp(e["q_f"], e["q_i"], FILTER_ALPHA)
        # force ceiling: the translational spring may pull at most force_max ...
        err = e["x_f"] - x
        lim = e["force_max"] / ctrl.p.k_trans
        n = float(np.linalg.norm(err))
        if n > lim:
            err = err * (lim / n)
        # ... and the rotational spring at most force_max * TORQUE_ARM (a wedged part turns hand torque into force)
        qc = e["q_f"]
        dq = quat_mul(quat_conj(q if float(q @ qc) >= 0 else -q), qc)
        sin_half = float(np.linalg.norm(dq[1:]))
        tmax = e["force_max"] * TORQUE_ARM
        if ctrl.p.k_rot * sin_half > tmax and sin_half > 1e-12:
            qc = slerp(q, qc, (tmax / ctrl.p.k_rot) / sin_half)
        ctrl.set_target(x + err, qc, direct=True)
        tau = ctrl.torque(d)
        d.ctrl[:7] = tau
        if e["grip_cmd"] == e["aux"]["GRIP_CLOSED"] and self._grip_width() < GRIP_EMPTY_W:
            e["grip_cmd"] = 0.0   # closed on nothing: hold the closed position like a failed Franka grasp
        d.ctrl[e["aux"]["grip"]] = e["grip_cmd"]
        mon.expect_ctrl(d.ctrl)
        for _ in range(e["aux"]["n_sub"]):
            for h in self.hooks:
                h(m, d, mon.step)
            mon.pre_step()
            mujoco.mj_step(m, d)
            mon.post_step()
            if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0 or not np.isfinite(d.qpos).all():
                e["nan"] = True
                return
        F, T = e["aux"]["sensor"].wrench(d)
        p, _, _ = ctrl.arm.pose(d)
        e["last_F"], e["last_T"] = F, T - np.cross(p, F)
        self._record()

    def _t(self) -> float:
        e = self.ep
        return float(e["d"].time - e["mon"].t0) if e["mon"].armed else 0.0

    def _record(self, force: bool = False):
        e = self.ep
        t = self._t()
        if not force and t + 1e-9 < e["next_rec"]:
            return
        x, q, _ = e["ctrl"].arm.pose(e["d"])
        e["traj"].append([round(t, 4), *_r(x), *_r(q), *_r(e["last_F"], 4), round(self._grip_width(), 6)])
        e["next_rec"] = (len(e["traj"])) / TRAJ_HZ

    def _grip_width(self) -> float:
        e = self.ep
        j1, j2 = e["aux"]["finger_qadr"]
        return float(e["d"].qpos[j1] + e["d"].qpos[j2])

    def _run(self, duration: float, stop=None) -> dict:
        """Tick until `duration` sim s pass or stop() says so. Returns {stopped, duration_s, max_force_N}."""
        e = self.ep
        t0 = self._t()
        n = 0
        fmax = 0.0
        while self._t() - t0 < duration - 1e-9:
            self._tick()
            fmax = max(fmax, float(np.linalg.norm(e["last_F"])))
            if e["nan"]:
                return dict(stopped="nan", duration_s=round(self._t() - t0, 4), max_force_N=round(fmax, 3))
            n += 1
            if stop is not None:
                why = stop()
                if why:
                    return dict(stopped=why, duration_s=round(self._t() - t0, 4), max_force_N=round(fmax, 3))
        return dict(stopped=None, duration_s=round(self._t() - t0, 4), max_force_N=round(fmax, 3))

    # ------------------------------------------------------------------------------------------- commands
    def command(self, req: dict, allowed=ACTOR_OPS) -> dict:
        """Dispatch one request (dict with 'op'). Errors come back as {'error': ...}, never as exceptions."""
        t0 = time.perf_counter()
        try:
            if not isinstance(req, dict):
                raise EnvError("request must be a JSON object")
            op = req.get("op")
            if op not in allowed:
                raise EnvError(f"op {op!r} not allowed here; allowed: {list(allowed)}")
            args = {k: v for k, v in req.items() if k != "op"}
            fn = {"task": self._op_task, "move": self.move, "gripper": self.gripper, "wait": self.wait,
                  "observe": self._op_observe, "status": self._op_status, "finish": self.finish,
                  "reset": self._op_reset, "charge_tokens": self.charge_tokens, "ping": lambda: dict(pong=True),
                  "close": self._op_close, "render": self.render, "end_limit": self.end_limit}[op]
            out = fn(**args)
        except TypeError as ex:
            out = dict(error=f"bad arguments for {req.get('op')!r}: {ex}")
        except (EnvError, PermissionError, KeyError) as ex:
            out = dict(error=f"{type(ex).__name__}: {ex}")
        except Exception as ex:  # noqa: BLE001  (a bug must not kill the session)
            out = dict(error=f"{type(ex).__name__}: {ex}", trace=traceback.format_exc(limit=3))
        if self.ep is not None and req.get("op") in ("move", "gripper", "wait") and "error" not in out:
            self._after_command(req, out, time.perf_counter() - t0)
        return out

    def _op_task(self):
        return self.task()

    def _op_observe(self):
        self._need_ep()
        return self.observe()

    def _op_status(self):
        return self.status()

    def _op_reset(self, start_id: str):
        return self.reset(start_id)

    def _op_close(self):
        if self.ep is not None and not self.ep["done"]:
            self._end_episode("abandoned")
        return dict(closed=True, episodes=self.n_episodes)

    def move(self, pos, quat=None, speed=None, force_max=None, stop_force=None, timeout=None) -> dict:
        from twin.control import quat_mul, quat_conj
        self._need_ep()
        e = self.ep
        p = _vec(pos, 3, "pos")
        for ax, k in zip(p, "xyz"):
            lo, hi = WORKSPACE[k]
            if not lo <= ax <= hi:
                raise EnvError(f"pos.{k} {ax:.4f} outside the workspace {WORKSPACE[k]}")
        q = e["q_i"] if quat is None else _vec(quat, 4, "quat")
        nq = float(np.linalg.norm(q))
        if nq < 0.5:
            raise EnvError("quat must be a unit quaternion (w, x, y, z)")
        q = q / nq
        if float(q @ e["q_i"]) < 0:
            q = -q
        v = min(_f(speed if speed is not None else DEFAULT_SPEED, "speed", 1e-4), SPEED_CAP)
        fm = min(_f(force_max if force_max is not None else DEFAULT_FORCE, "force_max", 0.5), FORCE_CAP)
        sf = None if stop_force is None else min(_f(stop_force, "stop_force", 0.5), FORCE_CAP)
        v_req = v
        if sf is not None:
            v = min(v, guard_speed(sf))     # a guarded approach slows down so its stop can be honoured
        x0, q0 = e["x_i"].copy(), e["q_i"].copy()
        dist = float(np.linalg.norm(p - x0))
        dq = quat_mul(quat_conj(q0), q)
        ang = 2 * math.atan2(float(np.linalg.norm(dq[1:])), abs(float(dq[0])))
        T = max(dist / v, ang / min(DEFAULT_ANG_SPEED, ANG_SPEED_CAP) if ang > 0 else 0.0)
        to = _f(timeout, "timeout", 0.01, MOVE_MAX_S) if timeout is not None else min(MOVE_MAX_S, T + 2.0)
        e["force_max"] = fm
        t_start = self._t()
        hit = [0]
        from twin.control import slerp

        def stop():
            s = min(1.0, (self._t() - t_start) / T) if T > 0 else 1.0
            e["x_i"] = x0 + s * (p - x0)
            e["q_i"] = slerp(q0, q, s)
            if sf is not None:
                hit[0] = hit[0] + 1 if float(np.linalg.norm(e["last_F"])) >= sf else 0
                if hit[0] >= STOP_TICKS:
                    x, qq, _ = e["ctrl"].arm.pose(e["d"])
                    e["x_i"], e["q_i"] = x.copy(), qq.copy()   # stop: hold where the hand is
                    e["x_f"], e["q_f"] = x.copy(), qq.copy()
                    return "stop_force"
            if s >= 1.0:
                x, _, _ = e["ctrl"].arm.pose(e["d"])
                if float(np.linalg.norm(x - p)) < 5e-4 or self._t() - t_start >= T + 1.0:
                    return "reached" if float(np.linalg.norm(x - p)) < 5e-4 else "settled"
            return None

        r = self._run(to, stop)
        if r["stopped"] is None:
            r["stopped"] = "timeout"
        if r["stopped"] == "stop_force":            # hold briefly so the reported peak includes the overshoot
            r2 = self._run(STOP_SETTLE_S)
            r["max_force_N"] = max(r["max_force_N"], r2["max_force_N"])
            r["duration_s"] = round(r["duration_s"] + r2["duration_s"], 4)
        x, _, _ = e["ctrl"].arm.pose(e["d"])
        r.update(target_error_mm=round(1000 * float(np.linalg.norm(x - p)), 3), speed_used=v, force_max_used=fm)
        if sf is not None:
            r.update(stop_force=sf, guard_speed_cap=round(guard_speed(sf), 5), speed_capped=bool(v < v_req),
                     stop_tolerance_N=round(GUARD_TOL(sf), 3))
            if r["stopped"] == "stop_force":
                r["stop_overshoot_N"] = round(max(0.0, r["max_force_N"] - sf), 3)
        return dict(result=r, obs=self.observe())

    def gripper(self, action: str, width=None) -> dict:
        self._need_ep()
        e = self.ep
        if action == "open":
            e["grip_cmd"] = 0.04
        elif action == "close":
            e["grip_cmd"] = e["aux"]["GRIP_CLOSED"]
        elif action == "width":
            e["grip_cmd"] = _f(width, "width", 0.0, 0.08) / 2
        else:
            raise EnvError("action must be open | close | width")
        last = [self._grip_width(), 0]

        def stop():
            w = self._grip_width()
            last[1] = last[1] + 1 if abs(w - last[0]) < 2e-6 else 0
            last[0] = w
            return "settled" if last[1] >= 50 else None

        r = self._run(GRIP_S, stop)
        r["stopped"] = r["stopped"] or "timeout"
        r["width_m"] = round(self._grip_width(), 6)
        return dict(result=r, obs=self.observe())

    def wait(self, seconds) -> dict:
        self._need_ep()
        s = _f(seconds, "seconds", 0.0, WAIT_MAX_S)
        r = self._run(s)
        r["stopped"] = r["stopped"] or "done"
        return dict(result=r, obs=self.observe())

    def observe(self) -> dict:
        e = self.ep
        if e is None:
            return dict(t=0.0, note="no episode")
        ctrl, d = e["ctrl"], e["d"]
        x, q, _ = ctrl.arm.pose(d)
        J = ctrl.arm.jac(d)
        v = J @ d.qvel[ctrl.arm.dadr]
        return dict(t=round(self._t(), 4), tcp=dict(pos=_r(x), quat=_r(q)), tcp_twist=_r(v, 5),
                    q=_r(d.qpos[ctrl.arm.qadr]), dq=_r(d.qvel[ctrl.arm.dadr], 5),
                    gripper_width_m=round(self._grip_width(), 6),
                    wrench=dict(force_N=_r(e["last_F"], 4), torque_Nm=_r(e["last_T"], 5)),
                    episode_over=bool(e["done"]))

    def status(self) -> dict:
        e = self.ep
        out = dict(session=self.session, start_set=self.start_set, cousin=self.cousin_id, episodes=self.n_episodes,
                   budget=self.budget.as_dict())
        if e is not None:
            mon = e["mon"]
            out.update(start_id=e["start"]["id"], part=e["start"]["part"], t=round(self._t(), 3),
                       commands_used=mon.n_commands, tokens_used=mon.tokens, wall_s=round(time.time() - e["wall0"], 1),
                       episode_over=e["done"], ended_by=e["ended_by"])
        return out

    def end_limit(self, what: str, used=None, limit=None, note: str = "") -> dict:
        """Harness only (brain runner): the brain's own budget ran out. Ends the episode as limit_reached, which is
        reported apart from failures (validation.yaml D.limit_hit_rate). Never a success, never invalidating."""
        self._need_ep()
        if what not in LIMIT_WHAT:
            raise EnvError(f"what must be one of {list(LIMIT_WHAT)}")
        detail = f"{what}: {used} used of {limit}" + (f" ({str(note)[:200]})" if note else "")
        self.ep["mon"].flag("budget.exhausted", "budget", f"brain budget: {detail}", invalidates=False)
        self.ep["limit"] = dict(what=what, used=used, limit=limit, note=str(note)[:200])
        info = self._end_episode("limit_reached")
        return dict(episode_over=True, ended_by="limit_reached", limit=self.ep["limit"], record=info.get("file"),
                    report=self.ep.get("report_actor"))

    def render(self, out: str, views=None, zoom: float = 3.0, overlays: bool = True, title: str = "") -> dict:
        """Harness only (env_observe_image): render the LIVE state from the calibrated cameras with the belief
        overlays of twin/practice_view.py and save a PNG. Returns the image path and the belief summary, which is
        derived from task() and observe() (what the actor already has). Never returns raw state; never steps or
        touches the physics model or data (it renders a posed copy)."""
        from twin import practice_view as PV
        if self.ep is None:
            raise EnvError("no episode: reset to a start first")
        t0 = time.perf_counter()
        views = list(views or PV.VIEWS)
        if not views or any(v not in PV.VIEWS for v in views):
            raise EnvError(f"views must be a non-empty subset of {list(PV.VIEWS)}")
        zoom = _f(zoom, "zoom", 1.0, 6.0)
        outp = Path(out).resolve()
        if not outp.is_relative_to(self.run_dir.resolve()) or outp.suffix.lower() != ".png":
            raise PermissionError(f"render writes PNGs inside the run directory only ({self.run_dir})")
        e = self.ep
        if getattr(self, "_rnd_model", None) is not e["m"]:
            if getattr(self, "_rnd", None) is not None:
                self._rnd.close()
            self._rnd = PV.SceneRenderer(e["m"], PV.load_cams(), PV.err_seed(self.run_id))
            self._rnd_model = e["m"]
        self._rnd.pose(e["d"])
        obs = self.observe()
        img, b = PV.compose(self._rnd, self.task(), obs, views, zoom, title=str(title)[:200], overlays=bool(overlays))
        outp.parent.mkdir(parents=True, exist_ok=True)
        img.save(outp)
        return dict(image=_relpath(outp), views=views, t=obs["t"], episode_over=bool(e["done"]),
                    belief=dict(t=round(b["t"], 3), tip_vs_hole_mm=[round(float(v), 2) for v in b["tip_hole_mm"]],
                                tilt_deg=round(b["tilt_deg"], 2)),
                    sync=dict(match=True, live=True), render_s=round(time.perf_counter() - t0, 3))

    def charge_tokens(self, tokens) -> dict:
        self._need_ep()
        n = int(_f(tokens, "tokens", 0))
        stop = self.ep["mon"].charge_tokens(n)
        if stop:
            self._end_episode(stop)
        return dict(tokens_used=self.ep["mon"].tokens, episode_over=self.ep["done"])

    def _after_command(self, req: dict, out: dict, wall_s: float):
        e = self.ep
        st = e["ev"].update(e["d"])
        e["commands"].append(dict(req=req, t=round(self._t(), 4), result=out.get("result")))
        if e["nan"]:
            self._end_episode("error")
        else:
            stop = e["mon"].on_command(req, st["stage"], st["depth"], time.time() - e["wall0"])
            if stop:
                self._end_episode(stop)
        if e["done"]:
            out["episode_over"] = True
            out["ended_by"] = e["ended_by"]
            out["report"] = e.get("report_actor")

    # ------------------------------------------------------------------------------------------- finish
    def finish(self, claim=None) -> dict:
        self._need_ep()
        if claim not in (None, "success", "failure"):
            raise EnvError("claim must be 'success', 'failure' or null")
        self.ep["claim"] = claim
        rec = self._end_episode(None)
        return dict(episode_over=True, report=self.ep.get("report_actor"), record=rec.get("file"))

    def _end_episode(self, ended_by: str | None) -> dict:
        from twin.envelope import check_episode, features
        from twin.evaluator import EVALUATOR_VERSION, SETTLE_S, code_hash, counted
        e = self.ep
        if e["done"]:
            return e.get("record_info", {})
        e["ended_by"] = ended_by
        if ended_by is None and not e["nan"]:
            self._run(SETTLE_S)                     # hold the last command, let the part settle
        oc = e["ev"].outcome(e["d"], ended_by=ended_by, nan=e["nan"])
        e["mon"].end()
        tr = np.array(e["traj"], float) if e["traj"] else np.zeros((0, 12))
        feat = features(tr[:, 0], tr[:, 1:4], tr[:, 4:8], tr[:, 8:11], tr[:, 11]) if len(tr) >= 4 else {}
        viol = check_episode(feat, e["start"]["part"], self.envelope) if (self.envelope and feat) else None
        dt_half = None
        if oc["outcome"] == "success" and self.dt_half_enabled:
            dt_half = self._rerun_half()
            if dt_half.get("outcome") != "success":
                e["mon"].flag("numerics.dt_half", "numerics", f"success flips at dt/2: rerun outcome "
                              f"{dt_half.get('outcome')!r}", invalidates=True)
        flags = e["mon"].flag_list()
        final, cnt, why = counted(oc["outcome"], flags, dt_half)
        if self.privileged:
            final, cnt = f"privileged:{final}", False
            why = why + ["privileged red-team run: never counted"]
        e["done"] = True
        rec = dict(
            session=self.session, episode=e["index"], brain=self.brain, run_id=self.run_id, actor=self.actor,
            start_id=e["start"]["id"], start_set=self.start_set, part=e["start"]["part"], cousin=self.cousin_id,
            privileged=self.privileged, outcome=final, raw_outcome=oc["outcome"], counted=cnt, why_not_counted=why,
            claim=e["claim"], false_claim=bool(e["claim"] == "success" and oc["outcome"] != "success"),
            ended_by=ended_by, limit=e.get("limit"), evaluation=oc, monitors=e["mon"].summary(), dt_half=dt_half,
            envelope=dict(features=feat, violations=viol), n_commands=len(e["commands"]),
            sim_s=round(self._t(), 3), wall_s=round(time.time() - e["wall0"], 2), build_s=e["t_build_s"],
            physics=e["ep"]["physics"], start=e["start"], episode_geometry=dict(board_err=e["ep"]["board_err"],
                                                                                 inhand=e["ep"]["inhand"]),
            ik=e["aux"]["ik"], evaluator=dict(version=EVALUATOR_VERSION, code_hash=code_hash()),
            replay_params=e["aux"]["rp"].as_dict(),
            commands=e["commands"],
            trajectory=dict(cols="t, tcp x y z, quat w x y z, F x y z, gripper width", hz=TRAJ_HZ, rows=e["traj"]))
        info = {}
        if self.record:
            d = self.out_dir / "episodes"
            d.mkdir(parents=True, exist_ok=True)
            f = d / f"{e['index']:03d}_{e['start']['id']}_{self.cousin_id}.json"
            f.write_text(json.dumps(rec, default=float))
            info["file"] = _relpath(f)
            with open(self.run_dir / "env" / "episodes.jsonl", "a") as fh:
                fh.write(json.dumps({k: rec[k] for k in ("session", "episode", "actor", "start_id", "start_set", "part",
                                                         "cousin", "outcome", "counted", "claim", "ended_by",
                                                         "n_commands", "sim_s", "wall_s", "privileged")}
                                    | dict(n_flags=len(flags), file=info["file"])) + "\n")
        e["record"] = rec
        e["record_info"] = info
        e["report_actor"] = self._actor_report(rec)
        return info

    def _actor_report(self, rec: dict) -> dict:
        """What the actor may see after an episode: everything on practice starts, nothing on held-out starts."""
        if self.start_set == "heldout":
            return dict(recorded=True, note="held-out outcome goes to the evaluator log only")
        ev = rec["evaluation"]
        return dict(outcome=rec["outcome"], counted=rec["counted"], max_stage=ev["max_stage"],
                    final=ev["final"], max_depth_mm=ev["max_depth_mm"],
                    flags=[f"{f['monitor']}: {f['reason']}" for f in rec["monitors"]["flags"]],
                    envelope_violations=[v["reason"] for v in (rec["envelope"]["violations"] or [])])

    def _rerun_half(self) -> dict:
        """Replay this episode's command log open-loop at half the timestep (fresh env, same start and cousin)."""
        e = self.ep
        try:
            sub = PracticeEnv(self.brain, self.run_id, self.start_set, self.cousin_id, budget=self.budget.as_dict(),
                              actor=self.actor + ":dt_half", session=self.session + "-dt2", dt_half=False,
                              record=False, run_dir=self.run_dir, timestep_scale=self.timestep_scale / 2,
                              privileged=self.privileged, physics_override=self.physics_override)
            sub.reset(e["start"]["id"], start_override=e["start"] if e.get("override") else None)
            for c in e["commands"]:
                r = sub.command(c["req"])
                if r.get("episode_over") or sub.ep["done"]:
                    break
            if not sub.ep["done"]:
                sub._end_episode(None)
            rr = sub.ep["record"]
            return dict(outcome=rr["raw_outcome"], timestep=rr["physics"]["timestep"],
                        final=rr["evaluation"]["final"], n_flags=len(rr["monitors"]["flags"]))
        except Exception as ex:  # noqa: BLE001
            return dict(outcome="rerun_error", error=f"{type(ex).__name__}: {ex}")


# ================================================================================================ servers
def serve(sock_path: str, cfg: dict, idle_s: float = 1800.0):
    """Session daemon for brain tool calls: one JSON request per connection on a Unix socket."""
    env = PracticeEnv(**cfg)
    p = Path(sock_path)
    if p.exists():
        p.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(p))
    os.chmod(p, 0o600)
    srv.listen(4)
    srv.settimeout(idle_s)
    (p.parent / "pid").write_text(str(os.getpid()))
    try:
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                if env.ep is not None and not env.ep["done"]:
                    env._end_episode("abandoned")
                break
            with conn:
                conn.settimeout(60)
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                    if len(buf) > 1_000_000:
                        break
                try:
                    req = json.loads(buf.decode() or "{}")
                except json.JSONDecodeError as ex:
                    req, rep = None, dict(error=f"bad JSON: {ex}")
                if req is not None:
                    rep = env.command(req, allowed=HARNESS_OPS)
                conn.sendall((json.dumps(rep, default=float) + "\n").encode())
                if req is not None and req.get("op") == "close":
                    break
    finally:
        srv.close()
        p.unlink(missing_ok=True)


def call_socket(sock_path: str, req: dict, timeout: float = 600.0) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(str(sock_path))
    with s:
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode())


def run_program(env: PracticeEnv, start_id: str, program, wall_s: float = 600.0, cpu_s: int = 600,
                stderr_path: Path | None = None) -> dict:
    """Run a skill program (file or workspace dir with skill.py) in the sandbox on one start; pump JSON between it and
    the env. Returns the episode record summary plus the sandbox mode."""
    from twin import sandbox as SB
    rep0 = env.reset(start_id)
    tmp, entry = SB.prepare(program)
    proc, mode = SB.popen(tmp, entry, cpu_s=cpu_s, stderr_path=stderr_path)
    t0 = time.time()
    n_req = 0
    why_end = None
    try:
        while True:
            left = wall_s - (time.time() - t0)
            if left <= 0:
                why_end = "wall_timeout"
                break
            r, _, _ = select.select([proc.stdout], [], [], min(left, 5.0))
            if not r:
                if proc.poll() is not None:
                    why_end = f"exit {proc.returncode}"
                    break
                continue
            line = proc.stdout.readline()
            if not line:
                why_end = f"exit {proc.wait()}"
                break
            n_req += 1
            if len(line) > 65536:
                rep = dict(error="request too long")
            else:
                try:
                    req = json.loads(line)
                    rep = env.command(req, allowed=ACTOR_OPS)
                except json.JSONDecodeError as ex:
                    rep = dict(error=f"bad JSON: {ex}")
            try:
                proc.stdin.write((json.dumps(rep, default=float) + "\n").encode())
                proc.stdin.flush()
            except BrokenPipeError:
                why_end = "program closed its input"
                break
            if env.ep is not None and env.ep["done"]:
                why_end = "episode over"
                break
    finally:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        proc.wait()
        SB.cleanup(tmp)
    if env.ep is not None and not env.ep["done"]:
        env._end_episode(None)  # the program ended without finish: score what it left
    rec = env.ep["record"]
    return dict(start_id=start_id, outcome=rec["outcome"], counted=rec["counted"], raw_outcome=rec["raw_outcome"],
                flags=rec["monitors"]["flags"], max_stage=rec["evaluation"]["max_stage"], claim=rec["claim"],
                false_claim=rec["false_claim"], n_commands=rec["n_commands"], sim_s=rec["sim_s"],
                wall_s=round(time.time() - t0, 2), sandbox=mode, program_end=why_end, requests=n_req,
                envelope=rec["envelope"], dt_half=rec["dt_half"], file=env.ep["record_info"].get("file"),
                final=rec["evaluation"]["final"], task_part=rep0["task"].get("part"))


class RawAccess:
    """Privileged handle for red-team scripts (run in-process): the live MjModel/MjData and pre-step hooks."""

    def __init__(self, env: PracticeEnv):
        self._env = env

    @property
    def m(self):
        return self._env.ep["m"]

    @property
    def d(self):
        return self._env.ep["d"]

    @property
    def env(self):
        return self._env

    def add_hook(self, fn):
        """fn(m, d, step) runs right before every mj_step (and before the monitors' pre-step checks)."""
        self._env.hooks.append(fn)

    def clear_hooks(self):
        self._env.hooks.clear()


class _InProcessAPI:
    """The command API as Python calls (for privileged scripts; mirrors envapi.py)."""

    def __init__(self, env: PracticeEnv):
        self._env = env

    def _c(self, op, **kw):
        r = self._env.command(dict(op=op, **{k: v for k, v in kw.items() if v is not None}))
        if "error" in r:
            raise EnvError(r["error"])
        return r

    def task(self):
        return self._c("task")

    def observe(self):
        return self._c("observe")

    def status(self):
        return self._c("status")

    def move(self, pos, quat=None, speed=None, force_max=None, stop_force=None, timeout=None):
        return self._c("move", pos=list(map(float, pos)), quat=None if quat is None else list(map(float, quat)),
                       speed=speed, force_max=force_max, stop_force=stop_force, timeout=timeout)

    def gripper(self, action, width=None):
        return self._c("gripper", action=action, width=width)

    def wait(self, seconds):
        return self._c("wait", seconds=seconds)

    def finish(self, claim=None):
        return self._c("finish", claim=claim)


def run_privileged(env: PracticeEnv, start_id: str, script, start_override: dict | None = None) -> dict:
    """Run a red-team script in THIS process with raw access. `script` is a path to a file defining run(api, raw),
    or such a callable (self-tests)."""
    import runpy
    if not env.privileged:
        raise PermissionError("env is not in privileged mode")
    env.hooks.clear()
    env.reset(start_id, start_override=start_override)
    if callable(script):
        fn = script
    else:
        ns = runpy.run_path(str(script))
        if "run" not in ns:
            raise EnvError("privileged script must define run(api, raw)")
        fn = ns["run"]
    err = None
    try:
        fn(_InProcessAPI(env), RawAccess(env))
    except Exception as ex:  # noqa: BLE001
        err = f"{type(ex).__name__}: {ex}"
    finally:
        env.hooks.clear()
    if env.ep is not None and not env.ep["done"]:
        env._end_episode(None)
    rec = env.ep["record"]
    return dict(start_id=start_id, outcome=rec["outcome"], raw_outcome=rec["raw_outcome"], counted=rec["counted"],
                flags=rec["monitors"]["flags"], max_seen=rec["monitors"]["max_seen"], script_error=err,
                file=env.ep["record_info"].get("file"), final=rec["evaluation"]["final"])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--socket", required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--idle-s", type=float, default=1800.0)
    a = ap.parse_args()
    if a.cmd == "serve":
        os.environ.setdefault("MUJOCO_GL", "glfw")   # before mujoco is imported: the render op needs GL on this box
        serve(a.socket, json.loads(Path(a.config).read_text()), a.idle_s)


if __name__ == "__main__":
    main()
