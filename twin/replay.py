"""Replay real REASSEMBLE insert attempts in the twin (stage B2, skill playbook/skills/replay-validate).

A replay starts from the real robot state ~1.5 s before contact (part in hand), drives the twin's copy
of the REASSEMBLE impedance controller (twin/control.py) toward a setpoint rebuilt from the log, and lets
MuJoCo decide whether the part seats or jams. Nothing is edited after t = 0: the command is fixed before
the run, the grasp is a compliant joint chain defined in the model, and the board is static.

    x_cmd(t) = x(t) - F_base(t)/K   (+ rotational analogue, + optional damping/inertia terms)

Modes
  compliant  controller tracks x_cmd with the REASSEMBLE gains (the validation replay)
  stiff      controller tracks the LOGGED pose with very high gains (geometry probe only: the force it
             produces says how far the twin's hole is from where the real part went)
  free       compliant, but scored on EE tracking only (free-space controller check)

The grasp: the part is a free body squeezed between the finger pads by a force-limited gripper servo
(grip_force per finger), set up in a pre-roll before t = 0. Slip and pivot come from pad contact friction:
sliding (pad_friction) and torsional (pad_torsion, the resistance to pivoting about the closing axis). The
logged gripper opening releases the part, so the window runs past the real release (after_release_s) and
the outcome is read from where the part ends up, which is what the dataset's success label describes.

Sim force/torque = contact forces on the hand and fingers from everything else (the part through the
pads, the board), which is what the wrist sensor measures. Real and sim forces are compared after
subtracting each one's value over the window's first frames (removes sensor bias; the part's weight is in
both).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from twin import REPO_DIR
from twin.control import (CartesianImpedance, ImpedanceParams, mat_to_quat, quat_to_mat, reconstruct_setpoint,
                          rot_error, slerp, tool_wrench_about_tcp)

DATA = REPO_DIR / "data" / "reassemble"
POSES = REPO_DIR / "data" / "reassemble-tudata" / "poses"
SPLITS = REPO_DIR / "playbook" / "splits.json"
CALIB = REPO_DIR / "twin" / "calib"
OUT = REPO_DIR / "runs" / "replay"
FPS = 30
HOLD_MAX = 0.075      # finger sum below this = holding something (open = 0.080)
RELEASE_OPEN = 0.003  # fingers this much wider than the holding width = released
TARE_FRAMES = 10      # frames at the window start used to tare the F/T bias (pre-contact)
CONTACT_N = 2.0       # contact onset: |F - F_ref| above this (F_ref = median of the segment's first 0.5 s)



# ------------------------------------------------------------------------------------------- data
@lru_cache(maxsize=1)
def splits() -> dict:
    return json.loads(SPLITS.read_text())["episodes"]


def split_of(ep: int) -> str:
    return splits()[str(int(ep))]["split"]


def recording_of(ep: int) -> str:
    return splits()[str(int(ep))]["recording"]


def guard(ep: int, allowed=("train", "gate")):
    s = split_of(ep)
    if s not in allowed:
        raise PermissionError(f"episode {ep} is in the {s} split; allowed here: {list(allowed)} (playbook/splits.json)")


@lru_cache(maxsize=1)
def _tasks() -> dict:
    import pandas as pd
    t = pd.read_parquet(DATA / "meta/tasks.parquet").reset_index()
    return dict(zip(t["task_index"].astype(int), t["task"]))


DATA_DIRS = (DATA, REPO_DIR / "data" / "reassemble-gate")  # gate parquet downloaded by board-from-contacts


def _data_file(ep: int) -> Path:
    import pandas as pd
    from twin import exam
    exam.check(ep)
    e = pd.read_parquet(DATA / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
    r = e.loc[int(ep)]
    rel = f"data/chunk-{int(r['data/chunk_index']):03d}/file-{int(r['data/file_index']):03d}.parquet"
    for d in DATA_DIRS:
        if (d / rel).exists():
            return d / rel
    raise FileNotFoundError(f"episode {ep}: {rel} not on disk (looked in {[str(d.relative_to(REPO_DIR)) for d in DATA_DIRS]})")


def signals(ep: int) -> dict:
    """Per-frame arrays of one episode (robot base frame). Reads data files only; exam episodes only for the
    exam-scorer caller (twin/exam.py)."""
    from twin import exam
    exam.check(ep)  # before the cache, so a cached exam episode is never served to another caller
    return _signals(ep)


@lru_cache(maxsize=8)
def _signals(ep: int) -> dict:
    import pyarrow.parquet as pq
    cols = ["timestamp", "segment.index", "segment.success", "task_index", "observation.state",
            "observation.force.base", "observation.torque.base"]
    tb = pq.read_table(_data_file(ep), columns=cols, filters=[("episode_index", "=", int(ep))]).sort_by("timestamp")
    arr = lambda c: np.asarray(tb.column(c).combine_chunks().flatten(), np.float64).reshape(tb.num_rows, -1)  # noqa
    st = arr("observation.state")
    return dict(t=np.asarray(tb.column("timestamp"), np.float64), q=st[:, 0:7], dq=st[:, 7:14],
                tau=st[:, 14:21], grip=st[:, 21:23], ee=st[:, 23:30], ee_vel=st[:, 30:36],
                F=arr("observation.force.base"), T=arr("observation.torque.base"),
                seg=np.asarray(tb.column("segment.index")), ok=np.asarray(tb.column("segment.success")),
                task=np.asarray(tb.column("task_index")))


def segments(ep: int) -> list[dict]:
    s = signals(ep)
    si = s["seg"]
    cut = np.flatnonzero(np.diff(si) != 0) + 1
    tasks = _tasks()
    return [dict(index=int(si[a]), task=tasks[int(s["task"][a])], a=int(a), b=int(b), success=bool(s["ok"][a]))
            for a, b in zip(np.r_[0, cut], np.r_[cut, len(si)])]


def part_of(task: str) -> str | None:
    if not task.startswith("Insert "):
        return None
    return task[len("Insert "):].rstrip(".")


def attempt_info(ep: int, seg: dict, lead_s: float = 1.5) -> dict | None:
    """Timing of one Insert attempt with the part in hand; None if the fingers were open (a push)."""
    s = signals(ep)
    a, b = seg["a"], seg["b"]
    g = s["grip"][a:b].sum(1)
    hold = g < HOLD_MAX
    if not hold[: max(1, len(g) // 3)].any():
        return None
    w = float(np.median(g[hold]))
    first_hold = int(np.argmax(hold))
    opened = np.flatnonzero(g > w + RELEASE_OPEN)
    opened = opened[opened > first_hold]
    rel = a + (int(opened[0]) - 1 if len(opened) else len(g) - 1)
    F_ref = np.median(s["F"][a:a + 15], axis=0)
    dF = np.linalg.norm(s["F"][a:rel + 1] - F_ref, axis=1)
    on = np.flatnonzero(dF > CONTACT_N)
    onset = a + int(on[0]) if len(on) else None
    k0 = max(a + first_hold, (onset if onset is not None else rel) - int(round(lead_s * FPS)))
    return dict(episode=int(ep), segment_index=seg["index"], task=seg["task"], part=part_of(seg["task"]),
                success=seg["success"], released=bool(len(opened)), grip_width_mm=round(1000 * w, 2),
                frames=dict(seg_start=a, seg_end=b - 1, start=int(k0), onset=onset, release=int(rel),
                            released=bool(len(opened))),
                t=dict(seg_start=round(float(s["t"][a]), 3), start=round(float(s["t"][k0]), 3),
                       onset=None if onset is None else round(float(s["t"][onset]), 3),
                       release=round(float(s["t"][rel]), 3)),
                real_peak_dF=round(float(dF.max()), 3), F_ref=F_ref.round(3).tolist())


def list_attempts(ep: int, family: str = "peg", lead_s: float = 1.5) -> list[dict]:
    out = []
    for seg in segments(ep):
        p = part_of(seg["task"])
        if p is None or (family and family not in p):
            continue
        info = attempt_info(ep, seg, lead_s)
        if info is not None:
            out.append(info)
    return out


def train_attempts(family: str = "peg", episodes=None) -> list[dict]:
    """Every in-hand attempt of the family in local TRAIN episodes (splits.json filter)."""
    from twin.world import _episodes_meta  # noqa: F401  (meta reader shared with the world tools)
    import pandas as pd
    e = pd.read_parquet(DATA / "meta/episodes/chunk-000/file-000.parquet")
    out = []
    for ep in sorted(int(x) for x in e.episode_index):
        if episodes is not None and ep not in episodes:
            continue
        if split_of(ep) != "train":
            continue
        try:
            _data_file(ep)
        except FileNotFoundError:
            continue
        out += list_attempts(ep, family)
    return out


# ------------------------------------------------------------------------------------------ board
BOARD_CONTACT_DIR = REPO_DIR / "runs" / "claude" / "run-standin-1" / "board"  # board-from-contacts outputs


def board_pose_for(recording: str, source: str = "auto", part: str | None = None) -> dict:
    """Board frame (mocap NIST_Board1 convention) in the world, and optionally the hole xy of `part`.

    source: auto | contact | calib_contact | calib | mocap
      contact        runs/claude/run-standin-1/board/<recording>.json (board-from-contacts: x, y, yaw from
                     this recording's insert seats / remove grasps, hole xy from its contact layout); z from
                     mocap, as that tool's calib entry does (contacts do not pin z)
      calib_contact  twin/calib/<recording>.json "board_contact"
      calib          twin/calib/<recording>.json "board"
      mocap          poses JSON NIST_Board1
      auto           first available in that order
    """
    raw = json.loads((POSES / f"{recording}_poses.json").read_text())["NIST_Board1"]
    x, y, z, w = raw[1]
    mocap = dict(pos=list(map(float, raw[0])), quat=[w, x, y, z], source=f"mocap NIST_Board1 ({recording}_poses.json)")
    if source in ("auto", "contact"):
        f = BOARD_CONTACT_DIR / f"{recording}.json"
        if f.exists():
            c = json.loads(f.read_text())
            if c.get("pose") and c.get("holes"):
                out = dict(pos=[c["pose"]["pos"][0], c["pose"]["pos"][1], mocap["pos"][2]],
                           quat=list(c["pose"]["quat_wxyz"]), source=f"{f.relative_to(REPO_DIR)} (x, y, yaw); z mocap",
                           sigma=c["pose"].get("sigma"))
                if part is not None and part in c["holes"]:
                    out["hole_xy"] = list(c["holes"][part]["board_xy"])
                return out
        if source == "contact":
            raise KeyError(f"no board-from-contacts result for {recording} in {BOARD_CONTACT_DIR.relative_to(REPO_DIR)}")
    cal = CALIB / f"{recording}.json"
    c = json.loads(cal.read_text()) if cal.exists() else {}
    if source in ("auto", "calib_contact") and "board_contact" in c:
        b = c["board_contact"]
        return dict(pos=list(b["pos"]), quat=list(b["quat"]), source=f"twin/calib/{recording}.json board_contact")
    if source in ("auto", "calib") and "board" in c:
        b = c["board"]
        return dict(pos=list(b["pos"]), quat=list(b["quat"]), source=f"twin/calib/{recording}.json board")
    if source in ("calib_contact", "calib"):
        raise KeyError(f"no {source} entry in twin/calib/{recording}.json")
    return mocap


# ----------------------------------------------------------------------------------------- params
@dataclass
class ReplayParams:
    # controller (REASSEMBLE gains; K bracketed 200/500/800 and 10/50/75)
    k_trans: float = 500.0
    k_rot: float = 50.0
    zeta: float = 1.0
    setpoint: str = "dynamic"        # static: x - F/K | dynamic: + K^-1 (D v + Lambda a)
    use_accel: bool = True           # Lambda a term (a from the logged EE twist; free-space RMSE 0.97 -> 0.73 mm)
    tare: bool = True                # remove the F/T bias (tool frame) seen in the window's first frames
    joint_friction_scale: float = 0.0  # x menagerie FR3 frictionloss (1.137 Nm joints 1-4). 0: the rebuilt
    #                                    setpoint already absorbs the real arm's friction (free-space check)
    ns_every: int = 10               # manipulability gradient every n ticks (nullspace only; 2.3x faster)
    ns_damping: float = 0.1          # REASSEMBLE nullspace gains (compliance_param.cfg)
    ns_manip: float = 0.0            # REASSEMBLE: 10. Off in the twin: without the real joint friction the nullspace
    ns_jl: float = 0.0               # REASSEMBLE: 10. potentials ring and leak into the EE (free space p90 4.1 -> 1.8 mm)
    joint_damping: float | None = None  # arm joint viscous damping, Nms (None: menagerie 0.21)
    # grasp: the part is a free body squeezed by the finger pads (contact, set up before t = 0)
    grasp_from_top: float = 0.009    # m, pad centre below the part's top end (re-identified from onset timing)
    grasp_dx: float = 0.0            # m, part axis offset from the TCP along hand x (along the pads)
    grip_force: float = 60.0         # N per finger (teleop_franka.py GripperWrapper.close(60))
    pad_friction: float = 1.0        # sliding friction of the pads (max rule with the part's)
    pad_torsion: float = 0.002       # m, torsional friction of the pads: resists pivoting about the closing axis
    pad_solref: float | None = None  # contact time constant of the pads (None: solref_tc)
    pad_tc_by_part: dict | None = None  # s, v5b soft pads: pad-part contact time constant per part, identified from
    #                                     the real hold-width deficit at 60 N; the pads' solmix is raised so the pair
    #                                     uses it (None or a part not listed: unchanged)
    reg_rotation: str = "none"       # inhand registration of the part's rotation in the hand: none (aligned with the
    #                                  hand) | pivot (about the closing axis, into the pad plane) | full (vertical)
    grasp_dy: float = 0.0            # m, part centre off the finger mid-plane (finger coupling offset; set by inhand)
    reg_height: str = "ref"          # inhand: ref = tip depth_ref at k_ref (released) / 0 at the lowest held TCP;
    #                                  global = keep grasp_from_top
    reg_depth: float = 0.002         # m, tip depth below the surface at the release frame (reg_height = ref)
    grasp_rock: float = 0.0          # rad, registered rock of the part in the hand (about hand x); the pads' rest angle
    grasp_pivot: float = 0.0         # rad, registered pivot (about hand y); the pads' rest angle when finger_pivot_k > 0
    finger_rock_k: float = 0.0       # Nm/rad per finger, hinge about the pad-width axis at the pad centre (0: rigid)
    finger_pivot_k: float = 0.0      # Nm/rad per finger, hinge about the pad normal at the pad centre (0: rigid)
    finger_rock_k_by_part: dict | None = None  # v5b: Nm/rad per finger for these parts (overrides finger_rock_k)
    finger_rock_axis: str = "pad"    # v5b: pad = rock hinge through each pad centre (as before) | mid = both rock
    #                                  hinges on the part's axis (grasp mid-plane, half the gripped width inside each
    #                                  pad face), so a rigid part can rock without shearing the pads
    finger_zeta: float | None = None  # v5b: hinge damping ratio on (armature + finger + half the part) inertia about
    #                                   the hinge; None: the old numerical rule 4 sqrt(k armature)
    finger_rock_couple: bool = False  # v5b: tie the two rock hinges (equal rotation in the hand frame, a joint coupling
    #                                   like the finger_joint1/2 one) so the pads stay parallel: a compliant jaw pair
    # contact (part-board); B1 defaults
    friction: float = 0.3            # part friction (max rule: board is 0.05)
    solref_tc: float = 0.004         # contact time constant, s
    solimp: tuple = (0.95, 0.99, 0.0005, 0.5, 2.0)
    cone: str = "elliptic"
    impratio: float = 10.0
    noslip_iterations: int = 0       # MuJoCo noslip post-solver iterations (0 = off, unchanged). It suppresses the soft-
    #                                  contact creep that lets a light part turn in the pads with no load (about 1 deg/s;
    #                                  replay/diag, 2026-10-25); a solver option set before t = 0
    board_solref_tc: float | None = None  # board geoms' own contact time constant (None: solref_tc, i.e. unchanged).
    #                                       The part-board pair mixes to the mean of part and board (solmix 1:1);
    #                                       the pads and the part keep solref_tc (v5a knob, 2026-10-25)
    board_solimp: tuple | None = None     # board geoms' own solimp (None: solimp); mixed with the part's like solref
    # board / part geometry
    surface_z: float = 0.0303        # plate top above the mocap board z: +30.3 mm (+-4), plate corners triangulated by
    #                                  the calibrated static cameras (twin/calib board entry; world-frame Lessons)
    seat_depth: float = 0.020        # hole depth: episode-0 Remove pull-outs 17-23 mm, hama1 peg tops ~30 mm up
    plate_thickness: float | None = None  # walled depth of the hole; None = seat_depth (holes walled to the bottom)
    board_source: str = "auto"
    board_offset: tuple = (0.0, 0.0, 0.0)  # m, added in the board frame (geometry probe)
    pad_offset: float = 0.0          # m, board shifted along the hand's finger-closing axis at the window start
    #                                  (projected into the board plane): the sensitivity sweep "across the pads"
    legacy_hand: bool = False        # ablation only: the pre-fix hand (pads on O_T_EE, F/T stack 70.6 mm)
    registration: str = "none"       # none (open-loop prediction) | inhand (B2 reconstruction, label-blind) |
    #                                  release_oracle / full_oracle (old diagnostics) | release_oracle (DIAGNOSTIC, uses the real release
    #                                  pose: puts the hole under the real TCP at release; not a valid B2 score)
    hole_widen: float = 0.0          # m per side (negative control)
    hole_widen_by_part: dict | None = None  # m per side added to the hole of these parts (effective clearance)
    # run
    timestep: float = 0.001
    lead_s: float = 1.5              # window starts this long before the real contact onset
    after_release_s: float = 1.0     # window ends this long after the real gripper starts opening
    max_s: float = 30.0
    preroll_s: float = 0.25          # before t = 0: grip the part, arm held at the logged start pose
    stiff_k: float = 2e4             # stiff kinematic replay gains (N/m; rot = stiff_k / 20 Nm/rad)
    # DIAGNOSIS ONLY (B2 arrival-error diagnosis, 2026-10-25; runs/claude/run-standin-1/replay/diag/). The oracles
    # read the log after t = 0 or change the model after t = 0, so they never go in a frozen, gate or practice
    # params file. Defaults reproduce the behaviour without them.
    diag_trace: bool = False         # per-frame sim/log hand pose, part tip and axis, forces -> result["diag"]
    diag_hand_oracle_k: float = 0.0  # N/m: extra wrench K_h (x_log - x) + D_h (v_log - v) pins the twin TCP to the
    #                                  LOGGED TCP (D_h critical on the arm's Lambda); 0 = off
    diag_hand_oracle_kr: float = 0.0  # Nm/rad: the same for the TCP orientation; 0 = off
    diag_hand_oracle_axes: str = "lateral"  # lateral = only in the board plane (the axial push stays compliant) | all
    diag_inhand_lock: bool = False   # weld the part to the hand at its registered (t = 0) pose until the logged release
    diag_board_oracle: bool = False  # after registration, shift the board so the registered tip sits on the hole axis
    #                                  at k_ref with the registered depth (absorbs the clipped residual and grasp height)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["solimp"] = list(self.solimp)
        d["board_offset"] = list(self.board_offset)
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "ReplayParams":
        d = dict(d or {})
        known = {f for f in cls.__dataclass_fields__}
        bad = set(d) - known
        if bad:
            raise ValueError(f"unknown replay params {sorted(bad)}; known: {sorted(known)}")
        for k in ("solimp", "board_offset", "board_solimp"):
            if k in d and d[k] is not None:
                d[k] = tuple(d[k])
        return cls(**d)


def impedance(rp: ReplayParams, stiff: bool = False) -> ImpedanceParams:
    if stiff:
        return ImpedanceParams(k_trans=rp.stiff_k, k_rot=rp.stiff_k / 20, zeta=1.0, filter_alpha=0.0, dtau_max=0.0,
                               ns_every=rp.ns_every)
    return ImpedanceParams(k_trans=rp.k_trans, k_rot=rp.k_rot, zeta=rp.zeta, filter_alpha=0.0, ns_every=rp.ns_every,
                           ns_damping=rp.ns_damping, ns_manip=rp.ns_manip, ns_jl=rp.ns_jl)


# ------------------------------------------------------------------------------------------ scene
FINGERS = ("gripper/left_finger", "gripper/right_finger")
from twin.scene.build_base import FT_STACK_TCP_MATCH, HAND_TCP, TCP_SITE_Z  # noqa: E402
PAD_TO_TCP = TCP_SITE_Z - HAND_TCP  # 0.0229 m: the logged O_T_EE lies this far beyond the pad centre (hand z)


def pad_to_tcp(rp) -> float:
    return 0.0 if getattr(rp, "legacy_hand", False) else PAD_TO_TCP
PAD_Z = 0.045          # pad centre on the finger body (menagerie hand: pads at finger z 0.0395-0.05)
GRIP_KP = 5000.0       # N/m on the finger tendon; the squeeze is set by the force limit, not by kp
GRIP_CLOSED = -0.03    # tendon target while holding: always past contact, so the force limit governs


def part_world_pose(part, tcp_pos, tcp_quat, rp: ReplayParams, R_inhand=None):
    """Part body pose that puts its grasp site at the pad centre (PAD_TO_TCP above the logged TCP, + grasp_dx/dy),
    axis along hand z (or rotated in the hand by R_inhand about the grasp site)."""
    R = quat_to_mat(tcp_quat)
    Ri = np.eye(3) if R_inhand is None else np.asarray(R_inhand)
    Rp = R @ Ri
    grasp = np.asarray(tcp_pos, float) + R @ np.array([rp.grasp_dx, rp.grasp_dy, -pad_to_tcp(rp)])
    origin = grasp + Rp @ np.array([0.0, 0.0, part.length / 2 - rp.grasp_from_top])
    return origin.tolist(), mat_to_quat(Rp).tolist()


def build_model(part_name: str, board_pose: dict, rp: ReplayParams, tcp_pose=None, cameras: dict | None = None,
                R_inhand=None):
    """base.xml arm (torque motors, firmware gravity compensation) + board with the part's hole + the part as
    a free body between the finger pads at the logged start pose. Returns (MjModel, info)."""
    import mujoco
    from twin import BASE_XML
    from twin.board import CONTACT, add_board, board_spec
    from twin.parts import add_part_body, load_layout, part_spec

    lay = load_layout()
    board = board_spec(lay)
    part = part_spec(part_name, lay)
    if board_pose.get("hole_xy") is not None:
        part = replace(part, hole_xy=tuple(board_pose["hole_xy"]))
    board.params["surface_z"] = replace(board.params["surface_z"], value=float(rp.surface_z))
    board.params["seat_depth"] = replace(board.params["seat_depth"], value=float(rp.seat_depth))
    board.params["plate_thickness"] = replace(board.params["plate_thickness"],
                                              value=float(rp.plate_thickness or rp.seat_depth))
    widen = float(rp.hole_widen) + float((rp.hole_widen_by_part or {}).get(part_name, 0.0))
    if widen:
        for k in ("hole_d", "hole_a", "hole_b"):
            if k in part.params:
                part.params[k] = replace(part.params[k], value=part.v(k) + 2 * widen)
    part.params["friction"] = replace(part.params["friction"], value=float(rp.friction))
    part.params["grasp_from_top"] = replace(part.params["grasp_from_top"], value=float(rp.grasp_from_top))
    contact = dict(CONTACT, solref=[float(rp.solref_tc), 1.0], solimp=list(rp.solimp))
    board.contact = dict(contact)
    if rp.board_solref_tc is not None:
        board.contact["solref"] = [float(rp.board_solref_tc), 1.0]
    if rp.board_solimp is not None:
        board.contact["solimp"] = list(rp.board_solimp)
    if rp.hole_widen:
        board.contact["board_friction"] = 0.0

    spec = mujoco.MjSpec.from_file(str(BASE_XML))
    spec.option.timestep = rp.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC if rp.cone == "elliptic" else mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.impratio = float(rp.impratio)
    if rp.noslip_iterations:
        spec.option.noslip_iterations = int(rp.noslip_iterations)
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.tolerance = 1e-10
    spec.option.iterations = 100
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET
    b = spec.body("task_board")
    if b is not None:
        spec.delete(b)
    tt = spec.geom("table_top")
    if tt is not None:  # the board's own base is the stop; the table stays visual only
        tt.contype = 0
        tt.conaffinity = 0
    # arm: torque motors (the controller is ours); firmware gravity compensation on arm + hand
    for i in range(1, 8):
        a = spec.actuator(f"fr3_joint{i}")
        lim = 87.0 if i <= 4 else 12.0
        a.set_to_motor()
        a.ctrlrange = [-lim, lim]
        a.ctrllimited = True
    for bb in spec.bodies:
        if bb.name and (bb.name.startswith("fr3_link") or bb.name.startswith("gripper/")):
            bb.gravcomp = 1.0
    if rp.legacy_hand:  # ablation: put the hand back where base.xml had it before the 22.9 mm fix
        spec.body("gripper/hand").pos = [0.0, 0.0, FT_STACK_TCP_MATCH]
        spec.site("tcp").pos = [0.0, 0.0, HAND_TCP]
    # gripper: force-limited position servo on the finger tendon (0.5 q1 + 0.5 q2): per-finger force = F_t / 2
    g = spec.actuator("gripper/actuator8")
    g.gainprm[0] = GRIP_KP
    g.biasprm[0], g.biasprm[1], g.biasprm[2] = 0.0, -GRIP_KP, -2 * math.sqrt(GRIP_KP * 0.2)
    g.ctrlrange = [GRIP_CLOSED, 0.04]
    g.forcerange = [-2 * rp.grip_force, 2 * rp.grip_force]

    # finger compliance (pads that rock / pivot), defined before t = 0
    k_rock = float((rp.finger_rock_k_by_part or {}).get(part_name, rp.finger_rock_k))
    # v5b "mid": in each finger's frame the part axis sits half the gripped width (section along hand y) inside
    # the pad face (pads 2-5 face at finger y = 0; the right finger's frame is turned 180 deg about z)
    rock_pos = [0, -part.section[1] / 2, PAD_Z] if rp.finger_rock_axis == "mid" else [0, 0, PAD_Z]
    if rp.finger_rock_axis not in ("pad", "mid"):
        raise ValueError("finger_rock_axis must be pad | mid")
    for fn, sgn in zip(FINGERS, (1.0, -1.0)):  # the right finger is turned 180 deg about z: its x and y flip
        fb = spec.body(fn)
        for nm, ax, k, ref in (("rock", [1, 0, 0], k_rock, rp.grasp_rock),
                               ("pivot", [0, 1, 0], rp.finger_pivot_k, rp.grasp_pivot)):
            if k > 0:
                arm_ = max(1e-6, k * (rp.timestep / 0.4) ** 2)
                fb.add_joint(name=f"{fn}/{nm}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=ax,
                             pos=rock_pos if nm == "rock" else [0, 0, PAD_Z],
                             stiffness=k, springref=sgn * ref, damping=4 * math.sqrt(k * arm_), armature=arm_,
                             range=[-0.35, 0.35], limited=True)
    if rp.finger_rock_couple and k_rock > 0:  # v5b: right rock = -left rock (its x axis is flipped), before t = 0
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_JOINT
        eq.name = "v5b/rock_couple"
        eq.name1, eq.name2 = f"{FINGERS[0]}/rock", f"{FINGERS[1]}/rock"
        eq.data[:5] = [0.0, -1.0, 0.0, 0.0, 0.0]

    from twin.assembly import _compose
    off = np.asarray(rp.board_offset, float)
    bpos, bquat = _compose(board_pose["pos"], board_pose["quat"], off.tolist(), [1, 0, 0, 0])
    mat = "board_mat" if spec.material("board_mat") is not None else None
    _, binfo = add_board(spec, spec.worldbody, [part], board, bpos, bquat, material=mat)

    # the part: a free body whose grasp site sits at the logged TCP at the window start
    pc = dict(contact, torsional=1e-5)  # the pads own the torsional friction (max rule)
    if tcp_pose is None:
        ppos, pquat = [0.5, 0.0, 0.3], [0.0, 1.0, 0.0, 0.0]
    else:
        ppos, pquat = part_world_pose(part, tcp_pose[0], tcp_pose[1], rp, R_inhand)
    add_part_body(spec, spec.worldbody, part, ppos, pquat, free=True, contact=pc)
    if getattr(rp, "diag_inhand_lock", False):  # DIAGNOSIS ONLY: relpose set in run_replay, released at the release
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name = "diag/inhand_lock"
        eq.name1, eq.name2 = "gripper/hand", part.slug
        eq.solref = [max(0.002, 2 * rp.timestep), 1.0]
        eq.solimp = [0.9999, 0.9999, 0.001, 0.5, 2.0]
    for cname, cam in (cameras or {}).items():
        c = spec.camera(cname)
        if c is None:
            continue
        c.pos = list(cam["pos"])
        c.quat = list(cam["quat"])
        c.fovy = float(cam["fovy_deg"])
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    m = spec.compile()
    # finger pads and finger collision meshes: sliding + torsional friction, B1 contact softness
    fids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f) for f in FINGERS}
    for gi in range(m.ngeom):
        if m.geom_bodyid[gi] in fids and m.geom_contype[gi]:
            m.geom_condim[gi] = 4
            m.geom_friction[gi] = [rp.pad_friction, rp.pad_torsion, 1e-5]
            m.geom_solref[gi] = [rp.pad_solref or rp.solref_tc, 1.0]
            m.geom_solimp[gi] = list(rp.solimp)
    if rp.finger_zeta is not None:  # v5b: physical damping ratio on the hinge modes (armature + finger + part/2)
        pbid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, part.slug)
        i_part = float(m.body_inertia[pbid].max()) + float(m.body_mass[pbid]) * (part.length / 2 - rp.grasp_from_top) ** 2
        for fn in FINGERS:
            fbid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, fn)
            for nm in ("rock", "pivot"):
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{fn}/{nm}")
                if j < 0:
                    continue
                r = np.asarray(m.jnt_pos[j]) - np.asarray(m.body_ipos[fbid])
                ax = np.asarray(m.jnt_axis[j])
                r_perp2 = float(r @ r - (r @ ax) ** 2)
                da = m.jnt_dofadr[j]
                i_eff = float(m.dof_armature[da]) + float(m.body_inertia[fbid].max()) + float(m.body_mass[fbid]) * r_perp2 + i_part / 2
                if nm == "rock" and rp.finger_rock_axis == "pad":
                    i_eff += float(m.body_mass[pbid]) * (part.section[1] / 2) ** 2 / 2
                m.dof_damping[da] = 2 * rp.finger_zeta * math.sqrt(float(m.jnt_stiffness[j]) * i_eff)
    tc_pad = (rp.pad_tc_by_part or {}).get(part_name)
    if tc_pad is not None:  # v5b soft pads: the pad's own softness governs the pad-part pair (solmix weight)
        for gi in range(m.ngeom):
            if m.geom_bodyid[gi] in fids and m.geom_contype[gi]:
                m.geom_solref[gi] = [float(tc_pad), 1.0]
                m.geom_solmix[gi] = 1e3
    if rp.grasp_dy:
        # finger coupling q1 = q2 (+ c0): with c0 = 2 dy the pads' mid-plane sits dy along hand y
        for e in range(m.neq):
            if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT:
                m.eq_data[e, 0] = 2 * rp.grasp_dy
        ec = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "v5b/rock_couple")
        if ec >= 0:  # the closing offset belongs to the finger slide coupling only
            m.eq_data[ec, 0] = 0.0
    if rp.joint_damping is not None:
        for i in range(1, 8):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"fr3_joint{i}")
            m.dof_damping[m.jnt_dofadr[j]] = rp.joint_damping
    if rp.joint_friction_scale != 1.0:
        for i in range(1, 8):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"fr3_joint{i}")
            m.dof_frictionloss[m.jnt_dofadr[j]] *= rp.joint_friction_scale
    info = dict(part=part.describe(), board_pose=board_pose, board_offset=list(off), board=binfo)
    return m, info


def close_fingers_on_part(m, d, slug: str, squeeze: float = 5e-5, dy: float = 0.0):
    """Finger opening at which the pads just touch the part (bisection on the signed geom distance)."""
    import mujoco
    fids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f) for f in FINGERS}
    pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, slug)
    pads = [g for g in range(m.ngeom) if m.geom_bodyid[g] in fids and m.geom_contype[g]]
    pg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == pb]
    j1 = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper/finger_joint1")]
    j2 = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper/finger_joint2")]
    ft = np.zeros(6)

    def gap(q):
        d.qpos[j1] = q + dy
        d.qpos[j2] = q - dy
        mujoco.mj_kinematics(m, d)
        return min(mujoco.mj_geomDistance(m, d, a, b, 0.05, ft) for a in pads for b in pg)

    lo, hi = abs(dy), 0.04
    if gap(lo) > 0:
        return lo
    for _ in range(40):
        mid = (lo + hi) / 2
        if gap(mid) > -squeeze:
            hi = mid
        else:
            lo = mid
    gap(hi)
    return hi


class ToolSensor:
    """What the wrist F/T sensor sees: contact forces on the hand and fingers from everything else
    (the held part through the pads, the board), world frame; torque about the world origin."""

    def __init__(self, m, part_slug: str):
        import mujoco
        self.tool = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("gripper/hand",) + FINGERS}
        self.tool.discard(-1)
        self.m = m
        self._f6 = np.zeros(6)

    def wrench(self, d) -> tuple[np.ndarray, np.ndarray]:
        import mujoco
        F = np.zeros(3)
        T = np.zeros(3)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            in1, in2 = b1 in self.tool, b2 in self.tool
            if in1 == in2 or c.efc_address < 0:
                continue
            mujoco.mj_contactForce(self.m, d, i, self._f6)
            f_on_2 = c.frame.reshape(3, 3).T @ self._f6[:3]
            f = f_on_2 if in2 else -f_on_2
            F += f
            T += np.cross(c.pos, f)
        return F, T


def _part_mass(m, slug: str) -> float:
    import mujoco
    return float(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, slug)])


# ---------------------------------------------------------------------------------------- replay
def _interp_targets(ts: np.ndarray, X: np.ndarray, Q: np.ndarray, t: float):
    if t <= ts[0]:
        return X[0], Q[0]
    if t >= ts[-1]:
        return X[-1], Q[-1]
    k = int(np.searchsorted(ts, t) - 1)
    a = (t - ts[k]) / (ts[k + 1] - ts[k])
    return X[k] + a * (X[k + 1] - X[k]), slerp(Q[k], Q[k + 1], a)


def tare(s: dict, k0: int, part_mass: float, n: int = TARE_FRAMES) -> dict:
    """Sensor bias in the TOOL frame from the window's first frames (pre-contact by construction: the
    window starts lead_s before contact). The part's weight is real (the twin's part has it too), so it is
    kept: bias = reading - weight. Episodes 2-3 carry a 5-6 Nm torque.base bias in free space."""
    sl = slice(k0, k0 + n)
    F0 = np.median(s["F"][sl], axis=0)
    x0 = s["ee"][k0, :3]
    T0 = np.median([tool_wrench_about_tcp(s["F"][k], s["T"][k], s["ee"][k, :3]) for k in range(k0, k0 + n)], axis=0)
    R0 = quat_to_mat(s["ee"][k0, 3:7])
    w = np.array([0.0, 0.0, -9.81 * part_mass])
    return dict(F_tool=R0.T @ (F0 - w), T_tool=R0.T @ T0, F0=F0, T0=T0, x0=x0, part_weight=w)


def wrench_at(s: dict, k: int, tb: dict | None):
    """Tared (F on the tool, torque about the TCP), base frame, at frame k."""
    F = s["F"][k]
    T = tool_wrench_about_tcp(F, s["T"][k], s["ee"][k, :3])
    if tb is None:
        return F, T
    R = quat_to_mat(s["ee"][k, 3:7])
    return F - R @ tb["F_tool"], T - R @ tb["T_tool"]


def reconstruct_window(m, s: dict, k0: int, k1: int, rp: ReplayParams, base_qpos=None, tb: dict | None = None):
    """Setpoint x_d (and q_d) for frames k0..k1 from the log."""
    from twin.control import ArmModel, stiffness_matrix
    arm = ArmModel(m)
    ip = impedance(rp)
    K = stiffness_matrix(ip)
    X, Q = [], []
    ee = s["ee"]
    vel = s["ee_vel"]
    acc = np.gradient(vel, s["t"], axis=0) if rp.use_accel else None
    for k in range(k0, k1 + 1):
        x, q = ee[k, :3], ee[k, 3:7]
        F, Ttcp = wrench_at(s, k, tb)
        if rp.setpoint == "dynamic":
            d = arm.state_at(s["q"][k], s["dq"][k], base_qpos)
            J, M, Lam, D = arm.lambda_d(d, K, ip.zeta)
            v = J @ s["dq"][k]
            xd, qd = reconstruct_setpoint(x, q, F, Ttcp, ip, "dynamic", v=v,
                                          a=None if acc is None else acc[k], Lam=Lam, D=D)
        else:
            xd, qd = reconstruct_setpoint(x, q, F, Ttcp, ip, "static")
        X.append(xd)
        Q.append(qd)
    return np.array(X), np.array(Q)


def _register_release(bp: dict, part: str, s: dict, k_rel: int, rp: ReplayParams) -> dict:
    """DIAGNOSTIC ONLY: shift the board in its plane so the part's hole sits under the real TCP at the real
    release frame (part assumed vertical in the hole with its axis through the TCP). It copies where the
    real part went, so it measures the physics given perfect registration, never the twin's prediction."""
    from twin.parts import load_layout, part_spec
    hxy = bp.get("hole_xy") or list(part_spec(part, load_layout()).hole_xy)
    Rb = quat_to_mat(bp["quat"])
    tcp_b = Rb.T @ (s["ee"][k_rel, :3] - np.asarray(bp["pos"]))
    off = np.array([tcp_b[0] - hxy[0], tcp_b[1] - hxy[1], 0.0]) + np.asarray(rp.board_offset, float)
    out = dict(bp, source=bp["source"] + " + release_oracle shift (diagnostic)", oracle_shift_mm=(1000 * off[:2]).round(2).tolist())
    out["pos"] = (np.asarray(bp["pos"]) + Rb @ off).tolist()
    return out


def ref_frame(s: dict, f: dict, bp: dict) -> int:
    """The attempt's reference frame for in-hand registration, from logged poses and the gripper only:
    the release frame if the gripper opened, else the frame where the held TCP was lowest along the board
    normal (from contact onset, or the window start, to the end of the held phase)."""
    if f.get("released", True) and f["release"] < f["seg_end"]:
        return int(f["release"])
    Rb = quat_to_mat(bp["quat"])
    a = f["onset"] if f["onset"] is not None else f["start"]
    h = (s["ee"][a:f["release"] + 1, :3] - np.asarray(bp["pos"])) @ Rb[:, 2]
    return int(a + int(np.argmin(h)))


def register_inhand(bp: dict, part: str, s: dict, f: dict, rp: ReplayParams):
    """B2 reconstruction registration: label-blind (one rule for every attempt), poses only, set before t = 0.

    At the reference frame k_ref (ref_frame: the release frame if the gripper opened, else the lowest held TCP):
      rotation  rp.reg_rotation: none = the part is aligned with the hand; pivot = rotated about the hand y
                (finger-closing) axis so its axis matches the hole axis within the pad plane; full = vertical;
      height    global rp.grasp_from_top (identified from onset timing, not per attempt);
      position  offsets dx (hand x, along the pads) and dy (hand y, the finger mid-plane offset, held by an offset
                on the finger-coupling equality) chosen so the part's TIP sits on the hole axis of the calibrated
                board at k_ref. dy absorbs board error along the closing axis; it is reported.
    The tip's depth at k_ref follows from the logged TCP height and the global grasp height; it is reported."""
    from twin.parts import load_layout, part_spec
    ps = part_spec(part, load_layout())
    hxy = bp.get("hole_xy") or list(ps.hole_xy)
    Rb = quat_to_mat(bp["quat"])
    H = np.asarray(bp["pos"]) + Rb @ np.array([hxy[0], hxy[1], rp.surface_z])
    k = ref_frame(s, f, bp)
    Rh = quat_to_mat(s["ee"][k, 3:7])
    p = s["ee"][k, :3] - Rh[:, 2] * pad_to_tcp(rp)  # pad centre (the logged TCP is PAD_TO_TCP beyond it)
    z_in = -Rb[:, 2]
    v = Rh.T @ z_in
    R_in = np.eye(3)
    phi = 0.0
    if rp.reg_rotation in ("pivot", "full"):
        phi = math.atan2(v[0], v[2])
        R_in = np.array([[math.cos(phi), 0, math.sin(phi)], [0, 1, 0], [-math.sin(phi), 0, math.cos(phi)]])
    theta = 0.0
    if rp.reg_rotation == "full":
        # R_in = R_x(theta) R_y(phi) maps the hand z onto the hole axis v (theta: rock, phi: pivot)
        theta = math.atan2(-v[1], math.hypot(v[0], v[2]))
        Rx = np.array([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]])
        R_in = Rx @ R_in
    elif rp.reg_rotation not in ("none", "pivot"):
        raise ValueError("reg_rotation must be none | pivot | full")
    a_ = Rh @ R_in @ np.array([0.0, 0.0, 1.0])
    released = (k == f["release"] and f.get("released", True))
    g = rp.grasp_from_top
    if rp.reg_height == "ref":
        # TCP height above the surface along the board normal, then the length below the TCP that puts the tip
        # depth_ref into the hole (released) or on the surface (held throughout), measured along the part axis
        h = float((p - H) @ Rb[:, 2])
        d_ref = rp.reg_depth if released else 0.0
        below_needed = (h + d_ref) / max(float(a_ @ z_in), 0.5)
        g = float(np.clip(ps.length - below_needed, 0.0, 0.030))
    elif rp.reg_height != "global":
        raise ValueError("reg_height must be ref | global")
    below = ps.length - g
    tip0 = p + a_ * below
    # in-plane (board) offsets of the grasp point along hand x and y that move the tip onto the hole axis
    Pb = np.eye(3) - np.outer(Rb[:, 2], Rb[:, 2])
    A = np.column_stack([Pb @ Rh[:, 0], Pb @ Rh[:, 1]])
    sol, *_ = np.linalg.lstsq(A, Pb @ (H - tip0), rcond=None)
    dx = float(np.clip(sol[0], -0.008, 0.008))
    dy = float(np.clip(sol[1], -0.006, 0.006))
    tip = tip0 + Rh[:, 0] * dx + Rh[:, 1] * dy
    resid = Pb @ (H - tip)
    info = dict(k_ref=k, ref="release" if released else "lowest_tcp", grasp_from_top_mm=round(1000 * g, 2),
                rotation=rp.reg_rotation, pivot_deg=round(math.degrees(phi), 3), rock_deg=round(math.degrees(theta), 3),
                pad_offset_mm=round(1000 * dx, 3), closing_offset_mm=round(1000 * dy, 3),
                residual_mm=round(1000 * float(np.linalg.norm(resid)), 3),
                part_tilt_deg=round(math.degrees(math.acos(float(np.clip(a_ @ z_in, -1, 1)))), 3),
                tip_depth_at_ref_mm=round(1000 * float((H - tip) @ Rb[:, 2]), 2))
    info["_angles"] = (theta, phi)
    return R_in, dx, dy, g, info


def _board_oracle(bp: dict, part: str, s: dict, f: dict, rp: ReplayParams, R_in, dx: float, dy: float, g: float,
                  k: int) -> dict:
    """DIAGNOSIS ONLY: move the board so that the registered tip (logged hand at k_ref, registered in-hand pose)
    sits on the hole axis at the registration's target depth. Undoes what the clipped dx/dy and the clipped
    grasp height could not absorb: the hole is put where this attempt's own pose says it was."""
    from twin.parts import load_layout, part_spec
    ps = part_spec(part, load_layout())
    hxy = bp.get("hole_xy") or list(ps.hole_xy)
    Rb = quat_to_mat(bp["quat"])
    n = Rb[:, 2]
    H = np.asarray(bp["pos"]) + Rb @ np.array([hxy[0], hxy[1], rp.surface_z])
    Rh = quat_to_mat(s["ee"][k, 3:7])
    p = s["ee"][k, :3] - Rh[:, 2] * pad_to_tcp(rp)
    a_ = Rh @ np.asarray(R_in) @ np.array([0.0, 0.0, 1.0])
    tip = p + a_ * (ps.length - g) + Rh[:, 0] * dx + Rh[:, 1] * dy
    released = (k == f["release"] and f.get("released", True))
    d_t = rp.reg_depth if (released and rp.reg_height == "ref") else 0.0
    Pb = np.eye(3) - np.outer(n, n)
    resid = Pb @ (H - tip)
    depth = float((H - tip) @ n)
    shift = -resid + (d_t - depth) * n
    out = dict(bp, source=bp["source"] + " + diag board oracle", pos=(np.asarray(bp["pos"]) + shift).tolist())
    out["diag_board_shift_mm"] = (1000 * shift).round(3).tolist()
    return out


def _inhand_oracle(bp: dict, s: dict, k_rel: int, rp: ReplayParams, depth: float = 0.002):
    """DIAGNOSTIC ONLY: in-hand pose that makes the part stand along the hole axis at the real release, with
    its tip `depth` into the hole: tilt = the hand's tilt at release, grasp height from the TCP height."""
    Rb = quat_to_mat(bp["quat"])
    Rh = quat_to_mat(s["ee"][k_rel, 3:7])
    z_in = -Rb[:, 2]  # insertion direction (into the board)
    zh = Rh[:, 2]
    ax = np.cross(zh, z_in)
    sn = float(np.linalg.norm(ax))
    Rw = np.eye(3)
    if sn > 1e-9:
        ang = math.atan2(sn, float(zh @ z_in))
        a = ax / sn
        Rw = quat_to_mat([math.cos(ang / 2), *(math.sin(ang / 2) * a)])
    R_inhand = Rh.T @ Rw @ Rh  # the world rotation Rw expressed in the hand frame
    tcp_b = Rb.T @ (s["ee"][k_rel, :3] - np.asarray(bp["pos"]))
    below = tcp_b[2] - rp.surface_z + depth  # tip this far below the TCP puts it `depth` into the hole
    g = float(np.clip(0.050 - below, 0.0, 0.03))
    return R_inhand, g


def _diag_setup(m, d, slug: str, s: dict, k0: int, rp: ReplayParams) -> dict:
    """DIAGNOSIS ONLY, at the initial (registered) state before the pre-roll: the part tip and axis in the LOGGED
    TCP frame of the window start (what the registration intends), and the in-hand weld's relative pose."""
    import mujoco
    tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{slug}/tip")
    R0 = quat_to_mat(s["ee"][k0, 3:7])
    x0 = s["ee"][k0, :3]
    out = dict(tip_site=int(tip), tip_reg_tcp=(R0.T @ (d.site_xpos[tip] - x0)).tolist(),
               axis_reg_tcp=(R0.T @ d.site_xmat[tip].reshape(3, 3)[:, 2]).tolist(), lock_eq=-1)
    e = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "diag/inhand_lock")
    if e >= 0:
        b1, b2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        R1 = d.xmat[b1].reshape(3, 3)
        m.eq_data[e, :3] = 0.0
        m.eq_data[e, 3:6] = R1.T @ (d.xpos[b2] - d.xpos[b1])
        qrel = np.zeros(4)
        qneg = np.zeros(4)
        mujoco.mju_negQuat(qneg, d.xquat[b1])
        mujoco.mju_mulQuat(qrel, qneg, d.xquat[b2])
        m.eq_data[e, 6:10] = qrel
        m.eq_data[e, 10] = 1.0
        mujoco.mj_forward(m, d)
        rows = [i for i in range(d.nefc) if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY and d.efc_id[i] == e]
        out["lock_eq"] = int(e)
        out["lock_violation_at_t0"] = float(np.abs(d.efc_pos[rows]).max()) if rows else None
    return out


class _HandOracle:
    """DIAGNOSIS ONLY: joint torques J^T [F; T] that pin the twin TCP to the logged TCP (uses the log after t = 0)."""

    def __init__(self, s: dict, k0: int, k1: int, ts: np.ndarray, rp: ReplayParams, n_board: np.ndarray):
        self.ts = ts
        self.X = s["ee"][k0:k1 + 1, :3].copy()
        self.Q = s["ee"][k0:k1 + 1, 3:7].copy()
        self.V = np.gradient(self.X, ts, axis=0)
        W = np.zeros_like(self.X)
        for i in range(len(ts)):
            a, b = max(0, i - 1), min(len(ts) - 1, i + 1)
            if b > a:
                W[i] = _quat_log(self.Q[a], self.Q[b]) / (ts[b] - ts[a])
        self.W = W
        self.k, self.kr = float(rp.diag_hand_oracle_k), float(rp.diag_hand_oracle_kr)
        if rp.diag_hand_oracle_axes not in ("lateral", "all"):
            raise ValueError("diag_hand_oracle_axes must be lateral | all")
        n = np.asarray(n_board, float)
        self.P = np.eye(3) - np.outer(n, n) if rp.diag_hand_oracle_axes == "lateral" else np.eye(3)
        self.last_F = 0.0

    def tau(self, ctrl, d, t: float) -> np.ndarray:
        x_l, q_l = _interp_targets(self.ts, self.X, self.Q, t)
        i = int(np.clip(np.searchsorted(self.ts, t), 0, len(self.ts) - 1))
        x, q, R = ctrl.arm.pose(d)
        J = ctrl.arm.jac(d)
        v = J @ d.qvel[ctrl.arm.dadr]
        Lam = ctrl.last["Lam"]
        lt, Qt = np.linalg.eigh(0.5 * (Lam[:3, :3] + Lam[:3, :3].T))
        Dt = Qt @ np.diag(2 * np.sqrt(self.k * np.clip(lt, 1e-3, None))) @ Qt.T
        F = self.P @ (self.k * (x_l - x) + Dt @ (self.V[i] - v[:3]))
        T = np.zeros(3)
        if self.kr > 0:
            lr = float(np.linalg.eigvalsh(0.5 * (Lam[3:, 3:] + Lam[3:, 3:].T)).max())
            T = -self.kr * rot_error(q, q_l, R) + 2 * math.sqrt(self.kr * max(lr, 1e-4)) * (self.W[i] - v[3:])
        self.last_F = float(np.linalg.norm(F))
        return J.T @ np.r_[F, T]


def run_replay(ep: int, segment_index: int, rp: ReplayParams | dict | None = None, mode: str = "compliant",
               allowed=("train", "gate"), record_video: dict | None = None, t_end: float | None = None,
               halved_timestep: bool = False) -> dict:
    """Replay one in-hand Insert attempt. Returns outcome, traces and metrics (see module doc)."""
    import mujoco
    from twin.predicates import insertion_state

    guard(ep, allowed)
    rp = rp if isinstance(rp, ReplayParams) else ReplayParams.from_dict(rp)
    if halved_timestep:
        rp = replace(rp, timestep=rp.timestep / 2)
    seg = next((g for g in segments(ep) if g["index"] == int(segment_index)), None)
    if seg is None:
        raise ValueError(f"episode {ep} has no segment {segment_index}")
    info = attempt_info(ep, seg, rp.lead_s)
    if info is None:
        raise ValueError(f"episode {ep} segment {segment_index} ({seg['task']}): fingers open at the start, "
                         "no part in hand (a push), nothing to replay")
    part = info["part"]
    rec = recording_of(ep)
    bp = board_pose_for(rec, rp.board_source, part)
    s = signals(ep)
    f = info["frames"]
    if rp.pad_offset:
        Rb = quat_to_mat(bp["quat"])
        yb = Rb.T @ quat_to_mat(s["ee"][f["start"], 3:7])[:, 1]
        yb[2] = 0.0
        yb /= max(np.linalg.norm(yb), 1e-9)
        bp = dict(bp, pos=(np.asarray(bp["pos"]) + Rb @ (rp.pad_offset * yb)).tolist(),
                  source=bp["source"] + f" + {1000 * rp.pad_offset:+.1f} mm across the pads")
    R_inhand = None
    if rp.registration in ("release_oracle", "full_oracle"):
        bp = _register_release(bp, part, s, f["release"], rp)
    elif rp.registration not in ("none", "inhand"):
        raise ValueError(f"registration must be none | inhand | release_oracle | full_oracle, got {rp.registration!r}")
    if rp.registration == "full_oracle":
        R_inhand, g = _inhand_oracle(bp, s, f["release"], rp)
        rp = replace(rp, grasp_from_top=g)
    reg_info = None
    if rp.registration == "inhand":
        R_inhand, dx, dy, g, reg_info = register_inhand(bp, part, s, f, rp)
        th, ph = reg_info.pop("_angles")
        rp = replace(rp, grasp_dx=dx, grasp_dy=dy, grasp_from_top=g, grasp_rock=th, grasp_pivot=ph)
        if rp.diag_board_oracle:  # DIAGNOSIS ONLY
            bp = _board_oracle(bp, part, s, f, rp, R_inhand, dx, dy, g, reg_info["k_ref"])
            reg_info["diag_board_shift_mm"] = bp["diag_board_shift_mm"]
    k0 = f["start"]
    n_all = len(s["t"])
    if mode == "free":
        k1 = (f["onset"] - 3) if f["onset"] is not None else f["release"]
    else:
        k1 = min(f["release"] + int(round(rp.after_release_s * FPS)), n_all - 1)
    k1 = min(k1, k0 + int(rp.max_s * FPS))
    if t_end is not None:
        k1 = min(k1, k0 + int(round(t_end * FPS)))
    if k1 <= k0 + 3:
        raise ValueError(f"window too short ({k1 - k0} frames)")
    cams = _calib_cameras(rec) if record_video else None
    m, minfo = build_model(part, bp, rp, tcp_pose=(s["ee"][k0, :3], s["ee"][k0, 3:7]), cameras=cams,
                           R_inhand=R_inhand)
    slug = minfo["part"]["slug"]

    d = mujoco.MjData(m)
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
    for i in range(7):
        d.qpos[m.jnt_qposadr[jid(f"fr3_joint{i + 1}")]] = s["q"][k0, i]
    for fn, sgn in zip(FINGERS, (1.0, -1.0)):
        for nm, ref in (("rock", rp.grasp_rock), ("pivot", rp.grasp_pivot)):
            j = jid(f"{fn}/{nm}")
            if j >= 0:
                d.qpos[m.jnt_qposadr[j]] = sgn * ref
    mujoco.mj_forward(m, d)
    dg = _diag_setup(m, d, slug, s, k0, rp) if (rp.diag_trace or rp.diag_inhand_lock) else None  # DIAGNOSIS ONLY
    q_open = close_fingers_on_part(m, d, slug, dy=rp.grasp_dy)
    mujoco.mj_forward(m, d)
    grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper/actuator8")
    base_qpos = d.qpos.copy()

    stiff = mode == "stiff"
    ctrl = CartesianImpedance(m, impedance(rp, stiff=stiff))
    ts = s["t"][k0:k1 + 1] - s["t"][k0]
    tb = tare(s, k0, _part_mass(m, slug)) if rp.tare else None
    if stiff:
        X, Q = s["ee"][k0:k1 + 1, :3].copy(), s["ee"][k0:k1 + 1, 3:7].copy()
    else:
        X, Q = reconstruct_window(m, s, k0, k1, rp, base_qpos, tb)
    sensor = ToolSensor(m, slug)
    n_sub = max(1, int(round(ctrl.p.period / m.opt.timestep)))
    hold_w = info["grip_width_mm"] / 1000.0

    def grip_cmd(k):
        w = float(s["grip"][k].sum())
        return GRIP_CLOSED if w < hold_w + RELEASE_OPEN else w / 2

    # pre-roll (before t = 0): squeeze the part, arm held at the logged start pose
    ctrl.reset(d, s["ee"][k0, :3], s["ee"][k0, 3:7])
    nan = False
    step = 0
    while d.time < rp.preroll_s - 1e-9:
        if step % n_sub == 0:
            d.ctrl[:7] = ctrl.torque(d)
            d.ctrl[grip_act] = GRIP_CLOSED
        mujoco.mj_step(m, d)
        step += 1
    t_pre = float(d.time)
    pre_state = insertion_state(m, d, slug)
    pb_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, slug)
    bd_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "board")
    start_pen = max([0.0] + [-float(d.contact[i].dist) for i in range(d.ncon)
                             if {m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]} == {pb_id, bd_id}])
    grip_ok = pre_state["depth"] < -0.001  # part still above the hole and held

    T = float(ts[-1])
    rec_t, sim_x, sim_q, sim_F, sim_T, ins, held = [], [], [], [], [], [], []
    oracle = None  # DIAGNOSIS ONLY (diag_* fields; off by default)
    if rp.diag_hand_oracle_k > 0 or rp.diag_hand_oracle_kr > 0:
        oracle = _HandOracle(s, k0, k1, ts, rp, quat_to_mat(bp["quat"])[:, 2])
    lock_eq = dg["lock_eq"] if dg is not None else -1
    dtrace = [] if rp.diag_trace else None
    renderer = _Recorder(m, record_video) if record_video else None
    next_rec = 0
    t0w = time.perf_counter()
    step = 0
    while True:
        t = d.time - t_pre
        if next_rec < len(ts) and t >= ts[next_rec] - 1e-9:
            x, q, _ = ctrl.arm.pose(d)
            Fs, Ts = sensor.wrench(d)
            st = insertion_state(m, d, slug)
            rec_t.append(float(ts[next_rec]))
            sim_x.append(x)
            sim_q.append(q)
            sim_F.append(Fs)
            sim_T.append(Ts)
            ins.append([st["depth"], st["lateral"], st["tilt_deg"], st.get("axial_resist", 0.0), float(st["seated"])])
            held.append(float(d.qpos[m.jnt_qposadr[jid("gripper/finger_joint1")]]))
            if dtrace is not None:  # DIAGNOSIS ONLY
                ts_ = dg["tip_site"]
                dtrace.append(np.r_[d.site_xpos[ts_], d.site_xmat[ts_].reshape(3, 3)[:, 2],
                                    0.0 if oracle is None else oracle.last_F,
                                    float(d.eq_active[lock_eq]) if lock_eq >= 0 else -1.0])
            if renderer is not None:
                renderer.capture(d, next_rec)
            next_rec += 1
        if t >= T - 1e-9:
            break
        if step % n_sub == 0:
            xt, qt = _interp_targets(ts, X, Q, t)
            ctrl.set_target(xt, qt, direct=True)
            d.ctrl[:7] = ctrl.torque(d)
            d.ctrl[grip_act] = grip_cmd(min(k0 + int(t * FPS), k1))
            if oracle is not None:  # DIAGNOSIS ONLY
                d.ctrl[:7] = d.ctrl[:7] + oracle.tau(ctrl, d, t)
            if lock_eq >= 0 and d.eq_active[lock_eq] and d.ctrl[grip_act] != GRIP_CLOSED:  # DIAGNOSIS ONLY
                d.eq_active[lock_eq] = 0
        mujoco.mj_step(m, d)
        step += 1
        if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0 or not np.isfinite(d.qpos).all():
            nan = True
            break
    wall = time.perf_counter() - t0w
    if renderer is not None:
        renderer.close()
    st_end = insertion_state(m, d, slug)
    res = _score(info, s, k0, rec_t, np.array(sim_x), np.array(sim_q), np.array(sim_F), np.array(ins),
                 np.array(held), st_end, nan, tb)
    res.update(episode=int(ep), recording=rec, segment_index=int(segment_index), part=part, mode=mode,
               params=rp.as_dict(), board_pose=bp, grip_ok=bool(grip_ok), finger_contact_mm=round(1000 * q_open, 3),
               registration=reg_info, start_part_board_pen_mm=round(1000 * start_pen, 3),
               setpoint=dict(x=X.round(6).tolist(), q=Q.round(6).tolist()),
               timestep=float(m.opt.timestep), wall_s=round(wall, 2), sim_s=round(float(d.time - t_pre), 3),
               window=dict(k0=k0, k1=k1, t0=float(s["t"][k0]), t1=float(s["t"][k1]),
                           release_s=round(float(s["t"][f["release"]] - s["t"][k0]), 3)))
    if reg_info is not None and sim_x:
        i_ref = min(reg_info["k_ref"] - k0, len(sim_x) - 1)
        if i_ref >= 0:
            res["registration"]["tcp_gap_at_ref_mm"] = round(1000 * float(np.linalg.norm(sim_x[i_ref] - s["ee"][k0 + i_ref, :3])), 3)
            res["registration"]["sim_depth_at_ref_mm"] = round(1000 * float(ins[i_ref][0]), 3)
            res["registration"]["sim_lateral_at_ref_mm"] = round(1000 * float(ins[i_ref][1]), 3)
            res["registration"]["sim_tilt_at_ref_deg"] = round(float(ins[i_ref][2]), 3)
    if dtrace is not None:  # DIAGNOSIS ONLY: everything the arrival-error decomposition needs, per log frame
        nr = len(rec_t)
        kk = np.arange(k0, k0 + nr)
        e_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{slug}_hole/entry")
        Fr = np.array([wrench_at(s, k, tb)[0] for k in kk]) if tb is not None else s["F"][kk]
        res["diag"] = dict(
            cols="sim tcp xyz, sim tcp quat wxyz, sim tip xyz, sim part axis xyz, sim F xyz, log tcp xyz, log quat wxyz, "
                 "real F (tared) xyz, oracle |F| N, lock active",
            tip_reg_tcp=np.round(dg["tip_reg_tcp"], 6).tolist(), axis_reg_tcp=np.round(dg["axis_reg_tcp"], 6).tolist(),
            lock_violation_at_t0=dg.get("lock_violation_at_t0"),
            hole_entry=np.round(d.site_xpos[e_id], 6).tolist(),
            hole_axis=np.round(d.site_xmat[e_id].reshape(3, 3)[:, 2], 6).tolist(),
            k_ref_i=None if reg_info is None else int(reg_info["k_ref"] - k0), release_i=int(f["release"] - k0),
            onset_real_i=None if f["onset"] is None else int(f["onset"] - k0),
            rows=np.round(np.column_stack([np.array(sim_x), np.array(sim_q), np.array(dtrace)[:, :6], np.array(sim_F),
                                           s["ee"][kk, :3], s["ee"][kk, 3:7], Fr, np.array(dtrace)[:, 6:8]]), 6).tolist())
    if renderer is not None:
        res["video_frames"] = renderer.files
        src = {v.get("_source") for v in (cams or {}).values()}
        res["camera_source"] = (f"twin/calib/{sorted(src)[0]}" if src else "base.xml placeholder cameras") + (
            "" if f"{rec}.json" in src else " (calibrated on another recording of the same day: check with render_overlay)")
    return res


def _score(info, s, k0, rec_t, X, Qs, F, ins, held, st_end, nan, tb) -> dict:
    n = len(rec_t)
    k = np.arange(k0, k0 + n)
    real_x = s["ee"][k, :3]
    real_q = s["ee"][k, 3:7]
    nref = min(TARE_FRAMES, n)
    dF_real = s["F"][k] - np.median(s["F"][k0:k0 + nref], axis=0)
    dF_sim = F - np.median(F[:nref], axis=0) if n else F
    pos_err = np.linalg.norm(X - real_x, axis=1) if n else np.zeros(0)
    ang_err = np.array([np.degrees(2 * np.arcsin(min(1.0, float(np.linalg.norm(rot_error(Qs[i], real_q[i]))))))
                        for i in range(n)]) if n else np.zeros(0)
    Fm_sim = np.linalg.norm(dF_sim, axis=1) if n else np.zeros(0)
    Fm_real = np.linalg.norm(dF_real, axis=1)
    rel = int(info["frames"]["release"] - k0)
    held_n = min(n, rel + 1)
    on_sim = np.flatnonzero(Fm_sim[:held_n] > CONTACT_N)
    on_real = np.flatnonzero(Fm_real[:held_n] > CONTACT_N)
    t_on_sim = float(rec_t[on_sim[0]]) if len(on_sim) else None
    t_on_real = float(rec_t[on_real[0]]) if len(on_real) else None
    seated = bool(st_end["seated"]) if not nan else False
    pk_sim = float(Fm_sim[:held_n].max()) if held_n else 0.0
    pk_real = float(Fm_real[:held_n].max()) if held_n else 0.0
    at_rel = ins[min(rel, n - 1)] if n else [np.nan] * 5
    out = dict(
        outcome="nan" if nan else ("seated" if seated else "not_seated"),
        sim_success=seated, real_success=bool(info["success"]), match=(seated == bool(info["success"])) and not nan,
        final=dict(depth_mm=round(1000 * st_end["depth"], 3), lateral_mm=round(1000 * st_end["lateral"], 3),
                   tilt_deg=round(st_end["tilt_deg"], 3), seat_depth_mm=round(1000 * st_end["seat_depth"], 3)),
        at_release=dict(depth_mm=round(1000 * float(at_rel[0]), 3), lateral_mm=round(1000 * float(at_rel[1]), 3),
                        tilt_deg=round(float(at_rel[2]), 3)),
        max_depth_mm=round(1000 * float(ins[:, 0].max()), 3) if n else None,
        force=dict(peak_sim_N=round(pk_sim, 3), peak_real_N=round(pk_real, 3),
                   peak_rel_err=round(abs(pk_sim - pk_real) / max(pk_real, 1e-6), 4),
                   onset_sim_s=t_on_sim, onset_real_s=t_on_real,
                   onset_err_s=None if (t_on_sim is None or t_on_real is None) else round(abs(t_on_sim - t_on_real), 3)),
        tracking=dict(pos_rmse_mm=round(1000 * float(np.sqrt(np.mean(pos_err ** 2))), 3) if n else None,
                      pos_max_mm=round(1000 * float(pos_err.max()), 3) if n else None,
                      ang_rmse_deg=round(float(np.sqrt(np.mean(ang_err ** 2))), 3) if n else None,
                      pre_contact_rmse_mm=round(1000 * float(np.sqrt(np.mean(pos_err[:max(1, (on_real[0] if len(on_real) else n))] ** 2))), 3) if n else None),
        attempt=info,
        traces=dict(cols="t, sim |dF| N, real |dF| N, sim dFz, real dFz, sim z mm, real z mm, depth mm, lateral mm, tilt deg, finger mm",
                    rows=[[round(rec_t[i], 4), round(float(Fm_sim[i]), 3), round(float(Fm_real[i]), 3),
                           round(float(dF_sim[i, 2]), 3), round(float(dF_real[i, 2]), 3),
                           round(1000 * float(X[i, 2]), 3), round(1000 * float(real_x[i, 2]), 3),
                           round(1000 * float(ins[i, 0]), 3), round(1000 * float(ins[i, 1]), 3),
                           round(float(ins[i, 2]), 3), round(1000 * float(held[i]), 3)] for i in range(n)]),
    )
    return out


# ---------------------------------------------------------------------------------------- render
def _calib_cameras(recording: str) -> dict:
    """Calibrated cameras for this recording, else the nearest calibrated recording (cameras did not move
    within a day; say which one was used)."""
    files = sorted(CALIB.glob("*.json"))
    own = CALIB / f"{recording}.json"
    src = own if own.exists() else (files[0] if files else None)
    if src is None:
        return {}
    c = json.loads(src.read_text())
    cams = {k: v for k, v in c.get("cameras", {}).items()}
    for v in cams.values():
        v["_source"] = src.name
    return cams


class _Recorder:
    def __init__(self, m, spec: dict):
        import os
        os.environ.setdefault("MUJOCO_GL", "glfw")
        import mujoco
        self.m = m
        self.cams = spec.get("cameras", ["hama1"])
        self.out = Path(spec["dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.r = mujoco.Renderer(m, spec.get("height", 480), spec.get("width", 640))
        self.opt = mujoco.MjvOption()
        self.files = {c: [] for c in self.cams}

    def capture(self, d, i: int):
        from PIL import Image
        for c in self.cams:
            self.r.update_scene(d, camera=c, scene_option=self.opt)
            p = self.out / f"{c}_{i:04d}.png"
            Image.fromarray(self.r.render()).save(p)
            self.files[c].append(str(p))

    def close(self):
        self.r.close()


# ------------------------------------------------------------------------------- stiffness bracket
def setpoint_roughness(ep: int, segment_index: int, k_list=(200.0, 500.0, 800.0), kr_list=None,
                       cutoff_hz: float = 3.0, allowed=("train", "gate"), rp: ReplayParams | None = None) -> dict:
    """How rough is the rebuilt setpoint x_d(K) during contact, for each candidate stiffness?

    The real x_d is the haptic target after a first-order filter with a 0.2 s time constant
    (filter_params_ = 0.005 at 1 kHz), so it carries little energy above ~3 Hz. Contact makes the measured
    pose and force jump; only the right K cancels those jumps in x - F/K (+ D v + Lambda a). Returns the
    RMS of the high-passed x_d (mm) and of the rotation target (deg) per K over the contact phase."""
    from scipy.signal import butter, filtfilt
    from twin.control import ArmModel, stiffness_matrix  # noqa: F401
    guard(ep, allowed)
    rp = rp or ReplayParams()
    seg = next(g for g in segments(ep) if g["index"] == int(segment_index))
    info = attempt_info(ep, seg, rp.lead_s)
    if info is None or info["frames"]["onset"] is None:
        raise ValueError("needs an in-hand attempt with contact")
    s = signals(ep)
    f = info["frames"]
    k0, k1 = f["start"], f["release"]
    kc0 = max(k0, f["onset"] - 15)
    m, minfo = build_model(info["part"], board_pose_for(recording_of(ep), rp.board_source, info["part"]), rp)
    tb = tare(s, k0, _part_mass(m, minfo["part"]["slug"]))
    b, a = butter(2, cutoff_hz / (FPS / 2), "high")
    kr_list = kr_list or [{200.0: 10.0, 500.0: 50.0, 800.0: 75.0}.get(float(k), 50.0) for k in k_list]
    out = []
    for K, Kr in zip(k_list, kr_list):
        X, Q = reconstruct_window(m, s, k0, k1, replace(rp, k_trans=float(K), k_rot=float(Kr)), None, tb)
        if len(X) < 20:
            raise ValueError("window too short for the high-pass")
        hx = filtfilt(b, a, X, axis=0)[kc0 - k0:]
        q0 = Q[0]
        rv = np.array([_quat_log(q0, q) for q in Q])
        hr = filtfilt(b, a, rv, axis=0)[kc0 - k0:]
        out.append(dict(k_trans=float(K), k_rot=float(Kr), hf_pos_mm=round(1000 * float(np.sqrt(np.mean(np.sum(hx ** 2, 1)))), 4),
                        hf_rot_deg=round(float(np.degrees(np.sqrt(np.mean(np.sum(hr ** 2, 1))))), 4)))
    xs = s["ee"][kc0:k1 + 1, :3]
    hraw = filtfilt(b, a, s["ee"][k0:k1 + 1, :3], axis=0)[kc0 - k0:]
    best = min(out, key=lambda r: r["hf_pos_mm"])
    best_r = min(out, key=lambda r: r["hf_rot_deg"])
    return dict(episode=int(ep), segment_index=int(segment_index), part=info["part"], success=info["success"],
                contact_frames=int(k1 - kc0 + 1), cutoff_hz=cutoff_hz, per_k=out,
                hf_measured_pose_mm=round(1000 * float(np.sqrt(np.mean(np.sum(hraw ** 2, 1)))), 4),
                peak_dF=info["real_peak_dF"], best_k_trans=best["k_trans"], best_k_rot=best_r["k_rot"], n=len(xs))


def _quat_log(q0, q) -> np.ndarray:
    """Rotation vector of q relative to q0 (world frame), rad."""
    from twin.control import quat_conj, quat_mul
    q = np.asarray(q, float)
    if float(q @ q0) < 0:
        q = -q
    qe = quat_mul(q, quat_conj(q0))
    v = qe[1:]
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3)
    return v / n * 2 * math.atan2(n, qe[0])


def release_events(ep: int, min_drop: float = 3.0, allowed=("train", "gate"), max_events: int = 50) -> list[dict]:
    """Sudden contact releases in an episode (|F| drops by > min_drop N within 2 frames, then stays low)
    and the arm's jump toward its unseen equilibrium. For each: the drop dF, the measured displacement
    along dF over the next 0.4 s (static estimate K = |dF| / |dx|), and the critically damped rise fit
    x(t) = A (1 - (1 + w t) e^(-w t)) giving w and K_dyn = w^2 m_eff, m_eff = 1 / (u^T Lambda^-1 u)."""
    from scipy.optimize import least_squares
    from twin.control import ArmModel
    guard(ep, allowed)
    s = signals(ep)
    rp = ReplayParams()
    m, _ = build_model("round peg 1", board_pose_for(recording_of(ep), "mocap"), rp)
    arm = ArmModel(m)
    Fm = np.linalg.norm(s["F"], axis=1)
    out = []
    i = 5
    n = len(Fm)
    while i < n - 15 and len(out) < max_events:
        pre = Fm[i - 1]
        if pre > min_drop + 0.5 and pre - Fm[i + 1] > min_drop and Fm[i + 1:i + 12].max() < 1.5:
            dF = s["F"][i + 1] - s["F"][i - 1]
            u = -dF / np.linalg.norm(dF)  # the arm moves along the force it was resisting: F on tool was +dF
            u = s["F"][i - 1] - s["F"][i + 5]
            u = -u / np.linalg.norm(u)
            x = s["ee"][i - 1:i + 13, :3] @ u
            x = x - x[0]
            t = np.arange(len(x)) / FPS
            d = arm.state_at(s["q"][i - 1])
            J = arm.jac(d)
            Mi = np.linalg.inv(arm.mass(d))
            Lt_inv = (J @ Mi @ J.T)[:3, :3]
            meff = 1.0 / float(u @ Lt_inv @ u)
            res = least_squares(lambda p: p[0] * (1 - (1 + p[1] * (t - p[2])) * np.exp(-p[1] * np.clip(t - p[2], 0, None)))
                                * (t >= p[2]) - x, x0=[x[-1] if abs(x[-1]) > 1e-4 else 0.005, 15.0, 0.02],
                                bounds=([-0.1, 2.0, 0.0], [0.1, 80.0, 0.07]))
            A, w, t0 = res.x
            Fdrop = float(np.linalg.norm(s["F"][i - 1] - s["F"][i + 5]))
            out.append(dict(t=round(float(s["t"][i]), 3), drop_N=round(Fdrop, 2), jump_mm=round(1000 * A, 3),
                            K_static=round(Fdrop / abs(A), 1) if abs(A) > 2e-4 else None,
                            omega=round(float(w), 2), m_eff=round(meff, 3), K_dynamic=round(float(w ** 2 * meff), 1),
                            fit_rms_mm=round(1000 * float(np.sqrt(np.mean(res.fun ** 2))), 3),
                            seg=int(s["seg"][i]), task=_tasks()[int(s["task"][i])]))
            i += 15
        else:
            i += 1
    return out


# ------------------------------------------------------------------------------------- evaluation
def _eval_job(args):
    ep, seg, params, mode = args
    try:
        r = run_replay(ep, seg, params, mode=mode)
        r.pop("setpoint", None)
        return r
    except Exception as e:  # report, don't crash a batch
        return dict(episode=ep, segment_index=seg, error=f"{type(e).__name__}: {e}")


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    tp = sum(1 for r in ok if r["sim_success"] and r["real_success"])
    tn = sum(1 for r in ok if not r["sim_success"] and not r["real_success"])
    P = sum(1 for r in ok if r["real_success"])
    N = sum(1 for r in ok if not r["real_success"])
    tpr = tp / P if P else float("nan")
    tnr = tn / N if N else float("nan")
    pk = [r["force"]["peak_rel_err"] for r in ok if r["force"]["peak_real_N"] > CONTACT_N]
    on = [r["force"]["onset_err_s"] for r in ok if r["force"]["onset_err_s"] is not None]
    no_on = sum(1 for r in ok if r["force"]["onset_real_s"] is not None and r["force"]["onset_sim_s"] is None)
    return dict(n=len(ok), errors=len(results) - len(ok), real_success=P, real_failure=N,
                sim_success=sum(1 for r in ok if r["sim_success"]),
                tpr=round(tpr, 3), tnr=round(tnr, 3), bacc=round((tpr + tnr) / 2, 3),
                force_peak_rel_err_median=round(float(np.median(pk)), 3) if pk else None,
                onset_err_s_median=round(float(np.median(on)), 3) if on else None,
                onset_missing_in_sim=no_on,
                pre_contact_rmse_mm_median=round(float(np.median([r["tracking"]["pre_contact_rmse_mm"] for r in ok])), 3) if ok else None,
                nan=sum(1 for r in ok if r["outcome"] == "nan"))


def evaluate(attempts: list[dict], params: dict | None = None, mode: str = "compliant", workers: int = 15) -> dict:
    """Replay many attempts (train/gate only) in parallel processes; per-attempt results + summary."""
    from multiprocessing import get_context
    jobs = [(a["episode"], a["segment_index"], dict(params or {}), mode) for a in attempts]
    for ep, *_ in jobs:
        guard(ep)
    if workers <= 1:
        res = [_eval_job(j) for j in jobs]
    else:
        with get_context("fork").Pool(min(workers, len(jobs))) as p:
            res = p.map(_eval_job, jobs, chunksize=1)
    return dict(summary=summarize(res), results=res)


# ---------------------------------------------------------------------------------- geometry probe
def _probe_job(args):
    ep, seg, params = args
    try:
        r = run_replay(ep, seg, params, mode="stiff")
        rows = np.array(r["traces"]["rows"], float)
        rel = int(round(r["window"]["release_s"] * FPS))
        held = rows[:rel + 1]
        pre = rows[max(0, rel - 15):rel + 1]  # the last 0.5 s before the real release
        return dict(offset_mm=[round(1000 * v, 2) for v in params["board_offset"][:2]],
                    peak_sim_N=r["force"]["peak_sim_N"], peak_real_N=r["force"]["peak_real_N"],
                    pre_release_sim_N=round(float(np.median(pre[:, 1])), 3),
                    pre_release_real_N=round(float(np.median(pre[:, 2])), 3),
                    median_sim_N_in_real_contact=round(float(np.median(held[held[:, 2] > CONTACT_N, 1])), 3)
                    if (held[:, 2] > CONTACT_N).any() else None,
                    max_depth_mm=r["max_depth_mm"], outcome=r["outcome"])
    except Exception as e:
        return dict(offset_mm=list(params.get("board_offset", (0, 0))[:2]), error=f"{type(e).__name__}: {e}")


def geometry_probe(ep: int, segment_index: int, offsets_mm=(0.0, 1.0, 2.0, 3.0), directions=("+x", "-x", "+y", "-y"),
                   params: dict | None = None, workers: int = 15, allowed=("train", "gate")) -> dict:
    """Stiff kinematic replay (the twin's arm tracks the LOGGED pose with stiff_k) with the board shifted in its
    own plane. The real part followed this path without crashing, so the force the twin produces measures
    how far its hole is from where the real part went. Geometry probe only, never a validation score."""
    from multiprocessing import get_context
    guard(ep, allowed)
    base = dict(params or {})
    jobs = []
    for o in offsets_mm:
        for dname in (directions if o > 0 else directions[:1]):
            sgn = 1.0 if dname[0] == "+" else -1.0
            v = [0.0, 0.0, 0.0]
            v["xy".index(dname[1])] = sgn * o / 1000.0
            p = dict(base, board_offset=tuple(np.add(base.get("board_offset", (0, 0, 0)), v)))
            jobs.append((ep, segment_index, p))
    with get_context("fork").Pool(min(workers, len(jobs))) as pool:
        res = pool.map(_probe_job, jobs, chunksize=1)
    by_off = {}
    for (_, _, p), r in zip(jobs, res):
        o = round(float(np.linalg.norm(np.asarray(p["board_offset"][:2]) - np.asarray(base.get("board_offset", (0, 0, 0))[:2]))) * 1000, 2)
        by_off.setdefault(o, []).append(r)
    summary = {str(o): dict(n=len(v), peak_sim_N_median=round(float(np.median([x["peak_sim_N"] for x in v if "peak_sim_N" in x])), 2),
                            pre_release_sim_N_median=round(float(np.median([x["pre_release_sim_N"] for x in v if "pre_release_sim_N" in x])), 2),
                            pre_release_real_N=v[0].get("pre_release_real_N"))
               for o, v in sorted(by_off.items())}
    return dict(episode=int(ep), segment_index=int(segment_index), stiff_k=ReplayParams.from_dict(base).stiff_k,
                summary_by_offset_mm=summary, runs=res)
