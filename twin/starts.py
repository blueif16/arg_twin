"""Frozen start conditions for peg-insertion practice (stage D) and held-out evaluation (D/E).

A start puts the part in the hand above its hole. Everything uncertain about that moment is drawn from what the
REASSEMBLE TRAIN recordings measured (playbook/splits.json; gate and exam are never read here):

  board_err  true board minus the nominal (believed) board, board frame: dx, dy from the leave-one-recording-out
             across-pad hole errors of board-from-contacts (board_err_train.json, every part, random sign, each axis
             drawn independently); dz ~ N(0, s_z) with s_z = median per-recording contact-fit z sigma (train), +-2 s_z.
  inhand     where the part sits in the fingers at the grasp: dx along the pads (hand x), dy across them (hand y),
             grasp_from_top (pad centre below the part's top end), pivot (about hand y, the closing axis) and rock
             (about hand x). Registered from every TRAIN Remove grasp of a peg: the part is seated vertically in its
             hole when the fingers close on it, so the hand pose relative to the hole axis IS the in-hand pose at
             grasp time (same registration as twin/replay.py register_inhand, applied to a known-vertical part).
             One whole event is drawn per start (keeps the correlations).
  hand       the TCP pose at the replay window start (contact onset - 1.5 s) of every TRAIN in-hand peg Insert
             attempt, relative to that recording's contact-fit hole: offset (entry frame: x, y lateral, z along the
             insertion axis, negative = above) and the hand rotation relative to the entry frame.

The nominal cell is one TRAIN recording's contact-fit board (NOMINAL_RECORDING, the recording with calibrated cameras):
its pose and hole xy are copied into the manifest, so later board refits cannot change a frozen start set. The actor is
told the nominal hole pose and the nominal grasp (dx = dy = 0, grasp_from_top = 9 mm, part along hand z), never the
start's offsets.

Sets (frozen separately, each with its own seed, file, sha256 in manifest.json):
  practice  P-<slug>-NNN   20 per part      for practice; outcomes are shown to the actor
  heldout   H-<slug>-NNN   46 per part      for D.success_rate / E.routine_success: >= 46 allows one failure at a
                                            >= 0.9 claim (validation.yaml header); the first 29 per part are a
                                            pre-declared subset (zero failures allowed) for expensive brains
  redteam   R-<slug>-NNN   4 per part       the only starts privileged red-team scripts may use

    .venv/bin/python -I -c "import sys; sys.path.insert(0, '.'); from twin.starts import main; main()"
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from twin import REPO_DIR

RUN_DIR = REPO_DIR / "runs" / "claude" / "run-standin-1"
OUT_DIR = RUN_DIR / "starts"
BOARD_DIR = RUN_DIR / "board"
NOMINAL_RECORDING = "2025-01-09-14-11-58"  # episode 0 (train): cameras calibrated (twin/calib), 15-part contact fit
PEGS = [f"round peg {i}" for i in range(1, 5)] + [f"square peg {i}" for i in range(1, 5)]
SETS = {"practice": dict(prefix="P", n=20, seed=20261024_1), "heldout": dict(prefix="H", n=46, seed=20261024_2),
        "redteam": dict(prefix="R", n=4, seed=20261024_3)}
HELDOUT_SUBSET = 29  # first 29 per part: pre-declared smaller held-out set (0 failures for >= 0.9)
SURFACE_Z = 0.0303   # plate top above the board frame z (twin/board.py surface_z, twin/replay.py ReplayParams)
SEAT_DEPTH = 0.020   # twin/board.py seat_depth
NOMINAL_GRASP_FROM_TOP = 0.009  # twin/parts.py grasp_from_top (onset-timing identification)
TIP_CLEAR_MIN = 0.003  # the tip must start at least 3 mm above the true surface ...
TIP_CLEAR_MAX = 0.080  # ... and at most 80 mm
PAD_DX_MAX = 0.0065    # in-hand offset along the pads: menagerie pad half-width 8.5 mm minus 2 mm
PAD_DY_MAX = 0.002     # across the pads (finger mid-plane offset)


def _rel(p) -> str:
    return str(Path(p).resolve().relative_to(REPO_DIR))


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def quat_mat(q):
    from twin.control import quat_to_mat
    return quat_to_mat(q)


def mat_quat(R):
    from twin.control import mat_to_quat
    return mat_to_quat(R)


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# --------------------------------------------------------------------------------------------- nominal cell
def nominal_board(recording: str = NOMINAL_RECORDING) -> dict:
    """Contact-fit board of the nominal recording (x, y, yaw from contacts; z from mocap, as twin.replay does)."""
    from twin.replay import POSES
    f = BOARD_DIR / f"{recording}.json"
    c = json.loads(f.read_text())
    raw = json.loads((POSES / f"{recording}_poses.json").read_text())["NIST_Board1"]
    holes = {p: list(map(float, c["holes"][p]["board_xy"])) for p in PEGS if p in c["holes"]}
    return dict(recording=recording, pos=[float(c["pose"]["pos"][0]), float(c["pose"]["pos"][1]), float(raw[0][2])],
                quat=[float(v) for v in c["pose"]["quat_wxyz"]], hole_xy=holes, surface_z=SURFACE_Z,
                seat_depth=SEAT_DEPTH, source=f"{_rel(f)} (x, y, yaw, hole xy) + mocap z ({recording}_poses.json)",
                sha256=sha256_file(f))


def entry_pose(part: str, board: dict, err: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """World position and rotation (columns x_e, y_e, z_e; z_e into the hole) of `part`'s hole entry, for a board
    dict {pos, quat, hole_xy, surface_z} optionally shifted by err {dx, dy, dz} in the board frame."""
    from twin.board import entry_frame
    from twin.parts import load_layout, part_spec
    ps = part_spec(part, load_layout())
    Rb = quat_mat(board["quat"])
    hx, hy = board["hole_xy"][part]
    e = err or {}
    local = np.array([hx + e.get("dx", 0.0), hy + e.get("dy", 0.0), board["surface_z"] + e.get("dz", 0.0)])
    Re, _ = entry_frame(ps)
    return np.asarray(board["pos"], float) + Rb @ local, Rb @ Re


def true_board(board: dict, err: dict) -> dict:
    """Board pose dict (twin.replay.build_model format) with the start's error applied in the board frame."""
    Rb = quat_mat(board["quat"])
    pos = np.asarray(board["pos"], float) + Rb @ np.array([err.get("dx", 0.0), err.get("dy", 0.0), err.get("dz", 0.0)])
    return dict(pos=pos.tolist(), quat=list(board["quat"]), source="nominal board + start board_err")


# ------------------------------------------------------------------------------------------------ IK
def solve_ik(m, q_init, pos, quat, iters: int = 300, tol: float = 1e-7) -> tuple[np.ndarray, dict]:
    """Damped least squares on the 7 FR3 joints for the 'tcp' site. Returns (q7, info: ok, err_mm, err_deg)."""
    import mujoco
    from twin.control import ArmModel, rot_error
    arm = ArmModel(m)
    d = mujoco.MjData(m)
    d.qpos[arm.qadr] = q_init
    lo = m.jnt_range[[m.dof_jntid[i] for i in arm.dadr], 0]
    hi = m.jnt_range[[m.dof_jntid[i] for i in arm.dadr], 1]
    pos = np.asarray(pos, float)
    quat = np.asarray(quat, float)
    ep = er = np.ones(3)
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        x, q, R = arm.pose(d)
        ep = pos - x
        er = -rot_error(q, quat, R)
        if np.linalg.norm(ep) < tol and np.linalg.norm(er) < tol:
            break
        J = arm.jac(d)
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), np.r_[ep, 2 * er])
        d.qpos[arm.qadr] = np.clip(d.qpos[arm.qadr] + dq, lo, hi)
    q7 = d.qpos[arm.qadr].copy()
    margin = float(np.min(np.minimum(q7 - lo, hi - q7)))
    ok = bool(np.linalg.norm(ep) < 1e-5 and np.linalg.norm(er) < 1e-5 and margin > 0.05)
    return q7, dict(ok=ok, err_mm=round(1000 * float(np.linalg.norm(ep)), 6),
                    err_deg=round(math.degrees(2 * float(np.linalg.norm(er))), 6), limit_margin_rad=round(margin, 4))


_ARM_MODEL = None


def arm_model():
    global _ARM_MODEL
    if _ARM_MODEL is None:
        import mujoco
        from twin import BASE_XML
        _ARM_MODEL = mujoco.MjModel.from_xml_path(str(BASE_XML))
    return _ARM_MODEL


# ------------------------------------------------------------------------------------- measured distributions
def _train_episodes() -> list[int]:
    from twin.replay import splits
    return sorted(int(k) for k, v in splits().items() if v["split"] == "train")


def _board_for(recording: str, part: str) -> dict | None:
    f = BOARD_DIR / f"{recording}.json"
    if not f.exists():
        return None
    c = json.loads(f.read_text())
    if not c.get("pose") or part not in c.get("holes", {}):
        return None
    from twin.replay import POSES
    raw = json.loads((POSES / f"{recording}_poses.json").read_text())["NIST_Board1"]
    return dict(pos=[c["pose"]["pos"][0], c["pose"]["pos"][1], raw[0][2]], quat=c["pose"]["quat_wxyz"],
                hole_xy={part: c["holes"][part]["board_xy"]}, surface_z=SURFACE_Z)


def _inhand_from_grasp(tcp, q_tcp, H, Rw, length) -> dict:
    """In-hand pose of a part seated vertically in its hole (axis = entry z) grasped at TCP pose (tcp, q_tcp)."""
    from twin.replay import PAD_TO_TCP
    Rh = quat_mat(q_tcp)
    z_in = Rw[:, 2]
    v = Rh.T @ z_in
    phi = math.atan2(v[0], v[2])                       # pivot about hand y (twin.replay.register_inhand)
    theta = math.atan2(-v[1], math.hypot(v[0], v[2]))  # rock about hand x
    p = np.asarray(tcp) - Rh[:, 2] * PAD_TO_TCP          # pad centre
    tip = H + z_in * SEAT_DEPTH
    below = float((tip - p) @ z_in)
    Pb = np.eye(3) - np.outer(z_in, z_in)
    A = np.column_stack([Pb @ Rh[:, 0], Pb @ Rh[:, 1]])
    sol, *_ = np.linalg.lstsq(A, Pb @ (H - p), rcond=None)
    return dict(dx=float(sol[0]), dy=float(sol[1]), grasp_from_top=float(length - below), pivot=phi, rock=theta)


def measure(out_dir: Path = OUT_DIR) -> dict:
    """Measure the three distributions from TRAIN data and write distributions.json."""
    from twin.board_fit import extract_events
    from twin.parts import load_layout, part_spec
    from twin.replay import attempt_info, recording_of, segments, signals, split_of
    lay = load_layout()
    inhand, hand = [], []
    for ep in _train_episodes():
        assert split_of(ep) == "train"
        rec = recording_of(ep)
        try:
            evs = extract_events(ep)
        except FileNotFoundError:
            continue
        try:
            s = signals(ep)
        except FileNotFoundError:
            continue
        # in-hand at grasp: Remove grasps of a seated peg
        for e in evs:
            if e["kind"] != "remove_grasp" or e["part"] not in PEGS:
                continue
            b = _board_for(rec, e["part"])
            if b is None:
                continue
            H, Rw = entry_pose(e["part"], b)
            k = int(e["frame"])
            ih = _inhand_from_grasp(s["ee"][k, :3], s["ee"][k, 3:7], H, Rw, part_spec(e["part"], lay).length)
            ih.update(episode=ep, part=e["part"], t=e["t"])
            inhand.append(ih)
        # hand at the replay window start of in-hand Insert attempts
        for seg in segments(ep):
            if not seg["task"].startswith("Insert ") or seg["task"][7:].rstrip(".") not in PEGS:
                continue
            a = attempt_info(ep, seg)
            if a is None:
                continue
            b = _board_for(rec, a["part"])
            if b is None:
                continue
            H, Rw = entry_pose(a["part"], b)
            k = a["frames"]["start"]
            off = Rw.T @ (s["ee"][k, :3] - H)
            Rrel = Rw.T @ quat_mat(s["ee"][k, 3:7])
            hand.append(dict(episode=ep, part=a["part"], segment_index=seg["index"], success=bool(a["success"]),
                             offset=off.tolist(), R_rel_quat=mat_quat(Rrel).tolist(),
                             tilt_deg=float(np.degrees(np.arccos(np.clip(Rrel[2, 2], -1, 1))))))
    # board error: leave-one-recording-out across-pad errors on TRAIN (board_err_train.json)
    be = json.loads((BOARD_DIR / "board_err_train.json").read_text())
    close = [float(v["contact_close_mm"]) for r in be["recordings"] if r.get("split") == "train"
             for v in (r.get("lopo") or {}).values() if v.get("contact_close_mm") is not None]
    zsig = [float(r["full_fit"]["sigma"]["z_mm"]) for r in be["recordings"] if r.get("split") == "train"
            and r.get("full_fit") and r["full_fit"]["sigma"].get("z_mm") is not None]

    def summ(vals, scale=1.0):
        v = np.asarray(vals, float) * scale
        return dict(n=int(len(v)), median=round(float(np.median(v)), 3),
                    mad=round(float(np.median(np.abs(v - np.median(v)))), 3),
                    p10=round(float(np.percentile(v, 10)), 3), p90=round(float(np.percentile(v, 90)), 3))

    out = dict(
        generated_by="twin/starts.py measure", generated_at=time.strftime("%Y-%m-%d %H:%M"),
        split="train only (playbook/splits.json)",
        board_err=dict(across_pad_abs_mm=close, z_sigma_mm=round(float(np.median(zsig)), 3),
                       source=f"{_rel(BOARD_DIR / 'board_err_train.json')}: per recording-part leave-one-part-out "
                              "contact_close_mm (layout refitted without the recording); z: median full_fit sigma z_mm",
                       summary=summ(close)),
        inhand=dict(rows=inhand, source="TRAIN Remove grasps of pegs (twin/board_fit.py extract_events), registered "
                                        "against the recording's contact-fit hole with the part seated vertically "
                                        f"(seat depth {SEAT_DEPTH * 1000:.0f} mm, surface +{SURFACE_Z * 1000:.1f} mm)",
                    summary=dict(dx_mm=summ([r["dx"] for r in inhand], 1000), dy_mm=summ([r["dy"] for r in inhand], 1000),
                                 grasp_from_top_mm=summ([r["grasp_from_top"] for r in inhand], 1000),
                                 pivot_deg=summ([math.degrees(r["pivot"]) for r in inhand]),
                                 rock_deg=summ([math.degrees(r["rock"]) for r in inhand]))),
        hand=dict(rows=hand, source="TRAIN in-hand peg Insert attempts (twin.replay attempt_info): TCP at the window "
                                    "start (contact onset - 1.5 s) relative to the contact-fit hole entry frame",
                  summary=dict(offset_x_mm=summ([r["offset"][0] for r in hand], 1000),
                               offset_y_mm=summ([r["offset"][1] for r in hand], 1000),
                               offset_z_mm=summ([r["offset"][2] for r in hand], 1000),
                               tilt_deg=summ([r["tilt_deg"] for r in hand]))))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "distributions.json").write_text(json.dumps(out, indent=0))
    return out


# --------------------------------------------------------------------------------------------- generation
def _tip_height(part: str, board: dict, err: dict, hand_row: dict, ih: dict) -> float:
    """Height of the part tip above the TRUE surface along the board normal, for a candidate start."""
    from twin.parts import load_layout, part_spec
    from twin.replay import PAD_TO_TCP
    ps = part_spec(part, load_layout())
    Hn, Rwn = entry_pose(part, board)
    tcp = Hn + Rwn @ np.asarray(hand_row["offset"])
    Rh = Rwn @ quat_mat(hand_row["R_rel_quat"])
    Rin = rx(ih["rock"]) @ ry(ih["pivot"])
    a = Rh @ Rin @ np.array([0.0, 0.0, 1.0])
    grasp = tcp + Rh @ np.array([ih["dx"], ih["dy"], -PAD_TO_TCP])
    tip = grasp + a * (ps.length - ih["grasp_from_top"])
    Ht, Rwt = entry_pose(part, board, err)
    return float(-(tip - Ht) @ Rwt[:, 2])


def draw_board_err(rng, dist: dict) -> dict:
    close = np.asarray(dist["board_err"]["across_pad_abs_mm"], float) / 1000
    sz = dist["board_err"]["z_sigma_mm"] / 1000
    dx, dy = rng.choice(close, 2) * rng.choice([-1.0, 1.0], 2)
    dz = float(np.clip(rng.normal(), -2, 2) * sz)
    return dict(dx=float(dx), dy=float(dy), dz=dz)


def draw_inhand(rng, dist: dict) -> dict:
    """One whole Remove-grasp event (dx, dy, pivot, rock). grasp_from_top is NOT taken from the remove grasp: those
    grip the exposed 30 mm of a seated part (17 mm below the top, MAD 4 mm), while insert grasps come from the pick.
    It is drawn from twin/parts.py grasp_from_top P(9 mm, sigma 5 mm, clip +-2 sigma, 2-30 mm), identified from
    insert contact-onset timing (replay-validate).
    Events with |dx| > PAD_DX_MAX or |dy| > PAD_DY_MAX are redrawn, not clipped: the twin's pad (menagerie hand,
    17 mm wide) cannot hold a part there, so those registrations are noise or edge grasps."""
    from twin.parts import load_layout, part_spec
    rows = [r for r in dist["inhand"]["rows"] if abs(r["dx"]) <= PAD_DX_MAX and abs(r["dy"]) <= PAD_DY_MAX]
    r = rows[int(rng.integers(len(rows)))]
    gp = part_spec("round peg 1", load_layout()).params["grasp_from_top"]
    g = float(np.clip(gp.value + float(np.clip(rng.normal(), -2, 2)) * gp.sigma, gp.lo, gp.hi))
    return dict(dx=float(r["dx"]), dy=float(r["dy"]), grasp_from_top=g, pivot=float(r["pivot"]),
                rock=float(r["rock"]),
                src=f"remove grasp ep {r['episode']} {r['part']} t={r['t']}; grasp_from_top from twin/parts.py prior")


HAND_LATERAL_MAX = 0.050  # "above its hole": window starts more than 50 mm off the hole axis are not used ...
HAND_Z_RANGE = (-0.120, -0.010)  # ... nor TCP heights outside 10-120 mm above the surface (keeps 286 of 308)


def hand_rows(dist: dict) -> list[dict]:
    """Real window-start hand poses that are above the hole. 4% of real windows start more than 50 mm away (the
    operator still travelling 1.5 s before contact, up to 0.58 m) and are left out."""
    return [r for r in dist["hand"]["rows"] if math.hypot(r["offset"][0], r["offset"][1]) <= HAND_LATERAL_MAX
            and HAND_Z_RANGE[0] < r["offset"][2] < HAND_Z_RANGE[1]]


def make_set(name: str, dist: dict, board: dict, parts=PEGS) -> list[dict]:
    cfg = SETS[name]
    rng = np.random.default_rng(cfg["seed"])
    m = arm_model()
    from twin.scene.build_base import START_ARM_QPOS
    from twin.env import check_start
    hands = hand_rows(dist)
    out = []
    rejected = dict(tip_height=0, ik=0, grasp=0)
    for part in parts:
        slug = part.replace(" ", "_")
        for i in range(cfg["n"]):
            for tries in range(200):
                err = draw_board_err(rng, dist)
                ih = draw_inhand(rng, dist)
                hr = hands[int(rng.integers(len(hands)))]
                h = _tip_height(part, board, err, hr, ih)
                if not (TIP_CLEAR_MIN <= h <= TIP_CLEAR_MAX):
                    rejected["tip_height"] += 1
                    continue
                Hn, Rwn = entry_pose(part, board)
                tcp = Hn + Rwn @ np.asarray(hr["offset"])
                q_hand = mat_quat(Rwn @ quat_mat(hr["R_rel_quat"]))
                q7, ik = solve_ik(m, START_ARM_QPOS, tcp, q_hand)
                if not ik["ok"]:
                    rejected["ik"] += 1
                    continue
                cand = dict(id=f"{cfg['prefix']}-{slug}-{i:03d}", set=name, part=part, index=i,
                            board_err=err, inhand=ih,
                            hand=dict(offset=list(map(float, hr["offset"])), R_rel_quat=list(map(float, hr["R_rel_quat"])),
                                      src=f"window start ep {hr['episode']} seg {hr['segment_index']} {hr['part']}"),
                            tip_clearance_mm=round(1000 * h, 2), q_start=[round(float(v), 8) for v in q7],
                            draws=tries + 1)
                chk = check_start(cand, board)
                if not chk["ok"]:
                    rejected["grasp"] += 1
                    continue
                cand["check"] = chk
                out.append(cand)
                break
            else:
                raise RuntimeError(f"no valid start for {part} #{i} in 200 draws")
    REJECTED[name] = rejected
    return out


REJECTED: dict[str, dict] = {}


def freeze(out_dir: Path = OUT_DIR, remeasure: bool = False) -> dict:
    """Measure (or reuse) the distributions, draw every set, write files + manifest with sha256. Refuses to
    overwrite an existing manifest (a frozen evaluator changes only with a new run)."""
    out_dir = Path(out_dir)
    man_f = out_dir / "manifest.json"
    if man_f.exists():
        raise FileExistsError(f"{man_f} exists: start sets are frozen. A change needs a new run directory.")
    dist_f = out_dir / "distributions.json"
    dist = measure(out_dir) if (remeasure or not dist_f.exists()) else json.loads(dist_f.read_text())
    board = nominal_board()
    files = {}
    counts = {}
    for name in SETS:
        starts = make_set(name, dist, board)
        f = out_dir / f"{name}.json"
        f.write_text(json.dumps(dict(set=name, nominal_board=board, starts=starts), indent=0))
        files[name] = dict(file=f.name, sha256=sha256_file(f), n=len(starts), seed=SETS[name]["seed"])
        counts[name] = {p: sum(1 for s in starts if s["part"] == p) for p in PEGS}
    n_in = len(dist["inhand"]["rows"])
    n_keep = sum(1 for r in dist["inhand"]["rows"] if abs(r["dx"]) <= PAD_DX_MAX and abs(r["dy"]) <= PAD_DY_MAX)
    man = dict(frozen_at=time.strftime("%Y-%m-%d %H:%M"), frozen_by="twin/starts.py freeze (sim-integrity stand-in)",
               nominal_board=board, files=files, counts=counts, rejected_draws=REJECTED,
               inhand_events_kept=f"{n_keep}/{n_in} remove grasps with |dx| <= {PAD_DX_MAX * 1000} mm and |dy| <= "
                                  f"{PAD_DY_MAX * 1000} mm",
               hand_rows_kept=f"{len(hand_rows(dist))}/{len(dist['hand']['rows'])} window starts within "
                              f"{HAND_LATERAL_MAX * 1000:.0f} mm of the hole axis and {-HAND_Z_RANGE[1] * 1000:.0f}-"
                              f"{-HAND_Z_RANGE[0] * 1000:.0f} mm above the surface",
               validity="every start was built in the nominal twin and its grasp held through the pre-roll and 0.5 s "
                        "(twin/env.py check_start)",
               heldout_subset=dict(first_n_per_part=HELDOUT_SUBSET,
                                   why="29/29 gives a one-sided 95% Clopper-Pearson lower bound 0.902 (>= 0.9 with 0 "
                                       "failures); the full 46 allow one failure (validation.yaml header, v6)"),
               distributions=dict(file=dist_f.name, sha256=sha256_file(dist_f)),
               directory=str(out_dir.resolve().relative_to(REPO_DIR)) if out_dir.resolve().is_relative_to(REPO_DIR)
               else str(out_dir),
               generator=dict(file="twin/starts.py", sha256=sha256_file(Path(__file__))),
               rules=["practice outcomes may be shown to the actor; held-out outcomes go to the evaluator log only",
                      "each held-out start is attempted once per brain run (the env refuses a second attempt)",
                      "privileged red-team scripts may use redteam starts only"])
    man_f.write_text(json.dumps(man, indent=1))
    (out_dir / "manifest.sha256").write_text(sha256_file(man_f) + "\n")
    return man


def load(set_name: str, out_dir: Path = OUT_DIR) -> tuple[dict, dict]:
    """(starts by id, nominal board) of a frozen set, after checking the file against the manifest hash."""
    out_dir = Path(out_dir)
    man = json.loads((out_dir / "manifest.json").read_text())
    want = (out_dir / "manifest.sha256").read_text().split()[0]
    if sha256_file(out_dir / "manifest.json") != want:
        raise RuntimeError("starts manifest.json does not match manifest.sha256: the frozen start sets were edited")
    info = man["files"].get(set_name)
    if info is None:
        raise KeyError(f"unknown start set {set_name!r}; frozen: {list(man['files'])}")
    f = out_dir / info["file"]
    if sha256_file(f) != info["sha256"]:
        raise RuntimeError(f"{info['file']} does not match its frozen sha256 in the manifest")
    body = json.loads(f.read_text())
    return {s["id"]: s for s in body["starts"]}, body["nominal_board"]


def set_of(start_id: str) -> str:
    return {v["prefix"]: k for k, v in SETS.items()}[start_id.split("-", 1)[0]]


def main():
    man = freeze()
    print(json.dumps({k: v for k, v in man.items() if k in ("files", "counts", "distributions")}, indent=1))


if __name__ == "__main__":
    main()
