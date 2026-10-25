"""Board pose relative to the robot from the robot's own contact events (stage B0, skill board-from-contacts).

Why contacts: the static cameras resolve ~2 mm per pixel at the board and peg clearances are 0.05-0.25 mm,
while the logged tool tip (O_T_EE, 0.174 m TCP) is good to ~0.1 mm. Every successful Insert ends, and every
Remove starts, with a part sitting in its hole between the fingers. Those moments place the holes.

Event model (one event = one moment a part sits in its hole while the fingers hold it):

    p_e = T_r * h_k + u_e * d_e + eps_e

    p_e   TCP xy in the robot base frame at the event
    T_r   board pose of recording r: x, y, yaw (the mocap NIST_Board1 frame convention, origin at the plate corner)
    h_k   hole of part k in the board frame: one rigid layout for all recordings, fitted on train recordings
    d_e   horizontal unit vector of the EE x axis = the finger-pad axis at the event
    u_e   in-hand offset of the part along the pad axis; new every grasp; prior N(0, sigma_pad)
    eps_e mostly along the closing axis n_e (EE y): the two coupled fingers centre the part, sigma_close ~ 1 mm

u_e is marginalised analytically, so each event gives a tight constraint across the pads and a loose one along
them (a point-to-line registration). Robust loss (Cauchy by default, Geman-McClure as in Kim et al. ICRA 2023).

Layout fit (train recordings, joint): h_k for every part, a small correction of each recording's mocap board pose
(dx, dy, dyaw, zero-mean prior that fixes the gauge to the average mocap frame), events mapped through mocap.
Per-recording fit (any recording, no mocap): robust 2-point RANSAC over parts for the start, then the line model.
NIST layout (nist_layout): the NIST laser-plate features placed with one rigid transform (reflection allowed) fitted
to the contact layout; all 17 parts agree to ~0.8 mm RMS, so it is both a check and an exact-CAD alternative.

Events (extract_events):
  insert_seat   successful Insert holding a part: deepest TCP in the last 3 s before the fingers open
                (release TCP kept too: validation.yaml B0.board_err_mm is defined on release)
  remove_grasp  successful Remove: the grasp still closed at the segment end, at the frame the fingers finish
                closing on the seated part
  insert_fail   failed Insert holding a part: deepest TCP (where the hole was NOT); diagnostics only
  onset         first |F_base| > 5 N in an Insert while holding (the contact cloud; diagnostics only)

Data: data/reassemble/data (local, episodes 0-14) and data/reassemble-gate/data (downloaded HF parquet, read as
data only). Splits come from playbook/splits.json; exam episodes are refused here, whatever the caller.

    .venv/bin/python -m twin.board_fit layout          # fit + write the train layout
    .venv/bin/python -m twin.board_fit fit 2025-01-11-11-12-32
    .venv/bin/python -m twin.board_fit eval --split gate
"""

from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from twin import REPO_DIR

DATA_DIRS = (REPO_DIR / "data" / "reassemble", REPO_DIR / "data" / "reassemble-gate")
META = REPO_DIR / "data" / "reassemble" / "meta"
POSES = REPO_DIR / "data" / "reassemble-tudata" / "poses"
SPLITS = REPO_DIR / "playbook" / "splits.json"
CACHE = REPO_DIR / "runs" / "cache" / "board_events"
OUT_DEFAULT = REPO_DIR / "runs" / "claude" / "run-standin-1" / "board"
EVENTS_VERSION = 3
FPS = 30
HOLD_MAX = 0.075      # finger sum below this = holding something (fully open is 0.080)
RELEASE_OPEN = 0.003  # fingers this much wider than the holding width = released
SEAT_WINDOW_S = 3.0   # deepest TCP within this window before release = seated
CONTACT_N = 5.0       # teleop contact threshold on |force.base| (reassemble-data-notes)
FIT_KINDS = ("insert_seat", "remove_grasp")

# Model hyper-parameters, chosen on train leave-one-part-out (see playbook/skills/board-from-contacts Lessons) and
# frozen before any gate recording was read.
PARAMS = dict(sigma_close_mm=1.0, sigma_pad_mm=6.0, loss="cauchy", f_scale=2.0,
              mocap_prior_mm=10.0, mocap_prior_yaw_deg=2.0, ransac_inlier_mm=15.0, outlier_close_mm=6.0)


# ------------------------------------------------------------------------------------------------ data access
@lru_cache(maxsize=1)
def splits() -> dict:
    return json.loads(SPLITS.read_text())["episodes"]


def recording_of(ep: int) -> str:
    return splits()[str(int(ep))]["recording"]


def episode_of(recording: str) -> int:
    for k, v in splits().items():
        if v["recording"] == recording:
            return int(k)
    raise KeyError(f"recording {recording} not in playbook/splits.json")


def split_of(ep: int) -> str:
    return splits()[str(int(ep))]["split"]


def _guard(ep: int):
    if split_of(ep) == "exam":
        raise PermissionError(f"episode {ep} is in the exam split (playbook/splits.json); refused")


def episodes_in(split: str) -> list[int]:
    """Episodes of a project split (train / gate) whose data file is on disk. Never exam."""
    if split == "exam":
        raise PermissionError("exam split is held out")
    return [int(k) for k, v in splits().items() if v["split"] == split and _data_file(int(k)) is not None]


@lru_cache(maxsize=1)
def _episode_index():
    import pandas as pd
    eps = pd.read_parquet(META / "episodes/chunk-000/file-000.parquet").set_index("episode_index")
    return {int(e): (int(r["data/chunk_index"]), int(r["data/file_index"])) for e, r in eps.iterrows()}


def _data_file(ep: int) -> Path | None:
    from twin import exam
    exam.check(ep)
    c, f = _episode_index()[int(ep)]
    for d in DATA_DIRS:
        p = d / f"data/chunk-{c:03d}/file-{f:03d}.parquet"
        if p.exists():
            return p
    return None


@lru_cache(maxsize=1)
def _tasks() -> dict:
    import pandas as pd
    t = pd.read_parquet(META / "tasks.parquet").reset_index()
    return dict(zip(t["task_index"].astype(int), t["task"]))


def _frames(ep: int) -> dict:
    """The columns this module needs for one (non-exam) episode, sorted by frame."""
    import pyarrow.parquet as pq
    _guard(ep)
    p = _data_file(ep)
    if p is None:
        raise FileNotFoundError(f"episode {ep}: data parquet not on disk (looked in {[str(d) for d in DATA_DIRS]})")
    cols = ["frame_index", "segment.index", "segment.success", "task_index", "observation.state.ee_pose",
            "observation.state.gripper_position", "observation.force.base"]
    tb = pq.read_table(p, columns=cols, filters=[("episode_index", "=", int(ep))]).sort_by("frame_index")
    out = {}
    for c in cols:
        col = tb.column(c).combine_chunks()
        out[c] = (np.asarray(col.flatten(), dtype=np.float64).reshape(len(col), -1) if c.startswith("observation")
                  else np.asarray(col))
    return out


def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mocap_board(recording: str) -> dict | None:
    """Raw mocap NIST_Board1 pose of a non-exam recording: pos (m), R, yaw/roll/pitch (deg), sane flag."""
    ep = episode_of(recording)
    _guard(ep)
    f = POSES / f"{recording}_poses.json"
    if not f.exists():
        return None
    pos, qxyzw = json.loads(f.read_text())["NIST_Board1"]
    x, y, z, w = qxyzw
    R = quat_wxyz_to_R([w, x, y, z])
    tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    return dict(pos=np.array(pos, float), R=R, quat_wxyz=[w, x, y, z],
                yaw_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), tilt_deg=tilt, sane=tilt < 5.0)


# ------------------------------------------------------------------------------------------------ events
def _event(kind, ep, part, frame, P, W, extra=None):
    R = quat_wxyz_to_R(P[frame, 3:7])
    d = R[:2, 0] / max(np.linalg.norm(R[:2, 0]), 1e-9)
    e = dict(kind=kind, episode=int(ep), part=part, frame=int(frame), t=round(frame / FPS, 3),
             tcp=[float(v) for v in P[frame, :3]], pad_dir=[float(d[0]), float(d[1])],
             ee_x_z=float(R[2, 0]), ee_y_z=float(R[2, 1]),
             tilt_deg=float(np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1)))),
             grip_mm=round(1000 * float(W[frame]), 2))
    if extra:
        e.update(extra)
    return e


def extract_events(ep: int, use_cache: bool = True) -> list[dict]:
    """Contact events of one episode (see module doc). Cached under runs/cache/board_events/."""
    _guard(ep)
    cf = CACHE / f"ep{int(ep):03d}.json"
    if use_cache and cf.exists():
        c = json.loads(cf.read_text())
        if c.get("version") == EVENTS_VERSION:
            return c["events"]
    f = _frames(ep)
    tasks = _tasks()
    P = f["observation.state.ee_pose"]
    W = f["observation.state.gripper_position"].sum(1)
    Fm = np.linalg.norm(f["observation.force.base"], axis=1)
    si = f["segment.index"]
    hold = W < HOLD_MAX
    cut = np.flatnonzero(np.diff(si) != 0) + 1
    ev = []
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(si)]):
        a, b = int(a), int(b)
        task = tasks[int(f["task_index"][a])]
        ok = bool(f["segment.success"][a])
        verb, _, part = task.rstrip(".").partition(" ")
        if verb == "Insert":
            g = W[a:b]
            h = g < HOLD_MAX
            if not h[: max(1, len(g) // 3)].any():
                continue  # pushed with open fingers: nothing held
            w = float(np.median(g[h]))
            fh = int(np.argmax(h))
            op = np.flatnonzero(g > w + RELEASE_OPEN)
            op = op[op > fh]
            rel = a + (int(op[0]) - 1 if len(op) else len(g) - 1)
            lo = max(a + fh, rel - int(SEAT_WINDOW_S * FPS))
            deep = lo + int(np.argmin(P[lo:rel + 1, 2]))
            con = np.flatnonzero((Fm[a:rel + 1] > CONTACT_N) & hold[a:rel + 1])
            if len(con):
                ev.append(_event("onset", ep, part, a + int(con[0]), P, W, dict(success=ok)))
            if ok:
                ev.append(_event("insert_seat", ep, part, deep, P, W, dict(
                    release_frame=int(rel), release_tcp=[float(v) for v in P[rel, :3]],
                    hold_width_mm=round(1000 * w, 2), peak_force_n=round(float(Fm[a:rel + 1].max()), 2))))
            else:
                ev.append(_event("insert_fail", ep, part, deep, P, W, dict(hold_width_mm=round(1000 * w, 2))))
        elif verb == "Remove" and ok:
            if not hold[b - 1]:
                continue
            s = b - 1
            while s > a and hold[s - 1]:
                s -= 1
            if s == a and a > 0 and hold[a - 1]:
                continue  # grasp started before the segment: not a grasp of the seated part
            w = float(np.median(W[s:b]))
            if w > HOLD_MAX - 0.001:
                continue
            c = s + int(np.argmax(W[s:b] < w + 0.0005))
            pull = float(f["observation.force.base"][c:min(b, c + 2 * FPS), 2].min())
            ev.append(_event("remove_grasp", ep, part, c, P, W, dict(hold_width_mm=round(1000 * w, 2),
                                                                     pull_fz_n=round(pull, 2))))
    CACHE.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(dict(version=EVENTS_VERSION, episode=int(ep), recording=recording_of(ep), events=ev)))
    return ev


# ------------------------------------------------------------------------------------------------ geometry
def _rot2(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _residuals(p, d, hw, sig_n, sig_d):
    """Closing-axis and pad-axis residuals (in sigmas) of TCP points p vs predicted holes hw, pad dirs d."""
    n = np.c_[-d[:, 1], d[:, 0]]
    v = p - hw
    return (v * n).sum(1) / sig_n, (v * d).sum(1) / sig_d


def _loss(name):
    if name == "gm":  # Geman-McClure: rho(z) = z / (1 + z)
        def gm(z):
            r = np.empty((3, z.size))
            r[0] = z / (1 + z)
            r[1] = 1 / (1 + z) ** 2
            r[2] = -2 / (1 + z) ** 3
            return r
        return gm
    return name


def _ransac_init(p, h, inlier_mm):
    """Robust 2D rigid start from point pairs (event TCP ~ hole + offset): p ~ R h + t. Returns (t, yaw)."""
    best = (-1, 0.0, None)
    m = len(p)
    if m == 1:
        return p[0] - h[0], 0.0
    for i in range(m):
        for j in range(i + 1, m):
            dh, dp = h[j] - h[i], p[j] - p[i]
            if np.linalg.norm(dh) < 30.0:
                continue
            yaw = np.arctan2(dp[1], dp[0]) - np.arctan2(dh[1], dh[0])
            Rm = _rot2(yaw)
            t = p[i] - Rm @ h[i]
            err = np.linalg.norm(p - (h @ Rm.T + t), axis=1)
            score = (err < inlier_mm).sum() - 1e-6 * np.median(err)
            if score > best[0]:
                best = (score, yaw, err < inlier_mm)
    yaw, inl = best[1], best[2]
    if inl is None or inl.sum() < 2:
        return (p - h).mean(0), 0.0
    # Kabsch on inliers
    P_, H_ = p[inl], h[inl]
    pc, hc = P_.mean(0), H_.mean(0)
    U, _, Vt = np.linalg.svd((H_ - hc).T @ (P_ - pc))
    Rm = (U @ Vt).T
    if np.linalg.det(Rm) < 0:
        Vt[-1] *= -1
        Rm = (U @ Vt).T
    yaw = float(np.arctan2(Rm[1, 0], Rm[0, 0]))
    return pc - Rm @ hc, yaw


# ------------------------------------------------------------------------------------------------ layout
def _board_frame_events(ev, mb):
    """Events of one recording mapped into its mocap board frame (mm): p_b, d_b, z_b."""
    Rb, tb = mb["R"], mb["pos"]
    out = []
    for e in ev:
        pb = Rb.T @ (np.array(e["tcp"]) - tb)
        db = (Rb.T @ np.r_[e["pad_dir"], 0.0])[:2]
        out.append((1000 * pb[:2], db / np.linalg.norm(db), 1000 * pb[2]))
    return out


def fit_layout(episodes: list[int], params: dict | None = None, kinds=FIT_KINDS) -> dict:
    """Joint fit over train recordings: hole xy per part (board frame) + per-recording mocap corrections."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    prm = dict(PARAMS, **(params or {}))
    for ep in episodes:
        if split_of(ep) == "exam":
            raise PermissionError("exam episode in layout fit")
    rows, recs, skipped = [], [], []
    for ep in episodes:
        rec = recording_of(ep)
        mb = mocap_board(rec)
        if mb is None or not mb["sane"]:
            skipped.append(dict(episode=ep, recording=rec, why="no mocap" if mb is None else
                                f"mocap tilt {mb['tilt_deg']:.1f} deg"))
            continue
        ev = [e for e in extract_events(ep) if e["kind"] in kinds]
        if not ev:
            continue
        recs.append(rec)
        for e, (pb, db, zb) in zip(ev, _board_frame_events(ev, mb)):
            rows.append(dict(rec=len(recs) - 1, part=e["part"], kind=e["kind"], p=pb, d=db, z=zb,
                             grip=e.get("hold_width_mm", e["grip_mm"])))
    parts = sorted({r["part"] for r in rows})
    pi = {p: i for i, p in enumerate(parts)}
    k = np.array([pi[r["part"]] for r in rows])
    ri = np.array([r["rec"] for r in rows])
    p = np.array([r["p"] for r in rows])
    d = np.array([r["d"] for r in rows])
    E, Pn, Rn = len(rows), len(parts), len(recs)
    h0 = np.array([np.median(p[k == i], axis=0) for i in range(Pn)])
    sy = np.radians(prm["mocap_prior_yaw_deg"]) * 1000  # mrad

    def unpack(x):
        return x[:2 * Pn].reshape(Pn, 2), x[2 * Pn:].reshape(Rn, 3)

    def fun(x):
        h, dr = unpack(x)
        yaw = dr[ri, 2] / 1000
        c, s = np.cos(yaw), np.sin(yaw)
        hk = h[k]
        hw = np.c_[c * hk[:, 0] - s * hk[:, 1] + dr[ri, 0], s * hk[:, 0] + c * hk[:, 1] + dr[ri, 1]]
        rn, rd = _residuals(p, d, hw, prm["sigma_close_mm"], prm["sigma_pad_mm"])
        return np.r_[rn, rd, dr[:, :2].ravel() / prm["mocap_prior_mm"], dr[:, 2] / sy]

    S = lil_matrix((2 * E + 3 * Rn, 2 * Pn + 3 * Rn), dtype=int)
    for e in range(E):
        for row in (e, E + e):
            S[row, 2 * k[e]:2 * k[e] + 2] = 1
            S[row, 2 * Pn + 3 * ri[e]:2 * Pn + 3 * ri[e] + 3] = 1
    for j in range(3 * Rn):
        S[2 * E + j, 2 * Pn + j] = 1
    # start: per-recording robust correction vs the per-part median layout
    dr0 = np.zeros((Rn, 3))
    for r in range(Rn):
        m = ri == r
        t, yaw = _ransac_init(p[m], h0[k[m]], prm["ransac_inlier_mm"])
        dr0[r] = [t[0], t[1], 1000 * yaw]
    dr0 -= dr0.mean(0)
    sol = least_squares(fun, np.r_[h0.ravel(), dr0.ravel()], jac_sparsity=S, loss=_loss(prm["loss"]),
                        f_scale=prm["f_scale"], x_scale="jac")
    h, dr = unpack(sol.x)
    res = fun(sol.x)
    rn = res[:E] * prm["sigma_close_mm"]
    rd = res[E:2 * E] * prm["sigma_pad_mm"]
    # per-part uncertainty from the Gauss-Newton covariance, scaled by the robust closing-residual spread
    J = sol.jac.toarray() if hasattr(sol.jac, "toarray") else sol.jac
    try:
        cov = np.linalg.pinv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((J.shape[1], J.shape[1]), np.nan)
    out_parts = {}
    for i, name in enumerate(parts):
        m = k == i
        kinds_here = {}
        for kd in kinds:
            mk = m & np.array([r["kind"] == kd for r in rows])
            if mk.any():
                zs = np.array([rows[j]["z"] for j in np.flatnonzero(mk)])
                kinds_here[kd] = dict(n=int(mk.sum()), tcp_z_board_mm=round(float(np.median(zs)), 2),
                                      tcp_z_mad_mm=round(float(np.median(np.abs(zs - np.median(zs)))), 2),
                                      pad_offset_median_mm=round(float(np.median(rd[mk])), 2))
        sd = np.sqrt(np.clip(np.diag(cov)[2 * i:2 * i + 2], 0, None))
        out_parts[name] = dict(
            xy=[round(float(h[i, 0]) / 1000, 5), round(float(h[i, 1]) / 1000, 5)],
            xy_sigma_mm=[round(float(s), 2) for s in sd],
            n_events=int(m.sum()), n_recordings=int(len(set(ri[m]))),
            close_resid_mad_mm=round(float(np.median(np.abs(rn[m]))), 2),
            grip_mm_median=round(float(np.median([rows[j]["grip"] for j in np.flatnonzero(m)])), 2),
            kinds=kinds_here)
    corr = [dict(recording=recs[r], dx_mm=round(float(dr[r, 0]), 2), dy_mm=round(float(dr[r, 1]), 2),
                 dyaw_deg=round(float(np.degrees(dr[r, 2] / 1000)), 3)) for r in range(Rn)]
    shift = np.hypot(dr[:, 0], dr[:, 1])
    return dict(
        generated_by="twin/board_fit.py fit_layout", generated_at=time.strftime("%Y-%m-%d %H:%M"),
        frame="board = mocap NIST_Board1 frame convention (plate corner origin, z up); gauge = average of the "
              "train recordings' mocap poses (zero-mean correction prior)",
        params=prm, kinds=list(kinds), episodes=sorted(int(e) for e in episodes),
        recordings=recs, skipped=skipped, n_events=E, parts=out_parts,
        fit=dict(close_resid_median_mm=round(float(np.median(np.abs(rn))), 3),
                 close_resid_p90_mm=round(float(np.percentile(np.abs(rn), 90)), 3),
                 pad_offset_median_mm=round(float(np.median(rd)), 2),
                 pad_offset_mad_mm=round(float(np.median(np.abs(rd - np.median(rd)))), 2),
                 cost=float(sol.cost), status=int(sol.status)),
        mocap_corrections=corr,
        mocap_correction_summary=dict(
            shift_median_mm=round(float(np.median(shift)), 2), shift_p90_mm=round(float(np.percentile(shift, 90)), 2),
            yaw_abs_median_deg=round(float(np.degrees(np.median(np.abs(dr[:, 2])) / 1000)), 3)))


# NIST Assembly Task Board 1 laser plate features (mm) in the plate-slice coordinates of
# runs/scene/experiments/stl_holes.log (GMC_Laser_Plate_Virtual.STL at mid-thickness). Holes typed by area
# (round: area = pi w h / 4; key: area = w h). Connector housings and the waterproof socket sit between their
# screw holes (midpoint of the pair / centre of the 4-hole pattern).
_MID = lambda *p: [float(v) for v in np.mean(np.array(p), axis=0)]  # noqa: E731
NIST_PLATE_FEATURES = {
    "round peg 1": [191.45, 347.15], "round peg 2": [116.44, 347.12], "round peg 3": [41.19, 122.1],
    "round peg 4": [266.2, 122.24], "square peg 1": [191.45, 197.4], "square peg 2": [266.66, 272.35],
    "square peg 3": [41.76, 195.61], "square peg 4": [40.75, 347.0], "BNC": [191.31, 279.65],
    "bolt 4": [341.28, 47.1], "large gear": [141.49, 47.17], "medium gear": [191.45, 47.22],
    "small gear": [221.49, 47.17], "D-SUB": _MID((266.47, 212.16), (266.47, 182.16)),
    "Ethernet": _MID((341.45, 332.16), (341.45, 362.16)), "USB": _MID((26.45, 272.14), (56.43, 272.11)),
    "waterproof": _MID((102.94, 168.69), (129.97, 168.67), (102.94, 225.69), (129.97, 225.69))}


def nist_layout(contact_layout: dict, fit_parts=None) -> dict:
    """The NIST laser-plate layout placed in the board frame: one rigid transform (reflection allowed, no scale)
    fitted between the contact layout and NIST_PLATE_FEATURES. Keeps the contact layout's per-part z data."""
    names = [n for n in contact_layout["parts"] if n in NIST_PLATE_FEATURES and (fit_parts is None or n in fit_parts)]
    A = np.array([[1000 * v for v in contact_layout["parts"][n]["xy"]] for n in names])
    B = np.array([NIST_PLATE_FEATURES[n] for n in names])
    ac, bc = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ac).T @ (B - bc))
    R = (U @ Vt).T
    t = bc - R @ ac
    out = json.loads(json.dumps(contact_layout))
    res = {}
    for n, v in out["parts"].items():
        if n not in NIST_PLATE_FEATURES:
            continue
        hb = R.T @ (np.array(NIST_PLATE_FEATURES[n]) - t)  # NIST -> board frame (mm)
        res[n] = round(float(np.linalg.norm(hb - 1000 * np.array(v["xy"]))), 3)
        v["xy_contact"] = v["xy"]
        v["xy"] = [round(float(hb[0]) / 1000, 5), round(float(hb[1]) / 1000, 5)]
    rr = np.array(list(res.values()))
    out["source"] = "nist"
    out["nist_fit"] = dict(fit_parts=names, det=round(float(np.linalg.det(R)), 3),
                           angle_deg=round(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), 3),
                           t_mm=[round(float(v), 3) for v in t], rms_mm=round(float(np.sqrt((rr ** 2).mean())), 3),
                           max_mm=round(float(rr.max()), 3), residual_mm=res,
                           note="contact-layout hole minus NIST feature mapped into the board frame")
    return out


def layout_holes_mm(layout: dict, parts=None) -> tuple[list[str], np.ndarray]:
    names = [n for n in layout["parts"] if parts is None or n in parts]
    return names, np.array([[1000 * v for v in layout["parts"][n]["xy"]] for n in names])


# ------------------------------------------------------------------------------------------------ per recording
def fit_board(events: list[dict], layout: dict, exclude_parts=(), params: dict | None = None,
              kinds=FIT_KINDS, init=None) -> dict:
    """Board pose (x, y, yaw; z from per-part TCP heights) from one recording's events and the rigid layout."""
    from scipy.optimize import least_squares
    prm = dict(PARAMS, **(params or {}))
    ev = [e for e in events if e["kind"] in kinds and e["part"] in layout["parts"]
          and e["part"] not in set(exclude_parts)]
    parts_used = sorted({e["part"] for e in ev})
    if len(parts_used) < 2:
        raise ValueError(f"need events from >= 2 parts, have {parts_used}")
    p = 1000 * np.array([e["tcp"][:2] for e in ev])
    d = np.array([e["pad_dir"] for e in ev])
    h = np.array([[1000 * v for v in layout["parts"][e["part"]]["xy"]] for e in ev])
    if init is None:
        # RANSAC on per-part medians (one point per part) first, so one part's many events can't dominate
        pm = np.array([np.median(p[[e["part"] == q for e in ev]], axis=0) for q in parts_used])
        hm = np.array([[1000 * v for v in layout["parts"][q]["xy"]] for q in parts_used])
        t0, yaw0 = _ransac_init(pm, hm, prm["ransac_inlier_mm"])
    else:
        t0, yaw0 = np.array(init[:2]) * 1000, init[2]

    def fun(x):
        hw = h @ _rot2(x[2]).T + x[:2]
        rn, rd = _residuals(p, d, hw, prm["sigma_close_mm"], prm["sigma_pad_mm"])
        return np.r_[rn, rd]

    sol = least_squares(fun, np.r_[t0, yaw0], loss=_loss(prm["loss"]), f_scale=prm["f_scale"],
                        x_scale=[10.0, 10.0, 0.01])
    x = sol.x
    res = fun(x)
    E = len(ev)
    rn, rd = res[:E] * prm["sigma_close_mm"], res[E:] * prm["sigma_pad_mm"]
    # covariance: Gauss-Newton with the robust residual scale (MAD of closing residuals)
    J = sol.jac
    s_close = max(1.4826 * float(np.median(np.abs(rn))), 0.3) / prm["sigma_close_mm"]
    try:
        cov = np.linalg.inv(J.T @ J) * s_close ** 2
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.nan)
    # z: TCP height minus each part's train TCP height above the board (per event kind)
    zs = []
    for e in ev:
        zk = layout["parts"][e["part"]]["kinds"].get(e["kind"])
        if zk:
            zs.append(1000 * e["tcp"][2] - zk["tcp_z_board_mm"])
    z = float(np.median(zs)) / 1000 if zs else 0.0
    yaw = float(x[2])
    out_ev = [dict(part=e["part"], kind=e["kind"], t=e["t"], close_resid_mm=round(float(a), 2),
                   pad_offset_mm=round(float(b), 2), outlier=bool(abs(a) > prm["outlier_close_mm"]))
              for e, a, b in zip(ev, rn, rd)]
    return dict(
        pos=[float(x[0]) / 1000, float(x[1]) / 1000, z], yaw_deg=float(np.degrees(yaw)),
        quat_wxyz=[float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))],
        sigma=dict(x_mm=round(float(np.sqrt(cov[0, 0])), 3), y_mm=round(float(np.sqrt(cov[1, 1])), 3),
                   yaw_deg=round(float(np.degrees(np.sqrt(cov[2, 2]))), 4),
                   z_mm=round(float(1.4826 * np.median(np.abs(np.array(zs) - 1000 * z)) / np.sqrt(len(zs))), 2)
                   if zs else None),
        n_events=E, n_parts=len(parts_used), parts=parts_used, excluded=sorted(exclude_parts),
        close_resid_median_mm=round(float(np.median(np.abs(rn))), 3),
        close_resid_p90_mm=round(float(np.percentile(np.abs(rn), 90)), 3),
        pad_offset_median_mm=round(float(np.median(rd)), 2),
        n_outliers=int(sum(o["outlier"] for o in out_ev)), events=out_ev, status=int(sol.status))


def world_holes(pose: dict, layout: dict) -> dict:
    """Hole xy (m, world) of every layout part under a board pose {pos, yaw_deg} or a mocap dict {pos, R}."""
    if "R" in pose:
        R, t = pose["R"], np.asarray(pose["pos"], float)
        return {n: (R @ np.r_[v["xy"], 0.0] + t).tolist() for n, v in layout["parts"].items()}
    yaw = np.radians(pose["yaw_deg"])
    R2 = _rot2(yaw)
    t = np.asarray(pose["pos"], float)
    return {n: [*(R2 @ np.array(v["xy"]) + t[:2]).tolist(), float(t[2])] for n, v in layout["parts"].items()}


def drift(events: list[dict], layout: dict, params: dict | None = None, min_parts: int = 4) -> dict | None:
    """Did the board move during the recording? Fit the first and second half of the fit events (by time)
    separately and compare the hole positions they predict."""
    ev = sorted([e for e in events if e["kind"] in FIT_KINDS and e["part"] in layout["parts"]], key=lambda e: e["t"])
    if len(ev) < 2 * min_parts:
        return None
    half = len(ev) // 2
    a, b = ev[:half], ev[half:]
    if len({e["part"] for e in a}) < min_parts or len({e["part"] for e in b}) < min_parts:
        return None
    fa, fb = fit_board(a, layout, params=params), fit_board(b, layout, params=params)
    ha, hb = world_holes(fa, layout), world_holes(fb, layout)
    dd = [1000 * np.linalg.norm(np.array(ha[k][:2]) - np.array(hb[k][:2])) for k in layout["parts"]]
    return dict(t_split_s=round(b[0]["t"], 1), n_first=len(a), n_second=len(b),
                hole_shift_median_mm=round(float(np.median(dd)), 2), hole_shift_max_mm=round(float(max(dd)), 2),
                dyaw_deg=round(float(fb["yaw_deg"] - fa["yaw_deg"]), 3),
                sigma_first_mm=[fa["sigma"]["x_mm"], fa["sigma"]["y_mm"]],
                sigma_second_mm=[fb["sigma"]["x_mm"], fb["sigma"]["y_mm"]])


# ------------------------------------------------------------------------------------------------ evaluation
def _event_errors(ev, holes_w):
    """Per event: closing-axis distance (mm) and xy distance (mm) from TCP to the predicted hole."""
    out = []
    for e in ev:
        hw = np.array(holes_w[e["part"]][:2]) * 1000
        p = np.array(e["tcp"][:2]) * 1000
        d = np.array(e["pad_dir"])
        n = np.array([-d[1], d[0]])
        v = p - hw
        out.append(dict(close=float(abs(v @ n)), xy=float(np.linalg.norm(v)), pad=float(v @ d)))
    return out


def evaluate_recording(ep: int, layout: dict, params: dict | None = None, min_parts: int = 3) -> dict:
    """Held-out errors for one recording: leave-one-part-out (contact) vs mocap, plus the B0 metric."""
    rec = recording_of(ep)
    events = extract_events(ep)
    fit_ev = [e for e in events if e["kind"] in FIT_KINDS and e["part"] in layout["parts"]]
    parts = sorted({e["part"] for e in fit_ev})
    mb = mocap_board(rec)
    full = fit_board(events, layout, params=params) if len(parts) >= 2 else None
    lopo = {}
    for q in parts:
        if len(parts) - 1 < min_parts - 1 or len(parts) < min_parts:
            break
        try:
            f = fit_board(events, layout, exclude_parts=[q], params=params)
        except ValueError:
            continue
        hw_c = world_holes(f, layout)
        ev_q = [e for e in fit_ev if e["part"] == q]
        ec = _event_errors(ev_q, hw_c)
        row = dict(n_events=len(ev_q), contact_close_mm=float(np.median([x["close"] for x in ec])),
                   contact_xy_mm=float(np.median([x["xy"] for x in ec])),
                   kinds=sorted({e["kind"] for e in ev_q}))
        if mb is not None:
            hw_m = world_holes(mb, layout)
            em = _event_errors(ev_q, hw_m)
            row.update(mocap_close_mm=float(np.median([x["close"] for x in em])),
                       mocap_xy_mm=float(np.median([x["xy"] for x in em])),
                       contact_vs_mocap_hole_mm=float(1000 * np.linalg.norm(
                           np.array(hw_c[q][:2]) - np.array(hw_m[q][:2]))))
        # B0 as written: hole entry vs median TCP at this part's successful insert RELEASES
        rel = [e["release_tcp"] for e in ev_q if e["kind"] == "insert_seat"]
        if rel:
            mrel = np.median(np.array(rel), axis=0)
            row["b0_contact_mm"] = float(1000 * np.linalg.norm(mrel[:2] - np.array(hw_c[q][:2])))
            # release TCP minus hole in the board frame (for the pooled-over-recordings reading of B0)
            hk = np.array(layout["parts"][q]["xy"])
            row["b0_vec_contact"] = (1000 * (_rot2(-np.radians(f["yaw_deg"])) @ (mrel[:2] - np.array(f["pos"][:2]))
                                             - hk)).round(3).tolist()
            if mb is not None:
                row["b0_mocap_mm"] = float(1000 * np.linalg.norm(mrel[:2] - np.array(world_holes(mb, layout)[q][:2])))
                row["b0_vec_mocap"] = (1000 * ((mb["R"].T @ (mrel - mb["pos"]))[:2] - hk)).round(3).tolist()
        lopo[q] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}
    summ = {}
    for key in ("contact_close_mm", "mocap_close_mm", "contact_xy_mm", "mocap_xy_mm", "b0_contact_mm",
                "b0_mocap_mm", "contact_vs_mocap_hole_mm"):
        vals = [r[key] for r in lopo.values() if key in r]
        if vals:
            summ[key] = dict(median=round(float(np.median(vals)), 3), n_parts=len(vals))
    fails = [e for e in events if e["kind"] == "insert_fail" and e["part"] in layout["parts"]]
    fail_stats = None
    if full is not None and fails:
        ef = _event_errors(fails, world_holes(full, layout))
        fail_stats = dict(n=len(fails), close_median_mm=round(float(np.median([x["close"] for x in ef])), 2),
                          xy_median_mm=round(float(np.median([x["xy"] for x in ef])), 2))
    cmp = None
    if full is not None and mb is not None:
        hc, hm = world_holes(full, layout), world_holes(mb, layout)
        dd = [1000 * np.linalg.norm(np.array(hc[q][:2]) - np.array(hm[q][:2])) for q in layout["parts"]]
        dyaw = (full["yaw_deg"] - mb["yaw_deg"] + 180) % 360 - 180
        cmp = dict(hole_shift_median_mm=round(float(np.median(dd)), 2), hole_shift_max_mm=round(float(max(dd)), 2),
                   dyaw_deg=round(float(dyaw), 3), dz_mm=round(1000 * (full["pos"][2] - float(mb["pos"][2])), 2),
                   mocap_sane=mb["sane"], mocap_tilt_deg=round(mb["tilt_deg"], 2))
    return dict(episode=int(ep), recording=rec, split=split_of(ep), n_parts=len(parts),
                n_events=len(fit_ev), kinds={k: sum(e["kind"] == k for e in fit_ev) for k in FIT_KINDS},
                lopo=lopo, summary=summ, contact_vs_mocap=cmp, failed_inserts=fail_stats,
                drift=drift(events, layout, params),
                full_fit=None if full is None else {k: full[k] for k in
                                                   ("pos", "yaw_deg", "sigma", "n_parts", "n_events",
                                                    "close_resid_median_mm", "n_outliers")})


def evaluate(episodes: list[int], layout_eps: list[int], params: dict | None = None, loro: bool = True,
             layout_source: str = "contact") -> dict:
    """Evaluate many recordings. A recording in layout_eps gets a layout fitted without it (leave-one-recording-out).
    layout_source 'nist' places the NIST laser-plate layout with the (train) contact layout and uses that."""
    def make(eps):
        lay = fit_layout(eps, params)
        return nist_layout(lay) if layout_source == "nist" else lay
    base = make(layout_eps)
    per = []
    for ep in episodes:
        lay = make([e for e in layout_eps if e != ep]) if (loro and ep in layout_eps) else base
        per.append(evaluate_recording(ep, lay, params))
    pooled = {}
    for key in ("contact_close_mm", "mocap_close_mm", "contact_xy_mm", "mocap_xy_mm", "b0_contact_mm",
                "b0_mocap_mm", "contact_vs_mocap_hole_mm"):
        vals = [r[key] for p in per for r in p["lopo"].values() if key in r]
        recm = [p["summary"][key]["median"] for p in per if key in p["summary"]]
        if vals:
            pooled[key] = dict(median_over_parts=round(float(np.median(vals)), 3),
                               p90_over_parts=round(float(np.percentile(vals, 90)), 3), n_parts=len(vals),
                               median_of_recording_medians=round(float(np.median(recm)), 3), n_recordings=len(recm))
    # B0 read as "per part, the median release TCP over all held-out recordings" (board frame, each recording
    # mapped with its own pose: contact leave-one-part-out, or mocap), then the median over parts
    for src in ("contact", "mocap"):
        by_part = {}
        for p in per:
            for q, r in p["lopo"].items():
                if f"b0_vec_{src}" in r:
                    by_part.setdefault(q, []).append(r[f"b0_vec_{src}"])
        if by_part:
            d = {q: float(np.linalg.norm(np.median(np.array(v), axis=0))) for q, v in by_part.items()}
            pooled[f"b0_pooled_{src}_mm"] = dict(
                median_over_parts=round(float(np.median(list(d.values()))), 3), n_parts=len(d),
                n_releases=int(sum(len(v) for v in by_part.values())),
                per_part={q: dict(mm=round(v, 2), n_recordings=len(by_part[q])) for q, v in sorted(d.items())})
    return dict(layout=dict(source=layout_source, episodes=base["episodes"], n_events=base["n_events"], fit=base["fit"],
                            nist_fit=base.get("nist_fit")),
                params=dict(PARAMS, **(params or {})), recordings=per, pooled=pooled)


# ------------------------------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("layout")
    s.add_argument("--out", type=Path, default=OUT_DEFAULT / "layout_contact.json")
    s = sub.add_parser("fit")
    s.add_argument("recording")
    s = sub.add_parser("eval")
    s.add_argument("--split", default="gate")
    a = ap.parse_args()
    train = episodes_in("train")
    if a.cmd == "layout":
        lay = fit_layout(train)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(lay, indent=1))
        print(json.dumps(lay["fit"]), json.dumps(lay["mocap_correction_summary"]))
    elif a.cmd == "fit":
        ep = episode_of(a.recording)
        lay = fit_layout([e for e in train if e != ep])
        print(json.dumps({k: v for k, v in fit_board(extract_events(ep), lay).items() if k != "events"}, indent=1))
    else:
        r = evaluate(episodes_in(a.split), train)
        print(json.dumps(r["pooled"], indent=1))


if __name__ == "__main__":
    main()
