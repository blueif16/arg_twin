"""Watcher tools for stage B0 (world frame): re-measure the checks independently, find what they can detect,
and triangulate marked points. The method is in playbook/skills/auditor/references/check-validation.md.

measure_b0         B0.overlay_hama1 / B0.overlay_hama2 / B0.overlay_hand / B0.tcp_err_mm, measured as
                   playbook/validation.yaml defines them, on held-out frames drawn with a seed. The draw skips
                   every frame the fitting touched (the world tools' defaults plus the frames named in the call
                   log's arguments) and any frame within `radius` of one.
b0_known_error     the same metrics under known errors (camera translation and rotation, board shift and turn,
                   tool length) plus negative controls (robot rendered at the wrong time, wrong camera). Returns
                   response curves, px per mm, and the smallest error each check detects at its threshold.
triangulate_marks  3D points from pixel marks in two calibrated static cameras. Reports per-view reprojection,
                   distance to model points (board frame) in mm, and the rigid board shift the marks imply.

Only data access (frames, logged state) and the MuJoCo render come from twin.world. Edge maps, boundaries,
forward kinematics, projection and aggregation are written again here, so a bug in the fitter's scoring code
can't hide in both. Outputs go to runs/<brain>/<run_id>/b0-watch/ (JSON, PNG).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from harness.registry import REPO, tool

os.environ.setdefault("MUJOCO_GL", "glfw")
SEED = 20261024
WATCHER_CALLERS = ("b0-watcher",)  # this watcher's own calls never count as fitting
FIT_TOOLS = ("refine_camera", "render_overlay", "real_frame", "check_world_frame", "save_world_frame")
THRESH = {"B0.overlay_hama1": 3.0, "B0.overlay_hama2": 3.0, "B0.overlay_hand": 6.0, "B0.tcp_err_mm": 2.0}


def _w():
    from twin import world  # lazy: needs MUJOCO_GL and a GL context
    return world


def _out_dir(brain: str, run_id: str) -> Path:
    p = REPO / "runs" / brain / run_id / "b0-watch"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _rel(p) -> str:
    return str(Path(p).resolve().relative_to(REPO))


def _r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


# ----------------------------------------------------------------------------------------------
# Frames the fit touched, and a held-out draw that avoids them
# ----------------------------------------------------------------------------------------------
def _still_picks(episode: int, n: int) -> list[int]:
    w = _w()
    st = w.still_frames(episode, max_speed=0.01, max_rate_deg=3.0, min_gap=30)
    return [st[i] for i in np.linspace(0, len(st) - 1, min(n, len(st))).astype(int)] if st else []


def _hand_median_frames(episode: int, n: int = 61) -> list[int]:
    s = _w().episode_state(episode)
    g = s.grip.sum(1)
    out = []
    for sel in (g > g.max() - 0.002, g < g.min() + 0.002):
        idx = np.flatnonzero(sel)
        if len(idx) >= 20:
            out += idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)].tolist()
    return out


def frames_used_by_fit(episode: int, log_dir: Path | None = None) -> dict[str, list[int]]:
    """Every frame the world-frame fitting could have looked at, by source. Reconstructed from the tools'
    deterministic frame choices (twin/world.py, harness/tools/world.py) and the *arguments* of logged calls in
    runs/calls/ (never the actor's reasoning). Calls by WATCHER_CALLERS are ignored."""
    w = _w()
    used: dict[str, set] = {}

    def add(k, fr):
        used.setdefault(k, set()).update(int(f) for f in fr)

    add("background_and_static_mask_41", w.sample_frames(episode, 41))  # used inside every static refine
    add("hand_open_closed_medians", _hand_median_frames(episode))      # used inside every hand refine
    for path in sorted((log_dir or REPO / "runs" / "calls").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            a = row.get("args") or {}
            if row.get("tool") not in FIT_TOOLS or row.get("caller") in WATCHER_CALLERS or a.get("episode") != episode:
                continue
            t, body = row["tool"], a.get("camera") == "hand"
            if t == "refine_camera":
                n = int(a.get("n_fit_frames", 16))
                add(f"refine_{a.get('camera')}", _still_picks(episode, 2 * n + 1) if body else w.sample_frames(episode, 2 * n - 1))
            elif t == "render_overlay":
                if a.get("frames"):
                    add("render_overlay_frames", a["frames"])
                else:
                    n = int(a.get("n_frames", 6))
                    add("render_overlay_default", _still_picks(episode, n) if body else w.sample_frames(episode, n))
                if a.get("score_frames"):
                    n = int(a["score_frames"])
                    add("render_overlay_score", _still_picks(episode, n) if body else w.sample_frames(episode, n))
            elif t == "check_world_frame":
                n = int(a.get("n_frames", 15))
                add("check_world_frame", w.sample_frames(episode, 2 * n + 1)[1::2] + _still_picks(episode, n))
            elif t == "real_frame":
                f = a.get("frame", None if a.get("t") is None else round(a["t"] * w.FPS))
                if f is not None:
                    add("real_frame", [f])
            for k in (a.get("keypoints") or []) + (a.get("points") or []):
                if isinstance(k, dict) and "frame" in k:
                    add("keypoint_frames", [k["frame"]])
    return {k: sorted(v) for k, v in used.items()}


def pick_heldout(episode: int, n: int, seed: int, exclude, radius: int = 30, ok: np.ndarray | None = None,
                 margin_s: float = 1.0) -> list[int]:
    """Stratified seeded draw: the episode is cut into n equal spans and one free frame is drawn per span.
    A frame is free if it is > radius frames from every excluded frame and ok[f] (optional filter)."""
    w = _w()
    N = w.episode_state(episode).n
    free = np.ones(N, bool)
    for f in exclude:
        free[max(0, int(f) - radius):int(f) + radius + 1] = False
    m = int(margin_s * w.FPS)
    free[:m] = free[N - m:] = False
    if ok is not None:
        free &= ok
    cand = np.flatnonzero(free)
    edges = np.linspace(0, N, n + 1)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        c = cand[(cand >= edges[i]) & (cand < edges[i + 1])]
        if len(c):
            out.append(int(rng.choice(c)))
    return out


def hand_view_ok(episode: int, board_pos, board_quat, z_range=(0.06, 0.25), max_speed=0.02) -> np.ndarray:
    """Frames where the tool is over the board plate, 6-25 cm above it and moving slowly: the wrist camera
    then sees board features (and video/log latency can't move them much)."""
    w = _w()
    s = w.episode_state(episode)
    R = w.quat2mat(board_quat)
    tb = (s.ee[:, :3] - np.asarray(board_pos)) @ R
    (x0, x1), (y0, y1) = w.BOARD_PLATE["x"], w.BOARD_PLATE["y"]
    v = np.linalg.norm(np.gradient(s.ee[:, :3], axis=0), axis=1) * w.FPS
    return ((tb[:, 0] > x0) & (tb[:, 0] < x1) & (tb[:, 1] > y0) & (tb[:, 1] < y1)
            & (tb[:, 2] > z_range[0]) & (tb[:, 2] < z_range[1]) & (v < max_speed))


# ----------------------------------------------------------------------------------------------
# Own geometry: edges, boundaries, FK, projection, perturbations
# ----------------------------------------------------------------------------------------------
def edge_dt(img: np.ndarray, precise: bool = True) -> np.ndarray:
    """Distance (px) to the nearest Canny edge. Same blur and thresholds as the frozen check (5x5, sigma 1.2,
    Canny 30/90) so values are comparable; exact Euclidean distance unless precise=False (3x3 mask)."""
    import cv2
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 1.2)
    e = cv2.Canny(g, 30, 90)
    return cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2,
                                 cv2.DIST_MASK_PRECISE if precise else 3)


def inner_boundary(mask: np.ndarray, border: int = 2) -> np.ndarray:
    import cv2
    m = mask.astype(bool)
    b = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if border > 0:
        b[:border, :] = b[-border:, :] = b[:, :border] = b[:, -border:] = False
    return b


class Kin:
    """Own forward kinematics on base.xml (hand body movable along the flange axis to model tool length)."""

    def __init__(self):
        import mujoco
        from twin import BASE_XML
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(BASE_XML))
        self.d = mujoco.MjData(self.m)
        self.hand = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "gripper/hand")
        self.hand_pos0 = self.m.body_pos[self.hand].copy()
        self.tcp = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")

    def set(self, q, grip, hand_dz=0.0):
        R = np.zeros(9)
        self.mj.mju_quat2Mat(R, self.m.body_quat[self.hand])
        self.m.body_pos[self.hand] = self.hand_pos0 + R.reshape(3, 3) @ np.array([0, 0, hand_dz])
        self.d.qpos[:7], self.d.qpos[7:9] = q, grip
        self.mj.mj_kinematics(self.m, self.d)

    def body(self, name):
        if name == "world":
            return np.eye(3), np.zeros(3)
        b = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_BODY, name)
        return self.d.xmat[b].reshape(3, 3).copy(), self.d.xpos[b].copy()

    def tcp_pose(self):
        return self.d.site_xmat[self.tcp].reshape(3, 3).copy(), self.d.site_xpos[self.tcp].copy()


def cam_from_calib(d: dict):
    """Camera from the calib file's numbers only: T_parent_cam_opencv + fx fy cx cy."""
    w = _w()
    T = np.asarray(d["T_parent_cam_opencv"], float)
    return w.CameraModel.from_cv(d["name"], T[:3, :3], T[:3, 3], d["fx"], d["fy"], d["cx"], d["cy"],
                                 parent=d.get("parent", "world"))


def project_cv(cam, P_parent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Own pinhole projection of points given in the camera's parent frame."""
    R = cam.R_cv
    Xc = (np.atleast_2d(P_parent) - np.asarray(cam.pos)) @ R
    z = Xc[:, 2]
    return np.stack([cam.fx * Xc[:, 0] / z + cam.cx, cam.fy * Xc[:, 1] / z + cam.cy], 1), z


def perturb_cam(cam, t_cam_mm=(0, 0, 0), r_cam_deg=(0, 0, 0)):
    """Move the camera in its own OpenCV axes (x right, y down, z forward) and turn it about them."""
    w = _w()
    R = cam.R_cv
    pos = np.asarray(cam.pos) + R @ (np.asarray(t_cam_mm, float) / 1000)
    R2 = R @ w.rotvec2mat(np.radians(np.asarray(r_cam_deg, float)))
    return w.CameraModel.from_cv(cam.name, R2, pos, cam.fx, cam.fy, cam.cx, cam.cy, parent=cam.parent)


def perturb_board(pos, quat, t_board_mm=(0, 0, 0), r_board_deg=(0, 0, 0)):
    """Shift the board in its own axes; turn it about its axes through the plate centre."""
    w = _w()
    R = w.quat2mat(quat)
    c = np.array([np.mean(w.BOARD_PLATE["x"]), np.mean(w.BOARD_PLATE["y"]), w.BOARD_PLATE["top_z"]])
    R2 = R @ w.rotvec2mat(np.radians(np.asarray(r_board_deg, float)))
    p2 = np.asarray(pos) + R @ (np.asarray(t_board_mm, float) / 1000) + R @ c - R2 @ c
    return p2, w.mat2quat(R2)


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
N_ORIENT = 8


def oriented_edge_dts(img: np.ndarray) -> np.ndarray:
    """Directional chamfer maps (Liu et al., CVPR 2010): Canny edges split into N_ORIENT gradient-orientation
    bins (mod 180 deg), one distance map per bin. A rendered outline pixel is then matched only to real edges
    running roughly parallel to it, which removes most chance matches in a cluttered scene."""
    import cv2
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 1.2)
    e = cv2.Canny(g, 30, 90) > 0
    gf = g.astype(np.float32)
    th = np.degrees(np.arctan2(cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3), cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3))) % 180
    b = (th / (180 / N_ORIENT)).astype(int) % N_ORIENT
    return np.stack([cv2.distanceTransform((~(e & (b == k))).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                     for k in range(N_ORIENT)])


def oriented_dist(odt: np.ndarray, mask: np.ndarray, bnd: np.ndarray) -> np.ndarray:
    """Distance from each outline pixel to the nearest real edge within +-1 orientation bin of the outline's
    own normal direction."""
    import cv2
    if not bnd.any():
        return np.zeros(0)
    mf = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 1.5)
    th = np.degrees(np.arctan2(cv2.Sobel(mf, cv2.CV_32F, 0, 1, ksize=3), cv2.Sobel(mf, cv2.CV_32F, 1, 0, ksize=3))) % 180
    ys, xs = np.nonzero(bnd)
    k = (th[ys, xs] / (180 / N_ORIENT)).astype(int) % N_ORIENT
    return np.minimum(np.minimum(odt[k, ys, xs], odt[(k + 1) % N_ORIENT, ys, xs]), odt[(k - 1) % N_ORIENT, ys, xs])


class StaticScorer:
    """B0.overlay_hama*: per frame, the rendered robot (arm + hand) boundary and board-plate boundary (board
    pixels next to the robot dropped) are scored by distance to the nearest real Canny edge; the frame value is
    the median over the pooled boundary pixels; the check value is the median over frames."""

    def __init__(self, episode: int, camera: str, frames: list[int]):
        w = _w()
        self.episode, self.camera, self.frames = episode, camera, list(frames)
        self.real = {f: w.read_frame(episode, camera, f) for f in self.frames}
        self.dt = {f: edge_dt(self.real[f]) for f in self.frames}
        self.dt3 = {f: edge_dt(self.real[f], precise=False) for f in self.frames}
        self.odt = {f: oriented_edge_dts(self.real[f]) for f in self.frames}
        self.s = w.episode_state(episode)

    def score(self, cam, board_pos, board_quat, hand_dz=0.0, q_shift=0, keep_labels=False) -> dict:
        import cv2
        w = _w()
        view = w.get_view()
        view.set_board(board_pos, board_quat)
        view.set_hand_offset(hand_dz)
        per, labels = [], {}
        for f in self.frames:
            fq = int(np.clip(f + q_shift, 0, self.s.n - 1))
            view.set_robot(self.s.q[fq], self.s.grip[fq])
            lab = view.render_labels(cam)
            robot = np.isin(lab, [w.LABELS["arm"], w.LABELS["hand"]])
            rb = inner_boundary(robot)
            near = cv2.dilate(robot.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            bb = inner_boundary(lab == w.LABELS["board"]) & ~near
            dt, dt3 = self.dt[f], self.dt3[f]
            dr, db = dt[rb], dt[bb]
            dc = np.concatenate([dr, db])
            dc3 = np.concatenate([dt3[rb], dt3[bb]])
            orr = oriented_dist(self.odt[f], robot, rb)
            orb = oriented_dist(self.odt[f], lab == w.LABELS["board"], bb)
            per.append({"frame": f, "combined": float(np.median(dc)) if dc.size else np.nan,
                        "orient_robot_mean10": float(np.minimum(orr, 10).mean()) if orr.size else np.nan,
                        "orient_board_mean10": float(np.minimum(orb, 10).mean()) if orb.size else np.nan,
                        "combined_mask3": float(np.median(dc3)) if dc3.size else np.nan,
                        "robot": float(np.median(dr)) if dr.size else np.nan,
                        "board": float(np.median(db)) if db.size else np.nan,
                        "mean_clip10": float(np.minimum(dc, 10).mean()) if dc.size else np.nan,
                        "within1px": float((dc <= 1.0).mean()) if dc.size else np.nan,
                        "n_robot_px": int(dr.size), "n_board_px": int(db.size)})
            if keep_labels:
                labels[f] = lab
        view.set_hand_offset(0.0)
        out = {"per_frame": per}
        for k in ("combined", "combined_mask3", "robot", "board", "mean_clip10", "within1px",
                  "orient_robot_mean10", "orient_board_mean10"):
            v = np.array([p[k] for p in per], float)
            v = v[np.isfinite(v)]
            out[k] = float(np.median(v)) if v.size else np.nan
        out["ci95_combined"] = bootstrap_median_ci([p["combined"] for p in per])
        if keep_labels:
            out["_labels"] = labels
        return out


def bootstrap_median_ci(vals, n_boot: int = 2000, seed: int = SEED) -> list:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size < 3:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    meds = np.median(v[rng.integers(0, v.size, (n_boot, v.size))], axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def hand_residuals(kin: Kin, cam, episode: int, kps: list[dict], board_pos, board_quat, hand_dz=0.0) -> np.ndarray:
    """Reprojection error (px) of marked wrist-camera keypoints. Each kp: {frame, uv, and one of board:[x,y,z]
    (NIST_Board1 frame), world:[x,y,z], or body:<name> + p:[x,y,z] (point fixed on a robot body, e.g. a finger)}."""
    w = _w()
    s = w.episode_state(episode)
    Rb = w.quat2mat(board_quat)
    out = []
    for k in kps:
        f = int(k["frame"])
        kin.set(s.q[f], s.grip[f], hand_dz)
        if "board" in k:
            P = Rb @ np.asarray(k["board"], float) + np.asarray(board_pos)
        elif "world" in k:
            P = np.asarray(k["world"], float)
        else:
            Rk, pk = kin.body(k["body"])
            P = Rk @ np.asarray(k["p"], float) + pk
        Rc, pc = kin.body(cam.parent)
        uv, z = project_cv(cam, (Rc.T @ (P - pc))[None])
        out.append(float(np.hypot(*(uv[0] - np.asarray(k["uv"], float)))) if z[0] > 0 else 500.0)
    return np.array(out)


def finger_corner_residuals(cam, episode: int, frames, dark: float = 70.0, view=None) -> list[dict]:
    """Finger keypoints for a camera on the hand: the top corners (inner and outer) of each rendered finger-pad
    blob vs the same corners of the matching dark blob in the real frame (lower half, gray < dark). Returns one
    row per corner with the px distance; frames where a real pad blob can't be matched are skipped."""
    import cv2
    w = _w()
    view = view or w.get_view()
    pads = view.pad_geoms
    s = w.episode_state(episode)
    out = []

    def corners(mask):
        ys, xs = np.nonzero(mask)
        if xs.size < 30:
            return None
        return {"top_left": np.array([xs[np.argmin(ys + xs)], ys[np.argmin(ys + xs)]], float),
                "top_right": np.array([xs[np.argmin(ys - xs)], ys[np.argmin(ys - xs)]], float)}
    for f in frames:
        view.set_robot(s.q[f], s.grip[f])
        ids = view.render_geom_ids(cam)
        rend = np.isin(ids, pads)
        rend[: w.H // 2] = False
        real = cv2.cvtColor(w.read_frame(episode, cam.name, f), cv2.COLOR_RGB2GRAY) < dark
        real[: w.H // 2] = False
        real = cv2.morphologyEx(real.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        nr, lr = cv2.connectedComponents(rend.astype(np.uint8))
        nq, lq = cv2.connectedComponents(real.astype(np.uint8))
        for i in range(1, nr):
            m = lr == i
            if m.sum() < 200:
                continue
            side = "left" if np.nonzero(m)[1].mean() < w.W / 2 else "right"
            ov = [(np.logical_and(m, lq == j).sum(), j) for j in range(1, nq)]
            ov = [o for o in ov if o[0] > 0.3 * m.sum() and 0.5 * m.sum() < (lq == o[1]).sum() < 2.0 * m.sum()]
            if not ov:
                continue
            j = max(ov)[1]
            cr, cq = corners(m), corners(lq == j)
            if cr is None or cq is None:
                continue
            # inner top corner only (the outer side merges with the finger housing in the real image)
            for k in (("top_right",) if side == "left" else ("top_left",)):
                out.append({"frame": int(f), "pad": side, "corner": k, "render": cr[k].tolist(), "real": cq[k].tolist(),
                            "px": float(np.linalg.norm(cr[k] - cq[k]))})
    return out


def tcp_errors(kin: Kin, episode: int, frames, hand_dz=0.0) -> tuple[np.ndarray, np.ndarray]:
    """Twin tool tip (site 'tcp' by own FK at the logged joints) vs logged O_T_EE: position (mm), angle (deg)."""
    w = _w()
    s = w.episode_state(episode)
    dp, da = [], []
    for f in frames:
        kin.set(s.q[f], s.grip[f], hand_dz)
        R, p = kin.tcp_pose()
        dp.append(np.linalg.norm(p - s.ee[f, :3]) * 1000)
        Re = w.quat2mat(s.ee[f, 3:])
        da.append(np.degrees(np.arccos(np.clip((np.trace(R.T @ Re) - 1) / 2, -1, 1))))
    return np.array(dp), np.array(da)


def px_per_mm(cam, P_world: np.ndarray) -> dict:
    """Image motion (px) of a world point per mm of its own motion: largest and smallest singular value of the
    projection Jacobian (the smallest is the direction along the viewing ray)."""
    uv0, z = project_cv(cam, P_world[None])
    J = np.zeros((2, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = 1e-3
        J[:, i] = (project_cv(cam, (P_world + d)[None])[0][0] - uv0[0]) / 1.0
    sv = np.linalg.svd(J, compute_uv=False)
    return {"depth_m": _r(z[0]), "px_per_mm_lateral": _r(sv[0]), "mm_per_px_lateral": _r(1 / sv[0]),
            "px_per_mm_worst_dir": _r(sv[1]), "px_per_deg_camera_turn": _r(cam.fx * np.pi / 180, 2)}


# ----------------------------------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------------------------------
def _calib(episode: int):
    w = _w()
    rec = w.recording_of(episode)
    c = w.load_calib(rec)
    if not c:
        raise ValueError(f"no twin/calib/{rec}.json")
    return rec, c


# The unsuffixed marks files in b0-watch/ (hand_keypoints.json, static_robot_keypoints.json, measure_b0.json) were
# made on episode 0, the only calibrated recording when they were written. Other recordings use <stem>_<recording>.
LEGACY_RECORDING = "2025-01-09-14-11-58"


def _saved(out_dir: Path, rec: str, stem: str):
    """(content, path) of a saved watcher file for this recording: <stem>_<recording>.json, else the unsuffixed
    legacy file for episode 0 only; (None, None) if there is none."""
    for p in [out_dir / f"{stem}_{rec}.json"] + ([out_dir / f"{stem}.json"] if rec == LEGACY_RECORDING else []):
        if p.exists():
            return json.loads(p.read_text()), _rel(p)
    return None, None


def twin_board(rec: str, c: dict) -> dict:
    """The board the twin uses for this recording, from twin/calib/<recording>.json:
    - pose: "board_contact" (x, y, yaw from robot contacts; z = the mocap gauge), as twin/replay.py board_pose_for
      ('calib_contact') reads it;
    - plate top: ReplayParams.surface_z (30.3 mm) above that z, as replay, starts and the scene builder place it;
    - holes: board_xy per part from the contact-layout file the entry names (board_contact.file).
    Returned pose = the board frame raised to the plate top (plate top at z = 0, the convention of TwinView.set_board
    and of board keypoints). Without a board_contact entry it falls back to the calib "board" entry and says so."""
    w = _w()
    try:
        from twin.replay import ReplayParams
        sz = float(ReplayParams().surface_z)
    except Exception:  # noqa: BLE001
        sz = 0.0303
    cal = c.get("board") or {}
    b = c.get("board_contact")
    if b:
        q = np.asarray(b["quat"], float)
        pos = np.asarray(b["pos"], float) + w.quat2mat(q) @ np.array([0.0, 0.0, sz])
        holes, hf = {}, b.get("file")
        if hf and (REPO / hf).exists():
            holes = {k: list(v["board_xy"]) for k, v in json.loads((REPO / hf).read_text()).get("holes", {}).items()
                     if isinstance(v, dict) and v.get("board_xy")}
        src = (f"twin/calib/{rec}.json board_contact (x, y, yaw from robot contacts; z mocap gauge) + surface_z "
               f"{sz * 1000:.1f} mm (twin/replay.py ReplayParams); holes: {hf or 'none'}")
    else:
        q, pos, holes, hf = np.asarray(cal["quat"], float), np.asarray(cal["pos"], float), {}, None
        src = f"twin/calib/{rec}.json board (no board_contact entry; plate top at its z)"
    out = dict(pos=pos, quat=q, holes=holes, holes_file=hf, surface_z=sz, source=src)
    if cal:
        cp = np.asarray(cal["pos"], float)
        out["calib_board"] = dict(pos=cp, quat=np.asarray(cal["quat"], float), source=cal.get("source", ""),
                                  delta_mm=[_r(1000 * v, 2) for v in (pos - cp)])
    return out


def resolve_board_keypoints(kps: list[dict], holes: dict) -> tuple[list[dict], list[str]]:
    """Board keypoints that name a hole ("hole": part, or a label '<part> hole ...') take the twin's hole xy for that
    part (their z, height above the plate top, is kept); other keypoints are unchanged. Returns the keypoints and,
    per keypoint, where its 3D point came from."""
    import re
    out, src = [], []
    for k in kps:
        if "board" not in k:
            out.append(k)
            src.append("world" if "world" in k else f"body {k.get('body')}")
            continue
        part = k.get("hole")
        if part is None:
            m = re.match(r"\s*(.+?)\s+hole\b", str(k.get("label", "")), re.I)
            part = m.group(1) if m else None
        if part in holes:
            out.append({**k, "board": [holes[part][0], holes[part][1], float(k["board"][2])]})
            src.append(f"twin hole: {part}")
        else:
            out.append(k)
            src.append("board coords as marked")
    return out, src


@tool("measure_b0",
      "Watcher: re-measure B0.overlay_hama1, B0.overlay_hama2, B0.overlay_hand and B0.tcp_err_mm from "
      "twin/calib/<recording>.json (numbers only) on held-out frames drawn with a seed, skipping every frame the "
      "fitting touched (reconstructed from the world tools' defaults and the call log's arguments) +- radius. "
      "Works for any calibrated recording. The board is the twin's: calib board_contact (x, y, yaw from contacts) "
      "with the plate top surface_z above it, and hole keypoints take the twin's contact-layout hole xy; the "
      "output names the source and also gives the part median under the calib 'board' entry with the marked "
      "coordinates (the previous method). The recording's residual tool offset (calib tcp.hand_offset_m minus "
      "base.xml's, 0 for episode 0) is applied to renders and hand points; latency_frames is not modelled. "
      "Hand keypoints are marked by the watcher on its own frames (board/world/body 3D points); if omitted, "
      "hand_keypoints_<recording>.json / static_robot_keypoints_<recording>.json in the output folder are reused "
      "(for episode 0 also the unsuffixed files). static_keypoints medians per camera are what B0.kp_hama1/2 "
      "measure. Writes measure_b0_<recording>.json and per-camera overlay PNGs under "
      "runs/<brain>/<run_id>/b0-watch/. It does not record checks; pass the values to record_check.",
      {"type": "object", "properties": {
          "episode": {"type": "integer"},
          "n_frames": {"type": "integer", "default": 24, "description": "static-camera held-out frames"},
          "seed": {"type": "integer", "default": SEED},
          "radius": {"type": "integer", "default": 30, "description": "frames kept away from any fit frame"},
          "hand_keypoints": {"type": "array", "items": {"type": "object"},
                             "description": "[{frame, uv:[u,v], board:[x,y,z] | world:[x,y,z] | body:name, p:[x,y,z], "
                                            "label?, kind?: 'part'|'finger'}] marked on held-out frames"},
          "n_finger_frames": {"type": "integer", "default": 12,
                              "description": "held-out open-gripper frames for the automatic finger-pad corner keypoints (0 = none)"},
          "finger_radius": {"type": "integer", "description": "fit-frame distance for the finger draw (default: radius). "
                            "Short recordings may need less; frame_counts shows what each component got"},
          "n_board_view_frames": {"type": "integer", "default": 0,
                                  "description": "also draw this many held-out wrist frames with the tool 6-25 cm over the "
                                                 "twin's plate and slow (candidates for marking board points); returned only"},
          "static_keypoints": {"type": "object",
                               "description": "diagnostic, not a B0 check: {hama1|hama2: [{frame, uv, body, p | world, label}]} marked "
                                              "robot points (e.g. finger pads) in the static cameras; reprojection px per camera"},
          "brain": {"type": "string", "default": "claude"}, "run_id": {"type": "string", "default": "run-standin-1"}},
       "required": ["episode"]})
def measure_b0(episode: int, n_frames: int = 24, seed: int = SEED, radius: int = 30, hand_keypoints=None,
               n_finger_frames: int = 12, static_keypoints=None, brain: str = "claude", run_id: str = "run-standin-1",
               finger_radius: int | None = None, n_board_view_frames: int = 0) -> dict:
    import cv2
    w = _w()
    t0 = time.time()
    rec, c = _calib(episode)
    out_dir = _out_dir(brain, run_id)
    # marks saved by an earlier watcher run are reused, so a rerun reproduces the numbers
    marks_files = {}
    if hand_keypoints is None:
        hand_keypoints, marks_files["hand_keypoints"] = _saved(out_dir, rec, "hand_keypoints")
    if static_keypoints is None:
        static_keypoints, marks_files["static_keypoints"] = _saved(out_dir, rec, "static_robot_keypoints")
    used = frames_used_by_fit(episode)
    excl = sorted({f for v in used.values() for f in v})
    frames = pick_heldout(episode, n_frames, seed, excl, radius)
    tb = twin_board(rec, c)
    bp, bq = tb["pos"], tb["quat"]
    hdz = w.calib_hand_offset(rec, c)  # residual tool offset on top of base.xml (0 for episode 0)
    finger_radius = radius if finger_radius is None else int(finger_radius)
    res = {"recording": rec, "episode": episode, "brain": brain, "run_id": run_id, "seed": seed, "radius": radius,
           "finger_radius": finger_radius, "frames_static": frames,
           "excluded_sources": {k: len(v) for k, v in used.items()}, "n_excluded": len(excl),
           "board": {"source": tb["source"], "pos": [_r(v, 5) for v in bp], "quat_wxyz": [_r(v, 7) for v in bq],
                     "surface_z_m": tb["surface_z"], "holes_file": tb["holes_file"],
                     "calib_board_entry": None if "calib_board" not in tb else
                     {"source": tb["calib_board"]["source"], "twin_minus_calib_board_mm": tb["calib_board"]["delta_mm"]}},
           "hand_offset_residual_m": _r(hdz, 5), "latency_frames": "not modelled",
           "marks_files": {k: v for k, v in marks_files.items() if v}, "checks": {}}
    for cam_name in ("hama1", "hama2"):
        if cam_name not in c["cameras"]:
            continue
        cam = cam_from_calib(c["cameras"][cam_name])
        sc = StaticScorer(episode, cam_name, frames)
        r = sc.score(cam, bp, bq, hand_dz=hdz, keep_labels=True)
        labels = r.pop("_labels")
        view = w.get_view()
        view.set_board(bp, bq)
        view.set_hand_offset(hdz)
        tiles = []
        for f in frames[:: max(1, len(frames) // 6)][:6]:
            view.set_frame(episode, f)
            tiles.append(w.overlay(sc.real[f], view.render_rgb(cam), labels[f],
                                   title=f"{cam_name} f{f} held-out  {next(p['combined'] for p in r['per_frame'] if p['frame'] == f):.2f}px"))
        view.set_hand_offset(0.0)
        grid = np.concatenate([np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles) - len(tiles) % 3, 3)], 0)
        png = out_dir / f"heldout_overlay_{cam_name}_{rec}.png"
        cv2.imwrite(str(png), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        res["checks"][f"B0.overlay_{cam_name}"] = {
            "value": _r(r["combined"]), "ci95": [_r(x) for x in r["ci95_combined"]],
            "robot_only": _r(r["robot"]), "board_only": _r(r["board"]), "mask3_dt": _r(r["combined_mask3"]),
            "mean_clip10": _r(r["mean_clip10"]), "within1px": _r(r["within1px"]), "overlay": _rel(png),
            "per_frame": [{k: (_r(v) if isinstance(v, float) else v) for k, v in p.items()} for p in r["per_frame"]]}
    kin = Kin()
    dp, da = tcp_errors(kin, episode, frames)
    allf = list(range(0, w.episode_state(episode).n, 10))
    dpa, _ = tcp_errors(kin, episode, allf)
    res["checks"]["B0.tcp_err_mm"] = {"value": _r(np.median(dp)), "max": _r(dp.max()), "angle_deg_median": _r(np.median(da), 4),
                                      "every_10th_frame_median": _r(np.median(dpa)), "every_10th_frame_max": _r(dpa.max())}
    if "hand" in c["cameras"] and (hand_keypoints or n_finger_frames):
        cam = cam_from_calib(c["cameras"]["hand"])
        marked = list(hand_keypoints or [])
        kps, kp_src = resolve_board_keypoints(marked, tb["holes"])
        bad = [k["frame"] for k in kps if any(abs(int(k["frame"]) - e) <= radius for e in excl)]
        rp = hand_residuals(kin, cam, episode, kps, bp, bq, hand_dz=hdz) if kps else np.zeros(0)
        prev = None  # the previous method: calib "board" entry with the keypoints' marked coordinates
        if kps and "calib_board" in tb:
            rq = hand_residuals(kin, cam, episode, marked, tb["calib_board"]["pos"], tb["calib_board"]["quat"], hand_dz=hdz)
            prev = {"part_median": _r(np.median(rq), 2), "board": "twin/calib board entry, keypoint coordinates as marked",
                    "per_point_px": [_r(v, 2) for v in rq]}
        s_ = w.episode_state(episode)
        g = s_.grip.sum(1)
        ffr = pick_heldout(episode, n_finger_frames, seed + 3, excl, finger_radius, ok=(g > g.max() - 0.002)) if n_finger_frames else []
        fview = w.get_view()
        fview.set_hand_offset(hdz)
        fing = finger_corner_residuals(cam, episode, ffr, view=fview)
        fview.set_hand_offset(0.0)
        rf = np.array([r["px"] for r in fing])
        pooled = np.concatenate([rp, rf])
        meds = [float(np.median(x)) for x in (rp, rf) if x.size]
        res["checks"]["B0.overlay_hand"] = {
            "value": _r(max(meds), 2) if meds else None,
            "metric": "worse of finger_median and part_median (validation.yaml B0.overlay_hand, since v5)",
            "pooled_median": _r(np.median(pooled), 2) if pooled.size else None, "n": int(pooled.size),
            "ci95_pooled": [_r(x, 2) for x in bootstrap_median_ci(pooled)],
            "part_median": _r(np.median(rp), 2) if rp.size else None, "n_part": int(rp.size),
            "finger_median": _r(np.median(rf), 2) if rf.size else None, "n_finger": int(rf.size),
            "part_board_source": tb["source"], "part_previous_method": prev,
            "part_frames": sorted({int(k["frame"]) for k in kps}), "finger_frames": ffr,
            "frames_too_close_to_fit": bad,
            "per_point": [{"frame": int(k["frame"]), "label": k.get("label", ""), "kind": k.get("kind", "part"),
                           "uv": k["uv"], "point": s, "board": k.get("board"), "px": _r(v, 2)}
                          for k, s, v in zip(kps, kp_src, rp)],
            "fingers": [{k: (_r(v, 2) if k == "px" else v) for k, v in r.items()} for r in fing]}
    if static_keypoints:
        diag = {}
        for cam_name, kl in static_keypoints.items():
            cam = cam_from_calib(c["cameras"][cam_name])
            r = hand_residuals(kin, cam, episode, kl, bp, bq, hand_dz=hdz)
            vec = []
            for k in kl:
                f = int(k["frame"])
                kin.set(w.episode_state(episode).q[f], w.episode_state(episode).grip[f], hdz)
                if "world" in k:
                    P = np.asarray(k["world"], float)
                else:
                    Rk, pk = kin.body(k["body"])
                    P = Rk @ np.asarray(k["p"], float) + pk
                vec.append((np.asarray(k["uv"], float) - project_cv(cam, P[None])[0][0]).tolist())
            g = px_per_mm(cam, np.asarray(bp))
            diag[cam_name] = {"median_px": _r(np.median(r), 2), "n": len(r), "median_vector_px": [_r(v, 2) for v in np.median(vec, 0)],
                              "median_mm_at_board": _r(np.median(r) * g["mm_per_px_lateral"], 1),
                              "per_point": [{"frame": int(k["frame"]), "label": k.get("label", ""), "px": _r(v, 2)} for k, v in zip(kl, r)]}
        res["diagnostic_static_robot_keypoints"] = diag
    hb = res["checks"].get("B0.overlay_hand", {})
    fc = {"static": {"requested": n_frames, "drawn": len(frames), "radius": radius},
          "finger": {"requested": n_finger_frames if "hand" in c["cameras"] else 0, "drawn": len(hb.get("finger_frames", [])),
                     "corners_matched": hb.get("n_finger", 0), "radius": finger_radius},
          "part_keypoints": {"n": hb.get("n_part", 0), "frames": len(hb.get("part_frames", [])),
                             "too_close_to_fit": len(hb.get("frames_too_close_to_fit", []))},
          "static_keypoints": {k: v["n"] for k, v in (res.get("diagnostic_static_robot_keypoints") or {}).items()}}
    if n_board_view_frames:
        ok = hand_view_ok(episode, bp, bq)
        bvf = pick_heldout(episode, n_board_view_frames, seed + 5, excl, finger_radius, ok=ok)
        res["board_view_frames_to_mark"] = bvf
        fc["board_view"] = {"requested": n_board_view_frames, "drawn": len(bvf), "radius": finger_radius,
                            "eligible_frames": int(ok.sum())}
    res["frame_counts"] = fc
    res["thin_evidence"] = [f"{k}: {v['drawn']} of {v['requested']} frames" for k, v in fc.items()
                            if isinstance(v, dict) and "requested" in v and v["drawn"] < v["requested"]]
    if fc["finger"]["drawn"] and fc["finger"]["corners_matched"] < fc["finger"]["drawn"]:
        res["thin_evidence"].append(f"finger: {fc['finger']['corners_matched']} corners matched on {fc['finger']['drawn']} frames")
    res["seconds"] = round(time.time() - t0, 1)
    p = out_dir / f"measure_b0_{rec}.json"
    p.write_text(json.dumps(res, indent=1, default=float))
    return {"path": _rel(p), "recording": rec, "run_id": run_id, "frames_static": frames, "board": res["board"],
            "hand_offset_residual_m": res["hand_offset_residual_m"], "marks_files": res["marks_files"],
            "frame_counts": fc, "thin_evidence": res["thin_evidence"],
            **({"board_view_frames_to_mark": res["board_view_frames_to_mark"]} if n_board_view_frames else {}),
            "checks": {k: {kk: vv for kk, vv in v.items() if kk not in ("per_frame", "per_point", "fingers")} for k, v in res["checks"].items()},
            "diagnostic_static_robot_keypoints": {k: {kk: vv for kk, vv in v.items() if kk != "per_point"}
                                                  for k, v in (res.get("diagnostic_static_robot_keypoints") or {}).items()},
            "seconds": res["seconds"]}


def _sweep_specs(kind: str) -> list[dict]:
    S = []
    if kind == "static_camera":
        for ax, an in enumerate("xyz"):
            for m in (2, 5, 10, 20, 50):
                for sg in (1, -1):
                    t = [0, 0, 0]
                    t[ax] = sg * m
                    S.append({"family": f"cam_t{an}", "mag": m, "unit": "mm", "sign": sg, "cam_t": t})
            for m in (0.1, 0.3, 1, 2, 5):
                for sg in (1, -1):
                    r = [0, 0, 0]
                    r[ax] = sg * m
                    S.append({"family": f"cam_r{an}", "mag": m, "unit": "deg", "sign": sg, "cam_r": r})
    if kind in ("static_board", "hand"):
        for ax, an in enumerate("xyz"):
            for m in (1, 2, 3, 5, 10, 20):
                for sg in (1, -1):
                    t = [0, 0, 0]
                    t[ax] = sg * m
                    S.append({"family": f"board_t{an}", "mag": m, "unit": "mm", "sign": sg, "board_t": t})
        for ax, an in ((2, "yaw"), (0, "tilt")):
            for m in (0.5, 1, 2):
                for sg in (1, -1):
                    r = [0, 0, 0]
                    r[ax] = sg * m
                    S.append({"family": f"board_{an}", "mag": m, "unit": "deg", "sign": sg, "board_r": r})
        for m in (1, 2, 5, 10):
            for sg in (1, -1):
                S.append({"family": "tool_length", "mag": m, "unit": "mm", "sign": sg, "hand_dz": sg * m / 1000})
    if kind == "hand":
        for ax, an in enumerate("xyz"):
            for m in (1, 2, 5, 10):
                for sg in (1, -1):
                    t = [0, 0, 0]
                    t[ax] = sg * m
                    S.append({"family": f"cam_t{an}", "mag": m, "unit": "mm", "sign": sg, "cam_t": t})
            for m in (0.1, 0.3, 1, 2):
                for sg in (1, -1):
                    r = [0, 0, 0]
                    r[ax] = sg * m
                    S.append({"family": f"cam_r{an}", "mag": m, "unit": "deg", "sign": sg, "cam_r": r})
    return S


def _detect_table(rows: list[dict], thr: float, base: float, base_ci: list) -> list[dict]:
    """Per family: smallest magnitude at which the check fails (median > thr) for both signs, and for both signs
    with the bootstrap 95% lower bound above thr ('reliable'); also the smallest magnitude whose change from the
    baseline exceeds the baseline CI half-width (detectable change, ISO 11843 style)."""
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    half = (base_ci[1] - base_ci[0]) / 2 if base_ci and np.all(np.isfinite(base_ci)) else np.nan
    out = []
    for fam, rs in fams.items():
        mags = sorted({r["mag"] for r in rs})

        def first(pred):
            for m in mags:
                both = [r for r in rs if r["mag"] == m]
                if both and all(pred(r) for r in both):
                    return m
            return None
        out.append({"family": fam, "unit": rs[0]["unit"],
                    "fails_at": first(lambda r: r["value"] > thr),
                    "reliably_fails_at": first(lambda r: (r["ci95"][0] if "ci95" in r and np.isfinite(r["ci95"][0]) else r["value"]) > thr),
                    "change_detectable_at": first(lambda r: abs(r["value"] - base) > half) if np.isfinite(half) else None,
                    "curve": [{"mag": m, "values": [_r(r["value"]) for r in rs if r["mag"] == m]} for m in mags]})
    return out


@tool("b0_known_error",
      "Watcher: known-error tests for the B0 checks. Perturbs the calibrated twin by known amounts (camera "
      "translation 2-50 mm and rotation 0.1-5 deg in camera axes; board shift 1-20 mm and turn 0.5-2 deg; tool "
      "length 1-10 mm) and runs negative controls (robot rendered at the wrong time, the other camera's "
      "calibration), re-measuring each check on the held-out frames from measure_b0. Returns per-family response "
      "curves, the smallest error each check fails at its validation.yaml threshold, px per mm at the board, and "
      "writes b0_known_error_<recording>.json + response-curve figures under runs/<brain>/<run_id>/b0-watch/. "
      "Uses the twin's board and the recording's saved marks as measure_b0 does (never another recording's).",
      {"type": "object", "properties": {
          "episode": {"type": "integer"},
          "frames": {"type": "array", "items": {"type": "integer"}, "description": "held-out static frames (default: measure_b0's)"},
          "hand_keypoints": {"type": "array", "items": {"type": "object"}, "description": "as in measure_b0"},
          "brain": {"type": "string", "default": "claude"}, "run_id": {"type": "string", "default": "run-standin-1"}},
       "required": ["episode"]})
def b0_known_error(episode: int, frames=None, hand_keypoints=None, brain: str = "claude",
                   run_id: str = "run-standin-1") -> dict:
    w = _w()
    t0 = time.time()
    rec, c = _calib(episode)
    out_dir = _out_dir(brain, run_id)
    mb, _ = _saved(out_dir, rec, "measure_b0")
    mb = mb or {}
    if not frames:
        if "frames_static" not in mb:
            raise ValueError(f"no measure_b0 result for {rec} in {_rel(out_dir)}: run measure_b0 first or pass frames")
        frames = mb["frames_static"]
    tb = twin_board(rec, c)
    bp, bq = tb["pos"], tb["quat"]
    hdz = w.calib_hand_offset(rec, c)  # residual tool offset on top of base.xml (0 for episode 0)
    res = {"recording": rec, "episode": episode, "brain": brain, "run_id": run_id, "frames_static": frames,
           "board_source": tb["source"], "hand_offset_residual_m": _r(hdz, 5), "static": {}, "hand": {}, "tcp": {}}
    board_c = bp + w.quat2mat(bq) @ np.array([np.mean(w.BOARD_PLATE["x"]), np.mean(w.BOARD_PLATE["y"]), 0.0])
    for cam_name in ("hama1", "hama2"):
        cam = cam_from_calib(c["cameras"][cam_name])
        sc = StaticScorer(episode, cam_name, frames)
        base = sc.score(cam, bp, bq, hand_dz=hdz)
        rows = []
        for sp in _sweep_specs("static_camera") + _sweep_specs("static_board"):
            cm = perturb_cam(cam, sp.get("cam_t", (0, 0, 0)), sp.get("cam_r", (0, 0, 0)))
            p2, q2 = perturb_board(bp, bq, sp.get("board_t", (0, 0, 0)), sp.get("board_r", (0, 0, 0)))
            r = sc.score(cm, p2, q2, hand_dz=hdz + sp.get("hand_dz", 0.0))
            rows.append({**{k: v for k, v in sp.items() if k in ("family", "mag", "unit", "sign")},
                         "value": r["combined"], "ci95": r["ci95_combined"], "robot": r["robot"], "board": r["board"],
                         "mean_clip10": r["mean_clip10"], "within1px": r["within1px"],
                         "orient_robot_mean10": r["orient_robot_mean10"], "orient_board_mean10": r["orient_board_mean10"]})
        neg = []
        for qs in (1, 2, 3, 5, 10, 30, 100, 300, 1000):
            for sg in (1, -1):
                r = sc.score(cam, bp, bq, hand_dz=hdz, q_shift=sg * qs)
                neg.append({"control": "robot_time_shift_frames", "shift": sg * qs, "value": r["combined"],
                            "ci95": r["ci95_combined"], "robot": r["robot"], "mean_clip10": r["mean_clip10"],
                            "orient_robot_mean10": r["orient_robot_mean10"], "orient_board_mean10": r["orient_board_mean10"]})
        other = "hama2" if cam_name == "hama1" else "hama1"
        oc = cam_from_calib(c["cameras"][other])
        oc.name = cam_name
        r = sc.score(oc, bp, bq, hand_dz=hdz)
        neg.append({"control": f"{other}_calibration_on_{cam_name}_images", "value": r["combined"], "ci95": r["ci95_combined"],
                    "robot": r["robot"], "mean_clip10": r["mean_clip10"], "orient_robot_mean10": r["orient_robot_mean10"],
                    "orient_board_mean10": r["orient_board_mean10"]})
        thr = THRESH[f"B0.overlay_{cam_name}"]
        blank = float(np.median([n["orient_robot_mean10"] for n in neg if abs(n.get("shift", 0)) >= 300]))
        prop_thr = 0.5 * blank
        res["static"][cam_name] = {
            "baseline": {"value": base["combined"], "ci95": base["ci95_combined"], "robot": base["robot"],
                         "board": base["board"], "mean_clip10": base["mean_clip10"],
                         "orient_robot_mean10": base["orient_robot_mean10"], "orient_board_mean10": base["orient_board_mean10"]},
            "threshold": thr, "rows": rows, "negative_controls": neg,
            "detect": _detect_table(rows, thr, base["combined"], base["ci95_combined"]),
            "detect_mean_clip10": _detect_table([{**r, "value": r["mean_clip10"], "ci95": [np.nan]} for r in rows],
                                                np.inf, base["mean_clip10"], [base["mean_clip10"] - 0.05, base["mean_clip10"] + 0.05]),
            "proposed": {"metric": "orient_robot_mean10 (directional chamfer, robot outline, clip 10 px; mean per frame, median over frames)",
                         "baseline": base["orient_robot_mean10"], "blank_robot_shift_ge_300": blank,
                         "threshold_half_blank": prop_thr, "baseline_board_term": base["orient_board_mean10"],
                         "detect": _detect_table([{**r, "value": r["orient_robot_mean10"], "ci95": [np.nan]} for r in rows],
                                                 prop_thr, base["orient_robot_mean10"], [np.nan, np.nan]),
                         "detect_board_term": _detect_table([{**r, "value": r["orient_board_mean10"], "ci95": [np.nan]} for r in rows],
                                                            np.inf, base["orient_board_mean10"], [base["orient_board_mean10"] - 0.25, base["orient_board_mean10"] + 0.25])},
            "geometry_at_board_centre": px_per_mm(cam, board_c)}
    kin = Kin()
    dp0, _ = tcp_errors(kin, episode, frames)
    trows = []
    for m in (0.5, 1, 2, 3, 5, 10):
        for sg in (1, -1):
            dp, _ = tcp_errors(kin, episode, frames, hand_dz=sg * m / 1000)
            trows.append({"family": "tool_length", "mag": m, "unit": "mm", "sign": sg, "value": float(np.median(dp))})
    res["tcp"] = {"baseline": float(np.median(dp0)), "threshold": THRESH["B0.tcp_err_mm"], "rows": trows,
                  "detect": _detect_table(trows, THRESH["B0.tcp_err_mm"], float(np.median(dp0)), [np.nan, np.nan])}
    hb = mb.get("checks", {}).get("B0.overlay_hand", {})
    if hand_keypoints is None:
        hand_keypoints, res["hand_keypoints_file"] = _saved(out_dir, rec, "hand_keypoints")
    ffr = hb.get("finger_frames", [])
    if (hand_keypoints or ffr) and "hand" in c["cameras"]:
        cam = cam_from_calib(c["cameras"]["hand"])
        kps, _ = resolve_board_keypoints(list(hand_keypoints or []), tb["holes"])

        def fingers(cm):
            v = w.get_view()
            v.set_hand_offset(hdz)
            try:
                return finger_corner_residuals(cm, episode, ffr, view=v)
            finally:
                v.set_hand_offset(0.0)

        def pooled(cm, p2, q2, dz, fingers_too):
            rp = hand_residuals(kin, cm, episode, kps, p2, q2, hand_dz=hdz + dz) if kps else np.zeros(0)
            rf = np.array([r["px"] for r in fingers(cm)]) if fingers_too else base_f
            return np.concatenate([rp, rf])
        base_f = np.array([r["px"] for r in fingers(cam)])
        base = pooled(cam, bp, bq, 0.0, False)
        rows = []
        for sp in _sweep_specs("hand"):
            cm = perturb_cam(cam, sp.get("cam_t", (0, 0, 0)), sp.get("cam_r", (0, 0, 0)))
            p2, q2 = perturb_board(bp, bq, sp.get("board_t", (0, 0, 0)), sp.get("board_r", (0, 0, 0)))
            r = pooled(cm, p2, q2, sp.get("hand_dz", 0.0), "cam_t" in sp or "cam_r" in sp)
            rows.append({**{k: v for k, v in sp.items() if k in ("family", "mag", "unit", "sign")},
                         "value": float(np.median(r)), "ci95": bootstrap_median_ci(r)})
        # synthetic truth: the twin's own projections / rendered corners as perfect marks, so the response is
        # the check's resolution for a correct twin (the real baseline above may already fail)
        s = w.episode_state(episode)
        syn_kps = []
        for k in kps:
            f = int(k["frame"])
            kin.set(s.q[f], s.grip[f], hdz)
            Rc, pc = kin.body(cam.parent)
            P = w.quat2mat(bq) @ np.asarray(k["board"], float) + bp
            syn_kps.append({**k, "uv": project_cv(cam, (Rc.T @ (P - pc))[None])[0][0].tolist()})
        syn_f = {(r["frame"], r["pad"]): np.array(r["render"]) for r in fingers(cam)}

        def pooled_syn(cm, p2, q2, dz, fingers_too):
            rp = hand_residuals(kin, cm, episode, syn_kps, p2, q2, hand_dz=hdz + dz) if syn_kps else np.zeros(0)
            if fingers_too:
                rf = np.array([np.linalg.norm(np.array(r["render"]) - syn_f[(r["frame"], r["pad"])])
                               for r in fingers(cm) if (r["frame"], r["pad"]) in syn_f])
            else:
                rf = np.zeros(len(syn_f))
            return np.concatenate([rp, rf])
        rows_syn = []
        for sp in _sweep_specs("hand"):
            cm = perturb_cam(cam, sp.get("cam_t", (0, 0, 0)), sp.get("cam_r", (0, 0, 0)))
            p2, q2 = perturb_board(bp, bq, sp.get("board_t", (0, 0, 0)), sp.get("board_r", (0, 0, 0)))
            r = pooled_syn(cm, p2, q2, sp.get("hand_dz", 0.0), "cam_t" in sp or "cam_r" in sp)
            rows_syn.append({**{k: v for k, v in sp.items() if k in ("family", "mag", "unit", "sign")},
                             "value": float(np.median(r)), "part_median": float(np.median(r[:len(syn_kps)])) if syn_kps else None,
                             "finger_median": float(np.median(r[len(syn_kps):])) if len(r) > len(syn_kps) else None})
        geo = []
        for f in sorted({int(k["frame"]) for k in kps}):
            kin.set(s.q[f], s.grip[f], hdz)
            Rc, pc = kin.body(cam.parent)
            P = w.quat2mat(bq) @ np.asarray(next(k["board"] for k in kps if int(k["frame"]) == f), float) + bp
            geo.append({"frame": f, **px_per_mm(cam, Rc.T @ (P - pc))})
        res["hand"] = {"baseline": float(np.median(base)), "baseline_ci95": bootstrap_median_ci(base),
                       "baseline_part": float(np.median(base[:len(kps)])) if kps else None,
                       "baseline_finger": float(np.median(base_f)) if base_f.size else None,
                       "threshold": THRESH["B0.overlay_hand"], "rows": rows,
                       "detect": _detect_table(rows, THRESH["B0.overlay_hand"], float(np.median(base)), bootstrap_median_ci(base)),
                       "synthetic_rows": rows_syn,
                       "synthetic_detect": _detect_table(rows_syn, THRESH["B0.overlay_hand"], 0.0, [np.nan, np.nan]),
                       "synthetic_note": "marks replaced by the twin's own projections (a perfect twin), then perturbed: "
                                         "the smallest error a correct twin would fail at",
                       "geometry_note": "px per mm at each part keypoint (board plane) under the wrist camera",
                       "geometry_at_keypoints": geo}
    res["seconds"] = round(time.time() - t0, 1)
    p = out_dir / f"b0_known_error_{rec}.json"
    p.write_text(json.dumps(res, indent=1, default=float))
    fig = _plot_known_error(res, out_dir / f"b0_known_error_curves_{rec}.png")
    fig2 = _plot_proposed(res, out_dir / f"b0_known_error_proposed_{rec}.png")
    summ = {cam: {"baseline": _r(v["baseline"]["value"]), "detect": [{k: d[k] for k in ("family", "unit", "fails_at", "reliably_fails_at", "change_detectable_at")} for d in v["detect"]],
                  "negative_controls": [{k: (_r(x) if isinstance(x, float) else x) for k, x in n.items() if k != "ci95"} for n in v["negative_controls"]],
                  "geometry": v["geometry_at_board_centre"],
                  "proposed": {k: (v["proposed"][k] if k != "detect" else [{kk: d[kk] for kk in ("family", "unit", "fails_at")} for d in v["proposed"]["detect"]])
                               for k in ("baseline", "blank_robot_shift_ge_300", "threshold_half_blank", "detect")}}
            for cam, v in res["static"].items()}
    if res["hand"]:
        summ["hand"] = {"baseline": _r(res["hand"]["baseline"]),
                        "synthetic_detect": [{k: d[k] for k in ("family", "unit", "fails_at")} for d in res["hand"]["synthetic_detect"]]}
    summ["tcp"] = {"baseline": _r(res["tcp"]["baseline"]), "detect": res["tcp"]["detect"]}
    return {"path": _rel(p), "figure": _rel(fig), "figure_proposed": _rel(fig2), "summary": summ, "seconds": res["seconds"]}


def _plot_known_error(res: dict, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = []
    for cam, v in res["static"].items():
        panels.append((f"{cam}: camera shift (mm)", v, ["cam_tx", "cam_ty", "cam_tz"], "mm"))
        panels.append((f"{cam}: camera turn (deg)", v, ["cam_rx", "cam_ry", "cam_rz"], "deg"))
        panels.append((f"{cam}: board / tool (mm)", v, ["board_tx", "board_ty", "board_tz", "tool_length"], "mm"))
    if res.get("hand"):
        v = {"rows": res["hand"].get("synthetic_rows", res["hand"]["rows"]), "threshold": res["hand"]["threshold"],
             "baseline": {"value": 0.0}, "measured": res["hand"]["baseline"]}
        panels.append(("hand (perfect-twin truth): camera shift (mm)", v, ["cam_tx", "cam_ty", "cam_tz"], "mm"))
        panels.append(("hand (perfect-twin truth): camera turn (deg)", v, ["cam_rx", "cam_ry", "cam_rz"], "deg"))
        panels.append(("hand (perfect-twin truth): board / tool (mm)", v, ["board_tx", "board_ty", "board_tz", "tool_length"], "mm"))
    panels.append(("tool tip: tool length (mm)", {"rows": res["tcp"]["rows"], "threshold": res["tcp"]["threshold"],
                                                   "baseline": {"value": res["tcp"]["baseline"]}}, ["tool_length"], "mm"))
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False)
    colors = ["#2a6fdb", "#e0663a", "#2e9e5b", "#8c4fc9"]
    for ax, (title, v, fams, unit) in zip(axs.ravel(), panels):
        for fam, col in zip(fams, colors):
            rs = [r for r in v["rows"] if r["family"] == fam]
            if not rs:
                continue
            mags = sorted({r["mag"] for r in rs})
            worst = [min(r["value"] for r in rs if r["mag"] == m) for m in mags]   # the more lenient sign
            best = [max(r["value"] for r in rs if r["mag"] == m) for m in mags]
            ax.fill_between([0] + mags, [v["baseline"]["value"] if isinstance(v["baseline"], dict) else v["baseline"]] + worst,
                            [v["baseline"]["value"] if isinstance(v["baseline"], dict) else v["baseline"]] + best, color=col, alpha=0.18, lw=0)
            ax.plot([0] + mags, [v["baseline"]["value"] if isinstance(v["baseline"], dict) else v["baseline"]] + worst, "-o", color=col, ms=3, label=fam.replace("_", " "))
        ax.axhline(v["threshold"], color="#c0392b", ls="--", lw=1.2, label=f"threshold {v['threshold']}")
        if "measured" in v:
            ax.axhline(v["measured"], color="#7f8c8d", ls=":", lw=1.2, label=f"measured on real frames {v['measured']:.1f}")
        ax.set_xscale("symlog", linthresh=0.1 if unit == "deg" else 1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"known error ({unit})", fontsize=9)
        ax.set_ylabel("check value", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axs.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("B0 known-error tests (episode 0, held-out frames): line = the sign that hurts the check less, band = both signs", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _plot_proposed(res: dict, path: Path) -> Path:
    """Response of the current check (median edge px) vs the directional-chamfer robot term, per camera."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cams = list(res["static"])
    fig, axs = plt.subplots(len(cams), 3, figsize=(15.6, 3.7 * len(cams)), squeeze=False)
    groups = [("camera shift (mm)", ["cam_tx", "cam_ty", "cam_tz"], "mm"), ("camera turn (deg)", ["cam_rx", "cam_ry", "cam_rz"], "deg"),
              ("board shift (mm), board term", ["board_tx", "board_ty", "board_tz"], "mm")]
    colors = ["#2a6fdb", "#e0663a", "#2e9e5b"]
    for i, cam in enumerate(cams):
        v = res["static"][cam]
        pr = v["proposed"]
        for j, (title, fams, unit) in enumerate(groups):
            ax = axs[i, j]
            key = "orient_board_mean10" if j == 2 else "orient_robot_mean10"
            b0 = v["baseline"][key]
            for fam, col in zip(fams, colors):
                rs = [r for r in v["rows"] if r["family"] == fam]
                mags = sorted({r["mag"] for r in rs})
                lo = [min(r[key] for r in rs if r["mag"] == m) for m in mags]
                ax.plot([0] + mags, [b0] + lo, "-o", color=col, ms=3, label=f"{fam} (directional)")
                cur = [min(r["value"] for r in rs if r["mag"] == m) for m in mags]
                ax.plot([0] + mags, [v["baseline"]["value"]] + cur, ":", color=col, lw=1.2, label=f"{fam} (current median)")
            if j < 2:
                ax.axhline(pr["threshold_half_blank"], color="#c0392b", ls="--", lw=1.2, label=f"0.5 x wrong-pose blank = {pr['threshold_half_blank']:.2f}")
                ax.axhline(pr["blank_robot_shift_ge_300"], color="#7f8c8d", ls="-.", lw=1, label="wrong-pose blank")
            ax.axhline(v["threshold"], color="#8e44ad", ls="--", lw=1, label=f"current threshold {v['threshold']}")
            ax.set_xscale("symlog", linthresh=0.1 if unit == "deg" else 1)
            ax.set_title(f"{cam}: {title}", fontsize=10)
            ax.set_xlabel(f"known error ({unit})", fontsize=9)
            ax.set_ylabel("px", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6, loc="upper left")
    fig.suptitle("Current B0 overlay metric (dotted) vs directional-chamfer robot term (solid); lines = the sign that hurts less", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)
    return path


def refine_edge(img: np.ndarray, a, b, inside, n_prof: int = 48, search: float = 7.0) -> dict:
    """Sub-pixel image line of a plate edge near the rough segment a-b. Intensity profiles are sampled normal to
    the segment (from the interior side, given by `inside`, outward); on each the edge is the *first* strong
    intensity drop going outward (so a plate top edge wins over the side face and rails below it), refined by a
    parabola. A robust line fit (two passes, 1.5 px cut) gives the line (a, b, c) with a u + b v + c = 0."""
    import cv2
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), (3, 3), 0.8)
    a, b, inside = (np.asarray(x, float) for x in (a, b, inside))
    d = (b - a) / np.linalg.norm(b - a)
    nrm = np.array([-d[1], d[0]])
    if (inside - a) @ nrm > 0:
        nrm = -nrm  # nrm points outward
    ss = np.arange(-search, search + 0.01, 0.25)
    pts = []
    for t in np.linspace(0.08, 0.92, n_prof):
        p0 = a + t * (b - a)
        P = p0[None] + ss[:, None] * nrm[None]
        prof = cv2.remap(g, P[:, 0].astype(np.float32)[None], P[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR).ravel()
        dv = -np.gradient(prof)  # drop going outward > 0
        if dv.max() <= 1.0:
            continue
        thr = 0.5 * dv.max()
        cand = [i for i in range(1, len(dv) - 1) if dv[i] >= thr and dv[i] >= dv[i - 1] and dv[i] >= dv[i + 1]]
        if not cand:
            continue
        i = cand[0]
        den = dv[i - 1] - 2 * dv[i] + dv[i + 1]
        off = 0.5 * (dv[i - 1] - dv[i + 1]) / den if abs(den) > 1e-6 else 0.0
        pts.append(p0 + (ss[i] + off * 0.25) * nrm)
    pts = np.array(pts)
    if len(pts) < 6:
        raise ValueError(f"edge {a}-{b}: only {len(pts)} profile hits")
    keep = np.ones(len(pts), bool)
    for _ in range(3):
        c0 = pts[keep].mean(0)
        _, _, Vt = np.linalg.svd(pts[keep] - c0)
        n = Vt[1]
        r = (pts - c0) @ n
        keep = np.abs(r) < max(1.5, 2.5 * np.median(np.abs(r[keep])))
    line = np.array([n[0], n[1], -n @ c0])
    return {"line": line.tolist(), "n_points": int(keep.sum()), "n_profiles": len(pts),
            "rms_px": float(np.sqrt(np.mean(r[keep] ** 2))), "points": pts[keep].tolist()}


def _line3d(Ps, lines):
    """3D line from image lines in two views: intersection of the back-projected planes pi_i = P_i^T l_i."""
    A = np.stack([P.T @ np.asarray(l, float) for P, l in zip(Ps, lines)])
    n1, n2 = A[0, :3] / np.linalg.norm(A[0, :3]), A[1, :3] / np.linalg.norm(A[1, :3])
    _, _, Vt = np.linalg.svd(A)
    X1, X2 = Vt[2], Vt[3]
    # a finite point and a direction from the two null vectors
    pts = [X[:3] / X[3] for X in (X1, X2) if abs(X[3]) > 1e-9]
    direction = np.cross(A[0, :3], A[1, :3])
    direction /= np.linalg.norm(direction)
    p = pts[0]
    return p, direction, float(np.degrees(np.arccos(min(1.0, abs(n1 @ n2)))))


def _closest(p1, d1, p2, d2):
    w0 = p1 - p2
    a, b, c, d, e = d1 @ d1, d1 @ d2, d2 @ d2, d1 @ w0, d2 @ w0
    den = a * c - b * b
    s_, t_ = (b * e - c * d) / den, (a * e - b * d) / den
    q1, q2 = p1 + s_ * d1, p2 + t_ * d2
    return (q1 + q2) / 2, float(np.linalg.norm(q1 - q2))


@tool("triangulate_marks",
      "Watcher: triangulate points marked in two calibrated static cameras (default hama1 + hama2 from "
      "twin/calib/<recording>.json). Each mark: {label, uv1:[u,v], uv2:[u,v], board?:[x,y,z] model point in the "
      "NIST_Board1 frame}. Returns 3D points in the robot base frame, per-view reprojection error (px, the two "
      "cameras' mutual consistency), distance to the model points under the calib board pose (mm), and the rigid "
      "board correction the marks imply. Writes the JSON to out (default runs/<brain>/<run_id>/b0-watch/board_camera_<recording>.json).",
      {"type": "object", "properties": {
          "episode": {"type": "integer"}, "marks": {"type": "array", "items": {"type": "object"}},
          "cameras": {"type": "array", "items": {"type": "string"}, "default": ["hama1", "hama2"]},
          "edges": {"type": "array", "items": {"type": "object"},
                    "description": "straight plate edges: [{label, frame, seg1:[[u,v],[u,v]], seg2:[[u,v],[u,v]], inside1:[u,v], "
                                   "inside2:[u,v]}] rough segments on the top-surface edge in each view; refined to sub-pixel "
                                   "lines and triangulated as 3D lines"},
          "corners": {"type": "array", "items": {"type": "object"},
                      "description": "[{label, edges:[labelA, labelB], board?:[x,y,z]}] corners as the closest point of two 3D edges"},
          "out": {"type": "string"}, "brain": {"type": "string", "default": "claude"},
          "run_id": {"type": "string", "default": "run-standin-1"}},
       "required": ["episode"]})
def triangulate_marks(episode: int, marks: list | None = None, cameras=("hama1", "hama2"), edges=None, corners=None,
                      out: str | None = None, brain: str = "claude", run_id: str = "run-standin-1") -> dict:
    import cv2
    from scipy.optimize import least_squares
    w = _w()
    rec, c = _calib(episode)
    cams = [cam_from_calib(c["cameras"][n]) for n in cameras]
    Ps = []
    for cm in cams:
        R = cm.R_cv.T
        Ps.append(cm.K @ np.hstack([R, (-R @ np.asarray(cm.pos))[:, None]]))
    bp, bq = np.array(c["board"]["pos"]), np.array(c["board"]["quat"])
    Rb = w.quat2mat(bq)
    marks = marks or []
    pts = []
    for mk in marks:
        u1, u2 = np.asarray(mk["uv1"], float), np.asarray(mk["uv2"], float)
        X = cv2.triangulatePoints(Ps[0], Ps[1], u1[:, None], u2[:, None]).ravel()
        X = X[:3] / X[3]

        def resid(x):
            return np.concatenate([project_cv(cams[0], x[None])[0][0] - u1, project_cv(cams[1], x[None])[0][0] - u2])
        X = least_squares(resid, X).x
        rr = resid(X)
        # ray angle: how well conditioned the depth is
        a = (X - np.asarray(cams[0].pos))
        b = (X - np.asarray(cams[1].pos))
        ang = np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
        row = {"label": mk.get("label", ""), "world": [round(float(v), 5) for v in X],
               "reproj_px": [_r(np.hypot(*rr[:2]), 2), _r(np.hypot(*rr[2:]), 2)], "ray_angle_deg": _r(ang, 1),
               "board_frame": [round(float(v), 5) for v in Rb.T @ (X - bp)]}
        if "board" in mk:
            M = Rb @ np.asarray(mk["board"], float) + bp
            row["model_world"] = [round(float(v), 5) for v in M]
            row["model_board"] = list(mk["board"])
            row["err_mm"] = [_r(v * 1000, 2) for v in (X - M)]
            row["err_board_axes_mm"] = [_r(v * 1000, 2) for v in Rb.T @ (X - M)]
            row["err_norm_mm"] = _r(np.linalg.norm(X - M) * 1000, 2)
            row["err_xy_mm"] = _r(np.linalg.norm((Rb.T @ (X - M))[:2]) * 1000, 2)
            row["model_source"] = mk.get("model_source", "")
            if mk.get("xy_only"):  # model height unknown (e.g. a peg top): compare in the board plane only
                row["xy_only"] = True
                for k in ("err_mm", "err_norm_mm"):
                    row.pop(k, None)
        pts.append(row)
    res = {"recording": rec, "episode": episode, "brain": brain, "run_id": run_id, "cameras": list(cameras),
           "board_pose_under_test": {"pos": bp.tolist(), "quat_wxyz": bq.tolist(), "source": c["board"].get("source", "")},
           "points": pts}
    with_m = [p for p in pts if "err_norm_mm" in p]
    xy_m = [p for p in pts if p.get("xy_only")]
    if xy_m:
        res["xy_only_summary"] = {"n": len(xy_m), "err_xy_mm": {p["label"]: p["err_xy_mm"] for p in xy_m},
                                  "err_xy_mm_median": _r(np.median([p["err_xy_mm"] for p in xy_m]), 2)}
    if with_m:
        e = np.array([p["err_norm_mm"] for p in with_m])
        exy = np.array([p["err_xy_mm"] for p in with_m])
        ez = np.array([p["err_board_axes_mm"][2] for p in with_m])
        res["summary"] = {"n": len(with_m), "err_norm_mm_median": _r(np.median(e), 2), "err_norm_mm_max": _r(e.max(), 2),
                          "err_xy_mm_median": _r(np.median(exy), 2), "err_z_board_mm_median": _r(np.median(ez), 2),
                          "reproj_px_median": _r(np.median([max(p["reproj_px"]) for p in pts]), 2)}
    if pts:
        res["two_view_consistency"] = {"reproj_px_median_all_marks": _r(np.median([max(p["reproj_px"]) for p in pts]), 2),
                                       "note": "after 2-view triangulation, the residual is how far the two calibrated cameras "
                                               "disagree about the same physical point (0 for a consistent pair, up to marking error)"}
        if len(with_m) >= 3:
            A = np.array([p["model_world"] for p in with_m])
            B = np.array([p["world"] for p in with_m])
            ca, cb = A.mean(0), B.mean(0)
            U, S_, Vt = np.linalg.svd((A - ca).T @ (B - cb))
            D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
            Rf = Vt.T @ D @ U.T
            tf = cb - Rf @ ca
            fit_res = np.linalg.norm(B - (A @ Rf.T + tf), axis=1) * 1000
            res["rigid_fit_model_to_cameras"] = {
                "rotation_deg": _r(np.degrees(np.arccos(np.clip((np.trace(Rf) - 1) / 2, -1, 1))), 3),
                "translation_at_centroid_mm": [_r(v * 1000, 2) for v in (Rf @ ca + tf - ca)],
                "translation_at_centroid_board_axes_mm": [_r(v * 1000, 2) for v in Rb.T @ (Rf @ ca + tf - ca)],
                "residual_mm_median": _r(np.median(fit_res), 2), "residual_mm_max": _r(fit_res.max(), 2),
                "note": "rigid transform taking the model points (calib board pose) onto the triangulated points; "
                        "its translation is evaluated at the points' centroid"}
            # board pose implied by the cameras
            Rb2 = Rf @ Rb
            res["board_pose_from_cameras"] = {"pos": [round(float(v), 5) for v in Rf @ bp + tf],
                                              "quat_wxyz": [round(float(v), 6) for v in w.mat2quat(Rb2)]}
    if edges:
        E = {}
        for e in edges:
            f = int(e.get("frame", 0))
            ims = [w.read_frame(episode, n, f) for n in cameras]
            r1 = refine_edge(ims[0], *e["seg1"], e["inside1"])
            r2 = refine_edge(ims[1], *e["seg2"], e["inside2"])
            p3, d3, ang = _line3d(Ps, [r1["line"], r2["line"]])
            E[e["label"]] = (p3, d3)
            res.setdefault("edges", []).append({
                "label": e["label"], "frame": f, "plane_angle_deg": _r(ang, 1),
                "view1": {k: (_r(v, 3) if k == "rms_px" else v) for k, v in r1.items() if k != "points"},
                "view2": {k: (_r(v, 3) if k == "rms_px" else v) for k, v in r2.items() if k != "points"},
                "point_world": [round(float(v), 5) for v in p3], "dir_world": [round(float(v), 5) for v in d3],
                "dir_board": [round(float(v), 5) for v in Rb.T @ d3]})
            res.setdefault("_edge_points", {})[e["label"]] = [r1["points"], r2["points"]]
        L = {e["label"]: (e["view1"]["line"], e["view2"]["line"]) for e in res["edges"]}
        for cn in corners or []:
            # corner = intersection of the two refined image lines in each view, then a 2-view point
            # triangulation (well conditioned even when an edge runs along the baseline)
            la, lb = L[cn["edges"][0]], L[cn["edges"][1]]
            uvs = []
            for vi in range(2):
                h = np.cross(np.asarray(la[vi], float), np.asarray(lb[vi], float))
                uvs.append(h[:2] / h[2])
            X = cv2.triangulatePoints(Ps[0], Ps[1], uvs[0][:, None], uvs[1][:, None]).ravel()
            X = X[:3] / X[3]

            def resid(x, uvs=uvs):
                return np.concatenate([project_cv(cams[0], x[None])[0][0] - uvs[0], project_cv(cams[1], x[None])[0][0] - uvs[1]])
            X = least_squares(resid, X).x
            rr = resid(X)
            (pa, da), (pb_, db) = E[cn["edges"][0]], E[cn["edges"][1]]
            row = {"label": cn["label"], "kind": "edge_corner", "uv": [[_r(v, 2) for v in u] for u in uvs],
                   "world": [round(float(v), 5) for v in X], "board_frame": [round(float(v), 5) for v in Rb.T @ (X - bp)],
                   "reproj_px": [_r(np.hypot(*rr[:2]), 2), _r(np.hypot(*rr[2:]), 2)]}
            if "board" in cn:
                M = Rb @ np.asarray(cn["board"], float) + bp
                row.update({"model_board": list(cn["board"]), "model_world": [round(float(v), 5) for v in M],
                            "err_board_axes_mm": [_r(v * 1000, 2) for v in Rb.T @ (X - M)],
                            "err_norm_mm": _r(np.linalg.norm(X - M) * 1000, 2),
                            "err_xy_mm": _r(np.linalg.norm((Rb.T @ (X - M))[:2]) * 1000, 2),
                            "model_source": cn.get("model_source", "")})
            res.setdefault("edge_corners", []).append(row)
        if len(res.get("edge_corners", [])) == 4:
            C = {c["label"]: np.array(c["world"]) for c in res["edge_corners"]}
            res["plate_from_cameras"] = _plate_summary(C, bp, Rb)
    edge_pts = res.pop("_edge_points", None)
    if edge_pts:
        _plot_edges(episode, cameras, edges, edge_pts, res, cams, _out_dir(brain, run_id) / f"board_camera_edges_{rec}.png")
        res["edges_figure"] = _rel(_out_dir(brain, run_id) / f"board_camera_edges_{rec}.png")
    p = Path(out) if out else _out_dir(brain, run_id) / f"board_camera_{rec}.json"
    if not p.is_absolute():
        p = REPO / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, default=float))
    return {"path": _rel(p), "summary": res.get("summary"), "rigid_fit": res.get("rigid_fit_model_to_cameras"),
            "plate_from_cameras": res.get("plate_from_cameras"), "edge_corners": res.get("edge_corners"),
            "edges": [{k: e[k] for k in ("label", "plane_angle_deg")} | {"rms_px": [e["view1"]["rms_px"], e["view2"]["rms_px"]]} for e in res.get("edges", [])],
            "points": [{k: p_[k] for k in ("label", "reproj_px", "err_norm_mm", "err_board_axes_mm") if k in p_} for p_ in pts]}


def _plate_summary(C: dict, bp, Rb) -> dict:
    """Plate rectangle from four triangulated corners (labels c00, c10, c11, c01 = board-frame x0y0, x1y0, x1y1,
    x0y1): side lengths, centre, top-plane normal, yaw/tilt relative to the mocap board frame."""
    w = _w()
    c00, c10, c11, c01 = (C[k] for k in ("c00", "c10", "c11", "c01"))
    ex = ((c10 - c00) + (c11 - c01)) / 2
    ey = ((c01 - c00) + (c11 - c10)) / 2
    nz = np.cross(ex, ey)
    nz /= np.linalg.norm(nz)
    xh = ex / np.linalg.norm(ex)
    yh = np.cross(nz, xh)
    Rp = np.stack([xh, yh, nz], 1)
    Rrel = Rb.T @ Rp
    centre = (c00 + c10 + c11 + c01) / 4
    planar = max(abs((c - centre) @ nz) for c in (c00, c10, c11, c01))
    return {"side_x_mm": [_r(np.linalg.norm(c10 - c00) * 1000, 1), _r(np.linalg.norm(c11 - c01) * 1000, 1)],
            "side_y_mm": [_r(np.linalg.norm(c01 - c00) * 1000, 1), _r(np.linalg.norm(c11 - c10) * 1000, 1)],
            "corner_angle_deg": _r(np.degrees(np.arccos(abs(xh @ (ey / np.linalg.norm(ey))))), 2),
            "centre_world": [round(float(v), 5) for v in centre], "centre_board": [round(float(v), 5) for v in Rb.T @ (centre - bp)],
            "normal_world": [round(float(v), 5) for v in nz],
            "yaw_vs_mocap_deg": _r(np.degrees(np.arctan2(Rrel[1, 0], Rrel[0, 0])), 3),
            "tilt_vs_mocap_deg": _r(np.degrees(np.arccos(np.clip(Rrel[2, 2], -1, 1))), 3),
            "corner_out_of_plane_mm": _r(planar * 1000, 2),
            "pose_quat_wxyz": [round(float(v), 6) for v in w.mat2quat(Rp)]}


def _plot_edges(episode, cameras, edges, edge_pts, res, cams, path: Path):
    """Both views with the refined edge points (cyan), fitted lines (yellow), triangulated corners reprojected
    (red) and the model plate corners under the calib board pose (blue)."""
    import cv2
    w = _w()
    f = int(edges[0].get("frame", 0))
    tiles = []
    for vi, (name, cm) in enumerate(zip(cameras, cams)):
        img = w.read_frame(episode, name, f).copy()
        for lab, pp in edge_pts.items():
            for u, v in pp[vi]:
                cv2.circle(img, (int(round(u)), int(round(v))), 1, (0, 255, 255), -1)
        for e in res.get("edges", []):
            a, b, c = e[f"view{vi + 1}"]["line"]
            if abs(b) > abs(a):
                p1, p2 = (0, int(-c / b)), (639, int(-(c + a * 639) / b))
            else:
                p1, p2 = (int(-c / a), 0), (int(-(c + b * 479) / a), 479)
            cv2.line(img, p1, p2, (255, 220, 0), 1, cv2.LINE_AA)
        for cr in res.get("edge_corners", []):
            uv = project_cv(cm, np.array(cr["world"])[None])[0][0]
            cv2.circle(img, (int(round(uv[0])), int(round(uv[1]))), 4, (255, 0, 0), 1, cv2.LINE_AA)
            if "model_world" in cr:
                uvm = project_cv(cm, np.array(cr["model_world"])[None])[0][0]
                cv2.drawMarker(img, (int(round(uvm[0])), int(round(uvm[1]))), (0, 80, 255), cv2.MARKER_CROSS, 8, 1)
        cv2.putText(img, f"{name} f{f}: edges cyan/yellow, triangulated corners red, model corners blue", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))
