"""Camera calibration from robot keypoints, board features and forward kinematics (stage B0, world-frame skill).

Why this works without a calibration target: the logged joint angles put every robot link at a known 3D pose in
every frame, and the board pose is known to about 1 mm from robot contacts (board-from-contacts skill). A camera
looking at the cell is therefore a camera looking at a moving calibration object. Collect 2D-3D correspondences
over many frames spread over the workspace, then bundle-adjust the camera pose, shared intrinsics, optional radial
distortion, the video/joint-log latency and the tool length together (robot-supported calibration, Werner et al.
2018; robot keypoints with FK 3D as in DREAM, Lee et al. 2020). Two static cameras fitted to the same 3D points
are consistent by construction; `two_view` measures it on points neither fit used.

Correspondences (`Obs`) come from two sources:
  * marks: image positions a brain marks on zoomed crops (`marking_crops`). Keys name the 3D point:
      robot keys (`ROBOT_KEYS`: finger pads, ...) are fixed on a robot body; their 3D comes from FK at the frame;
      board keys (`board_keys()`: NIST plate corners, hole centres) are fixed in the board frame.
  * auto: render-and-compare of distinctive robot features (`AUTO_FEATURES`: the black joint caps, the finger
    pads). The twin is rendered with the current camera; masked ZNCC of the rendered patch against the real frame
    gives each feature's 2D offset; its 3D point is the centroid of its visible rendered surface, fixed to its link.
    Iterate render -> fit -> render; the offsets shrink to the residual.

Everything is in the robot base frame (= FR3 link0 = MuJoCo world). Camera convention as in twin/world.py
(CameraModel; OpenCV axes R_cv). Distortion: OpenCV radial k1, k2 on normalised coordinates.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from twin import world as w  # noqa: E402

# ----------------------------------------------------------------------------------------------
# 3D keypoint definitions
# ----------------------------------------------------------------------------------------------
# Finger pad: centre of the black pad mesh (Menagerie panda hand, geom 'gripper/black' on each finger), in the
# finger body frame. The visible black blob a person marks is centred within ~1-2 mm of it in side views.
PAD_P = (0.0, 0.0047, 0.0448)
ROBOT_KEYS = {
    "left_pad": ("gripper/left_finger", PAD_P),
    "right_pad": ("gripper/right_finger", PAD_P),
}
# Robot features matched automatically: name -> (body, [geom ids in base.xml]). Black joint caps and pads.
AUTO_FEATURES = {
    "pad_l": ("gripper/left_finger", [46]), "pad_r": ("gripper/right_finger", [54]),
    "cap3": ("fr3_link3", [15]), "cap4": ("fr3_link4", [18]), "blk5": ("fr3_link5", [22]),
    "cap6": ("fr3_link6", [30]), "blk7": ("fr3_link7", [33, 36]),
}
HAND_BODIES = ("gripper/hand", "gripper/left_finger", "gripper/right_finger")

LAYOUT_NIST = w.REPO_DIR / "runs" / "claude" / "run-standin-1" / "board" / "layout_nist.json"
PLATE_MM = 384.0
PLATE_THICK = 0.009  # NIST laser plate, GMC_Laser_Plate_Virtual.STL 384 x 384 x 9 mm


def nist_transform(layout_path: Path = LAYOUT_NIST) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) with NIST_mm = R @ board_mm + t (a reflection: det -1), from board_contact_layout's nist_fit."""
    nf = json.loads(Path(layout_path).read_text())["nist_fit"]
    a = np.radians(nf["angle_deg"])
    R = np.array([[np.cos(a), np.sin(a)], [np.sin(a), -np.cos(a)]]) if nf["det"] < 0 else \
        np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return R, np.array(nf["t_mm"], float)


def nist_to_board(p_mm, layout_path: Path = LAYOUT_NIST) -> np.ndarray:
    R, t = nist_transform(layout_path)
    return ((np.atleast_2d(np.asarray(p_mm, float)) - t) @ np.linalg.inv(R).T) / 1000.0


def board_keys(layout_path: Path = LAYOUT_NIST) -> dict[str, np.ndarray]:
    """3D points in the board (NIST_Board1) frame, plate top at z = 0: the four plate top corners (named by NIST
    plate coordinates: c00 = NIST (0,0) ... ) and every hole centre of the layout (hole_<part>)."""
    out = {}
    for k, (x, y) in {"c00": (0, 0), "c10": (PLATE_MM, 0), "c11": (PLATE_MM, PLATE_MM), "c01": (0, PLATE_MM)}.items():
        b = nist_to_board([x, y], layout_path)[0]
        out[k] = np.array([b[0], b[1], 0.0])
    lay = json.loads(Path(layout_path).read_text())
    for n, p in lay["parts"].items():
        out["hole_" + n.replace(" ", "_")] = np.array([p["xy"][0], p["xy"][1], 0.0])
    return out


def plate_outline_board(layout_path: Path = LAYOUT_NIST) -> np.ndarray:
    """(4, 2) plate top corners in the board frame, in order c00, c10, c11, c01."""
    k = board_keys(layout_path)
    return np.array([k[n][:2] for n in ("c00", "c10", "c11", "c01")])


# ----------------------------------------------------------------------------------------------
# Camera parameters with distortion
# ----------------------------------------------------------------------------------------------
@dataclass
class Cam:
    """Pinhole + radial distortion, world->camera (OpenCV). Converts to/from twin.world.CameraModel."""
    name: str
    R: np.ndarray            # world -> camera rotation (rows = camera axes in world)
    t: np.ndarray            # world -> camera translation
    f: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    parent: str = "world"
    fy_ratio: float = 1.0

    @classmethod
    def from_model(cls, m: "w.CameraModel", k1=0.0, k2=0.0) -> "Cam":
        R = m.R_cv.T
        return cls(m.name, R, -R @ np.asarray(m.pos, float), float(m.fx), float(m.cx), float(m.cy), k1, k2,
                   m.parent, float(m.fy / m.fx))

    def to_model(self) -> "w.CameraModel":
        pos = -self.R.T @ self.t
        return w.CameraModel.from_cv(self.name, self.R.T, pos, self.f, self.f * self.fy_ratio, self.cx, self.cy,
                                     parent=self.parent)

    @property
    def pos(self) -> np.ndarray:
        return -self.R.T @ self.t

    def project(self, X_parent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return project_dist(X_parent, self.R, self.t, self.f, self.f * self.fy_ratio, self.cx, self.cy, self.k1, self.k2)


def project_dist(X, R, t, fx, fy, cx, cy, k1=0.0, k2=0.0):
    Xc = np.atleast_2d(X) @ R.T + t
    z = Xc[:, 2]
    x, y = Xc[:, 0] / z, Xc[:, 1] / z
    r2 = x * x + y * y
    d = 1 + k1 * r2 + k2 * r2 * r2
    return np.stack([fx * x * d + cx, fy * y * d + cy], 1), z


# ----------------------------------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------------------------------
@dataclass
class Obs:
    cam: str
    frame: int
    uv: np.ndarray
    body: str                 # robot body name, "board" or "world"
    X: np.ndarray             # 3D point in the body / board / world frame
    name: str = ""
    src: str = "mark"         # mark | auto
    w: float = 1.0
    score: float = 1.0

    def as_dict(self):
        return {"cam": self.cam, "frame": int(self.frame), "uv": [round(float(v), 2) for v in self.uv],
                "body": self.body, "X": [round(float(v), 5) for v in self.X], "name": self.name, "src": self.src,
                "w": self.w, "score": round(float(self.score), 3)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["cam"], int(d["frame"]), np.asarray(d["uv"], float), d["body"], np.asarray(d["X"], float),
                   d.get("name", ""), d.get("src", "mark"), float(d.get("w", 1.0)), float(d.get("score", 1.0)))


def marks_to_obs(marks: list[dict], layout_path: Path = LAYOUT_NIST) -> list[Obs]:
    """Marks {cam, frame, uv, key} -> Obs. key in ROBOT_KEYS or board_keys(); or give body + p explicitly."""
    bk = board_keys(layout_path)
    out = []
    for m in marks:
        k = m.get("key")
        if k in ROBOT_KEYS:
            body, p = ROBOT_KEYS[k]
        elif k in bk:
            body, p = "board", bk[k]
        elif "body" in m:
            body, p = m["body"], m["p"]
        else:
            raise KeyError(f"unknown key {k}")
        out.append(Obs(m["cam"], int(m["frame"]), np.asarray(m["uv"], float), body, np.asarray(p, float),
                       name=k or m.get("label", ""), src="mark", w=float(m.get("w", 1.0))))
    return out


# ----------------------------------------------------------------------------------------------
# FK with latency and tool offset
# ----------------------------------------------------------------------------------------------
class Kinematics:
    """World poses of robot bodies at integer frames (cached), for points fixed on a body."""

    def __init__(self, episode: int):
        self.episode = episode
        self.s = w.episode_state(episode)
        m = mujoco.MjModel.from_xml_path(str(w.BASE_XML))
        self.m, self.d = m, mujoco.MjData(m)
        self.cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}

    def pose(self, frame: int, body: str) -> tuple[np.ndarray, np.ndarray]:
        frame = int(np.clip(frame, 0, self.s.n - 1))
        k = (frame, body)
        if k not in self.cache:
            self.d.qpos[:7], self.d.qpos[7:9] = self.s.q[frame], self.s.grip[frame]
            mujoco.mj_kinematics(self.m, self.d)
            for b in [body, "gripper/hand"]:
                i = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
                self.cache[(frame, b)] = (self.d.xmat[i].reshape(3, 3).copy(), self.d.xpos[i].copy())
        return self.cache[k]


def obs_world_tracks(obs: list[Obs], kin: Kinematics, K: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """For each obs: world points at frames f-K..f+K (N, 2K+1, 3) and the hand z axis at f (N, 3)."""
    N = len(obs)
    P = np.zeros((N, 2 * K + 1, 3))
    Z = np.zeros((N, 3))
    for i, o in enumerate(obs):
        if o.body in ("board", "world"):
            P[i] = o.X[None]
            continue
        for j, df in enumerate(range(-K, K + 1)):
            R, p = kin.pose(o.frame + df, o.body)
            P[i, j] = R @ o.X + p
        Z[i] = kin.pose(o.frame, "gripper/hand")[0][:, 2]
    return P, Z


# ----------------------------------------------------------------------------------------------
# Bundle adjustment
# ----------------------------------------------------------------------------------------------
@dataclass
class FitConfig:
    cams: tuple = ("hama1", "hama2")
    shared_intrinsics: bool = True     # identical webcam model: one f, cx, cy, k1, k2
    fit_f: bool = True
    fit_pp: bool = True                # principal point
    fit_k1: bool = False
    fit_k2: bool = False
    fit_latency: bool = True           # one video/joint-log offset per camera (frames; + = video late)
    shared_latency: bool = False
    fit_tool: bool = False             # offset of the hand along its z axis (m), shared by every camera
    fit_board_z: bool = False          # board z offset (m)
    loss: str = "cauchy"
    f_scale: float = 2.0               # px
    w_auto: float = 1.0
    w_mark: float = 1.0
    K: int = 3


class BA:
    def __init__(self, cams: dict[str, Cam], obs: list[Obs], board_pose: tuple[np.ndarray, np.ndarray],
                 kin: Kinematics, cfg: FitConfig):
        self.cfg, self.kin = cfg, kin
        self.cams0 = cams
        self.names = [c for c in cfg.cams if c in cams]
        self.bp, self.bR = np.asarray(board_pose[0], float), w.quat2mat(board_pose[1])
        self.set_obs(obs)
        self.latency = {c: 0.0 for c in self.names}
        self.tool = 0.0
        self.board_dz = 0.0

    def set_obs(self, obs: list[Obs]):
        self.obs = [o for o in obs if o.cam in self.names]
        P, Z = obs_world_tracks(self.obs, self.kin, self.cfg.K)
        isb = np.array([o.body == "board" for o in self.obs])
        P[isb] = (P[isb] @ self.bR.T) + self.bp
        self.P, self.Z, self.isb = P, Z, isb
        self.ishand = np.array([o.body in HAND_BODIES for o in self.obs])
        self.uv = np.array([o.uv for o in self.obs]) if self.obs else np.zeros((0, 2))
        self.wt = np.array([o.w * (self.cfg.w_auto if o.src == "auto" else self.cfg.w_mark) for o in self.obs])
        self.ci = np.array([self.names.index(o.cam) for o in self.obs], int)

    # parameter vector --------------------------------------------------------------------------
    def _groups(self):
        return ["all"] if self.cfg.shared_intrinsics else list(self.names)

    def pack(self, cams: dict[str, Cam]) -> np.ndarray:
        v = []
        for c in self.names:
            v += list(cv2.Rodrigues(cams[c].R)[0].ravel()) + list(cams[c].t)
        for g in self._groups():
            c = cams[self.names[0] if g == "all" else g]
            if self.cfg.fit_f:
                v.append(c.f)
            if self.cfg.fit_pp:
                v += [c.cx, c.cy]
            if self.cfg.fit_k1:
                v.append(c.k1)
            if self.cfg.fit_k2:
                v.append(c.k2)
        if self.cfg.fit_latency:
            v += [self.latency[self.names[0]]] if self.cfg.shared_latency else [self.latency[c] for c in self.names]
        if self.cfg.fit_tool:
            v.append(self.tool)
        if self.cfg.fit_board_z:
            v.append(self.board_dz)
        return np.array(v, float)

    def unpack(self, v) -> tuple[dict[str, Cam], dict, float, float]:
        v = list(v)
        cams = {}
        for c in self.names:
            r, t = np.array(v[:3]), np.array(v[3:6])
            del v[:6]
            c0 = self.cams0[c]
            cams[c] = Cam(c, cv2.Rodrigues(r)[0], t, c0.f, c0.cx, c0.cy, c0.k1, c0.k2, c0.parent, c0.fy_ratio)
        for g in self._groups():
            tgt = self.names if g == "all" else [g]
            vals = {}
            if self.cfg.fit_f:
                vals["f"] = v.pop(0)
            if self.cfg.fit_pp:
                vals["cx"], vals["cy"] = v.pop(0), v.pop(0)
            if self.cfg.fit_k1:
                vals["k1"] = v.pop(0)
            if self.cfg.fit_k2:
                vals["k2"] = v.pop(0)
            for c in tgt:
                for k, x in vals.items():
                    setattr(cams[c], k, x)
        lat = dict(self.latency)
        if self.cfg.fit_latency:
            if self.cfg.shared_latency:
                x = v.pop(0)
                lat = {c: x for c in self.names}
            else:
                lat = {c: v.pop(0) for c in self.names}
        tool = v.pop(0) if self.cfg.fit_tool else self.tool
        bdz = v.pop(0) if self.cfg.fit_board_z else self.board_dz
        return cams, lat, tool, bdz

    # model -------------------------------------------------------------------------------------
    def world_points(self, lat: dict, tool: float, bdz: float, idx=None) -> np.ndarray:
        idx = np.arange(len(self.obs)) if idx is None else idx
        K = self.cfg.K
        d = np.array([lat[self.names[c]] for c in self.ci[idx]])
        d = np.clip(d, -K + 1e-6, K - 1e-6)
        j0 = np.floor(d).astype(int) + K
        a = (d - np.floor(d))[:, None]
        P = self.P[idx]
        X = (1 - a) * P[np.arange(len(idx)), j0] + a * P[np.arange(len(idx)), j0 + 1]
        X = X + tool * self.Z[idx] * self.ishand[idx, None]
        X[:, 2] += bdz * self.isb[idx]
        return X

    def predict(self, cams, lat, tool, bdz, idx=None) -> np.ndarray:
        idx = np.arange(len(self.obs)) if idx is None else idx
        X = self.world_points(lat, tool, bdz, idx)
        out = np.zeros((len(idx), 2))
        for k, c in enumerate(self.names):
            s = self.ci[idx] == k
            if s.any():
                out[s] = cams[c].project(X[s])[0]
        return out

    def residuals(self, v) -> np.ndarray:
        cams, lat, tool, bdz = self.unpack(v)
        r = (self.predict(cams, lat, tool, bdz) - self.uv) * self.wt[:, None]
        # weak priors (0 +- 50 mm, 0 +- 5 frames) keep the scalar offsets bounded when the few observations that
        # pin them (board marks) are down-weighted by the robust loss early on; negligible against 1000s of points
        pri = [tool / 0.05, bdz / 0.05] + [lat[c] / 5.0 for c in self.names]
        return np.r_[r.ravel(), np.array(pri) * self.cfg.f_scale]

    def fit(self, max_nfev: int = 200) -> dict:
        from scipy.optimize import least_squares
        v0 = self.pack(self.cams0)
        res = least_squares(self.residuals, v0, loss=self.cfg.loss, f_scale=self.cfg.f_scale, x_scale="jac",
                            max_nfev=max_nfev * len(v0))
        cams, lat, tool, bdz = self.unpack(res.x)
        self.cams0, self.latency, self.tool, self.board_dz = cams, lat, tool, bdz
        # parameter standard errors from the Gauss-Newton covariance (robust-weighted), scaled by residual
        J = res.jac
        dof = max(1, len(res.fun) - len(res.x))
        s2 = float(np.sum(res.fun ** 2) / dof)
        try:
            cov = np.linalg.pinv(J.T @ J) * s2
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            se = np.full(len(res.x), np.nan)
        return {"cost": float(res.cost), "nfev": int(res.nfev), "status": int(res.status), "x": res.x, "se": se}

    def per_obs_error(self, idx=None) -> np.ndarray:
        idx = np.arange(len(self.obs)) if idx is None else idx
        return np.linalg.norm(self.predict(self.cams0, self.latency, self.tool, self.board_dz, idx) - self.uv[idx], axis=1)


# ----------------------------------------------------------------------------------------------
# Render-and-compare features (auto marks)
# ----------------------------------------------------------------------------------------------
def render_rgb_ids_depth(view: "w.TwinView", cam: "w.CameraModel"):
    rgb = view.render_rgb(cam)
    ids = view.render_geom_ids(cam)
    r = view.renderer
    r.enable_depth_rendering()
    r.update_scene(view.data, camera=cam.name, scene_option=view.opt)
    depth = r.render().copy()
    r.disable_depth_rendering()
    return rgb, ids, depth


def backproject(cam: "w.CameraModel", u, v, z) -> np.ndarray:
    """Pixel (u, v) at z-depth z (m, along the optical axis) -> point in the camera's parent frame."""
    x = (np.asarray(u, float) - cam.cx) / cam.fx * z
    y = (np.asarray(v, float) - cam.cy) / cam.fy * z
    Xc = np.stack([x, y, np.asarray(z, float)], -1)
    return Xc @ cam.R_cv.T + np.asarray(cam.pos)


def _zncc_match(real_g, tmpl, mask, u0, v0, R):
    h, wd = tmpl.shape
    padw = R + max(h, wd)
    pad = np.pad(real_g, padw, mode="edge")
    sub = pad[v0 - R + padw:v0 + R + h + padw, u0 - R + padw:u0 + R + wd + padw]
    res = cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED, mask=mask.astype(np.float32))
    res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
    j, i = np.unravel_index(int(np.argmax(res)), res.shape)
    du, dv = float(i - R), float(j - R)
    if 0 < i < res.shape[1] - 1:
        a, b, c = res[j, i - 1], res[j, i], res[j, i + 1]
        den = a - 2 * b + c
        if den < 0:
            du += 0.5 * (a - c) / den
    if 0 < j < res.shape[0] - 1:
        a, b, c = res[j - 1, i], res[j, i], res[j + 1, i]
        den = a - 2 * b + c
        if den < 0:
            dv += 0.5 * (a - c) / den
    # sharpness: best score minus the best score outside a 3 px radius (ambiguity check)
    yy, xx = np.mgrid[:res.shape[0], :res.shape[1]]
    far = (xx - i) ** 2 + (yy - j) ** 2 > 9
    second = float(res[far].max()) if far.any() else -1.0
    return du, dv, float(res[j, i]), float(res[j, i]) - second, bool(i in (0, res.shape[1] - 1) or j in (0, res.shape[0] - 1))


def auto_features(view: "w.TwinView", cam: "w.CameraModel", episode: int, frame: int, real: np.ndarray,
                  features: dict = AUTO_FEATURES, R: int = 15, margin: int = 8, min_px: int = 40,
                  min_px_pad: int = 25) -> list[dict]:
    """Match each feature's rendered patch (robot pixels only, gray, ZNCC) against the real frame within +-R px.
    Returns per feature: body, X_body (visible-surface centroid of the feature, body frame), the observed uv
    (rendered centroid + shift), score, sharpness, edge flag."""
    view.set_frame(episode, frame)
    rgb, ids, depth = render_rgb_ids_depth(view, cam)
    lab = view.render_labels(cam, ids)
    robot = np.isin(lab, [w.LABELS["arm"], w.LABELS["hand"]])
    rg = cv2.cvtColor(real, cv2.COLOR_RGB2GRAY).astype(np.float32)
    tg = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = []
    for name, (body, gl) in features.items():
        mk = np.isin(ids, gl)
        n = int(mk.sum())
        if n < (min_px_pad if name.startswith("pad") else min_px):
            continue
        ys, xs = np.nonzero(mk)
        u0, u1 = max(int(xs.min()) - margin, 0), min(int(xs.max()) + margin, w.W - 1)
        v0, v1 = max(int(ys.min()) - margin, 0), min(int(ys.max()) + margin, w.H - 1)
        mm = robot[v0:v1 + 1, u0:u1 + 1]
        if mm.sum() < 30:
            continue
        du, dv, sc, sharp, edge = _zncc_match(rg, tg[v0:v1 + 1, u0:u1 + 1], mm, u0, v0, R)
        Xp = backproject(cam, xs, ys, depth[ys, xs]).mean(0)  # in the camera's parent frame
        Rp, pp = view.parent_pose(cam.parent)
        Xw = Rp @ Xp + pp
        b = mujoco.mj_name2id(view.model, mujoco.mjtObj.mjOBJ_BODY, body)
        Rb, pb = view.data.xmat[b].reshape(3, 3), view.data.xpos[b]
        Xb = Rb.T @ (Xw - pb)
        uvr, _ = w.project(cam, Xp[None])
        out.append({"name": name, "body": body, "X": Xb, "uv": uvr[0] + [du, dv], "shift": [du, dv], "score": sc,
                    "sharp": sharp, "edge": edge, "npx": n})
    return out


def collect_auto(episode: int, cams: dict[str, "w.CameraModel"], frames: list[int], R: int = 15,
                 min_score: float = 0.6, min_sharp: float = 0.02, board_pose=None, hand_offset: float = 0.0,
                 latency: dict | None = None, log=print) -> list[Obs]:
    """Auto observations on `frames` for every camera. hand_offset (m) moves the twin hand along its z axis
    (tool-length correction) before rendering. latency {cam: frames}: render the robot at frame + latency (rounded
    to the nearest frame) so moving frames match; the Obs keep the video frame index."""
    view = w.get_view()
    view.set_hand_offset(hand_offset)
    if board_pose is not None:
        view.set_board(*board_pose)
    out = []
    for k, f in enumerate(frames):
        for name, cm in cams.items():
            real = w.read_frame(episode, name, f)
            fr = int(np.clip(round(f + (latency or {}).get(name, 0.0)), 0, w.episode_state(episode).n - 1))
            for r in auto_features(view, cm, episode, fr, real, R=R):
                if r["score"] < min_score or r["sharp"] < min_sharp or r["edge"]:
                    continue
                out.append(Obs(name, f, np.asarray(r["uv"]), r["body"], np.asarray(r["X"]), r["name"], "auto",
                               1.0, r["score"]))
        if log and k % 20 == 0:
            log(f"[auto] {k + 1}/{len(frames)} frames, {len(out)} obs")
    view.set_hand_offset(0.0)
    return out


# ----------------------------------------------------------------------------------------------
# Frame choice
# ----------------------------------------------------------------------------------------------
def free_frames(episode: int, exclude: list[int], radius: int = 30, margin_s: float = 1.0) -> np.ndarray:
    """Boolean mask of frames more than `radius` from every excluded frame (e.g. a watcher's held-out frames)."""
    n = w.episode_state(episode).n
    ok = np.ones(n, bool)
    for f in exclude:
        ok[max(0, int(f) - radius):int(f) + radius + 1] = False
    m = int(margin_s * w.FPS)
    ok[:m] = ok[n - m:] = False
    return ok


def ee_speed(episode: int) -> np.ndarray:
    s = w.episode_state(episode)
    return np.linalg.norm(np.gradient(s.ee[:, :3], axis=0), axis=1) * w.FPS


# ----------------------------------------------------------------------------------------------
# Wrist camera: pose in the hand frame + intrinsics from finger points (near field) and board points (far field)
# ----------------------------------------------------------------------------------------------
@dataclass
class HandFitConfig:
    fit_f: bool = True
    fit_pp: bool = True
    fit_k1: bool = False
    f_prior: float | None = None      # px; soft prior (e.g. RealSense D435 colour at 640x480 ~ 615)
    f_prior_sigma: float = 30.0
    loss: str = "cauchy"
    f_scale: float = 3.0
    w_finger: float = 1.0
    w_board: float = 1.0


def hand_points(obs: list[Obs], kin: Kinematics, tool: float, board_pose) -> np.ndarray:
    """Each obs point in the (tool-corrected) hand frame. Finger points: finger body pose relative to the hand
    (depends only on the gripper width). Board / world points: through FK of the hand at the obs frame, with
    the hand moved by `tool` along its z axis."""
    bp, bR = np.asarray(board_pose[0], float), w.quat2mat(board_pose[1])
    out = np.zeros((len(obs), 3))
    for i, o in enumerate(obs):
        Rh, ph = kin.pose(o.frame, "gripper/hand")
        if o.body in ("board", "world"):
            Xw = bR @ o.X + bp if o.body == "board" else o.X
            out[i] = Rh.T @ (Xw - (ph + tool * Rh[:, 2]))
        elif o.body == "gripper/hand":
            out[i] = o.X
        else:
            Rb, pb = kin.pose(o.frame, o.body)
            out[i] = Rh.T @ (Rb @ o.X + pb - ph)
    return out


class HandBA:
    def __init__(self, cam: Cam, obs: list[Obs], kin: Kinematics, tool: float, board_pose, cfg: HandFitConfig):
        self.cam0, self.obs, self.cfg = cam, obs, cfg
        self.X = hand_points(obs, kin, tool, board_pose)
        self.uv = np.array([o.uv for o in obs])
        self.isb = np.array([o.body in ("board", "world") for o in obs])
        self.wt = np.array([o.w * (cfg.w_board if b else cfg.w_finger) for o, b in zip(obs, self.isb)])

    def pack(self, c: Cam):
        v = list(cv2.Rodrigues(c.R)[0].ravel()) + list(c.t)
        if self.cfg.fit_f:
            v.append(c.f)
        if self.cfg.fit_pp:
            v += [c.cx, c.cy]
        if self.cfg.fit_k1:
            v.append(c.k1)
        return np.array(v, float)

    def unpack(self, v) -> Cam:
        v = list(v)
        c0 = self.cam0
        c = Cam(c0.name, cv2.Rodrigues(np.array(v[:3]))[0], np.array(v[3:6]), c0.f, c0.cx, c0.cy, c0.k1, c0.k2,
                c0.parent, c0.fy_ratio)
        del v[:6]
        if self.cfg.fit_f:
            c.f = v.pop(0)
        if self.cfg.fit_pp:
            c.cx, c.cy = v.pop(0), v.pop(0)
        if self.cfg.fit_k1:
            c.k1 = v.pop(0)
        return c

    def residuals(self, v):
        c = self.unpack(v)
        uv, z = c.project(self.X)
        r = ((uv - self.uv) * self.wt[:, None]).ravel()
        r = np.where(np.repeat(z, 2) > 0, r, 500.0)
        if self.cfg.f_prior is not None and self.cfg.fit_f:
            r = np.r_[r, (c.f - self.cfg.f_prior) / self.cfg.f_prior_sigma * self.cfg.f_scale]
        return r

    def fit(self, max_nfev=200):
        from scipy.optimize import least_squares
        v0 = self.pack(self.cam0)
        res = least_squares(self.residuals, v0, loss=self.cfg.loss, f_scale=self.cfg.f_scale, x_scale="jac",
                            max_nfev=max_nfev * len(v0))
        self.cam0 = self.unpack(res.x)
        return res

    def errors(self, obs=None, kin=None, tool=None, board_pose=None) -> np.ndarray:
        X, uv = (self.X, self.uv) if obs is None else (hand_points(obs, kin, tool, board_pose), np.array([o.uv for o in obs]))
        p, z = self.cam0.project(X)
        e = np.linalg.norm(p - uv, axis=1)
        e[z <= 0] = 500.0
        return e


def finger_corner_obs(view: "w.TwinView", cam: "w.CameraModel", episode: int, frame: int, real: np.ndarray,
                      dark: float = 70.0, max_dist: float = 60.0, grip=None, by_side: bool = False,
                      corners: tuple = ("inner_top",)) -> list[Obs]:
    """Wrist camera, open gripper: the inner top corner of each finger pad. Real corner = extreme pixel of the
    matching dark blob in the lower half (gray < `dark`); its 3D point = the rendered pad's inner top corner,
    back-projected through the twin's depth and fixed to the finger body. Same corner definition as the
    watcher's automatic finger keypoints (inner top: the outer side merges with the finger housing)."""
    view.set_frame(episode, frame)
    if grip is not None:  # a hand median image: robot pose irrelevant, only the finger opening
        view.set_robot(view.data.qpos[:7].copy(), grip)
    rgb, ids, depth = render_rgb_ids_depth(view, cam)
    pads = view.pad_geoms
    rend = np.isin(ids, pads)
    rend[: w.H // 2] = False
    g = cv2.cvtColor(real, cv2.COLOR_RGB2GRAY)
    rmask = (g < dark)
    rmask[: w.H // 2] = False
    rmask = cv2.morphologyEx(rmask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nq, lq, stq, cq = cv2.connectedComponentsWithStats(rmask, 8)
    nr, lr, str_, cr = cv2.connectedComponentsWithStats(rend.astype(np.uint8), 8)
    Rp, pp = view.parent_pose(cam.parent)
    out = []
    for i in range(1, nr):
        if str_[i, cv2.CC_STAT_AREA] < 200:
            continue
        side = "left" if cr[i][0] < w.W / 2 else "right"
        best, bd = None, max_dist
        if by_side:  # largest dark blob on the same image half
            js = [j for j in range(1, nq) if (cq[j][0] < w.W / 2) == (side == "left") and stq[j, cv2.CC_STAT_AREA] > 150]
            best = max(js, key=lambda j: stq[j, cv2.CC_STAT_AREA]) if js else None
        else:
            for j in range(1, nq):
                a = stq[j, cv2.CC_STAT_AREA]
                if not 0.3 * str_[i, cv2.CC_STAT_AREA] < a < 3.0 * str_[i, cv2.CC_STAT_AREA]:
                    continue
                dd = np.linalg.norm(cq[j] - cr[i])
                if dd < bd:
                    best, bd = j, dd
        if best is None:
            continue

        def corner(mask, which):
            ys, xs = np.nonzero(mask)
            inner = (side == "left") == (which.startswith("inner"))  # inner corner of the left pad is on its right
            k = np.argmin(ys - xs) if inner else np.argmin(ys + xs)
            return np.array([xs[k], ys[k]], float)
        for which in corners:
            cr_ = corner(lr == i, which)
            cq_ = corner(lq == best, which)
            u, v = int(cr_[0]), int(cr_[1])
            Xp = backproject(cam, u, v, depth[v, u])
            Xw = Rp @ Xp + pp
            gid = ids[v, u]
            b = view.model.geom_bodyid[gid]
            body = mujoco.mj_id2name(view.model, mujoco.mjtObj.mjOBJ_BODY, b)
            Rb, pb = view.data.xmat[b].reshape(3, 3), view.data.xpos[b]
            out.append(Obs(cam.name, frame, cq_, body, Rb.T @ (Xw - pb), f"{side}_pad_{which}", "auto", 1.0, 1.0))
    return out


# ----------------------------------------------------------------------------------------------
# Pipelines (used by harness/tools/world.py: calibrate_static_cameras, calibrate_hand_camera, mark_candidates)
# ----------------------------------------------------------------------------------------------
class FrameCache:
    """Decode the requested frames of one episode/camera sequentially once (seeking per frame is ~10x slower)."""

    def __init__(self, episode: int):
        self.episode, self.d = episode, {}

    def prefetch(self, cam: str, frames):
        import av
        need = sorted({int(f) for f in frames} - {f for (c, f) in self.d if c == cam})
        if not need:
            return
        path, t0 = w.video_ref(self.episode, cam)
        want = set(need)
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            c.seek(int(max(t0 + need[0] / w.FPS - 0.5, 0) / st.time_base), stream=st, backward=True)
            for fr in c.decode(st):
                if fr.time is None:
                    continue
                fi = int(round((fr.time - t0) * w.FPS))
                if fi in want:
                    self.d[(cam, fi)] = fr.to_ndarray(format="rgb24")
                    want.discard(fi)
                if not want or fi > need[-1]:
                    break

    def get(self, cam: str, f: int) -> np.ndarray:
        if (cam, int(f)) not in self.d:
            self.d[(cam, int(f))] = w.read_frame(self.episode, cam, int(f))
        return self.d[(cam, int(f))]


def board_pose_with_z(calib: dict, board_dz: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Contact board pose (x, y, yaw from robot contacts) with z raised by board_dz (plate top from cameras)."""
    b = calib.get("board_contact") or calib["board"]
    return np.asarray(b["pos"], float) + np.array([0, 0, board_dz]), np.asarray(b["quat"], float)


def static_pipeline(episode: int, cams0: dict[str, "w.CameraModel"], frames: list[int], marks: list[dict],
                    cfg: FitConfig, iters: int = 2, R0: int = 25, R1: int = 8, min_score_arm: float = 0.7,
                    min_score_pad: float = 0.75, w_marks: float = 3.0, calib: dict | None = None,
                    holdout_every: int = 3, fc: FrameCache | None = None, log=print) -> dict:
    """Iterate: auto features (render-and-compare) with the current model -> BA (+ marks) -> repeat with a
    smaller search radius. Every `holdout_every`-th frame is held out of the fit and scored."""
    calib = calib or w.load_calib(w.recording_of(episode))
    fc = fc or FrameCache(episode)
    for c in cams0:
        fc.prefetch(c, frames)
    orig = w.read_frame
    w.read_frame = lambda ep, cam, f: fc.get(cam, f) if ep == episode else orig(ep, cam, f)
    try:
        kin = Kinematics(episode)
        cams = {k: Cam.from_model(v) for k, v in cams0.items()}
        lat, tool, bdz = {k: 0.0 for k in cams}, 0.0, 0.0
        mk = marks_to_obs(marks)
        for o in mk:
            o.w *= w_marks
        hist = []
        for it in range(iters):
            models = {k: v.to_model() for k, v in cams.items()}
            obs = collect_auto(episode, models, frames, R=R0 if it == 0 else R1, min_score=0.5, min_sharp=0.0,
                               hand_offset=tool, latency=lat, log=None)
            arm = [o for o in obs if not o.name.startswith("pad") and o.name != "blk7" and o.score >= min_score_arm]
            pads = [o for o in obs if o.name.startswith("pad") and o.score >= min_score_pad]
            fit_o = [o for o in arm + pads if (frames.index(o.frame) % holdout_every) != 0]
            ho_o = [o for o in arm + pads if (frames.index(o.frame) % holdout_every) == 0]
            ba = BA(cams, fit_o + mk, board_pose_with_z(calib, 0.0), kin, cfg)
            ba.latency, ba.tool, ba.board_dz = dict(lat), tool, bdz
            res = ba.fit()
            cams, lat, tool, bdz = ba.cams0, ba.latency, ba.tool, ba.board_dz
            ev = BA(cams, ho_o, board_pose_with_z(calib, 0.0), kin, cfg)
            ev.latency, ev.tool, ev.board_dz = lat, tool, bdz
            e = ev.per_obs_error()
            summ = {}
            for c in cams:
                for grp, sel in (("arm", lambda o: not o.name.startswith("pad")), ("pads", lambda o: o.name.startswith("pad"))):
                    s = np.array([o.cam == c and sel(o) for o in ev.obs])
                    summ[f"{c}_{grp}_heldout_px"] = round(float(np.median(e[s])), 2) if s.any() else None
            emk = ba.per_obs_error(np.array([i for i, o in enumerate(ba.obs) if o.src == "mark"], int)) if mk else []
            rec = {"iter": it, "n_fit": len(fit_o), "n_heldout": len(ho_o), "f": round(cams[ba.names[0]].f, 2),
                   "latency_frames": {k: round(v, 3) for k, v in lat.items()}, "tool_mm": round(tool * 1000, 2),
                   "board_dz_mm": round(bdz * 1000, 2), "marks_px": [round(float(x), 2) for x in emk], **summ}
            hist.append(rec)
            if log:
                log(f"[static] {rec}")
        se = dict(zip(["param_%d" % i for i in range(len(res["se"]))], [float(x) for x in res["se"]]))
        return {"cams": cams, "latency": lat, "tool": tool, "board_dz": bdz, "history": hist, "se": se,
                "n_auto": len(obs)}
    finally:
        w.read_frame = orig


def median_corner_obs(view: "w.TwinView", cam: "w.CameraModel", episode: int) -> list[Obs]:
    """Finger-pad top corners on the open / closed hand medians (clean near-field picture of the fingers)."""
    s = w.episode_state(episode)
    g = s.grip.sum(1)
    out = []
    for (grip, med), corners in zip(w.hand_medians(episode, cam.name), [("inner_top", "outer_top"), ("outer_top",)]):
        f = int(np.argmin(np.abs(g - grip.sum())))  # a frame with this opening, so FK puts the fingers there
        out += finger_corner_obs(view, cam, episode, f, med, grip=grip, by_side=True, corners=corners)
    return out


def hand_pipeline(episode: int, cam0: "w.CameraModel", hole_marks: list[dict], tool: float, board_pose,
                  cfg: HandFitConfig, iters: int = 3, w_near: float = 1.0, log=print) -> tuple[Cam, dict]:
    """Wrist camera in the (tool-corrected) hand frame from far-field board marks + near-field pad corners on
    the hand medians; the near-field 3D points are re-rendered with each new camera estimate. Leave-one-key-out
    over the board marks gives the held-out board error."""
    view = w.get_view()
    view.set_hand_offset(tool)
    kin = Kinematics(episode)

    def run(marks, n_it, quiet=True):
        cam = Cam.from_model(cam0)
        holes = marks_to_obs(marks)
        hb = None
        for it in range(n_it):
            near = median_corner_obs(view, cam.to_model(), episode)
            for o in near:
                o.w = w_near
            hb = HandBA(cam, holes + near, kin, tool, board_pose, cfg)
            hb.fit()
            cam = hb.cam0
            if log and not quiet:
                e = hb.errors()
                isn = np.array([o.body != "board" for o in hb.obs])
                log(f"[hand] it{it} f {cam.f:.1f} pos_mm {np.round(cam.pos * 1000, 1).tolist()} near {np.round(e[isn], 1).tolist()} "
                    f"far median {np.median(e[~isn]):.2f}")
        return cam, hb
    loo = []
    for k in sorted({m["key"] for m in hole_marks}):
        tr = [m for m in hole_marks if m["key"] != k]
        te = [m for m in hole_marks if m["key"] == k]
        if not tr:
            continue
        cam, _ = run(tr, max(1, iters - 1))
        e = HandBA(cam, marks_to_obs(te), kin, tool, board_pose, cfg).errors()
        loo += [{"key": k, "frame": m["frame"], "px": round(float(x), 2)} for m, x in zip(te, e)]
    cam, hb = run(hole_marks, iters, quiet=False)
    e = hb.errors()
    isn = np.array([o.body != "board" for o in hb.obs])
    view.set_hand_offset(0.0)
    return cam, {"fit_board_px_median": round(float(np.median(e[~isn])), 2),
                 "fit_near_px": [round(float(x), 2) for x in e[isn]],
                 "board_leave_one_key_out_px_median": round(float(np.median([r["px"] for r in loo])), 2) if loo else None,
                 "board_leave_one_key_out": loo}


def heldout_finger_corners(episode: int, cam: Cam, tool: float, frames: list[int], fc: FrameCache | None = None,
                           max_dist: float = 25.0) -> np.ndarray:
    """Inner top pad corners on individual open-gripper frames (not the medians): px error per corner."""
    view = w.get_view()
    view.set_hand_offset(tool)
    kin = Kinematics(episode)
    obs = []
    for f in frames:
        real = fc.get("hand", f) if fc else w.read_frame(episode, "hand", f)
        obs += finger_corner_obs(view, cam.to_model(), episode, f, real, max_dist=max_dist)
    view.set_hand_offset(0.0)
    if not obs:
        return np.zeros(0)
    hb = HandBA(cam, obs, kin, tool, (np.zeros(3), np.array([1.0, 0, 0, 0])), HandFitConfig())
    return hb.errors()


def eval_marks(cams: dict[str, Cam], latency: dict, tool: float, board_pose, marks: list[dict],
               episode: int = 0) -> dict:
    """Median / p90 px of marked static-camera keypoints under a calibration (held-out check)."""
    kin = Kinematics(episode)
    obs = marks_to_obs(marks)
    ba = BA(cams, obs, board_pose, kin, FitConfig(cams=tuple(cams)))
    ba.latency, ba.tool = {k: latency.get(k, 0.0) for k in ba.names}, tool
    e = ba.per_obs_error()
    out = {}
    for c in ba.names:
        s = np.array([o.cam == c for o in ba.obs])
        if s.any():
            out[c] = {"n": int(s.sum()), "median_px": round(float(np.median(e[s])), 2),
                      "p90_px": round(float(np.percentile(e[s], 90)), 2)}
    return out


def two_view(cams: dict[str, Cam], marks: list[dict], a: str = "hama1", b: str = "hama2") -> list[dict]:
    """Triangulate every key marked in both cameras at the same frame; the larger per-view reprojection residual
    (px) is how far the two calibrated cameras disagree about that point."""
    from scipy.optimize import least_squares
    by = {}
    for m in marks:
        by.setdefault((int(m["frame"]), m["key"]), {})[m["cam"]] = np.asarray(m["uv"], float)
    out = []
    for (f, k), d in sorted(by.items()):
        if a in d and b in d:
            def r(X, d=d):
                return np.concatenate([cams[a].project(X[None])[0][0] - d[a], cams[b].project(X[None])[0][0] - d[b]])
            sol = least_squares(r, np.array([0.5, 0.0, 0.2]))
            out.append({"frame": f, "key": k, "world": [round(float(v), 4) for v in sol.x],
                        "px": round(float(max(np.hypot(*sol.fun[:2]), np.hypot(*sol.fun[2:]))), 2)})
    return out


def dark_blobs(img: np.ndarray, cu: float, cv: float, half: int = 40, rel: float = 0.55, amin: int = 6,
               amax: int = 3000) -> list[dict]:
    """Dark connected blobs (gray < rel x local median) in a window: candidates for pads or empty holes. The
    position is the darkness-weighted centroid, so a brain only has to say which blob is which feature."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    u0, v0 = max(int(cu) - half, 0), max(int(cv) - half, 0)
    u1, v1 = min(int(cu) + half, img.shape[1] - 1), min(int(cv) + half, img.shape[0] - 1)
    win = g[v0:v1 + 1, u0:u1 + 1]
    bgv = float(np.median(win))
    n, lab, st, _ = cv2.connectedComponentsWithStats((win < rel * bgv).astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if amin <= a <= amax:
            ys, xs = np.nonzero(lab == i)
            wts = np.clip(bgv - win[ys, xs], 1e-3, None)
            out.append({"id": len(out), "uv": [round(float(u0 + np.average(xs, weights=wts)), 2),
                                                round(float(v0 + np.average(ys, weights=wts)), 2)], "area": a})
    return out


def crop_with_grid(img: np.ndarray, cu: float, cv: float, half: int = 40, zoom: int = 4, step: int = 10,
                   label: str = "", blobs: list[dict] | None = None) -> np.ndarray:
    """Zoomed crop with a labelled pixel grid (for reading positions) and numbered blob candidates."""
    u0, v0 = int(round(cu)) - half, int(round(cv)) - half
    P = 4 * half
    pad = np.zeros((img.shape[0] + 2 * P, img.shape[1] + 2 * P, 3), np.uint8)
    pad[P:P + img.shape[0], P:P + img.shape[1]] = img
    n = 2 * half + 1
    z = cv2.resize(pad[v0 + P:v0 + P + n, u0 + P:u0 + P + n], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    ov = z.copy()
    for i in range(n):
        for val, horiz in ((u0 + i, False), (v0 + i, True)):
            if val % step == 0:
                x = i * zoom + zoom // 2
                if horiz:
                    cv2.line(ov, (0, x), (z.shape[1] - 1, x), (0, 255, 255), 1)
                    cv2.putText(z, str(val), (2, x - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 0, 0), 1)
                else:
                    cv2.line(ov, (x, 0), (x, z.shape[0] - 1), (0, 255, 255), 1)
                    cv2.putText(z, str(val), (x + 2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 0, 0), 1)
    z = cv2.addWeighted(z, 0.65, ov, 0.35, 0)
    for b in blobs or []:
        x, y = (b["uv"][0] - u0) * zoom + zoom // 2, (b["uv"][1] - v0) * zoom + zoom // 2
        cv2.drawMarker(z, (int(x), int(y)), (255, 0, 0), cv2.MARKER_CROSS, 8, 1)
        cv2.putText(z, str(b["id"]), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    if label:
        cv2.putText(z, label, (4, z.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(z, label, (4, z.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return z


def write_calib(episode: int, cams: dict[str, Cam], statuses: dict, methods: dict, evals: dict, latency: dict,
                tool: float, board_dz: float, notes: str, extra: dict | None = None) -> Path:
    """Write twin/calib/<recording>.json: new cameras, the board (contact x, y, yaw + camera plate-top z), the
    tool correction; the cameras and board it replaces are kept under "previous" (once: an existing "previous"
    is not overwritten)."""
    import time
    rec = w.recording_of(episode)
    path = w.calib_path(rec)
    doc = w.load_calib(rec) or {}
    if "previous" not in doc:
        doc["previous"] = {"note": "Values before the robot-keypoint recalibration of " + time.strftime("%Y-%m-%d %H:%M")
                                   + ". The static cameras came from silhouette/edge refinement (refine_camera) and the "
                                   "edge-overlay check; an independent watcher measured them at 5.35 px (hama1) and "
                                   "16 px (hama2) on held-out finger pads, 7.1 px two-view, wrist camera 33 px. Kept for "
                                   "comparison only.",
                           "cameras": doc.get("cameras", {}), "board": doc.get("board"), "tcp": doc.get("tcp")}
    doc.setdefault("cameras", {})
    for name, c in cams.items():
        e = c.to_model().to_dict()
        e.update({"status": statuses.get(name, "unchecked"), "method": methods.get(name, ""),
                  "latency_frames": round(float(latency.get(name, 0.0)), 3), "dist_k1_k2": [c.k1, c.k2]})
        if name in evals:
            e["eval"] = evals[name]
        doc["cameras"][name] = e
    bpos, bquat = board_pose_with_z(doc, board_dz)
    doc["board"] = {
        "pos": [round(float(v), 5) for v in bpos], "quat": [round(float(v), 7) for v in bquat],
        "source": "x, y, yaw from robot contacts (board_contact, board_from_contacts); z = plate top from the two "
                  f"calibrated static cameras (plate corners triangulated with the robot-calibrated cameras: board_contact "
                  f"z {doc['board_contact']['pos'][2]:.4f} + {board_dz * 1000:.1f} mm, about +-4 mm)",
        "plate_in_board_frame": {"corners": w.BOARD_PLATE.get("corners"), "top_z": 0.0,
                                 "thickness": w.BOARD_PLATE.get("thickness"),
                                 "source": w.BOARD_PLATE.get("source")}}
    tcp = dict(doc.get("tcp") or {})
    tcp.update({"hand_offset_m": round(float(tool), 4),
                "hand_offset_note": "Physical hand (and fingers, and the wrist camera mount) sits this far along the hand z "
                                    "axis from where base.xml puts it: both static cameras see the finger pads "
                                    f"{-tool * 1000:.1f} mm closer to the flange than base.xml (F/T stack "
                                    f"{(0.0706 + tool) * 1000:.1f} mm, not 70.6; the REASSEMBLE URDF stack is 44.5 mm). The "
                                    "logged O_T_EE (0.174 m from the flange) therefore lies about "
                                    f"{-tool * 1000:.0f} mm beyond the pad centres. base.xml is unchanged (replay builds on "
                                    "it); renders apply it with view.set_hand_offset(calib_hand_offset(rec)). Proposed fix: "
                                    "FT_STACK = 0.0706 + hand_offset_m in build_base.py, with the tcp site moved by "
                                    "-hand_offset_m so it stays on O_T_EE."})
    doc["tcp"] = tcp
    doc["notes"] = notes
    if extra:
        doc.update(extra)
    doc["written_by"] = "twin/camcal.py write_calib (robot-keypoint recalibration)"
    doc["written_at"] = time.strftime("%Y-%m-%d %H:%M")
    path.write_text(json.dumps(doc, indent=1, default=float))
    return path


def calib_state(episode: int, calib: dict | None = None):
    """(cams as Cam, latency, tool, board_pose) from twin/calib/<recording>.json."""
    calib = calib or w.load_calib(w.recording_of(episode))
    cams = {k: Cam.from_model(w.CameraModel.from_dict(v), *(v.get("dist_k1_k2") or [0.0, 0.0]))
            for k, v in calib["cameras"].items()}
    lat = {k: float(v.get("latency_frames", 0.0)) for k, v in calib["cameras"].items()}
    b = calib["board"]
    return cams, lat, w.calib_hand_offset("", calib), (np.asarray(b["pos"], float), np.asarray(b["quat"], float))


def project_key(cam: Cam, key: str, frame: int, kin: Kinematics, tool: float, board_pose, latency: float = 0.0):
    """Predicted pixel of a named key (ROBOT_KEYS or board_keys()) in one frame, for the camera's parent frame."""
    bk = board_keys()
    f = int(round(frame + latency))
    if key in ROBOT_KEYS:
        body, p = ROBOT_KEYS[key]
        R, t = kin.pose(f, body)
        Xw = R @ np.asarray(p) + t + tool * kin.pose(f, "gripper/hand")[0][:, 2]
    else:
        Xw = w.quat2mat(board_pose[1]) @ bk[key] + board_pose[0]
    if cam.parent != "world":
        Rh, ph = kin.pose(f, cam.parent)
        ph = ph + tool * Rh[:, 2]
        Xw = Rh.T @ (Xw - ph)
    uv, z = cam.project(Xw[None])
    return uv[0], float(z[0])
