"""SO-101 track: cameras and joint map from the robot's own silhouette (stage T2, reconstruct-scene steps 1-3).

Why silhouettes and not patches: the SO-101 here is printed in matte black with no texture, so REASSEMBLE's
masked-ZNCC patches on joint caps (twin/camcal.py) have nothing to lock onto. Its outline against the white table
and wall is sharp instead. The robot is still the calibration object (robot-supported calibration, Werner et al.
2018; render-and-compare as in EasyHeC / CalibAll): render the twin at the logged joints, find the real outline
near each rendered outline point, then bundle-adjust everything that moves the outline.

Parameters fitted together (one least-squares problem, Cauchy loss, point-to-line residuals):
  * per camera: rotation (3), translation (3), focal length f (1), principal point (2, optional)
  * per arm joint 0-3: scale and offset of the dataset-unit -> radian map; wrist_roll: offset only (its scale is
    known exactly); the gripper does not move the outline we use (the white moving jaw is masked out)
  * per camera: latency (frames): frame i shows the arm at log time i + latency
Iterate: render -> match (search radius shrinks) -> fit.

Real robot mask: dark pixels (HSV value < thr) minus a static 'ignore' mask (dark background such as the cardboard
box in the side view, found as pixels that stay dark in the temporal median of frames where the arm is away).
Rendered pixels of the moving jaw (white on the real robot) and the image border are 'unknown': outline points
next to them are not used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from twin import so101_data as sd
from twin import so101_robot as sr

REPO = sr.REPO
ARM_BODIES = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper")
JAW_BODY = "moving_jaw_so101_v1"


# ---------------------------------------------------------------------------------------------- real masks
def dark_mask(img: np.ndarray, thr: int = 75) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 2] < thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m


def ignore_mask(cam: str, episodes: list[int], thr: int = 75, cache: Path | None = None) -> np.ndarray:
    """Static dark background: dark in the per-pixel median of frames sampled while the arm is mid-task."""
    if cache and cache.exists():
        return cv2.imread(str(cache), cv2.IMREAD_GRAYSCALE) > 0
    imgs = []
    for ep in episodes:
        n = sd.episodes()[ep]["length"]
        imgs.append(sd.frames(ep, cam, [n // 3, n // 2, 2 * n // 3]))
    med = np.median(np.concatenate(imgs), axis=0).astype(np.uint8)
    ig = dark_mask(med, thr) > 0
    # keep only large static blobs that are not the robot base: the base is dark in every frame too, so remove
    # blobs that touch the robot's rest silhouette region by requiring brown hue (cardboard) or image border
    hsv = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)
    brown = (hsv[..., 1] > 60) & (hsv[..., 0] < 30)
    n, lab = cv2.connectedComponents(ig.astype(np.uint8))
    out = np.zeros_like(ig)
    for k in range(1, n):
        blob = lab == k
        if brown[blob].mean() > 0.3:
            out |= blob
    out = cv2.dilate(out.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cache), out.astype(np.uint8) * 255)
    return out


# ---------------------------------------------------------------------------------------------- renderer
class SegRenderer:
    """Segmentation + depth renders of the bare robot model from an OpenCV camera."""

    def __init__(self, w: int = sr.W, h: int = sr.H, finger_ext: float = 0.0):
        spec = mujoco.MjSpec.from_file(str(sr.MJCF))
        spec.meshdir = str(sr.ASSETS / "assets")
        if finger_ext:
            sr.add_finger_ext(spec, finger_ext)
        spec.worldbody.add_camera(name="calib")
        self.m = spec.compile()
        self.m.vis.global_.offwidth = max(w, 640)
        self.m.vis.global_.offheight = max(h, 480)
        self.d = mujoco.MjData(self.m)
        self.w, self.h = w, h
        self.cid = self.m.camera("calib").id
        self.seg = mujoco.Renderer(self.m, h, w)
        self.seg.enable_segmentation_rendering()
        self.dep = mujoco.Renderer(self.m, h, w)
        self.dep.enable_depth_rendering()
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(j).id] for j in sr.JOINTS])
        self.jaw = self.m.body(JAW_BODY).id
        self.geom_body = self.m.geom_bodyid.copy()
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[2] = 1  # visual meshes only

    def render(self, q: np.ndarray, cam: sr.Camera, scale: float = 1.0, depth: bool = True):
        sr.set_camera(self.m, self.cid, cam, self.w, self.h, scale)
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)
        self.seg.update_scene(self.d, camera=self.cid, scene_option=self.opt)
        s = self.seg.render()
        z = None
        if depth:
            self.dep.update_scene(self.d, camera=self.cid, scene_option=self.opt)
            z = self.dep.render()
        gid = s[..., 0].astype(int)
        typ = s[..., 1]
        body = np.full(gid.shape, -1)
        ok = (gid >= 0) & (typ == mujoco.mjtObj.mjOBJ_GEOM)
        body[ok] = self.geom_body[gid[ok]]
        return body, z

    def close(self):
        self.seg.close()
        self.dep.close()


# ---------------------------------------------------------------------------------------------- parameters
class Params:
    """Vector layout: per camera [rotvec 3, t 3, f, cx, cy, latency, k1]; joints [s0 o0 ... s5 o5];
    cap: how far the teal cap's centroid sits beyond the CAD moving-jaw tip, along the tip direction (m)."""
    NC = 11

    def __init__(self, cams: dict[str, sr.Camera], jm: sr.JointMap, latency: dict[str, float], cap: float = 0.0,
                 ext: float = 0.0):
        self.names = list(cams)
        self.cams = cams
        self.jm = jm
        self.lat = dict(latency)
        self.cap = float(cap)
        self.ext = float(ext)

    def vec(self) -> np.ndarray:
        v = []
        for n in self.names:
            c = self.cams[n]
            v += list(Rotation.from_matrix(c.R).as_rotvec()) + list(c.t) + [c.f, c.cx, c.cy, self.lat[n], c.k1]
        for j in range(6):
            v += [self.jm.scale[j], self.jm.offset[j]]
        v += [self.cap, self.ext]
        return np.array(v, float)

    def set(self, v: np.ndarray) -> "Params":
        cams = {}
        lat = {}
        for k, n in enumerate(self.names):
            b = v[k * self.NC:(k + 1) * self.NC]
            cams[n] = sr.Camera(Rotation.from_rotvec(b[:3]).as_matrix(), b[3:6].copy(), b[6], b[7], b[8], b[10])
            lat[n] = b[9]
        j = v[len(self.names) * self.NC:]
        sc = np.array(j[0:12:2], float)
        of = np.array(j[1:12:2], float)
        return Params(cams, sr.JointMap(sc, of, "fit"), lat, j[12], j[13])


def interp_units(units: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear interpolation of per-frame units at fractional frame times t (clamped)."""
    t = np.clip(t, 0, len(units) - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(units) - 1)
    a = (t - i0)[:, None]
    return units[i0] * (1 - a) + units[i1] * a


# ---------------------------------------------------------------------------------------------- frames
def sample_frames(episodes: list[int], per_ep: int, phase: int = 0, motion_only: bool = False) -> list[tuple[int, int]]:
    out = []
    for ep in episodes:
        n = sd.episodes()[ep]["length"]
        st = sd.signals(ep)["state"]
        idx = np.linspace(5, n - 6, per_ep + 2)[1:-1].round().astype(int) + phase
        idx = np.clip(idx, 0, n - 1)
        if motion_only:
            sp = np.abs(np.gradient(st[:, :5], axis=0)).sum(1)
            idx = [i for i in idx if sp[i] > 0.3]
        out += [(ep, int(i)) for i in idx]
    return out


class FrameSet:
    """Real masks for (episode, frame) items of one camera, plus the log to interpolate joints."""

    def __init__(self, cam: str, items: list[tuple[int, int]], ignore: np.ndarray, thr: int = 75):
        self.cam = cam
        self.items = items
        self.ignore = ignore
        self.masks = []
        self.units = {}
        by_ep: dict[int, list[int]] = {}
        for ep, i in items:
            by_ep.setdefault(ep, []).append(i)
        imgs = {}
        for ep, idx in by_ep.items():
            fs = sd.frames(ep, cam, idx)
            for i, f in zip(idx, fs):
                imgs[(ep, i)] = f
            self.units[ep] = sd.signals(ep)["state"]
        self.teal = []
        for it in items:
            m = dark_mask(imgs[it], thr).astype(np.float32)
            self.masks.append(cv2.GaussianBlur(m, (3, 3), 0.7))
            self.teal.append(teal_centroid(imgs[it]))
        self.imgs = imgs


def teal_centroid(img: np.ndarray, min_area: int = 12):
    """Centroid of the largest teal blob (the cap on the moving jaw's tip), or None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= 75) & (hsv[..., 0] <= 100) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 50)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[k, cv2.CC_STAT_AREA] < min_area:
        return None
    return cen[k].astype(float)


def cap_local(cap: float, m: mujoco.MjModel | None = None) -> np.ndarray:
    """Teal cap centroid in the moving-jaw frame: the CAD tip moved `cap` metres along the tip's radial direction."""
    tip = sr._jaw_tips(m or sr.load_model())["moving_tip"]
    d = np.array([tip[0], tip[1], 0.0])
    return tip + cap * d / np.linalg.norm(d)


# ---------------------------------------------------------------------------------------------- matching
def outline_points(body: np.ndarray, depth: np.ndarray, cam: sr.Camera, jaw: int, ignore: np.ndarray, step: int = 3,
                   border: int = 4):
    """Rendered outline of the arm without the moving jaw: pixel, outward normal, 3D point, body id."""
    arm = (body >= 0) & (body != jaw)
    unknown = (body == jaw) | ignore
    unknown = cv2.dilate(unknown.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    a8 = arm.astype(np.uint8)
    er = cv2.erode(a8, np.ones((3, 3), np.uint8))
    edge = (a8 - er) > 0
    edge[:border] = edge[-border:] = False
    edge[:, :border] = edge[:, -border:] = False
    edge &= ~unknown
    ys, xs = np.nonzero(edge)
    if len(xs) == 0:
        return None
    sel = np.arange(0, len(xs), step)
    ys, xs = ys[sel], xs[sel]
    sm = cv2.GaussianBlur(a8.astype(np.float32), (7, 7), 2.0)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    n = -np.stack([gx[ys, xs], gy[ys, xs]], 1)
    nn = np.linalg.norm(n, axis=1)
    ok = nn > 1e-6
    ys, xs, n = ys[ok], xs[ok], n[ok] / nn[ok, None]
    z = depth[ys, xs]
    ok = z > 0
    ys, xs, n, z = ys[ok], xs[ok], n[ok], z[ok]
    uv = np.stack([xs, ys], 1).astype(float)
    c, d = cam.ray(uv)
    # depth from MuJoCo is distance along the optical axis
    zc = d @ cam.R[2]
    X = c + d * (z / zc)[:, None]
    return uv, n, X, body[ys, xs]


def match_outline(mask: np.ndarray, uv: np.ndarray, n: np.ndarray, radius: float, step: float = 0.5):
    """Search along each normal for the real mask's inside->outside crossing nearest to the rendered outline."""
    ts = np.arange(-radius, radius + step, step)
    P = uv[:, None, :] + ts[None, :, None] * n[:, None, :]
    vals = cv2.remap(mask, P[..., 0].astype(np.float32), P[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    inside = vals >= 0.5
    cross = inside[:, :-1] & ~inside[:, 1:]
    v0, v1 = vals[:, :-1], vals[:, 1:]
    tt = ts[None, :-1] + (v0 - 0.5) / np.maximum(v0 - v1, 1e-6) * step
    tt = np.where(cross, tt, np.inf)
    k = np.argmin(np.abs(tt), axis=1)
    best = tt[np.arange(len(uv)), k]
    best[~np.isfinite(best)] = np.nan
    return best


# ---------------------------------------------------------------------------------------------- fit
class Corr:
    def __init__(self):
        self.cam, self.ep, self.fr, self.body, self.local, self.n, self.target = [], [], [], [], [], [], []

    def arrays(self):
        return {k: np.array(getattr(self, k)) for k in ("cam", "ep", "fr", "body", "local", "n", "target")}


def collect(params: Params, sets: dict[str, FrameSet], rend: SegRenderer, radius: float, fk: sr.FK, step: int = 3):
    """Render every frame with the current params; return correspondences and per-frame stats."""
    C = Corr()
    stats = []
    sr.set_ext(rend.m, params.ext)
    for ci, (cn, fs) in enumerate(sets.items()):
        cam = params.cams[cn]
        for k, (ep, i) in enumerate(fs.items):
            u = interp_units(fs.units[ep], np.array([i + params.lat[cn]]))
            q = params.jm.q(u)[0]
            body, depth = rend.render(q, cam)
            o = outline_points(body, depth, cam, rend.jaw, fs.ignore, step=step)
            if o is None:
                continue
            uv, nrm, X, bid = o
            t = match_outline(fs.masks[k], uv, nrm, radius)
            ok = np.isfinite(t)
            stats.append({"cam": cn, "ep": ep, "frame": i, "n": int(len(uv)), "matched": int(ok.sum()),
                          "med_abs_px": float(np.median(np.abs(t[ok]))) if ok.any() else None})
            if not ok.any():
                continue
            uv, nrm, X, bid, t = uv[ok], nrm[ok], X[ok], bid[ok], t[ok]
            # world -> body local at this frame's pose
            ub, binv = np.unique(bid, return_inverse=True)
            P, Rm = fk.poses(q[None], list(ub))
            loc = np.einsum("nji,nj->ni", Rm[0, binv], X - P[0, binv])
            m_ = len(uv)
            C.cam += [ci] * m_
            C.ep += [ep] * m_
            C.fr += [i] * m_
            C.body += list(bid)
            C.local += list(loc)
            C.n += list(nrm)
            C.target += list(uv + t[:, None] * nrm)
    return C.arrays(), stats


def residuals(v: np.ndarray, base: Params, A: dict, sets: dict[str, FrameSet], fk: sr.FK, free_pp: bool,
              prior: dict | None = None, teal_w: float = 0.0, twoview=None, tv_w: float = 0.0):
    p = base.set(v)
    sr.set_ext(fk.m, p.ext)
    res = np.zeros(len(A["cam"]))
    for ci, cn in enumerate(sets):
        sel = A["cam"] == ci
        if not sel.any():
            continue
        cam = p.cams[cn]
        if not free_pp:
            cam.cx, cam.cy = (sr.W - 1) / 2, (sr.H - 1) / 2
        idx = np.nonzero(sel)[0]
        # group by (ep, frame)
        keys = A["ep"][idx] * 100000 + A["fr"][idx]
        uk, inv = np.unique(keys, return_inverse=True)
        eps = uk // 100000
        frs = uk % 100000
        U = np.zeros((len(uk), 6))
        for k in range(len(uk)):
            U[k] = interp_units(sets[cn].units[int(eps[k])], np.array([frs[k] + p.lat[cn]]))[0]
        Q = p.jm.q(U)
        bodies = A["body"][idx]
        ub, binv = np.unique(bodies, return_inverse=True)
        P, Rm = fk.poses(Q, list(ub))
        X = P[inv, binv] + np.einsum("nij,nj->ni", Rm[inv, binv], A["local"][idx])
        uv, z = cam.project(X)
        res[idx] = np.einsum("ni,ni->n", uv - A["target"][idx], A["n"][idx])
    extra = []
    if teal_w > 0:
        jaw = fk.m.body(JAW_BODY).id
        loc = cap_local(p.cap, fk.m)
        for cn, fs in sets.items():
            cam = p.cams[cn]
            if not free_pp:
                cam.cx, cam.cy = (sr.W - 1) / 2, (sr.H - 1) / 2
            ks = [k for k, t in enumerate(fs.teal) if t is not None]
            if not ks:
                continue
            U = np.array([interp_units(fs.units[fs.items[k][0]], np.array([fs.items[k][1] + p.lat[cn]]))[0] for k in ks])
            Xc = fk.points(p.jm.q(U), np.array([jaw]), loc[None])[:, 0]
            uv, _ = cam.project(Xc)
            extra.append(((uv - np.array([fs.teal[k] for k in ks])) * teal_w).ravel())
    if twoview and tv_w > 0:
        names = list(sets)
        cl = [p.cams[n] for n in names]
        tv = []
        for pair in twoview:
            X = triangulate(cl, [pair[n] for n in names])
            for c, n in zip(cl, names):
                tv.append((c.project(X)[0] - pair[n]) * tv_w)
        extra.append(np.ravel(tv))
    if extra:
        res = np.concatenate([res] + extra)
    if prior:
        extra = [(v[i] - mu) / sg for i, (mu, sg) in prior.items()]
        res = np.concatenate([res, np.array(extra)])
    return res


def fit(params: Params, sets: dict[str, FrameSet], radii=(30, 18, 10, 6), free_pp: bool = False,
        fit_joints: bool = True, fit_latency: bool = True, log=print, step: int = 3, prior_sigma: dict | None = None,
        teal_w: float = 0.0, fit_roll_scale: bool = True, fit_k1: bool = False, loss: str = "cauchy",
        twoview=None, tv_w: float = 0.0, fit_ext: bool = False, fit_cap: bool = True):
    rend = SegRenderer()
    fk = sr.FK(rend.m)
    history = []
    v = params.vec()
    nc = len(params.names) * Params.NC
    for it, R in enumerate(radii):
        t0 = time.time()
        A, stats = collect(params, sets, rend, R, fk, step=step)
        free = np.ones(len(v), bool)
        for k in range(len(params.names)):
            if not free_pp:
                free[k * Params.NC + 7:k * Params.NC + 9] = False
            if not fit_latency:
                free[k * Params.NC + 9] = False
            if not fit_k1:
                free[k * Params.NC + 10] = False
        if not fit_joints:
            free[nc:nc + 10] = False
        if not fit_roll_scale:
            free[nc + 8] = False
        if teal_w <= 0:
            free[nc + 10:] = False
        if not fit_ext:
            free[nc + 13] = False
        if not fit_cap:
            free[nc + 12] = False
        prior = {}
        if prior_sigma:
            for i, sg in prior_sigma.items():
                prior[i] = (v[i], sg)

        def fun(x):
            vv = v.copy()
            vv[free] = x
            r = residuals(vv, params, A, sets, fk, free_pp, teal_w=teal_w, twoview=twoview, tv_w=tv_w)
            if prior:
                r = np.concatenate([r, np.array([(vv[i] - mu) / sg for i, (mu, sg) in prior.items()])])
            return r

        r0 = fun(v[free])
        sol = least_squares(fun, v[free], loss=loss, f_scale=1.5, x_scale="jac", max_nfev=60, diff_step=1e-6)
        v[free] = sol.x
        params = params.set(v)
        if not free_pp:
            for c in params.cams.values():
                c.cx, c.cy = (sr.W - 1) / 2, (sr.H - 1) / 2
        nres = len(A["cam"])
        rec = {"iter": it, "radius": R, "n_corr": int(nres), "med_abs_before": float(np.median(np.abs(r0[:nres]))),
               "med_abs_after": float(np.median(np.abs(sol.fun[:nres]))), "nfev": int(sol.nfev),
               "sec": round(time.time() - t0, 1)}
        ntv = 4 * len(twoview) if (twoview and tv_w > 0) else 0
        if ntv:
            t1 = sol.fun[len(sol.fun) - ntv:] / tv_w
            t0_ = r0[len(r0) - ntv:] / tv_w
            rec["twoview_px_before"] = float(np.median(np.hypot(t0_[0::2], t0_[1::2])))
            rec["twoview_px_after"] = float(np.median(np.hypot(t1[0::2], t1[1::2])))
        if teal_w > 0 and len(sol.fun) - ntv > nres:
            tr = sol.fun[nres:len(sol.fun) - ntv] / teal_w
            tr0 = r0[nres:len(r0) - ntv] / teal_w
            rec["teal_px_before"] = float(np.median(np.hypot(tr0[0::2], tr0[1::2])))
            rec["teal_px_after"] = float(np.median(np.hypot(tr[0::2], tr[1::2])))
            rec["n_teal"] = int(len(tr) // 2)
        history.append(rec)
        log(json.dumps(rec))
    rend.close()
    return params, history


def score_outline(params: Params, sets: dict[str, FrameSet], radius: float = 10.0) -> dict:
    """Held-out outline residual: median |distance| (px) from rendered outline to the real one, per camera."""
    rend = SegRenderer()
    fk = sr.FK(rend.m)
    A, stats = collect(params, sets, rend, radius, fk)  # collect applies params.ext
    rend.close()
    out = {}
    for ci, cn in enumerate(sets):
        s = [x for x in stats if x["cam"] == cn and x["med_abs_px"] is not None]
        tot = sum(x["n"] for x in stats if x["cam"] == cn)
        mt = sum(x["matched"] for x in stats if x["cam"] == cn)
        out[cn] = {"frames": len(s), "median_of_frame_medians_px": float(np.median([x["med_abs_px"] for x in s])) if s else None,
                   "matched_fraction": mt / max(tot, 1)}
    return out


def overlay(img: np.ndarray, body: np.ndarray, jaw: int, color=(0, 255, 0)) -> np.ndarray:
    """Rendered outline (green: arm, magenta: moving jaw) drawn on the real frame."""
    out = img.copy()
    for mask, col in (((body >= 0) & (body != jaw), color), (body == jaw, (255, 0, 255))):
        m8 = mask.astype(np.uint8)
        cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, col, 1)
    return out


# ---------------------------------------------------------------------------------------------- keypoints
def triangulate(cams: list[sr.Camera], uvs: list[np.ndarray]) -> np.ndarray:
    """Linear (DLT) triangulation of one point from n views, then per-view residuals."""
    A = []
    for c, uv in zip(cams, uvs):
        K = np.array([[c.f, 0, c.cx], [0, c.f, c.cy], [0, 0, 1.0]])
        P = K @ np.hstack([c.R, c.t[:, None]])
        A.append(uv[0] * P[2] - P[0])
        A.append(uv[1] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


def load_calib(path: Path) -> tuple[dict[str, sr.Camera], sr.JointMap, dict[str, float]]:
    d = json.loads(Path(path).read_text())
    cams = {k: sr.Camera.from_dict(v) for k, v in d["cameras"].items()}
    jm = sr.JointMap.from_dict(d["joint_map"])
    lat = {k: float(v) for k, v in d["latency_frames"].items()}
    return cams, jm, lat


def brick_twoview(episodes: list[int]) -> list[dict]:
    """Brick blob centroids at frame 0 in both cameras (same physical point): two-view constraints for the fit."""
    from twin import so101_objects as so
    out = []
    for ep in episodes:
        pair = {}
        for c in sd.CAMS:
            img = sd.frame(ep, c, 0)
            b = so.brick_blob(img, dark_mask(img))
            if b is None:
                break
            ys, xs = np.nonzero(b)
            pair[c] = np.array([xs.mean(), ys.mean()])
        if len(pair) == len(sd.CAMS):
            out.append(pair)
    return out


def load_cap(path: Path) -> float:
    return float(json.loads(Path(path).read_text()).get("teal_cap_m", 0.0))


def load_ext(path: Path) -> float:
    return float(json.loads(Path(path).read_text()).get("gripper_ext_m", 0.0))


def load_fext(path: Path) -> float:
    return float(json.loads(Path(path).read_text()).get("finger_ext_m", 0.0))


# ---------------------------------------------------------------------------------------------- coarse seed
def coarse_align(cam: sr.Camera, jm: sr.JointMap, lat: float, fs: FrameSet, iters: int = 400, log=print) -> sr.Camera:
    """Seed refinement before the outline fit: maximise the mean IoU of the rendered arm (without the moving jaw)
    against the real dark mask over the frame set, at half resolution, over camera pose and focal length
    (joint map fixed). Nelder-Mead on [rotvec, t, log f]."""
    from scipy.optimize import minimize
    w, h = sr.W // 2, sr.H // 2
    rend = SegRenderer(w, h)
    real = [cv2.resize((m >= 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for m in fs.masks]
    ig = cv2.resize(fs.ignore.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    qs = [jm.q(interp_units(fs.units[ep], np.array([i + lat])))[0] for ep, i in fs.items]
    x0 = np.concatenate([Rotation.from_matrix(cam.R).as_rotvec(), cam.t, [np.log(cam.f)]])

    def cam_of(x):
        return sr.Camera(Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6], float(np.exp(x[6])), cam.cx, cam.cy)

    def loss(x):
        c = cam_of(x)
        ious = []
        for q, r in zip(qs, real):
            b, _ = rend.render(q, c, scale=0.5, depth=False)
            a = (b >= 0) & (b != rend.jaw)
            jaw = b == rend.jaw
            valid = ~ig & ~jaw
            inter = (a & r & valid).sum()
            uni = ((a | r) & valid).sum()
            ious.append(inter / max(uni, 1))
        return -float(np.mean(ious))

    l0 = loss(x0)
    step = np.array([0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.05])
    sim = np.vstack([x0] + [x0 + np.eye(7)[k] * step[k] for k in range(7)])
    res = minimize(loss, x0, method="Nelder-Mead", options={"initial_simplex": sim, "maxfev": iters, "xatol": 1e-4, "fatol": 1e-5})
    rend.close()
    log(json.dumps({"coarse_iou_before": -l0, "coarse_iou_after": -res.fun, "nfev": int(res.nfev)}))
    return cam_of(res.x)


# ---------------------------------------------------------------------------------------------- jaw extension
def fixed_tip_extension(cams: dict, jm: sr.JointMap, lat: dict, items: list[tuple[int, int]], cam: str = "side",
                        ext: float = 0.0, search_px: float = 60.0) -> dict:
    """How far the real fixed (black) jaw reaches beyond the CAD tip, along the jaw: walk from the projected CAD
    tip along the projected jaw direction through the real dark mask and take the last dark pixel (5 parallel rays,
    1 px apart, median). Returns the per-frame extension in mm (px converted with the jaw's own projected scale)."""
    m = sr.load_model(ext=ext)
    fk = sr.FK(m)
    gb = m.body("gripper").id
    tip = sr._jaw_tips(m)["fixed_tip"]
    ax = sr.gripper_axis(m)
    back = tip - 0.02 * ax
    out = []
    for ep, i in items:
        img = sd.frame(ep, cam, i)
        mask = dark_mask(img).astype(np.float32)
        q = jm.q(interp_units(sd.signals(ep)["state"], np.array([i + lat[cam]])))
        X = fk.points(q, np.array([gb, gb]), np.stack([tip, back]))[0]
        uv, z = cams[cam].project(X)
        d = uv[0] - uv[1]
        L = np.linalg.norm(d)
        if L < 5:
            continue
        d /= L
        nrm = np.array([-d[1], d[0]])
        ss = np.arange(-10, search_px, 0.5)
        ends = []
        for off in (-2, -1, 0, 1, 2):
            P = uv[0][None] + ss[:, None] * d[None] + off * nrm[None]
            v = cv2.remap(mask, P[:, 0].astype(np.float32)[None], P[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
            dark = v >= 0.5
            if not dark[:20].any():
                continue
            k = int(np.argmax(dark))
            while k + 1 < len(dark) and dark[k + 1]:
                k += 1
            ends.append(ss[k])
        if len(ends) >= 3:
            s_end = float(np.median(ends))
            out.append({"ep": ep, "frame": i, "ext_px": s_end, "mm_per_px": 20.0 / L, "ext_mm": s_end * 20.0 / L})
    e = [o["ext_mm"] for o in out]
    return {"n": len(out), "median_mm": float(np.median(e)) if e else None,
            "iqr_mm": float(np.percentile(e, 75) - np.percentile(e, 25)) if e else None, "rows": out}
