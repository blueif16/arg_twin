"""SO-101 track: the objects without CAD — the LEGO brick and the transparent box (stage T2, reconstruct-scene 4a/4b).

Brick. Identified by eye as a LEGO 2x4 brick (part 3001: 4 x 2 studs, one brick high). Standard LEGO dimensions give
metric scale: pitch 8.0 mm, play 0.1 mm per side -> 31.8 x 15.8 mm footprint, body height 9.6 mm, studs 4.8 mm
diameter x 1.7 mm. Pose from both calibrated cameras: the brick's projected outline (convex hull of body + stud
block corners) is fitted to the red/pink colour mask in both views at once (IoU, Nelder-Mead over x, y, z, yaw).

Box. The transparent box is a hard case for vision: its walls show no colour and little edge. We do not segment the
walls. The box stands on an orange tape outline, which is easy to see in both views, so the footprint comes from the
tape (a rectangle on the table plane fitted to the orange mask in both views); the rim height comes from a 1-D search
for the strongest image edge along the projected rim in both views, and is checked by eye on an overlay.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from twin import so101_data as sd
from twin import so101_robot as sr

# LEGO 3001 (2x4 brick), mm -> m
BRICK = {"length": 0.0318, "width": 0.0158, "height": 0.0096, "stud_d": 0.0048, "stud_h": 0.0017, "pitch": 0.008,
         "mass": 0.0023, "source": "LEGO system dimensions: 8.0 mm pitch, 0.1 mm play per side, 9.6 mm brick height, "
                                    "4.8 mm x 1.7 mm studs; 3001 mass about 2.3 g"}


def red_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    m = ((h <= 8) | (h >= 168)) & (s >= 80) & (v >= 90)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def orange_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    m = (h >= 10) & (h <= 28) & (s >= 70) & (v >= 90)
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def brick_blob(img: np.ndarray, robot_dark: np.ndarray | None = None, min_px: int = 30):
    """Largest red component not touching the arm (cables are red too). Returns a uint8 mask or None."""
    m = red_mask(img)
    if robot_dark is not None:
        near = cv2.dilate(robot_dark.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    best, area = None, 0
    for k in range(1, n):
        a = stats[k, cv2.CC_STAT_AREA]
        if a < min_px:
            continue
        blob = lab == k
        if robot_dark is not None and (blob & near).sum() > 0.5 * a:
            continue
        if a > area:
            best, area = blob, a
    return None if best is None else best.astype(np.uint8)


def brick_corners(x, y, z, yaw) -> np.ndarray:
    """16 corners (body box + stud block) of a 2x4 brick whose body centre is (x, y, z), long axis at yaw."""
    L, Wd, Hh = BRICK["length"], BRICK["width"], BRICK["height"]
    sl = L - 2 * (BRICK["pitch"] / 2 - BRICK["stud_d"] / 2) - 0.0001
    sw = Wd - 2 * (BRICK["pitch"] / 2 - BRICK["stud_d"] / 2) - 0.0001
    pts = []
    for (a, b, z0, z1) in ((L, Wd, -Hh / 2, Hh / 2), (sl, sw, Hh / 2, Hh / 2 + BRICK["stud_h"])):
        for sx in (-1, 1):
            for sy in (-1, 1):
                for zz in (z0, z1):
                    pts.append([sx * a / 2, sy * b / 2, zz])
    pts = np.array(pts)
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return pts @ Rz.T + np.array([x, y, z])


def hull_mask(cam: sr.Camera, X: np.ndarray, shape=(sr.H, sr.W)) -> np.ndarray:
    uv, z = cam.project(X)
    m = np.zeros(shape, np.uint8)
    if (z <= 0).any():
        return m
    hull = cv2.convexHull(np.round(uv * 4).astype(np.int32))
    cv2.fillConvexPoly(m, hull, 1, lineType=cv2.LINE_8, shift=2)
    return m


def _ray_point(cams: dict, blobs: dict) -> np.ndarray:
    """Closest point between the rays through the blob centroids (two views)."""
    cs, ds = [], []
    for k, b in blobs.items():
        ys, xs = np.nonzero(b)
        c, d = cams[k].ray(np.array([[xs.mean(), ys.mean()]]))
        cs.append(c)
        ds.append(d[0])
    A = np.zeros((3, 3))
    bb = np.zeros(3)
    for c, d in zip(cs, ds):
        P = np.eye(3) - np.outer(d, d)
        A += P
        bb += P @ c
    return np.linalg.solve(A, bb)


def fit_brick(cams: dict, blobs: dict, table_z: float | None = None, x0=None) -> dict:
    """Fit (x, y, z, yaw) of the brick body centre to the colour blobs of every camera that sees it.
    With table_z, z is fixed to table_z + height/2 (brick lying flat on the table)."""
    blobs = {k: v for k, v in blobs.items() if v is not None}
    if not blobs:
        return {"ok": False, "why": "no brick blob"}
    if x0 is None:
        p = _ray_point(cams, blobs) if len(blobs) > 1 else None
        if p is None and table_z is not None:
            # one view: intersect the blob-centroid ray with the plane z = table + half a brick
            (k, b), = blobs.items()
            ys, xs = np.nonzero(b)
            c, d = cams[k].ray(np.array([[xs.mean(), ys.mean()]]))
            zt = table_z + BRICK["height"] / 2
            p = c + d[0] * (zt - c[2]) / d[0][2]
        if p is None:
            return {"ok": False, "why": "one view only and no start"}
        # yaw seed from the up-view blob's principal axis projected onto the table (try both quadrants)
        x0 = [p[0], p[1], p[2] if table_z is None else table_z + BRICK["height"] / 2, 0.0]
    best = None
    for yaw0 in (x0[3], x0[3] + np.pi / 4, x0[3] + np.pi / 2, x0[3] - np.pi / 4):
        def loss(v):
            x, y, yaw = v[0], v[1], v[-1]
            z = table_z + BRICK["height"] / 2 if table_z is not None else v[2]
            X = brick_corners(x, y, z, yaw)
            ious = []
            for k, b in blobs.items():
                hm = hull_mask(cams[k], X)
                inter = (hm & b).sum()
                uni = (hm | b).sum()
                ious.append(inter / max(uni, 1))
            return -float(np.mean(ious))
        v0 = [x0[0], x0[1], yaw0] if table_z is not None else [x0[0], x0[1], x0[2], yaw0]
        step = [0.004, 0.004, 0.3] if table_z is not None else [0.004, 0.004, 0.004, 0.3]
        sim = np.vstack([v0] + [np.array(v0) + np.eye(len(v0))[i] * step[i] for i in range(len(v0))])
        r = minimize(loss, v0, method="Nelder-Mead", options={"initial_simplex": sim, "maxfev": 600, "xatol": 1e-5, "fatol": 1e-6})
        if best is None or r.fun < best.fun:
            best = r
    v = best.x
    z = table_z + BRICK["height"] / 2 if table_z is not None else v[2]
    yaw = float((v[-1] + np.pi / 2) % np.pi - np.pi / 2)  # the brick is symmetric under 180 deg
    return {"ok": True, "pos": [float(v[0]), float(v[1]), float(z)], "yaw": yaw, "iou": -float(best.fun),
            "views": sorted(blobs), "blob_px": {k: int(b.sum()) for k, b in blobs.items()}}


# ---------------------------------------------------------------------------------------------- box
def rect_outline(cx, cy, yaw, lx, ly, z, n: int = 60) -> np.ndarray:
    """Points along a rectangle's outline on the plane z (centre, yaw, full side lengths)."""
    c, s = np.cos(yaw), np.sin(yaw)
    pts = []
    corners = [(-lx / 2, -ly / 2), (lx / 2, -ly / 2), (lx / 2, ly / 2), (-lx / 2, ly / 2)]
    for k in range(4):
        a, b = np.array(corners[k]), np.array(corners[(k + 1) % 4])
        for t in np.linspace(0, 1, n, endpoint=False):
            p = a + (b - a) * t
            pts.append([cx + c * p[0] - s * p[1], cy + s * p[0] + c * p[1], z])
    return np.array(pts)


def tape_band_mask(cam: sr.Camera, v, z, width: float) -> np.ndarray:
    """Image mask of a tape band of the given width centred on the rectangle outline."""
    cx, cy, yaw, lx, ly = v
    m = np.zeros((sr.H, sr.W), np.uint8)
    for sgn in (-1, 1):
        pass
    outer = rect_outline(cx, cy, yaw, lx + width, ly + width, z, 2)
    inner = rect_outline(cx, cy, yaw, lx - width, ly - width, z, 2)
    uo, zo = cam.project(outer[::2])
    ui, zi = cam.project(inner[::2])
    if (zo <= 0).any() or (zi <= 0).any():
        return m
    cv2.fillPoly(m, [np.round(uo * 4).astype(np.int32)], 1, shift=2)
    cv2.fillPoly(m, [np.round(ui * 4).astype(np.int32)], 0, shift=2)
    return m


def fit_tape(cams: dict, masks: dict, z: float | None, v0, width: float = 0.012) -> dict:
    """Rectangle (centre x, y, yaw, side lengths) of the tape outline's centreline on the table plane.
    One-sided chamfer: every orange pixel should lie on the projected outline (distance clipped at 15 px). Only part
    of the tape is visible in each view (the rest is behind the transparent walls or the robot), so an IoU against the
    full band would be biased; each side is seen in at least one view."""
    pts = {k: np.argwhere(m > 0)[:, ::-1].astype(np.float32) for k, m in masks.items()}

    free_z = z is None

    def loss(v):
        tot = []
        zz_ = v[5] if free_z else z
        for k, P in pts.items():
            img = np.full((sr.H, sr.W), 255, np.uint8)
            outl = rect_outline(v[0], v[1], v[2], v[3], v[4], zz_, 30)
            uv, zz = cams[k].project(outl)
            if (zz <= 0).any():
                return 1e3
            cv2.polylines(img, [np.round(uv * 4).astype(np.int32)], True, 0, 1, shift=2)
            dt = cv2.distanceTransform(img, cv2.DIST_L2, 3)
            d = cv2.remap(dt, P[:, 0][None], P[:, 1][None], cv2.INTER_LINEAR)[0]
            tot.append(np.minimum(d, 15.0).mean())
        return float(np.mean(tot))
    x0 = list(v0)[:5] + ([v0[5] if len(v0) > 5 else 0.0] if free_z else [])
    step = [0.01, 0.01, 0.05, 0.01, 0.01, 0.005][:len(x0)]
    n = len(x0)
    best = None
    for k in range(3):
        sim = np.vstack([x0] + [np.array(x0) + np.eye(n)[i] * step[i] for i in range(n)])
        r = minimize(loss, x0, method="Nelder-Mead", options={"initial_simplex": sim, "maxfev": 1200, "xatol": 1e-5, "fatol": 1e-5})
        x0 = list(r.x)
        best = r
    v = best.x
    zf = float(v[5]) if free_z else float(z)
    return {"center": [float(v[0]), float(v[1]), zf], "yaw": float(v[2]), "size": [float(v[3]), float(v[4])],
            "z_free": free_z, "tape_width": float(width), "chamfer_px": float(best.fun), "nfev": int(best.nfev),
            "orange_px": {k: int(len(P)) for k, P in pts.items()}}


def rim_height_profile(cams: dict, imgs: dict, footprint: dict, inset: float, heights: np.ndarray) -> dict:
    """Mean image-gradient magnitude along the projected rim (box top outline) for each candidate height."""
    out = {}
    for k, img in imgs.items():
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        prof = []
        for h in heights:
            P = rect_outline(footprint["center"][0], footprint["center"][1], footprint["yaw"],
                             footprint["size"][0] - inset, footprint["size"][1] - inset, footprint["center"][2] + h, 80)
            uv, z = cams[k].project(P)
            ok = (uv[:, 0] > 2) & (uv[:, 0] < sr.W - 3) & (uv[:, 1] > 2) & (uv[:, 1] < sr.H - 3) & (z > 0)
            vals = cv2.remap(mag, uv[ok, 0].astype(np.float32)[None], uv[ok, 1].astype(np.float32)[None], cv2.INTER_LINEAR)
            prof.append(float(vals.mean()) if ok.any() else 0.0)
        out[k] = prof
    return out


# ---------------------------------------------------------------------------------------------- drivers
def brick_pose_at(cams: dict, ep: int, frame: int, table_z: float | None) -> dict:
    """Brick pose at one frame from both cameras (colour blobs not touching the dark arm)."""
    from twin import so101_camcal as cc
    blobs = {}
    for c in sd.CAMS:
        img = sd.frame(ep, c, frame)
        blobs[c] = brick_blob(img, cc.dark_mask(img))
    r = fit_brick(cams, blobs, table_z)
    r.update({"ep": int(ep), "frame": int(frame)})
    return r


def brick_starts(cams: dict, eps: list[int], table_z: float) -> dict:
    out = {}
    for ep in eps:
        r = brick_pose_at(cams, ep, 0, table_z)
        if r.get("ok"):
            out[str(ep)] = {"pos": r["pos"], "yaw": r["yaw"], "iou": r["iou"], "views": r["views"]}
    return out


def static_background(cam: str, eps: list[int]) -> np.ndarray:
    """Per-pixel median of mid-task frames (the arm moves away, the tape and box stay)."""
    imgs = []
    for ep in eps:
        n = sd.episodes()[ep]["length"]
        imgs.append(sd.frames(ep, cam, [n // 4, n // 2, 3 * n // 4]))
    return np.median(np.concatenate(imgs), axis=0).astype(np.uint8)


def fit_static_objects(cams: dict, box_seed=None, rim_inset: float | None = None, box_height: float | None = None,
                       eps: list[int] | None = None) -> dict:
    tr = eps or sd.splits()["train"]
    # 1. table height from free-z brick fits at frame 0 (brick lying flat on the table)
    zs = []
    for ep in tr:
        r = brick_pose_at(cams, ep, 0, None)
        if r.get("ok") and len(r["views"]) == 2 and r["iou"] > 0.6:
            zs.append(r["pos"][2] - BRICK["height"] / 2)
    brick_table_z = float(np.median(zs)) if zs else None
    spread = float(np.percentile(zs, 75) - np.percentile(zs, 25)) * 1000 if zs else None
    # 2. tape rectangle, both views, z free: the tape lies on the table, so its z is the table height
    bgs = {c: static_background(c, tr[::2]) for c in sd.CAMS}
    oms = {c: orange_mask(bgs[c]) for c in sd.CAMS}
    table_z = brick_table_z
    seed = list(box_seed or [0.0, -0.2, 0.0, 0.15, 0.10])
    tape = fit_tape(cams, oms, table_z, seed)
    tape_free_z = fit_tape(cams, oms, None, seed + [table_z])["center"][2]
    # 3. box: outer footprint = inner edge of the tape outline; rim height from image edges
    w = tape["tape_width"]
    outer = [tape["size"][0] - w, tape["size"][1] - w]
    foot = {"center": tape["center"], "yaw": tape["yaw"], "size": outer}
    heights = np.arange(0.01, 0.12, 0.002)
    prof = rim_height_profile(cams, bgs, foot, 0.0 if rim_inset is None else rim_inset, heights)
    tot = np.sum([np.array(v) / max(np.max(v), 1e-6) for v in prof.values()], axis=0)
    h_edge = float(heights[int(np.argmax(tot))])
    h = float(box_height) if box_height is not None else h_edge
    return {"table_z": table_z, "table_z_from": "median of free-z two-view brick fits at frame 0 (train), minus half the brick height",
            "tape_free_z_check": tape_free_z,
            "brick_table_z": brick_table_z, "brick_table_z_n": len(zs), "table_z_spread_mm": spread, "tape": tape,
            "box": {"center": tape["center"], "yaw": tape["yaw"], "outer": outer, "height": h,
                    "height_from": "override" if box_height is not None else "rim edge search",
                    "rim_edge_search_m": h_edge},
            "rim_profile": {"heights_m": heights.round(3).tolist(), **{k: np.round(v, 2).tolist() for k, v in prof.items()}},
            "brick_model": BRICK}


def table_plane_from_bricks(cams: dict, eps: list[int], min_iou: float = 0.6) -> dict:
    """Least-squares plane z = a x + b y + c through the free-z two-view brick fits at frame 0 (minus half a brick)."""
    P = []
    for ep in eps:
        b = brick_pose_at(cams, ep, 0, None)
        if b.get("ok") and b["iou"] > min_iou and len(b["views"]) == 2:
            P.append([b["pos"][0], b["pos"][1], b["pos"][2] - BRICK["height"] / 2])
    P = np.array(P)
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    coef = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
    r = P[:, 2] - A @ coef
    return {"plane": coef.tolist(), "n": len(P), "resid_sd_mm": float(r.std() * 1000), "flat_sd_mm": float(P[:, 2].std() * 1000)}


def plane_z(plane, x: float, y: float) -> float:
    a, b, c = plane
    return float(a * x + b * y + c)


def brick_starts_on_plane(cams: dict, eps: list[int], plane) -> dict:
    """Brick starts with z from the table plane at the brick: fit with z fixed to the flat-table value at the brick's
    first estimate, then re-fit once at the plane height there."""
    out = {}
    for ep in eps:
        r0 = brick_pose_at(cams, ep, 0, plane_z(plane, 0.2, 0.0))
        if not r0.get("ok"):
            continue
        z = plane_z(plane, r0["pos"][0], r0["pos"][1])
        r = brick_pose_at(cams, ep, 0, z)
        if r.get("ok"):
            out[str(ep)] = {"pos": r["pos"], "yaw": r["yaw"], "iou": r["iou"], "views": r["views"], "table_z_here": z}
    return out
