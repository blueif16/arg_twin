"""Sample frames from the intake's video, and undistort them when a calibration stream is supplied.

Lens models: pinhole (nothing to do), opencv (k1,k2,p1,p2[,k3]), fisheye624 (Project Aria).
Undistortion goes to a centred pinhole of a chosen field of view, optionally rotated by k*90 deg
clockwise so the image is upright (Aria's RGB sensor is mounted sideways).
Without calibration the frames are only resized; intrinsics are then estimated downstream.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .common import absp, cached, read_json, rel, sha256_file, write_json

VERSION = "0.2"


def fisheye624_project(P: np.ndarray, params) -> np.ndarray:
    """Project camera-frame points (N,3) with Aria's FisheyeRadTanThinPrism (6 radial, 2 tangential,
    4 thin-prism; single focal). params = [f, cx, cy, k0..k5, p0, p1, s0..s3]."""
    p = np.asarray(params, float)
    f, cx, cy = p[0], p[1], p[2]
    k = p[3:9]
    tp = p[9:11]
    s = p[11:15]
    P = np.atleast_2d(P).astype(float)
    ab = P[:, :2] / P[:, 2:3]
    r = np.linalg.norm(ab, axis=1)
    th = np.arctan(r)
    th2 = th * th
    rad = np.ones_like(th)
    pw = th2.copy()
    for ki in k:
        rad += ki * pw
        pw = pw * th2
    th_divr = np.where(r < 1e-9, 1.0, th / np.maximum(r, 1e-12))
    xr = (rad * th_divr)[:, None] * ab
    r2 = (xr ** 2).sum(1)
    temp = 2.0 * (xr @ tp)
    uv = xr * (1.0 + temp)[:, None] + r2[:, None] * tp[None, :]
    r4 = r2 * r2
    uv[:, 0] += s[0] * r2 + s[1] * r4
    uv[:, 1] += s[2] * r2 + s[3] * r4
    return np.stack([f * uv[:, 0] + cx, f * uv[:, 1] + cy], 1)


def R_upcam_from_cam(rot90_cw: int) -> np.ndarray:
    """Rotation taking camera-frame vectors to the frame of the image rotated rot90_cw x 90 deg clockwise."""
    R1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])  # one clockwise turn: x' = -y, y' = x
    R = np.eye(3)
    for _ in range(rot90_cw % 4):
        R = R1 @ R
    return R


def build_undistort_maps(calib: dict, out_size: int, fov_deg: float, rot90_cw: int):
    """Maps from an upright, centred pinhole (out_size^2) to the source image, plus its K."""
    f_out = (out_size / 2) / math.tan(math.radians(fov_deg) / 2)
    c = (out_size - 1) / 2
    vs, us = np.mgrid[0:out_size, 0:out_size].astype(float)
    rays_up = np.stack([(us - c) / f_out, (vs - c) / f_out, np.ones_like(us)], -1).reshape(-1, 3)
    R = R_upcam_from_cam(rot90_cw)
    rays = rays_up @ R  # = (R^T ray_up^T)^T : back to the source camera frame
    model = calib["model"]
    if model == "fisheye624":
        uv = fisheye624_project(rays, calib["params"])
    elif model in ("pinhole", "opencv"):
        fx, fy, cx, cy = calib["params"][:4]
        dist = np.asarray(calib["params"][4:] or [0, 0, 0, 0], float)
        Kc = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        uv, _ = cv2.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Kc, dist)
        uv = uv.reshape(-1, 2)
    else:
        raise ValueError(f"unknown lens model {model}")
    mx = uv[:, 0].reshape(out_size, out_size).astype(np.float32)
    my = uv[:, 1].reshape(out_size, out_size).astype(np.float32)
    K = np.array([[f_out, 0, c], [0, f_out, c], [0, 0, 1.0]])
    return mx, my, K, R


def _read_source_frames(intake: dict):
    v = intake["video"]
    if v["kind"] == "frames":
        files = sorted(p for p in absp(v["path"]).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        for i, f in enumerate(files):
            yield i, i / v["fps"], cv2.imread(str(f), cv2.IMREAD_COLOR)
    else:
        cap = cv2.VideoCapture(str(absp(v["path"])))
        i = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            yield i, i / v["fps"], im
            i += 1
        cap.release()


def extract_frames(intake_path: str, out_dir: str, fps: float = 15.0, t0: float = 0.0, t1: float | None = None,
                   max_side: int = 640, undistort: bool = True, pinhole_size: int = 512,
                   pinhole_fov_deg: float = 90.0, rot90_cw: int | None = None) -> dict:
    """Sample frames at `fps` in [t0, t1) into out_dir/frames/*.jpg and write out_dir/frames.json.

    If the intake lists a calibration stream and undistort is true, frames are undistorted to an
    upright centred pinhole (pinhole_size^2, pinhole_fov_deg) and K is known (source: calibration).
    Otherwise frames are resized so the long side is max_side and K is null (to be estimated).
    """
    intake = read_json(absp(intake_path))
    calib_p = intake["streams"].get("calibration", {})
    calib = read_json(absp(calib_p["where"])) if calib_p.get("status") == "present" else None
    args = dict(fps=fps, t0=t0, t1=t1, max_side=max_side, undistort=undistort and calib is not None,
                pinhole_size=pinhole_size, pinhole_fov_deg=pinhole_fov_deg, rot90_cw=rot90_cw)
    hashes = {"video": intake["video"]["sha256"], "calib": sha256_file(absp(calib_p["where"])) if calib else None}
    out = absp(out_dir)

    def run(cache_dir: Path) -> dict:
        fdir = cache_dir / "frames"
        fdir.mkdir(parents=True, exist_ok=True)
        src_fps = intake["video"]["fps"]
        dur = intake["video"]["duration_s"]
        end = min(t1 if t1 is not None else dur, dur)
        want = np.arange(t0, end - 1e-9, 1.0 / fps)
        src_idx = sorted(set(int(round(t * src_fps)) for t in want))
        maps = None
        rot = rot90_cw if rot90_cw is not None else int((calib or {}).get("rot90_cw", 0))
        K = None
        R_up = np.eye(3)
        if args["undistort"]:
            mx, my, K, R_up = build_undistort_maps(calib, pinhole_size, pinhole_fov_deg, rot)
            maps = (mx, my)
        kept, ts = [], []
        size = None
        for i, t, im in _read_source_frames(intake):
            if i not in src_idx:
                continue
            if maps is not None:
                im = cv2.remap(im, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            else:
                for _ in range(rot % 4):
                    im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
                h, w = im.shape[:2]
                sc = max_side / max(h, w)
                if sc < 1:
                    im = cv2.resize(im, (int(round(w * sc)) // 2 * 2, int(round(h * sc)) // 2 * 2),
                                    interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(fdir / f"{len(kept):06d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            kept.append(i)
            ts.append(i / src_fps)
            size = im.shape[:2]
        rec = {"n": len(kept), "fps": fps, "timestamps_s": ts, "src_indices": kept,
               "height": int(size[0]), "width": int(size[1]),
               "K": K.tolist() if K is not None else None,
               "K_source": "calibration" if K is not None else None,
               "undistorted": maps is not None, "rot90_cw": rot,
               "R_upcam_from_srccam": R_up.tolist(),
               "frames_dir": rel(fdir)}
        return rec

    r = cached("extract_frames", VERSION, args, run, hashes)
    # mirror into the run folder
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "frames.json", r)
    return r


def load_frames(frames_json: dict | str, idx=None, gray: bool = False):
    fj = read_json(absp(frames_json)) if isinstance(frames_json, str) else frames_json
    d = absp(fj["frames_dir"])
    ids = range(fj["n"]) if idx is None else idx
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    return [cv2.imread(str(d / f"{i:06d}.jpg"), flag) for i in ids]


def frame_path(fj: dict, i: int) -> Path:
    return absp(fj["frames_dir"]) / f"{i:06d}.jpg"
