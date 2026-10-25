"""Shared helpers: paths, JSON io, content hashing, the tool cache, and geometry.

Conventions used everywhere in recon/:
- Cameras are OpenCV: x right, y down, z forward. Poses are 4x4 T_world_from_cam (camera-to-world).
- Quaternions are wxyz.
- Lengths are metres unless a key says otherwise (_mm, _px, _pct).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "recon"
CACHE = RUNS / "cache"
VENV_RECON_PY = REPO / ".venv-recon" / "bin" / "python"
VENV_MAIN_PY = REPO / ".venv" / "bin" / "python"


# ----------------------------------------------------------------------------- io

def _default(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def dumps(x: Any, indent: int | None = 1) -> str:
    return json.dumps(x, indent=indent, default=_default)


def write_json(path: str | Path, x: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(dumps(x))
    os.replace(tmp, p)
    return p


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def rel(p: str | Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def absp(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


# ----------------------------------------------------------------------------- hashing + cache

def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_dir(path: str | Path, pattern: str = "*") -> str:
    """Hash of a directory listing + file hashes (used for frame folders)."""
    h = hashlib.sha256()
    for f in sorted(Path(path).glob(pattern)):
        if f.is_file():
            h.update(f.name.encode())
            h.update(sha256_file(f).encode())
    return h.hexdigest()


def key_of(tool: str, version: str, args: dict, input_hashes: dict | None = None) -> str:
    blob = json.dumps({"tool": tool, "v": version, "args": args, "inputs": input_hashes or {}},
                      sort_keys=True, default=_default)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def cached(tool: str, version: str, args: dict, fn: Callable[[Path], dict],
           input_hashes: dict | None = None, force: bool = False) -> dict:
    """Run fn(out_dir) once per (tool, version, args, input content); return its JSON result.

    Deterministic by construction: a repeat call with the same inputs returns the stored result.
    The result dict gets `cache_key`, `cache_dir` and `cached` (bool) added.
    """
    k = key_of(tool, version, args, input_hashes)
    d = CACHE / tool / k
    out = d / "out.json"
    if out.exists() and not force:
        r = read_json(out)
        r["cached"] = True
        return r
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = fn(d)
    r = dict(r)
    r.update(cache_key=k, cache_dir=rel(d), tool=tool, tool_version=version,
             wall_s=round(time.time() - t0, 2))
    write_json(d / "args.json", {"args": args, "inputs": input_hashes or {}})
    write_json(out, r)
    r["cached"] = False
    return r


# ----------------------------------------------------------------------------- geometry

def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def T_from(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def T_from_pose_dict(d: dict) -> np.ndarray:
    return T_from(quat_wxyz_to_R(d["quaternion_wxyz"]), d["translation_xyz"])


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return T_from(R.T, -R.T @ t)


def rot_angle_deg(R: np.ndarray) -> float:
    c = (np.trace(R) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def project_to_SO3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Find s, R, t minimising |dst - (s R src + t)|^2. src, dst: (N, 3)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var_s) if with_scale and var_s > 1e-12 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def align_poses(pred: np.ndarray, gt: np.ndarray, mode: str = "sim3"):
    """Align predicted camera poses (N,4,4 c2w) to GT poses.

    Rotation from the chordal mean of R_gt R_pred^T over all frames (well posed even when the camera
    centres are nearly collinear, as in short head-mounted clips); then scale and translation from
    the centres in closed form. mode: 'sim3' (fit scale), 'se3' (scale fixed to 1).
    Returns s, R, t with  gt ~= s R pred + t.
    """
    Rs = np.einsum("nij,nkj->ik", gt[:, :3, :3], pred[:, :3, :3])
    R = project_to_SO3(Rs)
    cp = pred[:, :3, 3] @ R.T
    cg = gt[:, :3, 3]
    mp, mg = cp.mean(0), cg.mean(0)
    if mode == "sim3":
        num = ((cp - mp) * (cg - mg)).sum()
        den = ((cp - mp) ** 2).sum()
        s = float(num / den) if den > 1e-12 else 1.0
        s = max(s, 1e-6)
    else:
        s = 1.0
    t = mg - s * mp
    return s, R, t


def apply_sim3_to_poses(poses: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.array(poses, dtype=float, copy=True)
    out[:, :3, :3] = np.einsum("ij,njk->nik", R, poses[:, :3, :3])
    out[:, :3, 3] = s * poses[:, :3, 3] @ R.T + t
    return out


def backproject(depth: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None, stride: int = 1):
    """depth (H,W) metres -> points (N,3) in the camera frame (and their pixel coords)."""
    H, W = depth.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[::stride, ::stride]
    ok = np.isfinite(d) & (d > 0)
    if mask is not None:
        ok &= mask[::stride, ::stride].astype(bool)
    u, v, z = us[ok].astype(float), vs[ok].astype(float), d[ok].astype(float)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], 1), np.stack([u, v], 1)


def project(P_cam: np.ndarray, K: np.ndarray):
    P = np.atleast_2d(P_cam)
    z = P[:, 2]
    u = K[0, 0] * P[:, 0] / z + K[0, 2]
    v = K[1, 1] * P[:, 1] / z + K[1, 2]
    return np.stack([u, v], 1), z


def K_from(fx, fy, cx, cy) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


# ----------------------------------------------------------------------------- masks

def rle_encode(mask: np.ndarray) -> dict:
    """Row-major (start, length) runs, the same layout HOT3D uses."""
    m = np.asarray(mask, bool).reshape(-1)
    d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    runs = np.stack([starts, ends - starts], 1).reshape(-1)
    return {"height": int(mask.shape[0]), "width": int(mask.shape[1]), "rle": runs.tolist()}


def rle_decode(r: dict) -> np.ndarray:
    H, W = r["height"], r["width"]
    m = np.zeros(H * W, bool)
    a = np.asarray(r["rle"], dtype=np.int64).reshape(-1, 2)
    for s, n in a:
        m[s:s + n] = True
    return m.reshape(H, W)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def robust_stats(x) -> dict:
    x = np.asarray([v for v in np.ravel(x) if v is not None and np.isfinite(v)], float)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)), "mean": float(x.mean()),
            "p90": float(np.percentile(x, 90)), "max": float(x.max())}
