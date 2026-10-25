"""SO-101 track: read lerobot/svla_so101_pickplace (LeRobot v3.0 layout) as data.

The dataset lives in data/so101/svla_so101_pickplace (downloaded with huggingface_hub; Apache-2.0). Everything
here only reads parquet and decodes video; nothing from the download is executed.

Layout facts (checked 2026-10-25, meta/info.json and meta/episodes):
  * one parquet for all 50 episodes (11,939 frames at 30 fps); per-episode rows dataset_from_index..to_index
  * one AV1 mp4 per camera (`up`, `side`) holding all episodes back to back; episode e frame i is global
    frame dataset_from_index[e] + i (from_timestamp = dataset_from_index / 30 exactly)
  * `observation.state` and `action` are 6 values per frame: shoulder_pan, shoulder_lift, elbow_flex,
    wrist_flex, wrist_roll in LeRobot's RANGE_M100_100 units and the gripper in RANGE_0_100 units
    (see twin/so101_robot.py for the unit evidence). `action` is the leader arm, read at the same tick.
OpenCV 4.13 here cannot decode AV1, so frames come from ffmpeg (libdav1d) through a pipe.
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "data" / "so101" / "svla_so101_pickplace"
SPLITS = REPO / "playbook" / "splits_so101.json"
CAMS = ("up", "side")
FPS = 30.0
W, H = 640, 480
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def video_path(cam: str) -> Path:
    if cam not in CAMS:
        raise ValueError(f"camera must be one of {CAMS}; got {cam!r}")
    return ROOT / "videos" / f"observation.images.{cam}" / "chunk-000" / "file-000.mp4"


@lru_cache(maxsize=1)
def episodes() -> list[dict]:
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
                      columns=["episode_index", "dataset_from_index", "dataset_to_index", "length", "tasks"])
    out = []
    for r in t.to_pylist():
        out.append({"episode": int(r["episode_index"]), "start": int(r["dataset_from_index"]),
                    "stop": int(r["dataset_to_index"]), "length": int(r["length"]), "task": list(r["tasks"])})
    return out


@lru_cache(maxsize=1)
def _table():
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data" / "chunk-000" / "file-000.parquet").to_pandas()
    return (np.stack(t["observation.state"].values).astype(np.float64),
            np.stack(t["action"].values).astype(np.float64),
            t["episode_index"].values.astype(int), t["frame_index"].values.astype(int),
            t["timestamp"].values.astype(np.float64))


def splits() -> dict:
    return json.loads(SPLITS.read_text())


def split_of(ep: int) -> str:
    s = splits()
    for k in ("train", "gate"):
        if int(ep) in s[k]:
            return k
    return "none"


def signals(ep: int) -> dict:
    """state, action (N x 6, dataset units), t (s from episode start), frame index."""
    st, ac, epi, fi, ts = _table()
    m = epi == int(ep)
    if not m.any():
        raise ValueError(f"no episode {ep}")
    return {"state": st[m], "action": ac[m], "t": ts[m], "frame": fi[m]}


def frames(ep: int, cam: str, idx=None) -> np.ndarray:
    """BGR frames (n, 480, 640, 3) of one episode; idx = list of frame indices (default: all)."""
    e = episodes()[int(ep)]
    n = e["length"]
    idx = list(range(n)) if idx is None else [int(i) for i in idx]
    if not idx:
        return np.zeros((0, H, W, 3), np.uint8)
    if min(idx) < 0 or max(idx) >= n:
        raise ValueError(f"frame index out of range 0..{n - 1}")
    lo, hi = min(idx), max(idx)
    t0 = (e["start"] + lo) / FPS
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.6f}", "-i", str(video_path(cam)), "-frames:v", str(hi - lo + 1),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    k = len(raw) // (W * H * 3)
    arr = np.frombuffer(raw[:k * W * H * 3], np.uint8).reshape(k, H, W, 3)
    if k < hi - lo + 1:
        raise RuntimeError(f"decoded {k} of {hi - lo + 1} frames for episode {ep} {cam}")
    return arr[[i - lo for i in idx]]


def frame(ep: int, cam: str, i: int) -> np.ndarray:
    return frames(ep, cam, [i])[0]
