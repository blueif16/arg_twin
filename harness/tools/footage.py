"""Stage A tools: read REASSEMBLE footage and signals (skill: playbook/skills/read-footage).

All times are seconds from the start of the episode, on the dataset's 30 fps grid (every camera
and sensor is resampled to the hand-camera clock, so one time means one frame on every stream).
Outputs go under runs/footage/ep<NNN>/ and are returned as paths.

Data roots: data/reassemble/ first; an episode whose parquet or video file is not there is read from the
gate copies (data/reassemble-gate/ parquet, data/reassemble-gate-video/ videos, same HF layout). The gate
copies never serve exam episodes (playbook/splits.json), and labels of gate episodes are withheld by
recording_info and cut_clip unless asked for explicitly. probe_video and query_signals never load the
label columns at all.

Tools:
  recording_info  episodes on disk; per episode its cameras and (unless blind) its segment labels
  probe_video     a window of one or more cameras -> labelled frame grid PNG (+ mp4 clip)
  cut_clip        a window as clean per-camera mp4s + posters, for people (the Bot18 page's explainer cards)
  query_signals   a window of F/T, gripper, EE motion, joints -> plot PNG + compact JSON summary
  score_labels    an agent's segment labels vs the dataset's (segment F1 at IoU, success accuracy)
"""

from __future__ import annotations

import json
import math
import re
import time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np

from harness.registry import REPO, tool

DATA = REPO / "data" / "reassemble"
GATE_DATA = REPO / "data" / "reassemble-gate"  # HF parquet files holding gate (and more train) episodes
GATE_VIDEO = REPO / "data" / "reassemble-gate-video"  # HF video files holding gate episodes (blind readers)
PB_SPLITS = REPO / "playbook" / "splits.json"
SIGNAL_COLS = ["timestamp", "observation.state", "observation.force.base", "observation.torque.base",
               "observation.torque"]
LABEL_COLS = ["timestamp", "segment.index", "segment.success", "task_index"]
OUT = REPO / "runs" / "footage"
FPS = 30
CAMS = ("hand", "hama1", "hama2")
IDLE = "No action."
TASK_RE = re.compile(r"^(pick|insert|remove|place) (.+)$")
GRID_MAX_PX = 1568  # long side an image reader (Claude) sees without downscaling


# ---------------------------------------------------------------------------------------- data
@lru_cache(maxsize=1)
def _meta():
    import pandas as pd
    tasks = pd.read_parquet(DATA / "meta/tasks.parquet").reset_index()
    tasks = dict(zip(tasks["task_index"].astype(int), tasks["task"]))
    eps = pd.read_parquet(DATA / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
    splits = json.loads((DATA / "meta/splits.json").read_text())
    return tasks, eps, splits


@lru_cache(maxsize=1)
def _pb_split() -> dict:
    """episode -> playbook split (train / gate / exam); empty if playbook/splits.json is missing."""
    if not PB_SPLITS.exists():
        return {}
    return {int(k): v["split"] for k, v in json.loads(PB_SPLITS.read_text())["episodes"].items()}


def _local(rel: str, gate_root: Path, ep: int) -> Path:
    """data/reassemble/<rel> if present, else the gate copy. Exam episodes are refused wherever they sit (they share
    the local data file with train episodes), except for the exam-scorer caller (twin/exam.py)."""
    from twin import exam
    exam.check(ep)
    p = DATA / rel
    if p.exists() or not (gate_root / rel).exists():
        return p
    return gate_root / rel


def _data_path(ep: int) -> Path:
    _, eps, _ = _meta()
    if ep not in eps.index:
        raise ValueError(f"episode {ep} not in dataset meta")
    r = eps.loc[ep]
    return _local(f"data/chunk-{int(r['data/chunk_index']):03d}/file-{int(r['data/file_index']):03d}.parquet",
                  GATE_DATA, ep)


def _video_src(ep: int, cam: str):
    """(path, from_timestamp) of a camera's video for an episode, or None if not on disk."""
    _, eps, _ = _meta()
    r = eps.loc[ep]
    k = f"videos/observation.images.{cam}"
    p = _local(f"videos/observation.images.{cam}/chunk-{int(r[k + '/chunk_index']):03d}/file-{int(r[k + '/file_index']):03d}.mp4",
               GATE_VIDEO, ep)
    return (p, float(r[k + "/from_timestamp"])) if p.exists() else None


def _read_ep(ep: int, cols: list[str]) -> dict:
    import pyarrow.parquet as pq
    p = _data_path(ep)
    if not p.exists():
        raise FileNotFoundError(f"episode {ep}: data file {p.relative_to(REPO)} not on disk")
    tb = pq.read_table(p, columns=cols, filters=[("episode_index", "=", ep)])
    tb = tb.sort_by("timestamp")
    out = {}
    for c in cols:
        col = tb.column(c).combine_chunks()
        if c.startswith("observation"):
            out[c] = np.asarray(col.flatten(), dtype=np.float64).reshape(len(col), -1)
        else:
            out[c] = np.asarray(col)
    out["t"] = out.pop("timestamp").astype(np.float64)
    return out


def _frames(ep: int) -> dict:
    """Per-frame signal arrays for one episode. No label columns: reading signals never touches labels."""
    from twin import exam
    exam.check(ep)  # before the cache, so a cached exam episode is never served to another caller
    return _frames_cached(ep)


@lru_cache(maxsize=4)
def _frames_cached(ep: int) -> dict:
    return _read_ep(ep, SIGNAL_COLS)


def _labels(ep: int) -> dict:
    """Per-frame dataset labels (segment index, success, task) for one episode."""
    from twin import exam
    exam.check(ep)
    return _labels_cached(ep)


@lru_cache(maxsize=4)
def _labels_cached(ep: int) -> dict:
    return _read_ep(ep, LABEL_COLS)


def _segments(ep: int) -> list[dict]:
    tasks, _, _ = _meta()
    f = _labels(ep)
    si = f["segment.index"]
    cut = np.flatnonzero(np.diff(si) != 0) + 1
    starts, ends = np.r_[0, cut], np.r_[cut, len(si)]
    t = f["t"]
    return [dict(index=int(si[a]), task=tasks[int(f["task_index"][a])],
                 start=round(float(t[a]), 3), end=round(float(t[b - 1]) + 1 / FPS, 3),
                 success=bool(f["segment.success"][a])) for a, b in zip(starts, ends)]


def _window(ep: int, t0: float, t1: float):
    f = _frames(ep)
    dur = float(f["t"][-1]) + 1 / FPS
    if not (0 <= t0 < t1 <= dur + 1e-6):
        raise ValueError(f"window [{t0}, {t1}] outside episode {ep} (0-{dur:.2f} s)")
    i0, i1 = int(math.floor(t0 * FPS + 1e-6)), min(int(math.ceil(t1 * FPS - 1e-6)), len(f["t"]) - 1)
    return f, i0, i1 + 1


def _derived(f: dict, sl: slice) -> dict:
    """Named per-frame series used by the plot, the tile labels and the summary."""
    from scipy.spatial.transform import Rotation
    st = f["observation.state"][sl]
    fb = f["observation.force.base"][sl]
    q = st[:, 23:30]  # x y z qw qx qy qz in the robot base frame
    R = Rotation.from_quat(q[:, 3:7], scalar_first=True).as_matrix()
    zax = R[:, :, 2]
    tilt = np.degrees(np.arccos(np.clip(-zax[:, 2], -1, 1)))  # tool axis vs straight down
    xax = R[:, :, 0]
    yaw = np.unwrap(np.arctan2(xax[:, 1], xax[:, 0]))
    return dict(
        t=f["t"][sl], F=fb, Fmag=np.linalg.norm(fb, axis=1),
        Tbase=f["observation.torque.base"][sl], Tsens=f["observation.torque"][sl],
        grip=(st[:, 21] + st[:, 22]) * 1000.0,
        pos=st[:, 23:26] * 1000.0, v=np.linalg.norm(st[:, 30:33], axis=1) * 1000.0,
        w=np.degrees(np.linalg.norm(st[:, 33:36], axis=1)),
        tilt=tilt, yaw=np.degrees(yaw - yaw[0]), joints=st[:, 0:7])


def _runs(mask: np.ndarray):
    """[a, b) index runs where mask is True."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _r(x, n=2):
    return None if x is None else round(float(x), n)


# ---------------------------------------------------------------------------------- recording_info
@tool(
    "recording_info",
    "List REASSEMBLE episodes on disk, or describe one episode: duration, split, cameras with local "
    "video, the task-label vocabulary and (unless labels=false) the dataset's segment labels "
    "(index, task, start/end s, success). Use labels=false when you are about to label an episode "
    "blind and will score it afterwards. Gate-split episodes (playbook/splits.json) hide their labels "
    "unless labels=true is passed explicitly (watchers only); exam episodes are never listed or served.",
    {"type": "object", "properties": {
        "episode": {"type": "integer", "description": "episode index; omit to list all local episodes"},
        "labels": {"type": "boolean", "description": "include dataset segment labels (default true, "
                                                       "except false for gate-split episodes)"}}},
)
def recording_info(episode: int | None = None, labels: bool | None = None) -> dict:
    tasks, eps, splits = _meta()
    vocab = sorted({m.group(1) for v in tasks.values() if (m := TASK_RE.match(v.lower().rstrip(".")))})
    parts = sorted({m.group(2) for v in tasks.values() if (m := TASK_RE.match(v.lower().rstrip(".")))})
    note = ("Times are s from episode start at 30 fps. Dataset segments are contiguous: each end is the "
            "next start (gaps carry the previous label). 'Remove' = pull out of the board, 'Place' = put "
            "down on the table. 'No action.' = idle.")

    def show(ep):  # explicit labels wins; by default gate labels stay hidden
        return labels if labels is not None else _pb_split().get(ep) != "gate"

    if episode is None:
        from twin import exam
        rows = []
        for ep in exam.visible(int(e) for e in eps.index):  # exam episodes only for exam-scorer
            try:
                if not _data_path(ep).exists():
                    continue
            except PermissionError:  # belt and braces: exam episodes are already filtered out
                continue
            row = dict(episode=ep, recording=splits[str(ep)]["recording"],
                       split=splits[str(ep)]["split"], playbook_split=_pb_split().get(ep),
                       duration_s=_r(eps.loc[ep, "length"] / FPS, 1),
                       cameras=[c for c in CAMS if _video_src(ep, c)])
            if show(ep):
                segs = _segments(ep)
                row.update(n_segments=len(segs), n_failed=sum(not s["success"] for s in segs))
            rows.append(row)
        return dict(episodes=rows, verbs=vocab, parts=parts, idle_label=IDLE, note=note)
    ep = int(episode)
    f = _frames(ep)
    out = dict(episode=ep, recording=splits[str(ep)]["recording"], split=splits[str(ep)]["split"],
               playbook_split=_pb_split().get(ep), duration_s=_r(f["t"][-1] + 1 / FPS), fps=FPS,
               cameras=[c for c in CAMS if _video_src(ep, c)], verbs=vocab, parts=parts,
               idle_label=IDLE, note=note,
               signals=["force", "torque", "torque_tool", "gripper", "speed", "pos", "orient", "joints"])
    if show(ep):
        out["segments"] = _segments(ep)
    else:
        out["segments"] = "hidden (labels=false)" if labels is False else "hidden (gate episode; pass labels=true explicitly)"
    return out


# ------------------------------------------------------------------------------------ probe_video
def _crop_box(crop, cam, w=640, h=480):
    c = crop.get(cam) if isinstance(crop, dict) else crop
    if not c:
        return None
    x0, y0, x1, y1 = [float(v) for v in c]
    if max(x0, y0, x1, y1) <= 1.0:  # normalised
        x0, x1, y0, y1 = x0 * w, x1 * w, y0 * h, y1 * h
    x0, x1 = sorted((int(round(max(0, x0))), int(round(min(w, x1)))))
    y0, y1 = sorted((int(round(max(0, y0))), int(round(min(h, y1)))))
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError(f"crop for {cam} too small: {c}")
    return x0, y0, x1, y1


def _decode(ep: int, cam: str, i0: int, i1: int):
    """Yield (frame_index, rgb ndarray) for frame indices i0..i1-1 of one camera."""
    import av
    src = _video_src(ep, cam)
    if src is None:
        raise FileNotFoundError(f"episode {ep}: no local video for camera {cam}")
    path, off = src
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        # pre-roll at most 0.5 s and never before this episode's first frame, so a file shared with other
        # episodes (exam ones included) is decoded only inside this episode when keyframes allow it
        c.seek(int(max(off + i0 / FPS - 0.5, off + 0.5 / FPS) / st.time_base), stream=st, backward=True)
        want, arr = i0, None
        for fr in c.decode(st):
            if fr.time is None:
                continue
            fi = int(round((fr.time - off) * FPS))
            if fi < want:
                continue
            arr = fr.to_ndarray(format="rgb24")
            while want <= fi and want < i1:  # repeat a frame if the stream skipped one
                yield want, arr
                want += 1
            if want >= i1:
                return
        if arr is None:
            raise RuntimeError(f"episode {ep}: no {cam} frames decoded for frames {i0}-{i1}")
        while want < i1:  # ran out of stream: hold the last frame
            yield want, arr
            want += 1


def _label(img, text, color=(255, 255, 255), bar=(0, 0, 0), h=18):
    import cv2
    out = np.zeros((img.shape[0] + h, img.shape[1], 3), np.uint8)
    out[:h] = bar
    out[h:] = img
    cv2.putText(out, text, (4, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def _even(x):
    return int(x) // 2 * 2


@tool(
    "probe_video",
    "Look at a time window of an episode. Writes (a) a labelled frame grid PNG, one row strip per "
    "sampled time with the chosen cameras side by side, each tile titled 'cam t |F| gripper' with a "
    "red frame when |F| > 5 N (for image readers), and (b) an mp4 clip of the full-rate window with "
    "burned-in time, optionally slowed down (for video readers such as Gemini). Use crop to zoom.",
    {"type": "object", "required": ["episode", "t0", "t1"], "properties": {
        "episode": {"type": "integer"},
        "t0": {"type": "number", "description": "window start, s from episode start"},
        "t1": {"type": "number", "description": "window end, s"},
        "cameras": {"type": "array", "items": {"type": "string", "enum": list(CAMS)},
                    "description": "default ['hand']"},
        "fps": {"type": "number", "description": "grid sampling rate, frames per s (default 1)"},
        "max_frames": {"type": "integer", "description": "cap on grid time steps (default 16); fps is lowered to fit"},
        "crop": {"description": "zoom box [x0,y0,x1,y1] in 0-1 (or pixels of 640x480), one for all "
                 "cameras, or an object {cam: box}", "anyOf": [
                     {"type": "array", "items": {"type": "number"}}, {"type": "object"}]},
        "tile_width": {"type": "integer", "description": "grid tile width px (default 320)"},
        "clip": {"type": "boolean", "description": "also write the mp4 clip (default true)"},
        "slowdown": {"type": "number", "description": "clip playback slow-down factor (default 1)"},
        "times": {"type": "array", "items": {"type": "number"},
                  "description": "exact grid sample times (s) inside [t0, t1] instead of fps, e.g. one per grasp; "
                                 "each is decoded by a seek, so long windows stay cheap. No clip is written."}}},
)
def probe_video(episode: int, t0: float, t1: float, cameras: list | None = None, fps: float = 1.0,
                max_frames: int = 16, crop=None, tile_width: int = 320, clip: bool = True,
                slowdown: float = 1.0, times: list | None = None) -> dict:
    import cv2
    from PIL import Image
    T = time.time()
    ep = int(episode)
    cams = list(cameras or ["hand"])
    for c in cams:
        if c not in CAMS:
            raise ValueError(f"unknown camera {c}; choose from {CAMS}")
    f, i0, i1 = _window(ep, float(t0), float(t1))
    d = _derived(f, slice(i0, i1))
    n_steps = int(math.floor((t1 - t0) * fps + 1e-6)) + 1
    capped = n_steps > max_frames
    if capped:
        n_steps = max_frames
    if times:
        clip = False
        want = sorted({min(i1 - 1, max(i0, int(round(float(x) * FPS)))) for x in times})
        capped = len(want) > max_frames
        if capped:
            want = [want[int(round(j))] for j in np.linspace(0, len(want) - 1, max_frames)]
    elif capped:
        want = sorted({int(round(i)) for i in np.linspace(i0, i1 - 1, n_steps)})
    else:
        want = sorted({min(i1 - 1, i0 + int(round(k * FPS / fps))) for k in range(n_steps)})
    eff_fps = (len(want) - 1) / max((want[-1] - want[0]) / FPS, 1e-9) if len(want) > 1 else 0.0
    boxes = {c: _crop_box(crop, c) for c in cams}

    tag = f"{t0:07.2f}-{t1:07.2f}_{'-'.join(cams)}" + ("_crop" if any(boxes.values()) else "")
    od = OUT / f"ep{ep:03d}"
    od.mkdir(parents=True, exist_ok=True)
    if times:
        import hashlib
        tag += f"_t{len(want)}-" + hashlib.sha1(",".join(map(str, want)).encode()).hexdigest()[:6]
    grid_path = od / (f"grid_{tag}.png" if times else f"grid_{tag}_{fps:g}fps.png")
    clip_path = od / f"clip_{tag}_x{slowdown:g}.mp4" if clip else None

    def prep(img, c, width):
        b = boxes[c]
        if b:
            img = img[b[1]:b[3], b[0]:b[2]]
        hh = max(2, _even(img.shape[0] * width / img.shape[1]))
        return cv2.resize(img, (width, hh), interpolation=cv2.INTER_AREA if img.shape[1] > width else cv2.INTER_CUBIC)

    tiles = {i: {} for i in want}
    enc = container = None
    if times:  # sparse: one seek per sample time
        frame_iter = (tuple(next(_decode(ep, c, i, i + 1)) for c in cams) for i in want)
    else:
        frame_iter = zip(*[_decode(ep, c, i0, i1) for c in cams])
    for frames in frame_iter:
        fi = frames[0][0]
        k = fi - i0
        if fi in tiles:
            for c, (_, img) in zip(cams, frames):
                contact = d["Fmag"][k] > 5.0
                ti = prep(img, c, tile_width)
                if contact:
                    cv2.rectangle(ti, (0, 0), (ti.shape[1] - 1, ti.shape[0] - 1), (230, 30, 30), 3)
                txt = f"{c} {d['t'][k]:.2f}s |F|{d['Fmag'][k]:.1f}N g{d['grip'][k]:.0f}mm"
                tiles[fi][c] = _label(ti, txt, bar=(110, 0, 0) if contact else (0, 0, 0))
        if clip:
            parts = []
            for c, (_, img) in zip(cams, frames):
                im = img
                b = boxes[c]
                if b:
                    im = im[b[1]:b[3], b[0]:b[2]]
                    im = cv2.resize(im, (_even(480 * im.shape[1] / im.shape[0]), 480), interpolation=cv2.INTER_CUBIC)
                parts.append(_label(np.ascontiguousarray(im), f"{c}  t={d['t'][k]:.2f}s  |F|={d['Fmag'][k]:.1f}N", h=22))
            row = np.concatenate(parts, axis=1)
            if enc is None:
                import av
                container = av.open(str(clip_path), mode="w")
                enc = container.add_stream("libx264", rate=Fraction(FPS) / Fraction(slowdown).limit_denominator(100))
                enc.width, enc.height = _even(row.shape[1]), _even(row.shape[0])
                enc.pix_fmt = "yuv420p"
                enc.options = {"crf": "23", "preset": "veryfast"}
            import av
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(row[:enc.height, :enc.width]), format="rgb24")
            for pkt in enc.encode(vf):
                container.mux(pkt)
    if enc is not None:
        for pkt in enc.encode():
            container.mux(pkt)
        container.close()

    th = max(im.shape[0] for ts in tiles.values() for im in ts.values()) if any(tiles.values()) else 0

    def pad(im):  # crops with different aspect ratios give different tile heights
        return im if im.shape[0] == th else np.pad(im, ((0, th - im.shape[0]), (0, 0), (0, 0)), constant_values=255)

    strips = [np.concatenate([pad(tiles[i][c]) for c in cams], axis=1) for i in want if len(tiles[i]) == len(cams)]
    if not strips:
        raise RuntimeError("no frames decoded")
    sh, sw = strips[0].shape[:2]
    n = len(strips)
    best = max(range(1, n + 1), key=lambda k: (min(1.0, GRID_MAX_PX / max(k * sw, math.ceil(n / k) * sh)),
                                               -max(k * sw, math.ceil(n / k) * sh)))
    cols, rows = best, math.ceil(n / best)
    grid = np.full((rows * sh, cols * sw, 3), 255, np.uint8)
    for j, s in enumerate(strips):
        r_, c_ = divmod(j, cols)
        grid[r_ * sh:(r_ + 1) * sh, c_ * sw:(c_ + 1) * sw] = s
    Image.fromarray(grid).save(grid_path)
    scale = min(1.0, GRID_MAX_PX / max(grid.shape[:2]))
    return dict(
        grid=str(grid_path.relative_to(REPO)), clip=str(clip_path.relative_to(REPO)) if clip else None,
        clip_seconds=_r((i1 - i0) / FPS * slowdown, 1) if clip else None,
        times=[_r(i / FPS) for i in want], effective_fps=_r(eff_fps), capped=capped,
        cameras=cams, crop={c: b for c, b in boxes.items() if b} or None,
        grid_px=[grid.shape[1], grid.shape[0]], tile_px_as_seen=int(tile_width * scale),
        contact_frames=[_r(d["t"][i - i0]) for i in want if d["Fmag"][i - i0] > 5.0],
        seconds=_r(time.time() - T))



# ---------------------------------------------------------------------------------- cut_clip
EXPLAIN = REPO / "runs" / "explain" / "reassemble"


@tool(
    "cut_clip",
    "Cut a window of one episode into clean clips for people to watch: one H.264 mp4 per camera with no "
    "overlays, plus a poster JPG from the middle of the window. Use it for explainer cards on the Bot18 page "
    "(list the files under an `explain` block in runs/status/story.yaml, see docs/design/ui-protocol.md). "
    "To read footage yourself, use probe_video.",
    {"type": "object", "required": ["episode", "t0", "t1"], "properties": {
        "episode": {"type": "integer"},
        "t0": {"type": "number", "description": "window start, s from episode start"},
        "t1": {"type": "number", "description": "window end, s (at most 60 s after t0)"},
        "cameras": {"type": "array", "items": {"type": "string", "enum": list(CAMS)},
                    "description": "default ['hama1', 'hama2', 'hand']"},
        "name": {"type": "string", "description": "file stem, default ep<NNN>_<t0>-<t1>"},
        "width": {"type": "integer", "description": "output width px (default 640, the source width)"}}},
)
def cut_clip(episode: int, t0: float, t1: float, cameras: list | None = None, name: str | None = None,
             width: int = 640) -> dict:
    import av
    import cv2
    from PIL import Image
    T = time.time()
    ep, t0, t1 = int(episode), float(t0), float(t1)
    cams = list(cameras or ["hama1", "hama2", "hand"])
    for c in cams:
        if c not in CAMS:
            raise ValueError(f"unknown camera {c}; choose from {CAMS}")
    if t1 - t0 > 60:
        raise ValueError("keep a clip under 60 s; cut several instead")
    _, i0, i1 = _window(ep, t0, t1)
    stem = re.sub(r"[^\w.-]", "_", name or f"ep{ep:03d}_{t0:06.1f}-{t1:06.1f}")
    EXPLAIN.mkdir(parents=True, exist_ok=True)
    mid, clips = (i0 + i1) // 2, {}
    for c in cams:
        path, poster = EXPLAIN / f"{stem}_{c}.mp4", EXPLAIN / f"{stem}_{c}.jpg"
        container = av.open(str(path), mode="w", options={"movflags": "+faststart"})
        st = None
        for fi, img in _decode(ep, c, i0, i1):
            if img.shape[1] != width:
                img = cv2.resize(img, (_even(width), _even(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
            if st is None:
                st = container.add_stream("libx264", rate=FPS)
                st.width, st.height, st.pix_fmt = _even(img.shape[1]), _even(img.shape[0]), "yuv420p"
                st.options = {"crf": "23", "preset": "medium"}
            if fi == mid:
                Image.fromarray(img).save(poster, quality=85)
            for pkt in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(img[:st.height, :st.width]), format="rgb24")):
                container.mux(pkt)
        for pkt in st.encode():
            container.mux(pkt)
        container.close()
        clips[c] = {"video": str(path.relative_to(REPO)), "poster": str(poster.relative_to(REPO))}
    segs = ([s for s in _segments(ep) if s["end"] > t0 and s["start"] < t1] if _pb_split().get(ep) != "gate"
            else "hidden (gate episode)")
    return {"clips": clips, "episode": ep, "window": [t0, t1], "segments": segs, "seconds": _r(time.time() - T, 1)}

# ---------------------------------------------------------------------------------- query_signals
def _contacts(d, thr, gap=0.3, min_dur=0.1):
    on = d["Fmag"] > thr
    off_thr = 0.6 * thr
    runs = []
    for a, b in _runs(on):  # hysteresis: extend while above the lower threshold
        while a > 0 and d["Fmag"][a - 1] > off_thr:
            a -= 1
        while b < len(on) and d["Fmag"][b] > off_thr:
            b += 1
        if runs and (a - runs[-1][1]) / FPS < gap:
            runs[-1] = (runs[-1][0], max(b, runs[-1][1]))
        elif not runs or a >= runs[-1][1]:
            runs.append((a, b))
    ev = []
    for a, b in runs:
        if (b - a) / FPS < min_dur:
            continue
        s = slice(a, b)
        F = d["F"][s]
        k = a + int(np.argmax(d["Fmag"][s]))
        mF = F.mean(0)
        lat = d["pos"][s, :2]
        lv = np.diff(lat, axis=0)
        # lateral reversals: sign changes of the dominant lateral velocity component when moving
        rev = 0
        if len(lv) > 3:
            ax = int(np.argmax(np.abs(lv).sum(0)))
            sg = np.sign(lv[:, ax][np.abs(lv[:, ax]) > 0.05])
            rev = int(np.sum(sg[1:] != sg[:-1])) if len(sg) > 1 else 0
        ev.append(dict(
            t_on=_r(d["t"][a]), t_off=_r(d["t"][b - 1] + 1 / FPS), dur=_r((b - a) / FPS),
            peak_N=_r(d["Fmag"][k], 1), peak_t=_r(d["t"][k]),
            mean_F=[_r(x, 1) for x in mF], peak_F=[_r(x, 1) for x in d["F"][k]],
            dz_mm=_r(d["pos"][b - 1, 2] - d["pos"][a, 2], 1),
            lateral_path_mm=_r(np.linalg.norm(lv, axis=1).sum(), 1),
            yaw_change_deg=_r(d["yaw"][b - 1] - d["yaw"][a], 1),
            yaw_range_deg=_r(np.ptp(d["yaw"][s]), 1), tilt_range_deg=_r(np.ptp(d["tilt"][s]), 1),
            lateral_reversals=rev, grip_mm=_r(np.median(d["grip"][s]), 1)))
    return ev


def _grip_events(d, min_change=5.0, max_dur=1.5):
    g = d["grip"]
    ev = []
    i, n, w = 0, len(g), int(max_dur * FPS)
    while i < n - 1:
        j = min(n - 1, i + w)
        seg = g[i:j + 1]
        k_hi, k_lo = int(np.argmax(seg)), int(np.argmin(seg))
        if abs(seg[k_hi] - seg[k_lo]) >= min_change:
            a, b = sorted((k_hi, k_lo))
            # extend b while the change continues in the same direction
            sgn = np.sign(seg[b] - seg[a])
            bb = i + b
            while bb + 1 < n and (g[bb + 1] - g[bb]) * sgn > 0.05:
                bb += 1
            aa = i + a
            ev.append(dict(t=_r(d["t"][aa]), t_end=_r(d["t"][bb]), kind="close" if sgn < 0 else "open",
                           from_mm=_r(g[aa], 1), to_mm=_r(g[bb], 1)))
            i = bb + 1
        else:
            i += max(1, w // 2)
    merged = []  # one physical close/open is often detected as two steps
    for e in ev:
        if merged and merged[-1]["kind"] == e["kind"] and e["t"] - merged[-1]["t_end"] <= 0.3:
            merged[-1].update(t_end=e["t_end"], to_mm=e["to_mm"])
        else:
            merged.append(e)
    return merged


def _pauses(d, v_thr=5.0, min_dur=0.7, gap=0.2):
    runs = []
    for a, b in _runs(d["v"] < v_thr):
        if runs and (a - runs[-1][1]) / FPS <= gap:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    out = []
    for a, b in runs:
        if (b - a) / FPS >= min_dur:
            out.append(dict(t0=_r(d["t"][a]), t1=_r(d["t"][b - 1] + 1 / FPS),
                            z_mm=_r(d["pos"][a, 2], 1), grip_mm=_r(d["grip"][a], 1),
                            F_N=_r(np.median(d["Fmag"][a:b]), 1)))
    return out


CHANNELS = {
    "force": "force.base x/y/z and |F| (N), external force on the tool in the robot base frame",
    "torque": "torque.base x/y/z (Nm); includes the r x F moment about the base origin",
    "torque_tool": "sensor-frame torque x/y/z (Nm), minus its median over low-force frames in the window",
    "gripper": "finger opening (mm, 80 open, ~2 closed on nothing)",
    "speed": "EE linear speed (mm/s) and angular speed (deg/s)",
    "pos": "EE x/y/z displacement from the window start (mm); absolute values in the summary",
    "orient": "tool tilt from straight down (deg) and yaw about vertical relative to window start (deg)",
    "joints": "7 joint positions (rad)",
}


@tool(
    "query_signals",
    "Read a time window of an episode's robot signals. Writes a stacked plot PNG (contact shaded "
    "where |F| > contact_n, gripper events dotted, optional marks dashed) and returns a compact "
    "summary: contact events (onset/offset, peak and mean force vector, dz, lateral path and "
    "reversals, yaw/tilt change), gripper close/open events, pauses, per-channel ranges, and an "
    "optional down-sampled series. Channels: " + "; ".join(f"{k}: {v}" for k, v in CHANNELS.items()),
    {"type": "object", "required": ["episode", "t0", "t1"], "properties": {
        "episode": {"type": "integer"},
        "t0": {"type": "number"}, "t1": {"type": "number"},
        "channels": {"type": "array", "items": {"type": "string", "enum": list(CHANNELS)},
                     "description": "default ['force','gripper','speed','pos','orient']"},
        "contact_n": {"type": "number", "description": "contact threshold on |force.base| N (default 5, the teleop's own)"},
        "marks": {"type": "array", "items": {"type": "number"}, "description": "times to draw as dashed lines (e.g. your proposed boundaries)"},
        "series_hz": {"type": "number", "description": "if > 0, also return t,|F|,Fz,grip,speed,z,yaw rows at this rate"}}},
)
def query_signals(episode: int, t0: float, t1: float, channels: list | None = None, contact_n: float = 5.0,
                  marks: list | None = None, series_hz: float = 0.0) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = int(episode)
    chans = list(channels or ["force", "gripper", "speed", "pos", "orient"])
    for c in chans:
        if c not in CHANNELS:
            raise ValueError(f"unknown channel {c}; choose from {list(CHANNELS)}")
    f, i0, i1 = _window(ep, float(t0), float(t1))
    d = _derived(f, slice(i0, i1))
    t = d["t"]
    low = d["Fmag"] < 1.0
    tt = d["Tsens"] - (np.median(d["Tsens"][low], 0) if low.any() else 0)
    contacts = _contacts(d, contact_n)
    grips = _grip_events(d)

    fig, axs = plt.subplots(len(chans), 1, figsize=(12, 1.7 * len(chans) + 0.6), sharex=True, squeeze=False)
    axs = axs[:, 0]
    xyz = ("x", "y", "z")
    for ax, c in zip(axs, chans):
        if c == "force":
            for j in range(3):
                ax.plot(t, d["F"][:, j], lw=0.9, label=f"F{xyz[j]}")
            ax.plot(t, d["Fmag"], "k", lw=1.2, label="|F|")
            ax.axhline(contact_n, color="r", lw=0.6, ls=":")
            ax.set_ylabel("N")
        elif c == "torque":
            for j in range(3):
                ax.plot(t, d["Tbase"][:, j], lw=0.9, label=f"T{xyz[j]} base")
            ax.set_ylabel("Nm")
        elif c == "torque_tool":
            for j in range(3):
                ax.plot(t, tt[:, j], lw=0.9, label=f"T{xyz[j]} sensor")
            ax.set_ylabel("Nm")
        elif c == "gripper":
            ax.plot(t, d["grip"], "g", lw=1.2, label="grip mm")
            ax.set_ylabel("mm")
        elif c == "speed":
            ax.plot(t, d["v"], "b", lw=1.0, label="EE mm/s")
            ax2 = ax.twinx()
            ax2.plot(t, d["w"], "m", lw=0.8, alpha=0.7, label="deg/s")
            ax2.set_ylabel("deg/s", color="m")
            ax.set_ylabel("mm/s")
        elif c == "pos":
            for j in range(3):
                ax.plot(t, d["pos"][:, j] - d["pos"][0, j], lw=1.0, label=f"d{xyz[j]}")
            ax.set_ylabel("mm")
        elif c == "orient":
            ax.plot(t, d["tilt"], "c", lw=1.0, label="tilt")
            ax.plot(t, d["yaw"], "orange", lw=1.0, label="yaw")
            ax.set_ylabel("deg")
        elif c == "joints":
            for j in range(7):
                ax.plot(t, d["joints"][:, j], lw=0.8, label=f"q{j + 1}")
            ax.set_ylabel("rad")
        for e in contacts:
            ax.axvspan(e["t_on"], e["t_off"], color="r", alpha=0.08, lw=0)
        for g in grips:
            ax.axvline(g["t"], color="g", ls=":", lw=0.8)
        for m in marks or []:
            ax.axvline(float(m), color="k", ls="--", lw=0.9)
        ax.legend(loc="upper left", fontsize=7, ncol=4, framealpha=0.6)
        ax.grid(alpha=0.25)
        ax.set_title(c, fontsize=8, loc="right", pad=2)
    axs[-1].set_xlabel("episode time (s)")
    span = t1 - t0
    step = next((s for s in (0.5, 1, 2, 5, 10, 20, 30, 60, 120) if span / s <= 24), 300)
    axs[-1].set_xticks(np.arange(math.ceil(t0 / step) * step, t1 + 1e-9, step))
    fig.suptitle(f"episode {ep}  {t0:.2f}-{t1:.2f} s  (shaded: |F| > {contact_n:g} N)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    od = OUT / f"ep{ep:03d}"
    od.mkdir(parents=True, exist_ok=True)
    path = od / f"signals_{t0:07.2f}-{t1:07.2f}_{'-'.join(chans)}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)

    def rng(x, name):
        k1, k2 = int(np.argmin(x)), int(np.argmax(x))
        return {name: dict(min=_r(x[k1], 1), t_min=_r(t[k1]), max=_r(x[k2], 1), t_max=_r(t[k2]), mean=_r(x.mean(), 1))}

    ranges = {}
    for x, name in ((d["Fmag"], "|F|_N"), (d["F"][:, 2], "Fz_N"), (d["grip"], "grip_mm"), (d["v"], "speed_mm_s"),
                    (d["pos"][:, 2], "z_mm"), (d["tilt"], "tilt_deg"), (d["yaw"], "yaw_deg")):
        ranges.update(rng(x, name))
    out = dict(
        plot=str(path.relative_to(REPO)), window=[_r(t0), _r(t1)], frames=int(i1 - i0), contact_n=contact_n,
        start_pos_mm=[_r(x, 1) for x in d["pos"][0]], end_pos_mm=[_r(x, 1) for x in d["pos"][-1]],
        ranges=ranges, contact_share=_r(np.mean(d["Fmag"] > contact_n), 3),
        moving_share=_r(np.mean(d["v"] > 10), 3),
        contacts=contacts[:25], n_contacts=len(contacts), gripper_events=grips[:25],
        pauses=_pauses(d)[:25])
    if series_hz and series_hz > 0:
        stp = max(1, int(round(FPS / series_hz)))
        out["series_cols"] = ["t", "|F|", "Fz", "grip", "speed", "z", "yaw"]
        out["series"] = [[_r(t[i]), _r(d["Fmag"][i], 1), _r(d["F"][i, 2], 1), _r(d["grip"][i], 1),
                          _r(d["v"][i], 0), _r(d["pos"][i, 2], 1), _r(d["yaw"][i], 1)]
                         for i in range(0, len(t), stp)]
    return out


# ----------------------------------------------------------------------------------- score_labels
def _norm(task: str) -> str:
    return re.sub(r"\s+", " ", str(task).strip().lower().rstrip("."))


def _verb_part(task: str):
    m = TASK_RE.match(_norm(task))
    return (m.group(1), m.group(2)) if m else (_norm(task), None)


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


@tool(
    "score_labels",
    "Score an agent's segment labels for an episode against the dataset's. Pass labels inline "
    "(list of {start, end, task, success}) or labels_path to a read-footage JSON (its 'subtasks'). "
    "Ground truth = dataset segments with at least half their length inside the window(s) (default: the "
    "file's 'windows', else the span of your labels). One-to-one matching maximises IoU; a pair counts at IoU >= iou. Returns "
    "segment precision/recall/F1, F1 with the task also right, verb/part/task accuracy and success "
    "accuracy on matched pairs, failure recall, and boundary errors. Writes the score JSON under runs/footage/.",
    {"type": "object", "required": ["episode"], "properties": {
        "episode": {"type": "integer"},
        "labels": {"type": "array", "items": {"type": "object"}},
        "labels_path": {"type": "string", "description": "repo-relative path to a read-footage JSON"},
        "window": {"type": "array", "description": "[t0, t1] or [[t0, t1], ...] to score over",
                   "items": {"anyOf": [{"type": "number"}, {"type": "array", "items": {"type": "number"}}]}},
        "iou": {"type": "number", "description": "match threshold (default 0.5)"},
        "ignore_idle": {"type": "boolean", "description": "drop 'No action.' from both sides (default true)"}}},
)
def score_labels(episode: int, labels: list | None = None, labels_path: str | None = None,
                 window: list | None = None, iou: float = 0.5, ignore_idle: bool = True) -> dict:
    from scipy.optimize import linear_sum_assignment
    ep = int(episode)
    src = "inline"
    doc_windows = None
    if labels is None:
        if not labels_path:
            raise ValueError("pass labels or labels_path")
        p = (REPO / labels_path) if not Path(labels_path).is_absolute() else Path(labels_path)
        doc = json.loads(p.read_text())
        labels = doc.get("subtasks", doc) if isinstance(doc, dict) else doc
        doc_windows = doc.get("windows") if isinstance(doc, dict) else None
        src = str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)
    pred = []
    for x in labels:
        s = float(x.get("start", x.get("t0")))
        e = float(x.get("end", x.get("t1")))
        pred.append(dict(start=s, end=e, task=str(x.get("task", "")), success=x.get("success")))
    if ignore_idle:
        pred = [p for p in pred if _norm(p["task"]) != _norm(IDLE)]
    if not pred:
        raise ValueError("no labels to score")
    pred.sort(key=lambda p: p["start"])
    window = window or doc_windows or [min(p["start"] for p in pred), max(p["end"] for p in pred)]
    wins = [window] if not isinstance(window[0], (list, tuple)) else window
    wins = [(float(a), float(b)) for a, b in wins]
    w0, w1 = min(a for a, _ in wins), max(b for _, b in wins)
    gt = []
    for g in _segments(ep):
        if ignore_idle and g["task"] == IDLE:
            continue
        inside = sum(max(0.0, min(g["end"], b) - max(g["start"], a)) for a, b in wins)
        if inside >= 0.5 * (g["end"] - g["start"]):
            gt.append(g)
    M = np.array([[_iou((p["start"], p["end"]), (g["start"], g["end"])) for g in gt] for p in pred]) if gt else np.zeros((len(pred), 0))
    pairs = []
    if M.size:
        ri, ci = linear_sum_assignment(-M)
        pairs = [(int(r), int(c)) for r, c in zip(ri, ci) if M[r, c] >= iou]
    n_m, n_p, n_g = len(pairs), len(pred), len(gt)
    prec = n_m / n_p if n_p else 0.0
    rec = n_m / n_g if n_g else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    rows, task_ok, verb_ok, part_ok, succ_ok, succ_n = [], 0, 0, 0, 0, 0
    ds, de, ious = [], [], []
    for r, c in pairs:
        p, g = pred[r], gt[c]
        pv, pp = _verb_part(p["task"])
        gv, gp = _verb_part(g["task"])
        tk = _norm(p["task"]) == _norm(g["task"])
        task_ok += tk
        verb_ok += pv == gv
        part_ok += pp == gp
        if p["success"] is not None:
            succ_n += 1
            succ_ok += bool(p["success"]) == g["success"]
        ds.append(p["start"] - g["start"])
        de.append(p["end"] - g["end"])
        ious.append(M[r, c])
        rows.append(dict(gt_index=g["index"], gt_task=g["task"], pred_task=p["task"], iou=_r(M[r, c], 3),
                         start_err=_r(p["start"] - g["start"]), end_err=_r(p["end"] - g["end"]),
                         gt_success=g["success"], pred_success=p["success"], task_ok=tk))
    f1_task = 0.0
    if n_p and n_g and task_ok:
        pt, rt = task_ok / n_p, task_ok / n_g
        f1_task = 2 * pt * rt / (pt + rt)
    gt_fail = [c for c, g in enumerate(gt) if not g["success"]]
    matched_c = {c: r for r, c in pairs}
    fail_hit = sum(1 for c in gt_fail if c in matched_c and pred[matched_c[c]]["success"] is False)
    false_alarm = sum(1 for r, c in pairs if pred[r]["success"] is False and gt[c]["success"])
    res = dict(
        episode=ep, labels=src, windows=[[_r(a), _r(b)] for a, b in wins], iou_threshold=iou, ignore_idle=ignore_idle,
        n_pred=n_p, n_gt=n_g, matched=n_m, precision=_r(prec, 3), recall=_r(rec, 3), f1=_r(f1, 3),
        f1_task=_r(f1_task, 3), mean_iou_matched=_r(np.mean(ious), 3) if ious else None,
        task_acc=_r(task_ok / n_m, 3) if n_m else None, verb_acc=_r(verb_ok / n_m, 3) if n_m else None,
        part_acc=_r(part_ok / n_m, 3) if n_m else None,
        success_acc=_r(succ_ok / succ_n, 3) if succ_n else None,
        gt_failures=len(gt_fail), failures_detected=fail_hit, false_failure_alarms=false_alarm,
        start_err_s=dict(mean_abs=_r(np.mean(np.abs(ds))), median=_r(np.median(ds))) if ds else None,
        end_err_s=dict(mean_abs=_r(np.mean(np.abs(de))), median=_r(np.median(de))) if de else None,
        pairs=rows,
        unmatched_gt=[dict(index=gt[c]["index"], task=gt[c]["task"], start=gt[c]["start"], end=gt[c]["end"],
                           success=gt[c]["success"]) for c in range(n_g) if c not in matched_c],
        unmatched_pred=[dict(task=pred[r]["task"], start=pred[r]["start"], end=pred[r]["end"])
                        for r in range(n_p) if r not in {p for p, _ in pairs}],
        note="Dataset segment ends are the next segment's start (gaps carry the previous label), so "
             "end errors partly reflect that convention.")
    od = OUT / "scores"
    od.mkdir(parents=True, exist_ok=True)
    path = od / f"ep{ep:03d}_{w0:07.2f}-{w1:07.2f}_{len(wins)}w_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(res, indent=1))
    res["score_file"] = str(path.relative_to(REPO))
    return res
