"""Comparison media for the status page and the demo: real vs sim, attempt 1 vs attempt N, and so on.

make_comparison  two inputs (a video, a still, or an image sequence given as a folder or a glob) -> one
                 browser-playable H.264 mp4, or a PNG when both inputs are stills and the layout is static.
                 Layouts: side_by_side, overlay (50/50 blend), wipe (a divider sweeping across).
ensure_playable  (helper, not a tool) re-encode a video that browsers can't play from file:// (e.g. OpenCV's
                 mp4v, or 4:4:4 H.264) into a sibling H.264 yuv420p mp4. publish_media calls it.

Outputs default to runs/compare/ and must stay under runs/. Publish one with publish_media(kind="compare"), on its
data track's step: stage="B2" for REASSEMBLE, stage="T2" (with step="B0" or "B2") or track="so101" for SO-101.
"""

from __future__ import annotations

import glob as globmod
import json
import re
import shutil
import subprocess
from pathlib import Path

from harness.registry import REPO, tool

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
OUT_DIR = REPO / "runs" / "compare"
LAYOUTS = ("side_by_side", "overlay", "wipe")
MAX_H = 720


def _path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO / q


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def _run(cmd: list[str], timeout: int = 600) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr.strip()[-600:]}")
    return r.stdout


def probe(path: Path) -> dict:
    out = json.loads(_run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=codec_name,pix_fmt,width,height,avg_frame_rate:format=duration,format_name",
                           "-of", "json", str(path)], timeout=60))
    s = (out.get("streams") or [{}])[0]
    num, _, den = (s.get("avg_frame_rate") or "0/1").partition("/")
    return {"codec": s.get("codec_name"), "pix_fmt": s.get("pix_fmt"), "w": s.get("width"), "h": s.get("height"),
            "fps": float(num) / float(den or 1) if float(den or 1) else 0.0,
            "duration": float(out.get("format", {}).get("duration") or 0), "format": out.get("format", {}).get("format_name", "")}


def is_playable(path: Path) -> bool:
    """True if Chrome/Firefox/Safari play it inline from file:// (H.264 4:2:0 in mp4/mov, or VP8/VP9/AV1 in webm)."""
    info = probe(path)
    if path.suffix.lower() == ".webm":
        return info["codec"] in ("vp8", "vp9", "av1")
    return (path.suffix.lower() in (".mp4", ".m4v", ".mov") and info["codec"] == "h264"
            and info["pix_fmt"] in ("yuv420p", "yuvj420p"))


def ensure_playable(path: Path) -> Path:
    """Return path if browsers can play it, else a re-encoded sibling <stem>.h264.mp4 (under runs/)."""
    if path.suffix.lower() not in VIDEO_EXT or is_playable(path):
        return path
    out = path.with_name(path.stem + ".h264.mp4")
    if not str(out.resolve()).startswith(str((REPO / "runs").resolve())):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / (path.stem + ".h264.mp4")
    if not out.exists() or out.stat().st_mtime < path.stat().st_mtime:
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
              "-an", str(out)])
    return out


def _source(spec: str, fps: float) -> dict:
    """Classify one input and return its ffmpeg input args, size and duration (None for a still)."""
    p = _path(spec)
    if p.is_dir() or any(ch in spec for ch in "*?["):
        pattern = str(p)
        if p.is_dir():
            exts = sorted({f.suffix.lower() for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT})
            if not exts:
                raise ValueError(f"{spec}: folder holds no images")
            pattern = str(p / f"*{exts[0]}")
        frames = sorted(globmod.glob(pattern))
        if not frames:
            raise ValueError(f"{spec}: no frames match")
        from PIL import Image
        with Image.open(frames[0]) as im:
            w, h = im.size
        return {"kind": "sequence", "args": ["-framerate", f"{fps:g}", "-pattern_type", "glob", "-i", pattern],
                "w": w, "h": h, "dur": len(frames) / fps, "n": len(frames)}
    if not p.is_file():
        raise FileNotFoundError(f"{spec}: no such file")
    ext = p.suffix.lower()
    if ext in IMAGE_EXT:
        from PIL import Image
        with Image.open(p) as im:
            w, h = im.size
        return {"kind": "still", "path": p, "w": w, "h": h, "dur": None}
    if ext in VIDEO_EXT:
        info = probe(p)
        return {"kind": "video", "args": ["-i", str(p)], "w": info["w"], "h": info["h"], "dur": info["duration"]}
    raise ValueError(f"{spec}: not a video, image or image folder")


def _label_filter(text: str, h: int, right: bool = False) -> str:
    text = re.sub(r"[^\w .,+()/-]", "", text or "")[:40].replace(",", r"\,")
    if not text:
        return ""
    size = max(14, min(32, h // 18))
    font = shutil.which("fc-match") and subprocess.run(["fc-match", "-f", "%{file}", "sans"], capture_output=True,
                                                         text=True).stdout.strip()
    x = f"w-tw-{size // 2}" if right else f"{size // 2}"
    return (f",drawtext=text='{text}'" + (f":fontfile='{font}'" if font else "") +
            f":fontsize={size}:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw={size // 3}:x={x}:y={size // 2}")


@tool(
    "make_comparison",
    "Build one browser-playable comparison from two inputs, e.g. real vs sim or attempt 1 vs attempt N. "
    "Each input is a video, a still image, or an image sequence (a folder or a glob like 'runs/x/frame_*.png'). "
    "Layouts: side_by_side, overlay (50/50 blend, shows misalignment), wipe (a divider sweeps across). "
    "Writes an H.264 mp4, or a PNG when both inputs are stills and the layout is not wipe. "
    "Show it on the status page with publish_media(kind='compare').",
    {"type": "object", "required": ["left", "right"], "properties": {
        "left": {"type": "string", "description": "repo-relative video, image, image folder or glob (e.g. the real footage)"},
        "right": {"type": "string", "description": "the same kinds (e.g. the sim render)"},
        "layout": {"type": "string", "enum": list(LAYOUTS), "description": "default side_by_side"},
        "labels": {"type": "array", "items": {"type": "string"}, "description": "two captions burned in, default ['real', 'sim']"},
        "out": {"type": "string", "description": "output path under runs/ (.mp4 or .png); default runs/compare/<left>-vs-<right>-<layout>"},
        "fps": {"type": "number", "description": "frame rate for image sequences and the output, default 30"},
        "max_seconds": {"type": "number", "description": "cap on output length, default 60"}}},
)
def make_comparison(left: str, right: str, layout: str = "side_by_side", labels: list[str] | None = None,
                    out: str | None = None, fps: float = 30, max_seconds: float = 60) -> dict:
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    fps = float(fps) if fps and 1 <= float(fps) <= 120 else 30.0
    labels = list(labels or ["real", "sim"]) + ["", ""]
    a, b = _source(left, fps), _source(right, fps)
    still_pair = a["kind"] == b["kind"] == "still"
    as_png = still_pair and layout != "wipe"
    durs = [s["dur"] for s in (a, b) if s["dur"]]
    dur = min(max(durs) if durs else 4.0, float(max_seconds or 60))

    stem = lambda s: re.sub(r"[^\w.-]", "", Path(s.rstrip("/*")).stem)[:40] or "input"
    target = _path(out) if out else OUT_DIR / f"{stem(left)}-vs-{stem(right)}-{layout}{'.png' if as_png else '.mp4'}"
    if not str(target.resolve()).startswith(str((REPO / "runs").resolve()) + "/"):
        raise ValueError("out must be under runs/ so sources are never overwritten")
    if target.suffix.lower() != (".png" if as_png else ".mp4"):
        target = target.with_suffix(".png" if as_png else ".mp4")
    target.parent.mkdir(parents=True, exist_ok=True)

    H = min(MAX_H, max(a["h"], b["h"])) // 2 * 2
    wa = int(round(a["w"] * H / a["h"] / 2)) * 2
    args = ["ffmpeg", "-y", "-v", "error"]
    for s in (a, b):
        if s["kind"] == "still":
            args += ["-loop", "1", "-framerate", f"{fps:g}", "-t", f"{dur:.3f}", "-i", str(s["path"])]
        else:
            args += s["args"]
    pad = "" if as_png else f",fps={fps:g},tpad=stop_mode=clone:stop_duration={dur:.3f}"
    if layout == "side_by_side":
        wb = int(round(b["w"] * H / b["h"] / 2)) * 2
        fa = f"[0:v]scale={wa}:{H},setsar=1{pad},pad={wa + 4}:{H}:0:0:black{_label_filter(labels[0], H)}[a]"
        fb = f"[1:v]scale={wb}:{H},setsar=1{pad}{_label_filter(labels[1], H)}[b]"
        graph = f"{fa};{fb};[a][b]hstack=inputs=2,format=yuv420p[v]"
        size = [wa + 4 + wb, H]
    else:
        fit = f"scale={wa}:{H}:force_original_aspect_ratio=decrease,pad={wa}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        fa = f"[0:v]scale={wa}:{H},setsar=1{pad},format=gbrp[a]"
        fb = f"[1:v]{fit}{pad},format=gbrp[b]"
        if layout == "overlay":
            mix = "blend=all_mode=normal:all_opacity=0.5"
            lab = _label_filter(f"{labels[0]} + {labels[1]} (50/50)".strip(" +"), H)
        else:
            period = max(2.0, min(dur, 6.0))
            pos = f"W*(0.5+0.45*sin(2*PI*T/{period:g}))"
            mix = f"blend=all_expr='if(lt(abs(X-{pos}),2),255,if(lt(X,{pos}),A,B))'"
            lab = _label_filter(labels[0], H) + _label_filter(labels[1], H, right=True)
        graph = f"{fa};{fb};[a][b]{mix},format=yuv420p{lab}[v]"
        size = [wa, H]
    args += ["-filter_complex", graph, "-map", "[v]"]
    if as_png:
        args += ["-frames:v", "1", str(target)]
    else:
        args += ["-t", f"{dur:.3f}", "-r", f"{fps:g}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(target)]
    _run(args)
    return {"out": _rel(target), "kind": "image" if as_png else "video", "layout": layout, "size": size,
            "duration_s": None if as_png else round(dur, 2), "inputs": [{"spec": left, "kind": a["kind"]},
                                                                      {"spec": right, "kind": b["kind"]}],
            "next": f"publish_media(stage=..., title=..., paths=['{_rel(target)}'], kind='compare', metrics={{...}}); "
                    "for another data track add its stage (T2) and step, or track=..."}
