"""Intake for any video: a file path, a folder of frames, or a URL (yt-dlp), plus optional extra streams.

Writes intake.json: what the source offers and how far to trust it, in the form reconstruct-scene's
intake uses (present / absent / unknown, where it lives, units, rate, trust). Supplied values enter as
priors to verify, never as truth.

Downloads land in a new empty folder under data/recon-downloads/<slug>/ and are treated as untrusted
data: nothing there is executed; only ffprobe/OpenCV read it.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .common import REPO, absp, read_json, rel, sha256_file, write_json

DOWNLOADS = REPO / "data" / "recon-downloads"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png"}

EXTRA_KINDS = {
    "calibration": "camera intrinsics (and lens model) as JSON: {model: pinhole|fisheye624|opencv, params, width, height, rot90_cw}",
    "depth": "per-frame depth maps (folder of .npy/.png in metres or mm)",
    "camera_poses": "per-frame camera poses (JSON list of 4x4 T_world_from_cam)",
    "robot_log": "robot joint/EE log (parquet/csv/json)",
    "imu": "IMU log",
    "audio": "separate audio track",
    "labels": "action or object labels (never used for fitting; scoring only)",
}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80].strip("_") or "video"


def download(url: str, max_height: int = 720) -> dict:
    """yt-dlp a public video into its own new folder. Returns path and metadata (incl. licence)."""
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{6,})", url)
    slug = _slug(m.group(1) if m else url.split("/")[-1])
    d = DOWNLOADS / slug
    if d.exists() and any(d.glob("video.*")):
        meta = read_json(d / "meta.json") if (d / "meta.json").exists() else {}
        return {"path": str(next(d.glob("video.*"))), "meta": meta, "folder": rel(d)}
    d.mkdir(parents=True, exist_ok=True)
    fmt = f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/b[height<={max_height}][ext=mp4]/b[height<={max_height}]"
    cmd = ["yt-dlp", "--no-playlist", "-f", fmt, "--merge-output-format", "mp4",
           "--write-info-json", "-o", str(d / "video.%(ext)s"), url]
    subprocess.run(cmd, check=True, cwd=str(d), capture_output=True, text=True, timeout=1800)
    info_p = next(d.glob("*.info.json"), None)
    meta = {}
    if info_p:
        info = json.loads(info_p.read_text())
        meta = {k: info.get(k) for k in ("id", "title", "uploader", "channel", "license", "duration",
                                          "webpage_url", "upload_date", "width", "height", "fps")}
    write_json(d / "meta.json", meta)
    vid = next(p for p in d.glob("video.*") if p.suffix in VIDEO_EXT)
    return {"path": str(vid), "meta": meta, "folder": rel(d)}


def ffprobe(path: str | Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
                        str(path)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def _fps(s: str | None) -> float | None:
    if not s or s == "0/0":
        return None
    a, b = s.split("/") if "/" in s else (s, "1")
    return float(a) / float(b) if float(b) else None


def intake(source: str, out_dir: str, extra: dict | None = None, task: str = "",
           frames_fps: float | None = None) -> dict:
    """Write out_dir/intake.json describing the source and every extra stream.

    source: video file, folder of frames (then frames_fps is required), or http(s) URL.
    extra: {kind: path} for kinds in EXTRA_KINDS; anything absent is recorded as absent.
    """
    out = absp(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    extra = dict(extra or {})
    rec: dict = {"schema": "recon.intake/v0", "source": source, "task": task}
    origin = "file"
    meta = {}
    if re.match(r"^https?://", source):
        dl = download(source)
        path = Path(dl["path"])
        meta = dl["meta"]
        origin = "url"
        rec["download_folder"] = dl["folder"]
    else:
        path = absp(source)
        if path.is_dir():
            origin = "frames"
    rec["origin"] = origin
    rec["meta"] = meta

    if origin == "frames":
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXT)
        assert files, f"no images in {path}"
        assert frames_fps, "frames_fps is required for a folder of frames"
        import cv2
        im = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
        video = {"path": rel(path), "kind": "frames", "n_frames": len(files), "fps": float(frames_fps),
                 "duration_s": len(files) / float(frames_fps), "width": int(im.shape[1]),
                 "height": int(im.shape[0]), "channels": 1 if im.ndim == 2 else int(im.shape[2]),
                 "sha256": sha256_file(files[0])[:16] + f"..x{len(files)}"}
        audio = {"status": "absent"}
    else:
        pr = ffprobe(path)
        vs = [s for s in pr["streams"] if s.get("codec_type") == "video"]
        aus = [s for s in pr["streams"] if s.get("codec_type") == "audio"]
        assert vs, "no video stream"
        v = vs[0]
        rot = 0
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = int(sd["rotation"])
        dur = float(pr["format"].get("duration") or v.get("duration") or 0)
        fps = _fps(v.get("avg_frame_rate")) or _fps(v.get("r_frame_rate"))
        video = {"path": rel(path), "kind": "video", "codec": v.get("codec_name"), "fps": fps,
                 "duration_s": dur, "n_frames": int(v.get("nb_frames") or round(dur * (fps or 0))),
                 "width": int(v["width"]), "height": int(v["height"]), "rotation_meta_deg": rot,
                 "size_mb": round(path.stat().st_size / 1e6, 2), "sha256": sha256_file(path)}
        audio = {"status": "present" if aus else "absent",
                 **({"codec": aus[0].get("codec_name"), "rate": aus[0].get("sample_rate")} if aus else {})}
    rec["video"] = video

    streams = {"video": {"status": "present", "where": video["path"], "rate_hz": video["fps"],
                         "trust": "the observation itself"},
               "audio": audio}
    for kind, desc in EXTRA_KINDS.items():
        if kind in extra and extra[kind]:
            p = absp(extra[kind])
            st = {"status": "present", "where": rel(p), "what": desc,
                  "trust": "prior to verify (supplied values were wrong more than once on REASSEMBLE)"}
            if p.is_file() and p.suffix == ".json":
                try:
                    st["summary"] = {k: v for k, v in read_json(p).items() if k != "params"}
                except Exception:
                    pass
            streams[kind] = st
        else:
            streams[kind] = {"status": "absent"}
    rec["streams"] = streams

    # What the pipeline will have to estimate, given what is absent.
    need = []
    if streams["calibration"]["status"] == "absent":
        need += ["intrinsics (focal, distortion) from video",
                 "lens check: strong distortion or zoom breaks camera recovery (MegaSaM failure modes)"]
    if streams["camera_poses"]["status"] == "absent":
        need.append("camera trajectory from video (MegaSaM; DA3 second opinion)")
    if streams["depth"]["status"] == "absent":
        need.append("metric scale: no sensor; needs >=2 independent sources (MoGe-2 metric depth, known-size anchor, hand size)")
    need.append("objects, surfaces, hands, contacts from video only")
    need.append("mass, friction, inertia: category priors with wide ranges (not observable)")
    rec["must_estimate"] = need
    rec["honest_limits"] = [
        "metric scale without an anchor: 9-53% error on HOT3D (docs/knowledge/video-to-twin.md)",
        "mass from a model prior: typically x1.3 off, up to x4",
        "contact from egocentric RGB: F1 <= 0.27",
        "unseen geometry (backs, interiors, transparent parts) is guessed",
    ]
    write_json(out / "intake.json", rec)
    rec["intake_path"] = rel(out / "intake.json")
    return rec
