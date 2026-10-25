"""Plots and videos for replays (twin/replay.py): real vs sim force and height traces, real camera frames of
the same window, and frames with a force strip underneath so a comparison video shows motion and force.

Rendering needs MUJOCO_GL=glfw on this box (set by twin.world on import).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")


def trace_plot(res: dict, path: Path) -> Path:
    """Two panels: |dF| real vs sim (onsets, release), and TCP z real vs sim + part depth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = np.array(res["traces"]["rows"], float)
    t = rows[:, 0]
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(t, rows[:, 2], color="#444", lw=1.6, label="real |dF|")
    ax[0].plot(t, rows[:, 1], color="#d1495b", lw=1.4, label="sim |dF|")
    fo = res["force"]
    for key, c in (("onset_real_s", "#444"), ("onset_sim_s", "#d1495b")):
        if fo.get(key) is not None:
            ax[0].axvline(fo[key], color=c, ls=":", lw=1)
    rel = res["window"].get("release_s")
    for a in ax:
        if rel is not None:
            a.axvline(rel, color="#2e86ab", ls="--", lw=1)
    ax[0].set_ylabel("N")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title(f"ep {res['episode']} seg {res['segment_index']} {res['part']}  real "
                    f"{'success' if res['real_success'] else 'failure'}  |  sim {res['outcome']}  "
                    f"({res['mode']}, K={res['params']['k_trans']:.0f})", fontsize=10)
    ax[1].plot(t, rows[:, 6], color="#444", lw=1.6, label="real TCP z")
    ax[1].plot(t, rows[:, 5], color="#d1495b", lw=1.4, label="sim TCP z")
    ax[1].set_ylabel("mm")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[2].plot(t, rows[:, 7], color="#2e86ab", lw=1.4, label="part depth in hole (sim)")
    ax[2].plot(t, rows[:, 8], color="#f18f01", lw=1.2, label="lateral from hole axis (sim)")
    ax[2].axhline(res["final"]["seat_depth_mm"], color="#2e86ab", ls=":", lw=1)
    ax[2].set_ylabel("mm")
    ax[2].set_xlabel("s from window start (dashed: real release)")
    ax[2].legend(loc="upper left", fontsize=8)
    ax[2].set_ylim(-15, max(25, res["final"]["seat_depth_mm"] + 3))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def real_frames(episode: int, cam: str, t0: float, t1: float, out_dir: Path, fps: int = 30) -> list[Path]:
    """Decode one camera's frames for [t0, t1] (episode seconds) sequentially into PNGs."""
    import av
    from PIL import Image
    from twin.world import video_ref
    path, start = video_ref(episode, cam)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [start + t0 + i / fps for i in range(int(round((t1 - t0) * fps)) + 1)]
    files = []
    k = 0
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        c.seek(int(max(targets[0] - 1.0, 0) / st.time_base), stream=st, backward=True)
        for fr in c.decode(st):
            if fr.time is None:
                continue
            while k < len(targets) and fr.time >= targets[k] - 0.5 / fps:
                p = out_dir / f"{cam}_{k:04d}.png"
                Image.fromarray(fr.to_ndarray(format="rgb24")).save(p)
                files.append(p)
                k += 1
            if k >= len(targets):
                break
    return files


def strip_frames(frames: list, t: np.ndarray, series: list[tuple], out_dir: Path, title: str = "",
                 height: int = 150) -> list[Path]:
    """Append a force strip (with a moving cursor) under each frame. series: [(values, color, label)]."""
    from PIL import Image, ImageDraw
    out_dir.mkdir(parents=True, exist_ok=True)
    vmax = max(2.0, max(float(np.nanmax(v)) for v, _, _ in series) * 1.1)
    files = []
    n = min(len(frames), len(t))
    for i in range(n):
        im = Image.open(frames[i]).convert("RGB")
        W, H = im.size
        canvas = Image.new("RGB", (W, H + height), (255, 255, 255))
        canvas.paste(im, (0, 0))
        dr = ImageDraw.Draw(canvas)
        x0, x1, y0, y1 = 40, W - 10, H + 18, H + height - 18
        dr.rectangle([x0, y0, x1, y1], outline=(200, 200, 200))
        T = float(t[n - 1]) if n > 1 else 1.0
        sx = lambda tt: x0 + (x1 - x0) * tt / max(T, 1e-6)  # noqa: E731
        sy = lambda v: y1 - (y1 - y0) * v / vmax  # noqa: E731
        for vals, col, lab in series:
            pts = [(sx(t[j]), sy(vals[j])) for j in range(n)]
            dr.line(pts, fill=col, width=2)
        cx = sx(t[i])
        dr.line([(cx, y0), (cx, y1)], fill=(46, 134, 171), width=2)
        dr.text((x0, H + 2), f"{title}  |dF| N (max {vmax:.0f})  t={t[i]:.2f}s", fill=(30, 30, 30))
        lx = x1 - 170
        for k, (_, col, lab) in enumerate(series):
            dr.line([(lx, H + 8 + 0), (lx + 18, H + 8)], fill=col, width=3) if k == 0 else None
            dr.text((lx + 22 + 80 * k, H + 2), lab, fill=col)
        p = out_dir / f"f_{i:04d}.png"
        canvas.save(p)
        files.append(p)
    return files
