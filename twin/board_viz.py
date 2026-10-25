"""Plots for board-from-contacts (twin/board_fit.py): a top view per recording, a gate summary, a hama overlay.

All plots are in the robot base frame, mm. Colours are per part; the contact pose is solid, mocap dashed/grey.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from twin import REPO_DIR

PLATE_BOARD = ((-0.0102, 0.3738), (-0.3833, 0.0007))  # plate outline in the board frame (twin/calib, grid fit)


def _plate(pose):
    (x0, x1), (y0, y1) = PLATE_BOARD
    c = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
    if "R" in pose:
        R, t = pose["R"], np.asarray(pose["pos"])
        return 1000 * ((np.c_[c, np.zeros(len(c))] @ R.T)[:, :2] + t[:2])
    a = np.radians(pose["yaw_deg"])
    R2 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return 1000 * (c @ R2.T + np.asarray(pose["pos"][:2]))


def _colors(names):
    import matplotlib
    cmap = matplotlib.colormaps["tab20"]
    return {n: cmap(i % 20) for i, n in enumerate(sorted(names))}


def top_view(ep: int, layout: dict, fit: dict, mocap: dict | None, out: Path, camera: dict | None = None) -> Path:
    """Board seen from above: holes under the contact pose (+) and mocap pose (o), event clouds per part with the
    finger-pad line through each event (the in-hand offset runs along it), failed inserts (x), contact onsets (.)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from twin import board_fit as bf

    events = bf.extract_events(ep)
    rec = bf.recording_of(ep)
    holes_c = bf.world_holes(fit, layout)
    holes_m = bf.world_holes(mocap, layout) if mocap is not None and mocap.get("sane", True) else None
    holes_cam = bf.world_holes(camera, layout) if camera is not None else None
    col = _colors(layout["parts"])
    names = sorted(layout["parts"])
    fig = plt.figure(figsize=(17, 8.2))
    gs = fig.add_gridspec(4, 9, width_ratios=[1.0] * 4 + [0.15] + [0.62] * 4, wspace=0.35, hspace=0.45)
    ax = fig.add_subplot(gs[:, :4])
    zoom = [fig.add_subplot(gs[i // 4, 5 + i % 4]) for i in range(min(16, len(names)))]
    pc = _plate(fit)
    ax.plot(pc[:, 0], pc[:, 1], "-", color="k", lw=1.2, label="plate, contact pose")
    if holes_m is not None:
        pm = _plate(mocap)
        ax.plot(pm[:, 0], pm[:, 1], "--", color="0.5", lw=1.0, label="plate, mocap pose")
    if camera is not None:
        pcam = _plate(camera)
        ax.plot(pcam[:, 0], pcam[:, 1], ":", color="tab:purple", lw=1.2, label="plate, camera pose")
    for e in events:
        if e["part"] not in col:
            continue
        p = 1000 * np.array(e["tcp"][:2])
        c = col[e["part"]]
        if e["kind"] in ("insert_seat", "remove_grasp"):
            d = np.array(e["pad_dir"])
            seg = np.array([p - 12 * d, p + 12 * d])
            ax.plot(seg[:, 0], seg[:, 1], "-", color=c, lw=0.8, alpha=0.7)
            ax.plot(*p, "o" if e["kind"] == "insert_seat" else "^", ms=4.5, color=c, mec="k", mew=0.3)
        elif e["kind"] == "insert_fail":
            ax.plot(*p, "x", ms=5, color="tab:red", mew=1.0)
        elif e["kind"] == "onset":
            ax.plot(*p, ".", ms=2.5, color=c, alpha=0.5)
    for n in layout["parts"]:
        hc = 1000 * np.array(holes_c[n][:2])
        ax.plot(*hc, "+", ms=13, mew=2.0, color=col[n])
        if holes_m is not None:
            hm = 1000 * np.array(holes_m[n][:2])
            ax.plot(*hm, "o", ms=11, mfc="none", mec="0.35", mew=1.0)
        ax.annotate(n, hc, xytext=(6, 6), textcoords="offset points", fontsize=7, color=col[n])
    ax.plot([], [], "o", color="0.6", mec="k", mew=0.3, label="insert seat (TCP) + finger-pad line")
    ax.plot([], [], "^", color="0.6", mec="k", mew=0.3, label="remove grasp (TCP)")
    ax.plot([], [], "x", color="tab:red", label="failed insert (deepest TCP)")
    ax.plot([], [], "+", color="k", ms=10, mew=2, label="hole, contact pose")
    ax.plot([], [], "o", mfc="none", mec="0.35", ms=9, label="hole, mocap pose")
    ax.set_aspect("equal")
    ax.set_xlabel("robot x (mm)")
    ax.set_ylabel("robot y (mm)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    title = f"{rec} (ep {ep}, {bf.split_of(ep)}): board from {fit['n_events']} contact events, {fit['n_parts']} parts"
    if holes_m is not None:
        dd = [1000 * np.linalg.norm(np.array(holes_c[n][:2]) - np.array(holes_m[n][:2])) for n in layout["parts"]]
        title += f"\ncontact vs mocap: holes {np.median(dd):.1f} mm median (max {max(dd):.1f}), " \
                 f"yaw {((fit['yaw_deg'] - mocap['yaw_deg'] + 180) % 360 - 180):+.2f} deg"
    ax.set_title(title, fontsize=9)
    # right: one zoom per part (+-R mm around the contact-pose hole), same scale everywhere
    R = 22.0
    ord_names = [n for n in names if any(e["part"] == n for e in events)] or names
    ord_names = sorted(ord_names, key=lambda n: (-sum(e["part"] == n and e["kind"] in ("insert_seat", "remove_grasp")
                                                     for e in events), n))[:len(zoom)]
    for z, n in zip(zoom, ord_names):
        hc = 1000 * np.array(holes_c[n][:2])
        for e in events:
            if e["part"] != n:
                continue
            p = 1000 * np.array(e["tcp"][:2]) - hc
            if e["kind"] in ("insert_seat", "remove_grasp"):
                d = np.array(e["pad_dir"])
                seg = np.array([p - 30 * d, p + 30 * d])
                z.plot(seg[:, 0], seg[:, 1], "-", color=col[n], lw=0.8)
                z.plot(*p, "o" if e["kind"] == "insert_seat" else "^", ms=5, color=col[n], mec="k", mew=0.4)
            elif e["kind"] == "insert_fail":
                z.plot(*p, "x", ms=6, color="tab:red", mew=1.2)
            elif e["kind"] == "onset":
                z.plot(*p, ".", ms=4, color="0.4")
        z.plot(0, 0, "+", ms=14, mew=2.0, color="k")
        if holes_m is not None:
            z.plot(*(1000 * np.array(holes_m[n][:2]) - hc), "o", ms=10, mfc="none", mec="0.35", mew=1.2)
        if holes_cam is not None:
            z.plot(*(1000 * np.array(holes_cam[n][:2]) - hc), "s", ms=9, mfc="none", mec="tab:purple", mew=1.2)
        z.set_xlim(-R, R)
        z.set_ylim(-R, R)
        z.set_aspect("equal")
        z.set_xticks([-20, 0, 20])
        z.set_yticks([-20, 0, 20])
        z.tick_params(labelsize=6)
        z.grid(alpha=0.25)
        z.set_title(n, fontsize=8, color=col[n])
    for z in zoom[len(ord_names):]:
        z.axis("off")
    fig.text(0.735, 0.005, f"zooms: +-{R:.0f} mm around each contact-pose hole (+); grey o = mocap hole"
             + ("; purple square = camera hole" if holes_cam is not None else "")
             + "; lines = finger-pad axis through each TCP (in-hand offset runs along it); . = contact onsets",
             ha="center", fontsize=7)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.07)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def gate_summary(err: dict, out: Path) -> Path:
    """Per gate recording: leave-one-part-out closing-axis error and B0 metric, contact vs mocap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = err["recordings"]
    lab = [f"ep {r['episode']}\n{r['n_parts']}p" for r in recs]
    x = np.arange(len(recs))
    YMAX = 20.0

    def get(r, k):
        return r["summary"].get(k, {}).get("median", np.nan)

    fig, axs = plt.subplots(2, 1, figsize=(14, 7.2), sharex=True)
    panels = (("contact_close_mm", "mocap_close_mm", "across-pad (closing-axis) error, mm"),
              ("b0_contact_mm", "b0_mocap_mm", "hole vs median release TCP, mm"))
    for ax, (kc, km, name) in zip(axs, panels):
        vc, vm = [get(r, kc) for r in recs], [get(r, km) for r in recs]
        ax.bar(x - 0.2, np.minimum(vc, YMAX), 0.4, label="contact pose, fitted without the part", color="tab:blue")
        ax.bar(x + 0.2, np.minimum(vm, YMAX), 0.4, label="mocap pose", color="0.6")
        for xi, v in zip(x, vm):
            if np.isfinite(v) and v > YMAX:
                ax.annotate(f"{v:.0f}", (xi + 0.2, YMAX), ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, YMAX * 1.12)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(3.0, color="tab:red", lw=0.8, ls="--")
        ax.legend(fontsize=8, loc="upper left")
    p = err["pooled"]
    g = lambda k: p.get(k, {}).get("median_over_parts", float("nan"))  # noqa: E731
    axs[0].set_title(f"Across the finger pads (free of in-hand offset): median contact {g('contact_close_mm'):.2f} mm vs "
                     f"mocap {g('mocap_close_mm'):.2f} mm over {p.get('contact_close_mm', {}).get('n_parts', 0)} "
                     f"recording-parts (insert seats + remove grasps)", fontsize=9)
    axs[1].set_title(f"B0.board_err_mm (insert releases). Per recording: contact {g('b0_contact_mm'):.2f} vs mocap "
                     f"{g('b0_mocap_mm'):.2f} mm. Pooled over recordings per part: contact {g('b0_pooled_contact_mm'):.2f} "
                     f"vs mocap {g('b0_pooled_mocap_mm'):.2f} mm", fontsize=9)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(lab, fontsize=7)
    fig.suptitle(f"Board from robot contacts, {len(recs)} held-out gate recordings (layout from train only); "
                 f"bars = median over parts per recording; red = 3 mm target", fontsize=10)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def hama_overlay(ep: int, camera: str, frame: int, layout: dict, fit: dict, mocap: dict | None, out: Path,
                 crop: tuple | None = None, scale: int = 1, axis_mm: float = 40.0) -> Path:
    """Project the plate outline and each hole's axis (entry point + a vertical segment axis_mm tall, which
    lies on a seated peg) into a calibrated static camera (twin/calib/<recording>.json) over the real frame.
    Contact pose: cyan; mocap pose: magenta. crop = (x0, y0, x1, y1) in px, scaled up by `scale`."""
    import cv2
    from harness.tools.world import real_frame
    from twin import board_fit as bf
    rec = bf.recording_of(ep)
    cal = json.loads((REPO_DIR / "twin" / "calib" / f"{rec}.json").read_text())["cameras"][camera]
    T = np.array(cal["T_parent_cam_opencv"])
    K = np.array([[cal["fx"], 0, cal["cx"]], [0, cal["fy"], cal["cy"]], [0, 0, 1.0]])
    img = cv2.imread(str(REPO_DIR / real_frame(ep, camera, frame=frame)["path"]))
    ox, oy = 0, 0
    if crop:
        ox, oy, x1, y1 = crop
        img = img[oy:y1, ox:x1]
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    SH = 4  # cv2 sub-pixel shift bits

    def proj(P):
        P = np.atleast_2d(P)
        pc = (P - T[:3, 3]) @ T[:3, :3]
        uv = (pc @ K.T)
        uv = uv[:, :2] / uv[:, 2:3]
        return ((uv + 0.5 - [ox, oy]) * scale - 0.5) * (1 << SH)

    def pt(u):
        return tuple(int(round(v)) for v in u)

    def draw(pose, color):
        if "R" in pose:
            R, t = pose["R"], np.asarray(pose["pos"])
            to_w = lambda xy, z=0.0: (np.c_[xy, np.full(len(xy), z)] @ R.T) + t  # noqa: E731
        else:
            a = np.radians(pose["yaw_deg"])
            R2 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            to_w = lambda xy, z=0.0: np.c_[xy @ R2.T + np.asarray(pose["pos"][:2]),  # noqa: E731
                                           np.full(len(xy), pose["pos"][2] + z)]
        (x0, x1), (y0, y1) = PLATE_BOARD
        c = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        cv2.polylines(img, [proj(to_w(c)).astype(np.int32).reshape(-1, 1, 2)], True, color, 1, cv2.LINE_AA, SH)
        hx = np.array([layout["parts"][n]["xy"] for n in layout["parts"]])
        for a, b in zip(proj(to_w(hx)), proj(to_w(hx, axis_mm / 1000))):
            cv2.line(img, pt(a), pt(b), color, 1, cv2.LINE_AA, SH)
            cv2.circle(img, pt(a), 3 << SH, color, 1, cv2.LINE_AA, SH)

    if mocap is not None:
        draw(mocap, (255, 0, 255))
    draw(fit, (255, 255, 0))
    cv2.putText(img, f"{rec} {camera} frame {frame}: cyan = contact pose, magenta = mocap pose "
                     f"(hole entry o + {axis_mm:.0f} mm axis)", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    out = Path(out)
    cv2.imwrite(str(out), img)
    return out
