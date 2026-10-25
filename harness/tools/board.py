"""Stage B0 tools: place the board from the robot's own contact events (skill: playbook/skills/board-from-contacts).

Core code: twin/board_fit.py (events, layout fit, per-recording fit, held-out evaluation); plots: twin/board_viz.py.
Outputs go under runs/<brain>/<run_id>/board/.

Tools:
  board_contact_layout  fit the rigid hole layout (board frame) on train recordings: holes, 75 mm grid check,
                        how far each train recording's mocap board pose had to move, grip width per part
  board_from_contacts   board pose of one recording from its insert/remove contact events (no mocap, no camera);
                        writes <recording>.json with pose, per-hole positions, residuals, mocap/camera deltas
  board_err             the held-out measurement: leave-one-part-out within each recording and the B0.board_err_mm
                        number (contact vs mocap) on gate (or train, leave-one-recording-out) recordings
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from harness.registry import REPO, tool

_RUN = {"brain": {"type": "string", "default": "claude", "description": "claude (stand-in) or argon: whose run directory"},
        "run_id": {"type": "string", "default": "run-standin-1", "description": "run id: outputs go to runs/<brain>/<run_id>/board/"}}


def _out(brain: str, run_id: str) -> Path:
    for s, f in ((brain, "brain"), (run_id, "run_id")):
        if not s or "/" in s or s.startswith("."):
            raise ValueError(f"bad {f}: {s!r}")
    p = REPO / "runs" / brain / run_id / "board"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def _resolve(recording: str | None, episode: int | None) -> tuple[int, str]:
    from twin import board_fit as bf
    if episode is not None:
        return int(episode), bf.recording_of(int(episode))
    if recording:
        return bf.episode_of(recording), recording
    raise ValueError("give recording or episode")


def _grid_check(layout: dict) -> dict:
    """Fit the 8 pegs to a square 75 mm grid (NIST task board pitch): origin + angle only."""
    from scipy.optimize import least_squares
    from twin.layout import PEG_CELLS
    names = [k for k in PEG_CELLS if k in layout["parts"]]
    if len(names) < 3:
        return {}
    pts = np.array([[1000 * v for v in layout["parts"][k]["xy"]] for k in names])
    cells = np.array([PEG_CELLS[k] for k in names], float)

    def f(x, pitch):
        c, s = np.cos(x[2]), np.sin(x[2])
        g = np.c_[pitch[0] * cells[:, 0], -pitch[1] * cells[:, 1]]
        return (pts - (g @ np.array([[c, s], [-s, c]]) + x[:2])).ravel()

    fixed = least_squares(lambda x: f(x, (75.0, 75.0)), [pts[:, 0].min(), pts[:, 1].max(), 0.0])
    free = least_squares(lambda x: f(x[:3], x[3:]), np.r_[fixed.x, 75.0, 75.0])
    r75 = f(fixed.x, (75.0, 75.0)).reshape(-1, 2)
    rfree = f(free.x[:3], free.x[3:]).reshape(-1, 2)
    return dict(nist_pitch_mm=75.0, fixed75_rms_mm=round(float(np.sqrt((r75 ** 2).sum(1).mean())), 3),
                fixed75_max_mm=round(float(np.linalg.norm(r75, axis=1).max()), 3),
                fixed75_angle_deg=round(float(np.degrees(fixed.x[2])), 3),
                free_pitch_mm=[round(float(v), 2) for v in free.x[3:]],
                free_rms_mm=round(float(np.sqrt((rfree ** 2).sum(1).mean())), 3),
                cells={k: list(PEG_CELLS[k]) for k in names})


def camera_board(recording: str, brain: str = "claude", run_id: str = "run-standin-1") -> dict | None:
    """Board pose from camera triangulation (written by a watcher) if present: {pos, yaw_deg} or None."""
    p = REPO / "runs" / brain / run_id / "b0-watch" / "board_camera.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    cands = [d] + (d.get("recordings", []) if isinstance(d.get("recordings"), list) else []) + \
        (list(d["recordings"].values()) if isinstance(d.get("recordings"), dict) else [])
    for c in cands:
        if not isinstance(c, dict) or c.get("recording", recording) != recording:
            continue
        b = c.get("board", c)
        pos = b.get("pos") or b.get("position")
        if pos is None:
            continue
        yaw = b.get("yaw_deg")
        q = b.get("quat") or b.get("quat_wxyz")
        if yaw is None and q is not None:
            w, x, y, z = q
            yaw = float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
        if yaw is None:
            continue
        return dict(pos=[float(v) for v in pos], yaw_deg=float(yaw), source=_rel(p))
    return None


def _holes_delta(pa: dict, pb: dict, layout: dict) -> dict:
    from twin import board_fit as bf
    ha, hb = bf.world_holes(pa, layout), bf.world_holes(pb, layout)
    d = {k: round(1000 * float(np.linalg.norm(np.array(ha[k][:2]) - np.array(hb[k][:2]))), 2) for k in layout["parts"]}
    v = list(d.values())
    ya = pa["yaw_deg"]
    yb = pb["yaw_deg"]
    return dict(hole_shift_median_mm=round(float(np.median(v)), 2), hole_shift_max_mm=round(float(max(v)), 2),
                dyaw_deg=round(float((ya - yb + 180) % 360 - 180), 3),
                dz_mm=round(1000 * (float(pa["pos"][2]) - float(pb["pos"][2])), 2), per_hole_mm=d)


@tool("board_contact_layout",
      "Fit the board's rigid hole layout (board frame = mocap NIST_Board1 convention) from contact events of all "
      "train recordings on disk: insert seats (deepest TCP before the fingers open) and remove grasps (fingers "
      "closing on a seated part), each a tight constraint across the finger pads and a loose one along them "
      "(in-hand offset), plus a small per-recording correction of the mocap board pose. Returns hole xy per part, "
      "a 75 mm NIST grid check on the 8 pegs, how far mocap had to move per recording, and grip width per part "
      "(peg size check). Also places the NIST laser-plate layout (CAD hole/feature positions) in the board frame "
      "with one rigid transform (reflection allowed) and reports how far each contact hole is from its NIST "
      "feature. Writes layout_contact.json and layout_nist.json under runs/<brain>/<run_id>/board/. "
      "Never reads exam recordings.",
      {"type": "object", "properties": dict(_RUN)})
def board_contact_layout(brain: str = "claude", run_id: str = "run-standin-1") -> dict:
    from twin import board_fit as bf
    out = _out(brain, run_id)
    lay = bf.fit_layout(bf.episodes_in("train"))
    lay["grid_check"] = _grid_check(lay)
    f = out / "layout_contact.json"
    f.write_text(json.dumps(lay, indent=1))
    nl = bf.nist_layout(lay)
    fn = out / "layout_nist.json"
    fn.write_text(json.dumps(nl, indent=1))
    return dict(file=_rel(f), file_nist=_rel(fn), n_recordings=len(lay["recordings"]), n_events=lay["n_events"],
                fit=lay["fit"], skipped=lay["skipped"], grid_check=lay["grid_check"], nist_fit=nl["nist_fit"],
                mocap_correction_summary=lay["mocap_correction_summary"],
                parts={k: dict(xy=v["xy"], n=v["n_events"], close_resid_mad_mm=v["close_resid_mad_mm"],
                               grip_mm=v["grip_mm_median"]) for k, v in lay["parts"].items()})


@tool("board_from_contacts",
      "Board pose of one REASSEMBLE recording in the robot base frame from that recording's own contact events "
      "(TCP when a part sits in its hole: insert seats and remove grasps), using a rigid hole layout fitted on the "
      "OTHER train recordings. No mocap or camera is used in the fit; both are compared afterwards. Returns pose "
      "(pos, yaw, quat w-first; roll/pitch 0), 1-sigma, per-hole world positions, per-event residuals (closing-axis "
      "residual should be ~1 mm; the pad-axis offset is the in-hand offset), and deltas vs mocap and camera. "
      "exclude_parts leaves parts out (leave-one-part-out by hand). Writes runs/<brain>/<run_id>/board/<recording>.json.",
      {"type": "object", "properties": dict(
          recording={"type": "string", "description": "recording id, e.g. 2025-01-11-11-12-32"},
          episode={"type": "integer", "description": "or the episode index"},
          exclude_parts={"type": "array", "items": {"type": "string"}, "description": "dataset part names to leave out"},
          layout={"type": "string", "enum": ["contact", "nist"], "default": "contact",
                  "description": "hole layout: fitted from train contacts, or the NIST laser plate placed by it"},
          write={"type": "boolean", "default": True}, plot={"type": "boolean", "default": True}, **_RUN)})
def board_from_contacts(recording: str | None = None, episode: int | None = None, exclude_parts: list | None = None,
                        layout: str = "contact", write: bool = True, plot: bool = True, brain: str = "claude",
                        run_id: str = "run-standin-1") -> dict:
    from twin import board_fit as bf
    ep, rec = _resolve(recording, episode)
    out = _out(brain, run_id)
    train = bf.episodes_in("train")
    lay_eps = [e for e in train if e != ep]
    lay = bf.fit_layout(lay_eps)
    if layout == "nist":
        lay = bf.nist_layout(lay)
    events = bf.extract_events(ep)
    fit = bf.fit_board(events, lay, exclude_parts=exclude_parts or ())
    holes = bf.world_holes(fit, lay)
    res = dict(
        recording=rec, episode=ep, split=bf.split_of(ep), frame="robot base frame (FR3 link0 = MuJoCo world = mocap world)",
        method="contact events (insert seats + remove grasps) -> robust point-to-line fit of the rigid layout "
               "(twin/board_fit.py fit_board); layout from train recordings excluding this one",
        layout=dict(source=layout, episodes_excluded=[ep] if ep in train else [], n_recordings=len(lay["recordings"]),
                    params=lay["params"], nist_fit=lay.get("nist_fit")),
        pose={k: fit[k] for k in ("pos", "yaw_deg", "quat_wxyz", "sigma")},
        fit={k: fit[k] for k in ("n_events", "n_parts", "parts", "excluded", "close_resid_median_mm",
                                  "close_resid_p90_mm", "pad_offset_median_mm", "n_outliers", "status")},
        holes={k: dict(world=[round(v, 5) for v in holes[k]], board_xy=lay["parts"][k]["xy"]) for k in holes},
        drift=bf.drift(events, lay),
        events=fit["events"])
    mb = bf.mocap_board(rec)
    if mb is not None:
        res["mocap"] = dict(pos=mb["pos"].tolist(), yaw_deg=mb["yaw_deg"], tilt_deg=mb["tilt_deg"], sane=mb["sane"],
                            delta_contact_minus_mocap=_holes_delta(fit, dict(pos=mb["pos"], R=mb["R"],
                                                                             yaw_deg=mb["yaw_deg"]), lay))
    cam = camera_board(rec, brain, run_id)
    res["camera"] = None if cam is None else dict(cam, delta_contact_minus_camera=_holes_delta(fit, cam, lay))
    if cam is not None and mb is not None:
        res["camera"]["delta_mocap_minus_camera"] = _holes_delta(dict(pos=mb["pos"], R=mb["R"], yaw_deg=mb["yaw_deg"]),
                                                                 cam, lay)
    f = out / (f"{rec}.json" if layout == "contact" else f"{rec}_{layout}.json")
    if write:
        f.write_text(json.dumps(res, indent=1))
        res["file"] = _rel(f)
    if plot:
        from twin.board_viz import top_view
        png = top_view(ep, lay, fit, mb, out / (f"{rec}_top.png" if layout == "contact" else f"{rec}_{layout}_top.png"),
                       camera=cam)
        res["plot"] = _rel(png)
    slim = {k: v for k, v in res.items() if k not in ("events", "holes")}
    slim["holes_world_xy"] = {k: [round(v["world"][0], 4), round(v["world"][1], 4)] for k, v in res["holes"].items()}
    if res.get("mocap"):
        slim["mocap"] = {k: v for k, v in res["mocap"].items()}
        slim["mocap"]["delta_contact_minus_mocap"] = {k: v for k, v in res["mocap"]["delta_contact_minus_mocap"].items()
                                                      if k != "per_hole_mm"}
    return slim


@tool("board_err",
      "Held-out board-placement error from contact events. For every recording of a split: leave-one-part-out (fit "
      "the board from the other parts' events, predict the held-out part's hole) and compare with the mocap pose on "
      "the same events. Reports per part (median over its events) the closing-axis error (perpendicular to the "
      "finger pads, free of in-hand offset), the xy error, and the B0.board_err_mm quantity as written in "
      "validation.yaml (xy distance between the hole and the median TCP at that part's successful insert releases), "
      "for contact and mocap poses; pooled = median over all (recording, part) pairs. split 'gate' uses the train "
      "layout (true hold-out); 'train' refits the layout without each recording. Writes "
      "runs/<brain>/<run_id>/board/board_err_<split>.json.",
      {"type": "object", "properties": dict(
          split={"type": "string", "enum": ["gate", "train"], "default": "gate"},
          recordings={"type": "array", "items": {"type": "string"}, "description": "subset of the split (default all on disk)"},
          layout={"type": "string", "enum": ["contact", "nist"], "default": "contact",
                  "description": "hole layout: fitted from train contacts (default, frozen), or the NIST laser plate placed by it"},
          **_RUN)})
def board_err(split: str = "gate", recordings: list | None = None, layout: str = "contact", brain: str = "claude",
              run_id: str = "run-standin-1") -> dict:
    from twin import board_fit as bf
    out = _out(brain, run_id)
    eps = bf.episodes_in(split)
    if recordings:
        want = {bf.episode_of(r) for r in recordings}
        bad = want - set(eps)
        if bad:
            raise ValueError(f"not {split} recordings on disk: {sorted(bf.recording_of(e) for e in bad)}")
        eps = sorted(want)
    r = bf.evaluate(eps, bf.episodes_in("train"), loro=(split == "train"), layout_source=layout)
    f = out / (f"board_err_{split}.json" if layout == "contact" else f"board_err_{split}_{layout}.json")
    f.write_text(json.dumps(r, indent=1))
    lines = []
    for p in r["recordings"]:
        s = p["summary"]
        g = lambda k: s.get(k, {}).get("median")  # noqa: E731
        lines.append(dict(recording=p["recording"], episode=p["episode"], n_parts=p["n_parts"], kinds=p["kinds"],
                          contact_close_mm=g("contact_close_mm"), mocap_close_mm=g("mocap_close_mm"),
                          b0_contact_mm=g("b0_contact_mm"), b0_mocap_mm=g("b0_mocap_mm"),
                          contact_vs_mocap=None if not p["contact_vs_mocap"] else
                          {k: p["contact_vs_mocap"][k] for k in ("hole_shift_median_mm", "dyaw_deg", "dz_mm")}))
    pooled = r["pooled"]
    return dict(file=_rel(f), split=split, layout=layout, n_recordings=len(eps), pooled=pooled,
                b0_board_err_mm=dict(
                    per_recording=dict(
                        contact_lopo=pooled.get("b0_contact_mm", {}).get("median_over_parts"),
                        mocap=pooled.get("b0_mocap_mm", {}).get("median_over_parts"),
                        reading="per (recording, part): xy distance hole entry vs the median TCP at that part's "
                                "successful insert releases in that recording; median over (recording, part). "
                                "Includes each grasp's in-hand offset along the finger pads (~3.5 mm median)."),
                    pooled_over_recordings=dict(
                        contact_lopo=pooled.get("b0_pooled_contact_mm", {}).get("median_over_parts"),
                        mocap=pooled.get("b0_pooled_mocap_mm", {}).get("median_over_parts"),
                        reading="per part: median release TCP over all held-out recordings, each mapped into the "
                                "board frame with its own pose; xy distance to the hole; median over parts. "
                                "Closest to the validation.yaml wording; in-hand offsets average out."),
                    closing_axis=dict(
                        contact_lopo=pooled.get("contact_close_mm", {}).get("median_over_parts"),
                        mocap=pooled.get("mocap_close_mm", {}).get("median_over_parts"),
                        reading="proposed: distance across the finger pads only (free of in-hand offset), insert "
                                "seats and remove grasps, median over (recording, part)"),
                    note="contact pose for each (recording, part) is fitted without that part (leave-one-part-out); "
                         "gate recordings use a layout fitted on train recordings only"),
                per_recording=lines)
