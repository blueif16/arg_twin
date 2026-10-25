"""Backfill the REASSEMBLE episode-0 reconstruction decisions of the Claude stand-in run as decision records.

    .venv/bin/python -m scripts.backfill_decisions_standin

1. Computes the few numbers that are derived rather than read (each from files under runs/), and writes them with
   their inputs to runs/claude/run-standin-1/reconstruct/derived.json:
   - native and silhouette camera hypotheses projected against the final keypoint calibration (17 board holes)
   - the twin part's distance from the hole axis at release, with and without in-hand registration
   - the label-blind failure split counts
2. Logs one log_decision record per decision through the tool registry (calls land in runs/calls/), skipping ids
   already in runs/claude/run-standin-1/decisions.jsonl, so it can be rerun.
3. Writes the native-vs-tools chart with decision_summary.

Every other number is copied from the evidence file or skill Lessons named in the record.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from harness.registry import REPO, call

RUN = REPO / "runs" / "claude" / "run-standin-1"
OUT = RUN / "reconstruct"
CALLER = "claude:reconstruct-backfill"
BRAIN, RUN_ID, DAY = "claude", "run-standin-1", "2026-10-24"


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def _load(p: str):
    return json.loads((REPO / p).read_text())


def project(cam: dict, pts: np.ndarray) -> np.ndarray:
    T = np.array(cam["T_parent_cam_opencv"])  # camera pose in world, OpenCV axes
    pc = (np.linalg.inv(T) @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    return np.c_[cam["fx"] * pc[:, 0] / pc[:, 2] + cam["cx"], cam["fy"] * pc[:, 1] / pc[:, 2] + cam["cy"]]


def derived() -> dict:
    calib = _load("twin/calib/2025-01-09-14-11-58.json")
    final = _load("runs/claude/run-standin-1/world/static_fit.json")["cams"]
    layout = _load("runs/claude/run-standin-1/board/layout_contact.json")["parts"]
    known = _load("runs/claude/run-standin-1/b0-watch/b0_known_error.json")
    b = calib["board"]
    w, x, y, z = b["quat"]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    holes = np.array([np.array(b["pos"]) + R @ np.array([p["xy"][0], p["xy"][1], 0.0]) for p in layout.values()])
    cams = {}
    for cam in ("hama1", "hama2"):
        ref = _load(f"runs/world/refine/refine_ep000_{cam}.json")
        uv_final = project(final[cam], holes)
        mm_per_px = known["static"][cam]["geometry_at_board_centre"]["mm_per_px_lateral"]
        row = {"mm_per_px_at_board": mm_per_px}
        for name, c in (("native", ref["init"]), ("silhouette", ref["final"])):
            d = np.linalg.norm(project(c, holes) - uv_final, axis=1)
            Ra, Rb = np.array(c["T_parent_cam_opencv"])[:3, :3], np.array(final[cam]["T_parent_cam_opencv"])[:3, :3]
            ang = np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))
            row[name] = {"hole_px_median": round(float(np.median(d)), 2), "hole_px_max": round(float(d.max()), 2),
                         "hole_mm_median": round(float(np.median(d)) * mm_per_px, 1),
                         "position_diff_mm": round(float(np.linalg.norm(np.subtract(c["pos"], final[cam]["pos"])) * 1000), 1),
                         "rotation_diff_deg": round(float(ang), 2), "f_px": c["fx"]}
        row["final_f_px"] = final[cam]["fx"]
        cams[cam] = row
    wrist = [k["mm_per_px_lateral"] for k in known["hand"]["geometry_at_keypoints"]]

    def lateral(fn):
        res = _load(f"runs/claude/run-standin-1/replay/{fn}")["results"]
        v = [r["at_release"]["lateral_mm"] for r in res
             if r.get("at_release") and isinstance(r["at_release"].get("lateral_mm"), (int, float))]
        return {"n": len(v), "median_mm": round(float(np.median(v)), 2),
                "iqr_mm": [round(float(np.percentile(v, 25)), 2), round(float(np.percentile(v, 75)), 2)]}

    fs = _load("runs/claude/run-standin-1/replay/failure_split.json")
    fail = [r for r in fs if not r["real"]]
    jam = [r for r in fail if r["sustained_s"] >= 0.3 and r["peak_dF"] > 5 and r["descent_mm"] < 2]
    return {
        "inputs": ["twin/calib/2025-01-09-14-11-58.json", "runs/claude/run-standin-1/world/static_fit.json",
                   "runs/claude/run-standin-1/board/layout_contact.json", "runs/world/refine/refine_ep000_hama1.json",
                   "runs/world/refine/refine_ep000_hama2.json", "runs/claude/run-standin-1/b0-watch/b0_known_error.json",
                   "runs/claude/run-standin-1/replay/v2_train.json", "runs/claude/run-standin-1/replay/v2_train_prediction.json",
                   "runs/claude/run-standin-1/replay/failure_split.json"],
        "method": {
            "cameras": "17 contact-layout holes at the plate top (calib board pose) projected with each camera hypothesis "
                       "(its own intrinsics) vs the final keypoint calibration; px to mm with mm_per_px_lateral at the "
                       "board centre from b0_known_error.json. This measures distance from the final calibration, whose "
                       "own held-out error is the watcher's pad reprojection.",
            "lateral": "median at_release.lateral_mm (twin part tip vs hole axis at the release frame) over the replay results",
            "failure_split": "real failures with |dF| > 5 N for >= 0.3 s and < 2 mm descent = physical-jam signature"},
        "cameras": cams,
        "wrist_mm_per_px_median": round(float(np.median(wrist)), 3),
        "inhand_lateral_at_release": {"prediction_none": lateral("v2_train_prediction.json"),
                                      "reconstruction_inhand": lateral("v2_train.json")},
        "failure_split": {"real_failures": len(fail), "jam_signature": len(jam),
                          "jam_signature_jammed_in_twin": sum(1 for r in jam if not r["v2"]),
                          "never_released": sum(1 for r in fail if not r["released"])},
    }


def records(dv: dict) -> list[dict]:
    c1, c2 = dv["cameras"]["hama1"], dv["cameras"]["hama2"]
    lat = dv["inhand_lateral_at_release"]
    fsplit = dv["failure_split"]
    D = rel(OUT / "derived.json")
    WF, BC, BT, MJ, RV, VC = (f"playbook/skills/{s}/SKILL.md" for s in
                             ("world-frame", "board-from-contacts", "build-twin", "mujoco", "replay-validate", "validate-checks"))
    R = "runs/claude/run-standin-1/replay/"
    recs = [
        dict(id="intake-reassemble", stage="B0", step="intake", quantity="what REASSEMBLE episode 0 provides",
             saw=["docs/research/reassemble-data-notes.md", "data/reassemble/frames/ep000_f02117_hand.png"],
             native={"source": "none", "estimate": ""}, tool=None,
             result={"summary": "FR3 + Franka hand: joints and O_T_EE per frame (30 fps, resampled to the hand-camera "
                                "clock), gripper width, wrist F/T (gravity- and bias-compensated, base frame), 3 cameras "
                                "(2 static webcams, wrist RealSense) with no intrinsics, mocap poses of camera brackets and "
                                "board (often wrong), segment labels with success, no commanded setpoint (action = next "
                                "measured pose), NIST board CAD/STL, peg clearances 0.05-0.25 mm per side."},
             error=None, verdict="adopt",
             decision="Robot recording with joints and video: the robot is the calibration object; with F/T, contacts "
                      "can place the fixture; commands must be rebuilt from pose and force.",
             evidence=["docs/research/reassemble-data-notes.md", BT], skill="reconstruct-scene"),
        dict(id="cam-hama1-pose", stage="B0", step="cameras", quantity="hama1 pose (static camera 1)",
             saw=["runs/world/frames/ep000_hama1_f00000.png", "runs/world/overlays/native_ep000_hama1.png"],
             native={"source": "native", "estimate": "look-at by eye: position (1.45, -0.70, 0.55) m, fovy 45 deg "
                     "(f 579 px); side from which way robot +x runs, azimuth from base->board (-26 deg by eye)",
                     "values": {"pos_m": [1.45, -0.70, 0.55], "f_px": 579.41,
                                "dist_to_final_mm": c1["native"]["position_diff_mm"],
                                "rot_to_final_deg": c1["native"]["rotation_diff_deg"]}},
             tool={"name": "calibrate_static_cameras", "call": "runs/claude/run-standin-1/world/static_fit.json"},
             result={"summary": "robot-keypoint bundle adjustment (arm caps + finger pads from FK, shared f, latency, "
                                "tool offset, plate height): pos (1.455, -0.672, 0.540) m, f 553.96 px. An earlier "
                                "silhouette fit reached 0.96 px edge distance but was 5.35 px off on held-out pads.",
                     "values": {"f_px": 553.96, "silhouette_hole_px_to_final": c1["silhouette"]["hole_px_median"]}},
             error={"metric": "before: native pose's board-hole projections vs the final calibration (median of 17); "
                              "after: watcher's held-out finger-pad reprojection", "unit": "px",
                    "before": c1["native"]["hole_px_median"], "after": 1.63, "heldout": True,
                    "mm_per_px": c1["mm_per_px_at_board"],
                    "reference": "final keypoint calibration (derived.json); watcher marks (measure_b0.json)"},
             verdict="adopt", decision="Keep the keypoint calibration; the native look set side, up and azimuth, the "
                                       "bundle adjustment set the pose to pixel level.",
             evidence=[D, "runs/claude/run-standin-1/world/static_fit.json",
                       "runs/claude/run-standin-1/b0-watch/measure_b0.json", "runs/world/refine/refine_ep000_hama1.json", WF],
             skill="world-frame", check_ids=["B0.kp_hama1"]),
        dict(id="cam-hama2-pose", stage="B0", step="cameras", quantity="hama2 pose (static camera 2)",
             saw=["runs/world/frames/ep000_hama2_f00000.png", "runs/world/overlays/native_ep000_hama2.png"],
             native={"source": "native", "estimate": "look-at by eye: position (1.40, 0.35, 0.60) m, fovy 45 deg (f 579 px)",
                     "values": {"pos_m": [1.40, 0.35, 0.60], "f_px": 579.41,
                                "dist_to_final_mm": c2["native"]["position_diff_mm"],
                                "rot_to_final_deg": c2["native"]["rotation_diff_deg"]}},
             tool={"name": "calibrate_static_cameras", "call": "runs/claude/run-standin-1/world/static_fit.json"},
             result={"summary": "keypoint bundle adjustment: pos (1.534, 0.278, 0.640) m, shared f 553.96 px. The "
                                "silhouette fit had a free focal of 518 px 12 cm closer (depth/focal valley) and was "
                                "16 px off on held-out pads.",
                     "values": {"f_px": 553.96, "silhouette_hole_px_to_final": c2["silhouette"]["hole_px_median"]}},
             error={"metric": "before: native pose's board-hole projections vs the final calibration (median of 17); "
                              "after: watcher's held-out finger-pad reprojection", "unit": "px",
                    "before": c2["native"]["hole_px_median"], "after": 1.51, "heldout": True,
                    "mm_per_px": c2["mm_per_px_at_board"],
                    "reference": "final keypoint calibration (derived.json); watcher marks (measure_b0.json)"},
             verdict="adopt", decision="Keep the keypoint calibration; silhouettes cannot separate focal length from distance.",
             evidence=[D, "runs/claude/run-standin-1/world/static_fit.json",
                       "runs/claude/run-standin-1/b0-watch/measure_b0.json", "runs/world/refine/refine_ep000_hama2.json", WF],
             skill="world-frame", check_ids=["B0.kp_hama2"]),
        dict(id="cam-two-view", stage="B0", step="cameras", quantity="hama1/hama2 agree in 3D",
             saw=["runs/claude/run-standin-1/b0-watch/board_camera.json"],
             native={"source": "native", "estimate": "silhouette-refined pair assumed consistent (both passed the "
                     "3 px edge check)", "values": {"edge_px_hama1": 0.955, "edge_px_hama2": 0.955}},
             tool={"name": "triangulate_marks"},
             result={"summary": "watcher two-view residual 7.11 px before, 0.47 px after the keypoint recalibration"},
             error={"metric": "median per-view residual after triangulating the same marked points", "unit": "px",
                    "before": 7.11, "after": 0.47, "heldout": True, "mm_per_px": c1["mm_per_px_at_board"],
                    "reference": "watcher marks in both cameras"},
             verdict="adopt", decision="Two-view consistency becomes a gated check (validation v5).",
             evidence=["runs/claude/run-standin-1/b0-watch/board_camera.json",
                       "runs/claude/run-standin-1/b0-watch/two_view_recal.json", VC],
             skill="validate-checks", check_ids=["B0.two_view_px"]),
        dict(id="cam-intrinsics", stage="B0", step="cameras", quantity="static camera focal length",
             saw=["docs/research/reassemble-data-notes.md"],
             native={"source": "prior", "estimate": "no intrinsics in the data; fovy 45 deg guess gives f 579.4 px, "
                     "principal point at the centre (319.5, 239.5)", "values": {"f_px": 579.41}},
             tool={"name": "calibrate_static_cameras"},
             result={"summary": "shared f 553.96 px, principal point (317.1, 222.2), se about 1 px; free k1 (-0.06 to "
                                "-0.13) did not improve held-out error, so pinhole",
                     "values": {"f_px": 553.96, "cx": 317.14, "cy": 222.22}},
             error={"metric": "relative focal-length error (|f - f_final| / f_final; after = se / f)", "unit": "fraction",
                    "before": round(abs(579.41 - 553.96) / 553.96, 4), "after": round(1 / 553.96, 4),
                    "reference": "bundle adjustment on about 4,300 FK points"},
             verdict="adopt", decision="Share intrinsics between the two identical webcams; keep the model pinhole.",
             evidence=["runs/claude/run-standin-1/world/static_fit.json", WF], skill="world-frame"),
        dict(id="cam-wrist", stage="B0", step="cameras", quantity="wrist camera pose and focal length",
             saw=["runs/world/overlays/native0_ep000_hand.png", "runs/world/cache/ep000_hand_median_open.png"],
             native={"source": "native", "estimate": "camera 70 mm along the hand, looking down past the fingers, "
                     "f 617 px (RealSense prior 615)", "values": {"pos_m": [0.07, -0.014, -0.01], "f_px": 617.16}},
             tool={"name": "calibrate_hand_camera", "call": "runs/claude/run-standin-1/world/hand_fit.json"},
             result={"summary": "fit after the tool offset and plate height were known: f 579.2 px; board "
                                "leave-one-hole-out 4.89 px, pad corners 4.47 px on held-out open frames; watcher "
                                "fingers 5.1 px", "values": {"f_px": 579.2}},
             error={"metric": "before: 8 marked part keypoints with the native pose; after: board leave-one-hole-out",
                    "unit": "px", "before": 84.94, "after": 4.89, "heldout": True,
                    "mm_per_px": dv["wrist_mm_per_px_median"],
                    "reference": "eye-identified marks, positions from blob centroids"},
             verdict="adopt", decision="Fit the wrist camera only after tool length and plate height, since its near "
                                       "field (fingers) and far field (board) disagree otherwise.",
             evidence=["runs/world/refine/refine_ep000_hand.json", "runs/claude/run-standin-1/world/hand_fit.json",
                       "runs/claude/run-standin-1/b0-watch/measure_b0.json", WF],
             skill="world-frame", check_ids=["B0.overlay_hand"]),
        dict(id="cam-handeye-motion", stage="B0", step="cameras", quantity="wrist camera by hand-eye from motion",
             saw=["runs/world/cache/ep000_hand_median_closed.png"],
             native={"source": "native", "estimate": "target-free AX = XB from SIFT + essential matrix over 400 frame "
                     "pairs should give the mount"},
             tool={"name": "twin.world.hand_eye_from_motion"},
             result={"summary": "45 of 400 frame pairs usable on blurry close-range footage; 2.4 deg rotation residual"},
             error={"metric": "rotation residual of the hand-eye solution", "unit": "deg", "before": None, "after": 2.4,
                    "reference": "Park-Martin AX = XB residual"},
             verdict="reject", decision="Drop motion-based hand-eye; calibrate the wrist camera on finger pads and "
                                        "board holes instead.",
             evidence=[WF], skill="world-frame"),
        dict(id="tool-length", stage="B0", step="robot_tool", quantity="hand position along the flange axis (tool length)",
             saw=["runs/world/ft_stack_hama1.png", "runs/claude/run-standin-1/world/beforeafter_hama2_heldout.png"],
             native={"source": "prior", "estimate": "the finger pads sit at the configured O_T_EE, 0.174 m from the "
                     "flange (offset 0 mm); FK + 0.174 m reproduces the logged O_T_EE to 0.24 mm",
                     "values": {"offset_mm": 0.0, "o_t_ee_m": 0.174}},
             tool={"name": "calibrate_static_cameras", "call": "fit_tool=true, runs/claude/run-standin-1/world/static_fit.json"},
             result={"summary": "one shared tool offset -22.9 mm (fits -22.9 to -24.4 mm across settings): the hand "
                                "sits on a 47.7 mm F/T stack; held-out auto pads 0.8 / 1.1 px with it, 12.3 / 11.5 px without",
                     "values": {"offset_mm": -22.9, "stack_mm": 47.7}},
             error={"metric": "hand position along its z axis; after = spread of the fitted offset across fit settings",
                    "unit": "mm", "before": 22.9, "after": 1.5, "heldout": False,
                    "reference": "bundle adjustment: arm caps fix the cameras, pads fix the hand"},
             verdict="adopt", decision="Mount the hand 22.9 mm closer in base.xml and keep the tcp site on O_T_EE; hold "
                                       "parts at the pad centre, never at tcp.",
             evidence=["runs/claude/run-standin-1/world/static_fit.json", "twin/calib/2025-01-09-14-11-58.json", WF],
             skill="world-frame", check_ids=["B0.tcp_err_mm", "B0.kp_hama1", "B0.kp_hama2"]),
        dict(id="tool-length-replay", stage="B2", step="contact", quantity="contact onset timing with the corrected hand",
             saw=["runs/claude/run-standin-1/replay/plots"],
             native={"source": "prior", "estimate": "legacy hand: pads on O_T_EE (22.9 mm too far out)",
                     "values": {"offset_mm": 0.0}},
             tool={"name": "replay_batch", "call": "legacy_hand ablation, 308 train windows"},
             result={"summary": "corrected hand vs legacy: onset error 0.433 -> 0.233 s, peak force rel. error 0.72 -> "
                                "0.36, pre-contact tracking 1.50 -> 0.64 mm",
                     "values": {"force_rel_before": 0.721, "force_rel_after": 0.358, "tracking_mm_before": 1.496,
                                "tracking_mm_after": 0.64}},
             error={"metric": "median contact-onset timing error, sim vs real", "unit": "s", "before": 0.433,
                    "after": 0.233, "reference": "logged F/T onset (|F - F_ref| > 2 N)"},
             verdict="adopt", decision="The tool-length fix explains the force and timing gap, not the over-jamming.",
             evidence=[R + "v2_train_legacyhand.json", R + "v2_train.json", RV],
             skill="replay-validate", check_ids=["B2.contact_onset_err_s", "B2.force_peak_err"]),
        dict(id="time-sync", stage="B0", step="time_sync", quantity="video-to-joint-log latency",
             saw=["docs/research/reassemble-data-notes.md"],
             native={"source": "prior", "estimate": "all streams on one clock, offset 0 frames",
                     "values": {"latency_frames": 0}},
             tool={"name": "calibrate_static_cameras", "call": "fit_latency=true (one offset per camera)"},
             result={"summary": "video lags the joint log by 0.78 frames (hama1) and 1.05 frames (hama2) = 26-35 ms, "
                                "about 4.5 mm at the median moving speed 0.135 m/s; a chamfer-vs-time-shift test gave "
                                "1.16 / 0.81 frames", "values": {"hama1_ms": 26.1, "hama2_ms": 35.0}},
             error={"metric": "timing error of the one-clock assumption (hama2); after not measured independently",
                    "unit": "ms", "before": 35.0, "after": None, "reference": "bundle-adjusted per-camera offset"},
             verdict="adopt", decision="Model latency per camera; negligible on still frames, up to 2 px on moving ones.",
             evidence=["runs/claude/run-standin-1/world/static_fit.json", WF], skill="world-frame"),
        dict(id="plate-height", stage="B0", step="object_pose", quantity="board plate-top height",
             saw=["runs/world/cache/ep000_hama1_bg41.png", "runs/world/cache/ep000_hama2_bg41.png"],
             native={"source": "prior", "estimate": "plate top at the mocap NIST_Board1 z (about 8 mm above the table)",
                     "values": {"board_z_m": 0.0079}},
             tool={"name": "calibrate_static_cameras", "call": "fit_board_z=true with 8 plate-corner marks"},
             result={"summary": "plate top 30.3 mm above the mocap board z (plate on an aluminium frame)",
                     "values": {"dz_mm": 30.3}},
             error={"metric": "plate-top height; after = corner-mark accuracy", "unit": "mm", "before": 30.3,
                    "after": 4.0, "reference": "two robot-calibrated cameras triangulating plate corners"},
             verdict="adopt", decision="Take board z from the cameras, x/y/yaw from contacts.",
             evidence=["runs/claude/run-standin-1/world/static_fit.json",
                       "runs/claude/run-standin-1/world/marks_board_ep000.json", WF], skill="world-frame"),
        dict(id="board-xy", stage="B0", step="object_pose", quantity="board pose x, y, yaw (24 gate recordings)",
             saw=["runs/claude/run-standin-1/board/gate_summary.png"],
             native={"source": "prior", "estimate": "mocap NIST_Board1 pose per recording; train recordings needed "
                     "7.0 mm median correction to agree with contacts", "values": {"mocap_shift_median_mm": 7.03}},
             tool={"name": "board_err", "call": "split gate, leave-one-part-out"},
             result={"summary": "point-to-line fit of insert seats and remove grasps (in-hand offset along the pads "
                                "marginalised): held-out across-pad error 1.23 mm (p90 3.85) vs mocap 4.08 mm (p90 8.85), "
                                "330 recording-parts", "values": {"contact_p90_mm": 3.851, "mocap_p90_mm": 8.848}},
             error={"metric": "held-out hole distance across the finger pads", "unit": "mm", "before": 4.079,
                    "after": 1.232, "heldout": True, "reference": "the left-out part's own contact events"},
             verdict="adopt", decision="Board pose comes from robot contacts; mocap and cameras only confirm.",
             evidence=["runs/claude/run-standin-1/board/board_err_gate.json", BC],
             skill="board-from-contacts", check_ids=["B0.board_err_mm"]),
        dict(id="board-stale-mocap-ep117", stage="B0", step="object_pose", quantity="board pose, gate recording 117",
             saw=["runs/claude/run-standin-1/board/2025-01-13-15-11-07_top.png"],
             native={"source": "prior", "estimate": "mocap pose file for 2025-01-13-15-11-07 (yaw -80.9 deg)",
                     "values": {"yaw_deg": -80.95}},
             tool={"name": "board_from_contacts"},
             result={"summary": "contact pose yaw -112.3 deg (sigma 0.78 / 1.28 mm, 0.26 deg); mocap holes 81.9 mm and "
                                "31.4 deg off; the pose file repeats an earlier recording's board pose",
                     "values": {"yaw_deg": -112.32, "hole_shift_mm": 81.94}},
             error={"metric": "held-out across-pad hole error in this recording", "unit": "mm", "before": 18.819,
                    "after": 1.375, "heldout": True, "reference": "leave-one-part-out contact events"},
             verdict="reject", decision="Reject the stale mocap pose; check every supplied pose against the robot's own data.",
             evidence=["runs/claude/run-standin-1/board/board_err_gate.json",
                       "runs/claude/run-standin-1/board/2025-01-13-15-11-07.json", BC],
             skill="board-from-contacts"),
        dict(id="hole-layout-mirrored", stage="B0", step="object_geometry", quantity="hole layout vs NIST CAD",
             saw=["runs/scene/experiments/stl_holes.log"],
             native={"source": "prior", "estimate": "NIST STL part centres as the layout, no reflection: 4-22 cm off",
                     "values": {"offset_mm_range": [40, 220]}},
             tool={"name": "board_contact_layout"},
             result={"summary": "all 17 contact holes match NIST laser-plate features under one mirrored rigid "
                                "transform (det -1): 0.76 mm RMS, max 1.45 mm; 8 pegs on the 75 mm grid at 0.51 mm RMS",
                     "values": {"rms_mm": 0.756, "max_mm": 1.447, "grid_rms_mm": 0.515}},
             error={"metric": "hole position vs CAD feature (before = lower end of the 4-22 cm range)", "unit": "mm",
                    "before": 40.0, "after": 0.756, "reference": "contact-fitted layout"},
             verdict="adopt", decision="Use the NIST plate with reflection for exact hole geometry, contacts for placement.",
             evidence=["runs/claude/run-standin-1/board/layout_nist.json", MJ, BC], skill="board-from-contacts"),
        dict(id="layout-method", stage="B0", step="object_pose", quantity="hole layout from robot data",
             saw=["runs/scene/experiments/layout_loo.log"],
             native={"source": "native", "estimate": "hole = median TCP at gripper release per part; per-event "
                     "scatter 9.3 mm read as hole scatter", "values": {"loo_median_mm": 9.3}},
             tool={"name": "board_contact_layout", "call": "point-to-line, in-hand offset per grasp"},
             result={"summary": "most of the scatter was in-hand offset along the pads plus mocap error; the 75 mm grid "
                                "fit drops from 2.9 mm RMS (release medians) to 0.51 mm RMS"},
             error={"metric": "8-peg fit to the NIST 75 mm grid, RMS", "unit": "mm", "before": 2.9, "after": 0.515,
                    "reference": "NIST 75 mm hole pitch"},
             verdict="revise", decision="Model the in-hand offset as a per-grasp variable instead of treating the TCP as the part.",
             evidence=["runs/scene/experiments/layout_loo.log", "runs/claude/run-standin-1/board/layout_nist.json", BT, BC],
             skill="board-from-contacts"),
        dict(id="plate-size", stage="B0", step="object_geometry", quantity="rendered plate size",
             saw=["runs/claude/run-standin-1/b0-watch/board_camera_edges.png"],
             native={"source": "prior", "estimate": "placeholder plate 400 x 400 x 25 mm in base.xml",
                     "values": {"side_mm": 400, "thickness_mm": 25}},
             tool={"name": "triangulate_marks"},
             result={"summary": "rendered plate 16 mm larger than the 384 mm NIST plate; triangulated sides 374-399 mm; "
                                "the twin now renders the NIST plate", "values": {"side_mm": 384}},
             error={"metric": "plate side length; after = corner-mark accuracy", "unit": "mm", "before": 16.0,
                    "after": 4.0, "reference": "NIST plate CAD and two-camera triangulation"},
             verdict="adopt", decision="Check the renderer against an independent projection before trusting overlays.",
             evidence=["runs/claude/run-standin-1/b0-watch/board_camera.json", VC, WF], skill="validate-checks"),
        dict(id="seat-depth", stage="B1", step="object_geometry", quantity="peg seat depth",
             saw=["runs/scene/experiments/insert_travel.log"],
             native={"source": "prior", "estimate": "seat depth = NIST plate thickness, 9 mm", "values": {"seat_mm": 9}},
             tool={"name": "query_signals", "call": "episode-0 Remove pull-out travel while the pull force lasts"},
             result={"summary": "pull-outs travel 17-23 mm: seat depth 20 +- 3 mm", "values": {"seat_mm": 20}},
             error={"metric": "seat depth; after = its sigma", "unit": "mm", "before": 11.0, "after": 3.0,
                    "reference": "Remove pull-out travel in the log"},
             verdict="adopt", decision="Seat depth from the robot's own pull-outs, not from the plate CAD.",
             evidence=[BT, RV, R + "b1_after_handfix.json"], skill="build-twin"),
        dict(id="contact-params", stage="B1", step="contact", quantity="contact softness for 0.05 mm fits",
             saw=["runs/scene/experiments/contact_params.log"],
             native={"source": "prior", "estimate": "MuJoCo default contacts: solref 0.02, solimp (0.9, 0.95, 1 mm)",
                     "values": {"solref": 0.02}},
             tool={"name": "insertion_probe", "call": "square peg 1, 0.0788 mm offset (0.024 mm rim overlap)"},
             result={"summary": "default contact slides the key 5.73 mm into the hole; solref 0.004, solimp (0.95, "
                                "0.99, 0.5 mm) stops it at 0.398 mm vs 0.376 mm geometric",
                     "values": {"depth_default_mm": 5.731, "depth_tuned_mm": 0.398, "geometric_mm": 0.376}},
             error={"metric": "jam depth minus geometric jam depth", "unit": "mm", "before": 5.355, "after": 0.022,
                    "reference": "geometry (chamfer minus overlap)"},
             verdict="adopt", decision="Fix timeconst and solimp in twin/board.py; check every fit with the probe, "
                                       "because dt/2 agreement passed the wrong default too.",
             evidence=["runs/scene/experiments/contact_params.log", MJ], skill="mujoco",
             check_ids=["B1.misaligned_jam_rate"]),
        dict(id="timestep", stage="B1", step="contact", quantity="simulation timestep for insertion",
             saw=["runs/scene/experiments/contact_params.log"],
             native={"source": "prior", "estimate": "base.xml timestep 2 ms", "values": {"dt_ms": 2}},
             tool={"name": "insertion_probe", "call": "dt 2 / 1 / 0.5 ms"},
             result={"summary": "at 2 ms the overlapping key went 1.405 mm deep; 1 and 0.5 ms agree at 0.398 mm",
                     "values": {"depth_2ms_mm": 1.405, "depth_1ms_mm": 0.398}},
             error={"metric": "jam depth minus geometric jam depth", "unit": "mm", "before": 1.029, "after": 0.022,
                    "reference": "geometry and the dt/2 rerun"},
             verdict="adopt", decision="dt = 1 ms for insertion work.",
             evidence=["runs/scene/experiments/contact_params.log", MJ], skill="mujoco",
             check_ids=["B1.dt_halving_agreement"]),
        dict(id="controller", stage="B2", step="controller", quantity="arm controller in free space",
             saw=["vendors/reassemble/catkin_reas/src/franka_reassemble/reassemble_controllers/src/cartesian_impedance_controller_damping_ratio.cpp"],
             native={"source": "native", "estimate": "replay the logged pose as the setpoint (x_d = x) on the "
                     "menagerie arm with its joint friction", "values": {"joint_friction_nm": 1.137}},
             tool={"name": "replay_controller_check", "call": "what=free_space"},
             result={"summary": "static setpoint 8.34 mm; rebuilt dynamic setpoint, no joint friction 0.97 mm; + "
                                "Lambda a 0.73 mm; nullspace potentials off 0.65 mm (303 windows); watcher 0.622 mm",
                     "values": {"static_mm": 8.335, "dynamic_mm": 0.97, "lambda_a_mm": 0.729, "ns_off_mm": 0.646}},
             error={"metric": "median EE position RMSE in free space, sim vs logged", "unit": "mm", "before": 8.335,
                    "after": 0.622, "heldout": False, "reference": "logged ee_pose on approach windows"},
             verdict="adopt", decision="Model the controller that ran (REASSEMBLE damping-ratio impedance) and rebuild "
                                       "the setpoint from pose and force; close the control gap before contact.",
             evidence=[R + "free_space_check.json", R + "free_space_v2.json", "runs/status/checks.jsonl", RV],
             skill="replay-validate", check_ids=["B2.free_space_mm"]),
        dict(id="commands-stiffness", stage="B2", step="commands", quantity="impedance stiffness K",
             saw=["vendors/reassemble/catkin_reas/src/franka_reassemble/reassemble_controllers/cfg/compliance_param.cfg"],
             native={"source": "prior", "estimate": "code default 500 N/m, 50 Nm/rad", "values": {"K_npm": 500}},
             tool={"name": "replay_controller_check", "call": "what=k_bracket"},
             result={"summary": "release jumps give K about 730 N/m (IQR 460-1,400) or 390 N/m; free-space error falls "
                                "with K (self-consistency, not evidence); force error favours softer K only because "
                                "softness hides misregistration"},
             error={"metric": "width of the K bracket left open for reconstruction", "unit": "N/m", "before": None,
                    "after": 600.0, "reference": "configured 200-800 N/m values"},
             verdict="open", decision="Keep 500 N/m and let reconstruction use the 200-800 N/m bracket; K is not "
                                      "identifiable from these logs.",
             evidence=["docs/research/reassemble-data-notes.md", R + "free_space_check.json", RV],
             skill="replay-validate"),
        dict(id="inhand-pose", stage="B2", step="hidden_state", quantity="part pose in the fingers",
             saw=["runs/claude/run-standin-1/replay/frames"],
             native={"source": "native", "estimate": "part centred between the pads and aligned with the hand "
                     "(prediction mode)", "values": {"pad_offset_mm": 0, "tilt_deg": 0}},
             tool={"name": "replay_batch", "call": "registration=inhand (label-blind, before t = 0)"},
             result={"summary": f"twin part tip vs hole axis at release: median {lat['prediction_none']['median_mm']} mm "
                                f"-> {lat['reconstruction_inhand']['median_mm']} mm over 308 train windows; bacc 0.506 "
                                "-> 0.620. Not independent: registration aims the tip at the hole at the reference "
                                "frame, so the residual is drift before release.",
                     "values": {"bacc_before": 0.506, "bacc_after": 0.62}},
             error={"metric": "median distance of the twin part tip from the hole axis at release", "unit": "mm",
                    "before": lat["prediction_none"]["median_mm"], "after": lat["reconstruction_inhand"]["median_mm"],
                    "reference": "hole axis of the contact-placed board"},
             verdict="adopt", decision="Register the unlogged in-hand pose per attempt before t = 0 (reconstruction); "
                                       "report open-loop prediction separately.",
             evidence=[D, R + "v2_train.json", R + "v2_train_prediction.json", RV],
             skill="replay-validate", check_ids=["B2.outcome_bacc", "B2.prediction_bacc"]),
        dict(id="part-friction", stage="B2", step="contact", quantity="part friction against tight holes",
             saw=["runs/claude/run-standin-1/replay/videos"],
             native={"source": "prior", "estimate": "part friction 0.30", "values": {"mu": 0.30}},
             tool={"name": "replay_batch", "call": "stratified subset, 69 failures + 69 successes"},
             result={"summary": "released pegs stuck 2-4 mm deep at 6-12 deg where real ones fell in; 0.15 seats them "
                                "(0.10: bacc 0.616, 0.20: 0.609)", "values": {"mu": 0.15}},
             error={"metric": "balanced error rate (1 - bacc) on the stratified train subset", "unit": "fraction",
                    "before": 0.42, "after": 0.355, "reference": "real seat/jam labels"},
             verdict="adopt", decision="Friction 0.15, chosen on train only.",
             evidence=[R + "screen_newhand.json", RV, MJ], skill="replay-validate"),
        dict(id="soft-layer-clearance", stage="B2", step="contact", quantity="does the soft-contact layer eat clearance",
             saw=["runs/claude/run-standin-1/replay/screen_newhand.json"],
             native={"source": "native", "estimate": "a 0.5 mm solimp width could take up 0.05-0.25 mm clearance and "
                     "cause over-jamming", "values": {"solimp_width_mm": 0.5}},
             tool={"name": "replay_batch", "call": "solimp width 0.1 mm"},
             result={"summary": "bacc 0.529 -> 0.551: no real change; with margin 0 the soft layer acts only at "
                                "negative distance, so it does not take up clearance"},
             error={"metric": "balanced error rate (1 - bacc), stratified subset", "unit": "fraction", "before": 0.471,
                    "after": 0.449, "reference": "real seat/jam labels"},
             verdict="reject", decision="Not the cause; keep the B1 contact parameters.",
             evidence=[R + "screen_newhand.json", RV, MJ], skill="replay-validate"),
        dict(id="effective-clearance", stage="B2", step="contact", quantity="effective hole clearance",
             saw=["runs/claude/run-standin-1/replay/clearance_grid.json"],
             native={"source": "prior", "estimate": "NIST nominal clearances 0.05-0.25 mm per side",
                     "values": {"widen_mm": 0.0}},
             tool={"name": "replay_batch", "call": "widen 0-0.5 mm per side, 5-fold recording CV"},
             result={"summary": "one global +0.2 mm holds out at bacc 0.645 vs 0.620; per-part widening unstable "
                                "across folds (0.620)", "values": {"widen_mm": 0.2}},
             error={"metric": "balanced error rate (1 - bacc), recording-level CV", "unit": "fraction",
                    "before": 0.38, "after": 0.355, "heldout": True, "reference": "real seat/jam labels"},
             verdict="adopt", decision="Freeze +0.2 mm (params_v3) and read it as unmodelled self-alignment, not geometry.",
             evidence=[R + "clearance_cv.json", RV], skill="replay-validate"),
        dict(id="operator-aborts", stage="B2", step="validation", quantity="what the real failures are",
             saw=["runs/claude/run-standin-1/replay/failure_split.json"],
             native={"source": "native", "estimate": "the 69 real peg failures are physical jams the twin must reproduce",
                     "values": {"real_failures": 69}},
             tool={"name": "label-blind failure split", "call": "runs/claude/run-standin-1/replay/failure_split.json"},
             result={"summary": f"{fsplit['jam_signature']} of {fsplit['real_failures']} show a physical-jam signature "
                                f"(the twin jams all {fsplit['jam_signature_jammed_in_twin']}); {fsplit['never_released']} "
                                "were never released; the rest are short-force aborts or misses. A permissive twin "
                                "scores 0.651 vs 0.620, so the per-attempt negative control fails.",
                     "values": {"jam_signature": fsplit["jam_signature"], "never_released": fsplit["never_released"]}},
             error={"metric": "real failures that contact physics can reproduce", "unit": "count",
                    "before": fsplit["real_failures"], "after": fsplit["jam_signature"],
                    "reference": "force, descent and release signals of each attempt"},
             verdict="escalate", decision="Escalated: the practice gate becomes distribution-level with a permissive "
                                          "negative control (validation v10); per-attempt accuracy stays as the "
                                          "reported stricter claim.",
             evidence=[D, R + "failure_split.json", R + "v2_train_permissive.json", "playbook/validation.yaml", RV],
             skill="replay-validate", check_ids=["B2.dist_negative_control", "B2.negative_control_drop"]),
        dict(id="edge-overlay-blind", stage="B0", step="validation", quantity="edge-overlay check as camera acceptance",
             saw=["runs/claude/run-standin-1/b0-watch/b0_known_error_curves.png"],
             native={"source": "prior", "estimate": "B0 v3: median edge distance <= 3 px certifies a camera",
                     "values": {"threshold_px": 3}},
             tool={"name": "b0_known_error"},
             result={"summary": "fails only at 50 mm lateral shift or 2 deg turn; a robot rendered 33 s off still "
                                "scores 2.2-3.0 px and passes; a directional chamfer fails at 10 mm (hama1) / 5 mm "
                                "(hama2); held-out keypoints replaced it (v5)"},
             error={"metric": "smallest lateral camera shift that fails the check (hama1)", "unit": "mm",
                    "before": 50.0, "after": 10.0, "reference": "known-error perturbations, both signs"},
             verdict="revise", decision="Gate cameras on held-out keypoints and two-view consistency; keep edges as a diagnostic.",
             evidence=["runs/claude/run-standin-1/b0-watch/b0_known_error.json", VC, "playbook/validation.yaml"],
             skill="validate-checks", check_ids=["B0.overlay_hama1", "B0.kp_hama1"]),
        dict(id="replay-dt-half", stage="B2", step="integrity", quantity="replay outcomes at half the timestep",
             saw=["runs/claude/run-standin-1/replay/dt_study.json"],
             native={"source": "native", "estimate": "replays at 1 ms are converged (48/48 agreed on the first subset)",
                     "values": {"agree_first_subset": 48}},
             tool={"name": "replay_batch", "call": "timestep 0.0005"},
             result={"summary": "288/308 outcomes unchanged at dt/2 (93.5%, below 95%); the 20 flips are chaotic "
                                "borderline cases: 0.5 vs 0.25 ms agree on 14 of them"},
             error={"metric": "fraction of replay outcomes that change at dt/2", "unit": "fraction",
                    "before": 0.0, "after": round(20 / 308, 3), "reference": "same windows at dt/2"},
             verdict="open", decision="Report dt/2 agreement with every B2 number; treat borderline keys as uncertain.",
             evidence=[R + "v2_train_dt05.json", R + "dt_study.json", RV], skill="replay-validate"),
    ]
    for r in recs:
        r.update(brain=BRAIN, run_id=RUN_ID, by=CALLER, backfilled=True, decided_on=DAY)
        if r["id"] in LABELS:
            r["label"] = LABELS[r["id"]]
    return recs


LABELS = {"cam-hama1-pose": "Static camera 1 pose", "cam-hama2-pose": "Static camera 2 pose",
          "cam-wrist": "Wrist camera pose", "tool-length": "Tool length (hand on flange)",
          "plate-height": "Board plate-top height", "board-xy": "Board x, y (24 held-out recs)",
          "board-stale-mocap-ep117": "Board pose, stale mocap file", "hole-layout-mirrored": "Hole layout vs CAD (mirrored)",
          "seat-depth": "Peg seat depth", "controller": "Arm tracking in free space",
          "inhand-pose": "Part in the fingers at release", "contact-params": "Contact softness, 0.05 mm fit",
          "tool-length-replay": "Contact onset timing"}


CHART_IDS = ["cam-hama1-pose", "cam-hama2-pose", "cam-wrist", "tool-length", "plate-height", "board-xy",
             "board-stale-mocap-ep117", "hole-layout-mirrored", "seat-depth", "controller", "inhand-pose",
             "contact-params", "tool-length-replay"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dv = derived()
    (OUT / "derived.json").write_text(json.dumps(dv, indent=1) + "\n")
    path = RUN / "decisions.jsonl"
    have = set()
    if path.exists():
        have = {json.loads(ln).get("id") for ln in path.read_text().splitlines() if ln.strip()}
    n_new = 0
    for r in records(dv):
        if r["id"] in have:
            continue
        res = call("log_decision", r, caller=CALLER)
        if "error" in res:
            raise SystemExit(f"{r['id']}: {res['error']}")
        n_new += 1
    s = call("decision_summary", {"brain": BRAIN, "run_id": RUN_ID, "chart": True, "chart_ids": CHART_IDS,
                                  "title": "Reconstruction error: native estimate vs native estimate plus tools",
                                  "subtitle": "Claude stand-in, REASSEMBLE episode 0; board rows: 24 held-out recordings"},
             caller=CALLER)
    print(json.dumps({"new_records": n_new, "total": s.get("n"), "by_step": s.get("by_step"),
                      "chart": s.get("chart"), "chart_rows": s.get("chart_rows"), "error": s.get("error")}, indent=1))


if __name__ == "__main__":
    main()
