"""SO-101 track tools (stage T2): calibration, keypoint and two-view checks, objects, grasp check, scene check, replay.

Deterministic measurement tools a watcher can rerun from files. Code: twin/so101_*.py. Data: data/so101 (read as data).
Episodes are passed as `ep` / `eps` (never `episode`): harness.registry's exam guard reads `episode(s)` against
REASSEMBLE's splits.json, whose exam ids (9, 10, 16, ...) collide with SO-101 episode numbers.
Splits: playbook/splits_so101.json (0-34 train, 35-49 gate). Tools that take episodes check them against `split`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from harness.registry import REPO, tool

os.environ.setdefault("MUJOCO_GL", "glfw")
RUN = REPO / "runs" / "claude" / "run-so101-1"


def _p(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if str(p).startswith(str(REPO)) else str(p)


def _eps(eps, split: str) -> list[int]:
    from twin import so101_data as sd
    s = sd.splits()
    if split not in ("train", "gate"):
        raise ValueError("split must be 'train' or 'gate'")
    allowed = set(s[split])
    eps = list(s[split]) if eps in (None, "all", []) else [int(e) for e in eps]
    bad = [e for e in eps if e not in allowed]
    if bad:
        raise ValueError(f"episodes {bad} are not in the {split} split (playbook/splits_so101.json)")
    return eps


def _json(obj, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return _rel(path)


# ---------------------------------------------------------------------------------------------- calibration
@tool(
    "so101_calibrate",
    "SO-101: fit both cameras (pose, focal), the per-joint map from dataset units to MJCF radians and a latency per "
    "camera, from the robot's own outline on train frames (render the twin at the logged joints, match the real dark "
    "outline along each rendered outline normal, bundle-adjust; radii shrink). Optional coarse IoU seed first. Slow "
    "(5-15 min). Writes a calib JSON.",
    {"type": "object", "required": ["seed", "out"], "properties": {
        "seed": {"type": "string", "description": "calib JSON to start from (cameras, joint_map, latency_frames)"},
        "out": {"type": "string", "description": "output calib JSON path"},
        "coarse": {"type": "boolean", "description": "run the IoU seed alignment first (default false)"},
        "radii": {"type": "array", "items": {"type": "number"}, "description": "search radii per round, px (default [30,18,10,6])"},
        "per_ep": {"type": "integer", "description": "fit frames per train episode (default 6)"},
        "fit_joints": {"type": "boolean"}, "fit_latency": {"type": "boolean"}, "free_pp": {"type": "boolean"},
        "teal_w": {"type": "number", "description": "weight of the teal-cap keypoint residuals (0 = outline only; default 6)"},
        "tv_w": {"type": "number", "description": "weight of the brick two-view residuals at frame 0 (default 4)"},
        "fit_ext": {"type": "boolean", "description": "fit the gripper extension along its axis (default true)"},
        "k1": {"type": "boolean", "description": "fit radial k1 per camera (default true)"},
        "step": {"type": "integer", "description": "outline subsampling in px (default 4)"}}},
)
def so101_calibrate(seed: str, out: str, coarse: bool = False, radii=None, per_ep: int = 5, fit_joints: bool = True,
                    fit_latency: bool = True, free_pp: bool = False, teal_w: float = 6.0, tv_w: float = 4.0,
                    fit_ext: bool = True, k1: bool = True, step: int = 4) -> dict:
    from twin import so101_camcal as cc, so101_data as sd, so101_robot as sr
    cams, jm, lat = cc.load_calib(_p(seed))
    tr = sd.splits()["train"]
    cdir = _p(out).parent
    ig = {c: cc.ignore_mask(c, tr[::3], cache=cdir / f"ignore_{c}.png") for c in sd.CAMS}
    logs = []
    if coarse:
        items = cc.sample_frames(tr[1::4], 1) + cc.sample_frames(tr[2::8], 3)
        for c in sd.CAMS:
            cams[c] = cc.coarse_align(cams[c], jm, lat[c], cc.FrameSet(c, items, ig[c]), log=logs.append)
    items = cc.sample_frames(tr, per_ep, phase=0)
    sets = {c: cc.FrameSet(c, items, ig[c]) for c in sd.CAMS}
    p = cc.Params(cams, jm, lat, 0.0, cc.load_ext(_p(seed)))
    tv = cc.brick_twoview(tr) if tv_w > 0 else None
    p, hist = cc.fit(p, sets, radii=tuple(radii or (14, 8, 5)), free_pp=free_pp, fit_joints=fit_joints,
                     fit_latency=fit_latency, log=logs.append, teal_w=teal_w, twoview=tv, tv_w=tv_w, fit_ext=fit_ext,
                     fit_cap=False, fit_k1=k1, loss="soft_l1", step=step)
    d = {"cameras": {k: c.to_dict() for k, c in p.cams.items()}, "joint_map": p.jm.to_dict(), "latency_frames": p.lat,
         "teal_cap_m": p.cap, "gripper_ext_m": p.ext, "teal_w": teal_w, "tv_w": tv_w, "fit_frames": items, "history": hist, "seed": seed, "log": logs,
         "source": "outline bundle adjustment (twin/so101_camcal.py::fit), train frames only"}
    return {"calib": _json(d, _p(out)), "history": hist,
            "joint_scale_deg_per_unit": np.degrees(p.jm.scale).round(4).tolist(),
            "joint_offset_deg": np.degrees(p.jm.offset).round(2).tolist(), "latency_frames": p.lat,
            "f_px": {k: c.f for k, c in p.cams.items()}}


@tool(
    "so101_outline_score",
    "SO-101: held-out outline residual of a calibration: median |distance| (px) between the rendered arm outline and "
    "the real dark outline on frames the fit never used (phase-shifted sampling of the chosen split). Diagnostic only "
    "(an outline metric can pass a wrong pose; gate on keypoints).",
    {"type": "object", "required": ["calib"], "properties": {
        "calib": {"type": "string"}, "split": {"type": "string", "description": "train (default) or gate"},
        "eps": {"type": "array", "items": {"type": "integer"}}, "per_ep": {"type": "integer"},
        "phase": {"type": "integer", "description": "frame offset from the fit sampling (default 11)"},
        "radius": {"type": "number"}}},
)
def so101_outline_score(calib: str, split: str = "train", eps=None, per_ep: int = 3, phase: int = 11,
                        radius: float = 10.0) -> dict:
    from twin import so101_camcal as cc, so101_data as sd
    cams, jm, lat = cc.load_calib(_p(calib))
    eps = _eps(eps, split)
    tr = sd.splits()["train"]
    ig = {c: cc.ignore_mask(c, tr[::3], cache=_p(calib).parent / f"ignore_{c}.png") for c in sd.CAMS}
    items = cc.sample_frames(eps, per_ep, phase=phase)
    sets = {c: cc.FrameSet(c, items, ig[c]) for c in sd.CAMS}
    out = cc.score_outline(cc.Params(cams, jm, lat, cc.load_cap(_p(calib)), cc.load_ext(_p(calib))), sets, radius)
    return {"calib": calib, "split": split, "frames": len(items), "per_camera": out}


@tool(
    "so101_overlay",
    "SO-101: draw the twin's rendered arm outline (green; moving jaw magenta) over real frames of both cameras for "
    "the given (ep, frame) items. Writes one PNG grid (rows = cameras).",
    {"type": "object", "required": ["calib", "items", "out"], "properties": {
        "calib": {"type": "string"}, "out": {"type": "string"},
        "items": {"type": "array", "items": {"type": "object"}, "description": "[{ep, frame}]"},
        "split": {"type": "string"}, "scale": {"type": "number", "description": "tile scale (default 0.5)"},
        "objects": {"type": "string", "description": "objects JSON: also draw the box outline and brick (frame-0 pose)"}}},
)
def so101_overlay(calib: str, items: list, out: str, split: str = "train", scale: float = 0.5, objects: str = "") -> dict:
    import cv2
    from twin import so101_camcal as cc, so101_data as sd, so101_objects as so, so101_robot as sr
    cams, jm, lat = cc.load_calib(_p(calib))
    _eps([it["ep"] for it in items], split)
    obj = json.loads(_p(objects).read_text()) if objects else None
    rend = cc.SegRenderer(finger_ext=cc.load_fext(_p(calib)))
    sr.set_ext(rend.m, cc.load_ext(_p(calib)))
    rows = []
    for cn in sd.CAMS:
        tiles = []
        for it in items:
            ep, i = int(it["ep"]), int(it["frame"])
            img = sd.frame(ep, cn, i)
            q = jm.q(cc.interp_units(sd.signals(ep)["state"], np.array([i + lat[cn]])))[0]
            b, _ = rend.render(q, cams[cn], depth=False)
            o = cc.overlay(img, b, rend.jaw)
            if obj:
                bx = obj["box"]
                for h in (0.0, bx["height"]):
                    P = so.rect_outline(bx["center"][0], bx["center"][1], bx["yaw"], bx["outer"][0], bx["outer"][1],
                                       bx["center"][2] + h, 40)
                    uv, _ = cams[cn].project(P)
                    cv2.polylines(o, [np.round(uv).astype(np.int32)], True, (255, 255, 0), 1)
                st = obj.get("starts", {}).get(str(ep))
                if st and i == 0:
                    X = so.brick_corners(*st["pos"], st["yaw"])
                    uv, _ = cams[cn].project(X)
                    hull = cv2.convexHull(np.round(uv).astype(np.int32))
                    cv2.polylines(o, [hull], True, (255, 255, 0), 1)
            cv2.putText(o, f"{cn} ep{ep} f{i}", (8, 22), 0, 0.7, (0, 255, 255), 2)
            tiles.append(cv2.resize(o, None, fx=scale, fy=scale))
        rows.append(np.concatenate(tiles, 1))
    rend.close()
    p = _p(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), np.concatenate(rows, 0))
    return {"png": _rel(p)}


# ---------------------------------------------------------------------------------------------- keypoints
@tool(
    "so101_keypoints",
    "SO-101: the robot keypoints a person can mark, with descriptions, and their projections for (ep, frame, cam) "
    "items under a calibration (to check marks against, after marking). Also writes a legend PNG rendering the keys "
    "on the model.",
    {"type": "object", "properties": {"calib": {"type": "string"},
                                      "items": {"type": "array", "items": {"type": "object"}},
                                      "split": {"type": "string"}}},
)
def so101_keypoints(calib: str = "", items=None, split: str = "train") -> dict:
    from twin import so101_camcal as cc, so101_data as sd, so101_robot as sr
    out = {"keys": sr.KEY_DESCR}
    if calib and items:
        cams, jm, lat = cc.load_calib(_p(calib))
        _eps([it["ep"] for it in items], split)
        fk = sr.FK(sr.load_model(ext=cc.load_ext(_p(calib))))
        fext = cc.load_fext(_p(calib))
        proj = []
        for it in items:
            ep, i, cn = int(it["ep"]), int(it["frame"]), it["cam"]
            q = jm.q(cc.interp_units(sd.signals(ep)["state"], np.array([i + lat[cn]])))
            K = fk.keypoints(q, cap=cc.load_cap(_p(calib)), finger_ext=fext)
            proj.append({"ep": ep, "frame": i, "cam": cn,
                         "uv": {k: cams[cn].project(v[0])[0].round(1).tolist() for k, v in K.items()}})
        out["projections"] = proj
    return out


def _load_marks(marks) -> list[dict]:
    if isinstance(marks, str):
        marks = json.loads(_p(marks).read_text())
        if isinstance(marks, dict):
            marks = marks.get("marks", [])
    return [dict(m) for m in marks]


@tool(
    "so101_keypoint_check",
    "SO-101 T2.kp_px / T2.two_view_px: score a calibration on marked robot keypoints. Each mark {ep, frame, cam, key, "
    "uv}; keys from so101_keypoints. Returns the median reprojection error (px) per camera and pooled, and for every "
    "(ep, frame, key) marked in both cameras the per-view residual after triangulating the two marks (px), plus the "
    "triangulated point's distance to the FK keypoint (mm). Uses the calibration's latency.",
    {"type": "object", "required": ["calib", "marks"], "properties": {
        "calib": {"type": "string"}, "marks": {"description": "marks JSON path or list"},
        "split": {"type": "string", "description": "split the marked episodes must belong to (default gate)"},
        "out": {"type": "string"}}},
)
def so101_keypoint_check(calib: str, marks, split: str = "gate", out: str = "") -> dict:
    from twin import so101_camcal as cc, so101_data as sd, so101_robot as sr
    cams, jm, lat = cc.load_calib(_p(calib))
    cap = cc.load_cap(_p(calib))
    mk = _load_marks(marks)
    _eps(sorted({int(m["ep"]) for m in mk}), split)
    fk = sr.FK(sr.load_model(ext=cc.load_ext(_p(calib))))
    fext = cc.load_fext(_p(calib))
    rows = []
    for m in mk:
        ep, i, cn, key = int(m["ep"]), int(m["frame"]), m["cam"], m["key"]
        q = jm.q(cc.interp_units(sd.signals(ep)["state"], np.array([i + lat[cn]])))
        X = fk.keypoints(q, [key], cap=cap, finger_ext=fext)[key][0]
        uv, _ = cams[cn].project(X)
        e = float(np.linalg.norm(uv - np.array(m["uv"], float)))
        rows.append({"ep": ep, "frame": i, "cam": cn, "key": key, "err_px": round(e, 2), "pred": uv.round(1).tolist(),
                     "mark": m["uv"]})
    per = {}
    for cn in sd.CAMS:
        es = [r["err_px"] for r in rows if r["cam"] == cn]
        per[cn] = {"n": len(es), "median_px": float(np.median(es)) if es else None,
                   "p90_px": float(np.percentile(es, 90)) if es else None}
    allv = [r["err_px"] for r in rows]
    # two-view
    tv = []
    by = {}
    for m in mk:
        by.setdefault((int(m["ep"]), int(m["frame"]), m["key"]), {})[m["cam"]] = np.array(m["uv"], float)
    for (ep, i, key), d in by.items():
        if len(d) < 2:
            continue
        cl = [cams[c] for c in sd.CAMS]
        uvs = [d[c] for c in sd.CAMS]
        Xt = cc.triangulate(cl, uvs)
        res = [float(np.linalg.norm(c.project(Xt)[0] - u)) for c, u in zip(cl, uvs)]
        # FK at the mean latency of both cameras
        q = jm.q(cc.interp_units(sd.signals(ep)["state"], np.array([i + np.mean(list(lat.values()))])))
        Xk = fk.keypoints(q, [key], cap=cap, finger_ext=fext)[key][0]
        tv.append({"ep": ep, "frame": i, "key": key, "resid_px": [round(r, 2) for r in res],
                   "tri_vs_fk_mm": round(float(np.linalg.norm(Xt - Xk) * 1000), 1)})
    tvr = [r for t in tv for r in t["resid_px"]]
    res = {"calib": calib, "n_marks": len(rows), "per_camera": per,
           "kp_px_median_pooled": float(np.median(allv)) if allv else None,
           "two_view_px_median": float(np.median(tvr)) if tvr else None, "n_two_view_points": len(tv),
           "tri_vs_fk_mm_median": float(np.median([t["tri_vs_fk_mm"] for t in tv])) if tv else None,
           "rows": rows, "two_view": tv}
    if out:
        res["json"] = _json(res, _p(out))
    return res


@tool(
    "so101_mark_sheet",
    "SO-101: help a person mark keypoints without seeing any projection: for each (ep, frame, cam) item write a 4x "
    "zoomed crop around `center` (or the whole frame at 1x) with a labelled 10 px grid, and list colour-blob "
    "candidates (teal jaw caps, white horn) with code-computed centroids. Mark by picking a candidate or reading uv off "
    "the grid.",
    {"type": "object", "required": ["items", "out_dir"], "properties": {
        "items": {"type": "array", "items": {"type": "object"}, "description": "[{ep, frame, cam, center?: [u,v], half?: 40}]"},
        "out_dir": {"type": "string"}, "split": {"type": "string"}}},
)
def so101_mark_sheet(items: list, out_dir: str, split: str = "gate") -> dict:
    import cv2
    from twin import so101_data as sd
    _eps(sorted({int(it["ep"]) for it in items}), split)
    od = _p(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    res = []
    for it in items:
        ep, i, cn = int(it["ep"]), int(it["frame"]), it["cam"]
        img = sd.frame(ep, cn, i)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        teal = ((hsv[..., 0] >= 75) & (hsv[..., 0] <= 100) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 50)).astype(np.uint8)
        cands = []
        n, lab, st, cen = cv2.connectedComponentsWithStats(teal)
        for k in range(1, n):
            if st[k, cv2.CC_STAT_AREA] >= 6:
                cands.append({"id": len(cands), "kind": "teal", "uv": cen[k].round(2).tolist(), "area": int(st[k, cv2.CC_STAT_AREA])})
        if "center" in it:
            u, v = [int(round(x)) for x in it["center"]]
            h = int(it.get("half", 40))
            x0, y0 = max(0, u - h), max(0, v - h)
            crop = img[y0:v + h, x0:u + h]
            z = 4
            big = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
            for gx in range((x0 // 10 + 1) * 10, x0 + crop.shape[1], 10):
                cv2.line(big, ((gx - x0) * z, 0), ((gx - x0) * z, big.shape[0]), (200, 200, 0), 1)
                cv2.putText(big, str(gx), ((gx - x0) * z + 2, 12), 0, 0.35, (0, 255, 255), 1)
            for gy in range((y0 // 10 + 1) * 10, y0 + crop.shape[0], 10):
                cv2.line(big, (0, (gy - y0) * z), (big.shape[1], (gy - y0) * z), (200, 200, 0), 1)
                cv2.putText(big, str(gy), (2, (gy - y0) * z - 2), 0, 0.35, (0, 255, 255), 1)
            for c in cands:
                cu, cv_ = c["uv"]
                if x0 <= cu < x0 + crop.shape[1] and y0 <= cv_ < y0 + crop.shape[0]:
                    cv2.putText(big, f"#{c['id']}", (int((cu - x0) * z) + 4, int((cv_ - y0) * z)), 0, 0.5, (255, 0, 255), 1)
        else:
            x0 = y0 = 0
            big = img.copy()
        p = od / f"ep{ep:03d}_f{i:03d}_{cn}.png"
        cv2.imwrite(str(p), big)
        res.append({"ep": ep, "frame": i, "cam": cn, "png": _rel(p), "origin": [x0, y0], "zoom": 4 if "center" in it else 1,
                    "candidates": cands})
    return {"sheets": res}


# ---------------------------------------------------------------------------------------------- objects
@tool(
    "so101_brick_pose",
    "SO-101: the LEGO 2x4 brick's pose (body centre x, y, z in the robot base frame, yaw) at one frame, fitted to the "
    "red/pink colour blob in both cameras (projected outline of body + stud block, IoU). With table_z (or an objects "
    "JSON) z is fixed to table + 4.8 mm.",
    {"type": "object", "required": ["calib", "ep", "frame"], "properties": {
        "calib": {"type": "string"}, "ep": {"type": "integer"}, "frame": {"type": "integer"},
        "objects": {"type": "string"}, "free_z": {"type": "boolean"}, "split": {"type": "string"}}},
)
def so101_brick_pose(calib: str, ep: int, frame: int, objects: str = "", free_z: bool = False, split: str = "train") -> dict:
    from twin import so101_camcal as cc, so101_objects as so
    cams, jm, lat = cc.load_calib(_p(calib))
    _eps([ep], split)
    tz = None if free_z or not objects else json.loads(_p(objects).read_text())["table_z"]
    return so.brick_pose_at(cams, ep, frame, tz)


@tool(
    "so101_objects_fit",
    "SO-101: fit the static objects from train frames with a calibration: table height (median of free-z brick fits at "
    "frame 0), the box footprint from the orange tape outline (rectangle on the table plane, both views, IoU), the box "
    "rim height (strongest image edge along the projected rim, 1-D search), and brick start poses (frame 0, z on the "
    "table) for the requested split. Writes an objects JSON.",
    {"type": "object", "required": ["calib", "out"], "properties": {
        "calib": {"type": "string"}, "out": {"type": "string"},
        "box_seed": {"type": "array", "items": {"type": "number"}, "description": "[cx, cy, yaw, lx, ly] native guess (m, rad)"},
        "rim_inset": {"type": "number", "description": "rim rectangle inset from the tape centreline (m, default = tape width)"},
        "box_height": {"type": "number", "description": "override the rim search with a value (m)"},
        "starts_split": {"type": "string", "description": "train (default) or gate: whose brick starts to add"}}},
)
def so101_objects_fit(calib: str, out: str, box_seed=None, rim_inset: float | None = None, box_height: float | None = None,
                      starts_split: str = "train") -> dict:
    from twin import so101_camcal as cc, so101_objects as so
    cams, jm, lat = cc.load_calib(_p(calib))
    res = so.fit_static_objects(cams, box_seed=box_seed, rim_inset=rim_inset, box_height=box_height)
    res["starts"] = so.brick_starts(cams, _eps(None, starts_split), res["table_z"])
    res["starts_split"] = starts_split
    res["calib"] = calib
    return {"objects": _json(res, _p(out)), "table_z": res["table_z"], "box": res["box"],
            "table_z_spread_mm": res.get("table_z_spread_mm"), "n_starts": len(res["starts"]),
            "start_iou_median": float(np.median([s["iou"] for s in res["starts"].values()]))}


@tool(
    "so101_add_starts",
    "SO-101: add brick start poses (frame 0, from both cameras, z on the fitted table) for a split to a copy of an "
    "objects JSON. Use it to freeze gate starts with the frozen tools. Since the frozen gate run (2026-10-25) a brick "
    "seen by one camera only is placed from that view on the table plane; such starts have views == [that camera].",
    {"type": "object", "required": ["calib", "objects", "split", "out"], "properties": {
        "calib": {"type": "string"}, "objects": {"type": "string"}, "split": {"type": "string"}, "out": {"type": "string"}}},
)
def so101_add_starts(calib: str, objects: str, split: str, out: str) -> dict:
    from twin import so101_camcal as cc, so101_objects as so
    cams, jm, lat = cc.load_calib(_p(calib))
    obj = json.loads(_p(objects).read_text())
    st = so.brick_starts(cams, _eps(None, split), obj["table_z"])
    obj["starts"] = {**obj.get("starts", {}), **st}
    return {"objects": _json(obj, _p(out)), "added": sorted(st, key=int),
            "iou_median": float(np.median([s["iou"] for s in st.values()]))}


@tool(
    "so101_grasp_object",
    "SO-101 T2.grasp_object_mm: for each episode, the brick centre reconstructed from both cameras just before the "
    "grasp (the frame where the leader starts closing the gripper; falls back to frame 0 if the brick is occluded) vs "
    "the gripper pad centre by forward kinematics at the grasp frame (first frame the follower gripper is held open by "
    "the brick). Returns per-episode distances and the median (mm).",
    {"type": "object", "required": ["calib", "objects"], "properties": {
        "calib": {"type": "string"}, "objects": {"type": "string"}, "split": {"type": "string"},
        "eps": {"type": "array", "items": {"type": "integer"}}, "out": {"type": "string"}}},
)
def so101_grasp_object(calib: str, objects: str, split: str = "gate", eps=None, out: str = "") -> dict:
    from twin import so101_camcal as cc, so101_data as sd, so101_objects as so, so101_replay as rp, so101_robot as sr, so101_scene as ss
    cams, jm, lat = cc.load_calib(_p(calib))
    obj = json.loads(_p(objects).read_text())
    eps = _eps(eps, split)
    fext = cc.load_fext(_p(calib))
    m = ss.compile_scene(ss.build(cfg={"gripper_ext": cc.load_ext(_p(calib)), "finger_ext": fext}), finger_ext=fext)
    fk = sr.FK(m)
    gb = m.body("gripper").id
    pad = m.site_pos[m.site("grip/pad_center").id]
    rows = []
    for ep in eps:
        gf = rp.grasp_frame(ep)
        cs = rp.close_start(ep)
        if gf is None:
            rows.append({"ep": ep, "ok": False, "why": "no grasp found in the gripper signal"})
            continue
        b = so.brick_pose_at(cams, ep, cs, obj["table_z"])
        b0 = so.brick_pose_at(cams, ep, 0, obj["table_z"])
        used = "close_start"
        if not b.get("ok") or b.get("iou", 0) < 0.5 * b0.get("iou", 0) or len(b.get("views", [])) < 2:
            b, used = b0, "frame0"
        q = jm.q(sd.signals(ep)["state"][gf][None])
        Xp = fk.points(q, np.array([gb]), pad[None])[0, 0]
        dist = float(np.linalg.norm(np.array(b["pos"]) - Xp) * 1000)
        rows.append({"ep": ep, "ok": True, "grasp_frame": gf, "close_start": cs, "brick_from": used,
                     "brick_pos": np.round(b["pos"], 4).tolist(), "brick_iou": round(b["iou"], 3),
                     "pad_center": Xp.round(4).tolist(), "dist_mm": round(dist, 1),
                     "dxyz_mm": ((np.array(b["pos"]) - Xp) * 1000).round(1).tolist()})
    ds = [r["dist_mm"] for r in rows if r.get("ok")]
    res = {"calib": calib, "objects": objects, "split": split, "n": len(ds),
           "median_mm": float(np.median(ds)) if ds else None, "p90_mm": float(np.percentile(ds, 90)) if ds else None,
           "rows": rows}
    if out:
        res["json"] = _json(res, _p(out))
    return res


# ---------------------------------------------------------------------------------------------- scene + replay
def _twin(calib: str, objects: str, cfg: dict | None):
    from twin import so101_replay as rp
    c = json.loads(_p(calib).read_text())
    o = json.loads(_p(objects).read_text())
    cfg = {**o.get("controller", {}), **(cfg or {})}
    return rp.Twin(c, o, cfg), o


@tool(
    "so101_scene_check",
    "SO-101 scene check (mujoco skill): build the twin with an episode's brick start, write the MJCF, reload it, settle "
    "under gravity with the arm held at its logged start; report resting and transient brick penetration (mm), brick "
    "drift (mm) and the same at dt/2.",
    {"type": "object", "required": ["calib", "objects", "ep"], "properties": {
        "calib": {"type": "string"}, "objects": {"type": "string"}, "ep": {"type": "integer"}, "split": {"type": "string"},
        "settle_s": {"type": "number"}, "cfg": {"type": "object"}, "out_dir": {"type": "string"}}},
)
def so101_scene_check(calib: str, objects: str, ep: int, split: str = "train", settle_s: float = 1.0, cfg=None,
                      out_dir: str = "") -> dict:
    import mujoco
    from twin import so101_data as sd, so101_scene as ss
    _eps([ep], split)
    tw, o = _twin(calib, objects, cfg)
    od = _p(out_dir) if out_dir else RUN / "scene"
    st = o["starts"][str(ep)]
    spec = ss.build(json.loads(_p(calib).read_text()), o, tw.cfg, brick_pose=st)
    xml = od / f"scene_ep{ep:03d}.xml"
    ss.save(spec, xml)
    m2 = mujoco.MjModel.from_xml_path(str(xml))
    reload_ok = bool(m2.ngeom == tw.m.ngeom and np.allclose(m2.body_mass, tw.m.body_mass, atol=1e-6)
                     and np.allclose(m2.geom_size, tw.m.geom_size, atol=1e-6) and np.allclose(m2.actuator_gainprm, tw.m.actuator_gainprm))
    res = {"xml": _rel(xml), "reload_ok": reload_ok}
    q0 = tw.jm.q(sd.signals(ep)["state"][0])
    for tag, dt in (("dt", tw.cfg["timestep"]), ("dt_half", tw.cfg["timestep"] / 2)):
        tw.set_timestep(dt)
        tw.reset(q0, st["pos"], st["yaw"])
        p0 = tw.brick()[0]
        pen_max = 0.0
        for _ in range(int(round(settle_s / dt))):
            mujoco.mj_step(tw.m, tw.d)
            for c in tw.d.contact[:tw.d.ncon]:
                pen_max = max(pen_max, -c.dist)
        pen_rest = max([-c.dist for c in tw.d.contact[:tw.d.ncon]] + [0.0])
        p1 = tw.brick()[0]
        res[tag] = {"timestep": dt, "final_penetration_mm": round(pen_rest * 1000, 4),
                    "transient_penetration_mm": round(pen_max * 1000, 4),
                    "brick_drift_mm": round(float(np.linalg.norm(p1 - p0) * 1000), 3),
                    "brick_final": p1.round(5).tolist(), "finite": bool(np.isfinite(tw.d.qpos).all())}
    res["dt_half_agrees_mm"] = round(float(np.linalg.norm(np.array(res["dt"]["brick_final"]) -
                                                          np.array(res["dt_half"]["brick_final"])) * 1000), 3)
    return res


@tool(
    "so101_replay_batch",
    "SO-101 T2.replay_success / T2.replay_negative_control and the free-space controller check: replay each "
    "episode's logged leader commands in the twin (brick start from the objects JSON, optionally displaced "
    "`displace_mm` perpendicular to the base-to-brick line, +ve = counter-clockwise seen from above) and read the "
    "outcome from sim state (brick inside the box). Reports the success rate with a 95% Clopper-Pearson interval, "
    "free-space pad-centre tracking (mm) and optional dt/2 agreement.",
    {"type": "object", "required": ["calib", "objects"], "properties": {
        "calib": {"type": "string"}, "objects": {"type": "string"}, "split": {"type": "string"},
        "eps": {"type": "array", "items": {"type": "integer"}}, "displace_mm": {"type": "number"},
        "half_dt": {"type": "boolean", "description": "also replay at dt/2 and compare outcomes"},
        "cfg": {"type": "object", "description": "scene/controller overrides (pre-t0 only)"}, "out": {"type": "string"}}},
)
def so101_replay_batch(calib: str, objects: str, split: str = "train", eps=None, displace_mm: float = 0.0,
                       half_dt: bool = False, cfg=None, out: str = "") -> dict:
    from scipy.stats import beta
    from twin import so101_replay as rp
    eps = _eps(eps, split)
    tw, o = _twin(calib, objects, cfg)
    rows = []
    t0 = time.time()
    for ep in eps:
        st = rp.brick_start(o, ep)
        disp = None
        if displace_mm:
            p = np.array(st["pos"])
            r = p[:2] / np.linalg.norm(p[:2])
            tang = np.array([-r[1], r[0], 0.0])
            disp = tang * displace_mm / 1000
        tw.set_timestep(tw.cfg["timestep"])
        r = rp.replay(tw, ep, st, displace=disp)
        row = {"ep": ep, "ok": r["ok"]}
        if r["ok"]:
            fs = rp.free_space_error(tw, r, ep)
            row.update({"success": r["success"], "brick_final_box_frame_mm": r["brick_final_box_frame_mm"],
                        "brick_lift_mm": round(r["brick_max_z_mm"], 1), "free_space_median_mm": round(fs["median_mm"], 2),
                        "free_space_p90_mm": round(fs["p90_mm"], 2)})
            if half_dt:
                tw.set_timestep(tw.cfg["timestep"] / 2)
                r2 = rp.replay(tw, ep, st, displace=disp)
                row["half_dt_success"] = r2.get("success")
                row["half_dt_final_diff_mm"] = round(float(np.linalg.norm(np.array(r2["brick_final_world"]) -
                                                                           np.array(r["brick_final_world"])) * 1000), 2)
                tw.set_timestep(tw.cfg["timestep"])
        else:
            row["why"] = r.get("why")
        rows.append(row)
    ok = [r for r in rows if r["ok"]]
    k, n = sum(r["success"] for r in ok), len(rows)
    lo = float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    fsm = [r["free_space_median_mm"] for r in ok]
    res = {"calib": calib, "objects": objects, "split": split, "displace_mm": displace_mm, "n": n, "successes": k,
           "success_rate": k / n if n else None, "ci95": [round(lo, 3), round(hi, 3)],
           "free_space_median_mm": float(np.median(fsm)) if fsm else None,
           "free_space_p90_of_medians_mm": float(np.percentile(fsm, 90)) if fsm else None,
           "half_dt_agreement": (sum(r.get("half_dt_success") == r["success"] for r in ok) / len(ok)) if half_dt and ok else None,
           "cfg": cfg or {}, "wall_s": round(time.time() - t0, 1), "rows": rows}
    if out:
        res["json"] = _json(res, _p(out))
    return res


@tool(
    "so101_replay_video",
    "SO-101: replay one episode and render the twin from the calibrated `up` and `side` cameras next to the real "
    "footage (real top row, twin bottom row), as an H.264 mp4.",
    {"type": "object", "required": ["calib", "objects", "ep", "out"], "properties": {
        "calib": {"type": "string"}, "objects": {"type": "string"}, "ep": {"type": "integer"}, "split": {"type": "string"},
        "out": {"type": "string"}, "cfg": {"type": "object"}}},
)
def so101_replay_video(calib: str, objects: str, ep: int, out: str, split: str = "train", cfg=None) -> dict:
    import subprocess
    import cv2
    import mujoco
    from twin import so101_data as sd, so101_replay as rp
    _eps([ep], split)
    tw, o = _twin(calib, objects, cfg)
    r = rp.replay(tw, ep, rp.brick_start(o, ep), record=True)
    rend = mujoco.Renderer(tw.m, 240, 320)
    d = mujoco.MjData(tw.m)
    real = {c: sd.frames(ep, c) for c in sd.CAMS}
    p = _p(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    for k, qpos in r["frames"]:
        d.qpos[:] = qpos
        mujoco.mj_forward(tw.m, d)
        top, bot = [], []
        for c in sd.CAMS:
            top.append(cv2.resize(real[c][min(k, len(real[c]) - 1)], (320, 240)))
            rend.update_scene(d, camera=c)
            bot.append(cv2.cvtColor(rend.render(), cv2.COLOR_RGB2BGR))
        fr = np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)
        cv2.putText(fr, f"real ep{ep} f{k}", (6, 16), 0, 0.5, (0, 255, 255), 1)
        cv2.putText(fr, "twin (replayed leader commands)", (6, 256), 0, 0.5, (0, 255, 255), 1)
        vw.write(fr)
    vw.release()
    rend.close()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
                    "-movflags", "+faststart", str(p)], check=True)
    tmp.unlink()
    return {"mp4": _rel(p), "success": r.get("success"), "brick_final_box_frame_mm": r.get("brick_final_box_frame_mm")}


CAP_VMIN = 70  # fixed after gate (2026-10-25), chosen on train: see runs/claude/run-so101-1/fix_train/teal_sweep.json


def _cap_blob(img, min_area: int = 12, vmin: int = CAP_VMIN):
    """The teal cap's centroid for so101_auto_marks: the largest teal blob that is also bright (V >= vmin).
    The frozen version (twin/so101_camcal.py::teal_centroid, V >= 50) also accepted dark blue-tinted specks on the
    black servo housing (V about 55) when the cap was hidden or dim in the up camera, which put marks 30-250 px off
    on train and gate. Train only: V >= 70 removes every > 25 px outlier (runs/claude/run-so101-1/fix_train/)."""
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= 75) & (hsv[..., 0] <= 100) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= vmin)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    ks = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= min_area]
    if not ks:
        return None
    k = max(ks, key=lambda k: st[k, cv2.CC_STAT_AREA])
    return cen[k].astype(float)


@tool(
    "so101_auto_marks",
    "SO-101: code-detected keypoint marks for so101_keypoint_check: the teal cap's centroid (key teal_cap) in each "
    "camera at near-still frames of each episode (3 frames before the leader starts closing the gripper, and 3 frames "
    "after the follower's gripper is blocked by the brick). A watcher's by-eye marks are the gated measurement; these "
    "are the actor's automatic stand-in. v2 (fixed after the frozen gate run): bright teal only (V >= 70), so dark "
    "specks on the black housing are no longer taken for the cap.",
    {"type": "object", "required": ["out"], "properties": {
        "split": {"type": "string"}, "eps": {"type": "array", "items": {"type": "integer"}}, "out": {"type": "string"}}},
)
def so101_auto_marks(out: str, split: str = "gate", eps=None) -> dict:
    from twin import so101_camcal as cc, so101_data as sd, so101_replay as rp
    eps = _eps(eps, split)
    marks = []
    for ep in eps:
        cs, gf = rp.close_start(ep), rp.grasp_frame(ep)
        if gf is None:
            continue
        for f in (max(cs - 3, 0), gf + 3):
            for cn in sd.CAMS:
                t = _cap_blob(sd.frame(ep, cn, f))
                if t is not None:
                    marks.append({"ep": ep, "frame": int(f), "cam": cn, "key": "teal_cap", "uv": [round(float(t[0]), 2), round(float(t[1]), 2)]})
    res = {"by": "so101_auto_marks (teal centroid, code)", "split": split,
           "detector": {"version": "v2, fixed after the frozen gate run (2026-10-25)", "hue": [75, 100], "s_min": 80,
                        "v_min": CAP_VMIN, "min_area_px": 12, "pick": "largest"},
           "marks": marks}
    return {"json": _json(res, _p(out)), "n": len(marks)}


@tool(
    "so101_render_compare",
    "SO-101: real frames next to the twin rendered from the calibrated cameras, for one (ep, frame): the arm at the "
    "follower's logged state (each camera's latency applied), the table, the box, and the brick at its frame-0 start "
    "(so use frame 0 or a frame before the grasp). Writes two PNGs with the cameras stacked (up over side): real and "
    "twin, for a wipe-slider comparison.",
    {"type": "object", "required": ["calib", "objects", "ep", "frame", "out_real", "out_sim"], "properties": {
        "calib": {"type": "string"}, "objects": {"type": "string"}, "ep": {"type": "integer"},
        "frame": {"type": "integer"}, "split": {"type": "string"}, "out_real": {"type": "string"},
        "out_sim": {"type": "string"}, "scale": {"type": "number", "description": "output scale (default 0.75)"}}},
)
def so101_render_compare(calib: str, objects: str, ep: int, frame: int, out_real: str, out_sim: str,
                         split: str = "train", scale: float = 0.75) -> dict:
    import cv2
    import mujoco
    from twin import so101_camcal as cc, so101_data as sd, so101_replay as rp
    _eps([ep], split)
    tw, o = _twin(calib, objects, None)
    cams, jm, lat = cc.load_calib(_p(calib))
    st = rp.brick_start(o, ep)
    rend = mujoco.Renderer(tw.m, 480, 640)
    real, sim = [], []
    for cn in sd.CAMS:
        q = jm.q(cc.interp_units(sd.signals(ep)["state"], np.array([frame + lat[cn]])))[0]
        tw.reset(q, st["pos"], st["yaw"])
        rend.update_scene(tw.d, camera=cn)
        im = cv2.cvtColor(rend.render(), cv2.COLOR_RGB2BGR)
        rl = sd.frame(ep, cn, frame)
        for img, lst, tag in ((rl, real, "real"), (im, sim, "twin")):
            img = cv2.resize(img, None, fx=scale, fy=scale)
            cv2.putText(img, f"{tag} {cn} ep{ep} f{frame}", (8, 22), 0, 0.6, (0, 200, 255), 2)
            lst.append(img)
    rend.close()
    outs = []
    for lst, path in ((real, out_real), (sim, out_sim)):
        p = _p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), np.concatenate(lst, 0))
        outs.append(_rel(p))
    return {"real_png": outs[0], "sim_png": outs[1], "brick_start": st}
