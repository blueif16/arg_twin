"""HOT3D-Clips ground truth: export a clip as "any video + calibration", and score a scene card against GT.

Scoring only. The GT file written here is never passed to the pipeline or to Argon; the pipeline sees
only the exported RGB video and (optionally) the calibration stream, as it would for any Aria video.

Clip format (bop-benchmark/hot3d, train_aria/clip-XXXXXX.tar, 150 frames at 30 fps):
  <frame>.image_214-1.jpg   RGB, 1408x1408 fisheye624, sensor mounted sideways (rotate 90 deg CW for upright)
  <frame>.cameras.json      per stream: T_world_from_camera, calibration (FISHEYE624 params)
  <frame>.objects.json      per object: T_world_from_object, boxes/masks (amodal, RLE), visibility
  <frame>.hands.json        left/right: umetrack_pose.T_world_from_wrist, MANO params, visibility
  object_models/models_info.json: name, bbox min and size in the object frame (metres)

Our protocol (one protocol for every tool, per docs/knowledge/video-to-twin.md):
  camera   ATE after Sim(3) alignment (shape), ATE after SE(3) at our own scale (world error),
           rotation error, scale ratio s_opt (1.0 = our metric scale right)
  scale    median ratio of our metric depth to GT depth on GT-object pixels (rendered from GT meshes)
  objects  matched by 2D mask IoU; centre error in the camera frame and in the world (after SE(3)),
           size error of sorted box extents vs the GT model's bbox extents
  hands    wrist position error, camera frame and world (UmeTrack wrist origin vs our wrist landmark)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .common import (REPO, T_from, T_from_pose_dict, absp, align_poses, apply_sim3_to_poses, inv_T, mask_iou,
                     read_json, rel, rle_decode, robust_stats, rot_angle_deg, write_json)
from .frames import build_undistort_maps, fisheye624_project

HOT3D = REPO / "data" / "hot3d-clips"
RGB = "214-1"
GT_DIR = REPO / "runs" / "recon" / "v0" / "gt"


def clip_dir(clip: str) -> Path:
    return HOT3D / "extracted" / f"clip-{clip}"


def models_info() -> dict:
    return read_json(HOT3D / "object_models" / "models_info.json")


def load_gt(clip: str) -> dict:
    d = clip_dir(clip)
    frames = sorted(int(p.name.split(".")[0]) for p in d.glob("*.info.json"))
    cams, objs, hands, ts = [], [], [], []
    for f in frames:
        c = read_json(d / f"{f:06d}.cameras.json")[RGB]
        cams.append(T_from_pose_dict(c["T_world_from_camera"]))
        info = read_json(d / f"{f:06d}.info.json")
        ts.append(info["image_timestamps_ns"][RGB])
        objs.append(read_json(d / f"{f:06d}.objects.json"))
        hands.append(read_json(d / f"{f:06d}.hands.json"))
    calib = read_json(d / f"{frames[0]:06d}.cameras.json")[RGB]["calibration"]
    return {"frames": frames, "T_world_from_cam": np.stack(cams), "objects": objs, "hands": hands,
            "timestamps_ns": ts, "calibration": calib, "info": read_json(d / f"{frames[0]:06d}.info.json")}


def export_source(clip: str, out_dir: str | None = None) -> dict:
    """Write the clip as a plain RGB mp4 (raw fisheye, 30 fps) plus a calibration.json stream.
    GT goes to runs/recon/v0/gt/<clip>/ (scoring only)."""
    d = clip_dir(clip)
    src = absp(out_dir) if out_dir else REPO / "runs" / "recon" / "v0" / "sources" / f"hot3d-{clip}"
    src.mkdir(parents=True, exist_ok=True)
    vid = src / "rgb.mp4"
    if not vid.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i",
                        str(d / f"%06d.image_{RGB}.jpg"), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                        str(vid)], check=True)
    gt = load_gt(clip)
    cal = gt["calibration"]
    calib = {"model": "fisheye624", "params": cal["projection_params"], "width": cal["image_width"],
             "height": cal["image_height"], "rot90_cw": 1,
             "source": "Aria factory calibration shipped with the recording (HOT3D clip cameras.json)"}
    write_json(src / "calibration.json", calib)
    g = GT_DIR / clip
    g.mkdir(parents=True, exist_ok=True)
    write_json(g / "gt_cameras.json", {"T_world_from_cam": gt["T_world_from_cam"], "frames": gt["frames"],
                                       "note": "scoring only; never shown to the pipeline or Argon"})
    return {"video": rel(vid), "calibration": rel(src / "calibration.json"), "gt_dir": rel(g),
            "sequence_id": gt["info"]["sequence_id"], "participant": gt["info"]["participant_id"]}


# ----------------------------------------------------------------------------- GT in our frame conventions

def gt_in_frames(clip: str, frames_json: dict) -> dict:
    """GT re-expressed for the sampled, undistorted, upright frames the pipeline used."""
    gt = load_gt(clip)
    src_idx = frames_json["src_indices"]
    R_up = np.asarray(frames_json["R_upcam_from_srccam"])
    T_cam_from_up = T_from(R_up.T, np.zeros(3))
    cams = np.stack([gt["T_world_from_cam"][i] @ T_cam_from_up for i in src_idx])
    info = models_info()
    objects = {}
    for k, i in enumerate(src_idx):
        for oid, lst in gt["objects"][i].items():
            o = lst[0]
            mi = info[str(o["object_bop_id"])]
            ctr = np.array([mi["min_x"] + mi["size_x"] / 2, mi["min_y"] + mi["size_y"] / 2,
                            mi["min_z"] + mi["size_z"] / 2])
            T = T_from_pose_dict(o["T_world_from_object"])
            e = objects.setdefault(oid, {"name": o["object_name"], "bop_id": o["object_bop_id"],
                                         "size_sorted": sorted([mi["size_x"], mi["size_y"], mi["size_z"]]),
                                         "centre_world": {}, "T_world_from_object": {}, "visibility": {},
                                         "mask_rle_src": {}})
            e["centre_world"][k] = (T[:3, :3] @ ctr + T[:3, 3]).tolist()
            e["T_world_from_object"][k] = T.tolist()
            e["visibility"][k] = o.get("visibilities_modeled", {}).get(RGB, 0.0)
            if RGB in o.get("masks_amodal", {}):
                e["mask_rle_src"][k] = o["masks_amodal"][RGB]
    hands = {"left": {}, "right": {}}
    for k, i in enumerate(src_idx):
        for side in ("left", "right"):
            h = gt["hands"][i].get(side)
            if not h:
                continue
            vis = h.get("visibilities_modeled", {}).get(RGB, 0.0)
            T = T_from_pose_dict(h["umetrack_pose"]["T_world_from_wrist"])
            hands[side][k] = {"wrist_world": T[:3, 3].tolist(), "visibility": vis}
    return {"T_world_from_cam": cams, "objects": objects, "hands": hands,
            "sequence_id": gt["info"]["sequence_id"], "calibration": gt["calibration"]}


def undistorted_gt_mask(rle: dict, frames_json: dict, calib: dict) -> np.ndarray:
    m = rle_decode(rle).astype(np.uint8) * 255
    size = frames_json["width"]
    K = np.asarray(frames_json["K"])
    fov = float(np.degrees(2 * np.arctan((size / 2) / K[0, 0])))
    mx, my, _, _ = build_undistort_maps({"model": "fisheye624", "params": calib["projection_params"]}, size, fov,
                                        frames_json["rot90_cw"])
    return cv2.remap(m, mx, my, cv2.INTER_NEAREST) > 127


# ----------------------------------------------------------------------------- GT depth on object pixels

_MESH_DIR = GT_DIR / "meshes"


def _mesh_obj(bop_id) -> Path:
    """GT model (glb, mm or m) -> obj in metres for MuJoCo. Downloaded on demand into data/hot3d-clips."""
    _MESH_DIR.mkdir(parents=True, exist_ok=True)
    out = _MESH_DIR / f"obj_{int(bop_id):06d}.obj"
    if out.exists():
        return out
    from huggingface_hub import hf_hub_download
    import trimesh
    p = hf_hub_download("bop-benchmark/hot3d", f"object_models/obj_{int(bop_id):06d}.glb", repo_type="dataset",
                        local_dir=str(HOT3D))
    sc = trimesh.load(p, force="mesh")
    ext = sc.extents.max()
    if ext > 5:  # millimetres
        sc.apply_scale(0.001)
    sc.export(out)
    return out


def render_gt_depth(gtf: dict, k: int, K: np.ndarray, size: int) -> np.ndarray:
    """Depth (m) of GT objects only, seen from the GT camera at sampled frame k (0 where no object)."""
    import mujoco
    spec = mujoco.MjSpec()
    for oid, o in gtf["objects"].items():
        if k not in o["T_world_from_object"]:
            continue
        mp = _mesh_obj(o["bop_id"])
        mname = f"m{oid}"
        spec.add_mesh(name=mname, file=str(mp))
        T = np.asarray(o["T_world_from_object"][k])
        from .common import R_to_quat_wxyz
        b = spec.worldbody.add_body(name=f"o{oid}", pos=T[:3, 3], quat=R_to_quat_wxyz(T[:3, :3]))
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname, contype=0, conaffinity=0)
    Tc = np.asarray(gtf["T_world_from_cam"][k])
    from .common import R_to_quat_wxyz
    R_gl = Tc[:3, :3] @ np.diag([1, -1, -1])
    fovy = float(np.degrees(2 * np.arctan((size / 2) / K[1, 1])))
    spec.worldbody.add_camera(name="c", pos=Tc[:3, 3], quat=R_to_quat_wxyz(R_gl), fovy=fovy)
    spec.visual.global_.offwidth = size
    spec.visual.global_.offheight = size
    spec.visual.map.znear = 0.01
    m = spec.compile()
    m.vis.map.znear = 0.005
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    with mujoco.Renderer(m, size, size) as r:
        r.enable_depth_rendering()
        r.update_scene(d, camera="c")
        dep = r.render().copy()
    dep[dep > 50] = 0
    return dep


# ----------------------------------------------------------------------------- scoring

def _cam_frame(T_wc: np.ndarray, p_world) -> np.ndarray:
    return inv_T(T_wc)[:3, :3] @ np.asarray(p_world) + inv_T(T_wc)[:3, 3]


def score(card: dict, clip: str, frames_json: dict, pred_depth_fn=None, out_path: str | None = None) -> dict:
    """Per-stage GT errors for one scene card. pred_depth_fn(k) -> metric depth used by the card (for scale)."""
    gtf = gt_in_frames(clip, frames_json)
    res: dict = {"clip": clip, "sequence_id": gtf["sequence_id"]}
    n = frames_json["n"]
    # ---- camera
    traj = card["camera"]["trajectory"]
    fids = traj["frame_ids"]
    P = np.asarray(traj["T_world_from_cam"])
    G = gtf["T_world_from_cam"][fids]
    path_len = float(np.linalg.norm(np.diff(G[:, :3, 3], axis=0), axis=1).sum())
    s, R, t = align_poses(P, G, "sim3")
    A = apply_sim3_to_poses(P, s, R, t)
    ate_sim3 = float(np.sqrt(((A[:, :3, 3] - G[:, :3, 3]) ** 2).sum(1).mean()))
    s1, R1, t1 = align_poses(P, G, "se3")
    A1 = apply_sim3_to_poses(P, 1.0, R1, t1)
    ate_se3 = float(np.sqrt(((A1[:, :3, 3] - G[:, :3, 3]) ** 2).sum(1).mean()))
    rot_err = [rot_angle_deg(A[i, :3, :3].T @ G[i, :3, :3]) for i in range(len(fids))]
    # relative rotation between consecutive kept frames (scale-free)
    rpe_r = [rot_angle_deg((inv_T(P[i]) @ P[i + 1])[:3, :3].T @ (inv_T(G[i]) @ G[i + 1])[:3, :3])
             for i in range(len(fids) - 1)]
    K_gt = np.asarray(frames_json["K"]) if frames_json.get("K") else None
    K_pred = np.asarray(card["camera"]["K"])
    res["camera"] = {"method": traj.get("method"), "n": len(fids), "gt_path_len_m": path_len,
                     "ate_sim3_mm": 1000 * ate_sim3, "ate_se3_ourscale_mm": 1000 * ate_se3,
                     "ate_sim3_pct_of_path": 100 * ate_sim3 / max(path_len, 1e-6),
                     "scale_ratio_traj": s, "rot_err_deg_median": float(np.median(rot_err)),
                     "rpe_rot_deg_mean": float(np.mean(rpe_r)) if rpe_r else None,
                     "focal_px_pred": float(K_pred[0, 0]),
                     "focal_px_gt": float(K_gt[0, 0]) if K_gt is not None else None,
                     "focal_err_pct": 100 * (K_pred[0, 0] / K_gt[0, 0] - 1) if K_gt is not None else None,
                     "K_source": card["camera"].get("K_source")}
    # ---- metric scale from depth on GT-object pixels
    if pred_depth_fn is not None and K_gt is not None:
        ratios = []
        per_frame = []
        for k in sorted(set(list(range(0, n, max(1, n // 8))))):
            gd = render_gt_depth(gtf, k, K_gt, frames_json["width"])
            pdp = pred_depth_fn(k)
            if pdp is None:
                continue
            if pdp.shape != gd.shape:
                pdp = cv2.resize(pdp, gd.shape[::-1], interpolation=cv2.INTER_NEAREST)
            ok = (gd > 0.05) & np.isfinite(pdp) & (pdp > 0.05)
            if ok.sum() < 200:
                continue
            r = np.median(pdp[ok] / gd[ok])
            ratios.append(r)
            per_frame.append({"k": k, "ratio": float(r), "n_px": int(ok.sum()),
                              "abs_rel": float(np.median(np.abs(pdp[ok] - gd[ok]) / gd[ok]))})
        res["scale"] = {"depth_ratio_median": float(np.median(ratios)) if ratios else None,
                        "scale_err_pct": 100 * (float(np.median(ratios)) - 1) if ratios else None,
                        "per_frame": per_frame,
                        "note": "ratio of the card's metric depth to GT-mesh depth on object pixels; 1.0 = exact"}
    # ---- align card world to GT world (SE3 at our scale) for world errors
    T_align = T_from(R1, t1)

    def to_gt_world(p):
        return (T_align[:3, :3] @ np.asarray(p, float) + T_align[:3, 3])

    # ---- objects
    obj_res = []
    calib = gtf["calibration"]
    gt_masks_cache: dict = {}
    for o in card.get("objects", []):
        mt = o.get("mask_track")
        best = None
        if mt and Path(absp(mt)).exists():
            z = np.load(absp(mt))
            fr = z["frames"]
            ms = z["masks"]
            for oid, g in gtf["objects"].items():
                ious = []
                for j, k in enumerate(fr):
                    k = int(k)
                    if k not in g["mask_rle_src"] or j % 3:
                        continue
                    key = (oid, k)
                    if key not in gt_masks_cache:
                        gt_masks_cache[key] = undistorted_gt_mask(g["mask_rle_src"][k], frames_json, calib)
                    ious.append(mask_iou(ms[j], gt_masks_cache[key]))
                if ious:
                    v = float(np.mean(ious))
                    if best is None or v > best[1]:
                        best = (oid, v)
        e = {"id": o["id"], "category": o["category"], "matched_gt": None}
        if best and best[1] > 0.05:
            g = gtf["objects"][best[0]]
            e.update(matched_gt=g["name"], match_iou=best[1])
            pred_sz = sorted(o["size_m"])
            e["size_pred_m"] = pred_sz
            e["size_gt_m"] = g["size_sorted"]
            rel_err = [abs(a / b - 1) for a, b in zip(pred_sz, g["size_sorted"])]
            e["size_err_pct_mean"] = 100 * float(np.mean(rel_err))
            e["size_err_pct_max_dim"] = 100 * abs(pred_sz[-1] / g["size_sorted"][-1] - 1)
            cam_err, world_err = [], []
            for tr in o.get("track", []):
                k = int(tr["frame"])
                if k not in g["centre_world"] or tr.get("pos") is None:
                    continue
                pc_pred = _cam_frame(P[fids.index(k)] if k in fids else np.asarray(card["camera"]["all_T_world_from_cam"][k]),
                                     tr["pos"])
                pc_gt = _cam_frame(gtf["T_world_from_cam"][k], g["centre_world"][k])
                cam_err.append(1000 * float(np.linalg.norm(pc_pred - pc_gt)))
                world_err.append(1000 * float(np.linalg.norm(to_gt_world(tr["pos"]) - np.asarray(g["centre_world"][k]))))
            e["centre_err_cam_mm"] = robust_stats(cam_err)
            e["centre_err_world_mm"] = robust_stats(world_err)
        obj_res.append(e)
    res["objects"] = obj_res
    # ---- hands
    hand_res = {}
    for h in card.get("hands", []):
        side = h["side"]
        cam_err, world_err = [], []
        for tr in h.get("track", []):
            k = int(tr["frame"])
            g = gtf["hands"][side].get(k)
            if not g or tr.get("wrist") is None:
                continue
            Tk = np.asarray(card["camera"]["all_T_world_from_cam"][k])
            pc_pred = _cam_frame(Tk, tr["wrist"])
            pc_gt = _cam_frame(gtf["T_world_from_cam"][k], g["wrist_world"])
            cam_err.append(1000 * float(np.linalg.norm(pc_pred - pc_gt)))
            world_err.append(1000 * float(np.linalg.norm(to_gt_world(tr["wrist"]) - np.asarray(g["wrist_world"]))))
        gt_frames = [k for k, g in gtf["hands"][side].items() if g["visibility"] > 0.5]
        hand_res[side] = {"wrist_err_cam_mm": robust_stats(cam_err), "wrist_err_world_mm": robust_stats(world_err),
                          "frames_pred": len(h.get("track", [])), "frames_gt_visible": len(gt_frames)}
    res["hands"] = hand_res
    # summary row
    objs = [o for o in obj_res if o.get("matched_gt")]
    res["summary"] = {
        "cam_ate_sim3_mm": res["camera"]["ate_sim3_mm"],
        "cam_ate_se3_mm": res["camera"]["ate_se3_ourscale_mm"],
        "cam_rot_err_deg": res["camera"]["rot_err_deg_median"],
        "focal_err_pct": res["camera"]["focal_err_pct"],
        "scale_err_pct": (res.get("scale") or {}).get("scale_err_pct"),
        "obj_matched": f"{len(objs)}/{len(obj_res)} pred matched; {len(gtf['objects'])} GT objects in clip",
        "obj_centre_err_cam_mm_median": float(np.median([o["centre_err_cam_mm"]["median"] for o in objs
                                                         if o["centre_err_cam_mm"].get("n")])) if objs else None,
        "obj_size_err_pct_median": float(np.median([o["size_err_pct_mean"] for o in objs])) if objs else None,
        "hand_wrist_err_cam_mm_median": float(np.median([v["wrist_err_cam_mm"]["median"] for v in hand_res.values()
                                                         if v["wrist_err_cam_mm"].get("n")])) if any(
            v["wrist_err_cam_mm"].get("n") for v in hand_res.values()) else None,
        "hand_wrist_err_world_mm_median": float(np.median([v["wrist_err_world_mm"]["median"] for v in hand_res.values()
                                                           if v["wrist_err_world_mm"].get("n")])) if any(
            v["wrist_err_world_mm"].get("n") for v in hand_res.values()) else None,
    }
    if out_path:
        write_json(absp(out_path), res)
    return res
