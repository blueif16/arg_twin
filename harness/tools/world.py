"""World-frame tools: put the cameras, the board and the tool tip in the robot base frame.

Stage B foundation (playbook/skills/build-twin/references/reassemble-world-frame.md). The robot's joint angles are logged every
frame, so the twin rendered at those angles shows exactly where the arm must appear in each real
camera; aligning that render with the real frame calibrates the camera.

Camera spec accepted by render_overlay / refine_camera (`cam` argument), any one of:
  "calib"                                     the camera saved in twin/calib/<recording>.json
  {"pos": [x,y,z], "look_at": [x,y,z], "fovy_deg": 45, "up": [0,0,1], "roll_deg": 0, "parent": "world"}
  {"pos": [...], "quat": [w,x,y,z], "fx": .., "fy": .., "cx": .., "cy": .., "parent": ..}  (CameraModel dict)
`parent` is "world" for hama1/hama2 and "gripper/hand" for the hand camera; pos/look_at/up are in the
parent frame (hand frame: x away from the arm when the tool points down, y along finger travel, z out
of the fingers). Quaternions are MuJoCo camera frames (x right, y up, looking along -z).

Keypoints (optional, `keypoints` / `points`): [{"frame": f, "uv": [u, v], "board": [x,y,z]}] with the
3D point in the NIST_Board1 frame, or "world": [x,y,z], or "part": "<layout name>" (hole xy from
twin/assets/board_layout.json, z = release TCP height + 5 mm, roughly the visible top of the part).
Outputs go to runs/world/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from harness.registry import tool

CAM_SPEC = {"description": 'camera spec: "calib", a look-at dict {pos, look_at, fovy_deg|fx, up?, roll_deg?, '
                           'parent?} or a CameraModel dict {pos, quat, fx, fy?, cx?, cy?, parent?}'}
KP_SPEC = {"type": "array", "items": {"type": "object"},
           "description": 'points [{"frame": f, "uv": [u,v], "board"|"world": [x,y,z] | "part": name, "label"?: str}]'}


def _w():
    from twin import world  # imported lazily: needs MUJOCO_GL and a GL context
    return world


def _rel(p: Path) -> str:
    from twin import REPO_DIR
    return str(Path(p).resolve().relative_to(REPO_DIR))


def _camera(camera: str, cam, episode: int):
    w = _w()
    if cam is None or cam == "calib":
        c = w.load_calib(w.recording_of(episode))
        if not c or camera not in c.get("cameras", {}):
            raise ValueError(f"no saved calibration for {camera} in {w.calib_path(w.recording_of(episode))}; "
                             "pass a camera spec")
        return w.CameraModel.from_dict(c["cameras"][camera])
    if isinstance(cam, str):  # path to a JSON camera dict (e.g. a refine result)
        d = json.loads(Path(cam).read_text())
        return w.CameraModel.from_dict(d.get("final", d))
    cam = dict(cam)
    parent = cam.get("parent", "world" if camera != "hand" else "gripper/hand")
    if "look_at" in cam:
        m = w.CameraModel.look_at(camera, cam["pos"], cam["look_at"], cam.get("fovy_deg", 45.0),
                                  up=cam.get("up", (0, 0, 1) if parent == "world" else (1, 0, 0)),
                                  roll_deg=cam.get("roll_deg", 0.0), parent=parent)
        if "fx" in cam:
            m.fx = m.fy = float(cam["fx"])
        return m
    cam.setdefault("name", camera)
    cam.setdefault("parent", parent)
    if "fy" not in cam and "fx" in cam:
        cam["fy"] = cam["fx"]
    return w.CameraModel.from_dict(cam)


def _keypoints(kps, episode):
    if not kps:
        return None
    w = _w()
    out = []
    lay = None
    for k in kps:
        k = dict(k)
        if "part" in k and "board" not in k and "world" not in k:
            if lay is None:
                lay = json.loads(w.LAYOUT_JSON.read_text())["parts"]
            p = lay[k["part"]]
            k["board"] = [p["xy"][0], p["xy"][1], p.get("tcp_z_release", 0.0) + 0.005]
            k.setdefault("label", k["part"])
        out.append(k)
    return out


def _frames(episode: int, frames, n_frames: int, still: bool = False) -> list[int]:
    w = _w()
    if frames:
        return [int(f) for f in frames]
    if still:
        st = w.still_frames(episode, max_speed=0.01, max_rate_deg=3.0, min_gap=30)
        return [st[i] for i in np.linspace(0, len(st) - 1, min(n_frames, len(st))).astype(int)]
    return w.sample_frames(episode, n_frames)


def _summary(ev: dict) -> dict:
    return {k: v for k, v in ev.items() if k not in ("per_frame", "frames")}


# ----------------------------------------------------------------------------------------------
@tool("real_frame",
      "Save one real camera frame of a REASSEMBLE episode as PNG (runs/world/frames/) and return the robot "
      "state logged at that frame. grid=true draws a labelled 40 px grid so pixel positions can be read off.",
      {"type": "object", "properties": {
          "episode": {"type": "integer"}, "camera": {"type": "string", "enum": ["hand", "hama1", "hama2"]},
          "frame": {"type": "integer", "description": "frame index (30 fps); or give t"},
          "t": {"type": "number", "description": "seconds from episode start"},
          "grid": {"type": "boolean", "default": False}},
       "required": ["episode", "camera"]})
def real_frame(episode: int, camera: str, frame: int | None = None, t: float | None = None, grid: bool = False) -> dict:
    import cv2
    w = _w()
    fi = w.frame_index(episode, frame, t)
    img = w.read_frame(episode, camera, fi)
    if grid:
        img = img.copy()
        for x in range(40, w.W, 40):
            cv2.line(img, (x, 0), (x, w.H - 1), (0, 255, 255) if x % 200 else (255, 0, 255), 1)
            cv2.putText(img, str(x), (x + 2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 0, 0), 1)
        for y in range(40, w.H, 40):
            cv2.line(img, (0, y), (w.W - 1, y), (0, 255, 255) if y % 200 else (255, 0, 255), 1)
            cv2.putText(img, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 0, 0), 1)
    path = w.save_png(img, w.RUNS_DIR / "frames" / f"ep{episode:03d}_{camera}_f{fi:05d}{'_grid' if grid else ''}.png")
    s = w.episode_state(episode)
    return {"path": _rel(path), "episode": episode, "recording": s.recording, "frame": fi, "t": round(fi / w.FPS, 3),
            "segment": int(s.seg[fi]), "task_index": int(s.task[fi]),
            "joints_rad": np.round(s.q[fi], 4).tolist(), "gripper_m": np.round(s.grip[fi], 4).tolist(),
            "ee_pose_xyz_wxyz": np.round(s.ee[fi], 4).tolist()}


@tool("render_overlay",
      "Render the twin (robot at the logged joint angles, board at its mocap pose) from a camera spec and blend "
      "it over the real frames: green = robot outline, magenta = board outline, circles = projected points "
      "(red) and their marked positions (yellow). Returns the PNG and alignment scores with the frozen "
      "world-frame checks (static cameras: robot outline-to-edge px, moving-silhouette IoU, board outline px; "
      "hand camera: finger-pad IoU on the open/closed medians, near-field edge px, keypoint px).",
      {"type": "object", "properties": {
          "episode": {"type": "integer"}, "camera": {"type": "string", "enum": ["hand", "hama1", "hama2"]},
          "cam": CAM_SPEC, "frames": {"type": "array", "items": {"type": "integer"}},
          "n_frames": {"type": "integer", "default": 6, "description": "frames spread over the episode if frames not given"},
          "points": KP_SPEC, "tag": {"type": "string", "default": "overlay"},
          "score_frames": {"type": "integer", "default": 0,
                           "description": "also score on this many frames spread over the episode (0 = only the shown frames)"}},
       "required": ["episode", "camera"]})
def render_overlay(episode: int, camera: str, cam=None, frames=None, n_frames: int = 6, points=None,
                   tag: str = "overlay", score_frames: int = 0) -> dict:
    import cv2
    w = _w()
    c = _camera(camera, cam, episode)
    fr = _frames(episode, frames, n_frames, still=(c.parent != "world"))
    kps = _keypoints(points, episode)
    view = w.get_view()
    rec = w.recording_of(episode)
    cal = w.load_calib(rec)
    bp, bq = w.board_pose(rec, cal)  # the calibrated board (contacts + camera plate height) when saved
    view.set_board(bp, bq)
    view.set_hand_offset(w.calib_hand_offset(rec, cal))  # physical hand position (tool-length correction)
    tiles = []
    for f in fr:
        view.set_frame(episode, f)
        real = w.read_frame(episode, camera, f)
        lab = view.render_labels(c)
        rgb = view.render_rgb(c)
        pts = []
        for k in kps or []:
            P = w.keypoint_world(k, episode)
            uv, z = view.project_world(c, P[None])
            if z[0] > 0:
                pts.append((uv, (255, 0, 0)))
            if int(k.get("frame", -1)) == f and "uv" in k:
                pts.append((np.array([k["uv"]], float), (255, 255, 0)))
        tiles.append(w.overlay(real, rgb, lab, points=pts, title=f"{camera} ep{episode} f{f}"))
    cols = 3 if len(tiles) > 4 else 2 if len(tiles) > 1 else 1
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    if grid.shape[1] > 1920:
        grid = cv2.resize(grid, None, fx=1920 / grid.shape[1], fy=1920 / grid.shape[1], interpolation=cv2.INTER_AREA)
    path = w.save_png(grid, w.RUNS_DIR / "overlays" / f"{tag}_ep{episode:03d}_{camera}.png")
    view.set_hand_offset(0.0)
    ev_frames = _frames(episode, None, score_frames, still=(c.parent != "world")) if score_frames else fr
    ev = w.evaluate_camera(c, episode, ev_frames, keypoints=[k for k in kps or [] if "uv" in k])
    return {"overlay": _rel(path), "camera": c.to_dict(), "frames_shown": fr, "scores": _summary(ev),
            "per_frame": ev["per_frame"]}


@tool("refine_camera",
      "Numerically refine a camera from an initial spec (your native estimate): Nelder-Mead over pose and focal "
      "length, coarse to fine. Static cameras (hama1/hama2) fit the moving-robot silhouette (IoU vs background "
      "subtraction) + robot and board outline-to-edge distance over frames spread over the episode. The hand "
      "camera fits the finger pads and housing on open/closed-gripper medians + board outline on still frames "
      "+ optional keypoints. Needs a start in the right region (a few cm / degrees). Writes "
      "runs/world/refine/<tag>_<camera>.json and an overlay; returns the refined camera and held-out scores.",
      {"type": "object", "properties": {
          "episode": {"type": "integer"}, "camera": {"type": "string", "enum": ["hand", "hama1", "hama2"]},
          "cam": CAM_SPEC, "n_fit_frames": {"type": "integer", "default": 16},
          "keypoints": KP_SPEC,
          "weights": {"type": "object", "description": "override term weights: iou, robot_edge, board_edge, "
                                                       "pad_iou, near_edge, keypoints, edge_clip_px"},
          "fix_focal": {"type": "boolean", "default": False,
                        "description": "keep the spec's focal length (e.g. shared by identical cameras); "
                                       "fit pose only. Use it when depth and focal length trade off"},
          "tag": {"type": "string", "default": "refine"}},
       "required": ["episode", "camera", "cam"]})
def refine_camera(episode: int, camera: str, cam, n_fit_frames: int = 16, keypoints=None, weights=None,
                  fix_focal: bool = False, tag: str = "refine") -> dict:
    w = _w()
    c0 = _camera(camera, cam, episode)
    kps = _keypoints(keypoints, episode)
    body = c0.parent != "world"
    if body:
        st = w.still_frames(episode, max_speed=0.01, max_rate_deg=3.0, min_gap=30)
        allf = [st[i] for i in np.linspace(0, len(st) - 1, min(2 * n_fit_frames + 1, len(st))).astype(int)]
    else:
        allf = w.sample_frames(episode, 2 * n_fit_frames - 1)
    fit, held = allf[0::2], allf[1::2]
    t0 = time.time()
    ev0 = w.evaluate_camera(c0, episode, held, keypoints=kps)
    log = []
    c1, info = w.refine_camera(c0, episode, fit, weights=weights, keypoints=kps, fix_focal=fix_focal,
                               log=log.append)
    ev1 = w.evaluate_camera(c1, episode, held, keypoints=kps)
    out = {"camera": camera, "episode": episode, "init": c0.to_dict(), "final": c1.to_dict(), "refine": info,
           "fit_frames": fit, "heldout_frames": held, "keypoints": kps, "eval_init": _summary(ev0),
           "eval_final": _summary(ev1), "seconds": round(time.time() - t0, 1)}
    p = w.RUNS_DIR / "refine" / f"{tag}_ep{episode:03d}_{camera}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float))
    ov = render_overlay(episode, camera, cam=c1.to_dict(), frames=held[:: max(1, len(held) // 6)][:6],
                        points=kps, tag=f"{tag}_final")
    return {"result": _rel(p), "overlay": ov["overlay"], "final": c1.to_dict(),
            "eval_init": out["eval_init"], "eval_final": out["eval_final"], "seconds": out["seconds"],
            "stages": info["stages"]}


@tool("save_world_frame",
      "Write twin/calib/<recording>.json: cameras (intrinsics + extrinsics in their parent frame), the board pose "
      "(NIST_Board1 frame in the robot base frame) and the tool tip. Cameras given here replace those saved; "
      "others are kept. Each camera entry may carry its eval/status/method so the file records how good it is.",
      {"type": "object", "properties": {
          "episode": {"type": "integer"},
          "cameras": {"type": "object", "description": "{name: camera spec or path to a refine_camera result JSON}"},
          "status": {"type": "object", "description": "{name: 'pass' | 'provisional: <why>'}"},
          "method": {"type": "object", "description": "{name: short text: how this camera was obtained}"},
          "board": {"type": "object", "description": "optional {pos, quat (wxyz), source}; default = mocap NIST_Board1"},
          "notes": {"type": "string"}},
       "required": ["episode", "cameras"]})
def save_world_frame(episode: int, cameras: dict, status: dict | None = None, method: dict | None = None,
                     board: dict | None = None, notes: str = "") -> dict:
    w = _w()
    from twin.scene import build_base as bb
    rec = w.recording_of(episode)
    path = w.calib_path(rec)
    doc = w.load_calib(rec) or {}
    doc.setdefault("cameras", {})
    evals = {}
    for name, spec in cameras.items():
        if isinstance(spec, str) and spec.endswith(".json"):
            r = json.loads(Path(spec).read_text())
            evals[name] = r.get("eval_final")
            spec = r.get("final", r)
        c = _camera(name, spec, episode)
        entry = c.to_dict()
        entry["status"] = (status or {}).get(name, "unchecked")
        entry["method"] = (method or {}).get(name, "")
        if evals.get(name):
            entry["eval"] = evals[name]
        doc["cameras"][name] = entry
    if board is None:
        b = w.mocap_poses(rec)["NIST_Board1"]
        board = {"pos": b["pos"], "quat": b["quat"], "source": "mocap NIST_Board1 (poses JSON), checked by hama overlays"}
    board.setdefault("plate_in_board_frame", {k: list(v) if isinstance(v, tuple) else v for k, v in w.BOARD_PLATE.items()})
    doc.update({
        "recording": rec, "episode": episode,
        "frame": "robot base frame = FR3 link0 = MuJoCo world = mocap world (docs/research/reassemble-data-notes.md)",
        "camera_convention": "pos/quat: MuJoCo camera frame (x right, y up, looks along -z) in its parent frame; "
                             "fx fy cx cy in pixels at width x height, OpenCV pixel centres; "
                             "T_parent_cam_opencv = OpenCV camera axes (x right, y down, z forward) in the parent frame",
        "board": board,
        "tcp": {"from_flange_m": bb.TCP_FROM_FLANGE, "rotz_deg": -45.0, "ft_stack_m": round(bb.FT_STACK, 4),
                "hand_tcp_m": bb.HAND_TCP, "site": "tcp on gripper/hand (= logged O_T_EE)",
                "source": "FK of logged joints vs logged O_T_EE (0.6 mm max over episode 0)"},
        "written_by": "harness/tools/world.py save_world_frame", "written_at": time.strftime("%Y-%m-%d %H:%M"),
    })
    if notes:
        doc["notes"] = notes
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=float))
    return {"path": _rel(path), "cameras": {k: v.get("status") for k, v in doc["cameras"].items()}}


@tool("check_world_frame",
      "The frozen world-frame check: load twin/calib/<recording>.json and score every saved camera on held-out "
      "frames spread over the episode (static cameras) or on the finger medians and keypoints (hand camera). "
      "Returns pass/fail per camera against the thresholds in twin/world.py CHECKS.",
      {"type": "object", "properties": {
          "episode": {"type": "integer"}, "n_frames": {"type": "integer", "default": 15},
          "keypoints": KP_SPEC},
       "required": ["episode"]})
def check_world_frame(episode: int, n_frames: int = 15, keypoints=None) -> dict:
    w = _w()
    rec = w.recording_of(episode)
    doc = w.load_calib(rec)
    if not doc:
        raise ValueError(f"no {w.calib_path(rec)}")
    kps = _keypoints(keypoints, episode)
    out = {}
    for name, cd in doc["cameras"].items():
        c = w.CameraModel.from_dict(cd)
        if c.parent == "world":
            fr = w.sample_frames(episode, 2 * n_frames + 1)[1::2]
        else:
            fr = _frames(episode, None, n_frames, still=True)
        ev = w.evaluate_camera(c, episode, fr, keypoints=[k for k in kps or [] if "uv" in k] if c.parent != "world" else None)
        out[name] = {"pass": ev["check"]["pass"], "checks": ev["check"]["checks"]}
    return {"recording": rec, "all_pass": all(v["pass"] for v in out.values()), "cameras": out,
            "thresholds": w.CHECKS}



# ----------------------------------------------------------------------------------------------
# Robot-keypoint calibration (twin/camcal.py; method in playbook/skills/build-twin/references/reassemble-world-frame.md)
# ----------------------------------------------------------------------------------------------
def _out(brain: str, run_id: str) -> Path:
    from twin import REPO_DIR
    p = REPO_DIR / "runs" / brain / run_id / "world"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_marks(paths) -> list[dict]:
    from twin import REPO_DIR
    out = []
    for p in paths or []:
        d = json.loads((REPO_DIR / p).read_text() if not Path(p).is_absolute() else Path(p).read_text())
        out += d["marks"] if isinstance(d, dict) else d
    return out


def _exclude(episode: int, exclude_frames, exclude_files) -> list[int]:
    """Frames to keep away from (a watcher's held-out frames, your own held-out marks): explicit list plus every
    integer stored under a 'frame' key (or a 'frames*' list) in the given JSON files."""
    from twin import REPO_DIR
    ex = set(int(f) for f in exclude_frames or [])

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "frame" and isinstance(v, (int, float)):
                    ex.add(int(v))
                elif "frames" in k and isinstance(v, list) and all(isinstance(i, (int, float)) for i in v):
                    ex.update(int(i) for i in v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for p in exclude_files or []:
        walk(json.loads((REPO_DIR / p).read_text()))
    return sorted(ex)


@tool("mark_candidates",
      "Help a brain mark keypoints: for each (frame, key) project the key with the saved calibration (robot keys "
      "left_pad/right_pad via FK + latency + tool correction; board keys c00..c01, hole_<part> via the board pose), "
      "find dark-blob candidates around it, and write one PNG sheet of zoomed crops with a pixel grid and numbered "
      "candidates plus a JSON of candidate centroids. The brain looks, says which blob is which key (or none), and "
      "turns those into marks {cam, frame, key, uv} (uv = the chosen blob's centroid). Keep frames away from any "
      "watcher's held-out frames.",
      {"type": "object", "properties": {
          "episode": {"type": "integer"}, "camera": {"type": "string", "enum": ["hand", "hama1", "hama2"]},
          "items": {"type": "array", "items": {"type": "object"}, "description": "[{frame, key}]"},
          "half": {"type": "integer", "default": 40}, "rel": {"type": "number", "default": 0.55,
                                                              "description": "dark threshold: gray < rel x local median"},
          "tag": {"type": "string", "default": "cands"}, "brain": {"type": "string", "default": "claude"},
          "run_id": {"type": "string", "default": "run-standin-1"}},
       "required": ["episode", "camera", "items"]})
def mark_candidates(episode: int, camera: str, items: list, half: int = 40, rel: float = 0.55, tag: str = "cands",
                    brain: str = "claude", run_id: str = "run-standin-1") -> dict:
    import cv2
    from twin import camcal as cc
    w = _w()
    cams, lat, tool, bpose = cc.calib_state(episode)
    kin = cc.Kinematics(episode)
    tiles, out = [], []
    for i, it in enumerate(items):
        f, key = int(it["frame"]), it["key"]
        uv, z = cc.project_key(cams[camera], key, f, kin, tool, bpose, lat.get(camera, 0.0))
        img = w.read_frame(episode, camera, f)
        bl = cc.dark_blobs(img, uv[0], uv[1], half=half, rel=rel, amin=6, amax=4 * half * half)
        tiles.append(cc.crop_with_grid(img, uv[0], uv[1], half=half, zoom=4, label=f"#{i} f{f} {key}", blobs=bl))
        out.append({"i": i, "frame": f, "key": key, "pred": [round(float(v), 1) for v in uv], "blobs": bl})
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    o = _out(brain, run_id)
    png = w.save_png(grid, o / f"{tag}_{camera}.png")
    js = o / f"{tag}_{camera}.json"
    js.write_text(json.dumps({"camera": camera, "episode": episode, "items": out}, indent=1))
    return {"sheet": _rel(png), "candidates": _rel(js), "n": len(out)}


@tool("calibrate_static_cameras",
      "Robot-keypoint calibration of hama1 + hama2 together: render-and-compare of the arm's black joint caps and "
      "the finger pads (masked ZNCC of the twin's rendered patch against the real frame) on frames spread over the "
      "episode, then bundle adjustment of both camera poses, shared intrinsics (f, cx, cy; optional k1), the "
      "video/joint-log latency per camera, the tool length (hand offset along its z axis) and the board plate-top "
      "height, with optional marks (board corners, pads). Iterates with a shrinking search radius. Every 3rd frame "
      "is held out of the fit and scored. Writes runs/<brain>/<run_id>/world/static_fit.json (not twin/calib).",
      {"type": "object", "properties": {
          "episode": {"type": "integer"}, "start": {"type": "string", "default": "calib",
                                                   "description": "'calib' or a static_fit.json path"},
          "marks": {"type": "array", "items": {"type": "string"}, "description": "marks JSON paths (repo-relative)"},
          "exclude_frames": {"type": "array", "items": {"type": "integer"}},
          "exclude_files": {"type": "array", "items": {"type": "string"},
                            "description": "JSON files whose 'frame' values (and 'frames*' lists) are kept out (+-radius)"},
          "radius": {"type": "integer", "default": 30}, "step": {"type": "integer", "default": 15},
          "iters": {"type": "integer", "default": 2}, "fit_tool": {"type": "boolean", "default": True},
          "fit_board_z": {"type": "boolean", "default": True}, "fit_k1": {"type": "boolean", "default": False},
          "shared_intrinsics": {"type": "boolean", "default": True},
          "brain": {"type": "string", "default": "claude"}, "run_id": {"type": "string", "default": "run-standin-1"}},
       "required": ["episode"]})
def calibrate_static_cameras(episode: int, start: str = "calib", marks=None, exclude_frames=None, exclude_files=None,
                             radius: int = 30, step: int = 15, iters: int = 2, fit_tool: bool = True,
                             fit_board_z: bool = True, fit_k1: bool = False, shared_intrinsics: bool = True,
                             brain: str = "claude", run_id: str = "run-standin-1") -> dict:
    from twin import camcal as cc
    w = _w()
    t0 = time.time()
    rec = w.recording_of(episode)
    calib = w.load_calib(rec)
    if start == "calib":
        cams0 = {k: w.CameraModel.from_dict(calib["cameras"][k]) for k in ("hama1", "hama2")}
    else:
        d = json.loads(Path(start).read_text())
        cams0 = {k: w.CameraModel.from_dict(v) for k, v in d["cams"].items()}
    ex = _exclude(episode, exclude_frames, exclude_files)
    ok = cc.free_frames(episode, ex, radius)
    n = w.episode_state(episode).n
    frames = [f for f in range(2 * step, n - 2 * step, step) if ok[f]]
    mk = [m for m in _load_marks(marks) if m["cam"] in ("hama1", "hama2")]
    cfg = cc.FitConfig(fit_tool=fit_tool, fit_board_z=fit_board_z, fit_k1=fit_k1, shared_intrinsics=shared_intrinsics)
    r = cc.static_pipeline(episode, cams0, frames, mk, cfg, iters=iters, calib=calib, log=None)
    res = {"recording": rec, "episode": episode, "frames": frames, "excluded": ex, "radius": radius,
           "cams": {k: {**v.to_model().to_dict(), "k1": v.k1, "k2": v.k2} for k, v in r["cams"].items()},
           "latency": r["latency"], "tool": r["tool"], "board_dz": r["board_dz"], "history": r["history"],
           "marks": mk, "config": cfg.__dict__, "seconds": round(time.time() - t0, 1)}
    p = _out(brain, run_id) / "static_fit.json"
    p.write_text(json.dumps(res, indent=1, default=float))
    last = r["history"][-1]
    return {"path": _rel(p), "n_frames": len(frames), "result": last,
            "cams": {k: {"pos": v["pos"], "fx": v["fx"], "cx": v["cx"], "cy": v["cy"]} for k, v in res["cams"].items()},
            "seconds": res["seconds"]}


@tool("calibrate_hand_camera",
      "Wrist camera in the tool-corrected hand frame: far-field board marks (empty holes marked with mark_candidates; "
      "3D from the board pose) + near-field finger-pad top corners on the open/closed hand medians (3D re-rendered from "
      "the twin each iteration), with the tool correction and board pose from a static fit. Fits pose + f (soft prior, "
      "default 615 px for a RealSense D435 colour stream at 640x480); principal point fixed unless fit_pp. Reports "
      "leave-one-key-out board error and inner-pad-corner error on held-out open-gripper frames. Writes "
      "runs/<brain>/<run_id>/world/hand_fit.json.",
      {"type": "object", "properties": {
          "episode": {"type": "integer"}, "cam": CAM_SPEC, "marks": {"type": "array", "items": {"type": "string"}},
          "static_fit": {"type": "string", "description": "static_fit.json (tool, board_dz); default: the saved calib"},
          "f_prior": {"type": "number", "default": 615.0}, "f_prior_sigma": {"type": "number", "default": 20.0},
          "fit_pp": {"type": "boolean", "default": False},
          "heldout_finger_frames": {"type": "array", "items": {"type": "integer"}},
          "brain": {"type": "string", "default": "claude"}, "run_id": {"type": "string", "default": "run-standin-1"}},
       "required": ["episode", "cam", "marks"]})
def calibrate_hand_camera(episode: int, cam, marks, static_fit: str | None = None, f_prior: float = 615.0,
                          f_prior_sigma: float = 20.0, fit_pp: bool = False, heldout_finger_frames=None,
                          brain: str = "claude", run_id: str = "run-standin-1") -> dict:
    from twin import camcal as cc
    w = _w()
    t0 = time.time()
    calib = w.load_calib(w.recording_of(episode))
    if static_fit:
        sf = json.loads(Path(static_fit).read_text())
        tool, bpose = sf["tool"], cc.board_pose_with_z(calib, sf["board_dz"])
    else:
        _, _, tool, bpose = cc.calib_state(episode, calib)
    c0 = _camera("hand", cam, episode)
    mk = [m for m in _load_marks(marks) if m["cam"] == "hand"]
    cfg = cc.HandFitConfig(fit_pp=fit_pp, f_prior=f_prior, f_prior_sigma=f_prior_sigma)
    camf, rep = cc.hand_pipeline(episode, c0, mk, tool, bpose, cfg, log=None)
    if heldout_finger_frames:
        e = cc.heldout_finger_corners(episode, camf, tool, [int(f) for f in heldout_finger_frames])
        rep["heldout_finger_corner_px_median"] = round(float(np.median(e)), 2) if e.size else None
        rep["heldout_finger_corner_n"] = int(e.size)
    res = {"camera": camf.to_model().to_dict(), "tool": tool, "board_pose": [list(map(float, bpose[0])), list(map(float, bpose[1]))],
           "report": rep, "marks": mk, "config": cfg.__dict__, "seconds": round(time.time() - t0, 1)}
    p = _out(brain, run_id) / "hand_fit.json"
    p.write_text(json.dumps(res, indent=1, default=float))
    return {"path": _rel(p), "camera": {k: res["camera"][k] for k in ("pos", "quat", "fx")},
            "report": {k: v for k, v in rep.items() if k != "board_leave_one_key_out"}, "seconds": res["seconds"]}


@tool("keypoint_check",
      "Score the saved calibration (twin/calib/<recording>.json, with its latency and tool correction) on marks "
      "the fit never used: median/p90 px per camera, and two-view consistency (triangulate every key marked in both "
      "static cameras at the same frame; per-view residual px). Optionally also score as the uncorrected twin would "
      "(no tool correction, no latency) to see what the base.xml hand costs.",
      {"type": "object", "properties": {
          "episode": {"type": "integer"}, "marks": {"type": "array", "items": {"type": "string"}},
          "also_uncorrected": {"type": "boolean", "default": True}},
       "required": ["episode", "marks"]})
def keypoint_check(episode: int, marks, also_uncorrected: bool = True) -> dict:
    from twin import camcal as cc
    cams, lat, tool, bpose = cc.calib_state(episode)
    mk = _load_marks(marks)
    static = [m for m in mk if m["cam"] in ("hama1", "hama2")]
    out = {"tool_m": tool, "latency_frames": lat}
    if static:
        sc = {k: cams[k] for k in ("hama1", "hama2")}
        out["static"] = cc.eval_marks(sc, lat, tool, bpose, static, episode)
        tv = cc.two_view(sc, static)
        out["two_view"] = {"n": len(tv), "median_px": round(float(np.median([t["px"] for t in tv])), 2) if tv else None,
                           "points": tv}
        if also_uncorrected:
            out["static_uncorrected_twin"] = cc.eval_marks(sc, {}, 0.0, bpose, static, episode)
    hand = [m for m in mk if m["cam"] == "hand"]
    if hand:
        kin = cc.Kinematics(episode)
        hb = cc.HandBA(cams["hand"], cc.marks_to_obs(hand), kin, tool, bpose, cc.HandFitConfig())
        e = hb.errors()
        out["hand"] = {"n": len(e), "median_px": round(float(np.median(e)), 2)}
    return out
