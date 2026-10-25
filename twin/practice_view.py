"""Observation images for practice episodes: the visual coordinate harness (stage D; skill playbook/skills/practice).

What a brain sees when it asks for an image of a practice episode:

  pixels    the twin's TRUE state rendered from the calibrated cameras (hama1, hama2, wrist "hand"), at their native
            640 x 480, as a real camera would see the scene. Static views may be cropped around the believed hole and
            upscaled (no extra detail: the crop is resampled from the native render, like a crop of a real frame).
            The render camera carries a fixed calibration error of the size the calibration measured on held-out
            marks (CAM_ERR_PX), so overlays sit as far off the pixels as they would on the real robot.
  overlays  drawn from what the actor BELIEVES, never from sim truth (Google's zero-shot arm prompt,
            docs/research/google-argon-twin-pipeline.md; [google-argon-robotics-2026]):
              yellow   world grid at the believed plate top, lines every 1 cm, labels = world (X, Y) in cm
              green    believed hole: outline (section + clearance) at the entry plane, centre, and the hole axis
                       drawn 60 mm out of the hole
              magenta  believed part: axis from the pad centre to the tip under the NOMINAL grasp, the tip, and the
                       tip's foot on the grid plane
              cyan     pinch centre (+) and the jaw closing line [ ] at the current gripper opening
            The believed board is the env's task() hole pose (the nominal contact-fit board; the true board differs
            by the start's board_err, which nobody outside the env and the evaluator sees).

The env process owns the simulator and offers no render op, so this module keeps a SHADOW copy of the episode: a fresh
PracticeEnv (same frozen start, cousin and code; record=False, no dt/2 rerun, never scored) that replays the actor's
logged commands (runs/calls/, the registry's replayable record) and is checked against the logged observations.
MuJoCo is deterministic here: two envs fed the same commands end bit-identical (checked 2026-10-25). The shadow lives
in a small daemon per session (`serve`), so each image costs only the sim time since the last one.
A shadow never writes episode records, never claims held-out attempts and never returns evaluator output.

Rendering needs MUJOCO_GL=glfw on this box.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import socket
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")

from twin import REPO_DIR  # noqa: E402
from twin.env import PracticeEnv  # noqa: E402

W, H = 640, 480
CALIB_RECORDING = "2025-01-09-14-11-58"   # twin/starts.py NOMINAL_RECORDING: the recording with calibrated cameras
# Held-out reprojection error of each camera's calibration (twin/calib/<rec>.json cameras.*.status): finger pads
# 1.7 px (hama1) / 1.0 px (hama2); wrist camera: board leave-one-hole-out 4.9 px. The render camera is turned by
# this many pixels (fixed per run) away from the calibrated camera used for the overlays.
CAM_ERR_PX = {"hama1": 1.7, "hama2": 1.0, "hand": 4.9}
VIEWS = ("hand", "hama1", "hama2")
GRID_STEP_M = 0.01
GRID_HALF_M = {"hand": 0.04, "hama1": 0.06, "hama2": 0.06}
AXIS_LEN_M = 0.06
ENV_TOOLS = ("env_reset", "env_move", "env_gripper", "env_wait", "env_finish")
CALLS_DIR = REPO_DIR / "runs" / "calls"

YELLOW = (255, 214, 10)
GREEN = (40, 220, 90)
MAGENTA = (255, 60, 220)
CYAN = (0, 230, 255)
WHITE = (245, 245, 245)


# ============================================================================================== small math
def quat_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mat_quat(R) -> np.ndarray:
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def axis_angle(axis, ang) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


FLIP = np.diag([1.0, -1.0, -1.0])   # MuJoCo camera axes (x right, y up, looks along -z) -> OpenCV (y down, z forward)


class Cam:
    """A pinhole camera in its parent frame (calibration convention of twin/calib/<rec>.json)."""

    def __init__(self, name: str, parent: str, pos, quat, fx, fy, cx, cy, width=W, height=H):
        self.name, self.parent = name, parent
        self.pos = np.asarray(pos, float)
        self.R = quat_mat(quat)          # MuJoCo camera frame in the parent frame
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.width, self.height = int(width), int(height)

    @classmethod
    def from_calib(cls, d: dict) -> "Cam":
        return cls(d["name"], d.get("parent", "world"), d["pos"], d["quat"], d["fx"], d["fy"], d["cx"], d["cy"],
                   d.get("width", W), d.get("height", H))

    @property
    def quat(self) -> np.ndarray:
        return mat_quat(self.R)

    def perturbed(self, px: float, seed: int) -> "Cam":
        """The same camera turned by `px` pixels about a fixed random axis in the image plane."""
        rng = np.random.default_rng(seed)
        a = rng.uniform(0, 2 * math.pi)
        axis_cam = np.array([math.cos(a), math.sin(a), 0.0])
        c = copy.copy(self)
        c.R = self.R @ axis_angle(axis_cam, px / self.fx)
        return c

    def project(self, X_world, parent_R=np.eye(3), parent_p=np.zeros(3)) -> tuple[np.ndarray, np.ndarray]:
        P = (np.atleast_2d(np.asarray(X_world, float)) - parent_p) @ parent_R
        Xc = (P - self.pos) @ (self.R @ FLIP)
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.stack([self.fx * Xc[:, 0] / z + self.cx, self.fy * Xc[:, 1] / z + self.cy], 1)
        return uv, z


def load_cams(recording: str = CALIB_RECORDING) -> dict[str, Cam]:
    cal = json.loads((REPO_DIR / "twin" / "calib" / f"{recording}.json").read_text())
    return {k: Cam.from_calib(v) for k, v in cal["cameras"].items() if k in VIEWS}


# ============================================================================================== shadow env
class ShadowEnv(PracticeEnv):
    """A PracticeEnv that never records or scores anything visible, with an optional frame callback (clips)."""

    def __init__(self, *a, **kw):
        kw.update(record=False, dt_half=False)
        kw.setdefault("budget", dict(max_commands=10 ** 7, max_sim_s=1e7, max_wall_s=1e9, max_tokens=None))
        super().__init__(*a, **kw)
        self.frame_cb = None
        self.frame_dt = 0.1
        self.next_frame = 0.0

    def _tick(self):
        super()._tick()
        if self.frame_cb is not None and self.ep is not None and self._t() + 1e-9 >= self.next_frame:
            self.frame_cb(self)
            self.next_frame += self.frame_dt

    def _actor_report(self, rec: dict) -> dict:  # a shadow's outcome is never handed to anyone
        return dict(note="shadow episode: no report")


def _req(tool: str, args: dict) -> dict | None:
    a = {k: v for k, v in args.items() if k not in ("session", "brain", "run_id")}
    if tool == "env_move":
        keys = ("pos", "quat", "speed", "force_max", "stop_force", "timeout")
        return {"op": "move", **{k: a[k] for k in keys if a.get(k) is not None}}
    if tool == "env_gripper":
        return {"op": "gripper", "action": a["action"], **({"width": a["width"]} if a.get("width") is not None else {})}
    if tool == "env_wait":
        return {"op": "wait", "seconds": a["seconds"]}
    if tool == "env_finish":
        return {"op": "finish", **({"claim": a["claim"]} if a.get("claim") is not None else {})}
    return None


def _call_files(n: int = 3) -> list[Path]:
    return sorted(CALLS_DIR.glob("*.jsonl"))[-n:] if CALLS_DIR.exists() else []


def episode_log(session: str, brain: str = "claude", run_id: str = "run-standin-1") -> dict:
    """The current episode of a session as the registry logged it: start id, the commands the env accepted, and the
    last observation. Episode key = the time of its env_reset call."""
    ep = None
    for f in _call_files():
        with open(f) as fh:
            for line in fh:
                if '"env_' not in line or session not in line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tool, args = r.get("tool"), r.get("args") or {}
                if tool not in ENV_TOOLS or args.get("session") != session:
                    continue
                if args.get("brain", "claude") != brain or args.get("run_id", "run-standin-1") != run_id:
                    continue
                res = r.get("result") or {}
                if not r.get("ok") or "error" in res:
                    continue
                if tool == "env_reset":
                    ep = dict(key=r["t"], start_id=args["start_id"], commands=[], obs=res.get("obs"), finished=False,
                              task=res.get("task"), over=False)
                    continue
                if ep is None:
                    continue
                req = _req(tool, args)
                ep["commands"].append(req)
                if tool == "env_finish":
                    ep["finished"] = True
                    ep["obs"] = None
                else:
                    ep["obs"] = res.get("obs", ep["obs"])
                    ep["over"] = bool(res.get("episode_over")) or ep["over"]
    if ep is None:
        raise ValueError(f"no logged env_reset for session {session!r} ({brain}/{run_id}) in {CALLS_DIR}")
    return ep


# ============================================================================================== rendering
class SceneRenderer:
    """Renders a copy of an episode's model posed from its MjData. The physics model is never touched: cameras are
    set on a deep copy, and the copy's own MjData gets the positions before each render."""

    def __init__(self, m, cams: dict[str, Cam], err_seed: int):
        import mujoco
        self.mj = mujoco
        self.m = copy.deepcopy(m)
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, W)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, H)
        # Appearance only, on the render copy: a hole seen from above reads dark on the real board; in the twin the
        # base under the plate (the hole's floor) has the plate's flat grey, which hides the hole at the static
        # cameras. The base is darkened; wall tops stay flush and plate-coloured.
        for g in range(self.m.ngeom):
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if nm == "board/base":
                self.m.geom_matid[g] = -1
                self.m.geom_rgba[g] = [0.08, 0.08, 0.09, 1.0]
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, H, W)
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0
        self.cams_belief = cams
        self.cams_true = {k: c.perturbed(CAM_ERR_PX.get(k, 0.0), err_seed + i) for i, (k, c) in enumerate(sorted(cams.items()))}
        for c in self.cams_true.values():
            self._apply(c)

    def close(self):
        try:
            self.r.close()
        except Exception:  # noqa: BLE001
            pass

    def _apply(self, c: Cam):
        mj, m = self.mj, self.m
        i = mj.mj_name2id(m, mj.mjtObj.mjOBJ_CAMERA, c.name)
        want = 0 if c.parent == "world" else mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, c.parent)
        if i < 0 or m.cam_bodyid[i] != want:
            raise ValueError(f"camera {c.name} missing or not on {c.parent}")
        m.cam_pos[i] = c.pos
        m.cam_quat[i] = c.quat
        m.cam_mode[i] = mj.mjtCamLight.mjCAMLIGHT_FIXED
        m.cam_resolution[i] = [W, H]
        m.cam_sensorsize[i] = [W, H]
        m.cam_intrinsic[i] = [c.fx, c.fy, W / 2 - c.cx - 0.5, H / 2 - c.cy - 0.5]   # as twin/world.py TwinView
        m.cam_fovy[i] = float(np.degrees(2 * np.arctan((H / 2) / c.fy)))

    def pose(self, d_src):
        mj = self.mj
        self.d.qpos[:] = d_src.qpos
        self.d.qvel[:] = d_src.qvel
        if self.m.nmocap:
            self.d.mocap_pos[:] = d_src.mocap_pos
            self.d.mocap_quat[:] = d_src.mocap_quat
        mj.mj_kinematics(self.m, self.d)
        mj.mj_camlight(self.m, self.d)

    def parent_pose(self, parent: str) -> tuple[np.ndarray, np.ndarray]:
        if parent == "world":
            return np.eye(3), np.zeros(3)
        b = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_BODY, parent)
        return self.d.xmat[b].reshape(3, 3).copy(), self.d.xpos[b].copy()

    def rgb(self, name: str) -> np.ndarray:
        self.r.update_scene(self.d, camera=name, scene_option=self.opt)
        return self.r.render().copy()


def _font(size: int):
    from PIL import ImageFont
    try:
        import matplotlib
        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"), size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def belief(task: dict, obs: dict) -> dict:
    """Believed geometry from the task (nominal hole, nominal grasp) and the robot's own state. No sim truth."""
    He = np.asarray(task["hole"]["entry_pos"], float)
    Re = quat_mat(task["hole"]["entry_quat"])            # columns x_e, y_e, z_e (z_e INTO the hole)
    p = np.asarray(obs["tcp"]["pos"], float)
    Rh = quat_mat(obs["tcp"]["quat"])                    # hand axes; z out of the fingers
    g = task["grasp_nominal"]
    tip = p + Rh @ np.asarray(g["tip_in_tcp_frame_m"], float)
    pad = p - Rh[:, 2] * g["pad_to_tcp_mm"] / 1000.0
    rel = Re.T @ (tip - He)                              # tip in the hole frame (z_e into the hole)
    cosang = float(np.clip(Rh[:, 2] @ Re[:, 2], -1, 1))
    F = np.asarray(obs["wrench"]["force_N"], float)
    w = float(obs["gripper_width_m"])
    sec = task.get("part_geometry", {}).get("section_mm", [0, 0])
    # the actor's own belief that it holds the part: the fingers stopped at about the part's width
    held = min(sec) / 1000 - 0.003 <= w <= max(sec) / 1000 + 0.003
    return dict(He=He, Re=Re, tcp=p, Rh=Rh, tip=tip, pad=pad, tip_hole_mm=rel * 1000, tilt_deg=math.degrees(math.acos(cosang)),
                F=F, width=w, held=held, t=float(obs["t"]))


def _hole_outline(task: dict, b: dict, n: int = 40) -> np.ndarray:
    sx, sy = task["part_geometry"]["section_mm"]
    cx, cy = task["hole"]["clearance_per_side_mm"]
    He, Re = b["He"], b["Re"]
    if task.get("family") == "round_peg":
        r = (sx / 2 + cx) / 1000
        a = np.linspace(0, 2 * math.pi, n)
        loc = np.stack([r * np.cos(a), r * np.sin(a), np.zeros(n)], 1)
    else:
        ax, ay = (sx / 2 + cx) / 1000, (sy / 2 + cy) / 1000
        loc = np.array([[-ax, -ay, 0], [ax, -ay, 0], [ax, ay, 0], [-ax, ay, 0], [-ax, -ay, 0]])
    return He + loc @ Re.T


def _draw_poly(dr, cam: Cam, pts, Rp, pp, fill, width=2, xf=None, dash=0):
    uv, z = cam.project(pts, Rp, pp)
    if xf is not None:
        uv = xf(uv)
    for i in range(len(uv) - 1):
        if z[i] <= 0.01 or z[i + 1] <= 0.01 or not np.isfinite(uv[i:i + 2]).all():
            continue
        if dash and i % 2:
            continue
        dr.line([tuple(uv[i]), tuple(uv[i + 1])], fill=fill, width=width)


def _draw_view(img, cam: Cam, rnd: SceneRenderer, task: dict, b: dict, xf, scale: float, view: str):
    """Overlays on one view. xf maps native pixel coordinates to the (cropped, scaled) panel."""
    from PIL import ImageDraw
    dr = ImageDraw.Draw(img, "RGBA")
    Rp, pp = rnd.parent_pose(cam.parent)
    zg = float(b["He"][2])
    half = GRID_HALF_M[view]
    x0, y0 = (round(float(b["He"][0]) / GRID_STEP_M) * GRID_STEP_M, round(float(b["He"][1]) / GRID_STEP_M) * GRID_STEP_M)
    ticks = np.arange(-half, half + 1e-9, GRID_STEP_M)
    f = _font(11)
    seg = np.linspace(-half, half, 25)
    for k in ticks:
        major = abs(round((k + 1e-9) / GRID_STEP_M)) % 2 == 0
        col = YELLOW + ((200,) if major else (110,))
        _draw_poly(dr, cam, np.stack([np.full_like(seg, x0 + k), y0 + seg, np.full_like(seg, zg)], 1), Rp, pp, col, 1, xf)
        _draw_poly(dr, cam, np.stack([x0 + seg, np.full_like(seg, y0 + k), np.full_like(seg, zg)], 1), Rp, pp, col, 1, xf)
    # labels on the grid border: X (cm) along the near edge, Y (cm) along the side edge, every 2 cm
    for k in ticks:
        if round(k / GRID_STEP_M) % 2:
            continue
        for P, s_ in (([x0 + k, y0 - half, zg], f"X{100 * (x0 + k):.0f}"), ([x0 - half, y0 + k, zg], f"Y{100 * (y0 + k):.0f}")):
            uv, z = cam.project([P], Rp, pp)
            if z[0] <= 0.01:
                continue
            u, v = xf(uv)[0]
            if 0 <= u < img.width - 30 and 0 <= v < img.height - 12:
                dr.text((u + 2, v + 1), s_, fill=YELLOW + (255,), font=f, stroke_width=2, stroke_fill=(0, 0, 0, 200))
    # believed hole: outline, centre, axis out of the hole
    He, Re = b["He"], b["Re"]
    _draw_poly(dr, cam, _hole_outline(task, b), Rp, pp, GREEN + (255,), 2, xf)
    _draw_poly(dr, cam, np.stack([He - Re[:, 2] * s for s in np.linspace(0, AXIS_LEN_M, 13)]), Rp, pp, GREEN + (230,), 2, xf, dash=1)
    uv, z = cam.project([He], Rp, pp)
    if z[0] > 0.01:
        u, v = xf(uv)[0]
        dr.line([(u - 6, v), (u + 6, v)], fill=GREEN + (255,), width=1)
        dr.line([(u, v - 6), (u, v + 6)], fill=GREEN + (255,), width=1)
    # believed part axis (nominal grasp), tip, foot of the tip on the grid plane: only while it is believed held
    if b["held"]:
        _draw_poly(dr, cam, np.stack([b["pad"], b["tip"]]), Rp, pp, MAGENTA + (255,), 3, xf)
        foot = np.array([b["tip"][0], b["tip"][1], zg])
        _draw_poly(dr, cam, np.stack([b["tip"] + (foot - b["tip"]) * s for s in np.linspace(0, 1, 9)]), Rp, pp,
                   MAGENTA + (160,), 1, xf, dash=1)
        for P, r_px in ((b["tip"], 5), (foot, 3)):
            uv, z = cam.project([P], Rp, pp)
            if z[0] > 0.01:
                u, v = xf(uv)[0]
                dr.ellipse([u - r_px, v - r_px, u + r_px, v + r_px], outline=MAGENTA + (255,), width=2)
    # pinch centre (+) and the jaw closing line [ ] at the current opening
    Rh, pad, w = b["Rh"], b["pad"], b["width"]
    uv, z = cam.project([pad], Rp, pp)
    if z[0] > 0.01:
        u, v = xf(uv)[0]
        dr.line([(u - 9, v), (u + 9, v)], fill=CYAN + (255,), width=2)
        dr.line([(u, v - 9), (u, v + 9)], fill=CYAN + (255,), width=2)
    for s in (-1, 1):
        c = pad + s * Rh[:, 1] * w / 2
        _draw_poly(dr, cam, np.stack([c - Rh[:, 0] * 0.008, c + Rh[:, 0] * 0.008]), Rp, pp, CYAN + (255,), 3, xf)
        _draw_poly(dr, cam, np.stack([c, c - s * Rh[:, 1] * 0.003]), Rp, pp, CYAN + (255,), 2, xf)
    fl = _font(14)
    dr.text((6, 4), {"hand": "wrist camera (native)", "hama1": "fixed camera hama1", "hama2": "fixed camera hama2"}[view]
            + ("" if scale == 1 else f"  (crop x{scale:g}, resampled)"), fill=WHITE + (255,), font=fl,
            stroke_width=2, stroke_fill=(0, 0, 0, 220))


def header_lines(b: dict, task: dict) -> list[str]:
    dx, dy, dz = b["tip_hole_mm"]
    F = b["F"]
    part = (f"believed tip vs believed hole entry (hole frame, mm): x {dx:+.1f}  y {dy:+.1f}  depth {dz:+.1f} "
            f"(negative = above the surface)   part-axis tilt to hole axis {b['tilt_deg']:.1f} deg") if b["held"] else \
        "part not believed held (gripper width far from the part's width): no part overlay"
    return [
        f"t {b['t']:.2f} s   {task.get('part', '')}   " + part,
        f"TCP (m) {b['tcp'][0]:.4f} {b['tcp'][1]:.4f} {b['tcp'][2]:.4f}   F base (N) {F[0]:+.1f} {F[1]:+.1f} {F[2]:+.1f} "
        f"(|F| {np.linalg.norm(F):.1f}; +z = pressing down)   gripper {1000 * b['width']:.1f} mm   grid 1 cm at plate top "
        f"z {1000 * b['He'][2]:.1f} mm, border labels X / Y in world cm",
        "overlays are BELIEFS (nominal board and grasp): yellow grid, green hole + axis, magenta part axis + tip + "
        "tip foot, cyan pinch centre + jaws. Pixels are the cameras.",
    ]


def _wrap(lines: list[str], font, width: int) -> list[str]:
    out = []
    for s in lines:
        cur = ""
        for word in s.split(" "):
            t = (cur + " " + word) if cur else word
            if font.getlength(t) <= width or not cur:
                cur = t
            else:
                out.append(cur)
                cur = word
        out.append(cur)
    return out


def compose(rnd: SceneRenderer, task: dict, obs: dict, views=VIEWS, zoom: float = 3.0, title: str = "",
            overlays: bool = True) -> tuple:
    """One PNG-ready composite: the requested views side by side under a text header. Returns (PIL image, belief)."""
    from PIL import Image, ImageDraw
    b = belief(task, obs)
    panels = []
    for v in views:
        cam_t, cam_b = rnd.cams_true[v], rnd.cams_belief[v]
        img = Image.fromarray(rnd.rgb(v))
        scale = 1.0
        xf = lambda uv: uv  # noqa: E731
        if v != "hand" and zoom and zoom > 1:
            Rp, pp = rnd.parent_pose(cam_b.parent)
            uv, _ = cam_b.project([b["He"]], Rp, pp)
            cw, ch = W / zoom, H / zoom
            u0 = float(np.clip(uv[0][0] - cw / 2, 0, W - cw))
            v0 = float(np.clip(uv[0][1] - ch * 0.62, 0, H - ch))     # hole a bit below centre: room for the hand
            img = img.resize((W, H), Image.BILINEAR, box=(u0, v0, u0 + cw, v0 + ch))
            scale = float(zoom)
            xf = (lambda u0_, v0_, s_: (lambda uv: np.stack([(uv[:, 0] + 0.5 - u0_) * s_ - 0.5,
                                                              (uv[:, 1] + 0.5 - v0_) * s_ - 0.5], 1)))(u0, v0, scale)
        if overlays:
            _draw_view(img, cam_b, rnd, task, b, xf, scale, v)
        panels.append(img)
    f = _font(14)
    hl = header_lines(b, task)
    lines = _wrap(([title] if title else []) + (hl if overlays else hl[:-1] + ["overlays off: pixels only"]),
                  f, W * len(panels) - 16)
    hh = 8 + 19 * len(lines)
    out = Image.new("RGB", (W * len(panels), H + hh), (18, 18, 22))
    for i, p in enumerate(panels):
        out.paste(p, (W * i, hh))
    dr = ImageDraw.Draw(out)
    for i, s in enumerate(lines):
        dr.text((8, 4 + 19 * i), s, fill=WHITE, font=f)
    return out, b


def err_seed(run_id: str) -> int:
    return int(hashlib.sha256(f"calib-error:{run_id}".encode()).hexdigest()[:8], 16)


# ============================================================================================== shadow daemon
class Shadow:
    """One session's shadow: replays the logged commands of the current episode and renders it."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.env: ShadowEnv | None = None
        self.key = None
        self.applied: list[dict] = []
        self.rnd: SceneRenderer | None = None
        self.cams = load_cams()

    def _fresh(self, start_id: str):
        if self.rnd is not None:
            self.rnd.close()
        c = self.cfg
        self.env = ShadowEnv(c["brain"], c["run_id"], c["start_set"], c.get("cousin", "c000"), actor="obs-shadow",
                             session=c["session"] + "-shadow")
        self.env.reset(start_id)
        self.applied = []
        self.rnd = SceneRenderer(self.env.ep["m"], self.cams, err_seed(c["run_id"]))

    def sync(self, key, start_id: str, commands: list[dict], expect: dict | None) -> dict:
        t0 = time.perf_counter()
        replayed = "incremental"
        if key != self.key or self.env is None or commands[:len(self.applied)] != self.applied:
            self._fresh(start_id)
            self.key = key
            replayed = "from_reset"
        for c in commands[len(self.applied):]:
            if self.env.ep["done"]:
                break
            self.env.command(c)
            self.applied.append(c)
        out = dict(replayed=replayed, commands=len(self.applied), sync_s=round(time.perf_counter() - t0, 2))
        if expect:
            obs = self.env.observe()
            dt = abs(obs["t"] - expect["t"])
            dp = float(np.linalg.norm(np.subtract(obs["tcp"]["pos"], expect["tcp"]["pos"])))
            out.update(match=bool(dt < 1e-6 and dp < 1e-5), dt_s=dt, dpos_m=dp)
        return out

    def render(self, req: dict) -> dict:
        s = self.sync(req["key"], req["start_id"], req["commands"], req.get("expect"))
        if s.get("match") is False:   # one rebuild before reporting a divergence
            self.key = None
            s = self.sync(req["key"], req["start_id"], req["commands"], req.get("expect"))
        if s.get("match") is False:
            raise RuntimeError(f"shadow replay does not reproduce the live state (off by {s.get('dt_s')} s, "
                               f"{s.get('dpos_m')} m): refusing to render a state that is not the robot's")
        obs = self.env.observe()
        self.rnd.pose(self.env.ep["d"])
        img, b = compose(self.rnd, req["task"], obs, req.get("views") or VIEWS, req.get("zoom", 3.0),
                         overlays=req.get("overlays", True))
        out = Path(req["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        return dict(sync=s, belief=dict(t=round(b["t"], 3), tip_vs_hole_mm=[round(float(v), 2) for v in b["tip_hole_mm"]],
                                        tilt_deg=round(b["tilt_deg"], 2)))


def serve(sock_path: str, cfg: dict, idle_s: float = 1800.0):
    sh = Shadow(cfg)
    p = Path(sock_path)
    p.unlink(missing_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(p))
    os.chmod(p, 0o600)
    srv.listen(2)
    srv.settimeout(idle_s)
    try:
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                break
            with conn:
                conn.settimeout(600)
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = conn.recv(1 << 20)
                    if not chunk:
                        break
                    buf += chunk
                try:
                    req = json.loads(buf.decode() or "{}")
                    if req.get("op") == "close":
                        conn.sendall(b'{"closed": true}\n')
                        break
                    rep = sh.render(req) if req.get("op") == "render" else dict(error=f"unknown op {req.get('op')!r}")
                except Exception as ex:  # noqa: BLE001  (a bug must not kill the daemon)
                    import traceback
                    rep = dict(error=f"{type(ex).__name__}: {ex}", trace=traceback.format_exc(limit=4))
                conn.sendall((json.dumps(rep, default=float) + "\n").encode())
    finally:
        srv.close()
        p.unlink(missing_ok=True)


def call_socket(sock_path: str, req: dict, timeout: float = 900.0) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(str(sock_path))
    with s:
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode())


# ============================================================================================== clips
def episode_clip(rec: dict, run_dir: Path, out: Path, fps: float = 10.0, views=("hand", "hama1"), zoom: float = 3.0,
                 speed: float = 1.0, title: str = "") -> dict:
    """Replay a finished episode record (its frozen start, cousin and command log) in a shadow env and write an H.264
    clip of the overlaid views, one frame per 1/fps sim s, played back at `speed` x sim time."""
    import av
    task_cache = {}
    env = ShadowEnv(rec["brain"], rec["run_id"], rec["start_set"], rec["cousin"], actor="obs-clip",
                    session=rec["session"] + "-clip", run_dir=run_dir)
    r = env.reset(rec["start_id"])
    task_cache["task"] = r["task"]
    rnd = SceneRenderer(env.ep["m"], load_cams(), err_seed(rec["run_id"]))
    frames = []

    def grab(e):
        rnd.pose(e.ep["d"])
        img, _ = compose(rnd, task_cache["task"], e.observe(), views, zoom, title)
        frames.append(np.asarray(img))

    env.frame_dt = 1.0 / fps
    env.next_frame = 0.0
    env.frame_cb = grab
    grab(env)
    env.next_frame = env.frame_dt
    for c in rec["commands"]:
        if env.ep["done"]:
            break
        env.command(c["req"])
    if not env.ep["done"]:
        env.command({"op": "finish"})
    rnd.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    with av.open(str(out), "w") as cont:
        st = cont.add_stream("libx264", rate=int(round(fps * speed)))
        st.width, st.height, st.pix_fmt = w2, h2, "yuv420p"
        st.options = {"crf": "23", "preset": "veryfast"}
        for fr in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(fr[:h2, :w2]), format="rgb24")):
                cont.mux(pkt)
        for pkt in st.encode():
            cont.mux(pkt)
    return dict(file=str(out.relative_to(REPO_DIR)) if out.is_relative_to(REPO_DIR) else str(out), frames=len(frames),
                sim_s=round(env._t(), 2), fps=fps * speed)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--socket", required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--idle-s", type=float, default=1800.0)
    a = ap.parse_args()
    if a.cmd == "serve":
        serve(a.socket, json.loads(Path(a.config).read_text()), a.idle_s)


if __name__ == "__main__":
    main()
