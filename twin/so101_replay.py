"""SO-101 track: replay the logged leader commands in the twin (stage T2, reconstruct-scene steps 5-9).

What runs, per episode:
  * scene from twin/so101_scene.py with the frozen calibration (joint map, cameras), table height, box and the
    brick's start pose reconstructed from the cameras at frame 0 (twin/so101_objects.py), nothing else;
  * t < 0: the arm is placed at the follower's logged state of frame 0 and held there by its own position servos
    for `preroll_s` while the brick settles on the table (before t = 0, allowed by rules.md);
  * t >= 0: the follower's targets are the logged leader `action`, mapped through the same joint map, held for one
    30 Hz tick each (what LeRobot's control loop sends), for the episode length plus `after_s`;
  * outcome from simulator state at the end: brick centre inside the box's inner footprint and below its rim.
No forces, welds, state writes or parameter changes after t = 0. Success is never read from the video.

Free-space check: the same replay, compared joint by joint with the follower's logged state while the gripper is
open and empty (before the jaws first close on the brick), converted to millimetres at the pad centre by FK.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np

from twin import so101_data as sd
from twin import so101_robot as sr
from twin import so101_scene as ss

FPS = sd.FPS


class Twin:
    def __init__(self, calib: dict, objects: dict, cfg: dict | None = None):
        self.calib = calib
        self.objects = objects
        self.cfg = {**ss.DEFAULTS, "gripper_ext": float(calib.get("gripper_ext_m", 0.0)),
                    "finger_ext": float(calib.get("finger_ext_m", 0.0)), **(cfg or {})}
        self.jm = sr.JointMap.from_dict(calib["joint_map"])
        spec = ss.build(calib, objects, self.cfg, brick_pose={"pos": [0.2, 0.0, 0.1], "yaw": 0.0})
        self.spec = spec
        self.m = ss.compile_scene(spec, calib, self.cfg["finger_ext"])
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(j).id] for j in sr.JOINTS])
        self.dadr = np.array([self.m.jnt_dofadr[self.m.joint(j).id] for j in sr.JOINTS])
        self.act = np.array([self.m.actuator(j).id for j in sr.JOINTS])
        bj = self.m.joint("brick/free")
        self.bq = self.m.jnt_qposadr[bj.id]
        self.bv = self.m.jnt_dofadr[bj.id]
        self.box_inner = self.m.numeric("box/inner_half").data.copy() if "box" in objects else None
        self.rim = float(self.m.numeric("box/rim_height").data[0]) if "box" in objects else None

    def set_timestep(self, dt: float):
        self.m.opt.timestep = dt

    def reset(self, q0: np.ndarray, brick_pos, brick_yaw: float):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.qadr] = q0
        self.d.qpos[self.bq:self.bq + 3] = brick_pos
        self.d.qpos[self.bq + 3:self.bq + 7] = [np.cos(brick_yaw / 2), 0, 0, np.sin(brick_yaw / 2)]
        self.d.ctrl[self.act] = q0
        mujoco.mj_forward(self.m, self.d)

    def brick(self) -> tuple[np.ndarray, np.ndarray]:
        return self.d.qpos[self.bq:self.bq + 3].copy(), self.d.qpos[self.bq + 3:self.bq + 7].copy()

    def in_box(self) -> dict:
        p, _ = self.brick()
        sid = self.m.site("box/center").id
        R = self.d.site_xmat[sid].reshape(3, 3)
        loc = R.T @ (p - self.d.site_xpos[sid])
        inside = bool(abs(loc[0]) < self.box_inner[0] and abs(loc[1]) < self.box_inner[1] and loc[2] < self.rim)
        return {"inside": inside, "brick_in_box_frame_mm": (loc * 1000).round(1).tolist()}

    def pad_center(self) -> np.ndarray:
        return self.d.site_xpos[self.m.site("grip/pad_center").id].copy()


def brick_start(objects: dict, ep: int) -> dict:
    s = objects["starts"][str(ep)]
    return {"pos": s["pos"], "yaw": s["yaw"]}


def replay(tw: Twin, ep: int, start: dict, preroll_s: float = 0.5, after_s: float = 1.5, displace=None,
           record: bool = False, rec_every: int = 1) -> dict:
    sig = sd.signals(ep)
    units_a = sig["action"]
    units_s = sig["state"]
    q0 = tw.jm.q(units_s[0])
    pos = np.array(start["pos"], float)
    if displace is not None:
        pos = pos + np.asarray(displace, float)
    tw.reset(q0, pos, start["yaw"])
    dt = tw.m.opt.timestep
    n_pre = int(round(preroll_s / dt))
    for _ in range(n_pre):
        mujoco.mj_step(tw.m, tw.d)
        if not np.isfinite(tw.d.qpos).all():
            return {"ok": False, "why": "diverged in preroll"}
    t0 = tw.d.time
    n = len(units_a)
    steps_per_tick = 1.0 / FPS / dt
    qs, bricks, frames = [], [], []
    total = n + int(round(after_s * FPS))
    k_step = 0
    off = np.radians(np.asarray(tw.cfg.get("cmd_offset_deg", [0.0] * 6), float))
    for k in range(total):
        a = units_a[min(k, n - 1)]
        tw.d.ctrl[tw.act] = tw.jm.q(a) + off
        if k < n:
            qs.append(tw.d.qpos[tw.qadr].copy())  # state at the start of tick k (what the follower reads)
            if record and k % rec_every == 0:
                frames.append((k, tw.d.qpos.copy()))
        target = int(round((k + 1) * steps_per_tick))
        while k_step < target:
            mujoco.mj_step(tw.m, tw.d)
            k_step += 1
        if not np.isfinite(tw.d.qpos).all():
            return {"ok": False, "why": f"diverged at tick {k}"}
        bricks.append(tw.brick()[0])
    res = tw.in_box()
    out = {"ok": True, "episode": ep, "success": res["inside"], "brick_final_box_frame_mm": res["brick_in_box_frame_mm"],
           "brick_final_world": tw.brick()[0].round(4).tolist(), "brick_start": pos.round(4).tolist(),
           "brick_max_z_mm": float((np.max(np.array(bricks)[:, 2]) - pos[2]) * 1000), "dt": dt,
           "sim_q": np.array(qs), "real_q": tw.jm.q(units_s)}
    if record:
        out["frames"] = frames
    return out


def grasp_frame(ep: int, thr: float = 3.0) -> int | None:
    """First frame where the follower gripper is held open by the brick: state - action > thr while the leader is
    nearly closed, for >= 5 frames. Label-blind: uses the gripper signal only."""
    s = sd.signals(ep)
    g, a = s["state"][:, 5], s["action"][:, 5]
    blocked = (g - a > thr) & (a < 5)
    for i in range(len(g) - 5):
        if blocked[i:i + 5].all():
            return i
    return None


def close_start(ep: int) -> int | None:
    """Frame where the leader starts closing the gripper for the grasp (last frame before the action falls below
    the follower's open state by 2 units), searched backwards from the grasp frame."""
    gf = grasp_frame(ep)
    if gf is None:
        return None
    s = sd.signals(ep)
    a = s["action"][:, 5]
    i = gf
    amax = a[max(0, gf - 30):gf + 1].max()
    while i > 0 and a[i] < amax - 1.0:
        i -= 1
    return i


def free_space_error(tw: Twin, r: dict, ep: int, lag_frames: int = 0) -> dict:
    """Pad-centre distance (mm) between the sim arm and the logged follower, frames before the first grasp
    contact (gripper empty)."""
    gf = grasp_frame(ep) or len(r["real_q"])
    cs = close_start(ep) or gf
    end = max(5, cs)
    fk = sr.FK(tw.m)
    sid = tw.m.site("grip/pad_center")
    body = tw.m.body("gripper").id
    loc = tw.m.site_pos[sid.id]
    Xs = fk.points(r["sim_q"][:end], np.array([body]), loc[None])[:, 0]
    Xr = fk.points(r["real_q"][:end], np.array([body]), loc[None])[:, 0]
    e = np.linalg.norm(Xs - Xr, axis=1) * 1000
    jerr = np.degrees(np.abs(r["sim_q"][:end, :5] - r["real_q"][:end, :5]))
    return {"episode": ep, "frames": int(end), "median_mm": float(np.median(e)), "p90_mm": float(np.percentile(e, 90)),
            "max_mm": float(e.max()), "joint_median_deg": np.median(jerr, 0).round(2).tolist()}
