"""fx copy of lstate.py: sim-state API over a custom-BDDL LIBERO env (runs under .venv-libero; imports playgrounds/libero/sim.py read-only).

TaskEnv(task) builds one OffScreenRenderEnv from the task's BDDL and exposes:
  load(flat_state)        set the flattened MuJoCo state and run mj_forward (contacts, poses)
  api_state(step, cmd)    the documented state dict handed to Argon-written code (API_DOC below)
  truth()                 LIBERO's own env.check_success() (ground truth; never shown to Argon)
  render(cam, size)       RGB frame, true (unmirrored) camera view
"""
from __future__ import annotations

import sys
from pathlib import Path

import os
os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/ran/arg_twin/runs/libero/config")
os.environ["MUJOCO_GL"] = "glx"
import numpy as np  # noqa: E402

CONTROL_HZ = 20


def r(x, n=4):
    return [round(float(v), n) for v in np.asarray(x).ravel()]


class TaskEnv:
    def __init__(self, bddl_path: str, cam: int = 256, seed: int = 142):
        """Copied from runs/argon/probes/reward/scripts/lstate.py (read-only source); built from any BDDL path."""
        from libero.libero.envs import OffScreenRenderEnv
        self.task_name, self.language = str(bddl_path), None
        self.env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=cam, camera_widths=cam)
        self.env.seed(seed)
        self.env.reset()
        e = self.env.env
        self.e = e
        self.sim = e.sim
        self.m = e.sim.model._model
        self.d = e.sim.data._data
        import mujoco
        self.mj = mujoco
        m = self.m
        self.objects = list(e.objects_dict.keys())
        self.fixtures = list(e.fixtures_dict.keys())
        # geom -> owner name
        owner = {}
        for name in self.objects + self.fixtures:
            for g in e.get_object(name).contact_geoms:
                gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                if gid >= 0:
                    owner[gid] = name
        grip = e.robots[0].gripper.important_geoms
        for side, key in (("gripper_left_finger", "left_finger"), ("gripper_right_finger", "right_finger")):
            for g in grip[key]:
                owner[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)] = side
        for gid in range(m.ngeom):
            if gid in owner:
                continue
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]) or ""
            if bname.startswith("gripper0") or gname.startswith("gripper0"):
                owner[gid] = "gripper_hand"
            elif bname.startswith("robot0") or gname.startswith("robot0") or bname.startswith("mount0"):
                owner[gid] = "robot_arm"
            elif "table" in gname or "table" in bname:
                owner[gid] = "table"
            elif "floor" in gname:
                owner[gid] = "floor"
            else:
                # walk up to the top body and match an object/fixture prefix
                b = m.geom_bodyid[gid]
                while m.body_parentid[b] != 0:
                    b = m.body_parentid[b]
                top = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                match = [n for n in self.objects + self.fixtures if top.startswith(n)]
                owner[gid] = max(match, key=len) if match else "other"
        self.owner = owner
        self.collision = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.geoms_of = {n: [g for g in self.collision if owner.get(g) == n] for n in self.objects + self.fixtures}
        self.body_id = {n: e.obj_body_id[n] for n in self.objects + self.fixtures}
        self.free_dof = {}
        for n in self.objects:
            j = e.get_object(n).joints[0]
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            self.free_dof[n] = (int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid]))
        self.fixture_joints = {}
        for n in self.fixtures:
            js = []
            for j in getattr(e.get_object(n), "joints", []) or []:
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                if jid < 0:
                    continue
                kind = {2: "slide", 3: "hinge"}.get(int(m.jnt_type[jid]), "other")
                js.append((j, int(m.jnt_qposadr[jid]), kind, r(m.jnt_range[jid])))
            self.fixture_joints[n] = js
        self.finger_adr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)])
                           for j in ("gripper0_finger_joint1", "gripper0_finger_joint2")]
        self.eef_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        tg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_collision")
        self.table_top_z = None
        if tg >= 0:
            self.table_top_z = float(self.d.geom_xpos[tg][2] + m.geom_size[tg][2]) if m.geom_type[tg] == 6 else None
        self.state_dim = int(self.sim.get_state().flatten().shape[0])

    # ------------------------------------------------------------------ state
    def load(self, flat):
        self.sim.set_state_from_flattened(np.asarray(flat, dtype=np.float64))
        self.sim.forward()

    def set_qpos_free(self, name, pos=None, quat=None):
        qa, _ = self.free_dof[name]
        if pos is not None:
            self.d.qpos[qa:qa + 3] = pos
        if quat is not None:
            self.d.qpos[qa + 3:qa + 7] = quat
        _, da = self.free_dof[name]
        self.d.qvel[da:da + 6] = 0
        self.sim.forward()

    def flat(self):
        return np.asarray(self.sim.get_state().flatten(), dtype=np.float64)

    def truth(self) -> bool:
        return bool(self.env.check_success())

    def aabb(self, name):
        m, d = self.m, self.d
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for g in self.geoms_of[name]:
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            R = d.geom_xmat[g].reshape(3, 3)
            ctr = d.geom_xpos[g] + R @ c
            ext = np.abs(R) @ h
            lo = np.minimum(lo, ctr - ext)
            hi = np.maximum(hi, ctr + ext)
        if not np.isfinite(lo).all():
            p = d.xpos[self.body_id[name]]
            return p, p
        return lo, hi

    def api_state(self, step: int, gripper_cmd: float) -> dict:
        m, d = self.m, self.d
        objs = {}
        for n in self.objects:
            b = self.body_id[n]
            qa, da = self.free_dof[n]
            lo, hi = self.aabb(n)
            R = d.xmat[b].reshape(3, 3)
            objs[n] = {"pos": r(d.xpos[b]), "quat": r(d.xquat[b]), "up_axis": r(R[:, 2]),
                       "lin_vel": r(d.qvel[da:da + 3]), "ang_vel": r(d.qvel[da + 3:da + 6]),
                       "aabb_min": r(lo), "aabb_max": r(hi)}
        fixs = {}
        for n in self.fixtures:
            b = self.body_id[n]
            lo, hi = self.aabb(n)
            fixs[n] = {"pos": r(d.xpos[b]), "quat": r(d.xquat[b]), "aabb_min": r(lo), "aabb_max": r(hi),
                       "joints": {j: {"q": round(float(d.qpos[a]), 5), "type": k, "range": rg}
                                  for j, a, k, rg in self.fixture_joints[n]}}
        pairs = set()
        lf, rf = set(), set()
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = self.owner.get(c.geom1, "other"), self.owner.get(c.geom2, "other")
            if a == b:
                continue
            pairs.add(tuple(sorted((a, b))))
            for x, y in ((a, b), (b, a)):
                if x == "gripper_left_finger":
                    lf.add(y)
                if x == "gripper_right_finger":
                    rf.add(y)
        q1, q2 = d.qpos[self.finger_adr[0]], d.qpos[self.finger_adr[1]]
        eef = d.site_xpos[self.eef_site]
        R = d.site_xmat[self.eef_site].reshape(3, 3)
        q = np.zeros(4)
        self.mj.mju_mat2Quat(q, d.site_xmat[self.eef_site])
        return {"step": int(step), "t": round(step / CONTROL_HZ, 3),
                "objects": objs, "fixtures": fixs,
                "contacts": [list(p) for p in sorted(pairs)],
                "gripper": {"eef_pos": r(eef), "eef_quat": r(q), "finger_qpos": r([q1, q2], 5),
                            "width": round(float(q1 - q2), 5), "command": float(gripper_cmd),
                            "left_finger_contacts": sorted(lf), "right_finger_contacts": sorted(rf)},
                "table_top_z": None if self.table_top_z is None else round(self.table_top_z, 4)}

    # ------------------------------------------------------------------ images
    def render(self, cam="agentview", size=512):
        img = self.sim.render(width=size, height=size, camera_name=cam)
        # MuJoCo returns rows bottom-up; [::-1] is the true camera view. (sim.py's [::-1, ::-1] for agentview is a
        # left-right mirror of it: the agentview camera's right axis is world +y, so +y must be on the image right.)
        return img[::-1].copy()

    def close(self):
        self.env.close()
