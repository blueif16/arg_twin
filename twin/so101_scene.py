"""SO-101 track: the pick-and-place scene as code (MjSpec), stage T2.

Robot: so101_new_calib.xml (twin/assets/so101/SOURCE.md). Its mesh collisions are convex hulls in MuJoCo, and the
hull of the gripper body (servo housing + fixed jaw in one mesh) and of the L-shaped moving jaw would fill the space
between the jaws where the brick sits. So every arm mesh is visual only, and each jaw gets one collision mesh built
from the CAD vertices of its finger alone (convex, so the hull is exact: reconstruct-scene "collision as primitives
where fits matter"). The arm cannot touch the table or box in this twin; only the fingers make contact.

Objects: the LEGO 2x4 brick (box + stud block, free joint) and the transparent box (floor + 4 walls, fixed), both
from twin/so101_objects.py fits. Table: a plane at the fitted height. Cameras `up` and `side` from the calibration.

Controller: the MJCF's position actuators (class sts3215: kp 998.22, kv 2.731, forcerange +-3.35 N m, derived by
TheRobotStudio for STS3215 at P = 16, which is what LeRobot writes: `so_follower.configure`, P 16, I 0, D 32).
The gripper's torque is capped at 50 % in LeRobot (Max_Torque_Limit 500), so its forcerange is halved here.

Sites (task frames, group 4): `brick/center`, `box/center` (floor top centre), `grip/pad_center` (midpoint of the two
finger pads near the tips). Numerics: `box/inner_half`, `box/rim_height`.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from twin import so101_objects as so
from twin import so101_robot as sr

DEFAULTS = {
    "timestep": 0.002,
    "table_z": 0.0,
    "brick_friction": [0.8, 0.01, 0.001],
    "finger_friction": [1.0, 0.02, 0.001],
    "table_friction": [0.5, 0.005, 0.0001],
    "gripper_force_frac": 0.5,
    "fixed_finger_cut": -0.050,   # gripper-frame z below which the fixed jaw is finger
    "moving_finger_cut": 0.030,   # distance from the jaw hinge beyond which the moving jaw is finger
    "box_wall": 0.002,
    "solref": [0.004, 1.0],
    "solimp": [0.95, 0.99, 0.0005, 0.5, 2],
    "kp_scale": 1.0,          # arm servos (joints 0-4): kp and kv relative to the MJCF sts3215 class
    "kv_scale": 1.0,
    "cmd_offset_deg": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # constant added to every follower target (replay freedom)
    "gripper_ext": 0.0,       # m, gripper body moved along its axis (camera-fitted physical tool correction)
    "finger_ext": 0.0,        # m, both jaws lengthened at the tips (camera-measured fingertip covers)
}


def _body_mesh_vertices(m: mujoco.MjModel, body: str) -> np.ndarray:
    b = m.body(body).id
    vs = [sr._mesh_vertices_in_body(m, g) for g in range(m.ngeom)
          if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 3]
    return np.concatenate(vs)


def finger_vertices(cfg: dict | None = None) -> dict[str, np.ndarray]:
    cfg = {**DEFAULTS, **(cfg or {})}
    m = sr.load_model()
    vg = _body_mesh_vertices(m, "gripper")
    vf = vg[vg[:, 2] < cfg["fixed_finger_cut"]]
    vj = _body_mesh_vertices(m, "moving_jaw_so101_v1")
    vm = vj[np.linalg.norm(vj[:, :2], axis=1) > cfg["moving_finger_cut"]]
    return {"fixed": vf, "moving": vm}


def build(calib: dict | None = None, objects: dict | None = None, cfg: dict | None = None,
          brick_pose: dict | None = None) -> mujoco.MjSpec:
    """calib: {'cameras': {name: Camera dict}}; objects: {'table_z', 'box': {...}}; brick_pose: {'pos', 'yaw'}."""
    cfg = {**DEFAULTS, **(cfg or {})}
    objects = objects or {}
    spec = mujoco.MjSpec.from_file(str(sr.MJCF))
    spec.meshdir = str(sr.ASSETS / "assets")
    spec.option.timestep = cfg["timestep"]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET if hasattr(mujoco.mjtDisableBit, "mjDSBL_AUTORESET") else 0

    # physical-tool correction: the gripper body moved along its own axis (fitted with the cameras)
    if cfg["gripper_ext"]:
        gb = spec.body("gripper")
        d = np.array(spec.site("gripperframe").pos, float)
        d /= np.linalg.norm(d)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.array(gb.quat, float))
        gb.pos = (np.array(gb.pos, float) + R.reshape(3, 3) @ (d * cfg["gripper_ext"])).tolist()
    # arm meshes: visual only
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.contype = 0
            g.conaffinity = 0
    # finger collision hulls from CAD vertices
    fv = finger_vertices(cfg)
    for name, body in (("fixed", "gripper"), ("moving", "moving_jaw_so101_v1")):
        mesh = spec.add_mesh(name=f"{name}_finger_hull")
        mesh.uservert = fv[name].ravel().tolist()
        b = spec.body(body)
        g = b.add_geom(name=f"{name}_finger", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{name}_finger_hull")
        g.group = 3
        g.contype, g.conaffinity = 1, 1
        g.condim = 4
        g.friction = cfg["finger_friction"]
        g.solref, g.solimp = cfg["solref"], cfg["solimp"]
        g.rgba = [0.1, 0.8, 0.6, 0.0]
        g.mass = 0.0
        g.density = 0.0
    if cfg["finger_ext"]:
        sr.add_finger_ext(spec, cfg["finger_ext"], collide=True, cfg=cfg)
    # pad centre: midpoint of the fingers' tip regions (filled in after compile, see pad_center_local)
    spec.body("gripper").add_site(name="grip/pad_center", pos=[0, 0, 0], group=4)

    # gripper force cap; arm servo gains
    for a in spec.actuators:
        if a.name == "gripper":
            fr = np.array(a.forcerange) * cfg["gripper_force_frac"]
            a.forcerange = fr.tolist()
        else:
            kp = a.gainprm[0] * cfg["kp_scale"]
            kv = -a.biasprm[2] * cfg["kv_scale"]
            g = list(a.gainprm)
            g[0] = kp
            a.gainprm = g
            b = list(a.biasprm)
            b[1], b[2] = -kp, -kv
            a.biasprm = b

    wb = spec.worldbody
    # table
    tp = objects.get("table_plane")  # z = a x + b y + c (fitted), else flat at table_z
    if tp:
        a, b, c0 = tp
        nrm = np.array([-a, -b, 1.0]) / np.linalg.norm([-a, -b, 1.0])
        ax_ = np.cross([0, 0, 1.0], nrm)
        ang = np.arcsin(min(np.linalg.norm(ax_), 1.0))
        qt = [1, 0, 0, 0] if ang < 1e-9 else [np.cos(ang / 2), *(np.sin(ang / 2) * ax_ / np.linalg.norm(ax_))]
        t = wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], pos=[0, 0, c0], quat=qt,
                        rgba=[0.92, 0.92, 0.93, 1])
    else:
        t = wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01],
                        pos=[0, 0, objects.get("table_z", cfg["table_z"])], rgba=[0.92, 0.92, 0.93, 1])
    t.friction = cfg["table_friction"]
    t.solref, t.solimp = cfg["solref"], cfg["solimp"]
    wb.add_light(name="top", pos=[0.2, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.7, 0.7, 0.7])
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960

    # brick
    if brick_pose is not None:
        B = so.BRICK
        p = brick_pose["pos"]
        yaw = brick_pose.get("yaw", 0.0)
        bb = wb.add_body(name="brick", pos=p, quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        bb.add_freejoint(name="brick/free")
        g = bb.add_geom(name="brick/body", type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[B["length"] / 2, B["width"] / 2, B["height"] / 2], rgba=[0.95, 0.3, 0.3, 1])
        g.mass = B["mass"] * 0.95
        g.friction = cfg["brick_friction"]
        g.condim = 4
        g.solref, g.solimp = cfg["solref"], cfg["solimp"]
        for ix in range(4):
            for iy in range(2):
                s = bb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[B["stud_d"] / 2, B["stud_h"] / 2],
                                pos=[(ix - 1.5) * B["pitch"], (iy - 0.5) * B["pitch"], B["height"] / 2 + B["stud_h"] / 2],
                                rgba=[0.95, 0.3, 0.3, 1])
                s.mass = B["mass"] * 0.05 / 8
                s.friction = cfg["brick_friction"]
                s.solref, s.solimp = cfg["solref"], cfg["solimp"]
        bb.add_site(name="brick/center", pos=[0, 0, 0], group=4)

    # transparent box: floor + 4 walls (outer footprint = tape inner edge)
    box = objects.get("box")
    if box:
        c = box["center"]
        yaw = box["yaw"]
        lx, ly = box["outer"]
        h = box["height"]
        w = cfg["box_wall"]
        bx = wb.add_body(name="box", pos=[c[0], c[1], c[2]], quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        rgba = [0.85, 0.9, 1.0, 0.25]
        parts = [("floor", [lx / 2, ly / 2, w / 2], [0, 0, w / 2]),
                 ("wall_xp", [w / 2, ly / 2, h / 2], [lx / 2 - w / 2, 0, h / 2]),
                 ("wall_xn", [w / 2, ly / 2, h / 2], [-lx / 2 + w / 2, 0, h / 2]),
                 ("wall_yp", [lx / 2, w / 2, h / 2], [0, ly / 2 - w / 2, h / 2]),
                 ("wall_yn", [lx / 2, w / 2, h / 2], [0, -ly / 2 + w / 2, h / 2])]
        for nm, size, pos in parts:
            g = bx.add_geom(name=f"box/{nm}", type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, rgba=rgba)
            g.friction = cfg["table_friction"]
            g.solref, g.solimp = cfg["solref"], cfg["solimp"]
        bx.add_site(name="box/center", pos=[0, 0, w], group=4)
        spec.add_numeric(name="box/inner_half", data=[lx / 2 - w, ly / 2 - w])
        spec.add_numeric(name="box/rim_height", data=[h])

    # cameras
    for name, cd in ((calib or {}).get("cameras") or {}).items():
        wb.add_camera(name=name)
    return spec


def compile_scene(spec: mujoco.MjSpec, calib: dict | None = None, finger_ext: float = 0.0) -> mujoco.MjModel:
    """Compile; put the calibrated cameras in (OpenCV pinhole, no distortion) and the pad-centre site."""
    m = spec.compile()
    for name, cd in ((calib or {}).get("cameras") or {}).items():
        sr.set_camera(m, m.camera(name).id, sr.Camera.from_dict(cd))
    sid = m.site("grip/pad_center").id
    m.site_pos[sid] = pad_center_local(m, finger_ext=finger_ext)
    m.site_sameframe[sid] = 0  # compiled at the body origin, so MuJoCo flagged it 'same frame'; clear that
    return m


def pad_center_local(m: mujoco.MjModel, q_grip: float | None = None, finger_ext: float = 0.0) -> np.ndarray:
    """Pad centre in the gripper frame: midpoint between the fixed finger's tip region and the moving finger's
    tip region (the 15 mm nearest each tip), with the moving jaw at q_grip (default: closed on a brick width)."""
    fv = finger_vertices()
    tips = sr._jaw_tips(sr.load_model())
    fixed = fv["fixed"]
    d = sr.load_model().site_pos[sr.load_model().site("gripperframe").id]
    d = d / np.linalg.norm(d)
    fz = fixed[(fixed @ d) > (tips["fixed_tip"] @ d) - 0.015].mean(0)
    mv = fv["moving"]
    r = np.linalg.norm(mv[:, :2], axis=1)
    mz = mv[r > r.max() - 0.015].mean(0)
    # moving jaw point into the gripper frame at angle q (hinge about jaw z)
    q = 0.0 if q_grip is None else q_grip
    jb = m.body("moving_jaw_so101_v1").id
    Rq = np.array([[np.cos(q), -np.sin(q), 0], [np.sin(q), np.cos(q), 0], [0, 0, 1]])
    Rb = np.zeros(9)
    mujoco.mju_quat2Mat(Rb, m.body_quat[jb])
    # the real pads (fingertip covers) reach finger_ext further: move both tip-region centres half of that
    fz = fz + sr.gripper_axis(sr.load_model()) * finger_ext / 2
    dj = np.array([mz[0], mz[1], 0.0]) / np.linalg.norm(mz[:2])
    mz = mz + dj * finger_ext / 2
    mz_g = Rb.reshape(3, 3) @ (Rq @ mz) + m.body_pos[jb]
    return (fz + mz_g) / 2


def save(spec: mujoco.MjSpec, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    xml = spec.to_xml()
    path.write_text(xml)
    return str(path)
