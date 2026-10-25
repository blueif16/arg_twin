"""Insertion probe: does a hole behave like a hole?

A scripted fixture, not a robot: a Cartesian rig (slides x, y along the entry frame, slide z along the
insertion axis, optional passive hinges about the grasp point) holds the part rigidly at its grasp site
and pushes it along the hole axis with a force-limited position servo. The rig is part of the model
from t = 0, so the part still moves only through mj_step contact with the board (rules.md).

Start: tip `start_gap` above the entry, shifted laterally by `offset_mm` along the entry x axis (or
`offset_dir_deg` from it). The z target ramps at `speed` to `seat_depth + 5 mm`, so a blocked part sees
exactly `push_force` from the servo (plus its own weight). Lateral servo stiffness `lateral_kp`
(N/m; 'stiff' = 2e5, 'compliant' = 500 like REASSEMBLE's impedance default) holds the offset; hinges
with `rot_k` Nm/rad (0 = rigid wrist) let the part tilt.

Expected if the geometry is right: aligned -> seated; offset beyond clearance with a stiff rig -> jams
on the rim at ~0 depth; with a compliant rig a chamfered part recentres for offsets up to about
clearance + chamfer.
"""

from __future__ import annotations

import math
import time

import numpy as np

from twin.board import BoardSpec, add_board, entry_frame, mat_to_quat
from twin.parts import MM, PartSpec, add_part_body
from twin.predicates import insertion_state

STIFF_KP = None  # rigid lateral axis
COMPLIANT_KP = 500.0


def build_probe_spec(part: PartSpec, board: BoardSpec, offset_mm: float = 0.0, offset_dir_deg: float = 0.0,
                     start_gap: float = 0.003, lateral_kp: float | None = STIFF_KP, rot_k: float = 0.0,
                     push_force: float = 10.0, timestep: float = 0.001, cone: str = "pyramidal",
                     tilt_deg: float = 0.0):
    import mujoco

    spec = mujoco.MjSpec()
    spec.modelname = f"probe_{part.slug}"
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC if cone == "elliptic" else mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET  # surface NaNs instead of silent resets
    spec.compiler.autolimits = True
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    wb = spec.worldbody
    wb.add_light(name="key", pos=[0.2, -0.3, 1.0], dir=[0, 0.3, -1], diffuse=[0.7, 0.7, 0.7])
    _, info = add_board(spec, wb, [part], board)

    R, _ = entry_frame(part)
    th = math.radians(offset_dir_deg)
    lat = offset_mm * MM * np.array([math.cos(th), math.sin(th)])
    entry = np.array([*part.hole_xy, board.v("surface_z")])
    tip0 = entry + R @ np.array([lat[0], lat[1], -start_gap])
    grasp0 = tip0 + R[:, 2] * (part.grasp_z - part.length / 2)  # grasp is (L/2 - grasp_z) behind the tip

    rig = wb.add_body(name="probe/rig", pos=grasp0.tolist(), quat=mat_to_quat(R), mass=0.001,
                      inertia=[1e-6, 1e-6, 1e-6])
    # Armatures keep the explicit servo/spring stiffness stable: omega*dt < ~0.5 at dt = 2 ms.
    # A rigid lateral axis (lateral_kp=None) has no joint at all: the part can only move along z.
    lat_axes = [] if lateral_kp is None else [([1, 0, 0], "x"), ([0, 1, 0], "y")]
    for ax, nm in lat_axes + [([0, 0, 1], "z")]:
        rig.add_joint(name=f"probe/{nm}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=ax,
                      armature=max(0.1, (lateral_kp or 0) * (timestep / 0.5) ** 2))
    if rot_k > 0:
        arm = max(1e-4, rot_k * (timestep / 0.5) ** 2)
        for ax, nm in (([1, 0, 0], "rx"), ([0, 1, 0], "ry")):
            rig.add_joint(name=f"probe/{nm}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=ax, stiffness=rot_k,
                          damping=2 * math.sqrt(rot_k * arm), armature=arm)
    from twin.board import CONTACT
    # optional held tilt of the part about the entry x axis through the grasp point (crooked grasp)
    ht = math.radians(tilt_deg) / 2
    qt = [math.cos(ht), math.sin(ht), 0.0, 0.0]
    Rt = np.array([[1, 0, 0], [0, math.cos(2 * ht), -math.sin(2 * ht)], [0, math.sin(2 * ht), math.cos(2 * ht)]])
    add_part_body(spec, rig, part, pos=(Rt @ [0, 0, -part.grasp_z]).tolist(), quat=qt, free=False, contact=CONTACT)

    for nm, kp in ((n_, lateral_kp) for _, n_ in lat_axes):
        a = spec.add_actuator(name=f"probe/{nm}", target=f"probe/{nm}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=kp, dampratio=1.0)
    a = spec.add_actuator(name="probe/z", target="probe/z", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    a.set_to_position(kp=3000.0, dampratio=1.0)
    a.forcelimited = True
    a.forcerange = [-push_force, push_force]
    cam_pos = (entry + np.array([0.06, -0.06, 0.05])).tolist()
    from twin.scene.build_base import lookat_xyaxes
    wb.add_camera(name="probe_side", pos=cam_pos, xyaxes=lookat_xyaxes(cam_pos, entry.tolist()), fovy=30)
    return spec, info


def run_probe(part: PartSpec, board: BoardSpec, offset_mm: float = 0.0, offset_dir_deg: float = 0.0,
              timestep: float = 0.001, push_force: float = 10.0, speed: float = 0.02,
              lateral: str | float = "stiff", rot_k: float = 0.0, start_gap: float = 0.003,
              settle_s: float = 0.5, cone: str = "pyramidal", trace_dt: float = 0.01, tilt_deg: float = 0.0,
              render_path: str | None = None) -> dict:
    import mujoco

    kp = {"stiff": STIFF_KP, "compliant": COMPLIANT_KP}.get(lateral, lateral) if isinstance(lateral, str) \
        else float(lateral)
    spec, info = build_probe_spec(part, board, offset_mm, offset_dir_deg, start_gap, kp, rot_k,
                                  push_force, timestep, cone, tilt_deg)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    seat = board.v("seat_depth")
    z_end = start_gap + seat + 0.005
    T = z_end / speed + settle_s
    iz = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "probe/z")
    pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, part.slug)
    moving_mass = float(m.body_subtreemass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "probe/rig")])
    trace, max_pen, max_resist, max_lat = [], 0.0, 0.0, 0.0
    next_rec = 0.0
    t0 = time.perf_counter()
    nan = False
    while d.time < T:
        d.ctrl[iz] = min(speed * d.time, z_end)
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
            nan = True
            break
        for i in range(d.ncon):
            c = d.contact[i]
            if pb in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                max_pen = max(max_pen, -float(c.dist))
        if d.time >= next_rec:
            s = insertion_state(m, d, part.slug)
            max_resist = max(max_resist, s.get("axial_resist", 0.0))
            max_lat = max(max_lat, s["lateral"])
            trace.append([round(d.time, 4), round(s["depth"] / MM, 3), round(s["lateral"] / MM, 4),
                          round(s["tilt_deg"], 3), round(s.get("axial_resist", 0.0), 3)])
            next_rec += trace_dt
    wall = time.perf_counter() - t0
    s = insertion_state(m, d, part.slug)
    resist = s.get("axial_resist", 0.0)
    weight = moving_mass * 9.81
    if nan:
        outcome = "nan"
    elif s["seated"]:
        outcome = "seated"
    elif resist >= 0.8 * push_force and s["depth"] < 0.003:
        outcome = "jammed_at_rim"
    elif resist >= 0.8 * push_force:
        outcome = "jammed_in_hole"
    else:
        outcome = "partial"
    res = dict(part=part.name, offset_mm=offset_mm, offset_dir_deg=offset_dir_deg, timestep=timestep,
               lateral_kp=kp, rot_k=rot_k, tilt_deg=tilt_deg, push_force=push_force, speed=speed, cone=cone,
               outcome=outcome, final_depth_mm=round(s["depth"] / MM, 3), seat_depth_mm=round(seat / MM, 3),
               final_lateral_mm=round(s["lateral"] / MM, 4), final_tilt_deg=round(s["tilt_deg"], 3),
               final_axial_resist_N=round(resist, 3), max_axial_resist_N=round(max_resist, 3),
               max_lateral_mm=round(max_lat / MM, 4), max_penetration_mm=round(max_pen / MM, 4),
               moving_weight_N=round(weight, 3), clearance_mm=[round(c / MM, 4) for c in part.radial_clearance],
               chamfer_mm=round(part.v("chamfer_frac") * max(part.section) / MM, 3),
               sim_s=round(d.time, 3), wall_s=round(wall, 3), steps=int(round(d.time / timestep)),
               n_plate_geoms=int(m.ngeom), trace_cols=["t", "depth_mm", "lateral_mm", "tilt_deg", "resist_N"],
               trace=trace)
    if render_path:
        with mujoco.Renderer(m, 480, 640) as r:
            opt = mujoco.MjvOption()
            opt.sitegroup[4] = 1
            r.update_scene(d, camera="probe_side", scene_option=opt)
            from PIL import Image
            Image.fromarray(r.render()).save(render_path)
        res["render"] = render_path
    return res
