"""Cartesian impedance rig: the arm replaced by its task-space behaviour, so thousands of worlds fit on the GPU.

Under the REASSEMBLE controller the end effector behaves as   Lambda a = -K (x - x_d) - D v + W_ext   (twin/control.py).
The rig reproduces that directly: the Franka hand hangs from a body at the logged TCP pose (O_T_EE) through three
slides and three hinges along the TCP axes (at t = 0), each with a position actuator
    slides: kp = K,     kv = diag(D)_trans
    hinges: kp = K_r/2 (the controller's quaternion error is K_r sin(theta/2)), kv = diag(D)_rot
and joint armature that brings the reflected inertia up to diag(Lambda) of the real arm at the window start
(Lambda, D from the full FR3 model at the logged joints, rotated into the rig frame). Off-diagonal Lambda/D terms,
the nullspace and configuration changes during the window are ignored: check against the full-arm replay
(`rig_vs_arm`, twin/replay.py) before trusting it.

Wrist F/T: MuJoCo force/torque sensors on a site at the TCP on the hand body. They measure what the rig applies
to the hand subtree; the external wrench on the tool is minus that (constant hand weight removed by taring).

Per-world variation (pre-t0 only, mujoco_warp batched model fields + initial state): in-hand pose (along-pad
slide, rock tilt, pivot, height, closing offset), part friction, hole widening.
"""

from __future__ import annotations

import math

import numpy as np

from twin.control import ImpedanceParams, damping_matrix, quat_to_mat, stiffness_matrix
from twin.scene.build_base import HAND_TCP, TCP_SITE_Z

RIG_JOINTS = ("rig/x", "rig/y", "rig/z", "rig/rx", "rig/ry", "rig/rz")
FINGERS = ("hand/left_finger", "hand/right_finger")
PAD_Z = 0.045


def _hand_spec():
    import mujoco
    from twin.scene.build_base import MENAGERIE
    spec = mujoco.MjSpec.from_file(str(MENAGERIE / "franka_emika_panda" / "hand.xml"))
    for mesh in spec.meshes:
        mesh.file = str(MENAGERIE / "franka_emika_panda" / "assets" / mesh.file)
    spec.meshdir = ""
    for key in list(spec.keys):
        spec.delete(key)
    return spec


def rig_gains(Lam: np.ndarray, R0: np.ndarray, ip: ImpedanceParams):
    """kp, kv and reflected inertia per rig DoF (rig axes = TCP axes at t = 0)."""
    K = stiffness_matrix(ip)
    D = damping_matrix(Lam, K, ip.zeta)
    B = np.zeros((6, 6))
    B[:3, :3] = R0
    B[3:, 3:] = R0
    Lr = B.T @ Lam @ B
    Dr = B.T @ D @ B
    kp = np.r_[[ip.k_trans] * 3, [ip.k_rot / 2] * 3]
    return kp, np.diag(Dr).copy(), np.diag(Lr).copy()


def build_rig_model(part_name: str, board_pose: dict, rp, x0, q0, Lam: np.ndarray, finger_rock_k: float = 0.0):
    """MjModel: board + part hole, the hand on the 6-DoF impedance rig at TCP pose (x0, q0), part free body
    (placed later per world), F/T sensors 'ft_force'/'ft_torque' at the TCP. Returns (model, info)."""
    import mujoco
    from twin.board import CONTACT, add_board, board_spec
    from twin.parts import add_part_body, load_layout, part_spec
    from twin.replay import impedance
    from dataclasses import replace

    lay = load_layout()
    board = board_spec(lay)
    part = part_spec(part_name, lay)
    if board_pose.get("hole_xy") is not None:
        part = replace(part, hole_xy=tuple(board_pose["hole_xy"]))
    board.params["surface_z"] = replace(board.params["surface_z"], value=float(rp.surface_z))
    board.params["seat_depth"] = replace(board.params["seat_depth"], value=float(rp.seat_depth))
    board.params["plate_thickness"] = replace(board.params["plate_thickness"], value=float(rp.plate_thickness or rp.seat_depth))
    widen = float(rp.hole_widen) + float((rp.hole_widen_by_part or {}).get(part_name, 0.0))
    if widen:
        for k in ("hole_d", "hole_a", "hole_b"):
            if k in part.params:
                part.params[k] = replace(part.params[k], value=part.v(k) + 2 * widen)
    part.params["friction"] = replace(part.params["friction"], value=float(rp.friction))
    part.params["grasp_from_top"] = replace(part.params["grasp_from_top"], value=float(rp.grasp_from_top))
    contact = dict(CONTACT, solref=[float(rp.solref_tc), 1.0], solimp=list(rp.solimp))
    board.contact = dict(contact)

    spec = mujoco.MjSpec()
    spec.modelname = f"rig_{part.slug}"
    spec.compiler.autolimits = True
    spec.option.timestep = rp.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC if rp.cone == "elliptic" else mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.impratio = float(rp.impratio)
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.tolerance = 1e-8
    spec.option.iterations = 50
    spec.option.ls_iterations = 50  # (autoreset stays on: mujoco_warp rejects the flag; non-finite worlds are dropped)
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    wb = spec.worldbody
    wb.add_light(name="key", pos=[0.4, -0.6, 2.0], dir=[0, 0.3, -1], diffuse=[0.6, 0.6, 0.6])
    _, binfo = add_board(spec, wb, [part], board, board_pose["pos"], board_pose["quat"])

    R0 = quat_to_mat(q0)
    kp, kv, lam = rig_gains(Lam, R0, impedance(rp))
    rig = wb.add_body(name="rig", pos=list(map(float, x0)), quat=list(map(float, q0)), mass=1e-3,
                      inertia=[1e-6] * 3, gravcomp=1.0)
    for nm, ax in zip(RIG_JOINTS, ([1, 0, 0], [0, 1, 0], [0, 0, 1]) * 2):
        jt = mujoco.mjtJoint.mjJNT_SLIDE if nm in RIG_JOINTS[:3] else mujoco.mjtJoint.mjJNT_HINGE
        rig.add_joint(name=nm, type=jt, axis=ax, pos=[0, 0, 0])
    hs = _hand_spec()
    hb = hs.body("hand")
    frame = rig.add_frame(pos=[0.0, 0.0, -TCP_SITE_Z], quat=[1, 0, 0, 0])
    frame.attach_body(hb, "hand/", "")
    hand = spec.body("hand/hand")
    hand.add_site(name="tcp", pos=[0, 0, TCP_SITE_Z], size=[0.004, 0, 0], group=4)
    hand.add_site(name="ft", pos=[0, 0, TCP_SITE_Z], size=[0.004, 0, 0], group=4)
    spec.add_sensor(name="ft_force", type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="ft")
    spec.add_sensor(name="ft_torque", type=mujoco.mjtSensor.mjSENS_TORQUE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="ft")
    for b in spec.bodies:
        if b.name and b.name.startswith("hand/"):
            b.gravcomp = 1.0
    # finger rock compliance (pads that conform to a rocked part), spring rest set per world
    for fn in FINGERS:
        if finger_rock_k > 0:
            arm_ = max(1e-6, finger_rock_k * (rp.timestep / 0.4) ** 2)
            spec.body(fn).add_joint(name=f"{fn}/rock", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0],
                                    pos=[0, 0, PAD_Z], stiffness=finger_rock_k, damping=4 * math.sqrt(finger_rock_k * arm_),
                                    armature=arm_, range=[-0.35, 0.35], limited=True)
    # rig actuators (impedance) + gripper servo
    for i, nm in enumerate(RIG_JOINTS):
        a = spec.add_actuator(name=f"act/{nm}", target=nm, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=float(kp[i]), kv=float(max(kv[i], 1e-6)))
        a.ctrllimited = False
    g = spec.actuator("hand/actuator8")
    from twin.replay import GRIP_CLOSED, GRIP_KP
    g.gainprm[0] = GRIP_KP
    g.biasprm[0], g.biasprm[1], g.biasprm[2] = 0.0, -GRIP_KP, -2 * math.sqrt(GRIP_KP * 0.2)
    g.ctrlrange = [GRIP_CLOSED, 0.04]
    g.forcerange = [-2 * rp.grip_force, 2 * rp.grip_force]
    pc = dict(contact, torsional=1e-5)
    add_part_body(spec, wb, part, [0.5, 0.0, 0.5], [1, 0, 0, 0], free=True, contact=pc)
    m = spec.compile()
    # reflected inertia: armature = Lambda diag - what the rig subtree already carries
    mujoco_data = mujoco.MjData(m)
    mujoco.mj_forward(m, mujoco_data)
    rid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rig")
    msub = float(m.body_subtreemass[rid])
    for i, nm in enumerate(RIG_JOINTS):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
        own = msub if i < 3 else 0.01
        m.dof_armature[m.jnt_dofadr[j]] = max(float(lam[i]) - own, 0.05 if i < 3 else 1e-3)
    fids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f) for f in FINGERS}
    for gi in range(m.ngeom):
        if m.geom_bodyid[gi] in fids and m.geom_contype[gi]:
            m.geom_condim[gi] = 4
            m.geom_friction[gi] = [rp.pad_friction, rp.pad_torsion, 1e-5]
            m.geom_solref[gi] = [rp.pad_solref or rp.solref_tc, 1.0]
            m.geom_solimp[gi] = list(rp.solimp)
    info = dict(part=part.describe(), board=binfo, kp=kp.tolist(), kv=kv.tolist(), lam_diag=lam.tolist(),
                subtree_mass=msub, slug=part.slug, length=part.length)
    return m, info


def rig_targets(x0, q0, X, Q):
    """Rig joint targets (6 per frame) for setpoint frames X (n,3), Q (n,4): position and rotation vector
    of the setpoint in the t = 0 TCP frame (hinges in x, y, z order: small-angle rotation vector)."""
    from twin.replay import _quat_log
    R0 = quat_to_mat(q0)
    out = np.zeros((len(X), 6))
    for i in range(len(X)):
        out[i, :3] = R0.T @ (X[i] - x0)
        rv_world = _quat_log(q0, Q[i])  # world-frame rotation vector q0 -> Q[i]
        out[i, 3:] = R0.T @ rv_world
    return out


# ------------------------------------------------------------------------------------------ attempt setup
def prepare_attempt(ep: int, segment_index: int, rp, allowed=("train", "gate"), finger_rock_k: float = 0.0):
    """Everything a rig replay of one attempt needs: signals, window, rebuilt setpoint as rig targets, the rig
    model (with Lambda/D from the full arm at the window start), the registered (label-blind) in-hand pose, and
    the real tared F/T (force on the tool, torque about the TCP, base frame) per frame."""
    import mujoco
    from twin.control import ArmModel
    from twin.replay import (FPS, attempt_info, board_pose_for, build_model, guard, impedance, recording_of,
                             reconstruct_window, register_inhand, segments, signals, tare,
                             wrench_at, _part_mass)
    guard(ep, allowed)
    seg = next(g for g in segments(ep) if g["index"] == int(segment_index))
    info = attempt_info(ep, seg, rp.lead_s)
    if info is None:
        raise ValueError("no part in hand")
    s = signals(ep)
    f = info["frames"]
    k0 = f["start"]
    k1 = min(f["release"] + int(round(rp.after_release_s * FPS)), len(s["t"]) - 1, k0 + int(rp.max_s * FPS))
    part = info["part"]
    bp = board_pose_for(recording_of(ep), rp.board_source, part)
    m_arm, ainfo = build_model(part, bp, rp)
    arm = ArmModel(m_arm)
    d = arm.state_at(s["q"][k0], s["dq"][k0])
    _, _, Lam, _ = arm.lambda_d(d, stiffness_matrix(impedance(rp)), rp.zeta)
    tb = tare(s, k0, _part_mass(m_arm, ainfo["part"]["slug"])) if rp.tare else None
    X, Q = reconstruct_window(m_arm, s, k0, k1, rp, None, tb)
    x0, q0 = s["ee"][k0, :3].copy(), s["ee"][k0, 3:7].copy()
    m, rinfo = build_rig_model(part, bp, rp, x0, q0, Lam, finger_rock_k)
    tgt = rig_targets(x0, q0, X, Q)
    W = np.array([np.r_[wrench_at(s, k, tb)] for k in range(k0, k1 + 1)])  # (n, 6) tared
    W = W - np.median(W[:10], axis=0)  # deltas from the window's first frames (sim is compared the same way)
    reg = register_inhand(bp, part, s, f, rp)
    hold_w = info["grip_width_mm"] / 1000.0
    grip = np.array([(-0.03 if float(s["grip"][k].sum()) < hold_w + 0.003 else float(s["grip"][k].sum()) / 2)
                     for k in range(k0, k1 + 1)])
    return dict(ep=ep, seg=segment_index, info=info, s=s, k0=k0, k1=k1, part=part, bp=bp, Lam=Lam, m=m, rinfo=rinfo,
                tgt=tgt, W_real=W, reg=reg, grip=grip, x0=x0, q0=q0, ts=s["t"][k0:k1 + 1] - s["t"][k0])


def inhand_rotation(rock: float, pivot: float) -> np.ndarray:
    Rx = np.array([[1, 0, 0], [0, math.cos(rock), -math.sin(rock)], [0, math.sin(rock), math.cos(rock)]])
    Ry = np.array([[math.cos(pivot), 0, math.sin(pivot)], [0, 1, 0], [-math.sin(pivot), 0, math.cos(pivot)]])
    return Rx @ Ry


def world_init(m, A: dict, rp, dx: float, dy: float, g: float, rock: float, pivot: float):
    """Initial qpos for one world (part pose from the in-hand pose; fingers at the contact opening)."""
    import mujoco
    from dataclasses import replace
    from twin.parts import load_layout, part_spec
    from twin.replay import part_world_pose
    ps = part_spec(A["part"], load_layout())
    rpw = replace(rp, grasp_dx=dx, grasp_dy=dy, grasp_from_top=g)
    pos, quat = part_world_pose(ps, A["x0"], A["q0"], rpw, inhand_rotation(rock, pivot))
    d = mujoco.MjData(m)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{ps.slug}/free")
    d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 7] = np.r_[pos, quat]
    return d, ps


def finger_opening(m, d, slug: str, dy: float) -> float:
    from twin.replay import close_fingers_on_part
    import mujoco
    # close_fingers_on_part looks fingers up by the replay scene's names; map to the rig's
    j1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hand/finger_joint1")
    j2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hand/finger_joint2")
    fids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f) for f in FINGERS}
    pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, slug)
    pads = [gg for gg in range(m.ngeom) if m.geom_bodyid[gg] in fids and m.geom_contype[gg]]
    pg = [gg for gg in range(m.ngeom) if m.geom_bodyid[gg] == pb]
    ft = np.zeros(6)

    def gap(q):
        d.qpos[m.jnt_qposadr[j1]] = q + dy
        d.qpos[m.jnt_qposadr[j2]] = q - dy
        mujoco.mj_kinematics(m, d)
        return min(mujoco.mj_geomDistance(m, d, a, b, 0.05, ft) for a in pads for b in pg)

    lo, hi = abs(dy), 0.04
    if gap(lo) > 0:
        return lo
    for _ in range(30):
        mid = (lo + hi) / 2
        if gap(mid) > -5e-5:
            hi = mid
        else:
            lo = mid
    gap(hi)
    return hi


def ft_world(m, d) -> np.ndarray:
    """External wrench on the tool from the wrist sensors: -(sensor) rotated to world (force; torque about the TCP)."""
    import mujoco
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ft")
    R = d.site_xmat[sid].reshape(3, 3)
    return np.r_[-R @ d.sensordata[0:3], -R @ d.sensordata[3:6]]


def run_rig_cpu(A: dict, rp, dx=0.0, dy=0.0, g=None, rock=0.0, pivot=0.0, rock_springref=True) -> dict:
    """One world on CPU MuJoCo: same physics as the Warp batch, for validation."""
    import mujoco
    from twin.predicates import insertion_state
    m = A["m"]
    g = rp.grasp_from_top if g is None else g
    d, ps = world_init(m, A, rp, dx, dy, g, rock, pivot)
    for e in range(m.neq):
        m.eq_data[e, 0] = 2 * dy
    q = finger_opening(m, d, ps.slug, dy)
    mujoco.mj_forward(m, d)
    n_rig0 = 1  # actuator 0 is the gripper, 1..6 the rig
    nsub = int(round((1 / 30) / m.opt.timestep))
    tgt, grip = A["tgt"], A["grip"]
    # pre-roll: hold the start pose, close the gripper
    while d.time < rp.preroll_s - 1e-9:
        d.ctrl[0] = -0.03
        d.ctrl[1:7] = 0.0
        mujoco.mj_step(m, d)
    t_pre = d.time
    rows = []
    n = len(tgt)
    for i in range(n):
        st = insertion_state(m, d, ps.slug)
        rows.append(np.r_[ft_world(m, d), st["depth"], st["lateral"], st["tilt_deg"]])
        if i == n - 1:
            break
        for k in range(nsub):
            a = k / nsub
            d.ctrl[1:7] = tgt[i] + a * (tgt[i + 1] - tgt[i])
            d.ctrl[0] = grip[i]
            mujoco.mj_step(m, d)
    rows = np.array(rows)
    st = insertion_state(m, d, ps.slug)
    return dict(ft=rows[:, :6] - np.median(rows[:10, :6], axis=0), ins=rows[:, 6:], seated=bool(st["seated"]),
                final=dict(depth=st["depth"], lateral=st["lateral"], tilt=st["tilt_deg"]))
