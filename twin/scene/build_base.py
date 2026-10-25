"""Generate twin/scene/base.xml from MuJoCo Menagerie.

base.xml is the scene the rest of the twin loads. It is self-contained MJCF:
FR3 arm (menagerie franka_fr3) + Franka Hand (menagerie franka_emika_panda/hand.xml)
on a table, a placeholder box for the NIST Assembly Task Board #1, and the three
REASSEMBLE cameras: hand (wrist RealSense D435i), hama1 and hama2 (HAMA C-600 Pro
webcams). Camera poses and the board are placeholders until calibrated.

World frame = FR3 base frame (link0 origin), table top at z = 0.

Run:  .venv/bin/python -m twin.scene.build_base
"""

from __future__ import annotations

import re

import numpy as np
import mujoco

from twin import REPO_DIR, SCENE_DIR, BASE_XML

MENAGERIE = REPO_DIR / "vendors" / "mujoco_menagerie"
MENAGERIE_REL = "../../vendors/mujoco_menagerie"  # relative to twin/scene/

# REASSEMBLE start pose (vendors/reassemble/.../reassemble_control/config/start_pose.yaml)
START_ARM_QPOS = [0.0, -0.785398163397, 0.0, -2.35619449019, 0.0, 1.57079632679, 0.785398163397]
GRIPPER_OPEN = 0.04  # per finger, metres
GRIPPER_CTRL_OPEN = 255.0

# Tool stack (world-frame skill, playbook/skills/build-twin/references/reassemble-world-frame.md). The logged O_T_EE sits 0.174 m from
# the flange (docs/research/reassemble-data-notes.md). The hand sits FT_STACK below the flange (the AIDIN
# F/T sensor + mounts); the `tcp` site sits at TCP_SITE_Z on the hand so it stays on O_T_EE, which is a
# configured point about 23 mm beyond the pad centres, not the fingertips.
TCP_FROM_FLANGE = 0.174
HAND_TCP = 0.1034  # menagerie hand: nominal Franka TCP = pad centre on the hand body
FT_STACK_TCP_MATCH = TCP_FROM_FLANGE - HAND_TCP  # 0.0706 m: the old stack, which put the pad centre on O_T_EE
# Camera bundle adjustment (world-frame skill Lessons, 2026-10-24) finds the physical hand 22.9 mm closer to the
# flange: both static cameras see the pads there. The value is read from the calibration, never typed here.
HAND_CALIB = REPO_DIR / "twin" / "calib" / "2025-01-09-14-11-58.json"


def _hand_offset() -> float:
    import json
    try:
        return float(json.loads(HAND_CALIB.read_text())["tcp"].get("hand_offset_m", 0.0))
    except (OSError, KeyError, ValueError):
        return 0.0


HAND_OFFSET = _hand_offset()                   # -0.0229 m
FT_STACK = FT_STACK_TCP_MATCH + HAND_OFFSET    # 0.0477 m (REASSEMBLE URDF stack: 0.0445)
TCP_SITE_Z = TCP_FROM_FLANGE - FT_STACK        # 0.1263 m on the hand: the logged O_T_EE, ~23 mm beyond the pads
FT_RADIUS = 0.04  # AFT200-D80 is 80 mm across

# Placeholder geometry (metres). Replace with calibrated values.
TABLE_HALF = (0.75, 0.75, 0.02)  # table top slab
TABLE_CENTER = (0.35, 0.0, -0.02)  # top surface at z = 0
BOARD_HALF = (0.20, 0.20, 0.0125)  # NIST ATB#1 placeholder plate
BOARD_POS = (0.45, 0.0, 0.0125)
BOARD_TARGET = (0.45, 0.0, 0.03)
HAMA_TARGET = (0.30, 0.0, 0.30)  # between board and arm so both are in frame


def lookat_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> list[float]:
    """MuJoCo camera xyaxes for a camera at pos looking at target (camera looks along -z, y up)."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, np.asarray(up, float))
    x /= np.linalg.norm(x)
    z = -f
    y = np.cross(z, x)
    return [*x.round(6).tolist(), *y.round(6).tolist()]


def _load(sub: str, fname: str) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MENAGERIE / sub / fname))
    for mesh in spec.meshes:
        mesh.file = str(MENAGERIE / sub / "assets" / mesh.file)
    spec.meshdir = ""
    return spec


def build_spec() -> mujoco.MjSpec:
    fr3 = _load("franka_fr3", "fr3.xml")
    hand = _load("franka_emika_panda", "hand.xml")
    # Drop menagerie keyframes; base.xml defines its own full-state keyframe.
    for key in list(fr3.keys):
        fr3.delete(key)
    for key in list(hand.keys):
        hand.delete(key)

    # Franka Hand on the FR3 flange, rotated -45 deg about z (as in franka_description), pushed out
    # along the flange axis by the F/T sensor stack.
    hand.body("hand").quat = [0.9238795, 0.0, 0.0, -0.3826834]
    hand.body("hand").pos = [0.0, 0.0, FT_STACK]
    fr3.site("attachment_site").attach_body(hand.body("hand"), "gripper/", "")

    # The FR3 spec is the top-level spec (FR3 base frame == world frame). Attaching it into a fresh
    # spec with an empty prefix makes to_xml emit an unnamed default class, which fails to reload.
    spec = fr3
    spec.modelname = "reassemble_twin_base"
    spec.compiler.autolimits = True
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]

    # Assets
    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.85, 0.88, 0.92], rgb2=[0.35, 0.38, 0.42], width=512, height=3072)
    spec.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.42, 0.42, 0.44], rgb2=[0.36, 0.36, 0.38], width=300, height=300)
    floor_mat = spec.add_material(name="floor_mat", texrepeat=[4, 4], reflectance=0.05)
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"
    spec.add_material(name="table_mat", rgba=[0.72, 0.62, 0.50, 1.0], specular=0.1)
    spec.add_material(name="board_mat", rgba=[0.70, 0.70, 0.68, 1.0], specular=0.05)  # matte white PVC base

    wb = spec.worldbody
    wb.add_light(name="key", pos=[0.4, -0.6, 2.0], dir=[0.0, 0.3, -1.0],
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.45, 0.45, 0.45], castshadow=1)
    wb.add_light(name="fill", pos=[1.5, 1.0, 1.5], dir=[-0.6, -0.4, -0.6],
                 diffuse=[0.3, 0.3, 0.3], castshadow=0)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0, 0, -0.75],
                size=[3, 3, 0.05], material="floor_mat")
    wb.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(TABLE_CENTER),
                size=list(TABLE_HALF), material="table_mat")

    # Placeholder NIST Assembly Task Board #1 (static).
    board = wb.add_body(name="task_board", pos=list(BOARD_POS))
    board.add_geom(name="board_plate", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(BOARD_HALF),
                   material="board_mat")
    board.add_site(name="board_center", pos=[0, 0, BOARD_HALF[2]], size=[0.005, 0.005, 0.005])

    # Cameras. hand = wrist RealSense D435i on the hand housing, looking along the approach axis
    # (tilted slightly outward), fingertips at the bottom of the image. Placeholder mount; calibrate later.
    hand_body = spec.body("gripper/hand")
    cam_pos = [0.065, 0.0, 0.035]
    hand_body.add_camera(name="hand", pos=cam_pos,
                         xyaxes=lookat_xyaxes(cam_pos, [0.08, 0.0, 0.45], up=[1.0, 0.0, 0.0]),
                         fovy=58.0)
    hand_body.add_site(name="tcp", pos=[0, 0, TCP_SITE_Z], size=[0.004, 0.004, 0.004],
                       rgba=[1, 0, 0, 0.5], group=4)
    # F/T sensor stack between flange and hand: visual only (the hand's explicit inertial keeps mass).
    hand_body.add_geom(name="ft_stack", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, -FT_STACK / 2],
                       size=[FT_RADIUS, FT_STACK / 2, 0], rgba=[0.62, 0.63, 0.65, 1.0], contype=0,
                       conaffinity=0, group=2)

    # hama1 / hama2 = external HAMA C-600 Pro webcams. Placeholder fixed poses, front-left and
    # front-right of the board. Real poses come per-recording from <ts>_poses.json (mocap).
    for name, pos in (("hama1", [1.45, 0.70, 0.85]), ("hama2", [1.45, -0.70, 0.85])):
        wb.add_camera(name=name, pos=pos, xyaxes=lookat_xyaxes(pos, HAMA_TARGET), fovy=50.0)

    # Overview camera for debugging (not a REASSEMBLE camera).
    ov = [1.6, 0.9, 0.9]
    wb.add_camera(name="overview", pos=ov, xyaxes=lookat_xyaxes(ov, [0.3, 0.0, 0.25]), fovy=45.0)

    return spec


def add_keyframe(spec: mujoco.MjSpec, model: mujoco.MjModel) -> None:
    qpos = np.zeros(model.nq)
    qpos[:7] = START_ARM_QPOS
    qpos[7:9] = GRIPPER_OPEN
    ctrl = np.zeros(model.nu)
    ctrl[:7] = START_ARM_QPOS
    ctrl[7] = GRIPPER_CTRL_OPEN
    spec.add_key(name="reassemble_start", qpos=qpos.tolist(), ctrl=ctrl.tolist())


def main() -> None:
    spec = build_spec()
    model = spec.compile()
    assert model.nq == 9 and model.nu == 8, (model.nq, model.nu)
    add_keyframe(spec, model)
    ref = spec.compile()

    xml = spec.to_xml().replace(str(MENAGERIE), MENAGERIE_REL)
    # to_xml drops mass="0" density="0" from fr3's collision default (MuJoCo 3.17), which would give
    # the massless-collision link0 a geom-derived mass. Restore it; the round-trip check guards this.
    xml, n = re.subn(r'(<default class="collision">\s*<geom type="mesh" group="3")/>',
                     r'\1 mass="0" density="0"/>', xml)
    assert n == 1, "fr3 collision default not found in to_xml output"
    header = (
        "<!-- GENERATED by twin/scene/build_base.py from vendors/mujoco_menagerie "
        "(franka_fr3 + franka_emika_panda/hand.xml).\n"
        "     Edit the builder (or this file, then stop regenerating). Board and camera poses are "
        f"PLACEHOLDERS until calibrated. Hand: F/T stack {FT_STACK:.4f} m (hand_offset_m {HAND_OFFSET:+.4f} from "
        f"{HAND_CALIB.name}); tcp site {TCP_SITE_Z:.4f} m on the hand = logged O_T_EE. -->\n"
    )
    BASE_XML.write_text(header + xml)

    # Round-trip check: the written XML must compile to the same model as the spec.
    m2 = mujoco.MjModel.from_xml_path(str(BASE_XML))
    for attr in ("body_pos", "body_quat", "geom_pos", "geom_size", "geom_type", "cam_pos",
                 "cam_quat", "jnt_range", "actuator_gainprm", "actuator_biasprm", "body_mass"):
        a, b = getattr(ref, attr), getattr(m2, attr)
        assert a.shape == b.shape and np.allclose(a, b, atol=1e-6), f"round-trip mismatch in {attr}"
    print(f"wrote {BASE_XML.relative_to(REPO_DIR)}: nq={m2.nq} nv={m2.nv} nu={m2.nu} "
          f"nbody={m2.nbody} ngeom={m2.ngeom} ncam={m2.ncam} nmesh={m2.nmesh}")


if __name__ == "__main__":
    main()
