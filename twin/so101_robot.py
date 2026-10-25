"""SO-101 track: the robot model, the dataset-unit -> joint-angle map, forward kinematics, cameras and renders.

Joint units (evidence: lerobot 0.6.1 `motors/motors_bus.py::_normalize`, `robots/so_follower/so_follower.py::calibrate`):
  * calibration puts the operator's "middle of range" pose at raw 2047 (half turn), then records each joint's
    min/max raw ticks by hand; wrist_roll is a full-turn joint with min 0, max 4095.
  * RANGE_M100_100: norm = (raw - min) / (max - min) * 200 - 100, so norm is linear in the angle; 0 is the middle
    of the recorded range. RANGE_0_100 (gripper): norm = (raw - min) / (max - min) * 100, 0 = closed.
  * raw ticks are 4096 per turn, so angle = raw * 2*pi/4096 + const.
  => per joint q = scale * norm + offset, with scale = (max - min) * 2*pi / (200 * 4096) and an offset that depends
  on where the operator's middle pose and range ends were. The dataset carries no calibration file, so scale and
  offset are fitted per joint from the video (twin/so101_camcal.py). wrist_roll's scale is known exactly:
  4095 * 2*pi / (200 * 4096) rad per unit.
  The prior (H0) maps -100..100 onto the MJCF joint range of so101_new_calib.xml (TheRobotStudio says its zero is the
  middle of each joint's range).

Camera convention: OpenCV pinhole, x_cam = R @ X_world + t, pixel (u, v) with pixel centres at integer indices.
World = the SO-101 base body frame of so101_new_calib.xml (z up; the table top is fitted, near z = 0).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "twin" / "assets" / "so101"
MJCF = ASSETS / "so101_new_calib.xml"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ROLL_SCALE = 4095 * 2 * np.pi / (200 * 4096)  # rad per dataset unit, wrist_roll (full-turn calibration)
W, H = 640, 480

# Keypoints a person can mark, as (body, point in body frame). The horn centre is the gripper servo axis on the
# outer face of the white horn (the moving jaw's hinge); the jaw tips are the CAD tips (the real jaws wear teal caps).
# Body-frame coordinates were read off the CAD meshes (twin/so101_robot.py::_jaw_tips computes the tips).
KEY_DESCR = {
    "teal_cap": "centre of the teal cap on the tip of the moving (white) jaw (3D: CAD tip + fitted cap offset)",
    "fixed_tip": "visible end of the fixed (black) jaw (3D: CAD tip + fitted finger extension along the gripper axis)",
    "moving_tip": "visible end of the moving (white) jaw, i.e. the dark end of its teal cap (3D: CAD tip + finger extension)",
}


@dataclass
class JointMap:
    """q[rad] = scale * units + offset, per joint (6)."""
    scale: np.ndarray
    offset: np.ndarray
    source: str = "H0"
    sigma: dict = field(default_factory=dict)

    @staticmethod
    def h0(model: mujoco.MjModel | None = None) -> "JointMap":
        m = model or load_model()
        lo = np.array([m.jnt_range[m.joint(j).id][0] for j in JOINTS])
        hi = np.array([m.jnt_range[m.joint(j).id][1] for j in JOINTS])
        scale = (hi - lo) / 200.0
        offset = (hi + lo) / 2.0
        scale[4], offset[4] = ROLL_SCALE, 0.0
        scale[5], offset[5] = (hi[5] - lo[5]) / 100.0, lo[5]
        return JointMap(scale, offset, "H0: -100..100 (gripper 0..100) spans the MJCF joint range; wrist_roll exact scale")

    def q(self, units: np.ndarray) -> np.ndarray:
        u = np.asarray(units, float)
        return u * self.scale + self.offset

    def units(self, q: np.ndarray) -> np.ndarray:
        return (np.asarray(q, float) - self.offset) / self.scale

    def to_dict(self) -> dict:
        return {"joints": list(JOINTS), "scale_rad_per_unit": self.scale.tolist(), "offset_rad": self.offset.tolist(),
                "source": self.source, "sigma": self.sigma}

    @staticmethod
    def from_dict(d: dict) -> "JointMap":
        return JointMap(np.array(d["scale_rad_per_unit"], float), np.array(d["offset_rad"], float),
                        d.get("source", ""), d.get("sigma", {}))


@dataclass
class Camera:
    """OpenCV pinhole: x_cam = R X + t; u = fx x/z + cx, v = fy y/z + cy (pixel centres at integers)."""
    R: np.ndarray
    t: np.ndarray
    f: float
    cx: float = (W - 1) / 2
    cy: float = (H - 1) / 2
    k1: float = 0.0

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """X (..., 3) world -> uv (..., 2), depth z (...)."""
        Xc = X @ self.R.T + self.t
        z = Xc[..., 2]
        x = Xc[..., 0] / z
        y = Xc[..., 1] / z
        if self.k1:
            r2 = x * x + y * y
            x, y = x * (1 + self.k1 * r2), y * (1 + self.k1 * r2)
        return np.stack([self.f * x + self.cx, self.f * y + self.cy], -1), z

    def ray(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel -> (camera centre, unit direction in world). Ignores k1 (pinhole)."""
        uv = np.atleast_2d(uv)
        d = np.stack([(uv[:, 0] - self.cx) / self.f, (uv[:, 1] - self.cy) / self.f, np.ones(len(uv))], -1)
        d = d @ self.R  # R^T d
        return self.center, d / np.linalg.norm(d, axis=1, keepdims=True)

    def to_dict(self) -> dict:
        return {"R": self.R.tolist(), "t": self.t.tolist(), "f": self.f, "cx": self.cx, "cy": self.cy, "k1": self.k1,
                "center_world_m": self.center.tolist(), "convention": "OpenCV x_cam = R X + t; pixel centres at integers"}

    @staticmethod
    def from_dict(d: dict) -> "Camera":
        return Camera(np.array(d["R"], float), np.array(d["t"], float), float(d["f"]), float(d["cx"]), float(d["cy"]),
                      float(d.get("k1", 0.0)))

    @staticmethod
    def look_at(pos, target, f, up=(0, 0, 1), cx=(W - 1) / 2, cy=(H - 1) / 2) -> "Camera":
        pos, target, up = (np.asarray(v, float) for v in (pos, target, up))
        z = target - pos
        z /= np.linalg.norm(z)
        x = np.cross(z, up)  # right = forward x up ... (OpenCV: x right, y down)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        return Camera(R, -R @ pos, float(f), cx, cy)


def load_model(path: Path | str = MJCF, ext: float = 0.0, finger_ext: float = 0.0) -> mujoco.MjModel:
    """The SO-101 model; `ext` (m) moves the gripper body (fixed jaw, servo, moving jaw) along its own axis
    (towards gripperframe); `finger_ext` (m) lengthens both jaws at their tips (the real follower's fingertip covers
    reach beyond the CAD tips). Both are physical-tool corrections fitted with the cameras (twin/so101_camcal.py)."""
    if finger_ext:
        spec = mujoco.MjSpec.from_file(str(path))
        spec.meshdir = str(ASSETS / "assets")
        add_finger_ext(spec, finger_ext)
        m = spec.compile()
    else:
        m = mujoco.MjModel.from_xml_path(str(path))
    if ext:
        set_ext(m, ext)
    return m


TIP_REGION = 0.012  # m: CAD vertices this close to a jaw tip (along the jaw) form the tip cross-section


def finger_tip_vertices(m: mujoco.MjModel | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per jaw: (tip-region CAD vertices in its body frame, unit direction the jaw points)."""
    m = m or mujoco.MjModel.from_xml_path(str(MJCF))
    tips = _jaw_tips(m)
    out = {}
    gb = m.body("gripper").id
    ax = gripper_axis(m)
    vg = np.concatenate([_mesh_vertices_in_body(m, g) for g in range(m.ngeom)
                         if m.geom_bodyid[g] == gb and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 3])
    vg = vg[(vg @ ax) > (tips["fixed_tip"] @ ax) - TIP_REGION]
    out["fixed"] = (vg, ax)
    jb = m.body("moving_jaw_so101_v1").id
    vj = np.concatenate([_mesh_vertices_in_body(m, g) for g in range(m.ngeom)
                         if m.geom_bodyid[g] == jb and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 3])
    t = tips["moving_tip"]
    dj = np.array([t[0], t[1], 0.0]) / np.linalg.norm(t[:2])
    vj = vj[(vj @ dj) > (t @ dj) - TIP_REGION]
    out["moving"] = (vj, dj)
    return out


def add_finger_ext(spec: mujoco.MjSpec, L: float, collide: bool = False, cfg: dict | None = None) -> None:
    """Add one convex mesh per jaw: the tip cross-section swept L metres further along the jaw (visual group 2,
    and collision group 3 when `collide`)."""
    base = mujoco.MjModel.from_xml_path(str(MJCF))
    for name, body in (("fixed", "gripper"), ("moving", "moving_jaw_so101_v1")):
        v, d = finger_tip_vertices(base)[name]
        pts = np.concatenate([v, v + d * L])
        mesh = spec.add_mesh(name=f"{name}_tip_ext")
        mesh.uservert = pts.ravel().tolist()
        g = spec.body(body).add_geom(name=f"{name}_tip_ext", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{name}_tip_ext")
        g.group = 3 if collide else 2
        g.contype, g.conaffinity = (1, 1) if collide else (0, 0)
        g.rgba = [0.1, 0.55, 0.5, 1] if name == "moving" else [0.1, 0.1, 0.1, 1]
        g.mass = 0.0
        g.density = 0.0
        if collide and cfg:
            g.condim = 4
            g.friction = cfg["finger_friction"]
            g.solref, g.solimp = cfg["solref"], cfg["solimp"]


_EXT_BASE: dict = {}


def set_ext(m: mujoco.MjModel, ext: float) -> None:
    """Shift the gripper body by `ext` metres along the gripper's axis (in place, idempotent per model)."""
    gid = m.body("gripper").id
    key = id(m)
    if key not in _EXT_BASE:
        _EXT_BASE[key] = m.body_pos[gid].copy()
    sp = m.site_pos[m.site("gripperframe").id]
    d = sp / np.linalg.norm(sp)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.body_quat[gid])
    m.body_pos[gid] = _EXT_BASE[key] + R.reshape(3, 3) @ (d * ext)


def gripper_axis(m: mujoco.MjModel) -> np.ndarray:
    sp = m.site_pos[m.site("gripperframe").id]
    return sp / np.linalg.norm(sp)


def robot_bodies(m: mujoco.MjModel) -> list[int]:
    names = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")
    return [m.body(n).id for n in names]


def keypoints_local(m: mujoco.MjModel | None = None, cap: float = 0.0, finger_ext: float = 0.0) -> dict[str, tuple[str, np.ndarray]]:
    """name -> (body, point in the body frame), from the CAD meshes; teal_cap moves the CAD moving-jaw tip by `cap`
    metres along its radial direction; the visible jaw ends sit `finger_ext` beyond the CAD tips (fitted with the
    cameras, twin/so101_camcal.py)."""
    m = m or mujoco.MjModel.from_xml_path(str(MJCF))
    tips = _jaw_tips(m)
    t = tips["moving_tip"]
    d = np.array([t[0], t[1], 0.0]) / np.linalg.norm(t[:2])
    return {
        "teal_cap": ("moving_jaw_so101_v1", t + cap * d),
        "fixed_tip": ("gripper", tips["fixed_tip"] + finger_ext * gripper_axis(m)),
        "moving_tip": ("moving_jaw_so101_v1", t + finger_ext * d),
    }


def _mesh_vertices_in_body(m: mujoco.MjModel, gid: int) -> np.ndarray:
    mid = m.geom_dataid[gid]
    a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    v = m.mesh_vert[a:a + n].astype(float)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[gid])
    return v @ R.reshape(3, 3).T + m.geom_pos[gid]


def _jaw_tips(m: mujoco.MjModel) -> dict[str, np.ndarray]:
    """CAD jaw tips: the vertex furthest along the jaw direction (gripper body -z, i.e. towards gripperframe)."""
    gb = m.body("gripper").id
    jb = m.body("moving_jaw_so101_v1").id
    site = m.site("gripperframe")
    out = {}
    # fixed jaw: vertices of the gripper body's meshes; direction = towards gripperframe from the body origin
    d = m.site_pos[site.id] / np.linalg.norm(m.site_pos[site.id])
    vs = np.concatenate([_mesh_vertices_in_body(m, g) for g in range(m.ngeom)
                         if m.geom_bodyid[g] == gb and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH])
    out["fixed_tip"] = vs[np.argmax(vs @ d)]
    vj = np.concatenate([_mesh_vertices_in_body(m, g) for g in range(m.ngeom)
                         if m.geom_bodyid[g] == jb and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH])
    # moving jaw: the vertex furthest from its hinge axis (z of the jaw frame) within the jaw plane
    r = np.linalg.norm(vj[:, :2], axis=1)
    out["moving_tip"] = vj[np.argmax(r)]
    return out


class FK:
    """Batch forward kinematics on a private MjData."""

    def __init__(self, m: mujoco.MjModel | None = None):
        self.m = m or load_model()
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(j).id] for j in JOINTS])

    def poses(self, q: np.ndarray, bodies: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """q (N, 6) rad -> xpos (N, B, 3), xmat (N, B, 3, 3)."""
        q = np.atleast_2d(q)
        P = np.zeros((len(q), len(bodies), 3))
        Rm = np.zeros((len(q), len(bodies), 3, 3))
        for i, qi in enumerate(q):
            self.d.qpos[self.qadr] = qi
            mujoco.mj_kinematics(self.m, self.d)
            P[i] = self.d.xpos[bodies]
            Rm[i] = self.d.xmat[bodies].reshape(-1, 3, 3)
        return P, Rm

    def points(self, q: np.ndarray, body_ids: np.ndarray, local: np.ndarray) -> np.ndarray:
        """World positions of points fixed in bodies: q (N, 6); body_ids (K,), local (K, 3) -> (N, K, 3)."""
        ub, inv = np.unique(body_ids, return_inverse=True)
        P, Rm = self.poses(q, list(ub))
        return P[:, inv] + np.einsum("nkij,kj->nki", Rm[:, inv], local)

    def keypoints(self, q: np.ndarray, names=None, cap: float = 0.0, finger_ext: float = 0.0) -> dict[str, np.ndarray]:
        kl = keypoints_local(self.m, cap, finger_ext)
        names = names or list(kl)
        bids = np.array([self.m.body(kl[n][0]).id for n in names])
        loc = np.array([kl[n][1] for n in names])
        X = self.points(q, bids, loc)
        return {n: X[:, k] for k, n in enumerate(names)}


def set_camera(m: mujoco.MjModel, cam_id: int, cam: Camera, w: int = W, h: int = H, scale: float = 1.0) -> None:
    """Put an OpenCV camera into a MuJoCo world-attached camera (pose + intrinsics; sensorsize in pixel units)."""
    R = cam.R
    xmj = R[0]
    ymj = -R[1]
    zmj = -R[2]
    mat = np.stack([xmj, ymj, zmj], 1)  # columns = camera axes in world
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.ravel())
    m.cam_pos[cam_id] = cam.center
    m.cam_quat[cam_id] = q
    m.cam_sensorsize[cam_id] = [w, h]
    m.cam_resolution[cam_id] = [w, h]
    f = cam.f * scale
    cx = (cam.cx + 0.5) * scale - 0.5
    cy = (cam.cy + 0.5) * scale - 0.5
    m.cam_intrinsic[cam_id] = [f, f, (w / 2 - 0.5) - cx, (h / 2 - 0.5) - cy]


def save_json(obj: dict, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return str(path.relative_to(REPO)) if str(path).startswith(str(REPO)) else str(path)
