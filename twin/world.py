"""World frame for the REASSEMBLE twin: real frames, camera models, twin renders, overlays, scores.

Everything here works in the robot base frame (= FR3 link0 = MuJoCo world, see twin/__init__.py).
The robot's joint angles are logged every frame, so a render of the twin at those angles is ground
truth for where the arm must appear; that is what aligns the cameras.

Camera convention: a CameraModel stores the MuJoCo camera frame (x right, y up, looks along -z) as
`pos` + `quat` (wxyz) in its parent frame ("world" or a body name such as "gripper/hand"), plus
pinhole intrinsics fx, fy, cx, cy in pixels (OpenCV convention: pixel centres at integers).
`R_cv` gives the OpenCV axes (x right, y down, z forward) in the parent frame.

Used by harness/tools/world.py. Rendering needs MUJOCO_GL=glfw on this box.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from twin import BASE_XML, PKG_DIR, REPO_DIR  # noqa: E402

DATA_DIR = REPO_DIR / "data" / "reassemble"
POSES_DIR = REPO_DIR / "data" / "reassemble-tudata" / "poses"
CALIB_DIR = PKG_DIR / "calib"
RUNS_DIR = REPO_DIR / "runs" / "world"
CACHE_DIR = RUNS_DIR / "cache"
FPS = 30
W, H = 640, 480
REAL_CAMERAS = ("hand", "hama1", "hama2")

# Board plate in the mocap NIST_Board1 frame. Inferred from data, not documented: the board-frame
# insert points (docs/research/reassemble-data-notes.md) span x 0.03..0.34, y -0.35..-0.04, so the
# marker origin sits near one corner of a 0.384 m plate that extends +x and -y. If the board-layout
# builder (twin/layout.py) has fitted the plate outline from the peg grid, use that. Checked by overlay.
LAYOUT_JSON = PKG_DIR / "assets" / "board_layout.json"
# The real plate: NIST laser plate (384 x 384 x 9 mm) placed in the board frame by the mirrored rigid fit of the
# contact hole layout (board-from-contacts skill). Written by twin/camcal.py. Preferred over the old grid fit,
# which was 16 mm too large in renders (400 x 400 x 25 mm placeholder box) and up to 12 mm off in x.
NIST_PLATE_JSON = PKG_DIR / "assets" / "board_nist_plate.json"


def _board_plate() -> dict:
    plate = {"x": (0.0, 0.384), "y": (-0.384, 0.0), "top_z": 0.0, "source": "default corner guess"}
    try:
        d = json.loads(NIST_PLATE_JSON.read_text())
        C = np.array(list(d["corners_board_m"].values()))
        yaw = (d["edge_c00_c10_yaw_deg"] + 45.0) % 90.0 - 45.0  # box yaw in the board frame (square: mod 90)
        plate.update(x=(float(C[:, 0].min()), float(C[:, 0].max())), y=(float(C[:, 1].min()), float(C[:, 1].max())),
                     corners=C.tolist(), centre=list(d["centre_board_m"]), yaw_deg=float(yaw),
                     size=list(d["size_m"]), thickness=float(d["thickness_m"]),
                     source="twin/assets/board_nist_plate.json (NIST laser plate at the contact layout)")
        return plate
    except (OSError, KeyError, ValueError):
        pass
    try:
        g = json.loads(LAYOUT_JSON.read_text())["grid"]
        (x0, y0), (sx, sy) = g["plate_xy_min"], g["plate_size"]
        plate.update(x=(x0, x0 + sx), y=(y0, y0 + sy), source="twin/assets/board_layout.json grid fit")
    except (OSError, KeyError, ValueError):
        pass
    return plate


BOARD_PLATE = _board_plate()


# ----------------------------------------------------------------------------------------------
# Data access (reads data files only; never imports anything from data/)
# ----------------------------------------------------------------------------------------------
@lru_cache
def splits() -> dict:
    return json.loads((DATA_DIR / "meta" / "splits.json").read_text())


def recording_of(episode: int) -> str:
    return splits()[str(episode)]["recording"]


@lru_cache
def _episodes_meta():
    import pandas as pd
    return pd.read_parquet(DATA_DIR / "meta" / "episodes" / "chunk-000" / "file-000.parquet")


def video_ref(episode: int, cam: str) -> tuple[Path, float]:
    """(mp4 path, start time of this episode inside that file in seconds)."""
    from twin import exam
    exam.check(episode)
    e = _episodes_meta()
    r = e[e.episode_index == episode].iloc[0]
    k = f"videos/observation.images.{cam}"
    path = DATA_DIR / f"{k}/chunk-{int(r[k + '/chunk_index']):03d}/file-{int(r[k + '/file_index']):03d}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"{path} (episode {episode} video not downloaded)")
    return path, float(r[k + "/from_timestamp"])


@dataclass
class EpisodeState:
    episode: int
    recording: str
    t: np.ndarray        # (N,) s
    q: np.ndarray        # (N, 7) joint angles, rad
    grip: np.ndarray     # (N, 2) finger positions, m
    ee: np.ndarray       # (N, 7) O_T_EE: x y z qw qx qy qz, robot base frame
    seg: np.ndarray      # (N,) segment index
    task: np.ndarray     # (N,) task index

    @property
    def n(self) -> int:
        return len(self.t)


def episode_state(episode: int) -> EpisodeState:
    """Logged robot state of one episode; exam episodes only for the exam-scorer caller (twin/exam.py)."""
    from twin import exam
    exam.check(episode)  # before the cache, so a cached exam episode is never served to another caller
    return _episode_state(episode)


@lru_cache(maxsize=4)
def _episode_state(episode: int) -> EpisodeState:
    import pyarrow.parquet as pq
    cols = ["frame_index", "timestamp", "observation.state.joint_position",
            "observation.state.gripper_position", "observation.state.ee_pose", "segment.index",
            "task_index"]
    tb = pq.read_table(DATA_DIR / "data" / "chunk-000" / "file-000.parquet", columns=cols,
                       filters=[("episode_index", "==", episode)]).to_pandas()
    if tb.empty:
        raise ValueError(f"episode {episode} not in the local parquet")
    tb = tb.sort_values("frame_index")
    st = lambda c: np.stack(tb[c].to_numpy()).astype(np.float64)  # noqa: E731
    return EpisodeState(episode, recording_of(episode), tb["timestamp"].to_numpy(np.float64),
                        st("observation.state.joint_position"), st("observation.state.gripper_position"),
                        st("observation.state.ee_pose"), tb["segment.index"].to_numpy(),
                        tb["task_index"].to_numpy())


def frame_index(episode: int, frame: int | None = None, t: float | None = None) -> int:
    n = episode_state(episode).n
    fi = int(round(t * FPS)) if frame is None else int(frame)
    if not 0 <= fi < n:
        raise ValueError(f"frame {fi} outside episode {episode} (0..{n - 1})")
    return fi


def read_frame(episode: int, cam: str, frame: int) -> np.ndarray:
    """RGB uint8 (480, 640, 3) of one camera at one episode frame (all cameras share one 30 fps grid)."""
    import av
    path, t0 = video_ref(episode, cam)
    target = t0 + frame / FPS
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        c.seek(int(max(target - 0.5, 0) / st.time_base), stream=st, backward=True)
        for fr in c.decode(st):
            if fr.time is not None and fr.time >= target - 0.5 / FPS:
                return fr.to_ndarray(format="rgb24")
    raise RuntimeError(f"no frame at {target:.3f}s in {path}")


def mocap_poses(recording: str) -> dict[str, dict]:
    """TU Wien per-recording mocap poses -> {name: {"pos": [x,y,z], "quat": [qw,qx,qy,qz]}}.
    File stores [qx,qy,qz,qw]; the mocap world frame equals the robot base frame (data notes)."""
    raw = json.loads((POSES_DIR / f"{recording}_poses.json").read_text())
    return {k: {"pos": list(map(float, v[0])), "quat": [float(v[1][3]), *map(float, v[1][:3])]}
            for k, v in raw.items()}


# ----------------------------------------------------------------------------------------------
# Rotation helpers
# ----------------------------------------------------------------------------------------------
def quat2mat(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float) / np.linalg.norm(q))
    return m.reshape(3, 3)


def mat2quat(R) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q if q[0] >= 0 else -q


def rotvec2mat(r) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(r, float).reshape(3, 1))[0]


def mat2rotvec(R) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, float))[0].ravel()


FLIP = np.diag([1.0, -1.0, -1.0])  # MuJoCo camera axes <-> OpenCV camera axes


# ----------------------------------------------------------------------------------------------
# Camera model
# ----------------------------------------------------------------------------------------------
@dataclass
class CameraModel:
    name: str                       # MuJoCo camera name in base.xml (hand, hama1, hama2)
    parent: str = "world"           # "world" or a MuJoCo body name
    pos: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    quat: list = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # wxyz, MuJoCo camera frame
    fx: float = 550.0
    fy: float = 550.0
    cx: float = (W - 1) / 2
    cy: float = (H - 1) / 2
    width: int = W
    height: int = H

    # -- construction ---------------------------------------------------------------------------
    @classmethod
    def from_cv(cls, name, R_cv, pos, fx, fy=None, cx=None, cy=None, parent="world", **kw):
        q = mat2quat(np.asarray(R_cv) @ FLIP)
        return cls(name=name, parent=parent, pos=[float(v) for v in pos], quat=q.tolist(), fx=float(fx),
                   fy=float(fy if fy is not None else fx), cx=float((W - 1) / 2 if cx is None else cx),
                   cy=float((H - 1) / 2 if cy is None else cy), **kw)

    @classmethod
    def look_at(cls, name, pos, target, fovy_deg=45.0, up=(0, 0, 1), roll_deg=0.0, parent="world"):
        """Camera at pos looking at target (both in the parent frame); `up` is the parent-frame
        direction that should point up in the image; roll_deg turns the image clockwise."""
        z = np.asarray(target, float) - np.asarray(pos, float)
        z /= np.linalg.norm(z)
        x = np.cross(z, np.asarray(up, float))
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z], 1)
        if roll_deg:
            R = R @ rotvec2mat([0, 0, np.radians(roll_deg)])
        f = (H / 2) / np.tan(np.radians(fovy_deg) / 2)
        return cls.from_cv(name, R, pos, f, parent=parent)

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})

    # -- views ----------------------------------------------------------------------------------
    @property
    def R_cv(self) -> np.ndarray:
        return quat2mat(self.quat) @ FLIP

    @property
    def fovy_deg(self) -> float:
        return float(np.degrees(2 * np.arctan((self.height / 2) / self.fy)))

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pos"] = [round(float(v), 5) for v in self.pos]
        d["quat"] = [round(float(v), 6) for v in self.quat]
        for k in ("fx", "fy", "cx", "cy"):
            d[k] = round(float(d[k]), 2)
        d["fovy_deg"] = round(self.fovy_deg, 2)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R_cv, self.pos
        d["T_parent_cam_opencv"] = np.round(T, 6).tolist()
        return d

    # -- parameter vector for optimisation: rotvec(R_cv) 3, pos 3, f 1 [, cx, cy] --------------
    def vec(self, principal=False) -> np.ndarray:
        v = [*mat2rotvec(self.R_cv), *self.pos, (self.fx + self.fy) / 2]
        return np.array(v + ([self.cx, self.cy] if principal else []))

    def with_vec(self, v) -> "CameraModel":
        v = np.asarray(v, float)
        c = CameraModel.from_cv(self.name, rotvec2mat(v[:3]), v[3:6], v[6], parent=self.parent,
                                cx=v[7] if len(v) > 7 else self.cx, cy=v[8] if len(v) > 8 else self.cy)
        return c

    def scaled(self, s: float) -> "CameraModel":
        """Same camera at a different image size (s=0.5 -> 320x240)."""
        return CameraModel(self.name, self.parent, list(self.pos), list(self.quat), self.fx * s,
                           self.fy * s, (self.cx + 0.5) * s - 0.5, (self.cy + 0.5) * s - 0.5,
                           int(round(self.width * s)), int(round(self.height * s)))


def project(cam: CameraModel, pts_parent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project points given in the camera's parent frame -> (uv (N,2), depth (N,))."""
    P = np.atleast_2d(np.asarray(pts_parent, float))
    Xc = (P - np.asarray(cam.pos)) @ cam.R_cv
    z = Xc[:, 2]
    uv = np.stack([cam.fx * Xc[:, 0] / z + cam.cx, cam.fy * Xc[:, 1] / z + cam.cy], 1)
    return uv, z


# ----------------------------------------------------------------------------------------------
# Twin view: the MuJoCo scene posed at a real frame, rendered from a CameraModel
# ----------------------------------------------------------------------------------------------
LABELS = {"background": 0, "arm": 1, "hand": 2, "board": 3, "table": 4, "other": 5}


class TwinView:
    def __init__(self, xml: Path = BASE_XML, width: int = W, height: int = H):
        # base.xml is shared with replay code, so the real board's base slab is added in memory, here only
        # (render path), after every existing geom: geom ids of the robot stay as in base.xml.
        spec = mujoco.MjSpec.from_file(str(xml))
        tb = spec.body("task_board")
        if tb is not None and spec.geom("board_base") is None:
            spec.add_material(name="board_base_mat", rgba=[0.45, 0.46, 0.47, 1.0], specular=0.1)
            tb.add_geom(name="board_base", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.19, 0.19, 0.01],
                        material="board_base_mat", contype=0, conaffinity=0, group=2)
        self.model = spec.compile()
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
        # No multisampling: blended segmentation colours at object borders decode to wrong ids.
        self.model.vis.quality.offsamples = 0
        self.data = mujoco.MjData(self.model)
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "reassemble_start")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0  # sites would pollute segmentation masks
        self.geom_label = self._geom_labels()
        mujoco.mj_forward(self.model, self.data)

    def close(self):
        self.renderer.close()

    def _geom_labels(self) -> np.ndarray:
        m = self.model
        parent = m.body_parentid

        def under(b, root):
            while b > 0:
                if b == root:
                    return True
                b = parent[b]
            return False

        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper/hand")
        board = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "task_board")
        lab = np.full(m.ngeom, LABELS["other"], np.int32)
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if hand >= 0 and under(b, hand):
                lab[g] = LABELS["hand"]
            elif base >= 0 and under(b, base):
                lab[g] = LABELS["arm"]
            elif board >= 0 and under(b, board):
                lab[g] = LABELS["board"]
            elif name in ("table_top", "floor"):
                lab[g] = LABELS["table"]
        return lab

    # -- posing ---------------------------------------------------------------------------------
    def set_robot(self, q7, grip2=(0.04, 0.04)):
        self.data.qpos[:7] = q7
        self.data.qpos[7:9] = grip2
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def set_frame(self, episode: int, frame: int):
        s = episode_state(episode)
        self.set_robot(s.q[frame], s.grip[frame])

    def set_board(self, pos, quat_wxyz):
        """Place the board given the NIST_Board1 frame pose (plate top at board z = 0, see BOARD_PLATE). With the
        NIST plate asset: a 384 x 384 x 9 mm plate turned by its small yaw in the board frame, on a darker base
        slab that reaches down to the table (z = 0). Otherwise the old placeholder box, top at board z = 0."""
        m = self.model
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "task_board")
        R = quat2mat(quat_wxyz)
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "board_plate")
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "board_base")
        if "corners" in BOARD_PLATE:
            t = BOARD_PLATE["thickness"]
            hx, hy = BOARD_PLATE["size"][0] / 2, BOARD_PLATE["size"][1] / 2
            ctop = np.array([*BOARD_PLATE["centre"], BOARD_PLATE["top_z"]])
            top_w = np.asarray(pos) + R @ ctop
            yq = np.array([np.cos(np.radians(BOARD_PLATE["yaw_deg"]) / 2), 0, 0, np.sin(np.radians(BOARD_PLATE["yaw_deg"]) / 2)])
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, np.asarray(quat_wxyz, float), yq)
            m.body_pos[b] = top_w
            m.body_quat[b] = q
            m.geom_size[gid] = [hx, hy, t / 2]
            m.geom_pos[gid] = [0, 0, -t / 2]
            m.geom_quat[gid] = [1, 0, 0, 0]
            if bid >= 0:
                h = max(float(top_w[2]) - t, 0.001)
                m.geom_size[bid] = [hx - 0.002, hy - 0.002, h / 2]
                m.geom_pos[bid] = [0, 0, -t - h / 2]
            for g in (gid, bid):
                if g >= 0:
                    m.geom_rbound[g] = float(np.linalg.norm(m.geom_size[g]))
        else:
            half = m.geom_size[gid]
            c = np.array([np.mean(BOARD_PLATE["x"]), np.mean(BOARD_PLATE["y"]), BOARD_PLATE["top_z"] - half[2]])
            m.body_pos[b] = np.asarray(pos) + R @ c
            m.body_quat[b] = quat_wxyz
            if bid >= 0:
                m.geom_size[bid] = [1e-4, 1e-4, 1e-4]
        mujoco.mj_kinematics(m, self.data)

    def set_hand_offset(self, dz: float):
        """Move the Franka hand (and everything on it, incl. the hand camera) by dz along the flange
        axis relative to base.xml. Used to test the F/T-stack length against images."""
        m = self.model
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper/hand")
        if not hasattr(self, "_hand_pos0"):
            self._hand_pos0 = m.body_pos[b].copy()
        m.body_pos[b] = self._hand_pos0 + quat2mat(m.body_quat[b]) @ np.array([0, 0, dz])
        mujoco.mj_kinematics(m, self.data)

    def parent_pose(self, parent: str) -> tuple[np.ndarray, np.ndarray]:
        if parent == "world":
            return np.eye(3), np.zeros(3)
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, parent)
        if b < 0:
            raise KeyError(f"no body {parent}")
        return self.data.xmat[b].reshape(3, 3).copy(), self.data.xpos[b].copy()

    def apply_camera(self, cam: CameraModel):
        m = self.model
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam.name)
        if i < 0:
            raise KeyError(f"no camera {cam.name} in model")
        want = 0 if cam.parent == "world" else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cam.parent)
        if m.cam_bodyid[i] != want:
            raise ValueError(f"camera {cam.name} is attached to body {m.cam_bodyid[i]}, not {cam.parent}")
        s = self.width / cam.width
        c = cam if abs(s - 1) < 1e-9 else cam.scaled(s)
        m.cam_pos[i] = c.pos
        m.cam_quat[i] = c.quat
        m.cam_mode[i] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        m.cam_resolution[i] = [self.width, self.height]
        m.cam_sensorsize[i] = [self.width, self.height]
        # MuJoCo's principal offset is from the sensor centre, sign-flipped, in GL pixel coordinates
        # (checked against project() to <0.1 px).
        m.cam_intrinsic[i] = [c.fx, c.fy, self.width / 2 - c.cx - 0.5, self.height / 2 - c.cy - 0.5]
        m.cam_fovy[i] = c.fovy_deg
        mujoco.mj_camlight(m, self.data)

    def project_world(self, cam: CameraModel, pts_world) -> tuple[np.ndarray, np.ndarray]:
        R, p = self.parent_pose(cam.parent)
        P = (np.atleast_2d(np.asarray(pts_world, float)) - p) @ R
        return project(cam, P)

    # -- rendering ------------------------------------------------------------------------------
    def render_rgb(self, cam: CameraModel) -> np.ndarray:
        self.apply_camera(cam)
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=cam.name, scene_option=self.opt)
        return self.renderer.render().copy()

    def render_geom_ids(self, cam: CameraModel) -> np.ndarray:
        """(H, W) geom id per pixel, -1 where no geom."""
        self.apply_camera(cam)
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=cam.name, scene_option=self.opt)
        seg = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        isg = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0)
        return np.where(isg, seg[..., 0], -1)

    def render_labels(self, cam: CameraModel, geom_ids: np.ndarray | None = None) -> np.ndarray:
        """(H, W) int label image, see LABELS."""
        ids = self.render_geom_ids(cam) if geom_ids is None else geom_ids
        out = np.zeros(ids.shape, np.int32)
        ok = ids >= 0
        out[ok] = self.geom_label[ids[ok]]
        return out

    @property
    def pad_geoms(self) -> np.ndarray:
        """Visual geoms of the black finger pads (finger bodies, material '*black')."""
        m = self.model
        out = []
        for g in range(m.ngeom):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            mat = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, m.geom_matid[g]) if m.geom_matid[g] >= 0 else ""
            if "finger" in body and (mat or "").endswith("black") and m.geom_group[g] == 2:
                out.append(g)
        return np.array(out, np.int32)


_VIEWS: dict = {}


def get_view(width: int = W, height: int = H, xml: Path = BASE_XML) -> TwinView:
    """Shared TwinView per image size. GL contexts are never freed: freeing one renderer while another
    is live corrupted the other's segmentation buffer (seen with GLFW on WSLg)."""
    key = (str(xml), int(width), int(height))
    if key not in _VIEWS:
        _VIEWS[key] = TwinView(xml, int(width), int(height))
    v = _VIEWS[key]
    v.set_hand_offset(0.0)
    return v


# ----------------------------------------------------------------------------------------------
# Real-image cues and scores
# ----------------------------------------------------------------------------------------------
def edge_distance(img: np.ndarray) -> np.ndarray:
    """Distance (px) from every pixel to the nearest Canny edge of a real frame."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 1.2)
    e = cv2.Canny(g, 30, 90)
    return cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 3)


def boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)


def sample_frames(episode: int, n: int, margin_s: float = 1.0) -> list[int]:
    N = episode_state(episode).n
    a, b = int(margin_s * FPS), N - 1 - int(margin_s * FPS)
    return [int(round(x)) for x in np.linspace(a, b, n)]


def background(episode: int, cam: str, n: int = 41, refresh: bool = False) -> np.ndarray:
    """Per-pixel temporal median over n frames spread over the episode. The arm moves, so this is
    the scene without the moving links (links that barely move, e.g. the base, stay in it)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"ep{episode:03d}_{cam}_bg{n}.png"
    if path.exists() and not refresh:
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    stack = np.stack([read_frame(episode, cam, f) for f in sample_frames(episode, n)])
    bg = np.median(stack, axis=0).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))
    return bg


def foreground(img: np.ndarray, bg: np.ndarray, thr: float = 28.0) -> np.ndarray:
    """Pixels that differ from the background (the moving arm, its cable, its shadow, moved parts)."""
    a = cv2.GaussianBlur(img, (5, 5), 0).astype(np.float32)
    b = cv2.GaussianBlur(bg, (5, 5), 0).astype(np.float32)
    # remove a global exposure change (webcam auto exposure) before differencing
    a *= np.median(b) / max(np.median(a), 1.0)
    d = np.abs(a - b).max(axis=2)
    m = (d > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return m.astype(bool)


@dataclass
class FrameScore:
    frame: int
    edge_px: float          # median distance, robot silhouette boundary -> nearest real edge
    edge_hit: float         # share of robot boundary pixels within 3 px of a real edge
    iou: float | None       # moving-robot silhouette IoU vs background-subtraction mask (static cams)
    board_edge_px: float | None = None  # median distance, board outline -> nearest real edge

    def as_dict(self):
        return {k: (None if v is None else round(float(v), 3)) for k, v in asdict(self).items()}


def score_frame(view: TwinView, cam: CameraModel, episode: int, frame: int, real: np.ndarray | None = None,
                bg: np.ndarray | None = None, static: np.ndarray | None = None,
                robot_labels=("arm", "hand"), labels: np.ndarray | None = None) -> tuple[FrameScore, np.ndarray]:
    real = read_frame(episode, cam.name, frame) if real is None else real
    if labels is None:
        view.set_frame(episode, frame)
        labels = view.render_labels(cam)
    if labels.shape != real.shape[:2]:
        real = cv2.resize(real, (labels.shape[1], labels.shape[0]), interpolation=cv2.INTER_AREA)
        if bg is not None:
            bg = cv2.resize(bg, (labels.shape[1], labels.shape[0]), interpolation=cv2.INTER_AREA)
    dt = edge_distance(real)
    robot = np.isin(labels, [LABELS[k] for k in robot_labels])
    bnd = boundary(robot)
    # ignore boundary pixels on the image border (the silhouette is cut there, not a real edge)
    bnd[:2, :] = bnd[-2:, :] = bnd[:, :2] = bnd[:, -2:] = False
    d = dt[bnd]
    edge_px = float(np.median(d)) if d.size else float("nan")
    edge_hit = float((d <= 3).mean()) if d.size else 0.0
    iou = None
    if bg is not None:
        fg = foreground(real, bg)
        keep = ~static if static is not None else np.ones_like(fg)
        a, b = robot & keep, fg & keep
        union = (a | b).sum()
        iou = float((a & b).sum() / union) if union else None
    board = labels == LABELS["board"]
    bb = boundary(board)
    bb[:2, :] = bb[-2:, :] = bb[:, :2] = bb[:, -2:] = False
    board_px = float(np.median(dt[bb])) if bb.any() else None
    return FrameScore(frame, edge_px, edge_hit, iou, board_px), labels


def static_robot_mask(view: TwinView, cam: CameraModel, episode: int, n: int = 41) -> np.ndarray:
    """Pixels the rendered robot covers in at least half of the background frames: there the median
    background shows the robot itself, so background subtraction cannot see it."""
    acc = None
    for f in sample_frames(episode, n):
        view.set_frame(episode, f)
        r = np.isin(view.render_labels(cam), [LABELS["arm"], LABELS["hand"]]).astype(np.float32)
        acc = r if acc is None else acc + r
    return acc >= n / 2


def overlay(real: np.ndarray, rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.45,
            points: list[tuple[np.ndarray, tuple]] | None = None, title: str = "") -> np.ndarray:
    """Real frame blended with the twin render; robot outline green, board outline magenta."""
    if rgb.shape != real.shape:
        rgb = cv2.resize(rgb, (real.shape[1], real.shape[0]), interpolation=cv2.INTER_LINEAR)
        labels = cv2.resize(labels.astype(np.uint8), (real.shape[1], real.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(np.int32)
    shown = labels > 0
    shown &= labels != LABELS["table"]
    out = real.astype(np.float32).copy()
    out[shown] = (1 - alpha) * out[shown] + alpha * rgb[shown].astype(np.float32)
    out = out.astype(np.uint8)
    robot = np.isin(labels, [LABELS["arm"], LABELS["hand"]])
    out[cv2.dilate(boundary(robot).astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)] = (0, 255, 0)
    out[boundary(labels == LABELS["board"])] = (255, 0, 255)
    for pts, color in points or []:
        for u, v in np.atleast_2d(pts):
            if np.isfinite(u) and np.isfinite(v):
                cv2.circle(out, (int(round(u)), int(round(v))), 4, color, 1, cv2.LINE_AA)
    if title:
        cv2.putText(out, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_png(img: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


# ----------------------------------------------------------------------------------------------
# Calibration files
# ----------------------------------------------------------------------------------------------
def calib_path(recording: str) -> Path:
    return CALIB_DIR / f"{recording}.json"


def load_calib(recording: str) -> dict | None:
    p = calib_path(recording)
    return json.loads(p.read_text()) if p.exists() else None


def board_pose(recording: str, calib: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """NIST_Board1 frame pose (pos, quat wxyz) in the robot base frame: calibrated if saved, else mocap."""
    if calib and calib.get("board"):
        return np.array(calib["board"]["pos"]), np.array(calib["board"]["quat"])
    b = mocap_poses(recording)["NIST_Board1"]
    return np.array(b["pos"]), np.array(b["quat"])


def board_corners_world(pos, quat) -> np.ndarray:
    R = quat2mat(quat)
    if "corners" in BOARD_PLATE:
        c = np.array([[x, y, BOARD_PLATE["top_z"]] for x, y in BOARD_PLATE["corners"]])
    else:
        (x0, x1), (y0, y1), z = BOARD_PLATE["x"], BOARD_PLATE["y"], BOARD_PLATE["top_z"]
        c = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])
    return c @ R.T + np.asarray(pos)


def calib_hand_offset(recording: str, calib: dict | None = None) -> float:
    """Residual tool-length correction (m) still to apply on top of base.xml: the recording's
    "tcp"."hand_offset_m" minus the offset base.xml was built with (build_base.HAND_OFFSET, read from the
    episode-0 calib). Zero for episode 0 now that base.xml carries the calibrated stack; applying the full
    calib value again would count it twice."""
    from twin.scene.build_base import HAND_OFFSET
    c = calib if calib is not None else load_calib(recording)
    return float(((c or {}).get("tcp") or {}).get("hand_offset_m", 0.0)) - HAND_OFFSET


# ----------------------------------------------------------------------------------------------
# Numeric refinement: silhouette IoU (static cameras) + edge chamfer of the robot and board outlines
# ----------------------------------------------------------------------------------------------
@dataclass
class FitData:
    frames: list
    reals: list
    fgs: list
    dts: list
    scale: float


def prepare_fit(episode: int, cam: str, frames: list[int], scale: float, bg: np.ndarray | None) -> FitData:
    w, h = int(W * scale), int(H * scale)
    reals, fgs, dts = [], [], []
    for f in frames:
        r = read_frame(episode, cam, f)
        fgs.append(None if bg is None else cv2.resize(foreground(r, bg).astype(np.uint8), (w, h),
                                                      interpolation=cv2.INTER_NEAREST).astype(bool))
        r = cv2.resize(r, (w, h), interpolation=cv2.INTER_AREA)
        reals.append(r)
        dts.append(edge_distance(r))
    return FitData(list(frames), reals, fgs, dts, scale)


DEFAULT_WEIGHTS = {"iou": 1.0, "robot_edge": 0.5, "board_edge": 0.3, "edge_clip_px": 8.0}


def fit_terms(view: TwinView, cam: CameraModel, episode: int, fd: FitData, static: np.ndarray | None,
              clip: float = 8.0) -> dict:
    """Per-term means over the fit frames: 1-IoU (moving robot silhouette vs background subtraction),
    robot outline -> real edge, board outline -> real edge (both clipped and divided by clip)."""
    ids = [LABELS["arm"], LABELS["hand"]]
    miou, er, eb = [], [], []
    for f, fg, dt in zip(fd.frames, fd.fgs, fd.dts):
        view.set_frame(episode, f)
        lab = view.render_labels(cam)
        robot = np.isin(lab, ids)
        rb = boundary(robot)
        rb[:1, :] = rb[-1:, :] = rb[:, :1] = rb[:, -1:] = False
        if rb.sum() > 10:
            er.append(float(np.minimum(dt[rb], clip).mean() / clip))
        bb = boundary(lab == LABELS["board"]) & ~cv2.dilate(robot.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        bb[:1, :] = bb[-1:, :] = bb[:, :1] = bb[:, -1:] = False
        if bb.sum() > 20:
            eb.append(float(np.minimum(dt[bb], clip).mean() / clip))
        if fg is not None:
            keep = ~static if static is not None else np.ones_like(fg)
            a, b = robot & keep, fg & keep
            u = (a | b).sum()
            miou.append(1.0 - (float((a & b).sum() / u) if u else 0.0))
    nan = float("nan")
    return {"1-iou": float(np.mean(miou)) if miou else nan, "robot_edge": float(np.mean(er)) if er else 1.0,
            "board_edge": float(np.mean(eb)) if eb else nan, "n_board": len(eb)}


def hand_medians(episode: int, cam: str = "hand", n: int = 61) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(gripper finger positions, temporal median image)] for the open and the closed gripper. For a
    camera fixed to the hand, the fingers and the hand housing are the only things that stay put in
    the image, so the median over many frames is a clean picture of them with the scene blurred out."""
    s = episode_state(episode)
    g = s.grip.sum(1)
    out = []
    for name, sel in (("open", g > g.max() - 0.002), ("closed", g < g.min() + 0.002)):
        idx = np.flatnonzero(sel)
        if len(idx) < 20:
            continue
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"ep{episode:03d}_{cam}_median_{name}.png"
        if path.exists():
            med = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        else:
            pick = idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)]
            med = np.median(np.stack([read_frame(episode, cam, int(i)) for i in pick]), 0).astype(np.uint8)
            save_png(med, path)
        out.append((np.median(s.grip[sel], 0), med))
    return out


def near_field_targets(episode: int, cam: str, scale: float, dark: float = 70.0):
    """Scaled hand medians with their dark-pad masks (lower half only) and edge distance maps."""
    w, h = int(W * scale), int(H * scale)
    out = []
    for grip, med in hand_medians(episode, cam):
        m = cv2.resize(med, (w, h), interpolation=cv2.INTER_AREA)
        pad = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY) < dark
        pad[: h // 2] = False
        out.append((grip, pad, edge_distance(m)))
    return out


def near_terms(view: TwinView, cam: CameraModel, targets, clip: float) -> dict:
    """Finger-pad IoU and hand outline -> median-image edge distance, averaged over open/closed."""
    pads = view.pad_geoms
    ious, edges = [], []
    for grip, pad_real, dt in targets:
        view.set_robot(view.data.qpos[:7].copy(), grip)
        ids = view.render_geom_ids(cam)
        pad = np.isin(ids, pads)
        pad[: pad.shape[0] // 2] = False
        u = (pad | pad_real).sum()
        ious.append(float((pad & pad_real).sum() / u) if u else 0.0)
        hand = view.render_labels(cam, ids) == LABELS["hand"]
        hb = boundary(hand)
        hb[:1, :] = hb[-1:, :] = hb[:, :1] = hb[:, -1:] = False
        edges.append(float(np.minimum(dt[hb], clip).mean() / clip) if hb.any() else 1.0)
    return {"1-pad_iou": 1.0 - float(np.mean(ious)), "near_edge": float(np.mean(edges))}


def keypoint_world(kp: dict, episode: int) -> np.ndarray:
    """A keypoint's 3D point in the robot base frame: {"world": [x,y,z]} or {"board": [x,y,z]}
    (NIST_Board1 frame, e.g. a hole from twin/assets/board_layout.json)."""
    if "world" in kp:
        return np.asarray(kp["world"], float)
    bp, bq = board_pose(recording_of(episode))
    return quat2mat(bq) @ np.asarray(kp["board"], float) + bp


def keypoint_residuals(cam: CameraModel, episode: int, keypoints: list[dict]) -> np.ndarray:
    """Pixel distance between each keypoint's marked image position {"frame", "uv"} and its projection.
    Marks are given at 640x480; points behind the camera count as 500 px."""
    if not keypoints:
        return np.zeros(0)
    frames = sorted({int(k["frame"]) for k in keypoints})
    Ts = dict(zip(frames, body_poses(episode, frames, cam.parent))) if cam.parent != "world" else {}
    s = cam.width / W
    out = []
    for k in keypoints:
        P = keypoint_world(k, episode)
        if cam.parent != "world":
            T = Ts[int(k["frame"])]
            P = T[:3, :3].T @ (P - T[:3, 3])
        uv, z = project(cam, P[None])
        u, v = np.asarray(k["uv"], float) * s
        out.append(np.hypot(uv[0, 0] - u, uv[0, 1] - v) / s if z[0] > 0 else 500.0)
    return np.array(out)


def fit_loss(terms: dict, weights: dict) -> float:
    L = 0.0
    for k, wk in (("1-iou", weights["iou"]), ("robot_edge", weights["robot_edge"]),
                  ("board_edge", weights["board_edge"]), ("1-pad_iou", weights.get("pad_iou", 0.0)),
                  ("near_edge", weights.get("near_edge", 0.0)), ("keypoints", weights.get("keypoints", 0.0))):
        if wk and k in terms and np.isfinite(terms[k]):
            L += wk * terms[k]
    return L


def refine_camera(cam: CameraModel, episode: int, frames: list[int], stages=None, weights=None,
                  principal: bool = False, keypoints: list[dict] | None = None, fix_focal: bool = False,
                  log=print) -> tuple[CameraModel, dict]:
    """Nelder-Mead over rotvec, position and focal length (and principal point if asked), coarse to
    fine, restarting the simplex at each stage. Needs a start inside the basin (right region, rough
    scale): that is what the native guess provides. Static cameras use the IoU term; cameras on a
    body (the hand camera) see no moving robot, so they use the finger/housing terms on the hand
    medians plus outlines. Optional keypoints (marked image points with known 3D) add a reprojection
    term: mean(min(px error, 200)) / 40."""
    from scipy.optimize import minimize
    weights_in = weights
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    stages = stages or [dict(scale=0.25, rot=0.04, pos=0.04, f=50.0, iters=700),
                        dict(scale=0.5, rot=0.01, pos=0.01, f=15.0, iters=700),
                        dict(scale=0.5, rot=0.004, pos=0.004, f=6.0, iters=500)]
    static_cam = cam.parent == "world"
    if not static_cam:  # body camera: no moving robot in view; fit fingers/housing + scene outlines
        weights = {**weights, "iou": 0.0, "robot_edge": 0.0, "pad_iou": 1.0, "near_edge": 0.5,
                   **{k: v for k, v in (weights_in or {}).items()}}
    if keypoints:
        weights.setdefault("keypoints", 1.0)
    bg = background(episode, cam.name) if static_cam and weights["iou"] else None
    bp, bq = board_pose(recording_of(episode))
    hist = []
    for st in stages:
        s = st["scale"]
        wh = (int(W * s), int(H * s))
        view = get_view(*wh)
        view.set_board(bp, bq)
        fd = prepare_fit(episode, cam.name, frames, s, bg)
        static = None
        if bg is not None:
            static = cv2.resize(static_robot_mask(view, cam, episode).astype(np.uint8), wh,
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        clip = weights["edge_clip_px"]
        near = near_field_targets(episode, cam.name, s) if not static_cam else None

        def terms(c):
            t = fit_terms(view, c, episode, fd, static, clip)
            if near is not None:
                t.update(near_terms(view, c, near, 2 * clip))
            if keypoints:
                t["keypoints"] = float(np.mean(np.minimum(keypoint_residuals(c, episode, keypoints), 200)) / 40)
            return t

        f_fixed = (cam.fx + cam.fy) / 2

        def full(v):  # with fix_focal the focal length is not a free parameter
            return np.r_[v[:6], f_fixed, v[6:]] if fix_focal else v

        def obj(v):
            return fit_loss(terms(cam.with_vec(full(v))), weights)

        x0 = cam.vec(principal)
        step = np.array([st["rot"]] * 3 + [st["pos"]] * 3 + [st["f"]] + ([st["f"] / 2] * 2 if principal else []))
        if fix_focal:
            x0, step = np.delete(x0, 6), np.delete(step, 6)
        simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * step[i] for i in range(len(x0))])
        t0 = terms(cam)
        res = minimize(obj, x0, method="Nelder-Mead",
                       options=dict(initial_simplex=simplex, maxfev=st["iters"], xatol=1e-5, fatol=1e-6))
        if res.fun <= fit_loss(t0, weights):
            cam = cam.with_vec(full(res.x))
        t1 = terms(cam)
        rec = {"scale": s, "loss_start": round(fit_loss(t0, weights), 4), "loss_end": round(fit_loss(t1, weights), 4),
               "nfev": int(res.nfev), "terms_end": {k: round(v, 4) for k, v in t1.items()}}
        hist.append(rec)
        log(f"[refine {cam.name}] {rec}")
    return cam, {"weights": weights, "stages": hist}


# ----------------------------------------------------------------------------------------------
# Evaluation: the numbers the world-frame checks use (full resolution, frames spread over the episode)
# ----------------------------------------------------------------------------------------------
# Frozen world-frame checks (playbook/skills/build-twin/references/reassemble-world-frame.md). Static cameras are scored on
# held-out frames spread over the recording; the hand camera on its finger medians and keypoints.
CHECKS = {
    "static": {"robot_edge_px_median": ("<=", 2.0), "robot_edge_px_worst": ("<=", 3.0),
               "robot_edge_hit_mean": (">=", 0.70), "iou_mean": (">=", 0.45), "board_edge_px_median": ("<=", 3.0)},
    "body": {"pad_iou_min": (">=", 0.65), "near_edge_px_median": ("<=", 3.0), "keypoint_px_median": ("<=", 10.0)},
}


def check(ev: dict) -> dict:
    kind = "static" if ev.get("parent", "world") == "world" else "body"
    res = {}
    for k, (op, thr) in CHECKS[kind].items():
        v = ev.get(k)
        ok = v is not None and (v <= thr if op == "<=" else v >= thr)
        res[k] = {"value": v, "need": f"{op} {thr}", "pass": bool(ok)}
    return {"pass": all(r["pass"] for r in res.values()), "checks": res}


def near_field_eval(cam: CameraModel, episode: int) -> dict:
    view = get_view()
    pads = view.pad_geoms
    out = {}
    ious, eds = [], []
    for grip, med in hand_medians(episode, cam.name):
        pad_real = cv2.cvtColor(med, cv2.COLOR_RGB2GRAY) < 70
        pad_real[: H // 2] = False
        view.set_robot(view.data.qpos[:7].copy(), grip)
        ids = view.render_geom_ids(cam)
        pad = np.isin(ids, pads)
        pad[: H // 2] = False
        u = (pad | pad_real).sum()
        ious.append(float((pad & pad_real).sum() / u) if u else 0.0)
        hb = boundary(view.render_labels(cam, ids) == LABELS["hand"])
        hb[:2, :] = hb[-2:, :] = hb[:, :2] = hb[:, -2:] = False
        eds.append(float(np.median(edge_distance(med)[hb])) if hb.any() else float("nan"))
    out["pad_iou"] = [round(v, 3) for v in ious]
    out["pad_iou_min"] = round(min(ious), 3) if ious else None
    out["near_edge_px_median"] = round(float(np.nanmedian(eds)), 3) if eds else None
    return out


def evaluate_camera(cam: CameraModel, episode: int, frames: list[int], keypoints: list[dict] | None = None) -> dict:
    view = get_view()
    bp, bq = board_pose(recording_of(episode))
    view.set_board(bp, bq)
    static_cam = cam.parent == "world"
    bg = background(episode, cam.name) if static_cam else None
    static = static_robot_mask(view, cam, episode) if static_cam else None
    per = []
    for f in frames:
        sc, _ = score_frame(view, cam, episode, f, bg=bg, static=static)
        per.append(sc.as_dict())

    def agg(k, fn=np.median):
        v = [p[k] for p in per if p[k] is not None and np.isfinite(p[k])]
        return round(float(fn(v)), 3) if v else None

    ev = {"camera": cam.name, "parent": cam.parent, "episode": episode, "frames": frames,
          "robot_edge_px_median": agg("edge_px"), "robot_edge_px_worst": agg("edge_px", np.max),
          "robot_edge_hit_mean": agg("edge_hit", np.mean), "robot_edge_hit_worst": agg("edge_hit", np.min),
          "iou_mean": agg("iou", np.mean), "iou_worst": agg("iou", np.min),
          "board_edge_px_median": agg("board_edge_px"), "per_frame": per}
    if not static_cam:
        ev.update(near_field_eval(cam, episode))
    if keypoints:
        r = keypoint_residuals(cam, episode, keypoints)
        ev["keypoint_px"] = [round(float(x), 1) for x in r]
        ev["keypoint_px_median"] = round(float(np.median(r)), 2)
    elif not static_cam:
        ev["keypoint_px_median"] = None
    ev["check"] = check(ev)
    return ev


# ----------------------------------------------------------------------------------------------
# Hand-eye calibration from motion (no target): AX = XB with camera motions from feature matches
# ----------------------------------------------------------------------------------------------
@lru_cache
def _fk_model():
    m = mujoco.MjModel.from_xml_path(str(BASE_XML))
    return m, mujoco.MjData(m)


@lru_cache(maxsize=64)
def _body_poses(episode: int, frames: tuple, body: str) -> tuple:
    m, d = _fk_model()
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    s = episode_state(episode)
    out = []
    for f in frames:
        d.qpos[:7], d.qpos[7:9] = s.q[f], s.grip[f]
        mujoco.mj_kinematics(m, d)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = d.xmat[b].reshape(3, 3), d.xpos[b]
        out.append(T)
    return tuple(out)


def body_poses(episode: int, frames: list[int], body: str = "gripper/hand") -> list[np.ndarray]:
    """4x4 world poses of a robot body at the logged joint angles (pure FK, no rendering)."""
    return [T.copy() for T in _body_poses(int(episode), tuple(int(f) for f in frames), body)]


def still_frames(episode: int, max_speed: float = 0.003, max_rate_deg: float = 1.0, min_gap: int = 15,
                 max_z: float | None = None) -> list[int]:
    """Frames where the tool is (nearly) still: video/joint-log latency cannot misplace these."""
    s = episode_state(episode)
    v = np.linalg.norm(np.gradient(s.ee[:, :3], axis=0), axis=1) * FPS
    q = s.ee[:, 3:]
    dq = np.abs(np.sum(q[1:] * q[:-1], axis=1)).clip(0, 1)
    w_ = np.degrees(2 * np.arccos(dq)) * FPS
    w_ = np.concatenate([w_[:1], w_])
    ok = (v < max_speed) & (w_ < max_rate_deg)
    if max_z is not None:
        ok &= s.ee[:, 2] < max_z
    out, last = [], -10 ** 9
    for f in np.flatnonzero(ok):
        if f - last >= min_gap:
            out.append(int(f))
            last = f
    return out


def _rot_angle_deg(R) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def hand_eye_from_motion(episode: int, cam: str = "hand", fx: float = 615.0, fy: float | None = None,
                         cx: float = (W - 1) / 2, cy: float = (H - 1) / 2, exclude_mask: np.ndarray | None = None,
                         gaps=(6, 10, 15), min_rot_deg: float = 4.0, max_rot_deg: float = 35.0,
                         max_pairs: int = 400, stride: int = 3, parent: str = "gripper/hand",
                         log=print) -> tuple[CameraModel, dict]:
    """Camera pose on a robot body from real frames alone (no target, no twin geometry).
    Pairs of frames a few frames apart (`gaps`) while the tool turns: relative camera motion from
    SIFT matches + essential matrix (rotation, translation direction); relative body motion from FK
    of the logged joints. Short gaps keep views overlapping, and under smooth motion a constant
    video/joint-log latency cancels in the relative motion. Rotation: Park-Martin; translation:
    linear least squares with one unknown scale per pair. exclude_mask (H, W bool) hides what moves
    with the camera (fingers, housing), since those points would vote for zero motion."""
    fy = fx if fy is None else fy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    N = episode_state(episode).n
    grid = np.arange(0, N, stride)
    T_all = dict(zip(grid.tolist(), body_poses(episode, grid.tolist(), parent)))
    cands = []
    for f in grid:
        for g in gaps:
            j = f + (g // stride) * stride
            if j not in T_all:
                continue
            A = np.linalg.inv(T_all[f]) @ T_all[j]
            ang = _rot_angle_deg(A[:3, :3])
            if min_rot_deg <= ang <= max_rot_deg:
                cands.append((int(f), int(j), A))
    if not cands:
        raise RuntimeError("no frame pairs with enough rotation")
    pick = np.linspace(0, len(cands) - 1, min(max_pairs, len(cands))).astype(int)
    cands = [cands[i] for i in sorted(set(pick.tolist()))]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.01)
    mask = None if exclude_mask is None else (~exclude_mask).astype(np.uint8) * 255
    cache: dict = {}

    def feat(f):
        if f not in cache:
            g = clahe.apply(cv2.cvtColor(read_frame(episode, cam, f), cv2.COLOR_RGB2GRAY))
            cache[f] = sift.detectAndCompute(g, mask)
        return cache[f]

    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = []
    for fi, fj, A in cands:
        (ki, di), (kj, dj) = feat(fi), feat(fj)
        if di is None or dj is None or len(ki) < 30 or len(kj) < 30:
            continue
        good = [m for m, n in bf.knnMatch(di, dj, k=2) if m.distance < 0.75 * n.distance]
        if len(good) < 30:
            continue
        pi = np.float64([ki[m.queryIdx].pt for m in good])
        pj = np.float64([kj[m.trainIdx].pt for m in good])
        E, inl = cv2.findEssentialMat(pi, pj, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            continue
        n_in, R, t, _ = cv2.recoverPose(E, pi, pj, K, mask=inl)
        if n_in < 25:
            continue
        Bm = np.eye(4)  # camera motion i -> j, same form as A: T_cam_i^-1 T_cam_j
        Bm[:3, :3], Bm[:3, 3] = R.T, (-R.T @ t).ravel()
        pairs.append((A, Bm, int(n_in), fi, fj))
    if len(pairs) < 5:
        raise RuntimeError(f"only {len(pairs)} usable frame pairs")
    al = np.array([mat2rotvec(p[0][:3, :3]) for p in pairs])
    be = np.array([mat2rotvec(p[1][:3, :3]) for p in pairs])

    def solve_R(idx):  # Park & Martin 1994: log(R_A) = R_X log(R_B)
        M = be[idx].T @ al[idx]
        U, _, Vt = np.linalg.svd(M)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        return Vt.T @ D @ U.T

    idx = np.arange(len(pairs))
    for _ in range(4):  # drop pairs whose rotation disagrees (bad essential matrices)
        Rx = solve_R(idx)
        res = np.degrees(np.linalg.norm(al - be @ Rx.T, axis=1))
        keep = np.flatnonzero(res < max(0.5, 3 * np.median(res[idx])))
        if len(keep) < 5:
            break
        idx = keep
    Rx = solve_R(idx)
    rot_res = np.degrees(np.linalg.norm(al[idx] - be[idx] @ Rx.T, axis=1))
    n = len(idx)  # translation: (R_A - I) t_X - s_k R_X tb_k = -t_A
    Amat = np.zeros((3 * n, 3 + n))
    b = np.zeros(3 * n)
    for k, ii in enumerate(idx):
        A, Bm = pairs[ii][0], pairs[ii][1]
        Amat[3 * k:3 * k + 3, :3] = A[:3, :3] - np.eye(3)
        Amat[3 * k:3 * k + 3, 3 + k] = -(Rx @ Bm[:3, 3])
        b[3 * k:3 * k + 3] = -A[:3, 3]
    sol, *_ = np.linalg.lstsq(Amat, b, rcond=None)
    tx, scales = sol[:3], sol[3:]
    trans_res = np.linalg.norm((Amat @ sol - b).reshape(-1, 3), axis=1)
    camm = CameraModel.from_cv(cam, Rx, tx, fx, fy, cx, cy, parent=parent)
    info = {"candidate_pairs": len(cands), "pairs": len(pairs), "pairs_used": int(n),
            "frames_read": len(cache), "rot_residual_deg_median": round(float(np.median(rot_res)), 3),
            "trans_residual_mm_median": round(float(np.median(trans_res)) * 1000, 2),
            "scale_negative": int((scales < 0).sum())}
    log(f"[hand-eye {cam}] {info} pos {np.round(tx, 4).tolist()}")
    return camm, info
