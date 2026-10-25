"""LIBERO simulator adapter (runs under .venv-libero, never under the repo's .venv).

One LiberoSim owns one robosuite/MuJoCo env. It reproduces the published LLM-on-LIBERO adapter
[llm-libero-2026] (IRVLUTD/LLM_LIBERO src/inspect_robots_libero/_worker.py, read 2026-10-25):

  reset:  OffScreenRenderEnv(bddl, 256x256), env.seed(142 + init_state), env.reset(),
          set_init_state(init_states[init_state]), 10 no-op steps with the gripper open,
          then the OSC controller is switched to absolute targets (use_delta = False); step counter = 0.
  action: 10-D absolute EEF pose in the world frame: x y z, rot6d (r00 r10 r20 r01 r11 r21 = the first two
          columns of the rotation matrix), gripper in [0, 1]. rot6d is orthonormalised (Gram-Schmidt) and
          converted to axis-angle; gripper -> 2 g - 1 (robosuite's Panda gripper follows only its sign).
  state:  eef_state = controller.ee_pos + controller.ee_ori_mat[:, :2] (column-major rot6d) + the gripper's
          current command in [0, 1]; gripper_qpos = the two measured finger joints.
  images: views="paper": agentview flipped on both axes, wrist as rendered (both are mirror images, see VIEWS);
          views="true" (default since 2026-10-25 14:50): both flipped vertically only, the true camera views.
  success: LIBERO's own goal predicates (env.check_success()) on sim state, after every control step.

Privileged state (object poses, the predicate) never leaves this module except through `record_state()`,
which the server writes to the episode record only (never to an actor).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO / "runs" / "libero" / "config"))
# robosuite's macro MUJOCO_GPU_RENDERING=True forces MUJOCO_GL=egl unless MUJOCO_GL is osmesa or glx. egl fails on
# this WSL box; "glx" is left alone by robosuite and mujoco/robosuite then fall through to the GLFW context (WSLg).
os.environ["MUJOCO_GL"] = os.environ.get("LIBERO_MUJOCO_GL", "glx")
os.environ.setdefault("NUMBA_CACHE_DIR", str(REPO / "runs" / "libero" / "numba_cache"))

import numpy as np  # noqa: E402

SUITE = "libero_10"
ENV_SEED_BASE = 142          # paper code tasks.py: env seed for init state i is 142 + i
SETTLE_STEPS = 10            # no-op steps after set_init_state, excluded from the step count
CAMERA_HW = 256
CONTROL_HZ = 20
DIM_LABELS = ("x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper")
# paper embodiment.py _action_space(): bounds and the declared per-20 Hz-step limit (upstream interpolates at it)
LOW = np.array([-0.6, -0.6, 0.42, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0])
HIGH = np.array([0.6, 0.6, 1.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
MAX_STEP = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1])


def _libero():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    return benchmark, get_libero_path, OffScreenRenderEnv


def suite(name: str = SUITE):
    benchmark, _, _ = _libero()
    return benchmark.get_benchmark_dict()[name]()


def task_table(name: str = SUITE) -> list[dict]:
    s = suite(name)
    return [dict(task_id=i, name=t.name, language=t.language, problem_folder=t.problem_folder,
                 bddl_file=t.bddl_file, n_init_states=int(len(s.get_task_init_states(i))))
            for i, t in enumerate(s.tasks)]


def evaluator_files() -> list[Path]:
    """Files whose content defines success: the predicate code and every BDDL goal and init state of the suites."""
    import libero.libero as L
    root = Path(L.__file__).resolve().parent
    files = []
    for sub in ("envs/predicates", "envs/object_states", "envs/problems"):
        files += sorted((root / sub).glob("*.py"))
    files += [root / "envs" / "bddl_base_domain.py", root / "envs" / "bddl_utils.py", root / "envs" / "env_wrapper.py"]
    for s in ("libero_10", "libero_90"):
        files += sorted((root / "bddl_files" / s).glob("*.bddl"))
        files += sorted((root / "init_files" / s).glob("*"))
    return files


def evaluator_hash() -> dict:
    import libero.libero as L
    root = Path(L.__file__).resolve().parent
    h = hashlib.sha256()
    files = evaluator_files()
    for f in files:
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    import mujoco
    import robosuite
    return dict(sha256=h.hexdigest(), n_files=len(files), mujoco=mujoco.__version__, robosuite=robosuite.__version__)


def rot6d_to_axisangle(r6: np.ndarray) -> np.ndarray:
    """paper _worker.py: orthonormalise the two columns, stack [a, b, a x b], -> quaternion -> axis-angle."""
    import robosuite.utils.transform_utils as T
    a = np.asarray(r6[:3], float)
    b = np.asarray(r6[3:6], float)
    if np.linalg.norm(a) < 1e-6:
        raise ValueError("Degenerate rot6d first column")
    a = a / np.linalg.norm(a)
    b = b - np.dot(a, b) * a
    if np.linalg.norm(b) < 1e-6:
        raise ValueError("Degenerate rot6d second column")
    b = b / np.linalg.norm(b)
    R = np.stack([a, b, np.cross(a, b)], axis=1)
    return T.quat2axisangle(T.mat2quat(R))


VIEWS = ("true", "paper")
# Image orientation. MuJoCo/robosuite ("opengl" convention) return rows bottom-up, so raw[::-1] is the true camera
# view. The agentview camera sits at +x looking back at the robot (quat 0.653 0.271 0.271 0.653: image right = world
# +y). "paper" reproduces the published adapter: agentview[::-1, ::-1] (= the true view mirrored left-right, so
# LIBERO's "left plate" at y = -0.30 appears on the image's right) and the wrist raw (= the true view mirrored
# top-bottom). "true" serves both as raw[::-1]. Found 2026-10-25 by the reward probe; default "true" from then on.


class LiberoSim:
    def __init__(self, suite_name: str = SUITE, views: str = "true"):
        if views not in VIEWS:
            raise ValueError(f"views must be one of {VIEWS}")
        self.views = views
        self.suite_name = suite_name
        self.suite = suite(suite_name)
        self.env = None
        self.obs = None
        self.steps = 0
        self.task_id = None
        self.init_state = None

    # ------------------------------------------------------------------ episodes
    def reset(self, task_id: int, init_state: int, env_seed: int | None = None) -> dict:
        _, get_libero_path, OffScreenRenderEnv = _libero()
        if self.env is not None:
            self.env.close()
            self.env = None
        task = self.suite.get_task(int(task_id))
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=CAMERA_HW, camera_widths=CAMERA_HW)
        seed = ENV_SEED_BASE + int(init_state) if env_seed is None else int(env_seed)
        env.seed(seed)
        env.reset()
        states = self.suite.get_task_init_states(int(task_id))
        obs = env.set_init_state(states[int(init_state)])
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = env.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
        env.env.robots[0].controller.use_delta = False
        self.env, self.obs, self.steps = env, obs, 0
        self.task_id, self.init_state, self.env_seed = int(task_id), int(init_state), seed
        self.instruction = task.language
        self.task_name = task.name
        self.bddl_sha = hashlib.sha256(bddl.read_bytes()).hexdigest()[:16]
        self.init_state_sha = hashlib.sha256(np.asarray(states[int(init_state)]).tobytes()).hexdigest()[:16]
        return self.pack()

    def step(self, action10) -> dict:
        a = np.asarray(action10, dtype=float)
        if a.shape != (10,) or not np.all(np.isfinite(a)):
            raise ValueError("Invalid action")
        act = np.concatenate([a[:3], rot6d_to_axisangle(a[3:9]), [2 * a[9] - 1]])
        self.obs, _, _, _ = self.env.step(act)
        self.steps += 1
        return self.pack()

    def success(self) -> bool:
        return bool(self.env.check_success())

    # ------------------------------------------------------------------ what an actor may see
    def eef_state(self) -> np.ndarray:
        robot = self.env.env.robots[0]
        c = robot.controller
        g = float(np.clip((robot.gripper.current_action[-1] + 1.0) / 2.0, 0, 1))
        return np.concatenate((c.ee_pos, c.ee_ori_mat[:, :2].T.reshape(-1), [g]))

    def pack(self) -> dict:
        o = self.obs
        if self.views == "paper":
            av, wr = o["agentview_image"][::-1, ::-1], o["robot0_eye_in_hand_image"]
        else:
            av, wr = o["agentview_image"][::-1], o["robot0_eye_in_hand_image"][::-1]
        return dict(images=dict(agentview=av.copy(), wrist=wr.copy()),
                    eef_state=self.eef_state(), gripper_qpos=np.asarray(o["robot0_gripper_qpos"], float).copy(),
                    steps=self.steps)

    # ------------------------------------------------------------------ privileged (records only)
    def record_state(self) -> np.ndarray:
        return np.asarray(self.env.sim.get_state().flatten(), dtype=np.float64)

    def physics_health(self) -> dict:
        """Plausibility signals from MuJoCo: NaN / bad-qacc warnings, worst contact penetration, fastest free body."""
        sim = self.env.sim
        m, d = sim.model._model, sim.data._data
        import mujoco
        bad = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        finite = bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        pen = 0.0
        if d.ncon:
            pen = float(max(0.0, -float(np.min(d.contact.dist[: d.ncon]))))
        vmax = 0.0
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                a = m.jnt_dofadr[j]
                vmax = max(vmax, float(np.linalg.norm(d.qvel[a:a + 3])))
        return dict(bad_qacc=bad, finite=finite, max_penetration_m=pen, max_free_body_speed_m_s=vmax)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
