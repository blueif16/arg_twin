"""Cartesian impedance controller matching REASSEMBLE's, plus setpoint reconstruction from logged state.

The real robot ran `cartesian_impedance_controller_damping_ratio.cpp` (vendors/reassemble/catkin_reas/src/
franka_reassemble/reassemble_controllers/src/, read-only), a TU Wien modification of franka_ros
`cartesian_impedance_example_controller.cpp`. One 1 kHz tick of that controller:

    Lambda = (J M^-1 J^T)^-1                       task-space inertia at the EE (O_T_EE, here site 'tcp')
    D      = Q diag(2 zeta sqrt(lambda_i)) Q^T      "double diagonalisation": Lambda = Q Q^T, K = Q diag(lambda) Q^T
    e_pos  = x - x_d
    e_rot  = -R vec(q^-1 * q_d)                     quaternion vector part, so the rotational spring is
                                                    K_r sin(theta/2) ~= (K_r / 2) theta for small angles
    e_vel  = J dq - v_d                             v_d = 0: the teleop never publishes desired_velocity
    tau    = J^T (-K e - D e_vel)
           + N ((1 - r)(-d_ns dq) + r(-P_man dW/dq)) + N(-JL dJL/dq)     N = I - J^T pinv(J^T)
           + coriolis                               (gravity is compensated by the Franka firmware)
    tau    = tau_prev + clip(tau - tau_prev, +-0.5 Nm)                     delta_tau_max per tick
    x_d   <- 0.005 x_target + 0.995 x_d             target filter per tick (tau ~ 0.2 s); slerp for q_d

with W = det(J J^T) and the joint-limit potential sum_i tanh(a(q-c)+b) + tanh(-a(q-c)+b) + 2 from
nullspace_optim.h. Gains as loaded by dynamic reconfigure (cfg/compliance_param.cfg): K = 500 N/m,
K_r = 50 Nm/rad, zeta = 1, nullspace_damping 0.1, manipulability_scale 10, mani_vs_damping 0.8,
joint_limit_cont 10. Commented-out older defaults: 200/10 and 800/75.

The teleop node (reassemble_haptic/scripts/teleop_franka.py) publishes only `equilibrium_pose` (haptic
pose x 7.5), so the controller's x_d is that target after the 0.2 s filter. The dataset logs neither.
`reconstruct_setpoint` rebuilds x_d from what was logged (pose, velocity, wrench on the tool):

    static : K e = W_ext                         ->  x_d = x - F/K,   vec(q^-1 q_d) = -R^T tau / K_r
    dynamic: K e = W_ext - Lambda a - D v        (adds the damping lead D v and inertia; v, a from the log)

The reconstructed x_d is the POST-filter equilibrium, so a replay drives the controller's x_d directly
(the target filter is off by default in replays; it is on when a raw teleop target is given).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))
# nullspace_optim.h computeJointLimitGradient coefficients (a, b, c) per joint
JL_COEF = np.array([[10, -23, 0], [10, -15, 0], [10, -25, 0], [9, -10, -1.6], [4, -10, 0], [6.1, -10, 2.5],
                    [3.7, -10, 0]], float)
FD_DELTA = 1e-5  # nullspace_optim.h finite-difference step


@dataclass
class ImpedanceParams:
    """Gains of the REASSEMBLE controller. Defaults = compliance_param.cfg as loaded at start."""
    k_trans: float = 500.0          # N/m
    k_rot: float = 50.0             # Nm/rad (applied to the quaternion vector part, see module doc)
    zeta: float = 1.0               # damping ratio
    ns_damping: float = 0.1         # nullspace_damping
    ns_manip: float = 10.0          # manipulability_scale
    ns_ratio: float = 0.8           # mani_vs_damping
    ns_jl: float = 10.0             # joint_limit_cont
    dtau_max: float = 0.5           # Nm per tick (header constant delta_tau_max_)
    filter_alpha: float = 0.005     # per tick; 0 = off (x_d follows the target directly)
    period: float = 0.001           # controller tick, s (franka_ros runs at 1 kHz)
    coriolis: bool = True           # add C(q, dq) dq (gravity comes from body gravcomp in the scene)
    ns_every: int = 1               # recompute the manipulability gradient every n ticks (1 = exact)

    def as_dict(self) -> dict:
        return asdict(self)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def slerp(q0, q1, a):
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    dot = float(q0 @ q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(dot)
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


def rot_error(q, q_d, R=None) -> np.ndarray:
    """Franka's orientation error: -R vec(q^-1 q_d), with q flipped to q_d's hemisphere."""
    q = np.asarray(q, float)
    if float(q @ q_d) < 0:
        q = -q
    qe = quat_mul(quat_conj(q), q_d)
    R = quat_to_mat(q) if R is None else R
    return -R @ qe[1:]


def damping_matrix(Lam: np.ndarray, K: np.ndarray, zeta: float) -> np.ndarray:
    """D from the controller's double diagonalisation (Lambda = L L^T, K~ = L^-1 K L^-T)."""
    L = np.linalg.cholesky(Lam)
    Li = np.linalg.inv(L)
    lam, V = np.linalg.eigh(Li @ K @ Li.T)
    H = L @ V
    return H @ np.diag(2 * zeta * np.sqrt(np.clip(lam, 0, None))) @ H.T


def stiffness_matrix(p: ImpedanceParams) -> np.ndarray:
    return np.diag([p.k_trans] * 3 + [p.k_rot] * 3)


class ArmModel:
    """Kinematics/dynamics queries on the arm's 7 DoF of a MuJoCo model (any scene containing base.xml)."""

    def __init__(self, m, site: str = "tcp"):
        import mujoco
        self.m = m
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site < 0:
            raise KeyError(f"site {site!r} missing")
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        if min(jids) < 0:
            raise KeyError("FR3 arm joints missing")
        self.qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([m.jnt_dofadr[j] for j in jids])
        self.scratch = mujoco.MjData(m)
        self._jp = np.zeros((3, m.nv))
        self._jr = np.zeros((3, m.nv))
        self._M = np.zeros((m.nv, m.nv))

    def jac(self, d) -> np.ndarray:
        import mujoco
        mujoco.mj_jacSite(self.m, d, self._jp, self._jr, self.site)
        return np.vstack([self._jp[:, self.dadr], self._jr[:, self.dadr]])

    def mass(self, d) -> np.ndarray:
        import mujoco
        mujoco.mj_fullM(self.m, d, self._M)
        return self._M[np.ix_(self.dadr, self.dadr)]

    def pose(self, d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        R = d.site_xmat[self.site].reshape(3, 3).copy()
        return d.site_xpos[self.site].copy(), mat_to_quat(R), R

    def jac_at(self, d, q7: np.ndarray) -> np.ndarray:
        """Jacobian at arm configuration q7 (other DoF as in d), via the scratch data."""
        import mujoco
        s = self.scratch
        s.qpos[:] = d.qpos
        s.qpos[self.qadr] = q7
        mujoco.mj_kinematics(self.m, s)
        mujoco.mj_comPos(self.m, s)
        return self.jac(s)

    def lambda_d(self, d, K: np.ndarray, zeta: float):
        J = self.jac(d)
        M = self.mass(d)
        Lam = np.linalg.inv(J @ np.linalg.solve(M, J.T))
        return J, M, Lam, damping_matrix(Lam, K, zeta)

    def state_at(self, q7: np.ndarray, dq7: np.ndarray | None = None, base_qpos=None):
        """Scratch MjData with the arm at q7 (forward-computed), for offline reconstruction."""
        import mujoco
        s = self.scratch
        if base_qpos is not None:
            s.qpos[:] = base_qpos
        s.qpos[self.qadr] = q7
        s.qvel[:] = 0
        if dq7 is not None:
            s.qvel[self.dadr] = dq7
        mujoco.mj_forward(self.m, s)
        return s


class CartesianImpedance:
    """REASSEMBLE's Cartesian impedance controller on a MuJoCo arm (torque-controlled joints)."""

    def __init__(self, m, params: ImpedanceParams | None = None, site: str = "tcp"):
        self.p = params or ImpedanceParams()
        self.arm = ArmModel(m, site)
        self.m = m
        self.K = stiffness_matrix(self.p)
        self._dc = None  # scratch data for the Coriolis term (the real arm's getCoriolis())
        self.tau_prev = None
        self.x_d = None
        self.q_d = None
        self.x_t = None
        self.q_t = None
        self.q_ns = None
        self._grad_w = np.zeros(7)
        self._tick = 0
        self.last = {}

    def reset(self, d, x_d=None, q_d=None):
        """Equilibrium = current EE pose unless given (controller `starting()`)."""
        x, q, _ = self.arm.pose(d)
        self.x_d = np.array(x if x_d is None else x_d, float)
        self.q_d = np.array(q if q_d is None else q_d, float)
        self.x_t, self.q_t = self.x_d.copy(), self.q_d.copy()
        self.q_ns = d.qpos[self.arm.qadr].copy()
        self.tau_prev = None
        self._tick = 0

    def set_target(self, x_t, q_t, direct: bool | None = None):
        """New equilibrium target. direct (default: filter_alpha == 0) bypasses the 0.2 s target filter,
        which is what a replay of a reconstructed (already filtered) x_d needs."""
        self.x_t = np.asarray(x_t, float)
        q_t = np.asarray(q_t, float)
        if float(q_t @ self.q_t) < 0:
            q_t = -q_t
        self.q_t = q_t
        if direct if direct is not None else self.p.filter_alpha <= 0:
            self.x_d, self.q_d = self.x_t.copy(), self.q_t.copy()

    def _coriolis(self, d) -> np.ndarray:
        import mujoco
        # C(q,dq)dq = rne(q, dq, acc=0) - rne(q, 0, acc=0); both terms from the same model and data copy
        if self._dc is None:
            self._dc = mujoco.MjData(self.m)
        s = self._dc
        s.qpos[:] = d.qpos
        s.qvel[:] = d.qvel
        mujoco.mj_kinematics(self.m, s)
        mujoco.mj_comPos(self.m, s)
        mujoco.mj_comVel(self.m, s)
        full = np.zeros(self.m.nv)
        mujoco.mj_rne(self.m, s, 0, full)
        s.qvel[:] = 0
        mujoco.mj_comVel(self.m, s)
        grav = np.zeros(self.m.nv)
        mujoco.mj_rne(self.m, s, 0, grav)
        return (full - grav)[self.arm.dadr]

    def _manip_grad(self, d, J) -> np.ndarray:
        q = d.qpos[self.arm.qadr].copy()
        W = float(np.linalg.det(J @ J.T))
        g = np.zeros(7)
        for i in range(7):
            qi = q.copy()
            qi[i] += FD_DELTA
            Ji = self.arm.jac_at(d, qi)
            g[i] = (float(np.linalg.det(Ji @ Ji.T)) - W) / FD_DELTA
        return g

    @staticmethod
    def _jl_grad(q) -> np.ndarray:
        a, b, c = JL_COEF.T
        f = lambda x: np.tanh(a * (x - c) + b) + np.tanh(-a * (x - c) + b) + 2  # noqa: E731
        return (f(q + FD_DELTA) - f(q)) / FD_DELTA

    def torque(self, d) -> np.ndarray:
        """Joint torques (without gravity) for the 7 arm joints at the current state; advances the filter."""
        p = self.p
        J, M, Lam, D = self.arm.lambda_d(d, self.K, p.zeta)
        x, q, R = self.arm.pose(d)
        dq = d.qvel[self.arm.dadr]
        e = np.zeros(6)
        e[:3] = x - self.x_d
        e[3:] = rot_error(q, self.q_d, R)
        v = J @ dq
        tau_task = J.T @ (-self.K @ e - D @ v)
        Jt_pinv = np.linalg.pinv(J.T)
        N = np.eye(7) - J.T @ Jt_pinv
        if self._tick % max(1, p.ns_every) == 0 and p.ns_manip:
            self._grad_w = self._manip_grad(d, J)
        qa = d.qpos[self.arm.qadr]
        tau_ns = (1 - p.ns_ratio) * (N @ (-p.ns_damping * dq)) + p.ns_ratio * (N @ (-p.ns_manip * self._grad_w)) \
            + N @ (-p.ns_jl * self._jl_grad(qa))
        tau = tau_task + tau_ns
        if p.coriolis:
            tau = tau + self._coriolis(d)
        if self.tau_prev is not None and p.dtau_max > 0:
            tau = self.tau_prev + np.clip(tau - self.tau_prev, -p.dtau_max, p.dtau_max)
        self.tau_prev = tau.copy()
        # parameter / target filter (runs after the torque, as in update())
        if p.filter_alpha > 0:
            a = p.filter_alpha
            self.x_d = a * self.x_t + (1 - a) * self.x_d
            self.q_d = slerp(self.q_d, self.q_t, a)
        self._tick += 1
        self.last = dict(e=e, v=v, Lam=Lam, D=D)
        return tau


# ----------------------------------------------------------------------------------- reconstruction
def tool_wrench_about_tcp(F_base: np.ndarray, T_base: np.ndarray, p_tcp: np.ndarray) -> np.ndarray:
    """torque.base is the moment about the base origin (force_2_base.py: R tau_s + t_sensor x F).
    Return the torque about the TCP: tau_tcp = tau_base - p_tcp x F."""
    return T_base - np.cross(p_tcp, F_base)


def reconstruct_setpoint(x: np.ndarray, quat: np.ndarray, F: np.ndarray, Ttcp: np.ndarray, p: ImpedanceParams,
                         mode: str = "static", v: np.ndarray | None = None, a: np.ndarray | None = None,
                         Lam: np.ndarray | None = None, D: np.ndarray | None = None):
    """Equilibrium pose (x_d, q_d) that balances the logged wrench on the tool for one frame.

    x (3), quat (wxyz), F (3, force ON the tool, base frame), Ttcp (3, torque on the tool about the TCP).
    mode 'static' ignores motion; 'dynamic' also uses v (6, EE twist), a (6) and the model's Lam, D (6x6).
    """
    W = np.r_[F, Ttcp]
    if mode == "dynamic":
        if v is None or D is None:
            raise ValueError("dynamic reconstruction needs v and D")
        W = W - D @ v - (Lam @ a if (a is not None and Lam is not None) else 0.0)
    elif mode != "static":
        raise ValueError(f"mode must be static | dynamic, got {mode!r}")
    e_pos = W[:3] / p.k_trans
    e_rot = W[3:] / p.k_rot  # e_rot = -R vec(q^-1 q_d)
    R = quat_to_mat(quat)
    vq = -R.T @ e_rot
    nv = float(np.linalg.norm(vq))
    if nv >= 1.0:
        vq = vq / nv * 0.999
    qe = np.r_[np.sqrt(max(0.0, 1 - float(vq @ vq))), vq]
    q_d = quat_mul(quat, qe)
    return x - e_pos, q_d / np.linalg.norm(q_d)
