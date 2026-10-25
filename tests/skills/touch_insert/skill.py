"""Scripted peg insertion through the env API only (smoke test of the capability boundary; not a tuned routine).

Uses nothing but envapi: the task's nominal hole pose and nominal grasp, robot state and the wrist F/T.
1. square the hand to the hole frame and hover the nominal tip 8 mm above the nominal hole
2. guarded descent; on contact, locate the touch point from the wrench (line of action of F through tau, cut with the
   surface plane), re-aim the hand by the hole-minus-touch offset, repeat (at most 3 times)
3. if still on the surface: small spiral under a light press
4. once the hand has dropped below the touch height: press to seat, open, retreat, finish
"""
import math

import numpy as np

import envapi as env


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def touch_point(p, F, tau, H, zin):
    """Where the part touches the surface: the line of action of F (through tau about the TCP) cut with the plane."""
    rp = np.cross(F, tau) / float(F @ F)
    n = F / np.linalg.norm(F)
    base = p + rp
    s = float((H - base) @ zin) / float(n @ zin) if abs(float(n @ zin)) > 0.3 else 0.0
    return base + s * n


def mat_to_quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return [w, x, y, z]


def hand_frame(t, q_now):
    """Hand rotation with z along the hole axis and the yaw allowed by the part's symmetry nearest the current one."""
    Re = quat_to_mat(t["hole"]["entry_quat"])
    zin = Re[:, 2]
    xc = quat_to_mat(q_now)[:, 0]
    xc = xc - zin * float(xc @ zin)
    xc /= np.linalg.norm(xc)
    sym = t["hole"]["yaw_symmetry_deg"]
    if sym == 0:
        x = xc
    else:
        cands = [Re[:, 0] * math.cos(math.radians(a)) + Re[:, 1] * math.sin(math.radians(a))
                 for a in np.arange(0, 360, sym)]
        x = max(cands, key=lambda c: float(c @ xc))
    y = np.cross(zin, x)
    return np.column_stack([x, y, zin])


def main():
    t = env.task()
    H = np.array(t["hole"]["entry_pos"])
    obs0 = env.observe()
    R = hand_frame(t, obs0["tcp"]["quat"])
    q = mat_to_quat(R)
    zin = R[:, 2]
    tip = np.array(t["grasp_nominal"]["tip_in_tcp_frame_m"])
    seat = t["hole"]["seat_depth_mm"] / 1000

    def tcp_for(tip_world):
        return tip_world - R @ tip

    def depth_below(p, ref):
        return float((np.asarray(p) - ref) @ zin)

    obs = env.observe()
    env.move(obs["tcp"]["pos"], quat=q, speed=0.02)                       # square the hand in place
    env.move(tcp_for(H - zin * 0.008), quat=q, speed=0.03)                # hover: nominal tip 8 mm above
    r = env.move(tcp_for(H + zin * 0.040), quat=q, speed=0.008, force_max=8.0, stop_force=2.5)
    if r["result"]["stopped"] != "stop_force":
        env.finish(claim="failure")                                        # never touched anything
        return
    p0 = np.array(r["obs"]["tcp"]["pos"])                                  # TCP at the first touch
    # the grasp height is unknown by about +-10 mm; a first stop more than 10 mm below the nominal surface means the
    # part went into the hole on the way down
    inside = depth_below(p0 + R @ tip, H) > 0.010
    for _ in range(0 if inside else 3):
        F = np.array(r["obs"]["wrench"]["force_N"])
        tau = np.array(r["obs"]["wrench"]["torque_Nm"])
        p = np.array(r["obs"]["tcp"]["pos"])
        c = touch_point(p, F, tau, H, zin)
        off = (H - c) - zin * float((H - c) @ zin)
        if np.linalg.norm(off) < 0.0003:
            break
        env.move(p - zin * 0.002, quat=q, speed=0.01)                      # lift 2 mm
        env.move(p - zin * 0.002 + off, quat=q, speed=0.01)                # shift by the hole-minus-touch offset
        r = env.move(p + off + zin * 0.010, quat=q, speed=0.006, force_max=8.0, stop_force=2.5)
        if depth_below(r["obs"]["tcp"]["pos"], p0) > 0.003:
            inside = True
            break
    if not inside:
        p0 = np.array(env.observe()["tcp"]["pos"])                         # touch height after re-aiming
        pc = p0.copy()
        for k in range(1, 49):                                             # small spiral, light press
            ang = k * math.pi / 6
            rad = 0.0003 * k / 2
            target = pc + rad * (math.cos(ang) * R[:, 0] + math.sin(ang) * R[:, 1]) + zin * 0.006
            r = env.move(target, quat=q, speed=0.004, force_max=4.0, timeout=2.0)
            if depth_below(r["obs"]["tcp"]["pos"], p0) > 0.003:
                inside = True
                break
    claim = "failure"
    if inside:
        p = np.array(env.observe()["tcp"]["pos"])
        r = env.move(p + zin * 0.025, quat=q, speed=0.005, force_max=6.0, timeout=8.0)
        if depth_below(r["obs"]["tcp"]["pos"], p0) > seat - 0.004:
            claim = "success"
    env.gripper("open")
    p = np.array(env.observe()["tcp"]["pos"])
    env.move(p - zin * 0.03, quat=q, speed=0.03)
    env.finish(claim=claim)


main()
