"""In-hand pose from force: N parallel rig worlds on mujoco_warp, weighted by F/T agreement (a particle filter
without resampling), after [inhand-gpu-2020] (docs/knowledge/real2sim-twins.md).

For one real attempt:
  1. prior: each world draws an in-hand pose (along-pad slide dx, closing offset dy, height g, pivot about the
     closing axis, rock about the pad-width axis) from the population spread of the label-blind registration over
     train attempts, and contact parameters from the cousins ranges (part friction, hole widening). All pre-t0:
     initial qpos, and per-world model fields (geom_friction, geom_pos of the hole walls, qpos_spring of the
     finger rock hinges, eq_data of the finger coupling).
  2. every world is driven by the same rebuilt setpoint (twin/rig.py targets) and the logged gripper.
  3. likelihood per world: Gaussian on the tared 6-axis wrench (force on the tool, torque about the TCP,
     base frame) at frames subsampled to 5 Hz, sigma_F = 2 N, sigma_T = 0.15 Nm, over frames up to t_end.
  4. weights w = softmax(log L); no resampling (it would copy states between worlds after t = 0).
     Prediction (causal): t_end = decision time = real contact onset + 0.3 s, capped at release - 0.3 s.
     Reconstruction: t_end = release.
  5. outputs: P(seat) = sum w * seated, weighted peak force, posterior moments of the in-hand pose, ESS.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from twin import REPO_DIR
from twin.control import quat_to_mat

FPS = 30
SIGMA_F = 2.0
SIGMA_T = 0.15
LIK_HZ = 5
DECISION_DELAY_S = 0.3
POP_PRIOR = REPO_DIR / "runs" / "claude" / "run-standin-1" / "replay" / "inhand_prior.json"
COUSINS = dict(friction=(0.10, 0.30), widen_m=(0.0, 0.0002))


def population_prior(attempts, rp) -> dict:
    """Spread of the label-blind registration over train attempts (no labels used)."""
    from twin.replay import board_pose_for, recording_of, register_inhand, signals
    vals = []
    for a in attempts:
        s = signals(a["episode"])
        bp = board_pose_for(recording_of(a["episode"]), rp.board_source, a["part"])
        _, dx, dy, g, info = register_inhand(bp, a["part"], s, a["frames"], rp)
        th, ph = info.pop("_angles")
        vals.append([dx, dy, g, th, ph])
    v = np.array(vals)
    names = ["dx", "dy", "g", "rock", "pivot"]
    return {n: dict(mean=float(np.mean(v[:, i])), std=float(np.std(v[:, i])), p5=float(np.percentile(v[:, i], 5)),
                    p95=float(np.percentile(v[:, i], 95))) for i, n in enumerate(names)} | {"n": len(v)}


REG_STD = dict(dx=0.001, dy=0.001, g=0.002, rock=math.radians(1.5), pivot=math.radians(1.5))


def sample_worlds(prior: dict, n: int, seed: int, permissive: bool = False, center: dict | None = None) -> dict:
    """Population prior (center None), or the registered proposal (variant 3c-reg): N(center, REG_STD)."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, lim in (("dx", 0.008), ("dy", 0.006), ("g", None), ("rock", math.radians(12)), ("pivot", math.radians(20))):
        p = prior[k]
        x = rng.normal(p["mean"], p["std"], n) if center is None else rng.normal(center[k], REG_STD[k], n)
        if k == "g":
            x = np.clip(x, 0.0, 0.03)
        else:
            x = np.clip(x, -lim, lim)
        out[k] = x
    if permissive:
        out["friction"] = np.full(n, 0.05)
        out["widen"] = np.full(n, 0.001)
    else:
        out["friction"] = rng.uniform(*COUSINS["friction"], n)
        out["widen"] = rng.uniform(*COUSINS["widen_m"], n)
    return out


def _wall_dirs(m, slug: str):
    """Hole wall geoms and their outward normals in the board body frame (wall box local x axis)."""
    import mujoco
    ids, dirs = [], []
    for gi in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if nm.startswith(f"{slug}_hole/wall"):
            ids.append(gi)
            R = quat_to_mat(m.geom_quat[gi])
            dirs.append(R[:, 0])
    return np.array(ids), np.array(dirs)


_KCACHE = {}


def _ctrl_kernel():
    import warp as wp
    if "k" in _KCACHE:
        return _KCACHE["k"]

    @wp.kernel
    def set_ctrl(ctrl: wp.array2d(dtype=float), c0: wp.array(dtype=float), c1: wp.array(dtype=float), a: float):
        w, j = wp.tid()
        ctrl[w, j] = c0[j] + a * (c1[j] - c0[j])

    _KCACHE["k"] = set_ctrl
    return set_ctrl


def run_attempt(ep: int, seg: int, rp, prior: dict, n: int = 1024, seed: int = 0, permissive: bool = False,
                registered: bool = False, finger_rock_k: float = 2.0, nconmax: int = 96, njmax: int = 600, keep_traces: bool = False) -> dict:
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    from twin.rig import prepare_attempt, world_init, finger_opening, FINGERS

    t_start = time.perf_counter()
    rp = replace(rp, timestep=1.0 / (FPS * 33))
    A = prepare_attempt(ep, seg, rp, finger_rock_k=finger_rock_k)
    mjm = A["m"]
    center = None
    if registered:
        _, rdx, rdy, rg, rinfo = A["reg"]
        th, ph = rinfo.get("_angles", (math.radians(rinfo.get("rock_deg", 0.0)), math.radians(rinfo.get("pivot_deg", 0.0))))
        center = dict(dx=rdx, dy=rdy, g=rg, rock=th, pivot=ph)
    S = sample_worlds(prior, n, seed, permissive, center)
    # nominal world -> finger opening (rock is taken up by the pre-rotated finger hinges; dy shifts both fingers)
    d0, ps = world_init(mjm, A, rp, 0.0, 0.0, float(np.median(S["g"])), 0.0, 0.0)
    q_nom = finger_opening(mjm, d0, ps.slug, 0.0)
    jid = lambda nm: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, nm)  # noqa: E731
    qa = lambda nm: mjm.jnt_qposadr[jid(nm)]  # noqa: E731
    pj = jid(f"{ps.slug}/free")
    qpos = np.zeros((n, mjm.nq))
    for w in range(n):
        dw, _ = world_init(mjm, A, rp, S["dx"][w], S["dy"][w], S["g"][w], S["rock"][w], S["pivot"][w])
        qpos[w] = dw.qpos
    qpos[:, qa("hand/finger_joint1")] = q_nom + S["dy"]
    qpos[:, qa("hand/finger_joint2")] = q_nom - S["dy"]
    rock_j = [jid(f"{fn}/rock") for fn in FINGERS]
    for sgn, j in zip((1.0, -1.0), rock_j):
        if j >= 0:
            qpos[:, mjm.jnt_qposadr[j]] = sgn * S["rock"]
    # per-world model fields
    gpart = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, f"{ps.slug}/collision")
    walls, wdirs = _wall_dirs(mjm, ps.slug)
    fr = np.repeat(mjm.geom_friction[None], n, 0)
    fr[:, gpart, 0] = S["friction"]
    gp = np.repeat(mjm.geom_pos[None], n, 0)
    gp[:, walls, :] += S["widen"][:, None, None] * wdirs[None]
    qs = np.repeat(mjm.qpos_spring[None], n, 0)
    for sgn, j in zip((1.0, -1.0), rock_j):
        if j >= 0:
            qs[:, mjm.jnt_qposadr[j]] = sgn * S["rock"]
    eq = np.repeat(mjm.eq_data[None], n, 0)
    eq[:, :, 0] = 2 * S["dy"][:, None]
    m = mjw.put_model(mjm, batch_sizes=dict(geom_friction=n, geom_pos=n, qpos_spring=n, eq_data=n))
    m.geom_friction.assign(wp.array(fr.astype(np.float32), dtype=wp.vec3))
    m.geom_pos.assign(wp.array(gp.astype(np.float32), dtype=wp.vec3))
    m.qpos_spring.assign(wp.array(qs.astype(np.float32), dtype=float))
    m.eq_data.assign(wp.array(eq.astype(np.float32), dtype=wp.types.vector(length=eq.shape[2], dtype=float)))
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = qpos[0]
    mujoco.mj_forward(mjm, mjd)
    d = mjw.put_data(mjm, mjd, nworld=n, nconmax=nconmax, njmax=njmax)
    d.qpos.assign(wp.array(qpos.astype(np.float32), dtype=float))
    nu = mjm.nu
    c0 = wp.zeros(nu, dtype=float)
    c1 = wp.zeros(nu, dtype=float)
    kern = _ctrl_kernel()
    nsub = 33
    with wp.ScopedCapture() as cap:
        for k in range(nsub):
            wp.launch(kern, dim=(n, nu), inputs=[d.ctrl, c0, c1, float(k) / nsub])
            mjw.step(m, d)
    graph = cap.graph
    t_setup = time.perf_counter() - t_start
    # pre-roll: hold the start pose, close the gripper
    pre = np.zeros(nu)
    pre[0] = -0.03
    c0.assign(pre.astype(np.float32))
    c1.assign(pre.astype(np.float32))
    for _ in range(int(round(rp.preroll_s * FPS))):
        wp.capture_launch(graph)
    sid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "ft")
    tgt, grip = A["tgt"], A["grip"]
    nf = len(tgt)
    FT = np.zeros((nf, n, 6), np.float32)
    t0 = time.perf_counter()
    for i in range(nf):
        sd = d.sensordata.numpy()
        R = d.site_xmat.numpy()[:, sid].reshape(n, 3, 3)
        FT[i, :, :3] = -np.einsum("nij,nj->ni", R, sd[:, 0:3])
        FT[i, :, 3:] = -np.einsum("nij,nj->ni", R, sd[:, 3:6])
        if i == nf - 1:
            break
        a0 = np.r_[grip[i], tgt[i]].astype(np.float32)
        a1 = np.r_[grip[i], tgt[i + 1]].astype(np.float32)
        c0.assign(a0)
        c1.assign(a1)
        wp.capture_launch(graph)
    wp.synchronize()
    t_run = time.perf_counter() - t0
    qf = d.qpos.numpy().astype(np.float64)
    nan = ~np.isfinite(qf).all(1)
    seated, depth, lateral, tilt = _seated(mjm, qf, ps)
    FT = FT - np.median(FT[:10], axis=0, keepdims=True)
    res = _weigh(A, FT, seated, nan, S)
    res.update(registered=bool(registered), episode=int(ep), segment_index=int(seg), part=A["part"], real_success=bool(A["info"]["success"]),
               released=bool(A["info"]["released"]), n=n, permissive=permissive, seed=seed,
               wall=dict(setup_s=round(t_setup, 2), run_s=round(t_run, 2), total_s=round(time.perf_counter() - t_start, 2)),
               nan_worlds=int(nan.sum()), frames=nf, real_peak_N=float(np.linalg.norm(A["W_real"][:_rel(A) + 1, :3], axis=1).max()),
               seat_rate_prior=float(seated[~nan].mean()) if (~nan).any() else 0.0)
    del graph, cap, d, m, c0, c1
    wp.synchronize()
    import gc
    gc.collect()
    if keep_traces:
        res["_FT"] = FT
        res["_W"] = A["W_real"]
        res["_S"] = S
        res["_seated"] = seated
    return res


def _seated(mjm, qf, ps):
    import mujoco
    d = mujoco.MjData(mjm)
    e = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, f"{ps.slug}_hole/entry")
    t = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, f"{ps.slug}/tip")
    from twin.predicates import numeric
    seat = float(numeric(mjm, f"{ps.slug}/seat_depth")[0])
    tol = numeric(mjm, f"{ps.slug}/tol")
    out = np.zeros((4, len(qf)))
    for w in range(len(qf)):
        if not np.isfinite(qf[w]).all():
            continue
        d.qpos[:] = qf[w]
        mujoco.mj_kinematics(mjm, d)
        ze = d.site_xmat[e].reshape(3, 3)[:, 2]
        zt = d.site_xmat[t].reshape(3, 3)[:, 2]
        dp = d.site_xpos[t] - d.site_xpos[e]
        dep = float(dp @ ze)
        lat = float(np.linalg.norm(dp - dep * ze))
        til = math.degrees(math.acos(float(np.clip(zt @ ze, -1, 1))))
        out[:, w] = [dep >= seat - tol[0] and lat <= tol[1] and til <= tol[2], dep, lat, til]
    return out[0].astype(bool), out[1], out[2], out[3]


def _rel(A) -> int:
    return int(min(A["info"]["frames"]["release"] - A["k0"], len(A["W_real"]) - 1))


def decision_frame(A) -> int:
    rel = _rel(A)
    on = np.flatnonzero(np.linalg.norm(A["W_real"][:rel + 1, :3], axis=1) > 2.0)
    k = (on[0] + int(round(DECISION_DELAY_S * FPS))) if len(on) else rel
    return int(max(1, min(k, rel - int(round(0.3 * FPS)))))


ESS_FRAC = 0.1  # adaptive tempering target: ESS >= N/10


def _weights(FT, W, k_end, nan, return_beta: bool = False):
    step = FPS // LIK_HZ
    idx = np.arange(0, k_end + 1, step)
    r = FT[idx] - W[idx][:, None, :]
    ll = -0.5 * (np.sum((r[..., :3] / SIGMA_F) ** 2, axis=(0, 2)) + np.sum((r[..., 3:] / SIGMA_T) ** 2, axis=(0, 2)))
    ll[nan] = -np.inf
    ll = ll - np.max(ll[~nan]) if (~nan).any() else ll
    n = len(ll)

    def wb(b):
        w = np.exp(b * ll)
        w[nan] = 0.0
        return w / w.sum()

    def ess(w):
        return 1.0 / np.sum(w ** 2)

    target = ESS_FRAC * (~nan).sum()
    if ess(wb(1.0)) >= target:
        beta = 1.0
    else:  # bisection on log beta: ESS falls monotonically as beta grows
        lo, hi = 1e-6, 1.0
        for _ in range(40):
            mid = math.sqrt(lo * hi)
            if ess(wb(mid)) >= target:
                lo = mid
            else:
                hi = mid
        beta = lo
    w = wb(beta)
    return (w, beta) if return_beta else w


def _weigh(A, FT, seated, nan, S) -> dict:
    rel = _rel(A)
    k_dec = decision_frame(A)
    peaks = np.linalg.norm(FT[:rel + 1, :, :3], axis=2).max(0)
    out = {}
    rng = np.random.default_rng(12345)
    for mode, k_end in (("prediction", k_dec), ("reconstruction", rel)):
        w, beta = _weights(FT, A["W_real"], k_end, nan, return_beta=True)
        post = {}
        for k in ("dx", "dy", "g", "rock", "pivot", "friction", "widen"):
            mu = float(np.sum(w * S[k]))
            post[k] = dict(mean=mu, std=float(np.sqrt(np.sum(w * (S[k] - mu) ** 2))), prior_std=float(np.std(S[k])))
        imap = int(np.argmax(w))
        idraw = int(rng.choice(len(w), p=w))
        half = len(w) // 2
        wa, wb_ = w[:half] / max(w[:half].sum(), 1e-300), w[half:] / max(w[half:].sum(), 1e-300)
        out[mode] = dict(p_seat=float(np.sum(w * seated)), ess=float(1.0 / np.sum(w ** 2)), beta=float(beta),
                         t_end_s=round(k_end / FPS, 3), peak_N_mean=float(np.sum(w * peaks)), peak_N_map=float(peaks[imap]),
                         peak_N_draw=float(peaks[idraw]), seated_draw=bool(seated[idraw]), seated_map=bool(seated[imap]),
                         p_seat_halves=[float(np.sum(wa * seated[:half])), float(np.sum(wb_ * seated[half:]))],
                         posterior=post)
    # onset error of the drawn reconstruction world vs real (|dF| > 2 N while held)
    on_r = np.flatnonzero(np.linalg.norm(A["W_real"][:rel + 1, :3], axis=1) > 2.0)
    on_s = np.flatnonzero(np.linalg.norm(FT[:rel + 1, idraw, :3], axis=0 if False else 1) > 2.0)
    out["onset_err_s_draw"] = (abs(on_s[0] - on_r[0]) / FPS) if (len(on_r) and len(on_s)) else None
    return out
