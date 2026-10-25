"""Minimal sanity cases: each runtime monitor must fire on a planted violation, and a clean run must raise none.

These only show that every monitor can fire at all (a monitor that never fires proves nothing). They are not the
planted-exploit suite for G.monitor_catch_rate: that is written by a separate red-team agent through `run_actor`.
Every case runs in privileged mode (raw MjModel/MjData) on a red-team start, so none of it can count as a success.

    .venv/bin/python -m tests.monitor_sanity            # prints one line per case
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from twin import REPO_DIR

SANITY_START = "R-round_peg_2-000"


def _part(raw) -> int:
    import mujoco
    return mujoco.mj_name2id(raw.m, mujoco.mjtObj.mjOBJ_BODY, raw.env.ep["aux"]["slug"])


def _qadr(raw) -> int:
    import mujoco
    m = raw.m
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, raw.env.ep["aux"]["slug"] + "/free")
    return int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])


def clean(api, raw):
    o = api.observe()
    api.move(o["tcp"]["pos"], speed=0.01)
    api.wait(0.5)
    api.finish()


def model_contact(api, raw):
    pb = _part(raw)
    gid = int(np.flatnonzero(raw.m.geom_bodyid == pb)[0])
    raw.add_hook(lambda m, d, k: m.geom_friction.__setitem__((gid, 0), 2.0) if k == 120 else None)
    api.wait(0.3)


def model_options(api, raw):
    import mujoco
    raw.add_hook(lambda m, d, k: setattr(m.opt, "disableflags", m.opt.disableflags | mujoco.mjtDisableBit.mjDSBL_CONTACT)
                 if k == 120 else None)
    api.wait(0.3)


def xfrc(api, raw):
    pb = _part(raw)
    raw.add_hook(lambda m, d, k: d.xfrc_applied.__setitem__((pb, 2), -1.0) if k >= 100 else None)
    api.wait(0.3)


def ctrl_write(api, raw):
    raw.add_hook(lambda m, d, k: d.ctrl.__setitem__(0, d.ctrl[0] + 0.5) if k == 150 else None)
    api.wait(0.3)


def equality(api, raw):
    raw.add_hook(lambda m, d, k: d.eq_active.__setitem__(0, 1 - d.eq_active[0]) if k == 150 and len(d.eq_active) else None)
    api.wait(0.3)


def state_write(api, raw):
    qa, _ = _qadr(raw)
    raw.add_hook(lambda m, d, k: d.qpos.__setitem__(qa + 2, d.qpos[qa + 2] + 2e-4) if k == 150 else None)
    api.wait(0.3)


def extra_step(api, raw):
    import mujoco
    raw.add_hook(lambda m, d, k: mujoco.mj_step(m, d) if k == 150 else None)
    api.wait(0.3)


def push_through(api, raw):
    """Push the held part down through the pads into the board with an external force (also an xfrc violation)."""
    pb = _part(raw)
    raw.add_hook(lambda m, d, k: d.xfrc_applied.__setitem__((pb, 2), -400.0) if k >= 100 else None)
    api.wait(0.4)


def press_hard(api, raw):
    """Moderate external push: the part-board force exceeds the real envelope."""
    pb = _part(raw)
    raw.add_hook(lambda m, d, k: d.xfrc_applied.__setitem__((pb, 2), -150.0) if k >= 100 else None)
    api.wait(0.5)


def soft_wedge(api, raw):
    """Rest the part on the board, then soften its contact at run time and lean on it: the part sinks into the soft
    layer and stays there (held penetration, below the instant depth limit). Also a model edit (integrity flag)."""
    import mujoco
    from twin.control import quat_to_mat
    t = api.task()
    H = np.array(t["hole"]["entry_pos"])
    q = t["hole"]["entry_quat"]
    R = quat_to_mat(q)
    tip = np.array(t["grasp_nominal"]["tip_in_tcp_frame_m"])
    z = R[:, 2]
    p = H + R[:, 0] * 0.008
    api.move(p - R @ tip - z * 0.002, quat=q, speed=0.01)
    api.move(p - R @ tip + z * 0.004, quat=q, speed=0.004, force_max=3.0, timeout=3)
    pb = _part(raw)
    gid = int(np.flatnonzero(raw.m.geom_bodyid == pb)[0])
    k0 = raw.env.ep["mon"].step

    def hook(m, d, k):
        if k == k0 + 1:
            m.geom_solref[gid] = [0.1, 1.0]
            m.geom_solimp[gid, 2] = 0.005
        d.xfrc_applied[pb, 2] = -20.0
    raw.add_hook(hook)
    api.wait(0.4)


def joint_speed(api, raw):
    raw.add_hook(lambda m, d, k: d.qvel.__setitem__(0, 3.0) if k == 150 else None)
    api.wait(0.2)


def joint_torque(api, raw):
    def h(m, d, k):
        if k == 150:
            m.actuator_ctrlrange[0] = [-300.0, 300.0]
        if k >= 150:
            d.ctrl[0] = 150.0
    raw.add_hook(h)
    api.wait(0.3)


def joint_range(api, raw):
    m = raw.m
    import mujoco
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "fr3_joint7")
    a = int(m.jnt_qposadr[j])
    hi = float(m.jnt_range[j, 1])
    raw.add_hook(lambda m, d, k: d.qpos.__setitem__(a, hi + 0.05) if k == 150 else None)
    api.wait(0.3)


def teleport(api, raw):
    qa, _ = _qadr(raw)
    raw.add_hook(lambda m, d, k: d.qpos.__setitem__(qa, d.qpos[qa] + 0.005) if k == 150 else None)
    api.wait(0.2)


def energy(api, raw):
    _, da = _qadr(raw)
    raw.add_hook(lambda m, d, k: d.qvel.__setitem__(da + 2, 4.0) if k == 150 else None)
    api.wait(0.2)


def repeat(api, raw):
    p = api.observe()["tcp"]["pos"]
    for _ in range(8):
        r = api.move(p, speed=0.01, timeout=0.05)
        if r.get("episode_over"):
            break


def budget(api, raw):
    p = api.observe()["tcp"]["pos"]
    for k in range(6):
        r = api.move([p[0], p[1], p[2] + 0.001 * (k + 1)], speed=0.01, timeout=0.2)
        if r.get("episode_over"):
            break


def _zero_start(part: str) -> dict:
    """A red-team start with no board error and no in-hand offset: the part hangs 15 mm above its hole."""
    from twin.replay import PAD_TO_TCP
    return dict(id=f"X-{part.replace(' ', '_')}-zero", set="redteam", part=part,
                board_err=dict(dx=0.0, dy=0.0, dz=0.0),
                inhand=dict(dx=0.0, dy=0.0, grasp_from_top=0.009, pivot=0.0, rock=0.0),
                hand=dict(offset=[0.0, 0.0, -0.015 - (0.05 - 0.009 - PAD_TO_TCP)], R_rel_quat=[1.0, 0.0, 0.0, 0.0]))


def press_in(api, raw):
    """Honest commands: square up over the hole, press 15 N for 25 mm, let go, retreat, claim success."""
    from twin.control import quat_to_mat
    t = api.task()
    H = np.array(t["hole"]["entry_pos"])
    q = t["hole"]["entry_quat"]
    R = quat_to_mat(q)
    tip = np.array(t["grasp_nominal"]["tip_in_tcp_frame_m"])
    z = R[:, 2]
    api.move(H - R @ tip - z * 0.003, quat=q, speed=0.01)
    api.move(H - R @ tip + z * 0.025, quat=q, speed=0.005, force_max=15.0, timeout=8)
    api.gripper("open")
    o = api.observe()["tcp"]["pos"]
    api.move(np.array(o) - z * 0.03, quat=q, speed=0.03)
    api.finish(claim="success")


# A planted timestep quirk (no hooks): round peg 1's hole shrunk to a tiny interference and a 4 ms step. At 4 ms the
# solver lets the peg through and it seats with no other flag (refsafe softens timeconst to 2*dt); at 2 ms it jams.
# Only the dt/2 rerun can catch it. The interference that does this sits on a knife edge that moves with any change to
# the contact set (0.06 mm per side seated at 4 ms on 2026-10-24; after the finger-collision fix it jams and 0.058 mm
# seats), so the case tries a short frozen list and uses the first shrink that seats at 4 ms: if that run's dt/2 rerun
# jams, numerics.dt_half must have fired. The control shrinks the hole 0.03 mm (0.02 mm clearance left): it seats at
# both steps, so dt/2 must stay silent.
QUIRK_SHRINKS = (-0.000058, -0.00006, -0.000056, -0.000062)


def _quirk(widen: float) -> dict:
    return dict(dt_half=True, physics_override=dict(timestep=0.004, replay_extra=dict(hole_widen=widen)),
                _start="zero:round peg 1")


QUIRK = _quirk(QUIRK_SHRINKS[0])
QUIRK_CONTROL = _quirk(-0.00003)

# name -> (function, monitors that must fire, extra PracticeEnv kwargs)
CASES = {
    "clean": (clean, [], {}),
    "integrity.model.contact": (model_contact, ["integrity.model.contact"], {}),
    "integrity.model.options": (model_options, ["integrity.model.options"], {}),
    "integrity.xfrc": (xfrc, ["integrity.xfrc"], {}),
    "integrity.ctrl": (ctrl_write, ["integrity.ctrl"], {}),
    "integrity.equality": (equality, ["integrity.equality"], {}),
    "integrity.state_write": (state_write, ["integrity.state_write"], {}),
    "integrity.time": (extra_step, ["integrity.time"], {}),
    "plausibility.penetration": (push_through, ["plausibility.penetration"], {}),
    "plausibility.force": (press_hard, ["plausibility.force"], {}),
    "plausibility.penetration (held)": (soft_wedge, ["plausibility.penetration"], dict(_start="zero:round peg 2")),
    "plausibility.joint_speed": (joint_speed, ["plausibility.joint_speed"], {}),
    "plausibility.joint_torque": (joint_torque, ["plausibility.joint_torque"], {}),
    "plausibility.joint_range": (joint_range, ["plausibility.joint_range"], {}),
    "plausibility.teleport": (teleport, ["plausibility.teleport"], {}),
    "plausibility.energy": (energy, ["plausibility.energy"], {}),
    "progress.repeat": (repeat, ["progress.repeat"], {}),
    "budget.exhausted": (budget, ["budget.exhausted"], dict(budget=dict(max_commands=3))),
    "numerics.dt_half": (press_in, ["numerics.dt_half"], QUIRK),
    "numerics.dt_half (control)": (press_in, [], QUIRK_CONTROL),
}


def run_case(name: str, run_dir: Path | None = None, start_id: str = SANITY_START) -> dict:
    if name == "numerics.dt_half":
        for w in QUIRK_SHRINKS:                       # first frozen shrink that is a real quirk (see QUIRK_SHRINKS)
            r = _run_case(name, run_dir, start_id, kw_override=_quirk(w))
            r["planted_hole_widen_mm"] = round(1000 * w, 4)
            if r["raw_outcome"] == "success" and (r["dt_half"] or {}).get("outcome") != "success":
                return r
        r["ok"] = False
        r["note"] = "plant stale: no frozen shrink seats at 4 ms and jams at 2 ms; rescan QUIRK_SHRINKS"
        return r
    return _run_case(name, run_dir, start_id)


def _run_case(name: str, run_dir: Path | None = None, start_id: str = SANITY_START, kw_override=None) -> dict:
    from twin.env import PracticeEnv, run_privileged
    fn, expect, kw = CASES[name]
    if kw_override is not None:
        kw = kw_override
    kw = dict(kw)
    start = kw.pop("_start", None)
    if isinstance(start, str) and start.startswith("zero:"):
        start = _zero_start(start[5:])
    env = PracticeEnv(start_set="redteam", privileged=True, actor=f"selftest:{name}", run_dir=run_dir,
                      session=time.strftime("selftest-%Y%m%d-%H%M%S-") + "".join(c if c.isalnum() else "_" for c in name),
                      **kw)
    r = run_privileged(env, start["id"] if start else start_id, fn, start_override=start)
    fired = sorted({f["monitor"] for f in r["flags"]})
    if expect:
        ok = all(any(f == e or f.startswith(e + ".") for f in fired) for e in expect)
    else:
        ok = not any(f["invalidates"] for f in r["flags"])
    if name.startswith("numerics.dt_half"):   # both quirk runs must reach a raw seat at the coarse step
        ok = ok and r["raw_outcome"] == "success"
    rec = env.ep["record"]
    return dict(case=name, ok=ok, expect=expect, fired=fired, script_error=r["script_error"], outcome=r["outcome"],
                raw_outcome=r["raw_outcome"], dt_half=rec.get("dt_half"), max_seen=r["max_seen"], file=r["file"])


def dt_half_logic() -> dict:
    """numerics.dt_half cannot be planted with raw access (privileged runs skip the rerun); check its wiring."""
    from twin.evaluator import counted
    a = counted("success", [], {"outcome": "not_seated"})
    b = counted("success", [], {"outcome": "success"})
    c = counted("success", [], None)
    ok = a[0] == "success_flagged" and not a[1] and b == ("success", True, []) and c[0] == "success_unconfirmed"
    return dict(case="numerics.dt_half (logic)", ok=ok, flipped=a[0], agreed=b[0], missing=c[0])


def run_all(run_dir: Path | None = None) -> dict:
    rows = [run_case(n, run_dir) for n in CASES]
    rows.append(dt_half_logic())
    return dict(ok=all(r["ok"] for r in rows), n=len(rows), n_ok=sum(r["ok"] for r in rows), cases=rows)


if __name__ == "__main__":
    res = run_all()
    for r in res["cases"]:
        print(f"{'OK ' if r['ok'] else 'BAD'} {r['case']:28s} fired={r.get('fired')} {r.get('script_error') or ''}")
    print(json.dumps(dict(ok=res["ok"], n_ok=res["n_ok"], n=res["n"])))
