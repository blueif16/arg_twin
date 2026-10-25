"""Does a board scene run on MuJoCo Warp, and does it agree with CPU MuJoCo?

Takes a scene from twin.assembly (best: part_init='above', so every peg falls into its hole and the
collision kernels for the hole walls are exercised), runs `nworld` identical worlds on the GPU for
`seconds`, and compares each part's insertion state (depth, lateral, seated) with a CPU run.

    .venv/bin/python -m twin.warp_check --xml runs/scene/<name>/scene.xml [--nworld 64 --seconds 1.0]

Writes runs/scene/<name>/warp_check.json. First run JIT-compiles kernels (minutes); later runs hit
the cache in ~/.cache/warp.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from twin import REPO_DIR


def _states_from_qpos(mjm, qpos):
    from twin.predicates import all_states
    d = mujoco.MjData(mjm)
    d.qpos[:] = qpos
    mujoco.mj_kinematics(mjm, d)
    return {k: dict(depth_mm=round(v["depth"] * 1000, 4), lateral_mm=round(v["lateral"] * 1000, 4),
                    seated=v["seated"]) for k, v in all_states(mjm, d).items()}


def run(xml: str, nworld: int = 64, seconds: float = 1.0, chunk: int = 50, nconmax: int = 256,
        njmax: int = 1024) -> dict:
    import mujoco_warp as mjw
    import warp as wp

    path = Path(xml) if Path(xml).is_absolute() else REPO_DIR / xml
    mjm = mujoco.MjModel.from_xml_path(str(path))
    mjd = mujoco.MjData(mjm)
    k = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, "reassemble_start")
    if k >= 0:
        mujoco.mj_resetDataKeyframe(mjm, mjd, k)
    mujoco.mj_forward(mjm, mjd)
    nstep = int(round(seconds / mjm.opt.timestep / chunk)) * chunk
    out = dict(xml=str(path.relative_to(REPO_DIR)), nworld=nworld, nstep=nstep, timestep=mjm.opt.timestep,
               ngeom=mjm.ngeom, nq=mjm.nq)

    cpu = mujoco.MjData(mjm)
    cpu.qpos[:], cpu.qvel[:], cpu.ctrl[:] = mjd.qpos, mjd.qvel, mjd.ctrl
    t0 = time.perf_counter()
    for _ in range(nstep):
        mujoco.mj_step(mjm, cpu)
    out["cpu_wall_s"] = round(time.perf_counter() - t0, 3)
    out["cpu_states"] = _states_from_qpos(mjm, cpu.qpos)

    wp.init()
    t0 = time.perf_counter()
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)  # both per world
    with wp.ScopedCapture() as cap:
        for _ in range(chunk):
            mjw.step(m, d)
    wp.synchronize()
    out["warp_capture_jit_s"] = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    for _ in range(nstep // chunk):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    run_s = time.perf_counter() - t0
    out["warp_run_s"] = round(run_s, 3)
    out["warp_steps_per_s"] = round(nworld * nstep / run_s)
    qpos = d.qpos.numpy()
    out["nan_worlds"] = int(np.isnan(qpos).any(axis=1).sum())
    out["nacon_last_all_worlds"] = int(d.nacon.numpy()[0])
    out["buffers_per_world"] = dict(nconmax=nconmax, njmax=njmax)
    w0 = _states_from_qpos(mjm, qpos[0].astype(np.float64))
    out["warp_world0_states"] = w0
    seated_all = []
    for w in range(nworld):
        st = _states_from_qpos(mjm, qpos[w].astype(np.float64))
        seated_all.append(all(v["seated"] for v in st.values()))
    out["worlds_all_seated"] = int(sum(seated_all))
    out["max_depth_diff_vs_cpu_mm"] = round(max(abs(w0[s]["depth_mm"] - out["cpu_states"][s]["depth_mm"])
                                                for s in w0), 4) if w0 else None
    out["max_qpos_diff_vs_cpu"] = float(np.abs(qpos[0] - cpu.qpos).max())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--nconmax", type=int, default=256, help="contacts per world")
    ap.add_argument("--njmax", type=int, default=1024, help="constraint rows per world")
    a = ap.parse_args()
    res = run(a.xml, a.nworld, a.seconds, nconmax=a.nconmax, njmax=a.njmax)
    p = (Path(a.xml) if Path(a.xml).is_absolute() else REPO_DIR / a.xml).parent / f"warp_check_n{a.nworld}.json"
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("states")}, indent=1))
    print("cpu  :", res["cpu_states"])
    print("warp0:", res["warp_world0_states"])
    print(f"-> {p.relative_to(REPO_DIR)}")


if __name__ == "__main__":
    main()
