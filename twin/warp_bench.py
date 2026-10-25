"""Batched GPU rollout of a MuJoCo model with MuJoCo Warp; reports steps/s and checks world 0 against CPU MuJoCo.

Run:
    .venv/bin/python -m twin.warp_bench                      # twin/scene/base.xml, 256 worlds
    .venv/bin/python -m twin.warp_bench --nworld 4096 \\
        --xml .venv/lib/python3.12/site-packages/mujoco_warp/test_data/humanoid/humanoid.xml

The first run of a new model JIT-compiles Warp kernels (~1-2 min for base.xml); later runs hit the
kernel cache in ~/.cache/warp.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from twin import BASE_XML


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", type=Path, default=BASE_XML)
    ap.add_argument("--nworld", type=int, default=256)
    ap.add_argument("--nstep", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=50, help="steps per captured CUDA graph")
    ap.add_argument("--key", default="reassemble_start", help="keyframe name (ignored if absent)")
    ap.add_argument("--noise", type=float, default=0.05, help="per-world ctrl noise std (world 0 gets none)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert args.nstep % args.chunk == 0, "--nstep must be a multiple of --chunk"

    mjm = mujoco.MjModel.from_xml_path(str(args.xml))
    mjd = mujoco.MjData(mjm)
    key = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, args.key)
    if key >= 0:
        mujoco.mj_resetDataKeyframe(mjm, mjd, key)
    mujoco.mj_forward(mjm, mjd)

    wp.init()
    print(f"warp {wp.config.version} on {wp.get_device()} | mujoco {mujoco.__version__} | "
          f"model nq={mjm.nq} nv={mjm.nv} nu={mjm.nu} ngeom={mjm.ngeom} dt={mjm.opt.timestep}")

    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=args.nworld)

    # Per-world ctrl: keyframe ctrl + Gaussian noise (scaled by ctrl range), world 0 exact.
    rng = np.random.default_rng(args.seed)
    ctrl = np.tile(mjd.ctrl, (args.nworld, 1))
    if mjm.nu:
        span = np.where(mjm.actuator_ctrllimited, mjm.actuator_ctrlrange[:, 1] - mjm.actuator_ctrlrange[:, 0], 1.0)
        ctrl[1:] += rng.normal(0.0, args.noise, ctrl[1:].shape) * span
        lo, hi = mjm.actuator_ctrlrange[:, 0], mjm.actuator_ctrlrange[:, 1]
        ctrl = np.where(mjm.actuator_ctrllimited, np.clip(ctrl, lo, hi), ctrl)
    d.ctrl.assign(ctrl.astype(np.float32))

    t0 = time.perf_counter()
    with wp.ScopedCapture() as capture:
        for _ in range(args.chunk):
            mjw.step(m, d)
    wp.synchronize()
    jit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(args.nstep // args.chunk):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    run_s = time.perf_counter() - t0
    # The capture itself does not execute the steps, so world state is exactly nstep steps ahead.

    qpos = d.qpos.numpy()
    nan_worlds = int(np.isnan(qpos).any(axis=1).sum())
    sps = args.nworld * args.nstep / run_s
    print(f"nworld={args.nworld} nstep={args.nstep}: capture/JIT {jit_s:.1f}s, run {run_s:.3f}s -> "
          f"{sps:,.0f} steps/s ({sps * mjm.opt.timestep:,.0f}x realtime), NaN worlds: {nan_worlds}")

    # CPU reference for world 0 (same ctrl, no noise).
    for _ in range(args.nstep):
        mujoco.mj_step(mjm, mjd)
    err = float(np.abs(qpos[0] - mjd.qpos).max())
    print(f"world 0 vs CPU mujoco after {args.nstep} steps: max |qpos diff| = {err:.2e}; "
          f"nacon={int(d.nacon.numpy()[0])} (all worlds)")


if __name__ == "__main__":
    main()
