"""Smoke render: load twin/scene/base.xml, step it, render hand/hama1/hama2 at 640x480 to PNG.

Run (headless WSL needs GLFW via WSLg; egl fails, osmesa is not installed):
    MUJOCO_GL=glfw .venv/bin/python -m twin.render
Writes runs/smoke/{hand,hama1,hama2}.png and runs/smoke/strip.png (all three side by side).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from twin import BASE_XML, CAMERAS, REPO_DIR  # noqa: E402

WIDTH, HEIGHT = 640, 480


def load(xml: Path = BASE_XML) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "reassemble_start")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return model, data


def render_cameras(model, data, cameras=CAMERAS, width=WIDTH, height=HEIGHT) -> dict[str, np.ndarray]:
    out = {}
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for cam in cameras:
            renderer.update_scene(data, camera=cam)
            out[cam] = renderer.render().copy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", type=Path, default=BASE_XML)
    ap.add_argument("--steps", type=int, default=500, help="physics steps before rendering")
    ap.add_argument("--out", type=Path, default=REPO_DIR / "runs" / "smoke")
    ap.add_argument("--extra", nargs="*", default=[], help="extra cameras to render, e.g. overview")
    args = ap.parse_args()

    model, data = load(args.xml)
    q0 = data.qpos.copy()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        mujoco.mj_step(model, data)
    dt = time.perf_counter() - t0
    drift = float(np.abs(data.qpos[:7] - q0[:7]).max())
    print(f"stepped {args.steps} x {model.opt.timestep}s in {dt*1e3:.1f} ms "
          f"({args.steps/dt:,.0f} steps/s CPU); max arm drift from keyframe {drift:.2e} rad; "
          f"ncon={data.ncon}")

    args.out.mkdir(parents=True, exist_ok=True)
    cams = list(CAMERAS) + list(args.extra)
    images = render_cameras(model, data, cams)
    for cam, img in images.items():
        path = args.out / f"{cam}.png"
        Image.fromarray(img).save(path)
        print(f"{cam:>9}: {img.shape[1]}x{img.shape[0]} mean={img.mean():.1f} -> {path.relative_to(REPO_DIR)}")
    strip = np.concatenate([images[c] for c in CAMERAS], axis=1)
    Image.fromarray(strip).save(args.out / "strip.png")
    print(f"    strip: -> {(args.out / 'strip.png').relative_to(REPO_DIR)}")


if __name__ == "__main__":
    main()
