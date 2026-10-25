# Environment setup notes (GPU sim + policy training)

Set up on 2026-10-24 on the WSL2 box (RTX 4070 SUPER 12 GB, Windows driver 610.88, CUDA UMD 13.3). Everything lives in **one venv, `.venv`**. A second `.venv-train` was not needed: LeRobot 0.6.1 resolves cleanly against the same torch, mujoco and mujoco_warp (see "LeRobot" below).

## Versions in `.venv` (Python 3.12.10)

| Package | Version | Notes |
|---|---|---|
| torch / torchvision | 2.11.0+cu130 / 0.26.0+cu130 | CUDA 13.0 wheels, cuDNN 9.19. torch 2.11 is the newest LeRobot accepts (`torch<2.12`). |
| mujoco | 3.17.0 | unchanged from before |
| mujoco-warp / warp-lang | 3.17.0 / 1.18.0 | mujoco-warp needs `mujoco>=3.13`, `warp-lang>=1.15` |
| lerobot | 0.6.1 | extras `smolvla,diffusion,training` |
| transformers / diffusers / accelerate | 5.5.4 / 0.39.0 / 1.15.0 | pulled in by the extras |
| torchcodec / av | 0.11.1 (CPU build) / 15.1.0 | torchcodec uses the system FFmpeg 4.x (`libavutil.so.56`) and decodes fine |
| numpy | 2.2.6 | **downgraded from 2.5.4** by LeRobot (`numpy<2.3`). mujoco, warp and torch all work with it. |
| google-genai | 2.33.0 | unchanged |

`uv pip check` reports all 128 packages compatible. The venv is about 5.9 GB.

## Install commands that worked

```sh
# 1. PyTorch with CUDA 13.0 (the driver supports CUDA 13.3)
uv pip install --python .venv/bin/python "torch==2.11.0" "torchvision==0.26.0" \
    --index-url https://download.pytorch.org/whl/cu130

# constraints.txt (repo root) pins torch/torchvision/mujoco. Pass it to EVERY later install
# so nothing silently swaps torch for a CPU or newer build:
uv pip install --python .venv/bin/python -c constraints.txt "mujoco-warp==3.17.0"
uv pip install --python .venv/bin/python -c constraints.txt "lerobot[smolvla,diffusion,training]==0.6.1"

# 2. Robot models: sparse clone of Menagerie with only the Franka directories
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git vendors/mujoco_menagerie
git -C vendors/mujoco_menagerie sparse-checkout set --no-cone '/franka_fr3/' '/franka_emika_panda/' '/LICENSE' '/README.md'
```

Menagerie commit: `f3b6013629c7b94f8ee93fd977b161fa8350f9bc` (2026-10-23). `vendors/` and `runs/` are gitignored.

## GPU check

`torch.cuda.is_available()` is True and the device is RTX 4070 SUPER, sm_89. A 4096² matmul runs at **20 TFLOP/s fp32** and **65 TFLOP/s fp16**, and the GPU result matches the CPU result. About 11.2 of 12.9 GB is free; the Windows desktop uses the rest.

## MuJoCo Warp benchmark

The built-in CLI uses the humanoid from the package's test data and adds control noise to each world:

```sh
.venv/bin/mjwarp-testspeed test_data/humanoid/humanoid.xml --nworld 1024 --nstep 1000 --overflow_behavior continue
```

| Model | nworld | steps/s | Tool |
|---|---|---|---|
| humanoid (nq 28, contacts) | 1024 | 1.59 M | mjwarp-testspeed (one sync per step) |
| humanoid | 4096 | 2.65 M | mjwarp-testspeed |
| humanoid | 8192 | 2.83 M | mjwarp-testspeed |
| humanoid | 1024 / 4096 | 1.51 M / 2.72 M | `twin.warp_bench` (CUDA graph of 50 steps) |
| **twin base.xml** (FR3 + hand) | **256** | **0.53 M** (1,050x realtime) | `twin.warp_bench` |
| twin base.xml | 256 | 0.36 M | mjwarp-testspeed |
| twin base.xml | 4096 | 4.83 M | `twin.warp_bench` |
| twin base.xml, batch render of 3 cameras at 128x128 | 256 | about 10.9 k world-renders/s, so about 33 k frames/s | `mjwarp-testspeed --function render` |

Our own script:

```sh
.venv/bin/python -m twin.warp_bench                  # base.xml, 256 worlds, 1000 steps
.venv/bin/python -m twin.warp_bench --nworld 4096 --xml <model.xml>
```

It loads the model, resets every world to the `reassemble_start` keyframe, adds control noise to every world except world 0, captures 50 `mjw.step` calls in one CUDA graph, and then times the graph launches. Finally it checks world 0 against CPU MuJoCo. For base.xml, world 0 matches CPU MuJoCo to 1.1e-5 rad after 1000 steps, and no world produces NaN. For the humanoid, world 0 drifts from the CPU result (up to 0.14). That is expected: the humanoid falls and makes contacts, which is chaotic, and Warp runs in float32 while CPU MuJoCo runs in float64.

## Twin skeleton (`twin/`)

| File | What it is |
|---|---|
| `twin/__init__.py` | Paths (`BASE_XML`, `SCENE_DIR`) and `CAMERAS = ("hand", "hama1", "hama2")` |
| `twin/scene/build_base.py` | Generates `base.xml` from Menagerie with mjSpec, then checks that the written file reloads to the same model |
| `twin/scene/base.xml` | Generated, self-contained MJCF. Contains FR3 + Franka Hand, a table, a placeholder board box (`task_board`, 0.40 x 0.40 x 0.025 m at x=0.45), cameras `hand`, `hama1`, `hama2` (plus `overview` for debugging), a `tcp` site, and keyframe `reassemble_start` |
| `twin/render.py` | Smoke render: steps 500 times, then renders the three cameras at 640x480 |
| `twin/warp_bench.py` | Batched GPU rollout and benchmark (described above) |

```sh
.venv/bin/python -m twin.scene.build_base           # regenerate base.xml (optional)
MUJOCO_GL=glfw .venv/bin/python -m twin.render       # -> runs/smoke/{hand,hama1,hama2}.png + strip.png
MUJOCO_GL=glfw .venv/bin/python -m twin.render --extra overview
```

Conventions:
- The world frame is the FR3 base frame (link0 origin), and the table top is at z = 0.
- Actuators: 7 arm position actuators (`fr3_joint1..7`, ctrl in rad) and `gripper/actuator8` (0 = closed, 255 = open, 0.04 m per finger).
- Joint and body names: `fr3_joint1..7`, `gripper/finger_joint1/2`, `gripper/hand`, `gripper/left_finger`, `gripper/right_finger`.
- The `reassemble_start` keyframe is the REASSEMBLE start pose `[0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4]` with the gripper open, taken from `vendors/reassemble/catkin_reas/src/franka_reassemble/reassemble_control/config/start_pose.yaml`.
- All camera poses and the board are **placeholders**:
  - `hand` is on `gripper/hand` at (0.065, 0, 0.035), looking along the approach axis, fovy 58.
  - `hama1` and `hama2` are at (1.45, ±0.70, 0.85), looking at (0.30, 0, 0.30), fovy 50.
  - In REASSEMBLE, the real hama1, hama2 and board poses are recorded in world coordinates for each recording, in `<timestamp>_poses.json` (keys `Hama1`, `Hama2`, `DAVIS346`, `NIST_Board1`, stored as `[x,y,z],[qx,qy,qz,qw]`) from mocap. The wrist-camera mount is not given.

## Which Franka is the REASSEMBLE robot

**FR3 (Franka Research 3).** Use `vendors/mujoco_menagerie/franka_fr3`. Evidence:
- The arXiv 2502.05086 teleoperation section says "a Franka Emika FR3 robotic arm". Its Table I says "Franka Emika Panda Reaserach 3" [sic].
- `vendors/reassemble/.docker/.env` sets `ROBOT_TYPE=fr3`.

The gripper is the standard Franka Hand: the launch files use `use_gripper`/`load_gripper` with franka_gripper, and the paper does not name a gripper. Menagerie's `franka_fr3` has **no hand**, so `build_base.py` attaches `franka_emika_panda/hand.xml` to the FR3 `attachment_site`, rotated -45° about z, which is how franka_description mounts it. The wrist camera is a RealSense D435i, the external cameras are HAMA C-600 Pro webcams, and the F/T sensor is an AIDIN AFT200-D80-C on the flange (not modelled yet). `franka_emika_panda` is kept only for the hand meshes and as a fallback.

## LeRobot

Real module paths in 0.6.1, all of which import:
- `lerobot.policies.act.modeling_act.ACTPolicy` (config: `lerobot.policies.act.ACTConfig`)
- `lerobot.policies.diffusion.modeling_diffusion.DiffusionPolicy` (`DiffusionConfig`)
- `lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy` (`SmolVLAConfig`)
- Each package also exports `make_<name>_pre_post_processors`. The factory is `lerobot.policies.factory`.

GPU training smoke test with dummy data (8-D state, one 96x96 camera, batch 8, 5 AdamW steps):
- ACT: 51.6 M params, 1.1 GB peak.
- Diffusion (default UNet `down_dims`): 263 M params, 5.3 GB peak. Shrink `down_dims` if you share the GPU with Warp.

ACT and Diffusion downloaded the ResNet-18 ImageNet weights to `~/.cache/torch/hub` (45 MB). SmolVLA was import-checked only. Building a SmolVLA policy downloads the SmolVLM2 backbone or `lerobot/smolvla_base` from Hugging Face, about 1 GB, on first use.

Why there is no second venv: the only changes LeRobot made were numpy 2.5.4 → 2.2.6 and fsspec. torch, mujoco and mujoco-warp stayed pinned through `constraints.txt`, and everything was re-verified after the LeRobot install: the smoke render, the Warp rollout, and torch CUDA.

## Known issues and gotchas

- **Rendering:** only `MUJOCO_GL=glfw` works (through WSLg). `egl` fails and `osmesa` is not installed. `twin/render.py` sets `glfw` by default. Warp's batch renderer is a CUDA ray tracer and does not use GL at all. It ran on base.xml, but its images were not inspected.
- **First Warp JIT is slow:** about 89 s the first time base.xml is compiled, then about 1 s from the kernel cache at `~/.cache/warp/1.18.0` (154 MB). Changing the model's geometry or materials can trigger part of a recompile.
- **`mjwarp-testspeed` on the humanoid** aborts with `overflow detected ... LS_ITERATIONS` in one world at 1024 worlds, because of its control noise. Pass `--overflow_behavior continue`.
- **The base.xml benchmark has no contacts:** the arm holds the start pose and `nacon=0`, so collision kernels are not exercised. Expect lower steps/s once the gripper touches parts. Raise `nconmax`/`njmax` in `put_data` if contact-rich worlds overflow; the defaults for base.xml are naconmax 12288 and njmax 64.
- **mjSpec `to_xml` quirks in MuJoCo 3.17, both handled in `build_base.py`:**
  1. Attaching a spec with an empty prefix writes an unnamed nested `<default>`, which fails to reload. The builder therefore uses the FR3 spec as the top-level spec.
  2. `to_xml` drops `mass="0" density="0"` from FR3's collision default, which would give link0 extra mass. The builder restores it with a regex.

  A round-trip assertion compares body, geom, camera, actuator and mass arrays between the spec and the reloaded file.
- **numpy is held below 2.3 by LeRobot.** Do not upgrade numpy without checking LeRobot.
- **torchcodec is the CPU build** and relies on the system FFmpeg 4.4 libraries. Video decoding works (checked on a test mp4). If it ever breaks, LeRobot can fall back to the `pyav` video backend.
