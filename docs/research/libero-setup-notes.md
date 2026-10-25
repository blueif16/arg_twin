# LIBERO setup notes (task playground)

Set up 2026-10-25 on the WSL2 box. LIBERO lives in its own venv, **`.venv-libero`** (gitignored by `.venv-*/`). The repo's `.venv` (MuJoCo 3.17, torch 2.11+cu130, LeRobot) is untouched. Code: `playgrounds/libero/` (README there for the boundary and the integration plan).

## Versions in `.venv-libero` (Python 3.12.10)

| Package | Version | Notes |
|---|---|---|
| hf-libero | 0.1.4 | Hugging Face's packaging of LIBERO; the one `lerobot[libero]` 0.6.1 asks for (`hf-libero>=0.1.4,<0.2.0`) and the one the published LLM-on-LIBERO code uses (`pip install hf-libero==0.1.4`) |
| robosuite | 1.4.0 | pinned by hf-libero |
| mujoco | 3.8.1 | hf-libero requires `mujoco>=3.0,<3.9`; separate from `.venv`'s 3.17 |
| bddl | 1.0.1 | pinned by hf-libero |
| torch / torchvision | 2.11.0+cpu / 0.26.0+cpu | CPU build on purpose: LIBERO needs torch only for `torch.load` of init states (robomimic also imports it); no GPU in sim workers |
| numpy | 2.5.2 | works with robosuite 1.4.0 and numba 0.68 for everything we use |
| imageio, imageio-ffmpeg, h5py, pillow | latest | rollout videos, demo HDF5s |

Pins: `playgrounds/libero/constraints-libero.txt`. The venv is 2.4 GB; assets add 409 MB in `~/.cache/libero/assets`.

## Install commands that worked

```sh
uv venv --python 3.12.10 .venv-libero
# 1. CPU torch first (9.5 min on this connection)
uv pip install --python .venv-libero/bin/python --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.11.0+cpu" "torchvision==0.26.0+cpu"
# 2. LIBERO; the CPU index stays visible so the +cpu pins resolve
uv pip install --python .venv-libero/bin/python -c playgrounds/libero/constraints-libero.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    "hf-libero==0.1.4" imageio imageio-ffmpeg h5py pillow
# 3. LIBERO config without the interactive prompt (see gotcha 2)
mkdir -p runs/libero/config   # then write config.yaml as below
.venv-libero/bin/python playgrounds/libero/smoke.py          # downloads assets on first use, ~14 min total
.venv-libero/bin/python playgrounds/libero/server.py freeze  # writes playgrounds/libero/frozen.json
```

`runs/libero/config/config.yaml` points every key at the installed package (`<site-packages>/libero/libero/{bddl_files,init_files,assets}`, `datasets` one level up). `playgrounds/libero/sim.py` sets `LIBERO_CONFIG_PATH` to that folder, so `~/.libero` is never created.

## Gotchas

1. **Rendering: use `MUJOCO_GL=glx`, not `glfw` or `egl`.** robosuite 1.4's `macros.MUJOCO_GPU_RENDERING = True` overwrites `MUJOCO_GL` with `egl` unless it is `osmesa` or `glx` (`robosuite/utils/binding_utils.py:37`). `egl` fails on this box. With `glx`, robosuite and mujoco both fall through to the GLFW context (`mujoco.glfw.GLContext`), which works through WSLg like the twin's `glfw`. `sim.py` sets it before any import (override with `LIBERO_MUJOCO_GL`).
2. **First import blocks on `input()`** ("Do you want to specify a custom path for the dataset folder? (Y/N)") when `$LIBERO_CONFIG_PATH/config.yaml` is missing. In a stdio server that would hang or eat a protocol line. Write the config file yourself (above).
3. **Assets are downloaded on the first scene load**, from the HF dataset `lerobot/libero-assets` into `~/.cache/libero/assets` (409 MB; the first reset took 467 s). Later resets read the cache.
4. **Resets are slow:** 15–85 s per env build (MJCF compile plus textures) on the shared CPU; control steps take 0.04–0.16 s each including two 256×256 renders. A 900-step episode is about 1–2 min of sim, far below Argon's ~30 s per call.
5. **Threads:** a bare LIBERO process starts ~70 threads (torch, numba, BLAS). The client sets `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMBA_NUM_THREADS=1`; `NUMBA_CACHE_DIR` goes to `runs/libero/numba_cache`.
6. **LIBERO prints to stdout.** The stdio server swaps `sys.stdout` to stderr so stdout carries protocol JSON only (as the paper's worker does).
7. **`torch.load` and `weights_only`:** hf-libero 0.1.4's `get_task_init_states` already passes `weights_only=False`, so torch 2.11 loads the numpy-pickled init states. (LeRobot reimplements the loader for the same reason.)
8. **Image orientation: the published adapter serves mirror images.** MuJoCo/robosuite return pixel rows bottom-up, so `raw[::-1]` is the true camera view. The agentview camera sits at +x looking back at the robot (`quat 0.653 0.271 0.271 0.653`), and its image-right axis is world +y. The paper's worker serves `agentview[::-1, ::-1]`, which is the true view mirrored left-right: world +y lands on the image's left, and LIBERO's "left plate" (region y ≈ −0.30) shows on the image's right. Argon's own hindsight says the same ("+y is LEFT in agentview"). By the same pixel-order reasoning, the raw wrist image is mirrored top-bottom; that hasn't been checked against the hand camera's pose.
   - Found by the reward probe at 14:45.
   - `sim.LiberoSim(views=...)`: `"paper"` gives the published mirrored views, `"true"` gives both views as `[::-1]`. `"true"` is the default from 14:50.
   - Every episode record stores `views`. The 6 feasibility episodes of `feas-argon-high-1` all ran with `"paper"`.
   - LIBERO-10 tasks 4 and 6 name left/right in their instructions. They are exactly the paper's two 0/180 tasks, which makes the mirror a candidate cause there (a hypothesis, not tested).
9. **Objects are still settling after the 10 no-op steps** at some init states (Argon's first note in a test: "objects are still dropping onto the table"). That is the published protocol; we keep it.
10. **Commanded vs achieved motion:** with absolute OSC targets interpolated at 0.01 m per 20 Hz step, a chunk ends before the arm arrives (a commanded 10 cm descent reached 7.7 cm in its 11-step chunk). The paper reports 64–72% achieved displacement; ours reproduces the mechanism. The next chunk then starts from the lagging measured pose, so the delta-limit approver clamps its first steps ("approver: 2 step(s) modified (delta_clamped ×2)").

## Smoke test (`playgrounds/libero/smoke.py`, init state 49 of every LIBERO-10 task)

All 10 tasks load, reset, step and render; the success predicate reads `False` at reset and after the scripted motion. Lift 3 cm → 2.98 cm; gripper closes to 0.0042 m per finger on nothing and opens to 0.0358; no BADQACC warnings, max contact penetration ≤ 0.02 mm, objects at rest (≤ 3e-14 m/s). Output: `runs/libero/smoke/` (`summary.json`, `strip.png`, per-task PNGs).

## Frozen evaluator

`frozen.json` holds a SHA-256 over 314 files: LIBERO's predicate code (`envs/predicates`, `envs/object_states`, `envs/problems`, `bddl_base_domain.py`, `bddl_utils.py`, `env_wrapper.py`) and every BDDL goal and init-state file of libero_10 and libero_90, plus the MuJoCo and robosuite versions. Hash `34709c4e…`. The server refuses to start if it changes.

## Human demonstrations (checked 2026-10-25, metadata only)

| Dataset | Content |
|---|---|
| `yifengzhu-hf/LIBERO-datasets` (official) | HDF5 (robomimic layout), 50 teleop demos per task for all 130 tasks (LIBERO-10: 0.9–2.1 GB per task file, ~13 GB). Stores sim states, so every demo can be replayed exactly in our sim |
| `lerobot/libero_10_image` | LeRobot v3 parquet, 379 episodes (OpenVLA-filtered regenerations), 10 fps, 256×256 agentview + wrist, 8-D state (pos, quat, gripper), 7-D action |
| `HuggingFaceVLA/libero` | LeRobot v3, 1,693 episodes, 40 tasks (spatial, object, goal, 10), 10 fps |

None was downloaded. The official HDF5s are the source for a real-data envelope (see `playgrounds/libero/README.md`).
