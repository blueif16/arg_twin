# Argon twin workflow: playbook, orchestration, evidence

Draft v1, 2026-10-24. This replaces v0. The product is the **workflow**: a playbook plus a tool surface that Argon runs end to end, with no human in the loop. It sets goals, builds and calibrates the twin, designs the learning, monitors it, and revises its own playbook. The policy is the product and the twin is where it trains: the learning path is in [policy-learning.md](policy-learning.md). Background: [Google's twin recipes](../research/google-argon-twin-pipeline.md), [AGI Factory's evaluator](../research/agi-factory-notes.md), [test plan](../test-plan-v0.md).

## Dataset choice: harder than Google's demos, with ground truth at every stage

| Role | Dataset | Why it beats Google's demo | Status (checked 2026-10-24) |
|---|---|---|---|
| **Primary** | **REASSEMBLE** ([HF LeRobot port](https://huggingface.co/datasets/robot-lev/reassemble), [paper](https://arxiv.org/abs/2502.05086), [repo](https://github.com/TUWIEN-ASL/REASSEMBLE)) | Franka arm doing contact-rich assembly and disassembly on the **NIST Task Board #1**: 17 tight-tolerance parts (pegs, gears, USB, connectors). 149 long-horizon recordings (about 5 min each), **4,551 attempts of which 516 are real failures**, hierarchical segment labels with per-segment success, wrist **force/torque**, joint and end-effector logs, 3 RGB cameras, an event camera and audio. Google's photo-to-twin demo was party props, and its value function and labeling demos had no ground truth to score against | CC-BY-4.0, 86 GB, ungated, downloadable per episode. The repo ships holder STLs, Franka and F/T sensor URDFs, and a Gazebo setup (a baseline twin). The NIST board CAD is public. The Franka MJCF is in Menagerie. **Unverified:** whether camera extrinsics (mocap-measured, per the paper) are in the HDF5 metadata |
| Stretch: Google's own sensor | **RoCo Task Board** ([HF](https://huggingface.co/datasets/SharpaIT/RoCo_TaskBoardAssembly)) | Bimanual dexterous **Sharpa hands with 10-fingertip 6-axis tactile**, the same signal Google's robot-to-sim recipe matched against, on a harder task. 562 episodes, 65-D state, 6 video streams, plus the IROS 2026 challenge | CC-BY-4.0, 1.6 TB. **No official Sharpa MJCF found**, so Argon would also have to build the hand. High payoff, high risk |
| Human footage with 3D ground truth | **OakInk2** ([paper](https://arxiv.org/abs/2403.19417)) | Long-horizon bimanual tasks with **task graphs (Primitive Dependency Graphs), initial conditions and goals as ground truth**, mocap object poses, articulated objects, and 4 calibrated views. Google's human-video demo was a single placement | Research license (check the terms). Used for footage-to-spec and footage-to-twin scoring |
| Twin validity without a robot | **real2sim-eval** ([site](https://real2sim-eval.github.io/)), **SIMPLER** ([site](https://simpler-env.github.io/)) | Released policies and data with **published real success rates**, so a twin can be scored by sim–real rank correlation (MMRV, Pearson) | Optional external-validity track |

Rejected: SO-101 pick-cube (easier than Google's demo) and RoboCap-IKEA (no calibration and no annotations, so nothing to score against).

## The playbook

The playbook is versioned skill files plus a protocol. It tells Argon how to do each job, not what the answer is:

- **Goals**: turn footage into a typed task spec. Steps form a DAG, and preconditions, postconditions and success conditions are predicates over named simulator quantities (insertion depth, part pose relative to its hole, contact state).
- **Objectives in the simulated world**: the calibration objective for the twin (below) and the success predicates the learning optimizes.
- **Learning**: the stages, data sources and improvement loop in [policy-learning.md](policy-learning.md).
- **Monitoring**: which metrics to read each generation, how to diagnose (zoom into video, plot signals, re-replay), and which edits are allowed.
- **Protocol** (run by the harness): preregister hypotheses and metrics, freeze artifacts in order, keep splits disjoint, score the exam once, run an independent grader instance.

## Tools Argon calls itself

| Tool | What it does | Why it matters |
|---|---|---|
| `probe_video(ep, cam, t0, t1, fps≤24, slowdown, crop)` | Re-encodes a window and attaches it as static mode with offsets | **Self-zoom**: Argon decides when a contact event needs slow motion. Every zoom is logged |
| `query_signals(ep, t0, t1, channels)` | Plots force/torque, joints, gripper and EE velocity as images plus JSON | Contact timing is sharper in force/torque than in pixels |
| `scene.compile / replay(ep) / render(cam, t) / diff(real, sim)` | Builds the MJCF, replays real attempts, renders from calibrated cameras, compares | The twin calibration loop |
| `learn.generate_demos / train / eval(seeds)` | Runs the learning path | The learning loop |
| `delegate(model, tasks[])` | Fans out to `gemini-3.8-flash` workers (per-episode labeling, triage); `gemini-3.5-flash-lite` validates schemas | Parallelism across 149 recordings |
| `protocol.preregister / freeze / submit_exam` | Locks artifacts and runs the held-out exam once | Scientific integrity without a human |

## Three loops of self-iteration

1. **Perception (within a task).** Argon reads the whole recording, finds uncertain events, zooms in (window, fps, slowdown, crop), checks against force/torque, and commits segment labels.
2. **Calibration and learning (within a stage).** It iterates twin parameters until the calibration objective plateaus, then iterates the learning path until held-out success plateaus. Each candidate is accepted by a paired gate.
3. **Playbook (across runs).** Argon proposes edits to its own playbook (new diagnostics, better prompts, tool-use rules). An edit is kept only if the next run improves on held-out recordings: AGI Factory's gate applied to the workflow itself. This is how the workflow gets smarter.

## Twin calibration objective (REASSEMBLE)

- **Outcome agreement on replay.** Replay every real attempt in the twin, including the 516 failures, and require the twin to reproduce the real outcome. Report **balanced accuracy**: a permissive twin that lets everything succeed scores 50%. Real failures are what separate a calibrated twin from a fake one.
- **Force/torque agreement.** The simulated wrist wrench matches the logged wrench in timing, magnitude and direction during insertion. This is Google's "physical match" objective, using a wrist F/T sensor instead of fingertip tactile.
- **Visual agreement.** Renders from calibrated cameras align with the real frames (reprojection error, silhouette IoU).
- **No-cheat rules**, as in Google's prompt: objects move only through contact, and nothing changes after t0.

## Evidence design

| Claim | Experiment | Metric |
|---|---|---|
| Footage becomes a correct spec | Spec on held-out recordings vs REASSEMBLE labels | Segment boundary F1 at IoU 0.5/0.75, order edit distance, per-segment success prediction (AUROC) |
| The twin is physically right | Replay held-out attempts | Outcome balanced accuracy, F/T error, reprojection error, compared with their Gazebo twin and a twin built from Google's single prompt |
| The learning works | Policies trained only in the twin; arms and ablations in [policy-learning.md](policy-learning.md#experiments) | Board success on the exam, per-skill success, real-data proxies, time per insertion compared with the human demos |
| **Model capability and workflow capability, separately and combined** | **2×2 factorial: {Argon, Flash} × {our playbook, Google-style single prompt}**, same data and budget, n ≥ 3 runs each | Main effect of the model, main effect of the workflow, and their interaction |
| Self-zoom matters | Ablation with `probe_video` disabled | Change in boundary F1 on fast contact events; zoom count and placement |
| The workflow improves itself | Playbook generations v0 → vN | Held-out metrics per generation; accepted vs rejected playbook edits |

Every run logs model, settings, fps, resolution, video mode, tokens, cost, latency, split, deciding checks, and failures with timestamps (test plan §4).

## Specific technical risks

- **Tight-tolerance insertion in MuJoCo.** Mesh collisions are convex hulls, so holes must be built from compound primitives. Clearance, `solref`/`solimp` and contact margin decide whether pegs jam the way the real ones did. This is exactly what the outcome-agreement objective calibrates.
- **The controller must match.** REASSEMBLE actions are end-effector targets under a Cartesian impedance controller. Replay has to emulate that controller (not joint PD), or the contact forces will not match. The F/T trace is the check. If `action` is the next measured pose, the commanded setpoint has to be reconstructed from the wrench first ([policy-learning.md](policy-learning.md#setpoint-reconstruction-comes-first)).
- **Camera extrinsics.** If they are not in the HDF5, Argon recovers them from robot kinematics: render the robot through forward kinematics and solve camera pose by alignment (the same idea DeepMind's SplatMesh paper used on ALOHA).

## Local setup facts (verified 2026-10-23)

- `.venv`: Python 3.12, `mujoco` 3.17.0, `google-genai` 2.33.0. GPU: RTX 4070 SUPER, 12 GB.
- Headless rendering in WSL works only with `MUJOCO_GL=glfw`.

## Demo visuals

MuJoCo is the physics. The look is layered on top: a real-vs-sim synced viewer with force/torque traces and contact arrows, background plates from the fixed cameras, an optional splat stage (gausscam, MuGS, or the vendored supersplat viewer), and a Blender hero clip.
