---
name: build-twin
description: Stage B (B0 world frame, B1 scene) on any data source, and the entry point for building a twin. Turn a robot recording, plain video, a single image or sketch into a MuJoCo scene written as code, to the precision the task needs - intake and triage, then at every quantity a native estimate, a precision tool, a physics check and a log_decision record. A simulator benchmark already is the twin; then only its state API is documented. Pick the adapter by what you were given; source facts, the world frame and board-from-contacts methods and MuJoCo conventions are in references/.
stage: build
steps: [B0, B1, B2]
---
## Goal

A MuJoCo scene, written as code, that reproduces what the input shows to the precision the task needs, plus a record of how every quantity in it was obtained:
- `runs/<brain>/<run_id>/reconstruct/intake.json`: what the input contains, what the task needs, the target error per quantity (tier exact / approximate / cosmetic);
- the scene from `MjSpec` code, every parameter with a sigma and a source; task frames as sites; success read from sim state;
- `runs/<brain>/<run_id>/decisions.jsonl` (`log_decision`): a pre-registered native estimate per quantity, then the decision after the tool;
- the native-vs-tools chart (`decision_summary {chart: true}`) published to the step.

Done when the source's B0 and B1 checks in `playbook/validation.yaml` pass as measured by a separate watcher (REASSEMBLE stages B0, B1; SO-101 stage T2 checks with `step: B0`/`B1`), every quantity has a decision record or is declared cosmetic, and Lessons holds the run. B2 (replaying real attempts) is the `replay-validate` skill.

## Choose your path

| You were given | Calibration object and scale | Fixture pose | Contact evidence | Adapter | Source facts |
|---|---|---|---|---|---|
| **Robot logs + cameras**, with F/T | the robot itself: forward-kinematics keypoints | the robot's own contact events; cameras and mocap only confirm | logged vs simulated force | `adapters/robot-logs.md` | `references/reassemble.md`, `references/reassemble-world-frame.md`, `references/reassemble-board-from-contacts.md` |
| **Robot logs + cameras**, no F/T | the robot itself | triangulation with robot-calibrated cameras plus grasp and release poses | outcomes and timing from video; widen sampled ranges | `adapters/robot-logs.md` | `references/so101.md` |
| **Video only** (human or egocentric) | supplied calibration, a target, or SfM; scale from calibration or a known-size object, never the eye | triangulation from calibrated views | none from sensors; physics checks on the retargeted motion | `adapters/video-only.md` | `recon/README.md` (pipeline), no scored instance |
| **Single image or sketch** | none; scale from known sizes or stated dimensions | one view, procedural layout | physics checks only | `adapters/video-only.md` | Google demos A and E (`docs/research/google-argon-twin-pipeline.md`) |
| **Simulator with a state API** | the benchmark's scene is the twin | exact | exact | `adapters/simulator.md` | `references/libero.md` |
| **Prior tested runs and lessons** | reuse frozen calibration and layouts after a held-out re-check | | | `adapters/prior-runs.md` | the source's reference |

MuJoCo conventions for every path (naming, sites, tight-hole collision, contact parameters, dt/2, Warp): `references/mujoco.md`.

## Inputs

- The recording or footage and its split; whatever geometry exists (CAD, STL, stated dimensions); the task's tolerance (clearance, what decides success).
- For robot logs: joint log, EE pose (configured frame or physical tool?), gripper, F/T and its frame, camera intrinsics and any extrinsics, mocap.
- Never exam recordings (rules.md); never the replay outcomes the B2 checks score.

## Tools

- World frame and board: `calibrate_static_cameras`, the robot-keypoint calibration tools in `harness/tools/world.py`, `board_*` in `harness/tools/board.py` (`references/reassemble-world-frame.md`, `references/reassemble-board-from-contacts.md`).
- Scene: `scene_build`, `scene_check`, `insertion_probe` (`references/reassemble.md`); `MjSpec`; MuJoCo Warp for sampled worlds (`references/mujoco.md`).
- Decisions: `log_decision`, `decision_summary` (`harness/tools/decisions.py`).
- Video only: `recon/` (intake, frames, scene card, twin builder; `recon/README.md`).
- **Registry tools carried from v1** (`--skills` allowlists read this section; usage in references/): `scene_parts`, `board_layout`, `scene_render`, `mark_candidates`, `calibrate_hand_camera`, `keypoint_check`, `render_overlay`, `real_frame`, `refine_camera`, `check_world_frame`, `save_world_frame`, `board_contact_layout`, `board_from_contacts`, `board_err`, `recording_info`, `query_signals`, `probe_video`, `publish_media`, `record_event`, `record_check`, `cut_clip`, `log_note`.

## Approach

The method is the one Google's Argon twin demos show (`docs/research/google-argon-twin-pipeline.md`): the native look sets the structure; the scene is written as code (primitives, procedural meshes, convex decomposition, analytic SDFs, never scans); render from the real cameras, compare, refine; tools supply precision where the eye can't; physics checks read from simulator state are the arbiter; external help only where perception is out of reach.

1. **Intake.** Write down what exists and how far to trust it (`recon/intake.py` writes `intake.json` in this form).

### 2. Triage: decide what precision each step needs

Start from the task's tolerance. Ask which quantities change the outcome at that tolerance, and identify only those [asid-2024, intent-real2sim-2026]. Frame errors of a few millimetres decide precision assembly [pnd-2026]; appearance matters only for visual checks and image-based policies (`docs/knowledge/real2sim-twins.md`).

Give each quantity a tier and a target error in `intake.json` before measuring anything. Use build-twin's tiers: exact (measured, held-out checked), approximate (sampled, flagged) or cosmetic.

| Quantity | Needs to be exact when | Error budget is set by |
|---|---|---|
| Tool point, fixture pose, hole or slot geometry | placement or insertion | the clearance |
| Cameras | any render-and-compare, anything triangulated, any grid drawn for a brain | the size the visual checks must resolve, in mm at the working distance |
| Time sync | frames compared while the robot moves | latency × speed, against the tolerance |
| Controller and commands | replaying real attempts | free-space tracking, against the tolerance |
| Contact parameters | force matters, or jam vs seat decides success | the measured force spread and the probe geometry |
| Hidden state (e.g. in-hand pose) | the outcome depends on it | the clearance |
| Appearance | image-based policies and overlays only | the visual check |

### 4. The loop at every step

For each quantity:

1. **Look, and write the native estimate first.** Give numbers and units. Use the eye for identity and structure: which side, which way is up, which object, which hole. Use code for position: on REASSEMBLE, reading absolute pixels off crops varied between readings (`build-twin` (references/reassemble-world-frame.md) Lessons). Log it as a pre-registration: `log_decision {verdict: "open"}` with no tool and no error. The call log's order then shows the guess came before the tool.
2. **Run the precision tool.** Measure the error before → after in physical units, against data the fit did not use: held-out marks, left-out contacts, the other camera. Convert pixels to mm at the working distance (`mm_per_px`).
3. **Run the step's check,** as named in its skill. A separate watcher records it with `record_check`.
4. **Log the decision:** `log_decision` with the tool, the result, the error, a verdict (adopt / reject / revise / escalate / open) and the evidence paths.
5. **Render from the real cameras and compare.** The scene is code: fix the generator and rebuild, never patch an output by hand.

Escalate after two rounds without progress (`playbook/orchestrator.md`).

### 7. Known traps

Each was found on REASSEMBLE. The numbers live in the decision records (ids in `runs/claude/run-standin-1/decisions.jsonl`) and in the named skill's Lessons.

| Trap | How it shows, and how to catch it | Evidence |
|---|---|---|
| The configured end-effector frame is not the physical tool | Forward kinematics matching the logged EE pose certifies the configuration, not the fingertips. No camera overlay sees tool length unless hand keypoints are scored. Fit one shared tool offset in the camera bundle adjustment, then confirm it with contact-onset timing in replay | `tool-length`, `tool-length-replay`; `build-twin` (references/reassemble-world-frame.md) Lessons; `replay-validate` §1 "Hand geometry" |
| Edge-overlay metrics cannot detect a wrong pose | A low edge distance still passes a robot rendered at the wrong time. Before trusting a metric, run known-error tests and a negative control. Gate on held-out keypoints and two-view consistency | `edge-overlay-blind`, `cam-two-view`; `auditor` (references/check-validation.md) Lessons |
| Stale mocap | A pose file can repeat an earlier recording's pose after the fixture moved, and camera bracket poses can be duplicated or swapped. Fit the fixture from the robot's own contacts for each recording, and flag any recording where the sources disagree | `board-stale-mocap-ep117`; `build-twin` (references/reassemble-board-from-contacts.md) Lessons; `docs/research/reassemble-data-notes.md` |
| Mirrored layouts | CAD part positions may match the real layout only under a reflection. Allow det = −1 when fitting a layout to CAD, and check the residual per feature | `hole-layout-mirrored`; `build-twin` (references/reassemble-board-from-contacts.md) Lessons |
| Video-to-log latency | Resampling every stream to one clock does not remove capture latency. Fit a time offset per camera together with the poses, and compare moving frames with still ones | `time-sync`; `build-twin` (references/reassemble-world-frame.md) Lessons |
| Soft-contact layers vs clearance | The soft layer looks like it eats clearance, but with margin 0 it acts only at negative distance. The real danger is the default contact parameters: they let a part slide through a rim overlap at every timestep, so the dt/2 check passes it, and only a geometric probe catches it | `contact-params`, `timestep`, `soft-layer-clearance`; `mujoco` "Contact parameters and timestep" |
| Operator aborts in the outcome labels | "Failure" labels include aborts and misses that no contact physics produces, so per-attempt accuracy plateaus and a permissive twin scores the same. Split failures label-blind by their signal signature before tuning on them, and validate at distribution level against a permissive negative control | `operator-aborts`; `replay-validate` Lessons; `playbook/validation.yaml` v10 |

**Core lessons that held on every source so far**
- **Supplied priors were wrong more often than the eye**, on both REASSEMBLE (stale mocap, mirrored CAD, tool length) and SO-101 (the table 13 mm below the base); only cross-checks against the robot's own kinematics and contacts found them (`references/reconstruct-scene-v1.md` Lessons).
- **A tool that looked precise was wrong twice**; each was caught by an independent measurement, so every precision tool gets a held-out or second-source check (same).
- **Not every "after" is a held-out error**: fit residuals and spreads are labelled as such on the chart (same).

## Per-model notes

- **Claude (stand-in).** Reads PNG renders and crops; good at identity and structure by eye, not at absolute pixels (crops varied between readings, `references/reassemble-world-frame.md`). Positions come from code.
- **Argon.** Ask for pixel picks as normalized `[y, x]` 0–1000 (clean 4/4); `(u, v)` came back in the 0–1000 frame for one or both axes in 6 of 10 runs (`runs/argon/probes/b0/findings.md`). Its 3D distance guesses run about 35 % short, so picks go to PnP in code and then to the calibration tool (seed 9–28 px). With the world-frame tools in an agentic run it reached 1.43 / 1.72 px and two-view 0.77 px for $0.32 (same). For numbers use `code_execution`, free text or high thinking, never JSON `response_format` at low/medium (`runs/argon/probes/api/findings.md` §1). Writing MJCF, it gave sound physics (10 of 12 files passed every check) but could not place the board: a handedness prior overrode what it saw (3 of 7 chains flipped to det +1), depth along the view was ambiguous, it anchored on mocap and claimed checks that never ran (`runs/argon/probes/scene/findings.md`). Give it the robot-contact layout and a check it must run, not overlays to judge. High thinking with 6 images never answered (b0 findings).

## Checks

The source's B0 and B1 checks in `playbook/validation.yaml`, measured by a watcher with `record_check` (method for proving what a check detects: `auditor/references/check-validation.md`). Core checks on every path:

| Check | How |
|---|---|
| held-out keypoints / two-view px per camera | marks the fit never used (REASSEMBLE `B0.kp_*`, `B0.two_view_px`; SO-101 `T2.kp_px`) |
| fixture / hole error in mm | leave-one-out contacts (REASSEMBLE `board_err`); triangulation for no-F/T sources |
| scene sanity | `scene_check`: loads, no contact deeper than 0.1 mm at t = 0, settle, same result at dt/2 |
| geometry probe | `insertion_probe` or the task's probe at the true clearance |
| decision coverage | every non-cosmetic quantity has a `log_decision` pre-registration and verdict |
| video only, provisional | scale from a reference object stated with its error; a human confirms the scene render against 2 frames |

## Lessons

Dated entries from every source, one line each; the full entry, with its numbers, is in the named reference. New lessons go here (`append_lesson`) with the source named; detail goes to `references/<source>.md`.

- **2026-10-24** (claude): claude-opus-5.5 stand-in, round and square pegs 1–4. Evidence: `runs/calls/20261024.jsonl`, `runs/scene/probe_campaign_20261024_final.txt`, `runs/scene/probe`; full entry in `references/reassemble.md` (v1 `build-twin`).
- **2026-10-24** (claude:b2-replay): Vertical stack corrected. Plate top +30.3 mm above the mocap board z. Seat depth 20 +- 3 mm. Pads 9 +- 5 mm below a peg's top end. This resolves the 'TCP drops 16-21 mm…. Evidence: `runs/claude/run-standin-1`; full entry in `references/reassemble.md` (v1 `build-twin`).
- **2026-10-24** (claude): episode 0, claude stand-in actor, robot-keypoint recalibration. Evidence: `runs/claude/run-standin-1`, `runs/status/checks.jsonl`; full entry in `references/reassemble-world-frame.md` (v1 `world-frame`).
- **2026-10-24** (claude): first pass, claude stand-in. Evidence: `runs/world/refine`, `runs/world/overlays`, `runs/claude/run-confirm-ep1`; full entry in `references/reassemble-world-frame.md` (v1 `world-frame`).
- **2026-10-24** (claude): Mocap is the weak link. Evidence: `runs/calls/20261024.jsonl`, `runs/claude/run-standin-1`; full entry in `references/reassemble-board-from-contacts.md` (v1 `board-from-contacts`).
- **2026-10-24** (claude): Primitive holes work as intended. Evidence: `runs/calls/20261024.jsonl`, `runs/scene/experiments`, `runs/scene/probe_campaign_20261024_final.txt`; full entry in `references/mujoco.md` (v1 `mujoco`).
- **2026-10-24** (claude:b2-replay): Replaying a real arm: torque-controlled FR3 in MuJoCo. Turn the menagerie position actuators into motors and set body gravcomp=1 on fr3_link and gripper/ to stand in for…. Evidence: `runs/claude/run-standin-1`; full entry in `references/mujoco.md` (v1 `mujoco`).
- **2026-10-24** (claude:b2-replay): Grasping with contacts so the part can be released by the logged gripper. Put the part as a free body between the pads and find the finger opening where the pads just to…. Evidence: `runs/claude/run-standin-1`; full entry in `references/mujoco.md` (v1 `mujoco`).
- **2026-10-24** (claude:b2-replay): dt/2 and cost for replays. 48 compliant peg replays gave 48/48 identical outcomes at dt 0.5 ms; median peak-force difference 0.5 N, p90 3.6 N. The Python controller cost…. Evidence: `runs/claude/run-standin-1`; full entry in `references/mujoco.md` (v1 `mujoco`).
- **2026-10-24** (claude:b2-replay): Torque-controlled arm replays: Cartesian impedance nullspace potentials ring without joint friction. REASSEMBLE's joint-limit potential gives several Nm once FR3 |q2| >…. Evidence: `runs/claude/run-standin-1`; full entry in `references/mujoco.md` (v1 `mujoco`).
- **2026-10-24** (claude:b2-replay): Hand frames: the logged Franka O_T_EE is a configured point, not the pad centre. In REASSEMBLE it lies 22.9 mm beyond the pads; base.xml now mounts the hand on a 47.7 mm…. Evidence: `runs/claude/run-standin-1`; full entry in `references/mujoco.md` (v1 `mujoco`).
- **2026-10-24** (claude:b2-replay): Part friction against tight holes after release: with friction 0.30 released pegs stuck 2-4 mm deep at 6-12 deg tilt where the real ones fell in. 0.15 seated more withou…. Evidence: `runs/claude/run-standin-1`; full entry in `references/mujoco.md` (v1 `mujoco`).
- **2026-10-24** (claude:b2-replay): MuJoCo Warp for per-attempt particle filters. Replace a Python per-tick controller by an equivalent Cartesian impedance rig, so the whole controller runs inside mjw.step…. Evidence: full entry in `references/mujoco.md` (v1 `mujoco`).
- **2026-10-25** (claude:b2-5b): Modelling a compliant parallel-jaw grasp. Soft MuJoCo contacts are a poor pad model: raising the pad-part timeconst per part to match a measured squeeze compression coll…. Evidence: `runs/claude/run-standin-1`; full entry in `references/mujoco.md` (v1 `mujoco`).
- **2026-10-24** (claude): claude stand-in, backfill of the episode-0 decisions. Evidence: `runs/claude/run-standin-1`; full entry in `references/reconstruct-scene-v1.md` (v1 `reconstruct-scene`).
- **2026-10-25** (claude:confirm-ep1): confirmatory B0 run on episode 1, playbook only, no episode-0 values. Helped: pre-registering a native look, rendering it once from the real cameras and correcting by ey…. Evidence: `runs/claude/run-confirm-ep1`; full entry in `references/reconstruct-scene-v1.md` (v1 `reconstruct-scene`).
- **2026-10-25** (claude:so101-actor): SO-101 track, first agent with no prior findings, playbook only. Helped: the branch row 'joints + video, no F/T' and the native -> tool -> check loop worked as written;…. Evidence: `runs/claude/run-so101-1`; full entry in `references/reconstruct-scene-v1.md` (v1 `reconstruct-scene`).
- **2026-10-25** (claude:so101-actor): SO-101 track, continued. Supplied-looking priors were wrong again: the table sat 13 mm below the base, the real jaws reach 4.3 mm past the CAD tips, and the MJCF servo g…. Evidence: `runs/claude/run-so101-1`; full entry in `references/reconstruct-scene-v1.md` (v1 `reconstruct-scene`).

## Sources

- v1 skills moved unchanged into references: `reconstruct-scene` → `references/reconstruct-scene-v1.md`; `build-twin` → `references/reassemble.md`; `world-frame` → `references/reassemble-world-frame.md`; `board-from-contacts` → `references/reassemble-board-from-contacts.md`; `mujoco` → `references/mujoco.md` (archived at `playbook/archive/skills-v1/`).
- `runs/argon/probes/b0/findings.md`, `runs/argon/probes/scene/findings.md`, `runs/argon/probes/api/findings.md`.
- docs/knowledge/real2sim-twins.md, calibration-and-localization.md, contact-localization.md, video-to-twin.md; `docs/research/google-argon-twin-pipeline.md`.
