---
name: reconstruct-scene
description: Entry point and loop for stage B. Turn a robot recording, human video, single image or sketch into a MuJoCo scene written as code. The agent takes its own inventory, decides what the task needs and how precisely, then at every step writes a native estimate, runs a precision tool and a physics check, and logs the decision with log_decision. Names the skill that implements each step.
stage: build
steps: [B0, B1, B2]
---
## Goal

A MuJoCo scene, written as code, that reproduces what the input shows to the precision the task needs, and a record of how every quantity in it was obtained:

- `runs/<brain>/<run_id>/reconstruct/intake.json`: what the input contains, what the task needs, and the target error per quantity.
- The scene, built by the step skills: MJCF generated with `MjSpec`, every parameter with a sigma and a source.
- `runs/<brain>/<run_id>/decisions.jsonl` (`log_decision`): for each quantity, a pre-registered native estimate, then the decision after the tool.
- The native-vs-tools chart (`decision_summary {chart: true}`), published to the step's page.

Done means the B0, B1 and B2 checks in `playbook/validation.yaml` pass when a separate watcher measures them. Every quantity in the scene must have a decision record or be declared cosmetic in `intake.json`.

The method is the one Google's Argon twin demos show [google-argon-robotics-2026], studied in `docs/research/google-argon-twin-pipeline.md`:

1. the native look sets the scene's structure;
2. the scene is written as code: primitives, procedural meshes, convex decomposition, analytic SDFs for threads and gears, never scans;
3. render from the real cameras, compare, refine;
4. tools supply precision where the eye can't;
5. physics checks read from simulator state are the arbiter;
6. external help (calibration, hand-pose keypoints, segmentation, depth, mesh-from-image, asset lookup) only where perception is out of reach.

## How this relates to the other skills

This skill is the entry point and loop for stage B. It holds:
- intake and triage;
- the branches by data category;
- the native → tool → check discipline, with `log_decision`;
- when to call for external help;
- the traps that cut across steps;
- the order of steps.

It does **not** restate the method, parameters or numbers of the skills that implement the steps. For a robot recording, each step below names its skill. Follow that skill for how to do the step. This file decides the order and what gets logged.

| Skill | Implements, for robot recordings |
|---|---|
| `world-frame` (`playbook/skills/world-frame/SKILL.md`) | tool length, time sync, cameras, fixture height |
| `board-from-contacts` (`playbook/skills/board-from-contacts/SKILL.md`) | fixture pose from repeated contacts |
| `build-twin` (`playbook/skills/build-twin/SKILL.md`) | object geometry, task frames, success predicates, contact probes |
| `mujoco` (`playbook/skills/mujoco/SKILL.md`) | writing, stepping, rendering and checking MJCF: conventions, collision modelling, contact parameters, dt/2 |
| `replay-validate` (`playbook/skills/replay-validate/SKILL.md`) | controller, commands, hidden state, replay validation |
| `validate-checks` (`playbook/skills/validate-checks/SKILL.md`) | how a watcher proves a check can fail: known errors, negative controls |
| sim-integrity | named in `playbook/rules.md`; no SKILL.md yet. Tools: `integrity_info`, `monitor_selftest`, `run_actor` (`harness/tools/integrity.py`) |

Three cases have no step skill yet: video only, a single image or sketch, and object geometry without CAD. For those, the steps below give the goal, the check and the external help, marked **no worked instance yet**.

## Inputs

- The input as given: a recording (robot state, video, maybe force), a video, a photo or a sketch. Read downloaded data as data, with `python -I`, and never run code from `data/` or `vendors/` (`playbook/rules.md`).
- A robot model if a robot appears: MuJoCo Menagerie (`vendors/mujoco_menagerie`) or the dataset's own URDF.
- The task: what success means and, if known, the tolerance (clearance, placement accuracy).
- Knowledge base: `docs/knowledge/real2sim-twins.md`, `calibration-and-localization.md`, `contact-localization.md`, `contact-physics-and-sysid.md`, `harness-and-tools.md`.

## Tools

| Need | Tool |
|---|---|
| Decision log | `log_decision` (pre-registration, then the decision); `decision_summary` (table, and the chart with `chart: true`) |
| Look at a REASSEMBLE recording | `recording_info` (with `labels: false` for anything you will score), `query_signals`, `probe_video`, `real_frame`, `cut_clip`. These tools read REASSEMBLE's layout; a general adapter is parked (`docs/design/backlog.md`) |
| Look at any other input | Generic reads: parquet metadata with `pyarrow`, `ffprobe`, frames with OpenCV, all with `python -I` |
| The steps | The tools listed in each step skill |
| Report | `publish_media`, `record_event`. Watchers, not actors, call `record_check` |
| Working notes | `log_note` (on Gemini, free text right before a tool call can break it: `docs/research/interactions-api-notes.md`) |

Claude reads frames as PNG images. Gemini can also take a video clip; slow it down over contact moments (`read-footage`).

## Approach

### 1. Intake: write down what exists

This is the agent's own job; there is no dataset adapter (`docs/design/backlog.md`). Read the metadata, a few frames and the signals with generic tools. Write `intake.json`, then call `log_decision {step: "intake"}` once.

For each item, record present / absent / unknown, where it lives, its units, rate and clock, and how far you trust it:

- **Robot:** model, joint log, end-effector pose (and whether it is a configured frame or the physical tool), gripper, and whether commands were logged or only measured state.
- **Force:** F/T or tactile, its frame and compensation.
- **Cameras:** count; fixed or wrist; resolution; intrinsics; where the extrinsics come from; depth.
- **Other poses:** mocap and what it tracks (markers or optical centres), and whether files agree with each other.
- **Labels:** segments, success, failure reasons.
- **Geometry:** CAD or STL, known assets, dimensions stated in documents.
- **The task's tolerance:** clearance, success predicate, what decides success.

Supplied values enter as **priors to verify, never as truth**. That covers mocap, a configured TCP, CAD layouts and simulator defaults. On REASSEMBLE, several of them were wrong (Known traps).

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

### 3. Branch by data category

| Category | Calibration object | Metric scale | Fixture pose | Contact evidence | Worked instance |
|---|---|---|---|---|---|
| Robot recording, joints + video + F/T | the robot itself: forward-kinematics keypoints | forward kinematics | repeated contacts; cameras confirm | logged force vs simulated force | REASSEMBLE episode 0, Claude stand-in (`runs/claude/run-standin-1/decisions.jsonl`) |
| Robot recording, joints + video, no F/T | the robot itself | forward kinematics | triangulation with the robot-calibrated cameras, plus grasp and release poses | outcomes and contact timing from video only; widen the sampled ranges | no worked instance yet |
| Video only (human or egocentric) | supplied calibration, a target, or structure-from-motion | external: calibration, a known-size object, or metric depth with an anchor | triangulation from calibrated views | none from sensors; physics checks on the retargeted motion (force closure, lift, stable placement) | no worked instance yet. Google's demo C took hand keypoints from an upstream tool |
| Single image or sketch | none | from understanding: known object sizes, stated dimensions | one view, procedural layout | physics checks only: task success in sim | no worked instance yet. Google's demos A (photo) and E (sketch) |

Google never estimated metric scale from one uncalibrated image (`docs/research/google-argon-twin-pipeline.md`). For video only, scale comes from calibration or a reference object, never from the eye alone.

### 4. The loop at every step

For each quantity:

1. **Look, and write the native estimate first.** Give numbers and units. Use the eye for identity and structure: which side, which way is up, which object, which hole. Use code for position: on REASSEMBLE, reading absolute pixels off crops varied between readings (`world-frame` Lessons). Log it as a pre-registration: `log_decision {verdict: "open"}` with no tool and no error. The call log's order then shows the guess came before the tool.
2. **Run the precision tool.** Measure the error before → after in physical units, against data the fit did not use: held-out marks, left-out contacts, the other camera. Convert pixels to mm at the working distance (`mm_per_px`).
3. **Run the step's check,** as named in its skill. A separate watcher records it with `record_check`.
4. **Log the decision:** `log_decision` with the tool, the result, the error, a verdict (adopt / reject / revise / escalate / open) and the evidence paths.
5. **Render from the real cameras and compare.** The scene is code: fix the generator and rebuild, never patch an output by hand.

Escalate after two rounds without progress (`playbook/orchestrator.md`).

### 5. Steps, in order

Each step lists its goal, its done-check and the skill that implements it for robot recordings.

| # | Step | Goal | Done when | Skill (robot recordings) / otherwise |
|---|---|---|---|---|
| 0 | Intake and triage | Know what exists, what the task needs, the target error per quantity | `intake.json` and the intake decision exist | this skill |
| 1 | Robot and tool length | The twin's hand and tool sit where the physical ones are, not where the configuration says | Held-out keypoints on the hand pass (`B0.kp_*`). `B0.tcp_err_mm` alone certifies only the configured frame | `world-frame` (tool offset), `mujoco` (base.xml) |
| 2 | Time sync | Video, joints and force on one clock | A latency is fitted per camera, and moving frames score no worse than still ones | `world-frame` (`playbook/skills/world-frame/SKILL.md`) |
| 3 | Cameras | Intrinsics and extrinsics good enough for renders to line up and for triangulation | `B0.kp_hama1`, `B0.kp_hama2`, `B0.two_view_px`, `B0.overlay_hand` on held-out marks, re-measured by a watcher | `world-frame`; watcher: `validate-checks`. Video only or image: **no worked instance yet**; calibration target or structure-from-motion (External help) |
| 4a | Object and fixture geometry | Shapes that decide the outcome are exact; the rest approximate or cosmetic. Source order: CAD > primitives fitted from images > retrieved assets. Collision as primitives where fits matter, convex decomposition for concave containers, analytic SDFs for threads and gears | B1: `settle_penetration_mm`, `aligned_insert_rate`, `misaligned_jam_rate`, `dt_halving_agreement` | `build-twin`, `mujoco`. Without CAD: **no worked instance yet**; segmentation, depth, mesh-from-image, asset lookup |
| 4b | Object and fixture pose | Fixtures placed where the robot found them. Source order: repeated contacts > triangulation > one view scaled by the robot | `B0.board_err_mm` and the held-out across-pad error | `board-from-contacts` (x, y, yaw), `world-frame` (height). Without contacts: triangulation, **no worked instance yet** |
| 5 | Controller | The sim arm follows the real arm in free space under the controller that actually ran | `B2.free_space_mm` | `replay-validate` §1 |
| 6 | Commands | The command the robot followed, rebuilt from the log when only measured state exists. Any freedom is declared before scoring | the same free-space check; freedoms written in replay-validate before gate scoring | `replay-validate` §2 |
| 7 | Contact probes | Aligned parts go in, misaligned parts jam, and the answer holds at dt/2, before any replay | `B1.aligned_insert_rate`, `B1.misaligned_jam_rate`, `B1.dt_halving_agreement` | `build-twin`, `mujoco` (`insertion_probe`) |
| 8 | Hidden state | What the log lacks (e.g. in-hand pose) is estimated before t = 0 without reading the outcome | a label-blind rule; prediction reported apart (`B2.prediction_bacc`) | `replay-validate` §3b; candidate: parallel simulations weighted by the F/T match [inhand-gpu-2020] |
| 9 | Validation | The twin reproduces what happened per attempt and as a distribution, and a permissive twin fails | `B2.dist_seat_rate_err`, `B2.dist_force_ks`, `B2.dist_negative_control`; `B2.outcome_bacc`, `B2.force_peak_err`, `B2.contact_onset_err_s` reported | `replay-validate`, `validate-checks` |
| 10 | Integrity layer | Nothing moves except through contact after t = 0, and counted successes are physically plausible | the G checks in `validation.yaml` | `playbook/rules.md`, sim-integrity (tools only, no SKILL.md yet) |

### 6. External help

Use these only where perception is out of reach, and log each use as its own decision. Status as of 2026-10-24: checked with the installed packages and the Hugging Face cache, GPU RTX 4070 SUPER (12 GB). Nothing heavy was installed. Ids marked on demand are as their authors publish them; they were not downloaded or run here.

| Trigger | Tool or model | Returns | Here now |
|---|---|---|---|
| No intrinsics; the robot is visible | robot-keypoint bundle adjustment: `calibrate_static_cameras`, `calibrate_hand_camera` | intrinsics, extrinsics, latency, tool offset | installed (REASSEMBLE-specific) |
| A calibration target is visible (checkerboard, ArUco, ChArUco) | OpenCV `calibrateCamera`, `solvePnP`, `cv2.aruco` | intrinsics, target pose | installed: opencv-python-headless 4.13, with aruco |
| Several uncalibrated views and no robot | COLMAP through `pycolmap`; VGGT (`facebook/VGGT-1B`) | camera poses up to scale, sparse points | on demand: `pip install pycolmap`; VGGT needs its repo and weights |
| Human hands, no robot (egocentric video) | HaMeR (github `geopavlakos/hamer`, needs the MANO model files), WiLoR (github `rolpotamias/WiLoR`); MediaPipe Hands (`mediapipe`) for 2.5D | 3D hand joints and wrist pose per frame | on demand; none installed |
| Object masks: silhouettes for render-and-compare, sizes | SAM ViT-B through transformers `SamModel` (`facebook/sam-vit-base`); SAM 2 (`facebook/sam2.1-hiera-large`, transformers `Sam2Model`) | masks from point or box prompts | SAM ViT-B: weights cached, runs with the installed transformers 5.5. SAM 2: weights on demand |
| Depth from one view: heights, or scale given an anchor | Depth Anything V2 Large (`depth-anything/Depth-Anything-V2-Large-hf`, relative depth); its metric variant (`depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf`); MoGe-2 (`Ruicheng/moge-2-vitl-normal`, metric point map) | per-pixel depth; relative depth needs a scale anchor | DA-V2 Large: weights cached, runs through transformers. Metric variant: on demand. MoGe-2: weights cached, `moge` package not installed. REASSEMBLE recorded no depth (`docs/research/reassemble-data-notes.md`) |
| Parts moving or slipping across frames | CoTracker3 (`facebook/cotracker3`) | 2D point tracks with visibility | weights cached; code on demand (`pip install git+https://github.com/facebookresearch/co-tracker`) |
| An object with no CAD and no simple primitive fit | Hunyuan3D-2mini (`tencent/Hunyuan3D-2mini`), TRELLIS (`microsoft/TRELLIS-image-large`) | a textured mesh at arbitrary scale. Rescale it, decompose it for collision, and use it as visual only where a fit matters | on demand. The Hunyuan3D cache is partial (config only); `hy3dgen` is not installed |
| Find the task objects by name | OWLv2 (`google/owlv2-base-patch16-ensemble`), Grounding DINO (`IDEA-Research/grounding-dino-base`) through transformers | boxes per text query | model classes installed; weights on demand |
| Collision for a concave container | CoACD (`coacd`), `trimesh` | convex pieces | installed (coacd 1.0.14, trimesh 5.2). Cannot make a precise hole (`mujoco`) |
| Threads, gear teeth | MuJoCo SDF plugins `mujoco.sdf.nut`, `bolt`, `gear`, `bowl`, `torus` [mujoco-sdf-plugins] | analytic collision | installed (bundled `libsdf_plugin.so`); CPU only, not in MuJoCo Warp |
| A known robot or object | MuJoCo Menagerie (`vendors/mujoco_menagerie`), NIST task board CAD (`vendors/nist-taskboard`); `robot_descriptions`, `objaverse` | MJCF, URDF, meshes | Menagerie and NIST are local and read-only; the others on demand |
| Fixture pose, given F/T and repeated contacts | `board_contact_layout`, `board_from_contacts`, `board_err` | pose with sigma and a held-out error | installed (REASSEMBLE-specific) |
| Hidden in-hand pose, given F/T | parallel MuJoCo Warp worlds weighted by the F/T match [inhand-gpu-2020] | a pose distribution per attempt | mujoco-warp 3.17 installed; method in `replay-validate` |
| Pose of a CAD object from RGB-D | FoundationPose (github `NVlabs/FoundationPose`) | object pose per frame | on demand; needs depth |
| Many frames to label quickly | Gemini Flash (`gemini-3.8-flash`) as a parallel reader under Argon | labels per clip | needs the Gemini API key, which is not set (`CLAUDE.md`) |

### 7. Known traps

Each was found on REASSEMBLE. The numbers live in the decision records (ids in `runs/claude/run-standin-1/decisions.jsonl`) and in the named skill's Lessons.

| Trap | How it shows, and how to catch it | Evidence |
|---|---|---|
| The configured end-effector frame is not the physical tool | Forward kinematics matching the logged EE pose certifies the configuration, not the fingertips. No camera overlay sees tool length unless hand keypoints are scored. Fit one shared tool offset in the camera bundle adjustment, then confirm it with contact-onset timing in replay | `tool-length`, `tool-length-replay`; `world-frame` Lessons; `replay-validate` §1 "Hand geometry" |
| Edge-overlay metrics cannot detect a wrong pose | A low edge distance still passes a robot rendered at the wrong time. Before trusting a metric, run known-error tests and a negative control. Gate on held-out keypoints and two-view consistency | `edge-overlay-blind`, `cam-two-view`; `validate-checks` Lessons |
| Stale mocap | A pose file can repeat an earlier recording's pose after the fixture moved, and camera bracket poses can be duplicated or swapped. Fit the fixture from the robot's own contacts for each recording, and flag any recording where the sources disagree | `board-stale-mocap-ep117`; `board-from-contacts` Lessons; `docs/research/reassemble-data-notes.md` |
| Mirrored layouts | CAD part positions may match the real layout only under a reflection. Allow det = −1 when fitting a layout to CAD, and check the residual per feature | `hole-layout-mirrored`; `board-from-contacts` Lessons |
| Video-to-log latency | Resampling every stream to one clock does not remove capture latency. Fit a time offset per camera together with the poses, and compare moving frames with still ones | `time-sync`; `world-frame` Lessons |
| Soft-contact layers vs clearance | The soft layer looks like it eats clearance, but with margin 0 it acts only at negative distance. The real danger is the default contact parameters: they let a part slide through a rim overlap at every timestep, so the dt/2 check passes it, and only a geometric probe catches it | `contact-params`, `timestep`, `soft-layer-clearance`; `mujoco` "Contact parameters and timestep" |
| Operator aborts in the outcome labels | "Failure" labels include aborts and misses that no contact physics produces, so per-attempt accuracy plateaus and a permissive twin scores the same. Split failures label-blind by their signal signature before tuning on them, and validate at distribution level against a permissive negative control | `operator-aborts`; `replay-validate` Lessons; `playbook/validation.yaml` v10 |

### 8. What Google's prompts ask for (verbatim)

From the robotics post [google-argon-robotics-2026] (`data/refs/google-argon-robotics-2026.html`). Google published the prompts only, not the harness.

| Principle | Google's words |
|---|---|
| The native look sets the structure | "Identify only the manipulation-relevant objects (the grasped target object and the receiving container or landmark)" (human view). "Parse the spatial layout from `hand_drawn_throw_sketch.jpg`: place the 76-DoF humanoid (32-DoF body equipped with two 22-DoF five-finger dexterous hands) at the origin facing a table holding a 3-2-1 pyramid of 6 rigid target boxes, with a projectile sphere grasped in the right five-finger hand." (sketch) |
| The scene is code | "Reconstruct procedural 3D meshes and compound MuJoCo collision primitives for all 9 party props matching their real-world dimensions, colors, and resting poses, including spring-hinged deformable crown panels for the soft plush cap." (single photo). "Replace crude bounding boxes with compound convex-decomposed 3D meshes so concave containers (nested bowls, recessed wooden trays, open bins) support physical interior placement." (human view) |
| Render from the real cameras and compare | "Build a MuJoCo scene of the robot and reconstructed task objects. When each episode is replayed frame by frame, by driving the robot joints toward the logged observation.state with joint-space PD." … "(1) Visual match: MuJoCo renders from the head_left, left_wrist, and right_wrist cameras line up with, and look like, those real observations." (robot view) |
| Tools supply precision | "triangulate their 3D tabletop coordinates from the egocentric camera intrinsics/extrinsics" (human view). "Segment the 12 balls, Gold/Teal pans, and Red/Blue trays in both stereo views and triangulate their 3D coordinates." (odd ball). For control, a calibrated grid is drawn on the image: "`Yellow numbers`: World (X, Y) in centimetres (2 cm grid at `grid_z`)" |
| Physics checks are the arbiter | "(2) Physical match: the simulated contact force/torque on each Sharpa fingertip, in the sensor frame, matches the logged tactile_force in timing, magnitude and direction." (robot view). "verify `reward == 1.0` in MuJoCo physics" (single photo) |
| Only what is set before t = 0 is tuned | "What you can tune happens before simulation starts: (1) Static object assets: realistic geometry, collision shapes, mass, inertia, friction and joints, with appearance close to the real objects, plus the initial object poses at t = 0." (robot view) |
| External help where perception runs out | The human-view prompt attaches `hand_3d_keypoints.json` from an upstream tool: "[Attached: egocentric_human_video.mp4, candidate_sim_views.png, hand_3d_keypoints.json]" |

## Checks

A watcher can run these from files alone:

1. `intake.json` exists. Every item has a status, and every quantity has a tier and a target error written before its first tool call (compare its timestamp with `runs/calls/`).
2. `decision_summary` lists, for every quantity marked exact or approximate: a pre-registration (verdict `open`, no tool) and, after it in the call log, a decision with a tool, an error in physical units and existing evidence. Backfilled records are marked `backfilled: true` and are not counted toward this check.
3. Every step's own check is recorded by a watcher with `record_check`: B0, B1 and B2 in `validation.yaml`. A step without a worked instance reports its check as not started, never as passed.
4. The native-vs-tools chart is published (`publish_media`, stage B0) with the run's brain, and Claude stand-in and Argon results are never combined.
5. Integrity: no exam recording was read (G.exam_reads); no quantity was tuned after t = 0 (`playbook/rules.md`).

### Confirmatory run

The first full test of this skill is an agent following it, with no prior findings, on a recording nobody has calibrated yet:

- Use a fresh `run_id` and an empty `runs/<brain>/<run_id>/`. Do not read `twin/calib/`, `runs/claude/run-standin-1/` or the task findings in the step skills' Lessons. Methods only, per "Keep two kinds of knowledge apart" in `playbook/orchestrator.md`.
- The recording is a train recording with every camera on disk other than episode 0. At the time of writing that is episode 1 (`recording_info`). Never fit on gate or exam recordings.
- A separate watcher scores it with the same B0, B1 and B2 checks, with the targets frozen at the run's start.
- Compare it with the hand-built episode-0 twin: each check side by side, the two native-vs-tools charts, tool calls, tokens and wall time.

Until then, everything here is exploratory: episode 0 shaped both the method and the targets (validation.yaml v3 and v5).

## Lessons

**2026-10-24, claude stand-in, backfill of the episode-0 decisions (caller `claude:reconstruct-backfill`).** Evidence:
- `runs/claude/run-standin-1/decisions.jsonl`: 28 records covering all 12 step types;
- `runs/claude/run-standin-1/reconstruct/derived.json`: derived numbers, with their inputs and method;
- `runs/claude/run-standin-1/reconstruct/native_vs_tools.png`;
- `scripts/backfill_decisions_standin.py`.

These records were written after the fact, so they show the pattern but cannot prove that the native estimates came first. The confirmatory run must pre-register.

- **The native look got the structure right, but not the precision.** By eye, the static cameras were on the right side, the right way up and at the right azimuth. Measured against the final keypoint calibration, though, they would have drawn the board holes about 74–83 mm off at the board. The tools took them to about 3–4 mm on held-out marks (`cam-hama1-pose`, `cam-hama2-pose`; `derived.json`).
- **Supplied priors were wrong more often than the eye.** Only cross-checks against the robot's own kinematics and contacts found these three, each off by more than any eye check could see (`tool-length`, `plate-height`, `board-stale-mocap-ep117`). Treat every supplied value as a prior.
  - the configured tool point;
  - the plate height taken from mocap;
  - one recording's stale mocap pose.
- **Not every "after" is a held-out error.** Some are fit residuals or the spread of the fit (tool length, plate height, seat depth, contact softness). The chart marks these with a diamond. Report held-out errors wherever a step allows them.
- **A tool that looked precise was wrong twice.** The silhouette camera fit and MuJoCo's default contacts both passed their own checks. Each was caught by an independent measurement: held-out keypoints, and a geometric probe. Plan each step's check before its tool, not after.
- **2026-10-25** (claude:confirm-ep1): 2026-10-25, confirmatory B0 run on episode 1 (train, never calibrated), playbook only, no episode-0 values. Helped: pre-registering a native look, rendering it once from the real cameras and correcting by eye, then the robot-keypoint BA, reached about 2 px held-out on both static cameras from a seed that was 20-65 px off; the eye got side, up and azimuth, one camera first went to the wrong quadrant and one render round fixed it. Missing or wrong: (1) the world tools need an existing twin/calib/<rec>.json (cameras, board_contact) before the first fit, the skill never says so, and write_calib assumes episode 0 (its previous-note text; the fitted tool is a residual on a base.xml that already carries another recording`s stack, so store base offset + residual); (2) triage must count evidence per step before fitting: this recording had 3 contact parts (board needs 5; leave-one-part-out then leans on the pad prior) and the wrist camera only ever saw one empty hole and one plate corner, so its mount was start-dependent and failed the board check while fingers passed; (3) plate corners are weak model points here: edge refinement picks the front-face edge in one view and the top edge in the other, no corner was clean in both views, and with both static cameras looking along the same axis a plate-edge pose fit slides along an x-z valley; prefer holes, and let a watcher triangulate them; (4) test shared vs separate intrinsics per recording; here separate improved held-out error; (5) on a short recording every real_frame look and the tools` own median/background frames blanket the episode, leaving a watcher no free finger or wrist board-view frames at the default radius: agree the watcher`s frames, or reuse frames already excluded, before looking. Evidence: `runs/claude/run-confirm-ep1/decisions.jsonl`, `runs/claude/run-confirm-ep1/reconstruct/intake.json`, `runs/claude/run-confirm-ep1/reconstruct/native_vs_tools.png`, `runs/claude/run-confirm-ep1/world/static_fit_adopted.json`, `runs/claude/run-confirm-ep1/world/hand_fit_adopted_from_prev.json`, `runs/claude/run-confirm-ep1/board/board_err_train.json`.
- **2026-10-25** (claude:so101-actor, run-so101-1): 2026-10-25, SO-101 track (lerobot/svla_so101_pickplace: 2 fixed cameras, joints and logged leader commands, no F/T, no object CAD), first agent with no prior findings, playbook only. Helped: the branch row 'joints + video, no F/T' and the native -> tool -> check loop worked as written; with the robot as the calibration object, held-out by-eye jaw-tip marks went from 83-85 px (native: right sides and azimuths, 9-12 cm off) to 6.4 px (up) and 3.8 px (side) on train. Missing or misleading: (1) intake has no 'joint units' item: LeRobot-normalised joints with no calibration file need a per-joint scale and offset fitted with the cameras (the nominal map gave 15 mm at the fingertips, fitted 6 mm); (2) the generic tools in Tools and External help (calibrate_static_cameras, mark_candidates, the keypoint checks) are hard-wired to one robot, camera names and board keys, and their patch features need textured joint caps; a black untextured arm needed an outline (silhouette) bundle adjustment in new modules, and the registry's exam guard reads 'episode' against another dataset's splits; (3) an outline-only fit passed its own residual (2 px) yet left the gripper 15-40 px off and put a low camera 12 cm too low, the tool-length trap in another form: add gripper keypoints (a coloured cap) and a two-view point near the table, with soft_l1 (a Cauchy loss let 100k outline residuals drown the few keypoints). Evidence: `runs/claude/run-so101-1/decisions.jsonl`, `runs/claude/run-so101-1/reconstruct/intake.json`, `runs/claude/run-so101-1/marks/kpcheck_final.json`, `runs/claude/run-so101-1/replay/train_v3.json`, `runs/claude/run-so101-1/gate_results_frozen_vs_fixed_after_gate.json`, `runs/claude/run-so101-1/fix_train/one_view_train.json`, `runs/claude/run-so101-1/fix_train/kpcheck_auto_train_v2.json`.
- **2026-10-25** (claude:so101-actor, run-so101-1): 2026-10-25, SO-101 track, continued. (4) Supplied-looking priors were wrong again: the table sat 13 mm below the base (a mounting plate), the real jaws reach 4.3 mm past the CAD tips (train replay success 13 / 25 / 27 of 35 for 0 / 4.3 / 8 mm), and the MJCF servo gains were about 30x stiffer than the real follower (free-space 15 -> 6 mm). (5) A transparent box was handled by a proxy: tape corners marked by eye (a colour-mask fit was unstable because each view sees two sides) plus one rim mark, approximate tier, enough at centimetre tolerance. (6) Triage must check that train covers the workspace the gate uses: the brick sits at 5 places in blocks of about 10 episodes and the index split left one place, 35 deg of pan beyond any train start, to gate only; there the cameras still agreed (two-view 1 px) but forward kinematics was about 2 cm off along one camera's line of sight and every replay missed (0/10, against 4/5 at a train place; fixed-after-gate numbers). Robot-calibrated cameras and joint maps are only validated on the poses they were fitted on. (7) Dry-run every gate-time tool on train failure modes first: two broke on the one gate run (a brick start that needed both views crashed the replay when one camera missed the brick; a colour detector took dark specks for the cap). The frozen run stays the result of record, fixes are validated on train, and any gate rerun is labelled fixed-after-gate. Evidence: `runs/claude/run-so101-1/decisions.jsonl`, `runs/claude/run-so101-1/reconstruct/intake.json`, `runs/claude/run-so101-1/marks/kpcheck_final.json`, `runs/claude/run-so101-1/replay/train_v3.json`, `runs/claude/run-so101-1/gate_results_frozen_vs_fixed_after_gate.json`, `runs/claude/run-so101-1/fix_train/one_view_train.json`, `runs/claude/run-so101-1/fix_train/kpcheck_auto_train_v2.json`.

## Sources

- Knowledge-base ids (`docs/knowledge/references.md`): [google-argon-robotics-2026], [asid-2024], [intent-real2sim-2026], [pnd-2026], [inhand-gpu-2020], [mujoco-sdf-plugins]. The topic syntheses for these ids are in `docs/knowledge/real2sim-twins.md`, `calibration-and-localization.md`, `contact-localization.md` and `contact-physics-and-sysid.md`.
- Google's prompts: https://developers.googleblog.com/gemini-4-argon-for-robotics/ (13 Oct 2026); the local copy is `data/refs/google-argon-robotics-2026.html`. The demos are studied in `docs/research/google-argon-twin-pipeline.md`.
- REASSEMBLE facts: `docs/research/reassemble-data-notes.md`.
