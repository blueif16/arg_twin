---
name: replay-validate
description: Stage B2/F. Replay real insert attempts in the twin from the real robot's rebuilt command (not its logged pose), with the in-hand pose registered label-blind (reconstruction) or not (prediction), and measure whether the twin reproduces seat vs jam, the contact force and the contact onset; includes the free-space controller check, a negative control, dt/2 and a geometry probe. Use it for any part family once its holes pass B1.
stage: build
steps: [B2, F]
---
Entry point: `reconstruct-scene` (`playbook/skills/reconstruct-scene/SKILL.md`)

## Goal

For one part family, a frozen parameter set (`runs/<brain>/<run_id>/replay/params_v2.json`; v1 `params.json` kept) with which `replay_batch` on held-out recordings reproduces what really happened:

1. **Outcome**: the part ends seated in the twin exactly when the dataset says the attempt succeeded. Real failures must be included (`B2.outcome_bacc`).
2. **Force**: the peak contact force matches the wrist sensor (`B2.force_peak_err`). Report it relative and in N.
3. **Timing**: contact starts at the same time (`B2.contact_onset_err_s`).
4. **Controls that show the test measures something**: a permissive twin loses the distinction, results hold at dt/2, and the free-space controller check passes.

Two modes (validation.yaml v8):
- **Reconstruction** (scored: `B2.outcome_bacc`, force, onset). Everything replays as logged except what the log lacks: the in-hand pose, registered per attempt before t = 0 by the label-blind rule in Approach 3b, and the stiffness inside 200–800 N/m. Any other freedom must be declared here before gate scoring.
- **Prediction** (`B2.prediction_bacc`, track): the same with `registration: none`.

In both, where the part goes is decided by `mj_step`. Nothing copies the logged pose or outcome after t = 0 (`playbook/rules.md`).

## Inputs

- Per-frame robot signals: joints, joint velocities, gripper, `ee_pose` (O_T_EE = site `tcp`), EE twist, `force.base` and `torque.base` (force on the tool; torque about the **base origin**), segment labels. Local parquet in `data/reassemble/`; gate and further train episodes are in `data/reassemble-gate/` (downloaded by board-from-contacts). Facts and caveats: `docs/research/reassemble-data-notes.md`.
- The controller that ran: `vendors/reassemble/catkin_reas/src/franka_reassemble/reassemble_controllers/src/cartesian_impedance_controller_damping_ratio.cpp`, gains in `cfg/compliance_param.cfg`, and the teleop node `reassemble_haptic/scripts/teleop_franka.py`. All read-only.
- B1 holes and parts (`playbook/skills/build-twin`, `twin/board.py`, `twin/parts.py`). Board pose per recording from board-from-contacts (`runs/<brain>/<run_id>/board/<recording>.json`), else `twin/calib`, else mocap.
- `playbook/splits.json`. Identify on **train** only. The tools refuse exam episodes.

## Tools

| Tool | Use |
|---|---|
| `replay_attempts {episode, family}` | In-hand Insert attempts with real outcome, contact onset (\|F − F_ref\| > 2 N), gripper release. Open-finger pushes are excluded (no part in hand) |
| `replay_attempt {episode, segment_index, mode, params, video}` | One replay. `compliant` = validation; `stiff` = geometry probe; `free` = approach only. Returns outcome, peak force, onset, tracking, a trace plot, and with `video` a real-vs-twin clip with force strips |
| `replay_batch {split, family, params, tag}` | All attempts of a split → bacc, force, onset (the B2 numbers) + per-attempt rows + JSON |
| `replay_controller_check {what: free_space \| k_bracket}` | Controller and setpoint vs the real arm in free space; stiffness bracket from release jumps and setpoint roughness |
| `replay_geometry_probe {episode, segment_index, offsets_mm}` | Stiff replay with the board shifted 0..3 mm: force vs hole offset |
| `make_comparison`, `publish_media`, `record_event` | Visuals and notes (watchers use `record_check`, never the actor) |

Code: `twin/control.py` (controller, reconstruction), `twin/replay.py` (attempts, scene, replay loop, scoring, sweeps), `twin/replay_viz.py`.
GPU particle filter: `twin/rig.py`, `twin/pf.py`, `twin/pf_batch.py`, `twin/pf_report.py` (§3c; known issue in Lessons).

**Frozen parameter files** (`runs/claude/run-standin-1/replay/`, all kept):

| file | `params_version` | what it is |
|---|---|---|
| `params.json` | v1 | open-loop, old hand, joint friction off |
| `params_v2.json` | v2 | corrected hand, `inhand` registration (`full`), friction 0.15, nullspace potentials off |
| `params_v3.json` | v3 (tool default) | v2 + all peg holes widened 0.2 mm per side |
| `params_v4.json` | v4 | `params` = v3, plus a `pf` block for the GPU particle filter (CPU tools use only `params`) |

**Running your own formulation** (CPU tools: `replay_attempt`, `replay_batch`, `replay_controller_check`, `replay_geometry_probe`):
- Write a JSON file under `runs/<brain>/<run_id>/...`, either `{"params": {...}}` or a flat dict of `twin.replay.ReplayParams` fields. Unknown keys are rejected.
- Pass it as `params_file`. Merge order: frozen file (`params_version`, only when `frozen` is true) < `params_file` < inline `params`. With `frozen: false` your file sits on top of `ReplayParams` defaults alone.

  ```
  .venv/bin/python -m harness.cli call replay_batch '{"split":"train","frozen":false,"params_file":"runs/<brain>/<run_id>/my_params.json","tag":"my_fit"}' --caller <agent>
  ```

- All 308 train peg windows take about 2–2.5 min with the default 15 processes (16-core box, 5–10 s per window).
- The result JSON goes to `runs/<brain>/<run_id>/replay/<tag>.json`. The reply has `summary` (bacc, TPR/TNR, force, onset), `summary_released_only` and per-attempt `rows`.
- Use `episodes: [...]` for a subset. Gate is for the watcher only; exam is refused.


## Approach

### 1. Reproduce the controller that ran, not a generic one

Read the vendor controller first. REASSEMBLE's controller differs from franka_ros [franka-ros-2024] in ways that change a replay:

| | franka_ros example | REASSEMBLE damping-ratio controller |
|---|---|---|
| damping | D = 2√K (unit mass) | D = Q diag(2ζ√λᵢ) Qᵀ from Λ = QQᵀ, K = Q diag(λ) Qᵀ (mass-aware; about 2.5× more damping at K = 500) |
| velocity term | D·J q̇ | D·(J q̇ − v_d); the teleop never publishes v_d, so it is D·J q̇ |
| nullspace | spring to the start q | (1−r)(−d q̇) + r(−P ∇det(JJᵀ)) + (−JL ∇limits), projected with N = I − Jᵀ pinv(Jᵀ) |
| defaults | 200 N/m, 10 Nm/rad, Δτ_max 1.0 | 500 N/m, 50 Nm/rad, ζ 1, Δτ_max 0.5 Nm/tick; older values 200/10 and 800/75 |

The parts both share:
- the orientation error is the **quaternion vector part**, so the rotational spring is K_r·sin(θ/2), about K_r/2 per rad;
- the target filter is x_d ← 0.005 x_target + 0.995 x_d per 1 kHz tick, a time constant of 0.2 s;
- Coriolis is added and gravity is left to the firmware.

In MuJoCo:
- arm actuators become torque motors;
- `gravcomp = 1` on the arm and hand bodies (not the held part, whose weight the real sensor sees);
- the controller runs at 1 kHz with a zero-order hold.

The manipulability gradient can be refreshed every 10 ticks (identical tracking, 2.3× faster).

**Declared deviation: nullspace potentials off** (`ns_jl = ns_manip = 0`; nullspace damping kept). REASSEMBLE's joint-limit potential (tanh wall, gain 10) produces several Nm once |q2| > ~1.2 rad. Without the real arm's joint friction (left out, see §2) that nullspace ringing goes undamped and leaks into the EE through the kinematic projection. Over 303 approach windows: median 0.99 → 0.65 mm, p90 4.1 → 1.8 mm, max 434 → 17 mm. They act only in the nullspace on the real arm, so the logged EE motion, which the setpoint is rebuilt from, already contains their effect.

**Hand geometry.** The logged O_T_EE (0.174 m from the flange) is a configured point **22.9 mm beyond the pad centres**, not the fingertips. The camera bundle adjustment puts the hand on a 47.7 mm F/T stack (`tcp.hand_offset_m`; world-frame Lessons). `base.xml` places the hand there and keeps the `tcp` site on O_T_EE, so the setpoint maths is unchanged. Parts are held at the pad centre (`<slug>/grasp` ↔ pad centre), never at `tcp`.

### 2. Rebuild the command from the log

The dataset stores no commanded pose: `action` is the next measured pose. The equilibrium follows from the controller's own dynamics, Λa = −K e − D v + W_ext:

    e = K⁻¹ (W_ext − D v − Λ a),   x_d = x − e_pos,   q_d = q ⊗ [√(1−|u|²), u],  u = −Rᵀ e_rot

- W_ext = [F_base; τ_tcp]. Here τ_tcp = torque.base − p_tcp × F_base, because `torque.base` is about the base origin (`force_2_base.py`).
- `static` (x − F/K) is DexForce's force-informed target with the sign flipped for a force on the tool [dexforce-2025]. `dynamic` adds the damping lead D v, which is as large as F/K: about 4–5 mm at 30 mm/s.
- **Tare the sensor in the tool frame** using the window's first 10 frames, which are pre-contact by construction. Subtract only the bias and keep the part's weight, because the twin's part has weight too.
- **Leave joint friction out of the twin.** The rebuilt x_d already contains the real arm's friction, so menagerie's 1.1 Nm `frictionloss` would count it twice.
- x_d is the post-filter equilibrium, so drive it directly with the target filter off.

### 3. The part in the hand: contact grasp, released by the log

The part is a free body between the finger pads.
- **Grip:** a force-limited tendon servo squeezes it (`grip_force` per finger; 60 N from `teleop_franka.py`).
- **Pads:** condim 4 with sliding `pad_friction` and torsional `pad_torsion`. Torsion is what resists pivoting about the closing axis.
- **Pre-roll before t = 0:** the arm is held at the logged start pose while the gripper closes. Fingers start at the opening where the pads touch, found by bisecting `mj_geomDistance`.
- **Release:** when the logged finger width opens, the servo follows it and the part is released. The window runs `after_release_s` past the release, and the outcome is the B1 `seated` predicate at the end.

This matters because operators let go once the tip has entered and gravity finishes the job.

An alternative is a joint-chain grasp: hinges with springs and `frictionloss` plus a slide. It identifies faster, but it cannot release, so it can only score "seated at release". That is wrong for this data.

### 3b. Reconstruction: registering the in-hand pose (label-blind)

The log has the hand pose and the gripper, not where the part sits in the fingers. One rule for every attempt uses only the attempt's logged poses, gripper signal and the calibrated board, never `segment.success`. It sets the part's initial pose in the hand before t = 0 (`registration: inhand`, `register_inhand` in `twin/replay.py`):

1. **Reference frame k_ref.** The release frame if the logged gripper opened; otherwise the frame where the held TCP was lowest along the board normal, from contact onset to the end of the hold.
2. **Rotation in the hand (`reg_rotation: full`).** The part's axis is set along the hole axis at k_ref: rock about the hand x axis, then pivot about the closing (y) axis. `none` keeps it aligned with the hand; `pivot` only rotates in the pad plane.
3. **Height (`reg_height: ref`).** The tip sits `reg_depth` = 2 mm below the plate top at k_ref if the gripper opened there, or on the plate top at the lowest held TCP if it never opened. This is converted to the pad-centre distance below the part's top end and clipped to 0–30 mm.
4. **Position.** The grasp point is offset along the pads (hand x, ±8 mm) and across them (hand y, ±6 mm) so the tip lies on the hole axis at k_ref.
   - The y offset is held by an offset on the finger-coupling equality (q1 = q2 + 2 dy), set before t = 0.
   - It absorbs board error across the pads and is reported per attempt.

What it leaks:
- The attempt's own pose at k_ref, which the v8 framing allows.
- Whether the gripper opened: 23 of the 24 never-opened train windows are real failures, and they score as failures in any mode because the part retracts with the hand.

Report bacc on released windows too. The registration is "the part was entering at k_ref" for every attempt, success or failure, so failures must jam through physics after that.

Diagnostics per replay: `registration.{pad_offset_mm, closing_offset_mm, rock_deg, pivot_deg, grasp_from_top_mm, tcp_gap_at_ref_mm, sim_lateral_at_ref_mm, sim_depth_at_ref_mm}` and `start_part_board_pen_mm`.

Physics alternative tried: finger rock/pivot hinges at the pad centres (`finger_rock_k`, `finger_pivot_k`, with spring rest at the registered angles). They hold any registered rotation and add compliance, but cost more TNR than they gain TPR on train.

### 3c. In-hand pose from force: parallel worlds (declared 2026-10-24, before running beyond a 1-attempt dev check)

After [inhand-gpu-2020]: many simulations run the same commands with perturbed in-hand pose and contact parameters, and the ones whose contact feedback matches the real sensors are kept. Code: `twin/rig.py` (Cartesian impedance rig) and `twin/pf.py`.

- **Simulator.** The arm is replaced by the **Cartesian impedance rig**: the hand on 3 slides + 3 hinges at the logged TCP pose of the window start.
  - Position actuators: kp = K (trans) and K_r/2 (rot).
  - kv and the reflected inertia (joint armature) come from diag(D) and diag(Λ) of the full FR3 model at the logged joints, in the TCP frame.
  - The wrist F/T is MuJoCo force/torque sensors at the TCP: the external wrench is −sensor.
  - dt = 1/990 s, so a 30 Hz frame is exactly 33 steps. The rig follows the rebuilt setpoint with linear interpolation inside each frame.
  - It runs on mujoco_warp with all worlds in one batch.
- **N = 1024 worlds per attempt.** Wall time is set by step latency, not by N: 64 worlds took 18 s and 1024 took 33 s on one 10 s window. N is justified per run by the ESS after tempering and by agreement of P(seat) between the two halves of the worlds.
- **Prior per world** (all before t = 0, no labels used):
  - In-hand pose drawn from N(mean, std) of the label-blind registration (§3b) over all train attempts (`runs/claude/run-standin-1/replay/inhand_prior.json`):

    | | spread |
    |---|---|
    | along-pad slide dx | 5.1 mm |
    | closing offset dy | 3.2 mm |
    | pad-centre depth below the part top g | 12 ± 7.8 mm |
    | rock | 3.5° |
    | pivot | 7.8° |

  - Rock is held by finger hinges at the pad centre (2 Nm/rad, spring rest = the world's rock); dy by the finger-coupling offset.
  - Contact: part friction U(0.10, 0.30), hole widening U(0, 0.2) mm per side (the cousins ranges).
  - Permissive twin: the same in-hand prior, friction 0.05, holes +1 mm per side.
- **Likelihood.**
  - Residual between sim and real tared 6-axis wrench (force on the tool, torque about the TCP, base frame), each as a delta from its window's first 10 frames.
  - Frames subsampled to 5 Hz up to t_end; σ_F = 2 N, σ_τ = 0.15 Nm.
  - log w = β · log L with **adaptive tempering**: β = the largest value in (0, 1] whose ESS ≥ N/10. The raw likelihood is degenerate (ESS 1) because of systematic model error.
- **No resampling or state copying between worlds** after t = 0: worlds are only weighted (selection), per rules.md.
- **Two label-blind modes:**
  - **Prediction:** t_end = decision time = the real contact onset (|ΔF| > 2 N) + 0.3 s, capped at release − 0.3 s; with no onset, release − 0.3 s.
  - **Reconstruction:** t_end = release.
  - Both report P(seat) = Σ w·seated (world outcome at window end), the weighted peak force, and posterior moments of the in-hand pose.
- **What is tuned, and how.** Only the decision threshold on P(seat) for the per-attempt bacc (track metrics), by 5-fold recording-level CV inside train. The distribution metrics use P(seat) directly:
  - seat rate = mean P(seat);
  - force KS uses one world per attempt drawn by weight, with a fixed seed.

**Variant 3c-reg** (declared 2026-10-25, after the population-prior run and before running this variant): **reconstruction only**.
- Why: with the population prior, reconstruction P(seat) was about 0.11 for both real successes and failures (interim, 74 attempts), against a real seat rate of 0.78. Few prior worlds put the part over the hole, and force does not find them.
- Proposal centred on the attempt's own label-blind registration (§3b): dx, dy ± 1 mm, g ± 2 mm, rock and pivot ± 1.5° (Gaussian SD), friction and hole widening as in 3c.
- N = 512 per attempt (a narrower proposal; wall time is latency-bound).
- Its prediction-mode output is **not** used: the registration uses the release frame.
- Same likelihood, tempering, no resampling. The permissive twin uses the same proposal.

### 4. Windows, outcome, metrics

- **Window:** from contact onset − 1.5 s to release + 1 s. Keep windows short, because the operator closed the loop on real forces.
- **Onset:** first frame with \|F − F_ref\| > 2 N, where F_ref is the median over the window's first 10 frames, in both real and sim. The 5 N teleop threshold misses a third of the attempts.
- **Sim force:** contacts on the hand and fingers from everything else (the part through the pads, the board). That is what the wrist sensor sees.
- **Peak force:** the maximum \|ΔF\| while the part is held.

### 5. Separate the error sources before identifying anything

Run these in order:
1. **Free-space check.** Approach windows: tracking RMSE sim vs logged. This tests controller, Λ, reconstruction and tare.
2. **Prediction replay**, `registration: none` (`B2.prediction_bacc`).
3. **Reconstruction replay**, `registration: inhand` (§3b; the scored B2 mode). Read `tcp_gap_at_ref_mm`: when the twin's hand ends > 2 mm from the real hand at k_ref, contact diverged before release and the registration no longer holds. The old `release_oracle` / `full_oracle` shifted the board and placed rotations the pads cannot hold; they are kept only for comparison.
4. **Negative control.** Holes widened 1 mm per side and friction 0.05, on the same windows. A twin with real discrimination must lose it here.
5. **dt/2 rerun** of the whole batch.
6. **Sensitivity sweep.** `pad_offset` 0/±1/±2/±3/±5 mm shifts the board across the pads (the closing axis). bacc and seat rate vs offset set the board-placement target.
7. **Geometry probe.** Stiff replay of the logged pose with board offsets. It is only informative when the in-hand pose is known.

**Identification** (train only):
- Pick parameters that the blind replay is sensitive to and that do not use the outcome. Onset timing pins the grasp height; it is a zero crossing of the signed onset error.
- Contact softness, friction and pad torsion need registered windows. Score them against the force sensor [mujoco-1124-2023], [qbit-2025] randomizes them to cover the real spread.
- Use `replay_batch` with 15 processes: about 2 min per 300 windows.
- MuJoCo Warp (1,024 worlds) only pays once the controller is a Warp kernel or is replaced by an equivalent Cartesian impedance rig. Not done here.

### 6. For gears and connectors

What carries over:
- Steps 1–2 and the free-space check (same arm and controller).
- The grasp, window and scoring code (`family` argument).
- The order in step 5.

What is new:
- The B1 geometry for shafts and keyed shells (`build-twin` §6).
- Where the part sits in the fingers (gears are gripped on the rim).
- Outcome predicates: twist-lock for BNC and waterproof, a latch for Ethernet.

Gears fail mostly in meshing with the neighbour, so teeth must be modelled before outcomes mean anything.

## Checks

A watcher runs these with the frozen params. Actors never run them on gate. Targets are in `validation.yaml` B2.

| id | how | pass |
|---|---|---|
| B2.outcome_bacc | `replay_batch {split: gate}` → `summary.bacc` | ≥ 0.8, counted only with ≥ 30 real failures in the scored set (validation v6) and a Clopper–Pearson interval |
| B2.force_peak_err | same → `force_peak_rel_err_median` (and the N error from the JSON) | ≤ 0.3 |
| B2.contact_onset_err_s | same → `onset_err_s_median` | ≤ 0.2 |
| dt/2 | `replay_batch` with `{timestep: 0.0005}` | ≥ 95% of outcomes unchanged |
| B2.free_space_mm | `replay_controller_check {what: free_space}` | median ≤ 1.0 mm |
| B2.negative_control_drop | frozen bacc − permissive bacc (`{hole_widen: 0.001, friction: 0.05}`), same windows and mode | ≥ 0.15 |
| B2.prediction_bacc | `replay_batch` with `{registration: none}` | track |
| registration | `params.registration` in the scored run | `inhand` for B2.outcome_bacc, force, onset; `none` for prediction |

## Lessons

**2026-10-24, claude-opus-5.5 stand-in, pegs (caller `claude-b2-replay` in `runs/calls/20261024.jsonl`; results in `runs/claude/run-standin-1/replay/`).**

**Controller and reconstruction**
- **Free space** (`free_space_check.json`, 48 approach windows):

  | setpoint | median RMSE |
  |---|---|
  | static x_d = x | 4.2 mm |
  | dynamic (+D v) | 0.97 mm |
  | dynamic + Λa | 0.73 mm, 0.22° |

  - With menagerie's joint friction kept, the error is 5.1 mm.
  - Without the tool-frame tare, p90 is 13.5 mm. Episodes 2–3 carry a 5–6 Nm `torque.base` bias in free space; tare brings p90 to 3.0 mm.
  - At dt/2: 1.04 mm. Refreshing the manipulability gradient every 10 ticks: identical.
- **K cannot be identified from these data.**
  - Free-space error falls with K (2.0 / 0.97 / 0.68 mm for 200/500/800). That is self-consistency amplified by compliance, not evidence.
  - Setpoint roughness always prefers stiffer K: the pose is smooth (0.09 mm above 3 Hz) and the F/T noise is not.
  - 20 clean release jumps give static K median 1920 (IQR 833–2611) and dynamic median 1013 (`release_events.json`), biased by friction and partial releases.
  - Blind force error favours K = 200 (0.58 vs 0.81 vs 0.99) only because a softer arm hides misregistration (`train_batches.json`).
  - Kept the code default, 500/50.
- **F/T contact-point localization is unusable.** The line of action from `torque.base` misses the hole by tens of mm, up to 140 mm, because a 0.6 m lever turns 0.1 N of error into 15 mm at 4 N.

**Geometry facts the replay needed**
- **Seat depth ≈ 20 mm, not B1's 9 mm.** Episode-0 Remove pull-outs travel 17–23 mm while the pull force lasts. Peg tops in hama1 sit ~25–35 mm above the board origin.
- **Grasp height:** from the zero crossing of the signed onset error, the part's top sits ~1.5 mm above the TCP (`scan_grasp_height.json`: −0.08 s at 0 mm, +0.05 s at 2.5 mm).
- **Operators release once the tip is in.** The TCP at insert release sits ~43 mm above the board, so gravity finishes the job. The outcome must therefore be read after a physical release.
- **The part pivots in the fingers.** The hand tilts 5.4° (median) at release while the part is in the hole. Assuming the part was vertical at the Pick does not explain it: predicted part tilt 5.3°.

**Results, frozen params** (`params.json`; `all_frozen.json`; 308 in-hand peg windows from 89 train recordings, 69 real failures)
- **bacc 0.53** (TPR 0.13, TNR 0.93).
- **Peak force:** relative error 0.71; 4.5 N in absolute terms (IQR 2.3–8.7), almost all over-prediction (+4.5 N signed). Real peak median 6.6 N.
- **Onset error:** median 0.42 s.
- **dt/2:** 296/308 outcomes unchanged (96%; 48/48 on the first local subset); bacc 0.54 at dt/2 (`all_dt05.json`).
- **Clopper–Pearson 95%:** TPR 31/239 (0.09–0.18), TNR 64/69 (0.84–0.98). The twin seats almost nothing, so it scores "failure" right by default.
- **Negative control fails.** Holes widened 1 mm per side with friction 0.05 seat 82 of 308 instead of 36, and score *higher* (bacc 0.57) than the frozen twin (0.53, `all_permissive.json`). The test currently measures nothing.
- **Why:**
  - At release the twin's part is a median 6.5 mm (IQR 2.3–11.3) from the hole axis, tilted 5.9°.
  - Peg clearances are 0.05–0.25 mm and chamfers 0.4–1.6 mm.
  - What decides the outcome is unobservable from the robot log: where the part sits in the fingers (board-from-contacts fits σ_pad ≈ 6 mm per event), its tilt in the hand, and the operator's closed-loop wiggle.
- **Oracle diagnostics** (not scores, `all_full_oracle.json`): with the whole in-hand pose registered at the real release, peak force error drops to 0.42 relative (2.6 N, IQR 1.2–7.3), but bacc is only 0.56 (TPR 0.23, TNR 0.88). Small divergences before release compound: the twin reaches the hole a few mm and degrees off and jams, or the part falls over. Scanning pad friction 0.2–1.0, torsion 0–0.002 m and grip 20–60 N seated 2–4 of 48 in every setting (`release_oracle`).
- **Geometry probe** (`geometry_probe_summary.json`; stiff replay at 2e4 N/m, full oracle, ep 1/4, 5/4, 7/19): pre-release force is 27 / 95 / 107 N against real 0.2 / 0.8 / 1.5 N, flat over 0–3 mm board offsets. A stiff replay of the hand with a fixed in-hand pose cannot follow a part that pivoted in the fingers, so hole offsets below 3 mm are invisible to it.
- **Board offset across the pads** (`pad_offset`, 308 windows each, `train_fit.json`): bacc stays at chance at every offset: 0.53 at 0 mm, 0.51/0.51 at ±1, 0.52/0.52 at ±2, 0.52/0.50 at ±3 and 0.51/0.50 at ±5 mm. Seated windows fall from 36 to 28–37, 25–27, 22–24 and 17–20. Replays cannot set a board-placement target until the in-hand pose is known.
- **Where it does work:** a real held jam (ep 2/38, square peg 2) is reproduced with peak force 8.5 vs 8.7 N (`videos/ep002_seg38_*`).

**What is missing** (for the next brain)
1. A per-attempt in-hand pose estimate from the wrist camera at the window start (offset along the pads, tilt, height). Without it, open-loop replay cannot score B2 at these clearances.
2. Contact parameters identified on registered windows against the sensor.
3. A closed-loop operator model if the wiggle-in matters. The orchestrator should treat B2 as **not met**; see the report for proposed validation changes.

**2026-10-24 (second pass, v2), claude-opus-5.5 stand-in: over-jamming in reconstruction mode** (`runs/claude/run-standin-1/replay/`: `params_v2.json`, `train_v2_summaries.json`, `screen_newhand.json`, `screen_oldhand.json`, `free_space_v2.json`, `scan_grasp_height_v2.json`, `b1_after_handfix.json`). All on train: 308 windows, 69 real failures.

What fixed what, in order of effect:
1. **The old oracle placed rotations the pads cannot hold.**
   - In 156/239 real successes the part reached the release frame more than 1 mm off the hole (median 3.7 mm, tilt 4.6°): pre-roll pads undo any rock about the hand x axis.
   - The real hand tilts mostly about the closing axis at release (median 4.7° pivot, 2.2° rock), and failures tilt more (5.6°/3.4° vs 3.9°/1.9°).
   - Replaced by the label-blind registration of §3b: on the stratified subset (69 failures + 69 successes), bacc 0.529 (old oracle) → 0.580 (`inhand`, rotation `full`).
2. **Friction.** The part's friction 0.30 → 0.15 gives 0.580 → 0.645 on the subset. Parts stuck 2–4 mm deep at 6–12° after release, where real ones fell in. 0.10 gives 0.616, 0.20 gives 0.609.
3. **Hand geometry (pads 22.9 mm above O_T_EE).**
   - The `legacy_hand` ablation on all 308 windows, with everything else equal:

     | | legacy hand | corrected hand |
     |---|---|---|
     | bacc | 0.604 | 0.620 |
     | bacc, released windows | 0.507 | 0.571 |
     | peak force rel. error | 0.72 | 0.36 |
     | onset error | 0.43 s | 0.23 s |
     | pre-contact tracking | 1.5 mm | 0.64 mm |
     | TPR | 0.60 | 0.44 |

   - It explains the force and timing gap, not the over-jamming: the corrected hand seats *fewer* parts.
   - Re-identified with it: grasp height 9 mm below the part top (onset zero crossing; 3 mm −0.63 s, 15 mm +0.77 s). With the camera plate top (+30.3 mm) this is consistent with contacts: onset fixes plate top + length below O_T_EE = 48.4 mm.
4. **Nullspace potentials off:** free space median 0.99 → 0.65 mm, p90 4.1 → 1.8 mm (see §1).
5. **No gain:**
   - solimp width 0.5 → 0.1 mm: 0.551 vs 0.529. The soft layer acts only at negative distance (margin 0), so it does not eat clearance.
   - Pad torsion 0.0005 m: 0.522.
   - impratio 1: 0.572.
   - Finger rock/pivot compliance 0.5–8 Nm/rad: 0.52–0.54; it raises TPR, TNR falls more.
   - K 800: 0.623 vs 0.645; K 200: 0.471.
   - Hole chamfers: NIST plate STL holes keep their size through the full 9 mm, so none are modelled.
   - 32-gon holes: the flats sit at the radius, so the polygon is circumscribed and is not a cause.

**v2 results on all 308 train windows** (reconstruction, frozen `params_v2.json`):

| | value |
|---|---|
| TPR | 106/239 = 0.44 (CI 0.38–0.51) |
| TNR | 55/69 = 0.80 (CI 0.68–0.88) |
| bacc | 0.62; released windows 0.57 (46 failures) |
| peak force | 0.36 relative; 2.1 N (IQR 1.0–5.3) |
| onset | 0.23 s |
| dt/2 outcome agreement | 288/308 = 93.5%, below 95% |
| free space | 0.62 mm median, p90 1.69 mm |
| prediction mode (`none`) | bacc 0.51 |

- The negative control **fails**: the permissive twin scores 0.651 (released 0.551), so the drop is −0.03 (released +0.02).
- Before → after for the oracle/reconstruction: TPR 0.23 → 0.44, TNR 0.88 → 0.80.

**What is left** (133 real successes not seated):
- 86 reach k_ref more than 1 mm off the hole axis: the twin's part drifted laterally during the seconds of contact before release.
- 26 have clipped registrations (closing offset beyond ±6 mm: board error across the pads).
- 19 sit over the hole and do not enter.

Seating tracks clearance:

| | round peg 4 | round peg 1 | square peg 1 |
|---|---|---|---|
| clearance | 0.25 mm | 0.05 mm | 0.055 mm |
| seated | 25/29 | 8/31 | 7/30 |

Next candidates:
- A declared, bounded lateral setpoint correction near contact (Google allows commands to change near contact). It must be declared before gate.
- Per-part contact softness identified against the sensor on registered windows.
- **2026-10-24** (claude:b2-replay): Effective clearance and what the outcome test can see (train, 308 windows). Widening every hole 0-0.5 mm per side moves the twin along one TPR/TNR line: bacc 0.62-0.66, no optimum. Per-part fits from recording-level 5-fold CV were unstable (round peg 4: 0.5/0/0.05/0.05/0.05 mm; held-out 0.620). One global +0.2 mm held out at 0.645 (frozen as params_v3). Above ~0.1 mm per side this exceeds plausible drilled-hole oversize, so read it as unmodelled self-alignment, not geometry. Split failures label-blind before tuning on them: only 14/69 show a physical-jam signature (|dF| > 5 N for >= 0.3 s, < 2 mm descent), and the twin jams all 14. The other 55 are aborts or misses (23 never released, 25 under 0.3 s of force), which no contact physics can separate. That is why the permissive twin scores as well. dt/2 flips (20/308, mostly keys) are chaotic borderline cases: 0.5 vs 0.25 ms agree on only 14/20 of them. Evidence: `runs/claude/run-standin-1/replay/clearance_cv.json`, `runs/claude/run-standin-1/replay/failure_split.json`, `runs/claude/run-standin-1/replay/dt_study.json`.
- **2026-10-25** (claude:b2-replay): In-hand pose from force (3c/3c-reg), train, stratified 69 failures + 69 successes. Force pins the grasp height (posterior/prior SD 0.50) and partly the closing offset (0.76) and slide (0.91). It does not pin rock, pivot, friction or hole widening (0.99-1.01). The tempered likelihood needs beta ~0.27 for ESS N/10 (the raw one collapses to ESS 1): systematic rig-vs-real force error. Distribution checks fail in every mode: population prior seat rate 0.10 (err 0.67, KS 0.38); registered proposal 0.20 (err 0.58, KS 0.33); v3 CPU arm with point registration 0.54 (err 0.24, KS 0.31); real 0.78. In each, the permissive twin is closer to reality (negative control -0.02 / -0.02 / -0.10). Per attempt (track), P(seat) still ranks attempts: CV bacc 0.63 prediction, 0.68 registered reconstruction. Before using a reduced model in a filter, match its seat rate to the full model on the same windows: the rig seated 8-15% of prior worlds against ~50% for the arm replay. Next: fix the rig (full 6x6 Lambda, or menagerie armature 0.195 inflating Lambda to 34/11/7 kg), then rerun. Evidence: `runs/claude/run-standin-1/replay/params_v4.json`, `runs/claude/run-standin-1/replay/pf/v4_stratified/summary.json`, `runs/claude/run-standin-1/replay/pf/v4reg_stratified/summary.json`, `runs/claude/run-standin-1/replay/inhand_prior.json`.
- **2026-10-25** (claude:b2-replay): KNOWN ISSUE, unfixed: the GPU Cartesian impedance rig (twin/rig.py) over-jams compared with the CPU full-arm replay (twin/replay.py). With the same registered in-hand poses, it seats 8-15% of prior worlds (3c 0.08, 3c-reg 0.15) against ~50% for the arm (v3 point registration 0.54 on 308 windows). Its peak forces run up to 2x the arm's (ep 1/4: 10.2 vs 5.4 N). Diagnosis, ranked by suspicion: (1) Lambda is taken diagonal in the TCP frame. The menagerie FR3 joint armature (0.195 kg m^2 on joints 1-4) makes Lambda 34/11/7 kg (ep 1 start), so the rig is a heavy, sluggish mass. The arm replay keeps the full coupled Lambda and nullspace, which can let the part self-align. Try the full 6x6 Lambda (a free body with spatial inertia), or check the real FR3 reflected inertia against menagerie. (2) Off-diagonal D is dropped. (3) Finger rock hinges (2 Nm/rad) with per-world rest angles: the arm replay has none. (4) Rotational kp = K_r/2 and the hinge order (small-angle) differ from the quaternion law at larger tilts. Fix test: run run_rig_cpu and run_replay on the same 50+ windows with the same registered pose (registration inhand, rotation full, finger_rock_k 0). Change one item at a time until rig and arm seat rates agree within ~0.05; only then rerun twin.pf_batch. Evidence: `twin/rig.py`, `twin/replay.py`, `runs/claude/run-standin-1/replay/pf/v4reg_stratified/summary.json`, `runs/claude/run-standin-1/replay/clearance_grid.json`.

## Sources

Ids refer to `docs/knowledge/references.md`.
- [franka-ros-2024] franka_ros `cartesian_impedance_example_controller.cpp` and `compliance_param.cfg`, local copy `data/refs/franka-ros-2024-*`.
- REASSEMBLE controller, teleop and F/T code, `vendors/reassemble/catkin_reas/src/` (read-only): `cartesian_impedance_controller_damping_ratio.cpp`, `nullspace_optim.h`, `compliance_param.cfg`, `teleop_franka.py`, `force_2_base.py`.
- [dexforce-2025] Chen et al., DexForce, RA-L 2025, arXiv:2501.10356: force-informed targets x_f = x_o + k_f·f.
- [mujoco-docs-2026] MuJoCo modelling docs: elliptic cones, impratio, Newton at tol 1e-10 against slip; timeconst ≥ 2 dt.
- [mujoco-1124-2023] MuJoCo maintainers: contact parameters only by system identification against a force sensor.
- [qbit-2025] QBIT, arXiv:2503.07479: randomized solref/solimp/friction covering real insertion force.
- [touching-gnn-2025] arXiv:2509.12151: an identified MuJoCo model kept ~3.5 N one-step F/T error; motion is predicted better than force.
- [yoon-2022] arXiv:2202.13098: contact simulation fitted to measured wrench (CMA-ES trains their contact networks, not engine parameters).
- [pnd-2026] arXiv:2610.07745: 5 mm fixture noise drops a FORGE insertion policy from 97–99% to 32–60% in sim.
