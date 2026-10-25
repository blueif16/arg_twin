---
name: replay-validate
description: Stage B2/F on any data source. Replay real attempts in the twin from the commands the real robot received (rebuilt from the log, never the logged pose) and measure whether the twin reproduces the outcome (seat vs jam, place vs drop), the contact force and the contact onset, with a free-space controller check, a negative control, dt/2 and a geometry probe; answer physics questions raised in practice (F) the same way. Pick the adapter by what you were given; source facts are in references/.
stage: build
steps: [B2, F]
---
## Goal

For each part or task family whose geometry passes B1: replays of held-out real attempts in the twin whose outcomes, force and onset match the real ones at distribution level, while a deliberately permissive twin does not. Two modes:
- **reconstruction** (scored): everything replays as logged except what the log lacks (e.g. the in-hand pose), registered per attempt label-blind before scoring;
- **prediction** (tracked): the same with `registration: none`.

Done when the source's B2 checks in `playbook/validation.yaml` pass as measured by a watcher (REASSEMBLE `B2.*`; SO-101 T2 checks with `step: B2`), each F question has an answer with its evidence, and Lessons holds the run.

## Choose your path

| You were given | What replays | Truth | Adapter | Source facts |
|---|---|---|---|---|
| **Robot logs + cameras** (commands or TCP, gripper, F/T) | the rebuilt command through the controller that ran | the recording's outcome labels, force and onset | `adapters/robot-logs.md` | `references/reassemble.md` (impedance Franka, F/T); `references/so101.md` (leader commands, no F/T) |
| **Simulator with a state API** | the recorded actions, open loop | the recorded states: exact replay to 1e-6 | `adapters/simulator.md` | `references/libero.md` |
| **Video only** | nothing commanded; only a retargeted motion | none: physics checks on the motion; report provisional | `adapters/video-only.md` | none yet |
| **Prior tested runs** | the frozen replay params version | the last scored distribution | `adapters/prior-runs.md` | the source's reference |

## Inputs

- Held-out attempts from the source's frozen splits (gate split for scoring; never exam).
- The twin from build-twin (B1 passed), the controller documentation, the log streams.
- Labels only for scoring, never for registration.

## Tools

`replay_*` and the free-space check in the harness (`references/reassemble.md` "Tools"); MuJoCo Warp for parallel worlds (`build-twin/references/mujoco.md`); `score_*` scripts named in the references.
- **Registry tools carried from v1** (`--skills` allowlists read this section; usage in references/): `make_comparison`, `publish_media`, `record_event`, `record_check`, `replay_attempt`, `replay_batch`, `replay_controller_check`, `replay_geometry_probe`.

## Approach

Core method (it holds wherever commands exist):
1. **Reproduce the controller that ran, not a generic one,** and check it in free space first: tracking error on approach windows against the tolerance (`references/reassemble.md` §1, free-space check on 48 windows).
2. **Rebuild the command from the log;** never replay the logged pose, which already contains the contact outcome (`references/reassemble.md` §2).
3. **Separate the error sources before identifying anything:** geometry, in-hand pose, contact parameters, controller. Identify only what the data can identify; K could not be identified from REASSEMBLE (`references/reassemble.md` Lessons).
4. **Distribution-level validation with a negative control.** Per-attempt accuracy plateaus on operator aborts that no physics produces; a permissive twin must fail the same checks the real twin passes (`references/reconstruct-scene-v1.md` §7 "Operator aborts").
5. **dt/2 and a geometry probe on every claim:** default contact parameters let a part slide through a rim overlap at every timestep, so dt/2 alone can pass a wrong twin (`references/reconstruct-scene-v1.md` §7).
6. **Label-blind registration only;** no resampling or state copying between worlds after t = 0 (rules.md).

## Per-model notes

- **Claude (stand-in)** ran every B2 round so far (`references/reassemble.md` Lessons, claude:b2-replay); it reads plots as PNG.
- **Argon** has not run B2. Give it the replay tools and the numbers as tool results; never ask for force or onset numbers in JSON at low/medium thinking (`runs/argon/probes/api/findings.md` §1).

## Checks

The source's B2 checks (distribution match of seat rate and force spread, peak force error, onset error, permissive-twin control, dt/2 agreement), recorded by a watcher. Per-attempt outcome accuracy and open-loop prediction are reported, not gated (validation v8, v10; `playbook/orchestrator.md`).

## Lessons

Dated entries from every source, one line each; the full entry, with its numbers, is in the named reference. New lessons go here (`append_lesson`) with the source named; detail goes to `references/<source>.md`.

- **2026-10-24** (claude): K cannot be identified from these data. Evidence: `runs/calls/20261024.jsonl`, `runs/claude/run-standin-1`; full entry in `references/reassemble.md` (v1 `replay-validate`).
- **2026-10-24** (second pass): claude-opus-5.5 stand-in: over-jamming in reconstruction mode. All on train: 308 windows, 69 real failures. Evidence: `runs/claude/run-standin-1`; full entry in `references/reassemble.md` (v1 `replay-validate`).
- **2026-10-24** (claude:b2-replay): Effective clearance and what the outcome test can see. Widening every hole 0-0.5 mm per side moves the twin along one TPR/TNR line: bacc 0.62-0.66, no optimum. Per-part…. Evidence: `runs/claude/run-standin-1`; full entry in `references/reassemble.md` (v1 `replay-validate`).
- **2026-10-25** (claude:b2-replay): In-hand pose from force, train, stratified 69 failures + 69 successes. Force pins the grasp height and partly the closing offset and slide. It does not pin rock, pivot,…. Evidence: `runs/claude/run-standin-1`; full entry in `references/reassemble.md` (v1 `replay-validate`).
- **2026-10-25** (claude:b2-replay): KNOWN ISSUE, unfixed: the GPU Cartesian impedance rig over-jams compared with the CPU full-arm replay. With the same registered in-hand poses, it seats 8-15% of prior wo…. Evidence: `runs/claude/run-standin-1`; full entry in `references/reassemble.md` (v1 `replay-validate`).
- **2026-10-25** (claude:b2-5b): v5b, formulation: identify grasp compliance from the log, label-blind, then CV by replaying each fold's held-out episodes with params identified on the other folds. Meth…. Evidence: `runs/claude/run-standin-1`; full entry in `references/reassemble.md` (v1 `replay-validate`).
- **2026-10-25** (claude:b2-5a): v5a, formulation: distribution calibration of global part-board contact parameters, train only. Method: screen one knob at a time on all 308 windows, then pick among the…. Evidence: `runs/claude/run-standin-1`; full entry in `references/reassemble.md` (v1 `replay-validate`).
- **2026-10-25** (claude:b2-6): v6, a consistent grasp, train only. Method: Test the grasp before tuning it. Run an offline static ramp: torque about the closing axis at 0.5 Nm/s, read the slip torque,…. Evidence: `runs/claude/run-standin-1`; full entry in `references/reassemble.md` (v1 `replay-validate`).

## Sources

- v1 skill text moved unchanged to `references/reassemble.md` (archived at `playbook/archive/skills-v1/replay-validate/`).
- docs/knowledge/contact-physics-and-sysid.md, real2sim-twins.md, evaluation-and-statistics.md.
