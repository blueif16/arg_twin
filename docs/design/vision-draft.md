# Vision (v1): two self-optimization loops

> Status: v1, 2026-10-25. Parts marked *agreed* were decided by the user. Everything else is a proposal and changes as results come in. v0 is in `archive/vision-draft-v0.md`.

## The claim

Argon learns a **general skill set** that turns footage into a working simulation and then improves at tasks inside it. With practice, work moves out of slow reasoning and into fast routines, so each attempt is more reliable and cheaper. The skills should handle varied scenes and tasks. A random egocentric or YouTube video should be characterized and twinned as well as a robot log is.

- **Datasets are benchmarks, not targets** (*agreed*). REASSEMBLE and SO-101 measure us; we don't tune the system to them.
- **The demo has no named end user** (*agreed*). It presents the flow, the measured performance boost and the self-learning framework. Everything is labelled sim-only until a real arm confirms it (rule D5).

## Two loops, one framework (*agreed*)

Both loops run on the same machinery:
- a fixed orchestrator;
- skills that state a goal, tools, rules and checks, and collect dated lessons;
- deterministic tools;
- a frozen evaluator;
- a watcher that never acts;
- a compile step that turns proven decisions into library code.

The loops differ in their objective and their playground.

### Loop 1: building the twin

**Objective.** A twin is better when it explains observations it wasn't fitted on:

1. **Appearance and geometry:** render vs real on held-out frames and views (keypoint px, mask IoU, depth error).
2. **Behaviour:** replaying the observed motion (robot commands, or retargeted hand motion) gives the observed results: object trajectories, contact events, outcomes, and forces where they were logged.
3. **A wrong-twin control:** a deliberately wrong twin (shifted, scaled, loosened) must score clearly worse. This stops a permissive twin from winning (the B2 lesson, 2026-10-25).
4. **Physics gates:** no penetration, stable at rest, results hold at half the timestep.
5. **No leakage:** fit only on data from before the event being predicted, and score on held-out data. The 2026-10-25 audit found the release-frame registration inflated the peg seat rate by about 0.10.

**Calibration.** On random footage the score is label-free, and published evidence says label-free checks only **reject, never certify** (`docs/knowledge/video-to-twin.md`):
- Reprojection rankings flip with the reference.
- Better PSNR came with worse geometry.
- Stability pass rates didn't rank pose accuracy.

So accuracy claims come only from benchmarks with ground truth:
- HOT3D-Clips (20 clips, about 4 GB);
- the ADT sample (500 MB) plus 3–5 sequences;
- LIBERO demo videos against their exact scene files;
- REASSEMBLE and SO-101 robot logs.

On random video the checks act as rejection gates, and the behaviour replay is the main discriminator.

**Honest limits** (same source):
- Metric scale is 9–53% off without an anchor.
- Model-guessed mass is off by about 1.3× on average, up to 4×.
- Contact estimated from video reaches F1 0.27.
- One internet-video pipeline failed 42% of its steps.

A twin that matches the video can still be physically wrong.

**Learning measure.** Over a stream of different videos, error and cost per new video fall as the skill's lessons and its perception tools improve.

**Scaffold.**
- The `reconstruct-scene` skill;
- generic perception tools (masks, depth, camera, hands);
- scene-as-code;
- render-and-compare and replay tools.

Most of this exists but is coupled to REASSEMBLE, so the intake has to become source-agnostic.

**Benchmarks.**
- With robot logs: REASSEMBLE, SO-101.
- With ground truth: to be chosen.
- Video-only, label-free: random videos.

### Loop 2: doing the task

**Objective.** Success on held-out starts within a frozen budget, plus cost per success (tokens, brain calls, time) and the brain's share of control. Integrity monitors and a real-data envelope are hard constraints.

**Learning measure.**
- Attempt 1 (the published zero-shot protocol) vs attempt N (practice, task tools, compiled routines) on the same held-out starts, with paired tests.
- Then transfer: new tasks start cheaper.

**Scaffold.**
- The practice env behind a capability boundary;
- frozen evaluator and monitors;
- task tools;
- the compile skill and library with its promotion gate.

**Playground** (*agreed*: LIBERO first).

| Playground | Why | Role |
|---|---|---|
| **LIBERO** (MuJoCo, 130 language-specified tasks) | The only published Argon-as-policy number: 43% on LIBERO-10 at high effort, $7.84 per episode, 41/100 limit hits [llm-libero-2026]. Skill libraries have shown gains there (ASPIRE 31% vs 4% on LIBERO-Pro) [aspire-2026]. | Main playground: LIBERO-90 for practice, LIBERO-10 held-out |
| RoboCasa or MetaWorld | Scene and task variety. LeRobot already ships the wrappers. | Second playground |
| REASSEMBLE peg twin | Contact-rich, built from footage | The case that joins both loops |

Attempt 1 must follow the published protocol: end-effector pose calls, the 80-call cap, no privileged sim state. Otherwise the comparison is void; for example, privileged-state agents reach 84.9% [faea-2026].

### How the loops connect

Loop 1 produces playgrounds for Loop 2. When a Loop 2 failure hinges on physics the twin never checked, it becomes a request back to Loop 1 (step F). The demo shows both learning curves.

## Reading footage (stage A, shared by both loops)

Long-video understanding underpins both loops: zooming out and in, actions, settings and outcomes.
- **Today's skill** is signals-first. It passes on robot logs: segment F1 0.992, success bacc 0.947 on 9 blind gate episodes.
- **What's untested:** video-only reading, which is what arbitrary footage needs.
- **A strategy study is running:** single pass vs hierarchical agentic zoom vs signals + video, on REASSEMBLE and one non-REASSEMBLE labelled dataset. It sets the "read any video" method.

## What we prove

1. **Loop 2 boost:** Argon plus the self-learning loop beats Argon alone under the published protocol on LIBERO-10 held-out, at lower cost per success.
2. **Loop 1 learning:** twin error, calibrated against ground truth, falls across diverse videos. Random videos get label-free scores and plausibility checks.
3. **Transfer:** skills and routines learned on some tasks or scenes make new ones cheaper from their first attempt.

## Open items

- **LIBERO feasibility:** install, boundary, and a 10-episode reproduction of attempt 1 (running).
- **Ground-truth benchmarks and the perception stack for Loop 1:** research running.
- **The REASSEMBLE B2 leak fix:** before-contact registration, and a leave-one-out board fit. Needed for honesty, not for the score.
- The Claude pilot on pegs (running) is the first end-to-end measurement of attempt 1 vs N.
