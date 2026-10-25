# ASPIRE: agentic skill discovery with an execution engine (2026)

`[aspire-2026]` · Lu, Wu, Kou et al. (with Mandlekar, Shi, Goldberg, Zhu, Fan; NVIDIA GEAR and others), arXiv 2607.00272 · status: verified · local: `data/refs/aspire-2026.html`

Read 2026-10-24 from the arXiv HTML (v2). Every number is theirs. Lines starting "For us" are our reading.

## Setup

- Coordinator plus actor coding agents (main runs: Claude Code with Opus 4.6) write code-as-policy programs against a fixed perception, geometry and motion API on LIBERO-Pro, Robosuite and BEHAVIOR-1K.
- **Execution engine:** per-primitive traces (API, inputs, outputs, return status, keyframes just before and after each call, overlays, grasp candidates, poses, planner results). No full video goes to the actor.
- **Skill library:** validated repairs, stored as failure signature + when-to-apply + repair strategy + optional code sketch. Admission: the actor proposes; a fresh verifier actor reproduces the source task on a subset of seeds; the coordinator checks API compliance.
- **Evolutionary search** over programs: K candidates per round, conditioned on the best previous programs and their failure traces. The coordinator also orders tasks as a curriculum and reruns early low scorers with the final library.

## Results

- Ablation (LIBERO-Pro macro average): execution engine 14% → 62%; adding evolutionary search → 72%.
- Library built on LIBERO-90 transfers zero-shot to unseen LIBERO-Pro Long tasks: 31% vs 4% for prior coding agents.
- Gains over the strongest baseline: up to 77% on LIBERO-Pro Object, Robosuite bimanual handover 20% → 92%. With Opus 5.5: 99% across all 80 LIBERO-Pro tasks.
- **Sim skills on a real bimanual YAM** (Codex with GPT-5.5 xhigh, different embodiment and API), tokens to first successful program:

  | Task | Total tokens w/o → w/ skill | Success w/o → w/ |
  |---|---|---|
  | Put bowl on plate | 8.65M → 5.11M | 20/20 → 20/20 |
  | Lift soda can | 61.94M → 6.58M | 13/20 → 19/20 |
  | Open/push drawer | 334.92M → 81.67M | 0/20 → 11/20 |

- **Online on the real robot** (GPT-6 Astra, Rubik's cube, 3 episodes): time per face turn 16 min 58 s → 7 min 48 s → 3 min 05 s; tokens per turn 11.27M → 2.01M; cost per turn $16.85 → $2.89. A human still scrambles, resets and gives task guidance.

## For us

- The engine's per-call trace is the single biggest lever they measured (+48 pp). Our harness call log should record, per primitive, inputs, outputs, status and the frames around the call, and hand that to Argon.
- Their admission rule (a fresh verifier reproduces the skill) is a ready-made promotion gate for our compile step.
- Skills here are mostly repair knowledge given as context, not code that runs without the model. Their token savings are per first success, not per routine execution.

## Limitations (verified in §5, 2026-10-24)

- The authors say it "is not yet a fully autonomous real-world lifelong learner". A human gave the initial guidance, scrambled the cube and reset the scene.
- "Unlike simulation, where success checking and scene resets are cheap and programmatic, real-world deployment still requires robust success detection, safe reset, safety monitoring, and calibration maintenance."
- For us: in the twin, success detection and resets are exact and cheap. Monitoring for stalls, exploits and implausible physics is the part that still needs a dedicated watcher.
