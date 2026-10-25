# reward-from-video: REASSEMBLE peg-twin reference

Evidence: `runs/argon/peg_reward/`. The revision was still running at 15:30, so update this file when `results/` holds a revised eval.

## Inputs Argon got (`specs/peg__prompt.md`)

- Two real successful insertions, one round peg and one square peg, each from a scene camera and the wrist camera.
- The recorded-signals plot around clip A.
- The practice-twin state API at 30 Hz: part pose, `tip_pos`, `tip_axis`, velocities and speed, target hole, gripper.
- The stage list to use: approach → contact → align → insert → seat → release, plus `done`.

Episodes start with the peg already held near and above its hole.

## Gate

- **Episodes:** twin episodes from the pilot arms (`run-argon-pilot-arm1-1`, and the stand-in pilot runs `run-pilot-arm1-2` and `run-pilot-armN-2`), replayed by `scripts/replay.py`.
- **Truth:** the twin's frozen evaluator.
- **Scoring:** `scripts/evaluate.py`.
- **Near-miss families:** depth short by 1.5 / 3 / 6 mm, tilt 4 / 6 / 10°, not released, wrong hole.

## Gate numbers (`results/gate_v0.summary.txt`, `results/gate_v1.summary.txt`)

59 pilot episodes (49 successes), 79,974 states, 392 near-misses.

| Version | P | R | End agreement | Near-miss reject | Progress ρ median (min) |
|---|---|---|---|---|---|
| v0, first write | 0.995 | 1.000 | 58/59 | 0.625 | |
| v1, after one revision (frozen) | 1.000 | **0.882** | 58/59 | 1.000 | 0.923 (0.409) |

**Near-miss families:**
- v0 accepted every depth-short-1.5 mm, depth-short-3 mm and tilt-4° near-miss (49/49 each).
- v1 accepts none.

**The cost of the fix is recall.** v1 misses 817 true timesteps. Its progress ends at 0.95 on both the lowest success and the highest failure.

**Why** (the coordinator's note on the run, not re-derived here): the revision over-tightened release so that it needs a wide-open gripper, and 2 successes stall at 0.95.

**Lesson:** gate on recall as well as on near-miss rejection. A revision that closes the near-miss leak can open a recall gap.

v0 cost $0.095, 50 s and 30k tokens, 21k of them video.

## Verification turn (`specs/peg__feedback.md`)

The turn showed the 2 episodes where `success` disagreed with the truth: early by 5 states and by 3 states. It did not show the accepted near-miss families. Under the core rule, the turn's near-misses come from its own 2 episodes, so the 1.5–3 mm and 4° acceptances become visible without exposing the gate set.
