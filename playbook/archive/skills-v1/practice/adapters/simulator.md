# practice: simulator input (a benchmark simulator with exact truth)

The twin is the simulator's own scene and success is the benchmark's goal predicate on exact sim state, so the task loop is measured with no twin error in the way. Source details are in `references/libero.md`.

## The boundary (same shape as `twin/env.py`)

- **One process owns the simulator.** Actors send commands and receive observations through a narrow op set: task, observe, move, done, give up, status.
- **Never sent to an actor or an observer:** object poses, sim state, the predicate or its value. Scored splits withhold the outcome as well; only practice splits may reveal it.
- **Frozen evaluator.** The predicate code plus every goal and init file are hashed, and the server refuses to run on a mismatch.
- **What counts as a success:** an exact open-loop replay and no invalidating flag (predicate true at reset, non-finite physics). Penetration and free-body speed are flagged.
- **Scored episodes run once per (run, actor),** and every attempt is logged.
- **The budget is frozen per run.** The server enforces it, and the brain runner asks before every model call.
- **The runner's accounting tool** (`libero_charge`) is not the brain's. Leave it out of a brain's tool allowlist.

## Attempt 1: the published protocol, unchanged

When the benchmark has a published LLM-as-policy number, attempt 1 reproduces that protocol exactly: prompt, tools, history, image horizon, limits and effort. It is the anchor every later number is paired against. Record each deviation (API wire, spend cap) next to the result. Report a protocol defect, such as mirrored views or tool text that contradicts the embodiment, but don't fix it inside the anchor. Fix it for attempt N and say so. Declare any smaller scored subset before its first episode.

## Practice: same-start retries with observer feedback

Design: `harness/brains/trials.py`; simulator runner: `runs/argon/libero_chain/chain.py`.

1. **The reward first.** The reward is frozen by hash before any trial (skill `reward-from-video`), and the server refuses on a mismatch. The benchmark predicate is recorded beside it, for validation only.
2. **Up to 4 trials per start.** Each trial is a short-budget run: Argon at medium, chained turns, medium-resolution images.
3. **Between trials, Argon reflects** on the failed trial's own trace, in at most 6 lines: STATUS / CAUSE / FIX / GUIDE / KEEP. Earlier notes on that start head the next trial's task.
4. **During a trial, a fast observer** (`gemini-3.8-flash`) wakes on a progress stall and injects 1–2 lines before the brain's next turn.
5. **Information honesty** (rules.md). Observers see what the brain received plus the frozen reward's progress, stage and success. A leak check runs over every observer input and output.
6. **The guide** that observers quote is method knowledge only. Task findings stay in the run.
7. **Score success@k by the benchmark predicate** (held at any step) and by the reward, separately. Each disagreement goes to the auditor.

## Method lessons that carry across sources

- **Measure the effector against the target before committing.** On pegs this means hole minus tip in one image, with the tip on the plate. On pick-and-place, lift 3–5 cm and confirm in both cameras that the object rose.
- **Treat one visual read as a probe start, not the aim,** and read the rim or edge, not the dark area.
- **Localise by contact where contact exists:** pressure centre and torque-sign flips.
- **Pull a stuck object gently.** After two equal failures, change the approach.
- **Test yaw at a stop.**
- **Claim done only on named evidence.**
