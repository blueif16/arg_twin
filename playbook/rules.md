# Rules

Every agent follows these. When a rule blocks you, stop and report it; don't work around it.

## Integrity of results

- **The evaluator is frozen.**
  - Success checks, thresholds and held-out starts are fixed before practice begins.
  - Actors never edit them. A change needs a new run.
- **Success is read from simulator state** by evaluator code: insertion depth, part pose relative to its hole, contact. A model's opinion of a frame is never the score.
- **Keep splits disjoint.** Exam recordings are never read, replayed or tuned on before the final score.
- **Actors and watchers are separate agents.** Nobody grades their own work.
- **Observers and auditors never pass sim truth.** Anything that feeds an actor (a mid-episode nudge, a between-trial reflection, an audit verdict a later trial reads) is built only from what the actor itself received, plus outcome reports the split allows and the frozen reward's own outputs. Object poses, raw state, the benchmark predicate's value and episode-record truth never reach it; a leak check over every observer input is part of the run's integrity (e.g. `L.observer_leaks`).
- **Cost is part of the result.** Report tokens, brain calls and wall time next to every success number.

## No sim cheating

These rules follow Google's twin prompts.

- **Objects move only through contact** computed by `mj_step`.
- **After t = 0, none of the following:**
  - changing model parameters
  - external forces on objects
  - welds or equality constraints added to fake a grasp
  - resetting or rolling back qpos/qvel
  - disabling collisions
  - reporting sensor values that don't come from the solver
- **What may be tuned:** only what is set before t = 0, meaning geometry, mass, friction, contact parameters, initial poses and controller gains.
- **Results must be numerically sound.** A result that only holds at one timestep isn't trusted. Re-check key results with the timestep halved.

## Capability boundary

Integrity holds by construction, not by promise (skill `sim-integrity`).

- **Actors act only through the env API:** `env_*` tools for brains, `envapi` for skill programs. Commands are absolute TCP targets with a speed and a force ceiling enforced in env code, the gripper, and wait. Observations are robot state, F/T, gripper width and time. Nobody else touches MjModel/MjData.
- **Skill code passes the static checker (`skill_check`) and runs in a separate, sandboxed process** that talks JSON to the env (`skill_run`). In-process Python restriction is not a boundary.
- **Monitor-flagged successes don't count.** A success counts only with no invalidating monitor flag and a dt/2 rerun that agrees.
- **Raw simulator access is for red-team scripts only:** `run_actor {privileged: true}`, red-team starts, logged, never counted.
- **Held-out starts are attempted once per run by each actor** (start, cousin, actor; `env/heldout_attempts.jsonl`), so attempt 1 and attempt N can be paired on the same starts. Their outcomes go to the evaluator log, not back to the actor.

## Grounding

- **Real recordings are the reference for physics.** A strategy faster, harder or different from anything in the real data gets flagged as outside the validated range.
- **Every claim in a skill's Lessons cites its evidence:** a run log, a file or a source.

## Interfaces

- **Tool names, arguments and MuJoCo site names are an interface.** Changing them needs a version bump and a check that existing skills still pass.
- **Don't edit `vendors/`.** It's read-only reference code. Don't run code from downloaded data or vendors.

## Records

- **Every tool call goes to `runs/calls/`,** and every loop round to `runs/loop/`. The integrity audit (`integrity_audit`) reads these logs for the G checks; anything done outside a tool call (shell reads, direct imports) is invisible to it.
- **No secrets** in logs, prompts or skill files.
