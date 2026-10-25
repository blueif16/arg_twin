---
name: compile-skill
description: Stage E. Turn what the brain worked out in practice into a routine that runs with zero brain calls (System 2 → System 1). Covers when a decision pattern is ready to compile, how to extract it from the practice ledger and command logs into a parameterized skill program, where the brain stays (hand-back codes), the promotion gate (routine success on held-out starts, cousins, envelope, G monitors, frozen files), demotion, and the versioned per-brain library. Use it after practice shows repeated successes, before attempt N, and whenever a routine fails later.
stage: compile
steps: [E]
---
## Goal

A versioned library of routines, each a parameterized skill program that runs in the sandbox against the env API with **zero brain calls**, promoted only through the frozen gate:

- **Compiled means it works alone:** `E.routine_success` on frozen held-out starts with no brain call.
- **Robust, not tuned to one twin:** `E.cousins_success` on the frozen (held-out start, cousin) pairs.
- **Inside the real regime:** `E.envelope_violations` = 0 (force, approach speed and duration within the real train attempts of the family, and the family's B2 passes).
- **Honest:** G monitors clean (no counted success with an invalidating flag) and the files the gate ran are the files the library holds.
- **The efficiency claim follows:** brain calls, tokens and time per success fall from attempt 1 to attempt N, because decisions moved into code. RAX measured 254k → 4k output tokens per episode once compiled [rax-2026].

A routine is not a pre-written plan for the whole task. It owns the motion phases it has seen succeed and hands control back at the decision points it cannot handle [urai-2026].

## Inputs

- A practice run of the **same brain**: `runs/<brain>/<run_id>/env/episodes.jsonl` with each episode's command log and 30 Hz trajectory, the ledger (`practice/ledger.json`: decisions and images per episode) and the run's task findings. Another brain's runs are refused by the tools (orchestrator.md: task findings stay with their brain).
- The env API a routine may use (`envapi`, `twin/sandbox.py`): `task`, `observe`, `move` (absolute TCP target, speed, force ceiling, guarded stop), `gripper`, `wait`, and `finish` when standalone. Nothing else: no images, no simulator, no files outside its workspace.
- The task tools' conventions (stage C, `task/v1`, `runs/<brain>/<run_id>/task_tools/manifest.json`), when they can run in the sandbox (Lessons: `twin/task_tools.py` does not pass `skill_check` yet).
- The frozen sets of the run: practice and held-out starts, cousins, the real envelope (`twin/envelope.py`), `env/budget.json`.
- Targets: `validation.yaml` E and G checks, and the sample-size header (one-sided 95% Clopper–Pearson lower bound ≥ 0.9 takes 29 / 46 starts with 0 / 1 failures).

## Tools

| Tool | Use |
|---|---|
| `library_extract` | The decision structure of a practice run's episodes from the command log (hover, touch, search, probe_in, release, retreat), the brain's parameters per phase, and `ready` |
| `skill_check` | Static check of a draft before adding it |
| `skill_run` | Run a draft on chosen practice starts while writing it (dev starts; zero brain calls) |
| `library_add` | Add a version: static check, frozen by hash, sources and structure recorded, sandbox smoke test; status candidate |
| `library_list` | Versions, status, gate results, frozen check; `usage: true` gives brain share per brain episode |
| `routine_run` | Inside a live brain episode: run a routine with zero brain calls until it hands back a code |
| `promotion_gate` | `mode: pilot` (anyone; practice starts and practice × cousin pairs; marks piloted). `mode: scored` (watchers only; held-out starts and frozen cousin pairs; promotes or rejects). `mode: review` (demotes or marks stale) |
| record_event | Notes, with `stage: E` |

Code: `harness/tools/compile.py` (tools, the sandbox pump for invoked routines, the gate), `twin/library.py` (library storage, decision-structure extraction, usage). The first routine: `runs/claude/run-compile-pilot-1/library/peg_insert_guarded/`.

## Approach

### 1. When a decision pattern is ready to compile

Compile when **repeated successes share one decision structure** and every decision in it can be made from what the robot measures:

- `library_extract` reads each episode's commands into phases and folds them into a canonical structure. Everything between the first touch and the last entry is one `search`; a push continues the entry. Ready = at least 3 counted successes with the same structure (the practice watcher's `near_promotion` wake fires at 3 in a row on one part).
- For every decision the brain took inside that structure, name what it was decided from. Decisions from numbers the env returns (stop reason, depth from TCP travel, wrench, gripper width) compile. Decisions from images do not: a sandboxed routine has no camera. Either replace them with a contact procedure and measure what that costs, or leave them as a hand-back.
- Compile motion phases first and keep branch decisions with the brain until held-out starts show the routine's own checks suffice. A program written in advance reached 24% where agents deciding after each call reached 56% with the same tools [urai-2026]. Distilled skills can also lose cues the raw traces held [skillevolbench-2026], so store the traces with the routine.

### 2. How to extract it into a routine

1. **Phases and parameters from the trace.** Use the phase order and the brain's parameters from `library_extract` (speeds, force ceilings, stop forces, lift heights, shift sizes) as the starting values. Change one only for a stated reason, such as the real envelope, a method lesson, or a measured failure.
2. **Write one checker-clean `skill.py`** against `envapi`. The checker allows no imports of sibling files, so a routine is one file. Put every number in `params.json`: `defaults`, `per_part` overrides (task findings such as where flat plate is), and `tunable`, the keys a brain may override.
3. **Measure what perception gave the brain.** A light touch gives the surface height, so depth becomes TCP travel past it. The line of action of the wrench gives where the part touched, which shows the rim side. A probe pattern sized by the capture width (chamfer + clearance) covers the rest.
4. **Each phase ends on a check the routine can read:** the stop reason, depth past the touch, gripper width. A routine never claims success on a hope: false claims are counted.
5. **Two modes, one program.** Standalone (no `invoke.json`) finishes the episode itself; this is what the gate runs. Invoked from a brain episode (`routine_run`), it never finishes. It writes `result.json` and returns its code.
6. **Develop on named dev starts only** (`skill_run {start_ids}` on practice starts), and list them in `library_add {dev_starts}` so the pilot excludes them. Never use held-out starts while writing.
7. **Add it** with `library_add`. The smoke test proves it boots in the sandbox, reads its params and speaks the protocol.

### 3. Where the brain stays: hand-back

A routine returns control with a code whenever it meets something it was not compiled for. This follows RAX's exception path [rax-2026] and URAI's decide-after-each-call [urai-2026]. It never guesses past one of these:

| Code | Meaning | What the brain typically does |
|---|---|---|
| OK | seated (by its own depth estimate) and released | confirm, `env_finish` with a claim |
| NOT_FOUND | pattern used up: the hole is outside the search area | look (`env_observe_image`), pass `center_mm`, re-invoke or act |
| JAMMED | stalled short of the seat after wiggles | decide: lift and retry, re-aim, act |
| ENVELOPE_TIME | another probe would leave the real duration envelope | decide whether to continue outside it (it will be flagged) |
| NO_CONTACT, UNEXPECTED_CONTACT | surface not where any prior puts it | look before moving |
| NOT_HOLDING, PULLED | the part is not where the hand thinks | look |
| BUDGET, EPISODE_OVER, ENV_ERROR, BAD_ARGS | limits or refusals | decide or end |

- The brain is woken on these events only, never on a timer. Periodic review cost NavGPT-3 13× the tokens for +6 SR [navgpt3-2026]; FORGE-plus calls its LLM before execution and on failure only [forge-plus-2026].
- `routine_run` keeps the routine's memory (surface height, hand frame, force reference) per episode. A second call after a correction skips the surface touch.
- Its commands go straight to the session, so `env_observe_image` cannot render that episode afterwards (the shadow replays logged `env_*` calls only).

### 4. The promotion gate

`promotion_gate` is deterministic code. The actor may run `mode: pilot`. A watcher reruns the pilot as a fresh verifier (ASPIRE's admission rule [aspire-2026]) and alone runs `mode: scored`.

| Criterion | Pilot (practice, never promotes) | Scored (held-out, promotes) |
|---|---|---|
| Routine success, zero brain calls | practice starts not used while compiling, nominal twin: rate ≥ target | the first 29 (or 46) held-out starts per part, attempted once: one-sided 95% lower bound ≥ 0.9 |
| Cousins | the same starts × a stratified pick of cousins (every dt × solver combo, both stiffness ends): rate ≥ target | the frozen (held-out start, cousin) pairs: lower bound ≥ 0.9, minimum over parts |
| Envelope (routine-level rule) | `twin/envelope.py` `check_routine` on the nominal episodes; the three features `validation.yaml` names are reported beside it | the same, plus the family's B2 state |
| G monitors | no counted success with an invalidating flag, no integrity flag | the same |
| Frozen | files unchanged since `library_add`, static check passes; the evaluator, starts, cousins and envelope hashes are stored with the result | the same |

- **Verdicts:** pilot pass → `piloted`. Scored pass → `promoted`; a scored fail → `rejected` (its held-out starts are spent for that version).
- **Never records checks.** The gate returns `suggested_checks`; the watcher records them. Pilot numbers are practice-start numbers. They go into a `record_event` note, never into an E check.
- Report cost next to success: env commands, sim s, wall s, brain calls (0) and tokens (0) per success.

### 5. Demotion and staleness

`promotion_gate {mode: review}` re-checks every piloted or promoted version:

- **Stale:** its files changed, or the evaluator code changed since its gate. The results are not comparable; re-gate.
- **Demoted:** in brain episodes, ≥ 5 invocations whose OK-rate has a two-sided 95% upper bound below the target; any later gate whose rate has an upper bound below the target; or any counted success with an invalidating flag.
- Demoted and stale versions are never invoked. A fix is a new version.
- Every change lands in `library/events.jsonl`; log the milestone with `record_event`.

### 6. The library

`runs/<brain>/<run_id>/library/`:

- `index.json`;
- `events.jsonl`;
- `invocations.jsonl`, every `routine_run`: sim time under the routine, commands, code;
- `<name>/v<N>/`: `skill.py`, `params.json`, `meta.json` (sources, structure, hashes, smoke), `gate/`.

It holds task findings. An Argon run starts with an empty library (orchestrator.md) and never reads a Claude library, or the reverse.

### 7. Counting the efficiency claim

- **Standalone routine episodes:** brain calls 0, tokens 0, brain share 0.
- **Brain episodes that invoke routines:** each `routine_run` is one brain decision. `library_list {usage: true}` gives the sim time under routines, so D.brain_share = 1 − routine sim s / episode sim s.
- Report inference time apart from robot time: compiled programs cut inference, not robot time [rax-2026].

## Checks

The watcher runs these, never the actor:

| Check | How | Pass |
|---|---|---|
| frozen | `library_list` → `frozen.ok` for the version; `skill_check` passes | true |
| `E.routine_success` | `promotion_gate {mode: scored}` (caller `watcher-*`) → `suggested_checks`, min over parts, one-sided lower bound in `criteria` | ≥ 0.9 lower bound |
| `E.cousins_success` | the same call, frozen cousin pairs | ≥ 0.9 lower bound |
| `E.envelope_violations` | the same call (full `check_routine` and the family's B2); cross-check with `envelope_check {part, result_file}` | 0 |
| `G.exploit_flags` | the same call → counted successes with an invalidating flag | 0 |
| reproduction | the watcher reruns `promotion_gate {mode: pilot}`; the verdict matches the actor's | same verdict |
| zero brain calls | the gate's episodes are sandboxed skill episodes (`actor routine:<name>@<v>:<sha>`); no brain tool call names their sessions | 0 |
| demotion logged | `promotion_gate {mode: review}`; every status change is in `library/events.jsonl` and on the timeline | logged |

## Lessons

(See the dated entries below.)

## Sources

Ids are in `docs/knowledge/references.md`; synthesis in `docs/knowledge/skill-libraries-and-compilation.md` and `docs/knowledge/two-speed-control.md`.

- [rax-2026] compile after success into staged programs with exceptions; 254k → 4k tokens per episode; peg insertion never compiled.
- [urai-2026] decide between tool calls: 56% vs 24% for a pre-written program with the same tools.
- [aspire-2026] admission by a fresh verifier reproducing the skill.
- [hexis-2026] accept a compiled skill only after static checks and replay of previously accepted traces.
- [skillevolbench-2026] distilled skills can lose cues: keep the raw traces next to the routine.
- [navgpt3-2026], [forge-plus-2026] wake the brain on events, not periodically.
- [agp-2026] documents alone cut tokens 11%; code is what cuts cost.
