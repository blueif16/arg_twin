# compile-skill: reassemble

The v1 skill `compile-skill`, moved here unchanged by skills-v2 (archived at `playbook/archive/skills-v1/compile-skill/SKILL.md`; only skill names and paths are updated to where their content now lives). v1 description: Stage E. Turn what the brain worked out in practice into a routine that runs with zero brain calls (System 2 → System 1). Covers when a decision pattern is ready to compile, how to extract it from the practice ledger and command logs into a parameterized skill program, where the brain stays (hand-back codes), the promotion gate (routine success on held-out starts, cousins, envelope, G monitors, frozen files), demotion, and the versioned per-brain library. Use it after practice shows repeated successes, before attempt N, and whenever a routine fails later.

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
- The task tools' conventions (stage C, `task/v1`, `runs/<brain>/<run_id>/task_tools/manifest.json`), when they can run in the sandbox. As of 2026-10-25 09:00, `twin/task_tools.py` still fails `skill_check` (a `getattr` with a non-literal name), so the first routine speaks `envapi` directly and copies the tools' conventions (believed geometry, surface reference, hand-back codes).
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
- It logs every command it relays as its own `env_*` call, with `parent` = the routine_run call (the task tools' convention, author-tools 4b). So `env_observe_image` can still render the episode afterwards, and `integrity_audit` sees the moves. Before 2026-10-25 it sent them straight to the session, and images refused for the rest of the episode.

### 4. The promotion gate

`promotion_gate` is deterministic code. The actor may run `mode: pilot`. A watcher reruns the pilot as a fresh verifier (ASPIRE's admission rule [aspire-2026]) and alone runs `mode: scored`.

| Criterion | Pilot (practice, never promotes) | Scored (held-out, promotes) |
|---|---|---|
| Routine success, zero brain calls | practice starts not used while compiling, nominal twin: rate ≥ target | the first 29 (or 46) held-out starts per part, attempted once: one-sided 95% lower bound ≥ 0.9 |
| Cousins | the same starts × a stratified pick of cousins (every dt × solver combo, both stiffness ends): rate ≥ target | the frozen (held-out start, cousin) pairs: lower bound ≥ 0.9, minimum over parts |
| Envelope (routine-level rule, validation.yaml v17) | `twin/envelope.py` `check_routine` on six features (peak_force_N, approach_speed_mm_s, insert_speed_mm_s, insert_duration_s, wiggle_mm, wiggle_deg), on the routine's own segment of every episode, nominal AND cousin: the whole episode when it kept it (code OK), up to the hand-back when it handed back; only the brain's later actions are excluded (hand-back segments also shown alone in `envelope_handbacks`) | the same, plus the family's B2 state |
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

**2026-10-25, first routine `peg_insert_guarded`, compiled from the Claude stand-in's dry run (pilot, practice starts only; `runs/claude/run-compile-pilot-1/`).** Dev runs 1–9 on six named dev starts (round peg 4 and square peg 3, 002–004) are in `skills/evals/` and `compile/dev/`; the routine is `library/peg_insert_guarded/v1`.

Pilot gate (`library/peg_insert_guarded/v1/gate/pilot_20261025-092011-176e.json`): practice starts not used while compiling, zero brain calls, verdict **not ready**.

| | Nominal (c000) | Cousins | Hand-back codes (nominal) | Envelope (full rule) |
|---|---|---|---|---|
| round peg 4 | 2/12 | 1/8 | ENVELOPE_TIME 10, OK 2 | violates: failed episodes never insert (insert speed low); 1 failure at 17.3 N |
| square peg 3 | 3/12 | 1/8 | NOT_FOUND 8, JAMMED 1, OK 3 | violates: failed episodes never insert |

- **Successes cost little.** A successful episode takes 24–47 env commands and 25–34 s of sim time (36–53 s wall at load 5–15), with 0 brain calls and 0 tokens, and every nominal success sits inside the six-feature envelope. The dry run's successes took 6–9 decisions, 1–3 images and 8–12k tokens each (`run-standin-practice-1/practice/ledger.json`). Per counted success, failures included, the pilot spent 186 s of sim time and 230 env commands.
- **The misses are perception, not motion** (`compile/diagnosis/aim_error_pilot_v1.json`, watcher-side). At the routine's first aim, the true tip sat 1.6–26 mm from the hole, and five starts had more than 6° of in-hand tilt (up to 24°). The routine never seated a start beyond 4.2 mm; 14 of its 19 misses were beyond 4.5 mm. A 4 mm contact search cannot cover that inside the 19 s contact envelope (search cost grows with area / pitch, insertion-strategies).
- **The dev starts hid this.** The six dev starts (002–004) had about 1 mm board error, the pilot starts up to 18.6 mm. The routine was tuned on the easy end of the distribution. Pick dev starts that span the error range (ask the watcher for the spread, not for which start is which).
- **With the hole position given, the motion code works.** In an oracle diagnostic (`compile/diagnosis/oracle/`, `center_mm` = the true offset, dt/2 off, never a check), the same v1 seats 16/24: round 9/12, square 7/12. The three round misses all have 7–24° of in-hand tilt; four of the five square misses have under 5°, so the square search and alignment still need work. 6 of the 16 oracle successes still break one envelope feature (wiggle_mm 4, insert speed 2, approach speed 1), so the envelope tuning on six dev starts did not fully generalize either. So hole-finding is the decision point that stays with the brain: one look, then `routine_run {center_mm}`.
- **The invoke path works.** A stand-in brain episode (`compile/tools/brain_episode.py`, run-compile-tooltest-1) made one `routine_run` (OK, 26 commands, 0 brain calls inside), then `env_finish`: success, counted, no flags, no envelope violations, brain share 0.02.
- **Gate design gaps (proposals, not changed):**
  - (a) The gate fails a routine on the six-feature `check_routine`, but `validation.yaml` E.envelope_violations names three features (force, approach speed, duration). Decide which rule is the gate. *Decided (user, 2026-10-25, validation v16): the six features.*
  - (b) The routine verdict counts hand-back episodes, whose missing insertion reads as insert speed and wiggle "below". Judge the speeds and wiggle on successful episodes, as the real envelope is built from successes, and keep the force check on all episodes.
  - (c) Cousin episodes are not envelope-checked. *Decided with (b) (user, 2026-10-25, validation v16): hand-back episodes do not count against the routine's envelope, and cousin episodes are checked. The gate does both since 2026-10-25. Re-judged under the new rule, v1's pilot nominal episodes no longer violate the envelope: the 2 + 3 kept episodes are clean, and the 10 + 9 hand-backs, including the 17.3 N one, are reported apart. Its one kept round peg 4 cousin episode does violate, on insert speed (`runs/orchestrator/pilot-loop-1/gate_v16_rejudge_v1pilot.txt`).*
  - (d) B2 is "not measured" per family, so no scored gate can pass yet.
  - (e) 20 practice starts per part leave 3 for a v2 after a 12-start pilot and 6 dev starts. Freeze more practice starts, or pilot smaller.
  - (f) The tunable `max_contact_s` lets a brain lengthen the contact cap with no range check.

Method lessons from dev runs 1–9:

- **Success was easy; the envelope was the work.** Dev runs 1–6 seated 2–3 of 3 per part, but every episode broke the approach-speed lower bound (12.4 mm/s round, 10.5 square). The contact onset (|F − F_ref| > 2 N, `twin/envelope.py`) came from the slow surface touch or a force jump, never after a fast approach. Fix the routine, not the envelope:
  - **Touch slowly, under 2 N.** The plate touch uses a ceiling of weight + 0.1 N at 1.5 mm/s. At rest that reads about 2 × weight + 0.1 N (1.63 N for round peg 4), so this only works for parts lighter than about 0.9 N. At 2–3 mm/s a tilted part's first touch spiked to 2.2–2.9 N for one 30 Hz sample, and that sample became the onset.
  - **Unload under the last ceiling, never a higher one.** Raising the ceiling while the filtered target is still deep makes the spring jump: a 4 N onset spike after the touch, a 9 N spike after the seat press.
  - **The hand lags the target.** The 0.2 s target filter plus the impedance make short fast segments slow at the hand: 25 mm/s over 7 mm gave 12 mm/s, and 60 mm/s over 10.5 mm gives 19–20 mm/s. Then 6 mm/s for the last 3.5 mm keeps the rim contact at 5–6 N; a 20 mm/s contact gave 16–28 N. The same lag is why a 12 mm push at 100 mm/s reaches 22–25 mm/s and 6 mm segments never reach the 17.8 mm/s insert-speed floor.
- **Touch flat plate well clear of the hole.** At 16–21 mm from the hole, a 3.6 mm in-hand offset put the touch on the chamfer. The depth estimate then read 1.5–3 mm short, and a seated part was handed back as JAMMED. The touch spots are now 30 mm (round peg 4) and 28 mm (square peg 3) away.
- **A straight-in entry is smoother than any real success.** With 18 mm from TCP to tip, a direct entry moved the TCP 0.3–0.4 mm (wiggle_mm; real minimum 0.64, 1st percentile 0.81). Once the tip is in the hole, lateral offsets do nothing: only tilt moves the TCP (lever × angle). The fix has two parts:
  - an entry rock: 16 steps, 5° about the tip at about 2.8 mm depth, only when the tip went straight in;
  - a push rock capped at the tilt the clearance allows at that depth (2c/depth). Together they give 0.92–1.09 mm.
  - A fixed 4° rock deep in the hole loaded the walls through the 0.4 N·m torque ceiling: 15–19 N peaks and a `plausibility.force` flag.
- **Duration and stalls cap the search.** Round contact must end within 19.2 s, so the routine hands back ENVELOPE_TIME after 10 s of contact, before the push; each probe takes about 3.2 s. The env's `progress.stuck` monitor ends an episode after 40 commands without progress, so square peg 3 stops at 5 probes and hands back NOT_FOUND first.
- **The guarded-stop change does not touch this routine.** It uses force-limited moves only, no `stop_force`, so the env never imposes slow guarded segments. Every envelope violation above was the routine's own choice.
- **Dev harness.** `compile/tools/inproc.py` runs a draft in-process through `twin.env._InProcessAPI` (the envapi surface, dt/2 off) at 20–30 s per episode, for parameter sweeps. It reproduces `skill_run`'s features exactly. Official numbers come from `skill_run` and `promotion_gate` only.

## Sources

Ids are in `docs/knowledge/references.md`; synthesis in `docs/knowledge/skill-libraries-and-compilation.md` and `docs/knowledge/two-speed-control.md`.

- [rax-2026] compile after success into staged programs with exceptions; 254k → 4k tokens per episode; peg insertion never compiled.
- [urai-2026] decide between tool calls: 56% vs 24% for a pre-written program with the same tools.
- [aspire-2026] admission by a fresh verifier reproducing the skill.
- [hexis-2026] accept a compiled skill only after static checks and replay of previously accepted traces.
- [skillevolbench-2026] distilled skills can lose cues: keep the raw traces next to the routine.
- [navgpt3-2026], [forge-plus-2026] wake the brain on events, not periodically.
- [agp-2026] documents alone cut tokens 11%; code is what cuts cost.
