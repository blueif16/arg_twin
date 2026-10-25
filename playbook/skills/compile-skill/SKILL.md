---
name: compile-skill
description: Stage E on any data source. Turn what the brain worked out in practice into a routine that runs with zero brain calls (System 2 to System 1) - when a pattern is ready, extracting it from the practice ledger and command logs into one parameterized program, where the brain stays (hand-back codes), the promotion gate on held-out starts, demotion, and the versioned per-brain library. Use it after practice shows repeated successes, before attempt N, and whenever a routine fails later. Source facts are in references/.
stage: compile
steps: [E]
---
## Goal

A promoted routine per decision pattern (one checker-clean `skill.py` + `params.json`, standalone and invoked modes, hand-back codes) that passes the promotion gate on held-out starts with zero brain calls, in the brain's versioned library; any demotion logged. Done when the source's E checks are recorded by a watcher and Lessons holds the run.

## Choose your path

| The practice ran on | Routine API | Gate starts | Adapter | Source facts |
|---|---|---|---|---|
| **Twin built from robot logs** | `envapi` in the skill sandbox | held-out practice starts, cousins, envelope | `adapters/robot-logs.md` | `references/reassemble.md` |
| **Benchmark simulator** | the benchmark's actor ops | the practice suite's held-out inits; never the scored suite | `adapters/simulator.md` | none yet |
| **Prior tested runs** | the library's last version | starts not yet spent on that version | `adapters/prior-runs.md` | the source's reference |

## Inputs

The practice ledger and command logs of successful episodes, the library, the frozen starts and envelope.

## Tools

`library_extract`, `library_add`, `library_list`, `skill_check`, `skill_run`, the gate (`references/reassemble.md` "Tools").
- **Registry tools carried from v1** (`--skills` allowlists read this section; usage in references/): `routine_run`, `promotion_gate`.

## Approach

1. **Compile only a pattern seen repeatedly,** with its parameters read from the trace (`references/reassemble.md` §1–§2).
2. **Each phase ends on a check the routine can read** (stop reason, depth past the touch, gripper width); never claim success on a hope (§2.4).
3. **Hand back to the brain** with a code when a check fails; the brain stays where perception decides (§3).
4. **Develop on named dev starts only;** the gate runs on held-out starts, which are spent for that version on a fail (§4).
5. **Never use the env's report as an oracle** during development: on practice starts it includes the true final offset, and Argon trilaterated the hole from it (`runs/argon/probes/compile/findings.md`). Dev feedback shows only what the routine itself would measure.
6. **Demote** on ≥ 5 invocations whose OK-rate upper bound falls below target; re-gate when files or the evaluator change (§5).

## Per-model notes

- **Argon** (high thinking, background mode): 879-line program, no sandbox crash or invented API in about 100 episodes; misses were perception and grip (10 of 16 NOT_FOUND, 6 JAMMED); thought summaries claimed reads it could not make (`runs/argon/probes/compile/findings.md`). The recorded runner has no file-writing tool; the probe added `write_file`/`read_file` confined to `ws/`.
- **Claude** wrote v1 routines from its own episodes (`references/reassemble.md` Lessons).

## Checks

The source's E checks (routine success on held-out starts with zero brain calls, cousins, envelope, G monitors, frozen files), recorded by a watcher.

## Lessons

Dated entries from every source, one line each; the full entry, with its numbers, is in the named reference. New lessons go here (`append_lesson`) with the source named; detail goes to `references/<source>.md`.

- **2026-10-25** (claude): first routine, compiled from the Claude stand-in's dry run. Dev runs 1–9 on six named dev starts are in and; the routine is. Evidence: `runs/claude/run-compile-pilot-1`, `runs/orchestrator/pilot-loop-1`; full entry in `references/reassemble.md` (v1 `compile-skill`).

## Sources

v1 skill text moved unchanged to `references/reassemble.md`; `runs/argon/probes/compile/findings.md`; docs/knowledge/skill-libraries-and-compilation.md, two-speed-control.md.
