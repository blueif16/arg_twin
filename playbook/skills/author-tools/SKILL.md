---
name: author-tools
description: Stage C on any data source. Design, test and version task-level tools (approach, guarded touch, search, insert, release; or move, grasp, place) on top of the env's actor ops, from the twin's sites and frames, acting only on believed geometry, with written feedback conventions; test them deterministically on practice starts with exact and believed information. Use it before practice whenever a brain needs actions above raw poses, or when a tool's interface changes. Source facts are in references/.
stage: tools
steps: [C]
---
## Goal

Versioned task tools (`<task>/vN`) for every target part or task, each with its interface text, its feedback conventions and a passing deterministic self-test on practice starts, run twice: with exact information and with believed information. Done when the source's C checks are recorded by a watcher and Lessons holds the run.

## Choose your path

| The env is | Tools sit on | Adapter | Source facts |
|---|---|---|---|
| **Twin built from robot logs** | `env_*` actor ops, the twin's sites and believed frames | `adapters/robot-logs.md` | `references/reassemble.md` |
| **Benchmark simulator** | the benchmark's actor ops (`libero_move_to`), no sim truth | `adapters/simulator.md` | none yet (attempt N on LIBERO uses raw `move_to`) |
| **Prior tested runs** | the released version | `adapters/prior-runs.md` | the source's reference |

## Inputs

The twin's sites and frames, the env API, practice starts, the brain's earlier traces (what it did by hand).

## Tools

`tools_*` release and self-test tools (`references/reassemble.md` "Tools"); `twin/task_tools.py`.
- **Registry tools carried from v1** (`--skills` allowlists read this section; usage in references/): `task_state`, `task_approach`, `task_move_until_contact`, `task_search`, `task_insert`, `task_release`, `task_tool_selftest`.

## Approach

1. **Act only on believed geometry;** a tool never reads sim truth (rules.md; `sim-integrity`).
2. **Make the interface text match the code.** The `task_insert` description listed `reaim`, which the code rejected with BAD_ARGS (`runs/argon/probes/practice/findings.md`); LIBERO's tool text caused degenerate rot6d (`practice` references/libero.md).
3. **Guard every descent:** an unguarded descent struck the plate at 45 N with no flag (`runs/argon/probes/practice/findings.md`).
4. **State preconditions and refuse outside them:** `task_insert` called 15 mm from the hole spiralled "here" and wasted 26 commands (same).
5. **Test with believed and exact information;** the gap between them is what perception costs.
6. **Tools can lose to raw poses:** pilot arm N (tools + routine) 16/24 vs arm 1 21/24 (`runs/orchestrator/pilot-loop-1/audit/final_1440.md`). Release a tool only if the paired test shows it helps.

## Per-model notes

- **Argon** seldom passes corrections, and when it does they come from its ~2 mm-wrong visual reads; give tools that measure (touch, probe) rather than tools that take a visual offset (`runs/argon/probes/practice/findings.md`).
- **Claude** composes tools well when preconditions are written in the description (`references/reassemble.md` Lessons).

## Checks

The source's C checks (self-test pass on practice starts, exact vs believed), recorded by a watcher.

## Lessons

Dated entries from every source, one line each; the full entry, with its numbers, is in the named reference. New lessons go here (`append_lesson`) with the source named; detail goes to `references/<source>.md`.

- **2026-10-25** (claude): The arm lags sideways while it moves, and a guarded stop freezes the lag. Evidence: `runs/claude/run-standin-1`, `runs/claude/run-standin-practice-1`; full entry in `references/reassemble.md` (v1 `author-tools`).

## Sources

v1 skill text moved unchanged to `references/reassemble.md`; `runs/argon/probes/practice/findings.md`; docs/knowledge/harness-and-tools.md.
