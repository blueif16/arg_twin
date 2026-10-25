---
name: sim-integrity
description: Stage watch (G, D, E) on any data source. Keep practice in the twin or simulator honest by construction - the capability boundary (env API, sandboxed skill processes, static checker), the frozen evaluator and reward by hash, runtime monitors, the real-data envelope, digital cousins and frozen starts, and red-team exploits. Use it before any practice run, when adding a monitor or a tool that touches the simulator, and when a red team plants exploits. Source facts are in references/.
stage: watch
steps: [G, D, E]
---
## Goal

A practice setup where an actor cannot reach sim truth or the evaluator, every counted success survives the monitors, and every planted exploit is caught. Done when the source's G checks in `playbook/validation.yaml` meet their targets (REASSEMBLE stage G; LIBERO `L.observer_leaks`, cheat catch rate) and Lessons holds the run.

## Choose your path

| The twin is | Boundary | Frozen evaluator | Monitors | Adapter | Source facts |
|---|---|---|---|---|---|
| **Built from robot logs** | env API, sandboxed skill processes, static checker | `twin/evaluator.py` + `monitors.py` hashed in every record | `twin/monitors.py`, envelope, cousins | `adapters/robot-logs.md` | `references/reassemble.md` |
| **Benchmark simulator** | the `libero_*` server; object poses, state and the predicate never sent | predicate + BDDL + init files hashed in `frozen.json`; exact replay | penetration > 20 mm, free body > 5 m/s, BADQACC | `adapters/simulator.md` | `references/libero.md` |
| **Built from video only** | as robot logs | the frozen, human-confirmed reward | cousins widened by the scale error | `adapters/robot-logs.md` | none yet |

## Inputs

The run manifest, the env and server code, the call log, run records; red-team exploit list.

## Tools

`integrity_audit` (scans the call log and records for exam reads and rule violations), `practice_manifest {verify: true}`, the monitors; the `auditor` skill for blind trial audits.
- **Registry tools carried from v1** (`--skills` allowlists read this section; usage in references/): `env_open`, `env_starts`, `env_reset`, `env_move`, `env_gripper`, `env_wait`, `env_observe`, `env_status`, `env_finish`, `env_close`, `env_charge_tokens`, `skill_check`, `skill_run`, `run_actor`, `monitor_selftest`, `envelope_check`, `integrity_info`.

## Approach

1. **Honest by construction, not by instruction:** capability boundary first, then frozen files, then monitors (`references/reassemble.md` §1–§3).
2. **Freeze the evaluator and the reward by hash;** the server refuses a mismatch (`references/reassemble.md` §2; `reward-from-video`).
3. **Observers and auditors never pass sim truth** to the actor; scan every observer input for truth names (rules.md; `L.observer_leaks`).
4. **Monitors plus a second line:** blind trial audits catch what monitors miss and vice versa (teleport, soften, widen-and-restore caught; a fallen peg read as seated; `runs/argon/probes/audit/findings.md`).
5. **Claims only inside the real envelope;** sample what is not identifiable as cousins (`references/reassemble.md` §4–§5).

## Per-model notes

- **Argon as a red team or auditor:** medium thinking, video at 4 fps in a user step, 3 calls and a majority vote; high thinking wrote confident fiction with 2/4 false alarms (`runs/argon/probes/audit/findings.md`).

## Checks

The source's G checks: no exam reads, no rule violations in the log audit, no counted success with a monitor flag, no sandbox escape, every planted exploit caught (`playbook/orchestrator.md`).

## Lessons

Dated entries from every source, one line each; the full entry, with its numbers, is in the named reference. New lessons go here (`append_lesson`) with the source named; detail goes to `references/<source>.md`.

- **2026-10-24** (claude): Soft contacts give legitimate transients over 0.5 mm. Evidence: `runs/claude/run-standin-1`; full entry in `references/reassemble.md` (v1 `sim-integrity`).
- **2026-10-24** (claude): red team. 16 attempts aimed at a counted success, and none counted, so the catch rate is 16/16. Evidence: `runs/claude/run-standin-1`; full entry in `references/reassemble.md` (v1 `sim-integrity`).
- **2026-10-24** (claude): builder follow-up to the red team. Evidence: full entry in `references/reassemble.md` (v1 `sim-integrity`).
- **2026-10-25** (claude): integrity audit of run-standin-1. The evaluator code hash changed 6 times inside this run, after its first practice episode. The 11 counted practice successes span three…. Evidence: `runs/claude/run-standin-1`; full entry in `references/reassemble.md` (v1 `sim-integrity`).
- **2026-10-25** (claude): builder fixes after the first practice dry run. Evidence: `runs/claude/run-standin-practice-1`, `runs/claude/run-standin-1`; full entry in `references/reassemble.md` (v1 `sim-integrity`).

## Sources

v1 skill text moved unchanged to `references/reassemble.md`; `runs/argon/probes/audit/findings.md`; docs/knowledge/evaluation-and-statistics.md.
