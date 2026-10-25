# B2 parallel search: launch brief for the orchestrator

Written 2026-10-25 from a cloud session, from the facts in `playbook/skills/replay-validate/SKILL.md` (Lessons, 5a and 5b) and `docs/design/progress-log.md` (09:45). It proposes how to run many B2 formulations at once. It changes no check, target or selection rule. Any change to those is the user's decision.

## Goal

Find a replay formulation that passes B2's distribution checks on train with recording-level CV, plus the loose-twin control, without breaking the integrity rules. If none can, show that with evidence, so B2 can be reported as unreachable for open-loop replay instead of being searched forever.

## What we know (train, 308 in-hand peg windows)

| Formulation | Seat rate (real 0.776) | Force KS | Loose-twin control (needs ≥ +0.10) |
|---|---|---|---|
| v3 | 0.54 | 0.31 | fails |
| v5a, softer board | 0.58 | 0.20 | −0.094, fails |
| v5b, grasp compliance | 0.630 | 0.295 | −0.036, fails |
| 5a softness mixture | passes the numbers | 0.15 | +0.104 |

The mixture is rejected: every sampled seat pushed the peg 0.8–1.7 mm into the walls.

- **The main lead:** 60 of 72 unseated real successes reach k_ref more than 1 mm off the hole (median 2.3 mm). In 52 of them the twin's hand is over 2 mm from the real hand. The replay drifts before release, which no contact setting can fix.
- **Not separable by physics:** 55 of 69 real failures are aborts or misses (23 never released, 25 under 0.3 s of force). Only 14 have a physical-jam signature, and the twin jams all 14.
- **Speed:** `replay_batch` with 15 workers takes about 2 min per 300 windows.
- **Not trusted:** the GPU rig (`twin/rig.py`) seats 8–15% where the CPU arm replay seats about 50%. Don't use it for scoring.

## Phase 1: diagnosis, 3 agents in parallel, about 1 hour

The running B2 diagnosis agent already owns this. Split its work three ways instead of starting a duplicate.

| Agent | Error source | Oracle ablation (diagnostic only, never a score) |
|---|---|---|
| D1 | Board pose | Place each hole from that recording's own contacts, leaving the replayed attempt's contact out |
| D2 | Arm tracking and drift | Hold the twin's hand on the logged path until first contact, then hand back to the rebuilt command |
| D3 | In-hand pose | Register from the same attempt's contact (the current 3b rule), with 5b compliance on and off |

Each agent reports, on all 308 windows:
- the arrival error at k_ref, split into its share of the 1 mm+ cases;
- seat rate, force KS and the loose-twin control with that source made perfect;
- the fraction of the 0.776 − 0.54 gap it closes.

Write results to `runs/claude/run-standin-1/replay/b2search/diag_<D1|D2|D3>.json`, plus a one-paragraph summary.

**Decision:** the family for Phase 2 is the source whose oracle closes the most of the gap, or a combination if no single one dominates. If even all three oracles together fail the loose-twin control, stop and report to the user: open-loop replay cannot reach the gate as defined.

## Phase 2: tournament, up to 3 hours

### Actors

One agent per formulation. Before running, each declares its formulation in `b2search/declarations/<id>.md`:
- what changes;
- which data it reads;
- why it is label-blind;
- the expected effect;
- the params file hash.

A changed variant is a new declaration. Candidates from the playbook and knowledge base, kept to the Phase 1 family:

- **Board:** per-recording board pose from that recording's contacts, leave-one-attempt-out.
- **Arm:**
  - a declared, bounded lateral setpoint correction that stops at first contact (Google's recipe B allows bounded command edits near contact; `replay-validate` lists it under next candidates);
  - command rebuild variants: K inside 200–800 N/m, force filter window ≤ 1 s (ACP);
  - controller Λ and armature.
- **In-hand:** registration variants, and the winner combined with 5b compliance.

### One evaluation queue

Actors never call `replay_batch` with their own workers. They submit a params file to one queue: a single runner, FIFO, 15 workers, about 2 min per evaluation. That is about 25 full evaluations an hour, enough for 8–12 actors. Always evaluate all 308 windows. If a stratified subset is ever used, reweight its TPR and TNR to the real base rate before computing a seat-rate error.

### One scorer

A deterministic script, written and hashed before Phase 2 starts, run by the watcher, never by an actor. For each candidate it reads:
- the candidate's train run;
- the loose twin (holes +1 mm per side, friction 0.05) on top of that candidate, not on top of v3;
- the dt/2 rerun;
- the per-window part–board penetration log (5a's recorder hook).

It outputs one row per candidate in `b2search/leaderboard.json`:

| Column | Pass |
|---|---|
| CV seat-rate error, round and square | ≤ 0.10 each (`B2.dist_seat_rate_err`) |
| Force KS | ≤ 0.20 (`B2.dist_force_ks`) |
| Loose-twin control margin | ≥ +0.10 (`B2.dist_negative_control`) |
| Max part–board penetration in seated windows | disqualify above 0.5 mm (IndustReal's filter, `docs/knowledge/contact-physics-and-sysid.md`) |
| dt/2 outcome agreement | report; disqualify if far below v3's 96% (296/308) |
| Command edits after first contact | disqualify if any: they feed the logged outcome back in |

### Guards against luck

- **Held-back slice:** before Phase 2, the watcher draws about 20% of train recordings by seed and lists them in `b2search/holdout.json`. No actor may replay them. The winner must also pass on them.
- **Selection rule, unchanged:** lowest train-CV seat-rate error among candidates that pass every row; ties go to fewer free parameters. Then the watcher scores gate once with the winner.

### Stop rule

Stop at the earlier of two tournament rounds with no pass, or a clock time the orchestrator sets now. Then report the best candidate, its leaderboard row and the Phase 1 evidence to the user. Gate stays unscored and all D/E results stay provisional.

## Compute

- Run Phase 1 and 2 now, while there is no Argon key.
- When Argon practice starts, drop the queue to 6–8 workers: practice images re-simulate on the CPU, and overnight contention made them about 10× slower.

## Records

- Everything goes under `runs/claude/run-standin-1/replay/b2search/`: declarations, queue log, leaderboard, holdout list.
- One progress-log entry per phase.
- Method lessons go to `replay-validate` Lessons. Parameter values stay in `runs/`.
