# Orchestrator

This is the fixed top level of the loop. It does not edit skills itself. It runs stages, watches every one of them, and decides when each is done. Skills, tools and the twin evolve; this file stays stable.

## The loop

```
A read footage ─► B build twin ─► C tools from twin ─► D practice ─► E compile ─► back to D
      ▲                ▲                                    │
      │                └──── F physics questions ◄──────────┤
      └──────────────── failure patterns ◄──────────────────┘
```

| Stage | Skills | Done when |
|---|---|---|
| A Read | `read-footage` | Labels for the target recordings exist and are scored against dataset labels; strategy notes and a failure taxonomy exist |
| B Build | `build-twin` (entry point: intake, world frame, board from contacts, scene as code; one adapter per input type), `replay-validate`; `reward-from-video` writes and gates the success check from footage | Held-out keypoint checks pass on every camera (`B0.kp_*`, `B0.two_view_px`, `B0.overlay_hand`; edge overlays are diagnostic since v5); the board pose comes from robot contacts (`board_err` on gate; cameras and mocap only confirm); the B1 probes pass; in reconstruction mode, replays of gate attempts match the real seat rate and force spread while a permissive twin does not (`B2.dist_*`, since v10), and peak force and contact onset meet their targets. Per-attempt outcome accuracy and open-loop prediction are reported, not gated (v8, v10) |
| C Tools | `author-tools` (being written) | Task-level tools exist for every target part, and each passes its own test in the twin |
| D Practice | `practice`; `auditor` explains paired outcomes and audits trials | The target skill reaches the success bar on frozen held-out starts, or the budget runs out |
| E Compile | `compile-skill` (not written yet) | Promoted routines pass held-out starts with no brain calls; any demotion is logged |
| F Feedback | `replay-validate` | Each physics question raised in D/E is answered against the real recordings |
| G Integrity | `sim-integrity`, `auditor`; watchers use `auditor` (check validation, trial audits) and `integrity_audit` | Every G check in `validation.yaml` meets its target: no exam reads, no rule violations in the log audit, no counted success with a monitor flag, no sandbox escape, every planted exploit caught |

**Data tracks** (`validation.yaml` `tracks:`). The same loop runs on different kinds of recordings, and a track's checks never mix with another's. `reassemble` runs the full loop. `so101` (LeRobot SO-101) covers reconstruction and replay only, as stage T2 with `build-twin`; each T2 check names the loop step it belongs to (`step: B0`, `B2`). `footage` (video only) has not started. `libero` (LIBERO, sim playground with exact ground truth) runs the task loop with the simulator as the twin: `reward-from-video` on B0, `practice` §9 on D, `auditor` on D and G; its checks are stage L.

**Skills are shared across sources** (skills v2, 2026-10-25). Each job has one skill; its "Choose your path" table picks an adapter by input type (`adapters/robot-logs.md`, `video-only.md`, `simulator.md`, `prior-runs.md`) and source facts live in `references/<source>.md`. v1 skills are in `playbook/archive/skills-v1/`.

## Watching

For every stage:

1. **Before work starts,** write down the stage goal and the budget (tokens and wall time). Checks and targets come from `validation.yaml` and are frozen at that point. `record_check` judges against the file as it is when called, so a mid-run edit changes later verdicts: a change needs a new run (`rules.md`).
2. **Watchers measure, the harness judges.** Watchers measure with domain tools and record each result with `record_check`; the harness, not the agent, decides pass or fail against the target. Milestones (a stage passes, a regression, a first success, a skill promoted) are recorded with `record_event`. The status page `runs/status/index.html` is rebuilt from these records automatically.
3. **Run actors and watchers as separate agents.** A watcher gets the outputs and the checks, never the actor's reasoning.
4. **Read the evidence each round:** check results, the call log, renders and overlays. Don't take an actor's summary on trust.
5. **Escalate** when checks stall for two rounds, the budget runs out, or a rule in `rules.md` is broken:
   - say what's blocking
   - name the skill to revise
   - re-run
6. **Log every round** to `runs/loop/<run-id>/`: what was tried, check results, tokens, time, which brain made each decision, and its decision points.

## Starting a run

- Pick the brain (`claude` stand-in or `argon`), the recordings, and the target parts.
- The splits are frozen in `splits.json` (exam = the dataset's own test split minus episode 0; gate = a hash-picked fifth of its train split):
  - train recordings: read and replayed freely
  - gate recordings: used to decide acceptance
  - exam recordings: scored once at the end. `harness.registry.call` refuses any tool call naming an exam episode or recording (arguments `episode`, `episodes`, `recording`, `recordings`) unless the caller is `exam-scorer`. Code that scans many episodes must filter by `splits.json` itself; listings are not covered by the guard. `integrity_audit` reports every call that touched exam data.
  - SO-101 has its own split file, `splits_so101.json` (episodes 0–34 train, 35–49 gate, no exam). SO-101 tools take `ep`/`eps`, never `episode`, because the guard reads `episode` against REASSEMBLE's exam ids.
- For an Argon run, start from an empty learned-skill library. Playbook skills stay; skills learned in earlier runs don't.
- **Keep two kinds of knowledge apart.**
  - *Method* lessons stay in a skill's Lessons and are given to every brain: how to use a tool, what order of steps works, boundary conventions.
  - *Task findings* go under `runs/<brain>/<run-id>/` and are never handed to another brain's run: strategy notes, failure taxonomy, parameter values, facts about a part.
  - This keeps "Argon worked it out" true.
- **Recordings whose labels are printed in `docs/`** (episode 0, in `docs/research/reassemble-data-notes.md`) are train-only. They are never used as gate or exam recordings.
