---
name: practice
description: Stage D on any data source. Practise a task in the twin (or the benchmark simulator) through the env API, in paired attempts on frozen starts. Attempt 1 is the strongest published uncompiled recipe for that source; attempt N adds task tools, the skill library, and retries with reflections and observer nudges. Covers freezing the run, the episode protocol, the frozen reward and evaluator, the event-woken watcher, paired statistics and where knowledge goes. Pick the adapter by what the twin is; source facts are in references/.
stage: practice
steps: [D]
---
## Goal

Paired attempts on the same frozen starts: attempt 1 (the published recipe, unchanged) and attempt N (our additions), each scored by the frozen evaluator (and, where one exists, the frozen reward beside it), with the comparison's paired statistics and cost per success. Done when the source's D checks in `playbook/validation.yaml` are recorded by a watcher (REASSEMBLE stage D; LIBERO stage L), the auditor has explained the outcomes, and Lessons holds the run.

## Choose your path

| The twin is | Actor ops | Success is | Attempt 1 anchor | Adapter | Source facts |
|---|---|---|---|---|---|
| **Built from robot logs** (stage B passed) | `env_*` (absolute TCP targets, gripper, waits, images with belief overlays) | the frozen evaluator on sim state (`twin/evaluator.py`) | absolute TCP targets, full history, calibrated geometry, no task tools | `adapters/robot-logs.md` | `references/reassemble.md` |
| **A benchmark simulator with a state API** | `libero_*` (`move_to` absolute pose, `observe`, `done`, `give_up`) | the frozen reward from `reward-from-video`, with the benchmark predicate recorded beside it | the published protocol, unchanged (43 % on LIBERO-10 at high effort) | `adapters/simulator.md` | `references/libero.md` |
| **Built from video only** | the twin's env API | the frozen reward, human-confirmed | none published; report as exploratory | `adapters/video-only.md` | none yet |
| **Resumed from prior tested runs** | as above | as above | the last frozen attempt-1 run, not a new one | `adapters/prior-runs.md` | the source's reference |

## Inputs

- A frozen run manifest (`practice_manifest`): starts, budgets, evaluator and reward hashes, the library version. Practice and gate starts only; exam starts never (rules.md).
- The skill library (attempt N), task tools (`author-tools`), the reward (`reward-from-video`).

## Tools

The env tools of the source (`env_*`, `library_*`, `routine_run`, or `libero_*`), `practice_manifest`, `record_check`; the brain runner (`brains` skill; `harness/brains/trials.py` for retries with feedback).
- **Registry tools carried from v1** (`--skills` allowlists read this section; usage in references/): `env_open`, `env_starts`, `env_reset`, `env_close`, `env_observe`, `env_observe_image`, `env_move`, `env_gripper`, `env_wait`, `env_status`, `env_finish`.

## Approach

Core method; it holds on both sources run so far.
1. **Freeze before attempt 1.** Budgets are enforced by the env, not the brain; a mid-run edit needs a new run (`references/reassemble.md` §1; `playbook/orchestrator.md`).
2. **Attempt 1 is the published recipe unchanged,** including its flaws; fix them only in attempt N and say so. LIBERO's published `move_to` text made Argon send all-zero rot6d (`references/libero.md`).
3. **Measure before acting.** Soft touch short of the target, then one image with the tool and target at the same height: 17/24 first-descent seats vs 1/24 for the arm that read the hole only (`runs/orchestrator/pilot-loop-1/audit/final_1440.md`). On LIBERO: lift 3–5 cm and confirm in both cameras that the object rose (chain L3).
4. **One visual read is a probe start, not the aim** (±2 mm on pegs, often the wrong sign); read the rim or edge, not the dark area (`runs/argon/probes/practice/findings.md`).
5. **Localise by contact where contact exists**; pull a stuck object gently; change the approach after two equal failures (`runs/orchestrator/libero-proof/skills/practice_libero.md`).
6. **Claim done only on named evidence.** LIBERO episodes fail by false `done` (2 of the first 4 failures; 287 of 1,800 in the paper) (same).
7. **Retries with feedback (attempt N):** up to k = 4 trials per start; between trials a reflection of at most 6 lines from the failed trial's own trace; mid-trial nudges on a progress stall. Observers see only what the actor received plus the frozen reward's outputs, never sim truth (rules.md; leak check `L.observer_leaks`).
8. **Score twice where two checks exist** (reward and benchmark predicate); their disagreement is a reward-hacking lead for the auditor.
9. **Paired statistics on matched starts** (McNemar exact) and cost per success; tasks findings stay in the run, method knowledge goes to Lessons.

## Per-model notes

- **Argon:** high thinking for attempt 1 on the peg twin gave the fewest calls and commands; avoid low; media resolution medium; 300 s client timeout with retries; sync `interactions.create` can hang, so prefer background or timeouts (`runs/argon/probes/practice/findings.md`). Medium mostly does not think (7 of 9 episodes 0 thought tokens) and reasons in long `log_note` essays. Cached input was 2.6 % on LIBERO because image eviction rewrites history (`references/libero.md`).
- **Claude (stand-in):** reads images as PNG; thinking is redacted in transcripts, so its notes and command descriptions are the record (`runs/orchestrator/pilot-loop-1/audit/final_1440.md`).

## Checks

The source's D checks in `playbook/validation.yaml` (success on frozen starts, attempt-1 anchor, success@k, cost per success, monitor-flag voids), recorded by a watcher; G checks run beside them (`sim-integrity`, `auditor`).

## Lessons

Dated entries from every source, one line each; the full entry, with its numbers, is in the named reference. New lessons go here (`append_lesson`) with the source named; detail goes to `references/<source>.md`.

- **2026-10-25** (claude): Claude stand-in dry run of the attempt-1 baseline. Evidence: `runs/claude/run-standin-practice-1`; full entry in `references/reassemble.md` (v1 `practice`).
- **2026-10-25** (agent): the dry run replayed under the slower guarded stops. Evidence: `runs/orchestrator/pilot-loop-1`; full entry in `references/reassemble.md` (v1 `practice`).
- **2026-10-25** (argon): simulator path: the anchor, the reward and the first chain. Evidence: `runs/libero/feas-argon-high-1`, `runs/argon/probes`, `runs/argon/libero_chain`; full entry in `references/libero.md` (v1 `practice`).

## Sources

- v1 skill text moved unchanged to `references/reassemble.md`; LIBERO section and lesson from `runs/orchestrator/libero-proof/skills/practice_libero.md` in `references/libero.md`.
- `runs/argon/probes/practice/findings.md`, `runs/orchestrator/pilot-loop-1/audit/final_1440.md`, `runs/argon/libero_chain/`.
- docs/knowledge/llm-as-robot-policy.md, insertion-strategies.md, evaluation-and-statistics.md.
