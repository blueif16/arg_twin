# Progress log

The orchestrator's running record: what was launched, what came back, what was decided, and what's next. Newest entries go at the bottom. The status page timeline (`runs/status/`) holds the per-check history; this file holds the reasoning that connects it.

## Standing mandate (user, 2026-10-25, before sleeping)

- Keep trying the recommended approaches, non-stop, until a real dead end. Don't give up easily.
- Get a **fully connected loop** running as fast as possible to prove the concept: read → twin → tools → practice → compile → practice again, cheaper.
- When confused or blocked: check the research library (`docs/knowledge/`) first. Then have subagents try alternative formulations in parallel.
- Keep a good record of progress and history.
- No Argon key needed: Claude subagents play the brain. The main thread only orchestrates; subagents build, test and watch.

## The loop and its critical path

```
A read ─► B twin (pegs) ─► C task tools ─► D practice ─► E compile ─► D again (cheaper) ─► F feedback to B
```

- **Shortest path to a closed loop:**
  1. Run C and D on the current peg twin, scoped to the best-reproduced pegs.
  2. Promote one routine in E, then rerun D.
  3. Results stay **provisional** until B2's distribution gate passes for pegs.
- B2 continues in parallel. The loop doesn't wait for it, but no result is reported as validated until it passes.
- **Scoring:** a dedicated watcher subagent, never the actor, records checks.
- **Evaluation discipline:** the evaluator is frozen, starts are frozen, and held-out starts are scored once per actor. Paired baseline vs attempt N, McNemar test, sample-size rules (validation header).

## State at 2026-10-25 (start of the overnight run)

| Stage | Pass | Notes |
|---|---|---|
| A | 1/3 | F1 0.933 (episode 0, actor-scored). Blind reader on gate is running |
| B0 | 5/6 | Recalibrated: hama1 1.63 px, hama2 1.51 px, two-view 0.47 px, board 2.37 mm. Wrist camera pending a measure_b0 board fix (audit agent) |
| B1 | 4/4 | Pegs only |
| B2 | 1/6 | free_space 0.62 mm passes. Per-attempt accuracy plateaued at ~0.65; gate is now distribution-level (v10). Round 4 (in-hand pose from force, warp particle filter) is running |
| C, D, E | 0 | Task-tools and practice-harness agents running. Compile not started |
| G | 2/5 | Catch rate 1.0 (red team 15/15, watcher rerun); boundary probe 0 escapes. Audit tool being built |
| T2 (SO-101) | 0/5 | Reconstruction agent running |

**Agents running:**
- B2 round 4
- SO-101 reconstruction
- episode-1 confirmatory calibration
- UI data-source switcher
- task tools (C)
- practice harness and stand-in dry run (D)
- audits, wrist fix and playbook consistency pass
- blind footage reader (A)

## Next actions, in order, as agents return

1. **Watcher subagent** (separate from all actors) scores everything pending:
   - B0.overlay_hand on episode 0, with the fixed measure_b0
   - episode-1 B0
   - A segment F1, success bacc and grounding on the blind gate reads
   - C.tool_success
   - G audits
   - T2 checks
   - B2 gate, once round 4 freezes params_v4
2. **Attempt 1, the baseline.** A stand-in runs it on frozen held-out starts with the frozen budget: absolute end-effector targets, full history, calibrated geometry in the prompt.
3. **Attempt N:** the same starts, with task tools and practice. Measures tokens, calls and time per success.
4. **Compile (E):** write the compile skill, promote the best routine into sandboxed code, run the cousins gate and envelope check. Then rerun with zero brain calls.
5. **Feedback (F):** physics questions raised in practice go to the replay agent.
6. **Then:** gears and the nut via MuJoCo SDF plugins (the H2 ladder), SO-101 replay, the footage-only track.

## Log

- 2026-10-25: overnight mandate received. The 8 agents above are running. Progress log started.
- 2026-10-25: episode-1 confirmatory calibration returned (actor numbers). The static cameras reproduce episode-0 quality from scratch: 1.72 and 1.77 px held-out, two-view 0.39 px; native guesses were 22 and 65 px. Tool length 48.9 mm, consistent with 47.7 mm on episode 0. The wrist board check fails at 17 px with thin evidence (3 parts, one hole in view): an evidence-sufficiency problem, not a method failure. Watcher tool blockers (run_id defaults, keypoint reuse, frame radius) sent to the audit agent. Playbook gaps queued in its pending list. Next: the watcher re-measures episode 1 once measure_b0 is per-recording.
- 2026-10-25 heartbeat: 7 agents running (B2 round 4, SO-101, UI switcher, task tools, practice harness, audits, blind footage reader). The critical path is active (C and D in progress). No action.
- 2026-10-25: audit agent returned.
  - The integrity_audit tool is built. It found exam leaks (episodes 9 and 10 via recording_info listings, plus one pre-guard query) → they're burned for the claude brain (exam_burned.json, v14). The root-cause fix of listing filters and a registry safety net went back to the audit agent.
  - measure_b0 is per-recording and scores against the contact board: episode-0 wrist part points 13.95 → 5.82 px.
  - Playbook consistency fixes are done. Pending fixes for skills other agents own are in runs/claude/run-standin-1/audit/playbook_fixes_pending.md.
  - Launched the standing watcher: B0 hand on episode 0, B0 on episode 1 (fresh run_id), G audits.
- 2026-10-25: UI data-source switcher done. The page switches REASSEMBLE / SO-101 / Footage-only under Real footage; every record carries track and step; publish_media(step="input") sets a track's footage. SO-101 shows 2 real clips, Footage-only is greyed with its reason. Viewport and switcher tests pass.
- 2026-10-25: blind footage reader returned. Gate episodes 68, 136 and 140 read cold (hand camera only; two HF video files downloaded; footage tools now hide gate labels by default). 6 failures called from signals before video. About 13k unique tokens per recording minute. The skill misled on connector grip widths, part height, slow-pull lift onset, and 'no nut travel = failure' (Lesson added). Sent to the watcher for A.segment_f1, success_bacc, claims_grounded and tokens_per_min.
- 2026-10-25: exam leak fixed at the root (twin/exam.py plus a registry results filter); audit per brain: claude exam_reads 4 (burned 9/10), 0 cheat violations in frozen runs. Ruling: run-standin-practice-1 is a dry run; scored runs freeze a fresh manifest.
- 2026-10-25 heartbeat: 5 agents running (B2 round 4, SO-101, task tools, practice harness, watcher). The critical path is active (C and D). No action.
- 2026-10-25: watcher scored the pending checks.
  - B0, episode 0: 6/6 (wrist 5.82 px).
  - B0, episode 1 confirmatory: static cameras pass (1.67 / 1.40 px, two-view 0.17 px, tool tip 0.14 mm). Wrist board fails at 26 px on thin evidence (4 marks).
  - A: F1 0.961 pass; success_bacc 0.78 fail (4/7, provisional); claims_grounded 0.667 fail (USB/Ethernet swap); 13k tokens per minute.
  - G: exam_reads 4 (burned episodes only); cheat_violations 0.
  - Launched read-footage refinement on train only: part identity from which hole (contact-board layout) and failure from end-state physics. A fresh blind re-test on 3 new gate episodes follows.
  - Queued: measure_b0 should exclude calibration fit frames. Wrist hand-eye transfer test (does the episode-0 mount calibration hold on episode 1?) is a cheap consistency check for later.
- 2026-10-25: **practice works end to end** (dry run, stand-in as brain, raw-pose baseline, 6 practice starts): 3 counted, 1 voided by the empty-close false flag, and round peg 1 stalls above target (5.5–7.1° in-hand tilt). 0 false done, 0 limit hits. About 1000 s per success with physics paused (images re-simulate).
  - Routed: env fixes to the env builder (empty-close false flag, render op, guarded-stop overshoot 1.5 → 5.6–10.9 N, limit_reached op). Task tools must log their env commands, and the round-peg-1 tilt stall goes to the task-tools agent.
  - Added D.limit_hit_rate (v15). Task findings saved in runs/claude/run-standin-practice-1/task_findings.md.
  - Launched E: the compile-skill, promotion gate tooling, and a first routine pilot on practice starts.
  - Next once env fixes land: a pilot loop on practice starts. Baseline (attempt 1) vs task tools (attempt N) vs compiled routine (zero brain calls) on the same practice starts, parallel stand-ins. Then the scored comparison on the pre-declared 29 held-out starts per peg, with a fresh manifest.
- 2026-10-25 heartbeat: 6 agents running (B2 round 4, SO-101, task tools, read-footage refinement, env fixes, compile E). The critical path is active (C, D fixes, E). No action.
- 2026-10-25 heartbeat: same 6 agents running; the compile agent is writing library code. No stalls; no action.
- 2026-10-25: read-footage refined on train only. On 29 held-out train recordings: success bacc 0.649 → 0.892, part ID 0.64 → 0.96 (part from hole via identify_part_by_hole; end_state_check), F1 0.926 → 0.961. Launched a fresh blind reader on gate episodes 99, 102 and 103; the watcher scores after.
- 2026-10-25: B2 round 4 negative. The force particle filter on a GPU impedance rig doesn't pass; the rig seats 8–15% of its worlds vs ~50% for the full-arm CPU replay. Best is still v3: seat rate 0.54 vs real 0.78 (error 0.24), KS 0.31; the permissive twin is closer to reality. Diagnosis: the twin is systematically too strict. The replay agent is wrapping up, documenting the rig issue. Next: two parallel formulations on train with recording-level CV. (a) Distribution calibration of global clearance, friction and contact softness to cover the real seat rate and force spread (QBIT style). (b) A physical grasp-compliance model (soft pads plus rotational give) identified from measured tilt and slip. Selection rule, declared now: pick the variant with the lower train-CV seat-rate error that also passes the train negative control; the watcher scores gate once with it.
- 2026-10-25 heartbeat: the B2 replay builder finished (documented). 7 running: SO-101, task tools, env fixes, compile E, blind footage re-test, B2 5a, B2 5b. Critical path active; no action.
- 2026-10-25 08:15: **interruption.** The machine restarted at 08:13. All 7 agents stopped mid-task at about 02:56 without reporting, so roughly 5 h were lost. Modules compile and the tool registry loads. Where each agent stopped:
  - B2 5a: one screen done. Soft contact (tc020) seats 77/308 train windows (25%) vs real 78%, bacc 0.52, worse than v3. Four screens (dmin050, fr005, fr030, k800) queued but never ran.
  - B2 5b: identification data extracted; the compliance-model fit was mid-way, no params.
  - SO-101: frozen and pre-registered, gate run once. Side camera 1.06 px, up camera 152 px (auto-marks picked the wrong teal blob), two-view 67 px, grasp-to-brick 16.5 mm. It then started fixing the mark tool after seeing the gate, and the fix is unvalidated.
  - Task tools: final selftest wrote 26 rows, no summary.
  - Compile E: dev run 7 (touch, unload and probe fixes) started but no result.
  - Blind re-test: episodes 99 and 102 were mid-read and 103 had just started; no read files saved.
  - Env fixes: was mid-edit of tests/monitor_sanity.py (dt/2 quirk selftest regression).
  - The repo has no commits.
- 2026-10-25 08:20: user approved the restart. Snapshot commit 34e3958 (code, playbook, docs; runs/ stays outside git per .gitignore). **SO-101 ruling (user):** the first gate run is the result of record. Mark-tool fixes are validated on train only, and any later gate number is labelled fixed-after-gate, never reported as a clean pass. Relaunching all 7 agents from their transcripts with CPU worker caps, so the heavy B2 jobs don't overload the 16 cores like last night.
- 2026-10-25 08:24: relaunched all 7 from resume packs (runs/orchestrator/resume-20261025/: original brief, mid-run messages, work digest). Worker caps on 16 cores:
  - B2 5a and B2 5b: 6 each
  - env fixes and task tools: 3 each
  - compile and SO-101: 2 each
  - blind re-test: 1 heavy process

  The SO-101 agent carries the gate ruling. The selection rule for B2 5a vs 5b is unchanged. The heartbeat cron (at :17 and :47) is still active.
- 2026-10-25 08:25 heartbeat: the 7 relaunched agents started a minute ago and are still reading their packs (load 0.4). Critical path active (env fixes, C, E); no action.
