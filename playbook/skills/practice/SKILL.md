---
name: practice
description: Stage D. Practice in-hand peg insertion in the twin through the env API, in paired attempts on frozen starts. Attempt 1 is the strongest published uncompiled recipe (absolute TCP targets, full history, calibrated geometry, no task tools); attempt N adds task tools and the skill library. Covers freezing the run, the episode protocol, the stage rubric, the event-woken watcher, paired statistics and where knowledge goes. Use it before any practice episode, and when watching or scoring one.
stage: practice
steps: [D]
---
## Goal

The target skill (insert the in-hand part of one family into its hole) passes `D.success_rate` on frozen held-out starts, or the frozen budget runs out. Every number comes with its cost: tokens, brain calls and time per success.

- **Paired attempts on the same frozen starts:** attempt 1 (baseline) and attempt N, compared with an exact McNemar test.
- **Attempt 1 is not a strawman.** It is the best published uncompiled recipe: Astranomy's agentic setup [astranomy-2026] plus Google's visual coordinate harness [google-argon-robotics-2026].
- **Limit hits are reported apart from failures.** 41 of Argon's 100 LIBERO-10 episodes hit a limit [llm-libero-2026].
- **Self-verification is measured.** Every episode records the actor's claim next to the evaluator's verdict (`D.false_done_rate`).

## Inputs

- A frozen run: `runs/<brain>/<run_id>/manifest.json`, written by `practice_manifest` before attempt 1, together with that run's `starts/`, `cousins/`, `envelope/` and `env/budget.json`.
- The env's task (from `env_reset`): the believed hole entry pose (the nominal contact-fit board), the nominal grasp, part geometry, clearance, limits and conventions. The start's board error and in-hand pose are never given. Finding them out by touch or by looking is the task.
- For attempt N: the task tools (stage C, interface `task/v1`; the list is in `runs/claude/run-standin-1/task_tools/manifest.json`) and the learned skill library, which starts empty in an Argon run.
- Method lessons below. Task findings come only from earlier runs of the **same** brain (section 6).

## Tools

The backticked names below are the actor's attempt-1 allowlist (`--skills practice` = these plus the core tools).

| Tool | Use |
|---|---|
| `env_open`, `env_starts`, `env_reset`, `env_close` | Session and episode. Pass the run's `brain` and `run_id` to every env call. |
| `env_observe` | Robot state, F/T in the base frame, gripper width and time. Free: it does not advance time or count as a command. |
| `env_observe_image` | The visual coordinate harness, one composite PNG (details in Approach §2). Free: no time, no command. |
| `env_move` | Absolute TCP target `pos` and `quat` (w, x, y, z) in the FR3 base frame, with `speed`, `force_max` (capped at 15 N in code) and `stop_force` for a guarded move. |
| `env_gripper`, `env_wait` | Open, close or set the width; hold the current targets. |
| `env_status`, `env_finish` | Budget used. End the episode with `claim: success / failure`. |

Not for the actor (written without backticks so they stay off the actor's allowlist):

| Tool | Who | Use |
|---|---|---|
| practice_manifest | runner | Freeze the run before attempt 1. `verify: true` re-hashes it later. |
| env_charge_tokens | runner | Charges each turn's tokens against the frozen per-episode token budget. |
| practice_ledger | watcher | Per-episode rows, wake events, D aggregates, the watcher packet, McNemar. |
| practice_clip | watcher, UI | An H.264 clip of a finished practice episode with the same overlays. |
| record_check, publish_media, record_event | watcher | D checks are recorded by the watcher, never by the actor. |
| task_state, task_approach, task_move_until_contact, task_search, task_insert, task_release | actor, attempt N only | Stage C task tools (`task/v1`). The manifest's `tools` list names them explicitly. |

Code: `harness/tools/practice.py`, `twin/practice_view.py`; the env and evaluator are in the `sim-integrity` skill.

## Approach

### 1. Freeze the run before attempt 1

Call `practice_manifest` once per run, before any episode. Example (the stand-in dry run):

```
practice_manifest {brain: claude, run_id: run-standin-practice-1, attempt: baseline, model: claude-opus-5.5,
                   start_set: practice, parts: [round peg 1, round peg 4, square peg 3], per_part: 2}
```

It writes `runs/<brain>/<run_id>/manifest.json` and its sha256, and refuses to run twice or after an episode exists. It freezes:

- the start ids, in order;
- the brain, model and thinking level;
- the tool allowlist (its declarations' hash);
- every skill's SKILL.md hash;
- the evaluator version and code hash, and the env, monitor and renderer code;
- validation.yaml's version;
- the per-episode budget, which it also writes into `env/budget.json`.

It copies the frozen start sets, cousins and envelope byte-identical from `source_run`, so every paired run attempts the same starts. A watcher runs `practice_manifest {verify: true}` before scoring: any change in evaluator, code, sets or budget means a new run.

**Freshness.** The runner calls `practice_manifest {require_fresh: true}` before attempt 1 and at every session start. A scored run refuses to continue if any frozen hash changed: freeze a new run_id, which re-hashes the current code. `dry_run: true` (at creation, or `mark_dry_run` later, one way only) marks a harness test. It may run on changed code but is never scored, and the ledger reports it `scorable: false`.

**The frozen per-episode budget (attempt 1 and attempt N alike):**

| Budget | Value | Why |
|---|---|---|
| brain calls (model turns) | 80 | LIBERO-10 capped Argon at 80 calls. It hit a limit in 41/100 episodes, averaging about 50 calls on two-subtask tasks [llm-libero-2026]. One in-hand insertion is one subtask, so 80 gives it twice LIBERO's per-subtask allowance, and a limit hit means the policy did not converge, not that the cap was tight. It is also 4× Google's fine-servoing range of 8–20 calls [google-argon-robotics-2026] and 4× Robocurve's 20-call budget [robocurve-astra-2026]. |
| env commands | 160 | Two per call: a move and a wait, or a gripper action. |
| sim time | 120 s | The env default. Real teleop insert segments have a median of 11.5–19.6 s, and the slowest real successful insertion took 27.0 s (`sim-integrity` envelope), so 120 s is 4.4× the slowest. |
| tokens | 12M | A chained run re-reads its context each turn. With a ~25k-token prefix and ~2.5k tokens per turn (a JSON result plus one composite image), 80 turns cost about 25k·80 + 2.5k·80²/2 ≈ 10.1M, so tokens should not bind before calls do. Most of this is cached input at 5% of the price. |
| wall time | 3,600 s | Physics is paused while the brain thinks. 80 calls × ~18.6 s (Astra's per-call latency [astranomy-2026]) + ~2 s of sim and render ≈ 1,650 s, so this cap only guards against hangs. |

- **Enforcement.** The env ends an episode `limit_reached` on commands, sim time, wall time or tokens. Tokens reach the env through the runner's `env_charge_tokens`.
- **Brain calls have no env op.** The runner (or the stand-in) stops at 80 and calls `env_finish` with no claim. The ledger marks the episode `limit_hit: brain_calls`, so it is counted apart from failures.
- **Expected cost per episode.** At Argon's introductory prices (brains skill), roughly $1–3. LIBERO-10 reported about $7–8 per Argon episode at high effort [llm-libero-2026].

### 2. What the actor sees: the visual coordinate harness

`env_observe_image {session, views?, zoom?}` returns one composite PNG. The panels are the wrist camera (`hand`, native 640×480) and the fixed cameras `hama1` and `hama2`, each cropped around the believed hole and resampled by `zoom` (default 3). A text header sits on top. The rules, which follow Google's zero-shot arm prompt, are:

- **Pixels are the cameras.** They show the twin's true state from the calibrated cameras at native resolution. Crops are resampled, so they add no detail a real 640×480 frame lacks.
- **The render cameras carry the calibration's measured error:** 1.7 px (hama1), 1.0 px (hama2) and 4.9 px (wrist). The error is fixed per run, so overlays sit off the pixels by as much as they would on the real robot. The wrist error of ~5 px is about 1 mm at 15 cm.
- **Overlays are beliefs** from the task the actor was given, never sim truth:
  - yellow: a 1 cm world grid at the believed plate top, with border labels X and Y in world cm (Google: "World (X, Y) in centimetres (2 cm grid at `grid_z`)");
  - green: the believed hole outline (section plus clearance) at the entry plane, its centre, and the hole axis drawn 60 mm out of the hole;
  - magenta: the believed part axis from the pad centre to the tip under the nominal grasp, the tip, and the tip's foot on the grid plane;
  - cyan: the pinch centre (+) and the jaw closing line [ ] at the current opening (Google's reticle).
- **The header** gives the believed tip in the believed hole frame (x, y lateral; depth, + into the hole), the part-axis tilt, the TCP, F in the base frame (+z means pressing down) and the gripper width.
- **Not modelled:** sensor noise, motion blur and lighting differences. The images are cleaner than real frames.

How it works: the env has no render op, so the tool keeps a shadow copy of the episode (`twin/practice_view.py`). The shadow is the same frozen start and code, replays the env_move / env_gripper / env_wait calls logged in `runs/calls/`, and is checked against the live env (command count and observation). MuJoCo is deterministic here: replays were bit-identical. The shadow never records, scores or reports outcomes.

- An image costs the sim time since the last image (the shadow simulates it again) plus about 1 s of rendering per view. On a loaded machine (load average 66 on 16 cores) that came to 4–58 s per image.
- If commands reached the env around the logged env_* tools (task_* tools drive the session socket directly), the tool refuses to render rather than show a stale state.
- `overlays: false` shows the pixels alone; the part overlay is drawn only while the gripper width says the part is held.

### 3. Episode protocol (the actor)

1. **Reset** at the next start id from the manifest (`env_reset`). The task arrives with the first observation.
2. **Observe** with `env_observe` and `env_observe_image` before the first move, and again whenever the state may have changed in a way the numbers do not show.
3. **Act** with absolute TCP targets (`env_move`), the gripper and waits, deciding after every result. Guidance from the published recipe, not a script:
   - Absolute end-effector targets beat deltas and joints (72.7 vs 66.0 and 48.7 [astranomy-2026]).
   - Keep the full history: the server-side chain for Argon, the conversation for the stand-in. Full history scored 100, images only 56.5 [astranomy-2026].
   - Calibrated geometry is in the task and the images. To put the believed tip at P with the hand axis on the hole axis, command TCP = P − R_hand · `tip_in_tcp_frame`.
   - Pick the call granularity on turn 1 (Google): coarse (2–4 calls), hover and verify (4–7) or fine servoing (8–20).
   - Near contact, use guarded moves (`stop_force`) and a modest `force_max`. The ceiling is enforced in code [forge-plus-2026].
   - Find out where the hole really is from contact and images. The task says the true board differs by about 1–4 mm, and the in-hand pose is not measured.
4. **Finish.** Release, retreat if you choose, then `env_finish` with your claim. The env holds the last command for 0.5 s and scores from simulator state. Practice starts return the outcome; held-out starts return nothing. Claim success only on evidence you can name, such as depth reached, a force signature or what the images show. False claims are counted.
5. **Next.** Close the session after the last start. A reset in the middle of an episode records it as `abandoned`.

Attempt 1 never uses task tools, the skill library, or task findings from other brains' runs.

### 4. The stage rubric (frozen evaluator, `twin/evaluator.py`; `sim-integrity` §2)

| Stage | Reached when |
|---|---|
| no_approach | none of the below |
| contact | a finger or the hand touches the part |
| lifted | both fingers hold it, its tip is ≥ 1 mm above the surface and it touches nothing else (in-hand starts begin here) |
| above_target | the tip is within 2 mm of the TRUE hole axis and ≤ 20 mm above the surface |
| seated | depth ≥ seat depth − 1 mm, lateral ≤ clearance + 0.2 mm, tilt ≤ 3° |

- **Outcomes:**
  - success: seated, released and at rest;
  - failures: seated_not_released, not_seated and dropped;
  - apart from failures: limit_reached, stalled, abandoned and error.
- **Counted:** a success counts only with no invalidating monitor flag and a dt/2 rerun that agrees.
- **False claims:** `false_claim` means the actor claimed success and the raw outcome was not success.
- **Reporting:** every episode reports its highest stage, so a stall shows where it happened [robocurve-astra-2026].

### 5. The watcher: a code monitor always on, an LLM only on events

**The code monitor is always on.** It is the env's monitors plus the wake rules in `practice_ledger`:
- env monitors: integrity, plausibility, `progress.stuck` (60 sim s or 40 commands), `progress.repeat` and the budget;
- **stuck:** no rubric progress for 20 sim s or 12 commands;
- **anomaly:** any monitor flag, a peak force ≥ 0.8 × the 15 N cap, or a tool error;
- **near_promotion:** 3 counted successes in a row on one part (hand to the compile step);
- **episode_end.**

**A separate LLM watcher is woken only on those events.** Periodic review is too costly: NavGPT-3's checks every 16 steps took tokens from 179k to 2.32M per episode for +6 SR [navgpt3-2026], while FORGE-plus calls its LLM only before execution and on failure [forge-plus-2026].
- **It sees** the watcher packet (`practice_ledger {episode: <start id>}`): tool calls and results, images and the episode record. It **never sees** the actor's reasoning. The packet drops `log_note` and `log_decision` text, and thoughts are never in the call log.
- **It may:**
  - end or invalidate an episode (`env_close` records it abandoned, with a `record_event` decision saying why);
  - pick the next start, from the manifest's list only;
  - raise a physics question for F (`record_event`, stage F: what looked wrong, with the episode file);
  - write the post-episode critique to `runs/<brain>/<run_id>/practice/critique/<start>.md`.
- **It never scores.** The evaluator scores. The watcher measures with `practice_ledger` and records the D checks with `record_check`. The actor never records D checks.
- **Its tokens count** in `D.tokens_per_success`. Add them per episode to `practice/tokens.json` as `watcher_tokens`.
- **Held-out runs:** the watcher sees no held-out outcome until every held-out start has been attempted.

### 6. Paired starts, held-out discipline, statistics

- **Same starts, same budget, same evaluator hash.** Baseline and attempt N use the same start ids, frozen in each manifest (`paired_with` names the partner run). Never add starts after looking at results.
- **Practice uses practice starts only** (`P-*`). Held-out starts (`H-*`) are attempted once per run, at the end, in a run whose manifest says `start_set: heldout`. Use the first 29 per part (0 failures allowed for ≥ 0.9) or all 46 (1 failure allowed). The env refuses a second attempt. The watcher never hands held-out outcomes back to the actor.
- **Rates** come with Clopper–Pearson 95% intervals; a rate passes a target only if its lower bound clears it.
- **Paired comparison:** an exact McNemar test on the discordant pairs (`practice_ledger {paired_with}`), as Astranomy did [astranomy-2026].

### 7. Two kinds of knowledge

- **Method lessons go to this skill's Lessons** and are given to every brain. Examples: how a tool behaves, a protocol step that saves calls, a convention that confused, a harness bug and its workaround.
- **Task findings go to `runs/<brain>/<run_id>/practice/findings.md`** and are never handed to another brain's run. Examples: "the true hole sat 2 mm toward −y on these starts", a force threshold that worked for round peg 4, a part's in-hand slip, a strategy that seated square pegs.
- When unsure, it is a task finding. This keeps "Argon worked it out" true (orchestrator.md).

### 8. Costs and time

- **Tokens.** The Gemini runner records them per turn (`runs/loop/<run-id>/turns.jsonl`). A Claude stand-in cannot measure its own: it estimates per decision, says so, and writes `practice/tokens.json` as `{start_id: {tokens, decisions, source, watcher_tokens}}`.
- **Time is reported two ways:**
  - **paused-physics wall:** what the episode took, with inference included;
  - **real-time equivalent:** sim s plus think s (wall minus tool time), the time a real arm would take if it held still while the brain thinks.
- **Brain share** is 1.0 for attempt 1, since every command is a brain decision. Attempt N reports the share of episode time under routine calls.
- **Notes** go to the timeline with `record_event(kind="note", stage="D", brain=...)`.

## Checks

The watcher runs these, never the actor:

| Check | How | Pass |
|---|---|---|
| frozen | `practice_manifest {verify: true}` before scoring | `ok: true` |
| `D.success_rate` | held-out run, `practice_ledger` → counted successes / attempted, Clopper–Pearson lower bound | ≥ 0.9 (validation.yaml) |
| `D.tokens_per_success`, `D.calls_per_success`, `D.time_per_success_s` (both modes), `D.false_done_rate`, `D.brain_share` | `practice_ledger` aggregates, with actor and watcher tokens | track |
| limit hits apart | `aggregates.limit_reached` reported next to the success rate, never inside the failures | reported |
| paired | `practice_ledger {paired_with}` → shared starts, discordant counts, exact McNemar p | reported |
| integrity | no counted success with an invalidating flag (`G.exploit_flags`); no `H-*` start in any practice session | 0, 0 |

A stage-D result is a twin number. It says something about the real arm only for part families whose B2 gate passes.

## Lessons

**2026-10-25, Claude stand-in dry run of the attempt-1 baseline** (caller `standin-practice`; `runs/claude/run-standin-practice-1/`, `practice/ledger.json`).

This was a harness test, never scored: the manifest is marked `dry_run`, and the audit changed frozen files after the freeze. Stand-in results say nothing about Argon (Claude as a direct policy scored 25.3 vs Astra's 72.7 [astranomy-2026]).

| Start | Rubric | Outcome | Decisions | Env commands | Sim s | Wall s | Claim |
|---|---|---|---|---|---|---|---|
| round peg 1 · 000 | above_target | not_seated | 19 | 30 | 63 | 579 | failure |
| round peg 1 · 001 | above_target | not_seated | 8 | 9 | 22 | 193 | failure |
| round peg 4 · 000 | seated | success_flagged (harness bug) | 17 | 31 | 77 | 1115 | success |
| round peg 4 · 001 | seated | success, counted | 9 | 18 | 44 | 464 | success |
| square peg 3 · 000 | seated | success, counted | 6 | 9 | 27 | 294 | success |
| square peg 3 · 001 | seated | success, counted | 8 | 19 | 53 | 345 | success |

- **Totals:** 3/6 counted (Clopper–Pearson 95% interval 0.12–0.88), 0 limit hits, 0/4 false done, 22.3 decisions per success.
- **Time per success:** 997 s paused-physics wall, 505 s real-time equivalent, 95 s sim.
- **Tokens (estimated, not measured):** about 29k new tokens per success. A fresh chained brain would re-read 170k–730k tokens per episode.
- **Envelope:** every success violates the real envelope as too slow (approach 3–7 mm/s vs the real 1st percentile of 10–12 mm/s).

Method lessons:
- **Measure the grasp before inserting.** One light touch on flat plate away from the hole (stop at 1.5 N, 3 mm/s) gives two numbers:
  - the tip's height below the TCP, which varied 10.8–22.3 mm across starts against the nominal 18.1;
  - the pressure centre from the wrench: r_y = (T_x + r_z F_y) / F_z and r_x = (r_z F_x − T_y) / F_z, after subtracting the free-air reading, since the wrench includes the part's weight.
- **Depth from travel.** TCP travel past that plate touch is the depth; "seated" is about 20 mm of travel, bottomed under 2–6 N. Every success claim made on this evidence was right (4/4).
- **The pressure centre locates the low edge, not the axis.** A slightly tilted flat-bottomed part loads its lowest edge. A rim contact sits about one half-width from the axis, on the side away from the hole.
- **Look first.** Moving the part 20 mm aside, on the far side of the hole from the wrist camera, and reading the hole against the grid gave 0.5–1.8 mm error, checked post hoc after all episodes. That is enough for 16 mm and 8×12 mm parts within one to three probes. It is not enough for 4 mm parts (~0.45 mm capture), where blind probing stalled.
- **Decide between probes.** A batch of probes that does not check each result undid a success: the second probe pulled the inserted part back out. This matches URAI's 56% vs 24% for pre-written programs [urai-2026].
- **Never close the gripper empty.** It drives the fingers into each other, and the monitor invalidates the episode (finger–finger penetration). To press on a released part, use `env_gripper {action: width, width: 0.004}`.
- **A guarded move overshoots its stop force.** It needs 3 ticks above threshold: 1.5 N stops peaked at 5.6–10.9 N at 3–5 mm/s on stiff contact. Approach at ≤ 3 mm/s.
- **The TCP sags 1–1.5 mm under the part's weight** (0.4–0.8 N, soft impedance). `reached` checks position only (0.5 mm), not orientation.
- **Images cost time.** Each image re-simulates since the last one. Ask for one when the numbers cannot answer: before the first approach, and when a contact makes no sense.


## Sources

Ids are in `docs/knowledge/references.md`.

- [astranomy-2026] Agentic recipe: absolute EEF, full history, calibrated geometry; McNemar on paired episodes; per-call latency.
- [google-argon-robotics-2026] Visual coordinate harness: 2 cm world grid, jaw reticle, call-granularity choice on turn 1.
- [llm-libero-2026] Argon 43% on LIBERO-10; 41/100 episodes hit the 80-call limit; done without success in 287/1,800.
- [robocurve-astra-2026] Stage rubric; 20-call budget; puzzle-in-groove 2/20 with 8 stalled above the groove.
- [navgpt3-2026] Periodic review cost (179k → 2.32M tokens) and the case for event-triggered review.
- [forge-plus-2026] Force ceilings in code; the LLM called before execution and on failure only.
- [hivla-study-2026] Success-detector termination; false positives hurt most.
- [urai-2026] Deciding after each call beats a pre-written program with the same tools (56% vs 24%).
- docs/knowledge/evaluation-and-statistics.md: Clopper–Pearson counts, McNemar, never add trials after looking.
