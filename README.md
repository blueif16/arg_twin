# Bot18: footage → digital twin → robot policy

A 1-day hackathon build on Google's **Gemini 4 Argon**. Argon reads real robot footage, rebuilds the scene as a MuJoCo simulation (a "digital twin"), writes the success checks, then practises the task inside the twin until it is reliable. The work it has figured out gets compiled into fast routines, so each later attempt costs fewer tokens and less time.

> **Status, 2026-10-25 09:45 PDT.** Everything below is **sim only** and **provisional**. No real arm has confirmed any result. The Argon API key is not set yet, so every result so far comes from **Claude agents standing in for Argon** (the "Claude stand-in" brain). When Argon runs, its skill library is wiped first, so its improvement is its own. Claude and Argon numbers are never merged.

Contents: [1. The plan](#1-the-plan) · [2. Status](#2-status) · [3. What's hard](#3-whats-hard-so-far) · [4. How results are kept honest](#4-how-results-are-kept-honest) · [5. Data tracks](#5-data-tracks) · [6. Repo map and reading order](#6-repo-map-and-reading-order) · [7. Running it](#7-running-it)

---

## 1. The plan

Source: [`docs/design/vision-draft.md`](docs/design/vision-draft.md). It is still a **proposal, not agreed**, except the two parts marked *agreed* below. The original test plan (tests T0–T7, rules D1–D5) is [`docs/test-plan-v0.md`](docs/test-plan-v0.md).

### The claim

Argon turns real robot footage into a simulated world shaped around the task, then teaches itself the task inside that world. With practice, work moves from slow reasoning into fast routines, so each attempt is more reliable and cheaper. The "policy" is Argon plus its harness: tools, compiled skills and a watcher. Small trained models are a later, optional experiment.

### The loop

```
 A read footage ─► B build twin ─► C task tools ─► D practice ─► E compile ─► D again, cheaper
                      ▲                                │
                      └──────── F physics questions ◄──┘
```

| Step | What happens | Done when |
|---|---|---|
| **A. Read footage** | Argon reads long recordings, zooms in at contact moments, and writes hierarchical labels, a progress score, strategy notes (how operators tilt, wiggle, twist) and a failure taxonomy. It also decides what the twin must get right: insertions fail most. | Segments, success calls and claims match the dataset's own labels |
| **B0. World frame** | Cameras calibrated with the robot as the calibration object, the board frame, the tool-tip offset | Overlays line up on every camera within a few pixels |
| **B1. Parts and contact** | Each part gets MuJoCo task frames (entry, axis, clearance, grasp), success checks and contact parameters | Aligned parts seat, misaligned parts jam, results hold when the timestep is halved |
| **B2. Replay against reality** | Replay real attempts, failures included, in the twin | The twin's seat rate and force spread match reality across held-out recordings, and a deliberately loose twin does worse |
| **C. Task tools** | Tools in task terms, e.g. `insert(part, hole, max_force)`, built from the twin's frames. They use only *believed* geometry (what perception would give), never sim truth. | Each tool succeeds on randomized practice starts |
| **D. Practice (System 2)** | Argon decides only at decision points; code runs the fast force loop. Runs happen with physics paused while it thinks, and in real time. | Measured on frozen starts against the baseline |
| **E. Compile (System 2 → 1)** | A decision pattern that keeps working becomes a sandboxed routine that runs with zero brain calls. Control returns to Argon when the routine meets something new. | It passes held-out starts, randomized "cousin" twins and the real-data force/speed envelope |
| **F. Feedback** | If a behaviour relies on physics the replays never checked, that becomes a reconstruction request back to B | — |

### What we want to prove

1. **The twin is faithful** (*agreed 2026-10-24*). Given only what the log lacks (the part's pose in the gripper, the stiffness setting), the twin seats the real successes and jams the real jams with matching force, and a permissive twin can't. Predicting outcomes open-loop is a stretch goal: the log leaves the in-hand pose about 6 mm uncertain against 0.05–0.25 mm clearance.
2. **Argon understands the crux.** The rules it states predict which held-out real attempts fail.
3. **Self-improvement against a strong baseline** (*agreed 2026-10-24*). Attempt 1 is the strongest published recipe without compilation (Astranomy: absolute end-effector targets, full history, calibrated geometry in the prompt), run on the same frozen starts as attempt N, with paired tests. From attempt 1 to attempt N, success goes up, tokens and time per part go down, and Argon's share of control shrinks.
4. **Transfer.** A harness built on about 8 parts makes the other 9 cheaper from their first attempt.

### The 3-minute demo

1. The footage and the twin snap into alignment, and a real jam is reproduced in the twin.
2. Argon's attempt 1: slow, thinking at every step, stalling at the insertion.
3. Attempt N: fast and routine-driven. A "who's in control" timeline shows Argon appearing only when something surprises it.
4. An unseen part: Argon steps back in, then fades out faster.

### How it's built

- **A fixed orchestrator** runs the stages, enforces the rules and decides when each stage is done ([`playbook/orchestrator.md`](playbook/orchestrator.md)).
- **The playbook is the main asset.** Each step is a skill file stating its goal, tools, rules, definition of done and dated lessons. Skills give goals and checks, not recipes ([`playbook/`](playbook/)).
- **A thin, deterministic harness**: tool registry, simulator, frozen evaluator, call recorder ([`harness/`](harness/), [`twin/`](twin/)).
- **A swappable brain.** Claude agents now; Argon later runs the same skills through the same tools, and every Argon call is recorded and replayable.

---

## 2. Status

As of 2026-10-25 09:45 PDT. The full running record, with every decision and its reasoning, is [`docs/design/progress-log.md`](docs/design/progress-log.md).

### The overnight run

The overnight run had one mandate from the user: get a **fully connected loop** (read → twin → tools → practice → compile → practice again, cheaper) running as fast as possible, with Claude subagents as the brain and a separate watcher agent doing all the scoring.

**Interruption:** the machine restarted at 08:13. All 7 agents had stopped mid-task at about 02:56, so roughly 5 hours were lost. They were relaunched at 08:24 from handover packs.

### Checks per stage (Claude stand-in, REASSEMBLE track)

Checks and targets are frozen in [`playbook/validation.yaml`](playbook/validation.yaml) (v15). Only the watcher records them.

| Stage | Checks passing | Key numbers | Notes |
|---|---|---|---|
| **A** Read footage | **3/3 ✓** | Segment F1 **0.992** (321 of 326 labelled); success balanced accuracy **0.947** (326 attempts, 30 real failures, 27 caught); claims grounded **0.947** (54 of 57, all checked); ~21k tokens per recording minute (tracked) | Blind reads of 9 gate episodes with the refined skill: 3 re-test plus 6 drawn by seed, scored once together as declared in advance. The refined skill names the part from *which hole* it enters and judges the outcome from end-state physics. The old skill scored 0.961 / 0.78 / 0.667 on 3 other gate episodes. The 6 new episodes alone: 0.990 / 0.942 / 31 of 33. |
| **B0** World frame | 5/6 | Static cameras 1.4–1.7 px, two-view 0.17 px, tool tip 0.14 mm, board 2.37 mm | Episode 0 passes 6/6 (wrist 5.82 px). Episode 1 re-run from scratch: the wrist camera fails at 26 px because too few board points are visible. The latest record shows as a regression. |
| **B1** Parts and contact | 4/4 | Settle penetration 0.038 mm; aligned seat 100%, misaligned jam 100%, dt/2 agreement 100% | Pegs only |
| **B2** Replay vs reality | 1/6 | Free-space tracking **0.62 mm** ✓. Seat rate vs real **78%** (needs within 10 points), train cross-validation: v3 **54%**; grasp compliance (v5b) **63%**; softer board contact (v5a) **58%**, force KS 0.20 | **The main blocker.** Neither round-5 fix passes the negative control: a deliberately loose twin is still as close to reality. So nothing went to gate. Most missed seats arrive more than 1 mm off the hole before release (see §3). A diagnosis agent is tracing why. |
| **C** Task tools | not scored yet | Tools built; self-test running | |
| **D** Practice | not scored yet | First end-to-end dry run: 3 of 6 practice starts counted | Dry run, never scored. All four harness fixes from it are done: false flag, fast rendering, force stops within 0.31 N, budget stop. Red team still 0 counted exploits. |
| **E** Compile | not scored yet | First routine (`peg_insert_guarded`, zero brain calls): round peg 4 seated 3/3 on practice starts, but too slow for the real-data speed envelope | Pilot |
| **G** Integrity | 3/5 | Red-team catch rate **1.0** (15/15 planted cheats caught); sandbox escapes **0**; rule violations **0**; exam reads **4** ✗ | Exam episodes 9 and 10 leaked through a listing tool before a guard existed, so they're burned for the Claude brain only. They stay clean for Argon. The leak is fixed at the root. |
| **T2** SO-101 track | 1/5, parked | Watcher, frozen gate run (the result of record): keypoints **7.44 px** (side 2.5, overhead 14.9) ✗; two-view 5.1 px ✗; grasp-to-brick **16.5 mm** ✗; replay **4/15** bricks in the box ✗; shifted-brick control 0/9 ✓ | On train the tools work: cameras 3.8–6.4 px, brick start 2.0 mm. The gate fails because the split cut by episode number puts gate bricks 35° of pan beyond any train start, where the joint map is ~2 cm off. All 50 episodes are used now, so a fix can only be shown on new data. |

### Running now

| Agent | Working on |
|---|---|
| B2 diagnosis | Split the >1 mm arrival error into board pose, hand drift and in-hand pose; measure how much of the seat-rate gap each one explains with "make this one source perfect" replays; propose round-6 fixes from the knowledge base, or show the gate can't be met by open-loop replay |
| Task tools (C) | Rerun the self-test on the fixed env (exact vs believed info), handle the 5.5–7.1° in-hand tilt on round peg 1, log every env command |
| Compile (E) | Fix the routine's envelope violations on the fixed env, cousins pilot |

Finished since the restart:
- **Environment fixes.** Side effect: force-guarded moves are slower now, so brains guard only the last 3–5 mm.
- **SO-101.**
- **The blind re-test and the 6-episode A confirmation.** Stage A now passes firmly.
- **B2 round 5:** both fixes, global contact (5a) and grasp compliance (5b). Neither is selectable.
- **A page fix:** steps with work but no scores now show "◐ not scored yet" on the flowchart.

**B2 selection rule, fixed in advance:** the formulation with the lower train-CV seat-rate error wins, provided it also passes the loose-twin negative control. The watcher then scores gate once with the winner.

### Next, in order

1. The watcher scores C tool success when task tools reports, and B2 gate once a formulation passes the train negative control.
2. **Attempt 1, the baseline**: a stand-in on frozen held-out starts (29 per part, enough for a ≥ 0.9 claim) with a frozen budget.
3. **Attempt N** on the same starts, with task tools and practice. Measures tokens, calls and time per success.
4. **Compile**: promote the best routine through the full gate, then rerun with zero brain calls.
5. Then: gears and the nut via MuJoCo SDF collision, SO-101 replay, the footage-only track.

---

## 3. What's hard so far

- **The twin is too strict (B2).** Real operators seat 78% of attempts, and the twin seats only 54% of the same attempts replayed. Tellingly, a *looser* twin (holes +1 mm per side, low friction) is closer to reality. That suggests reality has self-alignment the twin lacks, not bigger holes. The evidence points at compliance in the grasp:
  - the tool tilts 3–5° at release
  - the twin jams at ~1° where geometry allows ~0.64°
  - parts slide ~3.5 mm along the pads

  What's been tried:
  - **Round 4, a force-matched particle filter on the GPU:** made things worse. The rig seated 8–15% of its worlds against ~50% for the CPU replay.
  - **Round 5, two fixes in parallel:**
    - **Grasp compliance:** spring hinges on the fingers, stiffness from measured pad compression. Seat rate went 54% → 63%.
    - **Global contact settings:** friction and wider holes top out at 64%; only softer contact goes further. The one setting that "passed" every number did so by sinking pegs up to 15 mm into the hole bottom, so it was rejected.
  - **The common finding:** 60 of the 72 real seats the compliant twin still misses arrive more than 1 mm off the hole, against 0.05–0.25 mm clearance. The replayed motion drifts before release, and no contact or grasp setting can fix that. Next is to find where that drift comes from: board pose, tracking or in-hand pose.
- **The log hides the in-hand pose.** The recording leaves the part's pose in the gripper about 6 mm uncertain, against 0.05–0.25 mm clearance. That's why the agreed B2 gate is distribution-level (seat rate and force spread) rather than per-attempt prediction, which plateaued at ~0.65 accuracy.
- **The wrist camera needs evidence.** Calibration works when enough board points are in view (5.8 px on episode 0) and fails on thin evidence (17–26 px on episode 1).
- **Reading footage:** naming the part from the hole it goes into, and judging success from the end state, fixed most errors. The skill had misled the reader on connector grip widths and on "no nut travel means failure". Reading costs about 14–21k tokens per recording minute, measured from transcript usage: 20.8k pooled over 9 episodes, 13.9k for the latest reader. The first re-test cost more because it was resumed after the restart.
- **Splits must cover the workspace (SO-101).** Cutting train and gate by episode number left the gate's brick positions outside anything train covered. The playbook now asks for a coverage check, and a joint-units step, before fitting.
- **Practice:** the brain controls through a narrow env API with force and speed caps enforced in code. The first dry run found three harness bugs: a false integrity flag, a slow image path, and guarded stops overshooting their force limit. Fixes are in progress.

Published evidence behind each decision is in [`docs/knowledge/`](docs/knowledge/). Every claim there is referenced; the originals are kept locally and not in git.

---

## 4. How results are kept honest

- **Frozen splits.** REASSEMBLE is split into train, gate and exam; SO-101 into train and gate ([`playbook/splits.json`](playbook/splits.json), [`playbook/splits_so101.json`](playbook/splits_so101.json)). Fitting uses train only, gate is scored once per frozen version, and exam is refused by the tools.
- **Actor ≠ watcher.** The agent that does the work never records its own checks. A separate watcher reruns deterministic tools and records the numbers, and the harness, not the agent, decides pass or fail.
- **No sim cheating** ([`playbook/rules.md`](playbook/rules.md)). Only parameters set before t = 0 are allowed: no welds, forces or state writes afterwards. Runtime monitors flag:
  - model changes
  - applied forces
  - penetration
  - teleports
  - energy spikes
  - outcomes that flip at half the timestep

  A flagged success doesn't count. A blind red team planted 15 cheats and all 15 were caught.
- **Capability boundary.** Actors reach the simulator only through a command API with force and speed ceilings in code. Compiled routines pass a static checker and run in a separate sandboxed process.
- **Information honesty.** Task tools use the *believed* board pose, with the measured perception error, never ground truth.
- **Statistics.** Paired baseline vs attempt N on the same frozen starts, McNemar tests, and sample-size rules: at least 29 starts per part for a ≥ 0.9 claim.
- **Records.** Every tool call is logged to `runs/calls/` and every native estimate is logged before the tool result it's compared against. Decisions go to the status timeline and the progress log.

---

## 5. Data tracks

| Track | Data | Status |
|---|---|---|
| **REASSEMBLE** (full loop) | NIST Task Board assembly by a Franka arm: 149 recordings, 30 fps, three cameras, joint states, end-effector pose, gripper and force/torque. Notes: [`docs/research/reassemble-data-notes.md`](docs/research/reassemble-data-notes.md) | Active. Pegs first; gears and the nut later via SDF collision |
| **SO-101** (stage T2) | `lerobot/svla_so101_pickplace`: 50 pick-and-place episodes, two cameras, no CAD for the objects. Tests whether the general reconstruction playbook works on a setting unlike REASSEMBLE, run by an agent with no REASSEMBLE findings | Active |
| **Footage-only** | Human or egocentric video with no robot logs | Listed, not started: hard to validate ([`docs/design/backlog.md`](docs/design/backlog.md)) |

---

## 6. Repo map and reading order

**To study the project, read in this order:**

1. This README.
2. [`docs/design/vision-draft.md`](docs/design/vision-draft.md): the plan in full.
3. [`docs/design/progress-log.md`](docs/design/progress-log.md): what was launched, what came back, what was decided.
4. [`playbook/README.md`](playbook/README.md), [`playbook/orchestrator.md`](playbook/orchestrator.md), [`playbook/rules.md`](playbook/rules.md): how agents run the loop.
5. [`playbook/validation.yaml`](playbook/validation.yaml): every check, its target and the version history.
6. [`playbook/skills/`](playbook/skills/): one skill per job. The Lessons sections hold the real numbers.
7. [`docs/knowledge/`](docs/knowledge/): the published best practice behind each decision.
8. [`docs/research/google-argon-twin-pipeline.md`](docs/research/google-argon-twin-pipeline.md): how Google's Argon demos build twins.

| Path | What |
|---|---|
| `harness/` | Tool registry and CLI. Tools live in `harness/tools/*.py`, auto-discovered; `harness/status.py` builds the status page |
| `twin/` | MuJoCo scene, replay, controller, env with its capability boundary, monitors, cousins, envelope, task tools, skill library, SO-101 modules |
| `playbook/` | Orchestrator, rules, skills, validation checks, splits |
| `docs/knowledge/` | Topic syntheses and `references.md`; per-paper study notes in `studies/` |
| `docs/research/` | Data notes, environment setup, API notes, Google pipeline breakdown |
| `docs/design/` | Vision draft, progress log, backlog, UI protocol |
| `tests/` | Sim-integrity suite, monitor sanity checks, UI tests |
| `runs/` | Run outputs: call logs, replays, footage reads, status page. **Not in git** (about 900 MB) |
| `data/`, `vendors/` | Downloaded datasets and reference repos. **Not in git** |

---

## 7. Running it

Environment: `.venv` (uv, Python 3.12) with torch 2.11 (CUDA), mujoco 3.17, mujoco-warp 3.17, lerobot 0.6.1, google-genai 2.33. Details: [`docs/research/env-setup-notes.md`](docs/research/env-setup-notes.md) and [`CLAUDE.md`](CLAUDE.md).

```sh
.venv/bin/python -m harness.cli list                                # every tool's declaration
.venv/bin/python -m harness.cli call status_summary '{}'            # every stage's state
.venv/bin/python -m harness.status                                  # rebuild runs/status/index.html
MUJOCO_GL=glfw .venv/bin/python -m twin.render                      # smoke render (WSL needs glfw)
.venv/bin/python -m tests.test_sim_integrity --quick                # integrity tests
```

The status page (`runs/status/index.html`) is built locally from the run records. It has a cover flowchart with clips, one chapter per step, and a ledger of every event and cost.
