# Bot18: footage → digital twin → robot policy

A 1-day hackathon build on Google's **Gemini 4 Argon**. Argon reads real robot footage, rebuilds the scene as a MuJoCo simulation (a "digital twin"), writes the success checks, then practises the task inside the twin until it is reliable. The work it has figured out gets compiled into fast routines, so each later attempt costs fewer tokens and less time.

> **Status, 2026-10-25 12:30 PDT.** Everything below is **sim only** and **provisional**. No real arm has confirmed any result. The Argon API key is not set yet, so every result so far comes from **Claude agents standing in for Argon** (the "Claude stand-in" brain). When Argon runs, its skill library is wiped first, so its improvement is its own. Claude and Argon numbers are never merged.

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

As of 2026-10-25 12:30 PDT. The full running record, with every decision and its reasoning, is [`docs/design/progress-log.md`](docs/design/progress-log.md).

### The overnight run

The overnight run had one mandate from the user: get a **fully connected loop** (read → twin → tools → practice → compile → practice again, cheaper) running as fast as possible, with Claude subagents as the brain and a separate watcher agent doing all the scoring.

**Interruption:** the machine restarted at 08:13. All 7 agents had stopped mid-task at about 02:56, so roughly 5 hours were lost. They were relaunched at 08:24 from handover packs.

### Where the loop stands

Every step of the loop now exists and has run. Stage A passes. B2, the twin matching reality, is the step that keeps results provisional, and the first "practice again, cheaper" comparison is frozen and ready to run.

```
 A read ✓ ─► B0 ◐ ─► B1 ✓ ─► B2 ✕ ─► C ◐ ─► D ◐ ─► E ◐ ─► D again: pilot frozen, about to run
```

### Checks per stage (Claude stand-in, REASSEMBLE track)

Checks and targets are frozen in [`playbook/validation.yaml`](playbook/validation.yaml): v15 now, with v16 being written from today's user decisions (below). Only the watcher records them.

| Stage | Checks passing | Key numbers | Notes |
|---|---|---|---|
| **A** Read footage | **3/3 ✓** | Segment F1 **0.992** (321 of 326 labelled); success balanced accuracy **0.947** (326 attempts, 30 real failures, 27 caught); claims grounded **0.947** (54 of 57, all checked); ~21k tokens per recording minute (tracked) | Blind reads of 9 gate episodes with the refined skill, scored once together as declared in advance. The refined skill names the part from *which hole* it enters and judges the outcome from end-state physics. The old skill scored 0.961 / 0.78 / 0.667 on 3 other gate episodes. |
| **B0** World frame | 5/6 | Static cameras 1.4–1.7 px, two-view 0.17 px, tool tip 0.14 mm, board 2.37 mm | Episode 0 passes 6/6 (wrist 5.82 px). Episode 1, re-run from scratch, fails on the wrist camera (26 px) because too few board points are visible. The latest record shows as a regression. |
| **B1** Parts and contact | 4/4 ✓ | Settle penetration 0.038 mm; aligned seat 100%, misaligned jam 100%, dt/2 agreement 100% | Pegs only. Still passes with every new twin model. |
| **B2** Replay vs reality | 1/6 | Free-space tracking **0.62 mm** ✓. Train cross-validation, seat rate vs real **78%**: v3 54% → v5b 63% → **v6 69%**, error −0.084 (needs within 0.10). Force KS **0.211** (needs ≤ 0.20) | **The main blocker.** A diagnosis showed the old negative control can't be met by any faithful replay. The user changed it to a board-shift control (v16), and v6 passes that on train (+0.319). Before any gate look: an independent audit checks that v6's per-attempt board shift doesn't encode the real outcome, and force KS is still just over target (round pegs 0.277). |
| **C** Task tools | 1/2 (v15 rule) | Full tool chain seats **22/24** practice starts with exact information and **10/24** with believed information, vs 9/24 and 1/24 for one raw descent. No interface regressions | Fails as v15 defines it: the minimum over tools, and search ran only 2 times (0/2). v16 gates the chain instead, on a sample fixed in advance. |
| **D** Practice | not scored yet | First dry run counted 3 of 6. **Pilot comparison frozen:** arm 1 (raw-pose baseline) vs arm N (task tools + compiled routine) on 24 fresh practice starts, 180 s budget | Practice starts only, never scored. Being re-frozen under v16, then two fresh stand-ins run it. |
| **E** Compile | not scored yet | Routine `peg_insert_guarded` v1, zero brain calls: **5/24** practice starts; **16/24** when given the true hole position; each success 24–47 env commands and 25–34 s, inside the real envelope | Pilot verdict "not ready". Its misses had the hole 4.5–26 mm from the first aim, so finding the hole stays a brain decision and the routine does the rest. Arm N tests exactly that split. |
| **G** Integrity | 3/5 | Red-team catch rate **1.0** (15/15 planted cheats caught); sandbox escapes **0**; rule violations **0**; exam reads **4** ✗ | Exam episodes 9 and 10 leaked through a listing tool before a guard existed, so they're burned for the Claude brain only and stay clean for Argon. The leak is fixed at the root. |
| **T2** SO-101 track | 1/5, parked | Frozen gate run: keypoints 7.44 px ✗, two-view 5.1 px ✗, grasp-to-brick 16.5 mm ✗, replay 4/15 ✗, shifted-brick control 0/9 ✓ | The tools work on train (cameras 3.8–6.4 px, brick start 2.0 mm). The gate fails because the episode-number split put gate bricks 35° of pan beyond any train start. A fix can only be shown on new data. |

### User decisions today (validation v16)

1. **B2's negative control** becomes a board-shift control: a twin whose board is displaced beyond the measured board error must do clearly worse, by at least 0.10 in seat-rate error. B2 replay scoring judges "seated" on insertion depth only.
   - Reason: a released part seats in 93.5% of feasible replays, including 39 of 69 real failures (mostly operators giving up). So a deliberately loose twin always tied, and the old control could never pass.
2. **C** gates the full tool chain's success, on a sample size fixed in advance. Per-tool rates are tracked and count only with at least 10 trials.
3. **Scored runs** get a 180 s sim-time budget for both arms, because the env's force-capped stops make the same commands about 1.5× slower.
4. **E's promotion gate:**
   - lists all 6 envelope features it applies;
   - excludes episodes handed back to the brain from the routine's envelope;
   - envelope-checks cousin twins.

### Running now

| Agent | Working on |
|---|---|
| Validation v16 + pilot re-freeze | Writes the four decisions into `validation.yaml`, then re-freezes both pilot arms under v16 |
| v6 leak audit | An independent check that v6's per-attempt board registration uses no data from after first contact. Includes a before-contact-only rerun and a permutation test. |

Finished since the restart:
- **Environment fixes.** Side effect: force-guarded moves are slower now, so brains guard only the last 3–5 mm.
- **SO-101.**
- **Blind re-test and the 6-episode A confirmation.**
- **Task tools.**
- **Compile.**
- **B2 rounds 5 and 6, and the B2 diagnosis.**
- **Pilot setup.**
- **A page fix:** steps with work but no scores show "◐ not scored yet".

### Next, in order

1. **Run the pilot comparison:** two fresh Claude stand-ins, arm 1 and arm N, on the same 24 practice starts. The watcher then compares success, brain calls, tokens and time per success.
2. **B2:**
   1. finish the audit;
   2. one more train round on round-peg force spread;
   3. the watcher scores gate once under v16.
3. **Scored runs on frozen held-out starts:** attempt 1 vs attempt N (29 per part), then promote the routine through the E gate and rerun with zero brain calls.
4. **Then:** gears and the nut via MuJoCo SDF collision, the footage-only track, and Argon once its key is set.

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
  - **The common finding:** 60 of the 72 real seats the compliant twin still misses arrive more than 1 mm off the hole, against 0.05–0.25 mm clearance. The replayed motion drifts before release, and no contact or grasp setting can fix that.
  - **The diagnosis:**
    - Missed seats arrive 2.33 mm off the hole at release; seated ones 0.39 mm.
    - The main causes: the part slowly creeping in the simulated grasp (~1°/s, a MuJoCo artefact), and board registrations that clipped on 82 of 308 attempts.
    - It also found the old negative control unreachable. Once released in a feasible spot, the twin seats 93.5% of attempts, including real failures that were operators giving up.
  - **Round 6 (v6),** a consistent grasp:
    - a no-slip solver that stops the creep;
    - parts placed from the twin's own kinematics;
    - limit-surface pad torsion;
    - a per-attempt board shift capped at the measured board error.

    It seats 69% (error −0.084), passes the new board-shift control on train, and keeps every B1 probe passing. Force spread (KS 0.211) is still just over target.
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
