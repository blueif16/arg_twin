# Bot18: footage → digital twin → robot policy

A 1-day hackathon build on Google's **Gemini 4 Argon** (`gemini-4-argon`).

**The story in one line.** Argon reads footage, builds a sim of the task, writes its own reward from the video, practises with retries and feedback, and compiles what works into cheap routines. A watcher scores every run and an auditor reads every trace, so the system catches its own mistakes. Everything is sim-only.

**[Presenter's guide](docs/presentation.md)**

> **Status, 2026-10-25 16:50 PDT.** Every result is **sim only**. No real arm has confirmed anything (rule D5). Argon now runs on the paid key, and Claude and Argon numbers are kept apart. Every number below is copied from the orchestrator's sheet, [`runs/orchestrator/story_numbers.md`](runs/orchestrator/story_numbers.md), or from the findings file named next to it. `runs/` is not in git, so those paths exist on the build machine only.

**Where to see it:** the status page, `runs/status/index.html`, served locally:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory /home/ran/arg_twin
# open http://localhost:8765/runs/status/index.html
```

It opens on a flowchart with clickable clips, then one page per step, scrolled sideways. Rebuild it with `.venv/bin/python -m harness.status`.

Contents: [1. The plan](#1-the-plan) · [2. Results as of today](#2-results-as-of-today) · [3. Not proven yet](#3-not-proven-yet) · [4. Next](#4-next) · [5. How results are kept honest](#5-how-results-are-kept-honest) · [6. Repo map](#6-repo-map-and-reading-order) · [7. Running it](#7-running-it)

---

## 1. The plan

Source: [`docs/design/vision-draft.md`](docs/design/vision-draft.md) (v1), including its "Agreed since v1" section. Parts marked *agreed* were decided by the user; the rest is a proposal.

### The claim

Argon learns a **general skill set** that turns footage into a working simulation and then gets better at the task inside it. With practice, work moves out of slow reasoning and into fast routines, so each attempt is more reliable and cheaper. The skills should handle any scene: a random egocentric or YouTube video, as well as a robot log.

- **Datasets are benchmarks, not targets** (*agreed*). We measure on them; we don't tune to them.
- **The demo has no named end user** (*agreed*). It shows the flow, the measured boost and the self-learning framework.

### Two self-optimization loops, one framework (*agreed*)

Both loops share the same machinery: a fixed orchestrator, shared skills with dated lessons, deterministic tools, a frozen evaluator, a watcher that never acts, an auditor, and a compile step that turns proven decisions into library code.

| | Loop 1: build the twin | Loop 2: do the task |
|---|---|---|
| **Objective** | The twin explains observations it wasn't fitted on: appearance and geometry on held-out frames, replayed motion gives the observed results, and a deliberately wrong twin scores clearly worse | Success on held-out starts within a frozen budget, plus cost per success (tokens, brain calls, time) and the brain's share of control |
| **Hard limits** | Physics gates (no penetration, stable at rest, same result at half the timestep); no leakage from after the predicted event | Integrity monitors and the real-data envelope |
| **Learning measure** | Over a stream of different videos, error and cost per new video fall | Attempt 1 (the published zero-shot protocol) vs attempt N (practice, tools, routines) on the same held-out starts; then new tasks start cheaper |
| **Ground truth** | Label-free checks only reject, never certify, so accuracy claims come from benchmarks with ground truth (HOT3D, LIBERO scene files, REASSEMBLE and SO-101 logs) | The simulator's own success check |

Loop 1 produces playgrounds for Loop 2. When a Loop 2 failure hinges on physics the twin never checked, it goes back to Loop 1 as a request.

```
 A read footage ─► B build twin ─► reward from video ─► D practice (retries + feedback) ─► E compile ─► D again, cheaper
                      ▲                                            │
                      └────────────── F physics questions ◄────────┘
        watcher scores every run · auditor reads every trace → advance / iterate / roll back
```

### Agreed since v1 (2026-10-25 afternoon)

- **Shared skills** (*agreed*, installed 16:12 as skills v2). **One shared skill per job, 10 in all,** down from 13 skills that were mostly tied to REASSEMBLE: `read-footage`, `build-twin`, `replay-validate`, `reward-from-video`, `author-tools`, `practice`, `compile-skill`, `sim-integrity`, `auditor`, `brains` ([`playbook/skills/`](playbook/skills/)).
  - Each core `SKILL.md` has a **"Choose your path"** table that branches by input type: robot logs + cameras, video only, simulator demos, prior runs.
  - The core skill is the same for REASSEMBLE, LIBERO, SO-101 and web video; only `adapters/<input-type>.md` and `references/<source>.md` differ.
  - All 44 dated lessons are kept as index entries in each core skill. v1 is archived at [`playbook/archive/skills-v1/`](playbook/archive/skills-v1/).
- **A watcher and an auditor on every run** (*agreed*). The watcher scores. The auditor reads the reasoning and action traces, starting mid-run, and recommends advance, iterate or roll back. Argon can audit too, from clip + log with a 3-vote majority. The auditor is advisory and never sets the reward.
- **Retries with observer feedback** (*agreed*). Up to 4 short trials per start. Between trials Argon reflects (status from the progress score, cause, fix, one guide line); during a trial a fast model nudges when progress stalls. Observers never pass sim truth.
- **Reward from video** (*agreed*). From one demo, Argon writes the task spec, a progress score and the success code. A gate checks it with no model calls: demo ends, starts, failed siblings, generated near-misses. One revision, then it's frozen and used as the evaluator and the feedback status.
- **LIBERO is the main demo track** (*agreed*). The full chain runs there: demo → reward → practice with feedback. The outside anchor is the published Argon attempt-1 baseline (43% on LIBERO-10). REASSEMBLE stays as the real-footage benchmark; SO-101 is parked.

---

## 2. Results as of today

All Argon unless marked Claude. "Practice starts" means the numbers are not held-out scores (see [section 3](#3-not-proven-yet)).

### Reading footage: our skill vs Google's published prompt (stage A)

Sources: `runs/argon/hindsight/report.md`, `runs/argon/hindsight/results/fairness.txt`.

**Setup.** We ran Google's published hindsight prompt (from its origami / UMI demos) verbatim, against our shared `read-footage` skill. Both run on Argon, on the same labelled recordings, 4 per source. **These are not Google's scores: Google published none,** and we don't have its videos.

**Raw result.** Our skill beats or ties the prompt on every recording. Segment F1: Assembly101 0.41 → 0.54, REASSEMBLE 0.60 → 0.94, LIBERO 0.72 → 0.85.

**The fairness controls show that most of the gap is conventions and frame rate, not method.**

| Source | What survives the controls |
|---|---|
| **REASSEMBLE** | None. Without the source conventions our skill scores 0.59 vs Google's 0.60. Those conventions were measured on these same train episodes, so the +0.34 is leakage |
| **Assembly101** | About 40%, on the same 3 recordings: Google 0.38 → our method alone 0.45 → with conventions 0.55. Mistake recall (0.95 vs 0.79) and progress ρ (0.95 vs 0.81) also hold without conventions |
| **LIBERO, progress curve** | Frame rate explains it: Google's prompt at 4 fps reaches ρ 0.97 vs our 0.98 |
| **LIBERO, primitive segmentation** | The gap survives at equal frame rate: 0.63 vs 0.85 |

**The claim we can defend.** Our skill's gain comes mostly from writing the source's conventions down and reading at the right frame rate, which is what the skill's per-source references and Choose-your-path encode. The method alone adds a modest gain on human footage (Assembly101) and on LIBERO segmentation. A held-out claim needs new recordings that no lesson has seen.

Other notes: Google's prompt at the default 1 fps loses short clips. Our skill costs about 1.5× per recording minute ($0.025 vs $0.017), $3.51 in total. Adding lessons improved timing (LIBERO event error 0.34 → 0.20 s) but lowered mistake-flag precision, so that isn't a lasting gain yet. On Google's own origami clip both arms read Google's burned-in overlay, so it doesn't count as evidence.

### LIBERO chain (main track): demo → reward → practice with feedback

Sources: `runs/argon/probes/reward/findings.md`, `runs/argon/libero_chain/findings.md` (round 1), `runs/argon/libero_chain/r2/findings.md` and `r2/results/` (rewards v2 and v3, round 2, one-shot).

**Reward from video.** From one demo per task, the drawer check is exact after one revision. (The same skill on the REASSEMBLE peg: precision 1.0, recall 0.88, 100% of near-misses rejected; `runs/argon/peg_reward/findings.md`.)

**Practice round 1:** 9 starts (3 tasks × 3 inits), judged by LIBERO's own check.

| Measure | Result |
|---|---|
| Succeed at the first try | 3/9 |
| Succeed by the second try | 5/9 |
| LIBERO's check measured 1 s after the trial ends | 6/9 |
| Reflections that found the true cause | 5/5 |
| Median cost to first success | $0.70 |
| Total cost | $5.46 |

**The reward over-corrected once, and the gate caught both mistakes.**

1. **Reward v1 was too loose.** The audit found the mug reward accepted placements 4–6 cm off the plate, which gave 3 false successes. The probe's 4.5 cm near-miss had already predicted this. Argon revised the reward using only gate evidence.
2. **Reward v2 was too strict, and the gate showed it.** The gate checks the share of demo ends the reward calls true: v2 gave mug 0.18, butter 0.94, drawer 1.0, with 0 near-misses accepted. Re-scoring round 1's trial ends offline, v1 agreed with LIBERO on 10/14 (4 false positives, 0 misses) and v2 on 9/14 (0 false positives, 5 misses): v2 traded false wins for missed real ones. The cause: 10 of 17 mug demos end with a finger still on the mug. Replaying round 1's v1 successes with a 1 s hold, v2 and LIBERO agree on 9/9. Sources: `r2/findings.md`, `r2/results/rescore_r1_v2.json`, `rescore_r1_v2_hold.json`.
3. **Reward v3** fixes the mug at the gate (demo-end true: mug 1.0, butter 0.94, drawer 1.0; 0 near-misses accepted) but was an offline iteration only, never used in practice. Round 2 practised on v2, with the result below.

**Round 2 vs one-shot: one-shot Argon beat our loop on the paired starts.** Source: `runs/argon/libero_chain/r2/findings.md`. These are the 5 of the 9 fresh starts where both arms finished, all judged by LIBERO's check:

| Arm | Success |
|---|---|
| One-shot Argon (no reward, no feedback) | **2/5** |
| Our first trial, practising on reward v2 | 0/5 |
| Our loop within 2 tries | 1/5 |

- Exact McNemar p = 0.5 and 1.0, so no difference is shown either way.
- One-shot on all 9 starts: 4/9, at $1.00 per success. Its own "done" claim was false 2 of 5 times.
- **Why we lost:** round 2 practised on reward v2, which was too strict. It never fired on a correct mug placement, because 10/17 demos end with a finger still touching the mug. A wrong reward hurts practice, which is exactly why the gate exists.
- **Reward v3** on the round-1 re-score gets 8/9, and one mug placed 4.0 cm off still passes.
- **Honest summary:** with the good reward (v1 on drawer and butter) plus reflections, round 1 went 3/9 → 5/9. Round 1 and the one-shot run used different starts, so 5/9 vs 4/9 is not a paired comparison. **The provable boost over one-shot is not shown on LIBERO today.** The next run is v3 on held-out starts, with the full ladder.

**LIBERO compile (stage E): practice turned into a cheap routine.** Source: `runs/libero/compile-e/report.md`; clips are in `runs/libero/compile-e/sim/episodes/`.

- Argon compiled its 3 successful drawer practice trials into one routine, in one call costing $0.043.
- On 3 new starts the routine opened the drawer 3/3, judged by LIBERO's check and still true 1 s later.

| Per success | Practice | Routine |
|---|---|---|
| Brain calls | 18.3 | 1 (a look) + 0 |
| Wall time | 758 s | 40 s |
| Cost | $0.61 | $0.006 |

That's about 19× faster and about 100× cheaper. **Caveat:** n = 3 in one scene, and the handle moves at most 2 cm. This shows the cost collapse, not robustness.

**Compiled routine vs one-shot Argon, on the same drawer starts** (inits 6, 18, 38). Source: `runs/libero/compile-e/paired.jsonl`.

| | One-shot Argon | Compiled routine (after practice) |
|---|---|---|
| Success, by LIBERO's check | 1/3 | **3/3** |
| Calls per episode | 22 | 1 |
| Cost per success | $1.09 | $0.0057 |
| Wall time per episode | 301–805 s | 34–36 s |

- The 2 discordant starts both favour the routine. Exact McNemar p = 0.5: n = 3 can't reach significance.
- **Claim:** on the same starts, practice-then-routine succeeded where one-shot failed, at about 1/190 of the cost per success. It shows direction, not proof. A larger run on fresh inits was still going at the time of writing (until about 16:57).

**The anchor.** The published Argon attempt-1 on LIBERO-10 is 43%. Our reproduction of the paper's protocol got 2/6, against the paper's 31/60 on the same tasks, at $5.70 vs $5.90 per episode. The published camera views are mirrored.

### Web video track (new): real footage → sim task

Sources: `playgrounds/footage/run_report.md`, `runs/footage/summary.json`.

Three clips: a YouTube "put the bowl in the microwave" (11 s), a HOT3D egocentric clip (5 s) and a YouTube first-person table setting (51 s). Argon reads each clip into steps, objects, layout and success conditions, then:

| Step | Result |
|---|---|
| Writes a LIBERO scene file | 3/3 parsed on the first try, no repairs |
| LIBERO builds it | 3/3, in 20–63 s |
| LIBERO's goal check: False at reset, True at the goal | 3/3 |
| Argon's video-written check agrees with LIBERO's on saved states | microwave 4/4, table 4/4, HOT3D 3/4 |
| HOT3D placement vs ground truth | 2.4 cm mean over object-to-object distances; 2 of 6 objects have LIBERO assets |
| Cost, end to end | about $0.05 per clip ($0.158 for 3) |

- The HOT3D disagreement: LIBERO accepts the can resting on an oversized stand-in box; Argon's check is stricter.
- **Honest:** a one-prompt baseline ties ours on these 3 simple clips (parse, goals, check agreement) at about 40% of the cost. Ours is better only on HOT3D placement (4.3 vs 6.3 cm). Asset size isn't in the loop yet (hence the oversized stand-in), and practice on these scenes needs one server change.

### REASSEMBLE A–G (real robot footage, NIST task board)

| Stage | Result | Source |
|---|---|---|
| **A, read** | Claude and Argon both pass 3/3 on 9 blind gate episodes: segment F1 0.992, balanced accuracy 0.947. Argon $0.22 per recording minute, Claude $0.32. Self-learning: the skill went from bacc 0.78 to 0.95 | `runs/argon/probes/longvideo/findings.md` + the A checks |
| **B0, cameras** | 1.43 / 1.72 px with the skill's recipe; 4.46 px with the skill alone | `runs/argon/probes/b0/findings.md` |
| **B1, scene** | Passes 4/4. Argon's MJCF is physically sound on 10/12 checks but places parts 7 mm off | `runs/argon/probes/scene/findings.md` |
| **B2, replay** | Seat rate vs real (78%): 54% → 63% → 69% over three rounds, train only. Our audit found a leak worth about 0.10, so it's not certified and is parked | |
| **D, Argon attempt 1** | 11/14 seated (79%, 95% CI 49–95%), 0 false claims; $1.48 and 20.9 calls per success. Same starts, Claude raw: 12/14 (p = 1.0), 10.2 calls per success. Five episodes are marked as deviations because of unlogged pings | `runs/argon/pilot/findings.md` |
| **D, retries with feedback** | 4 hard starts: 0/4 at the first try, 1/4 by the second. 5 of 6 reflections named the real cause, 0 truth leaks. Nudges had no measurable effect | `runs/argon/trials/findings.md` |
| **D, Claude pilot** | Raw 21/24 vs tools + routine 16/24. The auditor's verdict was ITERATE (measure the tool tip first): the system refusing to fool itself | `runs/orchestrator/pilot-loop-1/audit/final_1440.md` |
| **E, compile** | Argon's routine went 1/6 → 5/6 in development. On unseen starts it seated 8/24 vs Claude's 5/24, using 42 s vs 186 s of sim time and 0 brain calls | `runs/argon/probes/compile/findings.md` |

### Integrity (G)

Source: `runs/argon/probes/audit/findings.md`.

- The Argon auditor caught **4/4** planted cheats and judged 13/16 outcomes right, at $0.05 per audit.
- The red team was caught **15/15**, with **0** sandbox escapes.

---

## 3. Not proven yet

Copied from the sheet:

- The vs-Google reading gain is mostly conventions learned on the same recordings, plus frame rate. The method alone adds a modest gain on Assembly101 and on LIBERO segmentation.
- No scored held-out boost: every attempt-1 vs attempt-N number above is on practice starts.
- Loop 1 (the twin) isn't certified.
- On the LIBERO anchor we're below the paper.
- The rungs "retries without feedback" and "one-shot after practice" are designed but not run.
- On LIBERO, one-shot Argon (2/5) beat our loop practising on the over-strict reward v2 (1/5 within 2 tries) on 5 paired starts. After practice, the compiled drawer routine beat one-shot 3/3 vs 1/3 on the same starts (p = 0.5). The direction favours practice, but significance is not shown.

---

## 4. Next

### The evaluation ladder for a provable boost

Every rung runs on **the same held-out starts** and is judged by **the sim's own check** (LIBERO's goal predicate), never by our reward.

| Rung | What it isolates |
|---|---|
| 1. **One-shot** | Argon alone, the published attempt-1 protocol |
| 2. **Retries without feedback** | How much is just extra tries |
| 3. **Our full loop** | Retries with reflection and observer feedback, the frozen reward, task tools, compiled routines |
| 4. **One-shot after practice** | Whether what was learned carries to a single try on new starts |

A provable boost means rungs 3 and 4 beat rungs 1 and 2, with paired tests on the same starts. The first run of the full ladder uses reward v3 on held-out LIBERO starts.

### Footage-to-sim follow-ups

- **The practice server change for scenes built from footage**, so the web video scenes can be practised on, not only built and checked.
- **Asset footprints fed into scene writing**, so Argon picks stand-in objects of the right size (the HOT3D oversized box).

---

## 5. How results are kept honest

- **Frozen splits** ([`playbook/splits.json`](playbook/splits.json), [`playbook/splits_so101.json`](playbook/splits_so101.json), [`playgrounds/libero/splits.json`](playgrounds/libero/splits.json)). Fitting uses train only, gate is scored once per frozen version, and exam is refused by the tools.
- **Actor ≠ watcher ≠ auditor.** The agent that does the work never records its own checks. The watcher reruns deterministic tools and records the numbers; the harness decides pass or fail. The auditor reads traces and can stop a stage (the Claude pilot's ITERATE, the reward v1 hole).
- **No sim cheating** ([`playbook/rules.md`](playbook/rules.md)). Only parameters set before t = 0 are allowed. Runtime monitors flag model changes, applied forces, penetration, teleports, energy spikes and outcomes that flip at half the timestep. A flagged success doesn't count.
- **Capability boundary.** Actors reach the simulator only through a command API with force and speed ceilings in code. Compiled routines pass a static checker and run in a separate sandboxed process.
- **Observers never pass sim truth.** Reflections and nudges are grepped for leaks (0 found).
- **Records.** Every tool call is logged to `runs/calls/`. Checks and targets are frozen in [`playbook/validation.yaml`](playbook/validation.yaml).

---

## 6. Repo map and reading order

1. This README.
2. [`docs/design/vision-draft.md`](docs/design/vision-draft.md): the plan in full.
3. [`playbook/README.md`](playbook/README.md), [`playbook/orchestrator.md`](playbook/orchestrator.md), [`playbook/rules.md`](playbook/rules.md): how agents run the loop.
4. [`playbook/skills/`](playbook/skills/): the 10 shared job skills, each with Choose-your-path, adapters and references.
5. [`playbook/validation.yaml`](playbook/validation.yaml): every check, its target and the version history.
6. [`docs/knowledge/`](docs/knowledge/): published best practice behind each decision; today's lessons in [`docs/knowledge/lessons-2026-10-25.md`](docs/knowledge/lessons-2026-10-25.md).
7. [`docs/research/google-argon-twin-pipeline.md`](docs/research/google-argon-twin-pipeline.md): how Google's Argon demos build twins.

| Path | What |
|---|---|
| `harness/` | Tool registry and CLI (tools in `harness/tools/*.py`, auto-discovered, including the LIBERO tools); `harness/brains/` runs Gemini over the tools, with trials and an offline fake; `harness/status.py` builds the page |
| `twin/` | MuJoCo scene, replay, controller, env with its capability boundary, monitors, task tools, skill library |
| `playbook/` | Orchestrator, rules, shared skills, validation checks, splits; `archive/skills-v1/` holds the old skills |
| `playgrounds/libero/` | The LIBERO playground: server, sim wrapper, Argon loop, feasibility runs and report. `report.py` also reads `paper_argon_episodes.json` (the paper's published per-episode numbers), which is kept out of git as downloaded data |
| `playgrounds/footage/` | The web video track: read → LIBERO scene file → build → reward gate, plus the one-prompt baseline |
| `recon/` | Loop 1 for any video (intake → scene card → twin). Written, not run: stopped when scope was cut |
| `docs/` | Knowledge base, research notes, design (vision, UI protocol) |
| `tests/` | Sim-integrity suite, decision tests, UI and handback checks |
| `runs/` | Run outputs, findings and the status page. **Not in git**, except `runs/status/story.yaml` |
| `data/`, `vendors/` | Downloaded datasets and reference repos. **Not in git** |

---

## 7. Running it

Environment: `.venv` (uv, Python 3.12) with torch 2.11 (CUDA), mujoco 3.17, mujoco-warp 3.17, lerobot 0.6.1, google-genai 2.33. LIBERO has its own setup: [`docs/research/libero-setup-notes.md`](docs/research/libero-setup-notes.md). Details: [`docs/research/env-setup-notes.md`](docs/research/env-setup-notes.md) and [`CLAUDE.md`](CLAUDE.md).

```sh
.venv/bin/python -m harness.cli list                                # every tool's declaration
.venv/bin/python -m harness.cli call status_summary '{}'            # every stage's state
.venv/bin/python -m harness.status                                  # rebuild runs/status/index.html
.venv/bin/python -m harness.brains.test_offline                     # brain runtime, offline
MUJOCO_GL=glfw .venv/bin/python -m twin.render                      # smoke render (WSL needs glfw)
.venv/bin/python -m tests.test_sim_integrity --quick                # integrity tests
```

The Gemini key lives outside the repo (see `CLAUDE.md`) and is passed per command.
