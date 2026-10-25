---
name: read-footage
description: Stage A on any data source. Turn task footage (robot recordings with logs, plain video, simulator demos, or a source read before) into hierarchical labels (task, steps, primitives with times), an outcome per attempt with its evidence, strategy notes and a failure taxonomy, then score the labels against the source's own where it has them. Pick the adapter by what you were given; source facts are in references/.
stage: read
steps: [A]
---
## Goal

A labels file (`schema: read-footage/v1`, defined in `references/reassemble.md` "Output schema") for each target recording:
- task → steps → primitives, each with start and end times;
- an outcome per attempt (success or the failure kind) and the evidence it rests on (a signal window, a frame time, or both);
- strategy notes per skill and a failure taxonomy.

Done when the labels are scored against the source's labels where they exist (Checks), the cost per recording minute is recorded, and Lessons holds the run.

## Choose your path

| You were given | Read it like this | Adapter | Source facts |
|---|---|---|---|
| **Robot logs + cameras** (joints or TCP, gripper, F/T or commands; 1+ cameras) | Signals first: the deterministic draft from the logs, then the model fills names and checks; video only for identity and "did it move" | `adapters/robot-logs.md` | `references/reassemble.md` (Franka, F/T, 3 cameras); `references/so101.md` (leader-arm commands, 2 cameras) |
| **Video only** (no logs) | One whole-recording 1 fps overview, then at most 2 levels of zoom on a budget; outcome from the end state after release | `adapters/video-only.md` | `references/assembly101.md` (human toy assembly) |
| **Simulator demos with a state API** (a benchmark that stores full sim states) | Read one demo clip for the task spec; times and outcomes come from the recorded states, not from pixels | `adapters/simulator.md` | `references/libero.md` |
| **Prior tested runs and lessons** (this source was read before) | Start from the frozen rules and the last scored run; re-score on held-out recordings before trusting a rule on new data | `adapters/prior-runs.md` | the source's reference, Lessons below |

Several rows can apply. A REASSEMBLE recording is robot logs + cameras and also has prior runs: read both adapters.

## Inputs

- The recordings, cut to the target window, and the split they belong to (`playbook/splits.json`, `splits_so101.json`, or the benchmark's own). Exam recordings are never read before the exam (rules.md).
- What the source documents about its labels: step granularity, failure definitions, part names. Write it down before labelling (it becomes the source's conventions section in `references/<source>.md`).
- Never the labels of the recordings being scored; they are for scoring only.

## Tools

- Robot logs: `end_state_check`, `query_signals`, `identify_part_by_hole`, `probe_video`, `score_labels` (registry; `.venv/bin/python -m harness.cli list`). Details in `references/reassemble.md`.
- Video: a 1 fps whole-recording video, frame grids (about 1.1k tokens per image whatever the tile count, `runs/argon/probes/api/findings.md` §2) and video zooms in a user step. The probe scripts are `runs/argon/probes/longvideo/scripts/` (`lvmedia.py`, `zooms.py`); they are not registry tools yet.
- Scoring: `score_labels` (segment F1, boundary error, failure balanced accuracy).
- **Registry tools carried from v1** (`--skills` allowlists read this section; usage in references/): `recording_info`.

## Approach

Core method; it holds on every source so far.

1. **Write the conventions first.** Losses on a new source come from its conventions, not from perception: on Assembly101 every setting scored F1 0.45–0.49 because four wheel attachments were one step there and part names differed (`runs/argon/probes/longvideo/findings.md`). Put the source's step granularity, failure definitions and a part glossary in the prompt and ask the reader to restate them before labelling.
2. **Take times from the most direct signal you have.** Force and gripper signals beat pixels for contact moments: the camera shows the peg arrive about 0.75 s before force rises, in every video condition (`runs/argon/probes/api/findings.md` §2). With logs, video added nothing measurable over signals (S3 F1 0.93 vs S4 within noise, `runs/argon/probes/longvideo/findings.md`). With recorded sim states, read times from the states.
3. **Judge the outcome from the end state, not the motion.** Operators release once the tip is in and gravity finishes the job, so the outcome is read after release (`references/reassemble.md`). For video, read a frame 0.5–1 s after release and the next sighting of the object (longvideo proposal 6).
4. **A deterministic draft first, the model second.** On REASSEMBLE the `end_state_check` draft alone scores F1 0.92 and Argon mostly adopts it (`runs/argon/probes/longvideo/findings.md`). Where no draft exists, the overview pass plays that role.
5. **Zoom with a budget and a reason.** Cap zooming at 2 levels and 30 % of the recording, refuse identical windows, record what each zoom changed. Zooming bought +0.04–0.06 F1 and found zero extra failures (`runs/argon/probes/longvideo/findings.md`).
6. **Every claim cites its evidence** (a signal window or a frame time) so a watcher can spot-check it (`runs/argon/probes/longvideo/scripts/claims.py`).
7. **Tune rules on train, report on held-out.** A rule tuned on the recordings it is scored on is not evidence (`references/reassemble.md`: 60 tuning / 29 held-out recordings).

## Per-model notes

- **Claude (stand-in).** Reads PNG frames and frame grids; it cannot take a video file. Use `probe_video` grids at exact times and the signal tools (`references/reassemble.md` Lessons; about 9–13k unique tokens per recording minute).
- **Argon.** Takes video directly. Put video in a **user step**: inside a `function_result` its `fps` is ignored and a zoom comes back at 1 fps (`runs/argon/probes/api/findings.md` §2, the runner's `--video-placement user_input`). 4 fps is the knee for contact moments; 1 fps snaps answers early by about 0.6 s. A 2 fps frame grid was the most accurate and cheapest for one moment (release error 0.14 s). Agentic video works only as `"processing": "agentic"` (a string) and is the costliest, least stable condition. Medium thinking for labelling; never ask for numbers in JSON `response_format` at low or medium: it emits a guess, while free text or `code_execution` gets them right (`runs/argon/probes/api/findings.md` §1). Expect $0.01–0.03 per recording minute on video only and $0.041 with signals (`runs/argon/probes/longvideo/findings.md`).
- **Any model.** Time burned into frames or tile titles (`t=…s`) keeps answers tied to the clock.

## Checks

Recorded by a watcher with `record_check`; targets are the source's stage-A checks in `validation.yaml` (REASSEMBLE stage A; SO-101 and footage tracks have none yet, so their numbers are tracked).

| Check | How |
|---|---|
| segment F1, boundary error | `score_labels` against the source labels, held-out recordings only |
| failure balanced accuracy | the same, per attempt |
| claims grounded | share of cited evidence that holds on a spot-check sheet (`claims.py`) |
| cost per recording minute | tokens and $ from the run record |
| video-only provisional | with no labels, a human confirms 3 sampled outcome frames; report the labels as unscored |

## Lessons

Dated entries from every source, one line each; the full entry, with its numbers, is in the named reference. New lessons go here (`append_lesson`) with the source named; detail goes to `references/<source>.md`.

- **2026-10-24** (claude): claude-opus-5.5 stand-in, episodes 0, 1 and 4. Evidence: `runs/footage/ep000`, `runs/footage/ep001`, `runs/footage/ep004`; full entry in `references/reassemble.md` (v1 `read-footage`).
- **2026-10-25** (claude): claude-opus-5.5 stand-in, blind cold read of gate episodes 68, 136, 140. Not scored when written; a separate watcher scores it. Evidence: `runs/claude/run-standin-1`, `runs/calls/20261025.jsonl`, `runs/footage/ep140`; full entry in `references/reassemble.md` (v1 `read-footage`).
- **2026-10-25** (claude): claude-opus-5.5 stand-in, refinement on train recordings only. No gate label was used. The watcher's gate files were read only to see which kinds of error to look for. Evidence: `runs/footage/refine-20261025`, `runs/claude/run-standin-1`; full entry in `references/reassemble.md` (v1 `read-footage`).
- **2026-10-25** (claude): claude-opus-5.5 stand-in, blind retest of gate episodes 99, 102, 103 with the refined tools. Not scored when written. Evidence: `runs/claude/run-standin-1`, `runs/footage/ep102`, `runs/footage/ep103`; full entry in `references/reassemble.md` (v1 `read-footage`).
- **2026-10-25** (claude): claude-opus-5.5 stand-in, blind confirmation read of gate episodes 24, 42, 58, 65, 92, 117. Not scored when written. Evidence: `runs/claude/run-standin-1`, `runs/footage/ep042`, `runs/footage/ep092`; full entry in `references/reassemble.md` (v1 `read-footage`).

## Sources

- `runs/argon/probes/longvideo/findings.md` (settings per input type, Assembly101 transfer, proposals), `runs/argon/probes/api/findings.md` (§1 thinking levels, §2 contact moments and video placement).
- v1 skill text, moved unchanged to `references/reassemble.md` (archived at `playbook/archive/skills-v1/read-footage/`).
- docs/knowledge/video-to-twin.md, docs/knowledge/evaluation-and-statistics.md.
