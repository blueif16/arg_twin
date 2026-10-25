---
name: read-footage
description: Stage A. Turn a REASSEMBLE recording (3 cameras + F/T + robot state) into hierarchical hindsight labels, per-skill strategy notes and a failure taxonomy, then score the labels against the dataset's.
stage: read
steps: [A]
---
## Goal

For each target recording, a labels file `runs/footage/ep<NNN>/labels_<run>.json` (schema below) that holds, in the style of Google's Argon robotics labelling demo (1 goal → 6 subtasks → 17 contact-level primitives, progress held flat while the robot fumbled; `docs/test-plan-v0.md` §1):

1. **Episode goal**: one sentence.
2. **Subtasks**: one per dataset-style segment, named exactly from the dataset vocabulary (`<Verb> <part>.`, verbs Pick / Insert / Remove / Place, 17 parts; `recording_info` lists them), with start/end, success, confidence and evidence.
3. **Primitives** inside each subtask: contact-level steps (approach, align, grasp, regrasp, lift, pull, push, insert, twist, wiggle, nudge, transport, release), each with times and the signal or frame that shows it.
4. **Progress curve** per subtask: knots `[t, p]`, p in 0–1 toward that subtask's goal. Progress rises only when the task state changes (part grasped, part moves along its axis, part free). It stays **flat during fumbles** (pulling or twisting with no displacement) and **drops** when a grasp is given up.
5. **Quality weight** per subtask, 0–1, for later use as a demonstration weight: 1 = clean success, 0.5–0.9 = success after a retry or with fumbles, 0 = failure (still valuable as a negative and for twin replay).
6. **Strategy notes** per skill family (e.g. `remove/gear`, `insert/peg`), saying how the operator aligns, tilts, wiggles and twists, each with force/motion numbers.
7. **Failure taxonomy**: codes, each with a signal signature, its instances and what the twin must reproduce.

Done when the labels are scored with `score_labels`, and the strategy notes and taxonomy are merged into `runs/footage/strategy_notes.json` and `runs/footage/failure_taxonomy.json`.

## Inputs

- `data/reassemble/` (LeRobot v3; see `docs/research/reassemble-data-notes.md` for frames, units and known gaps), plus the train and gate episodes in `data/reassemble-gate/`: 113 train and gate episodes have signals on disk. Episodes 9 and 10 are exam (`playbook/splits.json`) but sit inside the local data file: never read them. Video: hand camera for episodes 0–5 plus the HF files holding 60–68, 99–104 and 134–141 (`data/reassemble-gate-video/`), hama1 for 0–2, hama2 for 0–1 (`recording_info` reports what is on disk). Gate episodes hide their labels unless `labels: true` is passed.
- Dataset facts worth knowing: `Remove` = pull out of the board, `Place` = put down on the table. The teleop contact threshold is 5 N on |force.base|. force.base is the external force on the tool in the base frame: **Fz < 0 = pulling up against the board, Fz > 0 = pushing down**. On the 89 train recordings 304 of 2,662 action segments failed: Insert 172/754, Remove 67/569, Pick 58/757, Place 7/582.

## Tools

| Tool | Use |
|---|---|
| `recording_info` | Episodes on disk, cameras, vocabulary; per episode the dataset segments. **Use `labels: false` for any episode you will score**: the labels must not be seen before scoring. |
| `end_state_check` | **Start here.** Label-free, one call per episode. Per gripper cycle: where the part was grasped and released (at a hole or off the board), lift onset, how far a removed part was carried, insert attempts while held (each ended by a retreat), what follows a release at a hole (open-hand return or press, regrasp and whether it lifts the part out, the same part inserted again later), open-hand dips with no close. Returns a **draft** of dataset-style segments, each with the rule that decided it and that rule's train precision. |
| `identify_part_by_hole` | Label-free part names from WHERE each cycle grasps or seats the part: the hole within 25 mm, its runner-up, the across/along-pad distances, and a hold-width check against train widths. Use it for every part name you write. |
| `query_signals` | Signals plot PNG + JSON: contact events (onset/offset, peak and mean force vector, dz, lateral path and reversals, yaw/tilt change), gripper open/close events, pauses, ranges, optional `series_hz` rows. Channels: force, torque, torque_tool, gripper, speed, pos, orient, joints. `marks` draws your proposed boundaries. |
| `probe_video` | Frame-grid PNG (tiles titled `cam t |F| gripper`, red frame when |F| > 5 N) and an mp4 clip (burned-in time, `slowdown`). `crop` zooms; `max_frames` caps the grid. |
| `score_labels` | Your labels vs the dataset's: segment P/R/F1 at IoU ≥ 0.5, F1 with the task right, verb/part/task accuracy, success accuracy, failures detected, boundary errors. Reads `windows` from your file. |

Call them as `.venv/bin/python -m harness.cli call <tool> '<json>' --caller <you>`. Every call is logged to `runs/calls/`.

`end_state_check` and `identify_part_by_hole` (`harness/tools/footage_id.py`) read only robot signals, the recording's mocap board pose and the hole layout fitted on train recordings (`runs/claude/run-standin-1/board/layout_contact.json`). They never read labels, so they are safe on gate recordings. Their board pose is the mocap pose refined so that the recording's own grasp and seat points sit on holes; `board.events_on_holes` < 3 means the mocap pose is used unrefined.

**Image readers vs video readers.** Claude reads PNGs only: use the grids and plots. Gemini can take the mp4 clip directly; slow it down (`slowdown` 2–4) over contact moments, because Gemini samples video at about 1 fps by default. Both should read the signals plot: most boundaries and every failure call came from signals, not pixels (Lessons).

## Approach

Guidance, not a recipe. Measure what works and record it under Lessons. The draft is a proposal from deterministic rules; you own every label and must be able to defend it from a signal or a frame.

1. **Draft from signals (2 calls).** `end_state_check` over the whole episode, then `query_signals` over the whole window with the draft's boundaries as `marks`. Walk the draft against the plot: every segment should sit on a gripper close…open cycle, a retreat, or an open-hand return. The sign of Fz still tells assembly (push, Fz > 0) from disassembly (pull, Fz < 0).
2. **Name every part from where it goes** (`identify_part_by_hole`, or `part`/`basis` in `end_state_check`). A grasp or seat within 25 mm of a hole names the part; on train this matched the labels on 1,186 of 1,194 cycles. Remove grasps sit within 12 mm of their hole (p95) and table grasps 101 mm or more from every hole (p5), so the location also gives the verb (grasp at a hole = Remove, off the board = Pick; release at a hole = Insert, off the board = Place). Round vs square and peg index come with the hole: do not name them from the video or from height.
   - Hold width is a tie-break, not a name: round and square pegs of the same index differ by ≤ 0.6 mm, and several parts have a second grasp mode (train widths in `GRIP_MM`, `footage_id.py`). It names failed attempts that hover between holes (lowest, closest, width fits).
   - A cycle that never comes near a hole (picked and set down on the table) takes the name of the adjacent cycle that grasps the same object where this one released it; failing that, the width guess, which you must confirm in the hand camera.
3. **Look at the video only where the draft is weak** (one coarse hand grid at 0.2–0.33 fps, then zooms with `crop` about [0.15, 0.35, 0.85, 1.0]):
   - a part named by `hold width only`, or `board.events_on_holes` < 3;
   - a peg called unseated by an open-hand return (39/51 right on train): one hand frame 0.5–1 s after the release shows whether the peg stands proud;
   - a close on nothing inside an otherwise good Pick (labels sometimes fold the miss into the Pick: 61/69 right);
   - a `width_check` that disagrees, gear or nut regrasps in place, connectors released and touched again;
   - open-hand dips with no close (`open_hand_dips_before`): an aborted approach is labelled a failed Pick/Remove, but the dips are not drafted because on held-out train recordings they caught nothing and added false segments.
4. **Read the contact physics** for each attempt with `query_signals` on the contact window (`force`, `torque_tool`, `pos`, `orient`). Twist = yaw change during contact. Wiggle = lateral reversals plus small yaw oscillation. Jam = |F| high with dz ≈ 0. Slip = |F| drops suddenly while z jumps and the grip width shrinks. A slow pull (twist-lock BNC, a nut unscrewing) creeps up 5–17 mm while the part is still attached: the lift onset is the fast rise (> 8 mm in 0.5 s), which is what `end_state_check` reports.
5. **Boundaries (dataset conventions, measured on train with the tool's onsets; median end error on matched segments within ±0.15 s for every common rule, retreats −0.3 s).** Write the ones you use into `boundary_rules`.
   - A handling cycle is Remove→Place or Pick→Insert. The first verb ends **1.05 s after the lift onset** (the fast rise).
   - Insert and Place end **0.35 s after the middle of the gripper opening**. An open-hand press on a gear or connector belongs to its Insert, which then ends 0.35 s after the press.
   - **Each retreat while holding ends a failed Insert**: after a push near the hole the TCP climbs > 35 mm above the attempt's lowest point (or leaves sideways > 70 mm) with the part still held. The segment ends 0.5 s after the climb starts; the next attempt is a new `Insert`. If the part is then set down on the table, the last attempt is a failed Insert followed by a `Place`.
   - A close on nothing ends 1.0 s before the middle of the re-opening.
   - After a failed Insert, a regrasp that lifts the part out (> 15 mm) or carries it away is a new `Pick`.
   - A part grasped, lifted and set down on the table again with no insert is a successful `Pick` plus a successful `Place`.
   - Idle (`No action.`) covers the start of the recording until the arm first moves, and the end after the last release. Scoring ignores it by default.
6. **Decide success from the end state**, with these rules (train precision on matched segments, 89 recordings):
   - **Remove** succeeds only if the part is carried **> 80 mm from its hole** before the fingers open (tuning recordings: every failed Remove opened within 68 mm, successes at 119 mm or more, p5). A z rise is not enough. 45/51 right.
   - **Insert** attempts that end in a retreat failed (76/76).
   - **Peg released at its hole, then the open hand comes back down near it** (a drop of > 10 mm within 65 mm of the hole, back to within 40 mm of the release height, before the next close): the release Insert failed and the return is a new, successful `Insert` (the nudge, ending at the bottom of the return). 39/51 failures right, 38/38 nudges right.
   - **Gears and connectors pressed with the open hand after release**: one Insert, normally a success (train: 90 of 94 gear and 36 of 38 connector releases with a press were labelled successful, press inside the segment).
   - **Regrasp after an insert release** (next grasp within 45 mm of that hole): lifted out or carried → that Insert failed (16/19); in place → gears, nuts and pegs stay one Insert (22/25 success right), connectors are a failed Insert plus a new Insert (3/3).
   - **The same part inserted into the same hole again later**, with no Remove in between: the earlier Insert failed (3/4; e.g. a nut that never caught its thread and was picked up again).
   - **Closed on nothing**: < 1 mm, < 10 s and no contact > 5 N. A USB plug can be held at 0.9 mm for 30 s of insertion. 61/69.
   - **Place** is a success unless you see the part fall (1% of Places fail on train).
   - Not a failure signature: seat depth relative to the part's train seat height (failure-vs-success AUC 0.48), peak insert force (0.40), z travel while a nut turns.
   If you cannot see or measure the end state, say so in `confidence`.
7. **Write strategy notes and failure codes a watcher can confirm.** Each statement names the part from step 2, gives the window, and quotes numbers from a tool output (peak and mean force, duration, dz, yaw, reversals). State per instance, not "all X did Y": the first gate read lost a claim by bounding all eight peg inserts at ≤ 5 N and asserting every peg stayed seated; the watcher measured 7–8 N peaks and a failed insert among them. Candidate codes: F-JAM-AXIAL, F-ABORT-EARLY, F-GRASP-POSE, F-GRASP-SLIP, F-THREAD-NO-CATCH, F-GRASP-MISS, F-PEG-UNSEATED, F-RETREAT. Reuse them only when the signature matches; add new codes otherwise.
8. **Score last**, then write what the score taught you into Lessons: boundary offsets, label confusions, misses.

### Output schema (`schema: read-footage/v1`)

```json
{
 "schema": "read-footage/v1", "episode": 0, "recording": "…", "brain": "<model / caller>",
 "blind": true, "contamination": "<what you had seen about this episode before labelling>",
 "windows": [[t0, t1], …],
 "goal": "<one sentence>",
 "boundary_rules": {"<rule name>": "<rule>"},
 "subtasks": [{
   "id": "s1", "start": 0.0, "end": 0.0, "task": "<Verb> <part>.", "success": true, "confidence": 0.0,
   "evidence": ["<tool output path or numbers>"],
   "primitives": [{"start": 0.0, "end": 0.0, "type": "approach|align|grasp|regrasp|lift|pull|push|insert|twist|wiggle|nudge|transport|release",
                   "contact": false, "evidence": "<numbers>"}],
   "progress": [[t, p], …], "quality": 0.0, "failure_mode": "<code or null>", "notes": "…"}],
 "strategy_notes": {"<verb>/<part family>": ["<observation with numbers>"]},
 "failure_taxonomy": [{"code": "F-…", "name": "…", "signature": "<signal pattern>", "instances": ["s3"],
                       "twin_implication": "<what the twin must reproduce>"}],
 "twin_requirements": ["<what stage B must get right, with numbers>"]
}
```

`windows` must cover only stretches you labelled completely. A segment cut by your window edge is either labelled in full or left out, never half-labelled.

## Checks

A watcher can run these without the actor's reasoning:

1. The labels file parses, follows the schema, and every `task` is in the `recording_info` vocabulary.
2. Blindness: `runs/calls/` shows no `recording_info` call with labels for this episode by this caller before `score_labels`. `blind` and `contamination` are filled in honestly.
3. `score_labels` result on file. Bar for a scored episode: segment F1 ≥ 0.8 at IoU 0.5, success accuracy ≥ 0.8 on matched pairs, and every dataset failure in the window detected or explicitly missed in Lessons.
4. Subtasks don't overlap. Primitives lie inside their subtask. Progress knots are time-ordered and in [0, 1], and stay flat (±0.05) across every primitive whose evidence shows no displacement.
5. Every strategy note and failure code cites numbers from a tool output that exists in `runs/footage/`.
6. Cost is reported: tool calls, images read, wall time, from `runs/calls/`.
7. Every part name agrees with `identify_part_by_hole` for that cycle, or the subtask's `notes` say why not (the frame or signal that overrides it). Every success call that differs from the `end_state_check` draft says the same.

## Lessons

**2026-10-24, claude-opus-5.5 stand-in, episodes 0, 1 and 4** (labels: `runs/footage/ep000/labels_trial1.json`, `runs/footage/ep001/labels_blind1.json`, `runs/footage/ep004/labels_blind2.json`; scores: `runs/footage/scores/`; calls: `runs/calls/20261024.jsonl`, callers `claude-read-footage`, `claude-read-footage-blind`, `claude-read-footage-blind2`).

- **Scores.**
  - Episode 0, three windows (31.7–101.4, 179.0–230.5, 356.6–386.6 s): 11/11 segments matched, F1 1.0, task accuracy 1.0, success accuracy 1.0, 5/5 failures detected (all four small-gear failures and the Ethernet one), mean IoU 0.84, mean |boundary error| 0.85 s. **Not blind:** the brain had read the episode-0 segment table in `docs/research/reassemble-data-notes.md` beforehand, so treat this as an upper bound.
  - Episode 1, blind (whole episode, labels never read): 7 predicted, 8 dataset segments, 7 matched, F1 0.933, task/verb/part accuracy 1.0 (peg indices right), success accuracy 1.0, 1/1 failure detected, mean IoU 0.84.
  - The one miss: the dataset puts a `Pick bolt 4.` (95.0–105.0 s) between a failed and a successful `Insert bolt 4.`. I had assumed "retries have no new Pick"; rule 5 now says otherwise.
  - Episode 4, blind, window 57.7–135.6 s, labelled with the rules as revised after episode 1: 7 predicted, 8 dataset segments, 7 matched, F1 0.933, task accuracy 1.0, success accuracy 0.857, mean IoU 0.86, mean |start error| 0.51 s. Pick ends from "lift start + 0.9 s" were 0.17–0.47 s late, so the offset is now 0.75 s.
  - The miss and the one false failure were the same event: a D-SUB lifted 70 mm and set down 4 s later. I called it a failed Pick; the dataset has `Pick D-SUB.` (success) + `Place D-SUB.` (65.5–70.1 s). Rule 5 now covers it.
  - Over the two blind runs: 14/16 segments matched, every task name right on matched segments, 13/14 success calls right, 1/1 failure found.
- **Cost.** Episode 0: 14 calls (7 probe_video, 6 query_signals, 1 score), 2 of them failed on a since-fixed crop bug, 24.7 s of tool time, 9 images read. Episode 1: 8 calls, 8.6 s of tool time, 3 images read. Episode 4: 8 calls, 8.1 s of tool time, 2 images read. A blind episode or window took 1 signals read of the whole span, 1 coarse grid, 1 zoom and 2–3 short `series_hz` reads for lift times. Brain tokens were not metered for the Claude stand-in.
- **What worked.**
  - Signals before pixels. Every boundary and every success/failure call came from the gripper trace, contact events and dz. Video was needed only to name the part and to confirm end states.
  - The gripper close→open cycle is the segment skeleton.
  - The `|F|` + grip text in each tile title made grids readable without the plot.
- **Boundary offsets measured.** The first verb ends 0.43–1.13 s after the part comes free or the lift starts, median 0.73 s over 11 cases: +0.97, 0.80, 1.00, 0.63 on episode 0 Remove→Place; +1.10, 1.13, 0.67 on episode 1 Pick→Insert; +0.43, 0.43, 0.50, 0.73 on episode 4 Pick→Insert/Place. End-of-release boundaries scatter ±1 s. Episode 1's first Pick starts at 3.4 s, during what looked like an idle pause (3.1–6.5 s), so the dataset starts a segment at the first motion, not at the end of a pause (start error 3.1 s).
- **The small gear (episode 0, 4/4 failed): what the twin must reproduce.** Grasp at 14–19 mm, pull Fz −8 to −13 N (peak 13.8 N) for 12–16 s with dz < 1.5 mm.
  - Yaw twists of ±5° to ±45° meet no resistance: tool-axis torque stays ≈ 0 (`signals_0056.00-0081.00_force-torque_tool-pos-orient.png`). The gear turns freely on its shaft but stays held axially.
  - In contrast, round peg 4 came out at about 8 N mean (dz 25 mm in 4.9 s, yaw −20°), the D-SUB at < 2 N, and the Ethernet plug at 4.8 N once squeezed to 8 mm. A first grasp at 18 mm was released without a pull.
  - The M16 nut unscrews with about 110° of yaw at < 3 N while z rises 17 mm. Screwing it on failed once (+8 N, −105°, no travel) and succeeded after re-seating (+17 N push, then −53° at +5 N).
- **What didn't work / open.**
  - At grid scale (tiles about 260 px as seen), the external cameras added nothing for part ID; a 0.5-scale crop of hama2 didn't either.
  - Peg index from grip width is a guess built on "index follows size". It was right on 3/3 pegs, but no calibration table exists yet. Build one from train episodes (median grip width per part label).
  - Whether the Ethernet regrasp worked by pressing the latch is not visible at 640×480.
  - Primitive and progress labels have no ground truth in the dataset. Only a watcher check (rule 4 under Checks) guards them.

**2026-10-25, claude-opus-5.5 stand-in, blind cold read of gate episodes 68, 136, 140** (16.5 recording minutes, hand camera only; labels: `runs/claude/run-standin-1/footage_blind/ep*.json`; calls: `runs/calls/20261025.jsonl`, caller `claude-standin-blind-A`; native-before-zoom records `a-ep068-*`, `a-ep136-outcomes`, `a-ep140-*` in `runs/claude/run-standin-1/decisions.jsonl`). Not scored when written; a separate watcher scores it.

- **Held.**
  - Signals first worked on unseen recordings. The gripper cycles gave every segment skeleton (61 cycles). A label-free survey of the Fz sign showed each of the 24 gate episodes is all push or all pull, so assembly vs disassembly is known before any pixels.
  - Every failure call came from signals before video, and zooms only confirmed them. Three signatures: closes to < 1 mm (nothing grasped), opens at the board with no free lift, and release followed by an in-place regrasp.
  - A `probe_video` grid with `times` at lift + 0.8–1 s (one cropped tile per cycle, 20–21 tiles) named most parts in one image. 1–3 more zoom grids per episode settled the rest. Cost: 33 tool calls and 20 images for the three episodes.
- **Broke or misled.**
  - The grip-width table names peg size (≈15/11/7/2.5 mm for index 4/3/2/1 on 16 pegs) but not connectors. The same connector was held at 2.5, 7.0 and 10.0 mm (Ethernet), 7.1, 9.4 and 15.3 mm (USB), and 8.0 and 38 mm (D-SUB) in different episodes. Name connectors from the target seen in the hand view: the RJ45 flange, USB bracket or bayonet sleeve.
  - Round vs square pegs needs the holder's hole shape or the peg's top face in the hand view.
  - EE height at the board is not a part cue: one socket type sat 33 mm higher in one episode than in another, and a call made from height was wrong.
  - "Lift start = first z rise" fires during slow pulls (a twist-lock part crept 5 mm before it came free). For Remove, use the onset of the fast rise (> 8 mm in 0.5 s).
  - "No z travel during a nut twist" is not a failure signature. In one recording a nut held −6 N of pull for its first +100° of unscrewing with EE z flat (`runs/footage/ep140/signals_0440.00-0446.00_force-pos-orient.png`). A tug before release that meets resistance (Fz −3.7 N with z +0.6 mm) is a better cue for engagement.
  - F/T can freeze to a constant at the end of a recording (|F| exactly 5.9 N for the last 11 s of one episode). Treat a perfectly flat trace as missing data, not contact.
- **Situations the rules did not cover.** I labelled them as follows; all are unverified until scored.
  - Open-finger presses after release on gears and a connector (5–19 N, up to −10.8 mm): the Insert runs to the end of the press.
  - Gear released unmeshed, then regrasped in place, turned 15–34° and sunk 4–7 mm: a failed Insert plus a new Insert, with no Pick.
  - A nut unscrewed off its bolt during an assembly episode: Remove + Place.
  - A close on nothing: a failed Pick or Remove ending at the re-open.
- **Cost.** About 13k unique tokens per recording minute (10.5k–19.3k per episode, falling with length). This is counted once per token from the session counter, with setup excluded. Billed input with the context re-read every turn is about 1M per recording minute.

**2026-10-25, claude-opus-5.5 stand-in, refinement on train recordings only** (caller `claude-footage-refine`; tools `harness/tools/footage_id.py`; evaluation `runs/footage/refine-20261025/eval_train.json`, scripts `runs/footage/refine-20261025/scripts/`). No gate label was used. The watcher's gate files (`runs/claude/run-standin-1/a-watch/success_bacc_gate.json`, `claims_grounded_gate.json`) were read only to see which kinds of error to look for.

- **Why.** The first gate test (68, 136, 140) scored segment F1 0.961 but success balanced accuracy 0.78 (4 of 7 failures caught) and claims_grounded 0.667. Of the 4 sampled claims the watcher rejected, 2 named the wrong connector (USB vs Ethernet), 1 read a gear regrasp as a failed plus a new Insert, and 1 generalised over all pegs.
- **Protocol.** 89 train recordings: 60 tuning, 29 held out (episode index % 3 == 0; 778 segments, 96 failures). Hole layout and width table fitted on the tuning recordings only, thresholds chosen on them. "Before" is a deterministic proxy of the old rules: the same cycle skeleton, part from hold width, Remove failed only without a z rise, Insert failed only on a regrasp in place, presses merged, first verb ending lift + 0.75 s. "After" is the `end_state_check` draft. Scored like `score_labels`, plus the watcher's balanced accuracy (every labelled segment is an attempt). The held-out part errors were inspected once; the same-object and width fallbacks were added after that.
- **Held-out result.** Segment F1 0.926 → 0.961. Task accuracy 0.628 → 0.951. Part accuracy 0.640 → 0.959. Success balanced accuracy 0.649 → 0.892: failures caught 31/96 → 77/96, successes right 665/682 → 669/682. Tuning recordings: F1 0.938 → 0.970, part 0.647 → 0.974, balanced accuracy 0.699 → 0.908 (171/204 failures). The proxy names parts from width alone; the old reader also had video, so its real part accuracy was higher than 0.64.
- **What each change bought** (held-out balanced accuracy with that rule switched off, against 0.892):
  - peg open-hand return: 0.833;
  - carried-away Remove: 0.840;
  - retreats as failed Inserts: 0.860;
  - 45 mm regrasp radius: 0.887.
  Part accuracy was 0.931 from hole location alone and 0.959 with the same-object and width fallbacks for table-only cycles.
- **Location names the part.**
  - The label-free board pose is the mocap pose refined by the recording's own grasp and seat points. It agrees with the labelled `board_from_contacts` fit to 0.6 mm median (p90 1.6 mm, worst 20.9 mm, 85 train recordings). 2 of 89 needed the RANSAC fallback; one of them had a mocap pose 388 mm off.
  - At-hole names matched the labels on 1,186 of 1,194 train cycles, and on 13 of the 15 where the hold width disagreed.
  - In 3 square-peg cases, hole and width agree with each other against the label (episode 15 at 513 s, 120 at 213 s, 144 at 397 s). These are likely label errors.
- **Rules the first gate read flagged, checked on train.**
  - Connector grip widths: widths are tight per part (MAD 0.2–0.6 mm), but connectors have second modes (D-SUB 15.5 or 38.7 mm, Ethernet 6.9–10.1, USB 2.3–9.4), and round and square pegs of one index sit ≤ 0.6 mm apart. Width is now a tie-break only.
  - Part height: dropped. The xy location of the grasp or seat names the hole.
  - Lift onset in slow pulls: use the fast rise. On the tuning recordings, 11 of 36 failed Removes rose more than 15 mm (up to 157 mm) while still within 25 mm of the hole, so a z rise does not mean the part came free. "Carried more than 80 mm from the hole" does (45/51).
  - "No nut travel = failure": removed. Insert outcomes are not separated by seat depth against the part's train seat height (AUC 0.48) or by peak force (0.40). A nut that never caught its thread shows when it is picked up and inserted again (3/4).
  - Gear regrasp in place: one Insert (22/25 success right), not a failed plus a new Insert.
  - Open-finger presses: on gears and connectors they are part of a successful Insert (90 of 94 and 36 of 38 releases with a press). On pegs, an open-hand return after the release makes the release a failed Insert followed by a new successful one (39/51 failures right, 38/38 nudges right).
  - Retreats while holding: each is a failed Insert (76/76). On train, failed Inserts were followed by another Insert 85 times, a Place 39 times and a Pick 33 times.
- **Still missed** (tuning recordings, 33 of 204 failures):
  - gear regrasps in place that the labels split into a failure plus a new Insert (3);
  - Insert→Place with no detected attempt (4);
  - Place failures, not modelled (4 of 7);
  - nut releases that are not re-inserted later (3);
  - closes on nothing that the labels fold into a successful Pick (6 false failures);
  - aborted approaches with no close (13 failed Picks/Removes in all train). Drafting these added false segments on held-out, so the tool only reports them.
- **Cost.** About 1 s of tool time per recording. Two calls (`end_state_check`, then one whole-window `query_signals`) give the skeleton, the parts and a first success call before any image.

## Sources

- REASSEMBLE dataset and labels: `docs/research/reassemble-data-notes.md` [S1–S5 there]. Splits: `playbook/splits.json` (tuning and evaluation used train recordings only).
- Board from contacts (hole layout, point-to-line fit, mocap errors): `playbook/skills/board-from-contacts/SKILL.md`, `twin/board_fit.py`, `runs/claude/run-standin-1/board/layout_contact.json`.
- First gate test and its watcher: `runs/claude/run-standin-1/footage_blind/ep068.json`, `ep136.json`, `ep140.json`; `runs/claude/run-standin-1/a-watch/`.
- Why success comes from physical end state and not from a frame or the model's own "done": models call done falsely (287 of 1,800 LIBERO-10 episodes [llm-libero-2026]) and cannot tell which physical condition is unmet [sensenova-astra-2026] (`docs/knowledge/llm-as-robot-policy.md`); VLM judges get exploited [sole-r1-2026] and progress should follow procedure, not time [procvlm-2026] (`docs/knowledge/learning-signals.md`).
- Train evaluation of the 2026-10-25 rules: `runs/footage/refine-20261025/eval_train.json`.
