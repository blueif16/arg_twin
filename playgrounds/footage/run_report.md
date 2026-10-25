# Footage → LIBERO scene, front-end run (2026-10-25, 16:08–16:27 PDT)

This run covers the front end only: there is no practice, and everything is sim-only. Each footage goes through these steps:

1. Argon reads the video: a 1 fps overview, then grid zooms.
2. Argon writes a BDDL file from its read.
3. The BDDL is parsed offline, with one repair retry allowed.
4. LIBERO builds and renders the scene. The builds ran one at a time, and only while MemAvailable ≥ 4.5 GB.
5. A side-by-side image is made.
6. Gate states are saved, each with LIBERO's check value.
7. Argon writes a success check from the video.
8. A gate with no model calls compares that check with LIBERO's check on the saved states.

All Argon calls use `gemini-4-argon` at `thinking_level` medium on the Interactions API. Ground truth and sim states were used only for scoring and were never shown to Argon.

## Results, one row per footage

| Footage | Read | BDDL (parse) | Build | Goal check (LIBERO) | Reward gate vs LIBERO | Placement error vs GT |
|---|---|---|---|---|---|---|
| **put-bowl-in-microwave** (web, 11 s) | "open the microwave, put the white bowl inside it, and close the microwave"; twin = yes; $0.018, 27.5 s | Pass, 0 repairs. Goal `In(white_bowl_1, microwave_1_heating_region)`, `Close(microwave_1)`; a basket distractor; door starts closed, as in the clip. $0.010 | 63.0 s build+reset under load (96.6 s with states) | reset F, goal T, door-open F (`nm_offset` T: the bowl stays inside the cavity) | **4/4** with history reset → door-open state; **2/4** with reset-only history. The reward requires the door to have been opened during the episode, which the teleported goal skips. $0.018 | n/a |
| **HOT3D clip-002066** (egocentric, 5 s) | "pick up and put down the tomato sauce can …"; twin = partial (pitcher and yellow container have no asset); missed the milk and OJ cartons; $0.016, 33.8 s | Pass, 0 repairs. Goal `On(tomato_sauce_1, target_region)`; a `desk_caddy` stands in for the small box. $0.010 | 23.2 s (37.9 s total) | reset F, goal T, offset F, hover F | **3/4.** On the goal state, Argon's check says False and LIBERO says True, for two reasons: the target region overlaps the oversized caddy stand-in, so the can rests on the caddy (LIBERO's region-based `On` still passes); and the reward requires a lift in the history. $0.017 | **2.4 cm** mean over 3 pairwise distances: box–can_end 0.1, box–can_start 2.9, can_start–can_end 4.3 cm. Box = `holder_gray` is an identity assumption. 2 of 6 GT objects mapped |
| **pov-setting-the-table** (web, 51 s) | "set the table by placing the plate, napkin, fork, knife, spoon, and glass"; twin = partial (only the plate has an asset); $0.033, 65.8 s | Pass, 0 repairs (first attempt hit an API disconnect; a plain rerun passed). Goal `On(plate_1, plate_target_region)`. $0.011 | 20.0 s (31.4 s total) | reset F, goal T, offset F, hover F | **4/4.** $0.024 | n/a |

All three builds held LIBERO's goal check False at reset and True at the goal.

**Cost of our arm per footage** (read + BDDL + reward):

| Footage | Cost |
|---|---|
| put-bowl-in-microwave | $0.046 |
| HOT3D clip-002066 | $0.044 |
| pov-setting-the-table | $0.068 |
| **Total** | **$0.158** |

Each Files API upload adds 9–31 s.

## Ours vs one-prompt baseline

The one-prompt arm gets the whole video and a single prompt asking for the BDDL scene and a success check. It has the same asset list, rules and examples as context, and no read step, zooms or repair. Its success check is scored on our saved states.

| Footage | Arm | Parse | Objects and goal match the task | HOT3D region placement error (pairwise) | Success check vs LIBERO on our states | Cost / latency |
|---|---|---|---|---|---|---|
| microwave | ours | pass | yes; same goal pair, plus a basket distractor | — | 4/4 (strict 2/4) | $0.046 (3 calls) |
| microwave | one-prompt | pass | yes; identical goal, bowl only | — | 4/4 (strict 2/4), same behaviour | $0.020 / 15.0 s |
| HOT3D | ours | pass | partial: can and box, pitcher/yellow container declared unsupported | 2.4 cm mean; can start–end pair 4.3 cm | 3/4 | $0.044 |
| HOT3D | one-prompt | pass | partial: can, plus `white_storage_box` and `corn` stand-ins | can start–end pair 6.3 cm; no other pair maps | 3/4 (same goal-state disagreement) | $0.017 / 15.8 s |
| table | ours | pass | partial: plate placement only | — | 4/4 | $0.068 |
| table | one-prompt | pass | partial: plate fixed, glass → `porcelain_mug` as the goal | — | n/a: its check names `porcelain_mug_1`, which our scene doesn't have | $0.025 / 13.5 s |

**Reading:**
- On these 3 short, single-task clips, the one-prompt baseline matches our arm on parse (3/3), on goals, and on success-check agreement. It costs about 40% as much.
- Our read-first route is measurably better only on HOT3D placement: 4.3 vs 6.3 cm on the one shared pair.
- The sample is 3 clips, so treat this as a tie with a small placement edge for ours. The read step's value should show on longer, multi-step footage, where the earlier long-video probe found zooms helped; that wasn't tested here.

## Gotchas found this run

1. **Asset scale isn't in the loop.** Argon places region centres well (2.4 cm), but the chosen stand-in (a 35 cm desk caddy for a ~10 cm box) swallowed the goal region. The fix is to add each asset's footprint to the BDDL-writing context, or to run a post-build overlap check.
2. **LIBERO's `On(obj, table_region)` is lenient.** It passed with the can resting on top of the caddy. An Argon check that requires table contact is stricter, and arguably right.
3. **The gate needs a plausible history.** Checks written from video encode process steps ("the door was opened", "the can was lifted"). A teleported goal with a reset-only history fails them, so the gate should replay a short scripted history or report both variants (done for the microwave).
4. **Harness pitfalls:**
   - `pgrep -f` self-matches the calling shell's command line.
   - `exec -a` breaks the venv's Python prefix.
   - Both are fixed in `fx_run_build.sh`.
5. Builds take 20–63 s while 5 chain sims share the CPU.

## Status page

- Published three side-by-sides on track footage, step B1, brain argon:
  - microwave;
  - HOT3D, with the oversized stand-in noted;
  - table.
- Recorded one `record_event` note. The harness CLI worked after the installs.
- Each web clip appears only as one still inside its side-by-side image.

## Paths

- **Code** (`playgrounds/footage/`):

  | File | Job |
  |---|---|
  | `fx_argon_read.py` | read |
  | `fx_bddl_write.py` | BDDL writing and repair |
  | `fx_parse.py` | offline parse |
  | `fx_run_build.sh`, `fx_build.py` | build and gate states |
  | `fx_lstate.py` | state API; a copy of the reward probe's `lstate.py` that builds from any BDDL |
  | `fx_reward.py` | success check from video |
  | `fx_gate.py` | the no-model-call gate |
  | `fx_oneprompt.py` | baseline arm |
  | `fx_hot3d_err.py` | HOT3D placement error |
  | `fx_summary.py` | collects the summary |
  | `fx_context.py` | shared asset list, rules and examples |
  | `fx_bddl_load.py` | scout loader |
  | `bddl/` | hand-written examples |

- **Outputs:**
  - `runs/footage/summary.json`
  - `runs/footage/argon_read/<slug>/read.json`
  - `runs/footage/scenes/<slug>/`: `scene.bddl`, `write.json`, `build/{agentview.png, agentview_goal.png, states.json, flat_states.npz, build.json}`, `reward.py`, `reward.json`, `gate.json`, `side_by_side.png`, plus `placement_err.json` for HOT3D
  - `runs/footage/oneprompt/<slug>/`: `scene.bddl`, `reward.py`, `call.json`, `parse.json`, `gate_on_our_states.json`, plus `placement_err.json` for HOT3D
- **Clips:** `data/web-footage/<slug>/` (standard YouTube licence; keep local) and `data/web-footage/hot3d-clip-002066/` (converted from the HOT3D JPG frames).
- **Not run:** microwave-video-model (cut). Its read finished before the cut and is in `runs/footage/argon_read/microwave-video-model/`.
