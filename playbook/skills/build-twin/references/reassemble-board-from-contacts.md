# build-twin: reassemble-board-from-contacts

The v1 skill `board-from-contacts`, moved here unchanged by skills-v2 (archived at `playbook/archive/skills-v1/board-from-contacts/SKILL.md`; only skill names and paths are updated to where their content now lives). v1 description: Stage B0/B. Place the task board (pose per recording and every hole) in the robot base frame from the robot's own contact events (TCP when a part sits in its hole), with a held-out error in mm, and compare with mocap and cameras. Use it before build-twin and replay-validate whenever holes must be right to better than a camera pixel (about 2 mm at the board).

Entry point: `build-twin` (`playbook/skills/build-twin/SKILL.md`)

## Goal

For any REASSEMBLE recording (not exam):

- a board pose (x, y, yaw in the robot base frame, mocap `NIST_Board1` frame convention) fitted only from that recording's contact events, with a 1-sigma;
- every hole's position under that pose;
- a held-out error in mm that a watcher can rerun: leave one part out inside a recording, and a split of recordings that never touched the hole layout;
- the same error for the mocap pose (and the camera pose, if one exists), on the same events.

Done means `board_err {split: gate}` gives a held-out error the watcher accepts, and `board_from_contacts` has written `runs/<brain>/<run_id>/board/<recording>.json` for the recordings the twin will use.

Why contacts: the static cameras resolve about 2.2 mm per pixel at the board, mocap board poses carry errors of several mm per recording (Lessons), and peg clearances are 0.05–0.25 mm. The logged tool tip (`O_T_EE`, 0.174 m from the flange; a configured point about 23 mm beyond the finger-pad centres, world-frame Lessons) is good to about 0.1 mm, and every successful Insert ends, and every Remove starts, with a part sitting in its hole between the fingers.

## Inputs

- Robot signals per episode: `observation.state.ee_pose` (TCP, w-first quaternion), `observation.state.gripper_position`, `observation.force.base`, segment labels. Local data is `data/reassemble/data/`. Gate episodes come from HF parquet downloaded to `data/reassemble-gate/data/`, which is read as data only, with `python -I`. Facts and caveats: `docs/research/reassemble-data-notes.md`.
- Splits: `playbook/splits.json`. The layout uses train only. Gate is the held-out test. Exam is refused by the tools and by `twin/board_fit.py`.
- For comparison only:
  - mocap `data/reassemble-tudata/poses/<recording>_poses.json` (`NIST_Board1`)
  - the camera triangulation a watcher writes to `runs/<brain>/<run_id>/b0-watch/board_camera.json`
  - calibrated cameras in `twin/calib/<recording>.json`
- For a layout cross-check: the NIST laser-plate holes in `runs/scene/experiments/stl_holes.log` (from `vendors/nist-taskboard`).

## Tools

| Tool | Use |
|---|---|
| `board_contact_layout` | Fit the rigid hole layout (board frame) on all train recordings. Also checks the 8 pegs against a 75 mm grid and places the NIST laser plate with the layout (reflection allowed). Writes `layout_contact.json` and `layout_nist.json`. |
| `board_from_contacts` | One recording, using a layout fitted without that recording. Returns pose, sigma, per-hole world xy and per-event residuals (closing-axis residual and pad offset), plus deltas vs mocap and camera and a first-half/second-half drift check. Writes `<recording>.json` and a `_top.png` (overview plus one zoom per part). `exclude_parts` leaves parts out; `layout: nist` uses the CAD layout. |
| `board_err` | The held-out measurement. `split: gate` uses the train layout. `split: train` uses a layout refitted without each recording (slow, about 4 min). Leave-one-part-out inside each recording. Writes `board_err_<split>.json`. |
| `recording_info`, `query_signals`, `probe_video` | Look at the segments, gripper and force around an event when a residual looks wrong. |
| `publish_media`, `record_event` | Report. An actor never calls `record_check`; the watcher runs `board_err` itself. |

Core code: `twin/board_fit.py` (events, layout, fit, evaluation) and `twin/board_viz.py` (plots). The CLI is `python -m twin.board_fit {layout | fit <recording> | eval --split gate}`. Events are cached in `runs/cache/board_events/`.

## Approach

Options, with what to try first and how to tell whether each works.

1. **Pick the moments a part is in its hole.**
   - *Insert seat:* in a successful Insert while holding (finger sum < 75 mm in the first third), take the deepest TCP within the last 3 s before the fingers open by 3 mm beyond the holding width.
   - *Remove grasp:* in a successful Remove, take the grasp that is still closed at the segment end, at the frame where the fingers finish closing (width within 0.5 mm of the hold width). If you take the first closed frame instead, you catch half-closed approaches at 60–75 mm width and get 100–500 mm outliers.
   - Contact onsets (|F_base| > 5 N) and failed-insert deepest points are diagnostics only. A failed insert is where the hole was not.
2. **Model the in-hand offset. This is the main decision.**
   - The fingers of the Franka hand are coupled and symmetric, so they centre a part across the pads (along EE y, the closing axis). Along the pads (EE x) the part can sit anywhere on the about 2 cm pad, and it can pivot about the closing axis.
   - So each event says: the hole lies on the line through the TCP along the horizontal EE x. The model is `p = T·h + u·d + ε`, with u free per grasp under a N(0, 6 mm) prior that is marginalised, and ε about 1 mm across the pads.
   - This is the per-grasp "rest pose" variable of factor-graph in-hand estimators [S5, S6] reduced to one dimension. Slip under misalignment torque [S3] is why the gripper pose is not the part pose.
   - Alternatives:
     - isotropic point-to-point: worse, see Lessons;
     - one shared u for all events of one grasp: rarely useful, since a grasp usually has one seat;
     - a particle filter over contact configurations [S7, S8]: worth it only when you do not know which hole an event belongs to.
3. **Fix a rigid hole layout and keep it out of the test.**
   - Fit the hole positions jointly over train recordings, with each recording's mocap pose as the start and a small per-recording correction (dx, dy, dyaw).
   - The correction has a zero-mean prior (10 mm, 2°), which fixes the gauge to the average mocap frame. This is bundle adjustment. Registration from peg-insertion poses used as fiducials is NIST's method [S2].
   - Cross-check the layout against known geometry: the NIST 75 mm hole pitch, and the NIST laser plate with reflection allowed. If the residuals there are about 1 mm, the layout is real, not an artefact of the fit.
4. **Fit one recording.**
   - Start with a 2-point RANSAC over per-part medians against the layout, then refine x, y and yaw by robust least squares on the line constraints.
   - Cauchy loss at 2 sigma is the default. Geman–McClure [S1] works as well. A plain L2 loss does worse.
   - You need events from at least 5 parts spread over the board. Yaw and the in-pad direction are weak when every pad line points the same way, so check the returned sigma.
5. **z and tilt.** Contacts barely pin z, because grasp height varies 2–7 mm (MAD) per part and remove grasps sit about 20 mm lower than insert seats. Take z from mocap or from build-twin's stack, and roll and pitch as 0. The plate top sits 30.3 mm (±4) above the mocap z (camera triangulation on episode 0, world-frame Lessons; `surface_z` in the twin).
6. **Compare and look.**
   - Compare the contact pose with mocap, and with the camera pose if `board_camera.json` exists.
   - Run the drift check (first half vs second half of the events).
   - Open the `_top.png` zooms. Every finger-pad line should pass through the contact hole (+). If a source's hole sits off the lines, that source is wrong for this recording.

## Checks

The watcher runs these. Hyper-parameters are frozen before any gate recording is read (`PARAMS` in `twin/board_fit.py`, chosen on train in `tuning_train.json`).

| Check | Command | What it measures |
|---|---|---|
| Held-out recordings | `board_err {split: gate}` | Layout from train only; inside each gate recording, the board is fitted without one part and that part is scored. Reports `contact_close_mm` and `mocap_close_mm`: per part, the median across-pad distance of its events to the predicted hole, free of in-hand offset. Also reports `b0_board_err_mm` in two readings: per recording (xy to the median release TCP) and pooled per part over recordings. |
| Train, no circularity | `board_err {split: train}` | The layout is refitted without each recording, then leave-one-part-out. Use it to choose hyper-parameters, never gate. |
| One recording | `board_from_contacts {recording, exclude_parts: [p]}` | Leave-one-part-out by hand, and look at the zooms. |
| Layout sanity | `board_contact_layout {}` | `grid_check.fixed75_rms_mm` and `nist_fit.rms_mm` should be about 1 mm. Mocap corrections show how far mocap is from the contact frame. |
| Integrity | tool layer | No exam recording is read. Gate is never used for fitting or tuning. |

Validation note: `B0.board_err_mm` as written (per-part median since v2) includes each grasp's in-hand offset along the pads (median about 3.5 mm), so it cannot reach 3 mm per recording even with a perfect board. The across-pad leave-one-part-out error is the board-only part of it.

## Lessons

**2026-10-24, claude-opus-5.5 stand-in (caller `board-contacts` in `runs/calls/20261024.jsonl`; outputs in `runs/claude/run-standin-1/board/`).**

**Data**
- 89 train and 24 gate recordings were on disk. Train events: 540 insert seats and 502 remove grasps. Gate events: 138 insert seats and 196 remove grasps.
- Extracting every episode takes about 1.5 s in total with pyarrow filters. Every HF data file mixes splits, so filter by `playbook/splits.json` before reading.

**In-hand offset (train layout fit, 1,025 events, `layout_contact.json`)**
- Across the pads the residual is 0.75 mm median (p90 3.0). Along the pads the offset u has MAD 3.6 mm, median |u| about 3.5–3.9 mm and p90 about 9.5 mm.
- By kind: insert seats 0.85 mm across the pads, remove grasps 0.69 mm.
- The tool tilts 3–4° at the event, mostly about the closing axis (median 3.1° about EE y vs 1.8° about EE x), which fits a part pivoting between flat pads.
- Without the per-recording mocap correction, the across-pad residual is 2.4 mm. So most of the old "6–9 mm per-release scatter" (build-twin Lessons) was in-hand offset along the pads plus mocap error, not hole scatter.

**Mocap is the weak link**
- To make all contacts agree, each train recording's mocap board pose had to move 7.0 mm median (p90 11.5 mm) and 0.31° in yaw.
- On gate, contact and mocap holes differ by 6.7 mm median.
- Mocap failures that contacts caught:
  - gate ep 117 (`2025-01-13-15-11-07`): mocap is 82 mm and 31° off. Its pose file repeats the board pose of ep 55 and ep 11, recorded earlier. The contact pose (−112.3°) matches the mocap and contact poses of ep 103, recorded 7 minutes later, within 4 mm and 0.1°. So the board had moved and the pose file is stale.
  - train ep 49: mocap flipped (tilt 172°).
- Gate ep 42 differs by 4.3° in yaw, but it has only 4 parts and a contact yaw sigma of 2.4°, so this is not a confirmed failure.

**Held-out errors (`board_err_gate.json`, `board_err_train.json`)**

| Scheme | Across the pads: contact | Across the pads: mocap | B0 per recording (contact / mocap) | B0 pooled over recordings (contact / mocap) |
|---|---|---|---|---|
| gate, 24 recordings, 330 recording-parts | 1.23 mm (p90 3.85) | 4.08 mm (p90 8.85) | 5.58 / 7.99 mm | 2.37 / 4.45 mm |
| train, leave-one-recording-out, 85 recordings, 1,033 recording-parts | 1.28 mm (p90 4.0) | 3.61 mm | 5.87 / 8.47 mm | 1.25 / 1.54 mm |

- Recordings with only 3–4 parts are unreliable: gate ep 29 with 4 parts gave 5.5 mm.
- First-half vs second-half fits agree to 2.6 mm median (p90 5.2) on gate, which is within half-fit noise. There is no sign of the board moving within a recording.

**Hyper-parameters (train only, `tuning_train.json`)**
- Across-pad error by setting:
  - line model with Cauchy loss: 1.28 mm
  - line model with Geman–McClure: 1.32 mm
  - line model with linear (L2) loss: 1.61 mm
  - isotropic point model (4 mm): 1.61 mm
- The pad prior (3, 6 or 15 mm) barely matters (1.28–1.29 mm). Frozen: line model, Cauchy, sigma_close 1 mm, sigma_pad 6 mm.

**Layout cross-checks**
- On the 75 mm grid, the 8 contact-layout pegs fit with 0.51 mm RMS at fixed 75 mm pitch, and a free pitch comes out at 75.2 × 75.1 mm. The old release-median layout gave 77.3 × 74.8 mm at 2.9 mm RMS.
- All 17 holes match NIST laser-plate features at 0.76 mm RMS (max 1.45 mm, scale 0.998) under one mirrored rigid transform:
  - peg holes;
  - the 9 mm BNC hole and the 17 mm M16 hole;
  - the three 6.6 mm gear-shaft holes;
  - the midpoints of the connector housings' screw-hole pairs;
  - the centre of the waterproof socket's 4-hole pattern.
  Fitting on the 8 pegs alone predicts the other 9 parts at 0.80 mm RMS.
- The earlier conclusion in `reassemble-data-notes.md` that the layout does not match NIST came from the STL assembly part centres, without reflection.
- The NIST-placed layout scores 1.45 mm across the pads on gate, against 1.23 mm for the contact layout. The contact layout absorbs per-part grasp habits; the NIST layout gives exact CAD hole positions for build-twin.

**Peg sizes**
- Grip width at the event (median over about 30 grasps each, insert / remove):
  - round pegs 1–4: 2.2/3.3, 6.5/7.2, 10.7/11.2, 14.8/15.0 mm;
  - square pegs 1–4: 3.1/3.5, 7.5/6.8, 11.3/11.1 (also 7.6 when held across the short side), 15.3/10.4 (also 9.7) mm.
- The hole positions give the same answer on their own: the 8 pegs land on the NIST holes of 4/8/12/16 mm round and 4×4 / 8×7 / 12×8 / 16×10 keys in label order, and the next-best grid symmetry matches 4 of 8.

**Cameras**
- On ep 0, hama1 frame 0 with both poses projected (`ep000_hama1_f00000_holes.png`): the 9.6 mm contact–mocap difference shows as only about 1.5–4 px, partly along the view ray. A by-eye read cannot pick a winner, so use the watcher's triangulation.
- The world-frame board-outline check is not independent of mocap: the plate outline came from the old layout in the mocap frame.

**Open**
- Camera triangulation as a third source.
- The contact z sits 3–4 mm above mocap z (median); the cause is unknown.
- Model per-part closing biases for asymmetric connectors.
- Use failed inserts as negative evidence.

## Sources

Knowledge-base ids (`docs/knowledge/references.md`, synthesis in `docs/knowledge/contact-localization.md`): [kim-2023] = S1, [cheok-2018] = S2, [chen-2022] = S3, [drigalski-2020] = S4, [vandermerwe-2026] = S5, [kimrodriguez-2022] = S6, [taguchi-2010] = S7 (not read), [chhatpar-2005] = S8 (not read).

- [S1] Kim, Kramberger, Buch, Sloth, "Contact-Based Pose Estimation of Workpieces for Robotic Setups", ICRA 2023, doi:10.1109/ICRA48891.2023.10161465. Contact points and force normals, robust Geman–McClure cost, 0.51 mm / 0.55°.
- [S2] Cheok et al., NIST IR 8198r1 (2018), "Improving Automated Insertion Applications by Restoring Rigid-Body Condition in Point-Based Registration", https://doi.org/10.6028/NIST.IR.8198r1. Hole positions measured by inserting a peg with the robot, then point-based rigid registration; error is judged at held-out targets (TRE), not at the fitted fiducials (FRE).
- [S3] Chen et al., "Insights from an Industrial Collaborative Assembly Project", arXiv:2205.14340. Misalignment torque during insertion makes the part slip in the gripper.
- [S4] von Drigalski et al., "Contact-based in-hand pose estimation using Bayesian state estimation and particle filtering", ICRA 2020, doi:10.1109/ICRA40945.2020.9196640. The dual problem: known environment, unknown in-hand pose.
- [S5] Van der Merwe, Ota, Berenson, Fazeli, Jha, "Simultaneous Extrinsic Contact and In-Hand Pose Estimation via Distributed Tactile Sensing" (TacGraph), RA-L 2026, arXiv:2512.23856. One in-hand rest pose per grasp in a factor graph.
- [S6] Kim, Rodriguez, "Active Extrinsic Contact Sensing: Application to General Peg-in-Hole Insertion", ICRA 2022, arXiv:2110.03555. Factor graph over gripper, object and contact.
- [S7] Taguchi, Marks, Okuda, "Rao-Blackwellized particle filtering for probing-based 6-DOF localization in robotic assembly", ICRA 2010. Verified by the knowledge-base agent [taguchi-2010].
- [S8] Chhatpar, Branicky, "Particle filtering for localization in robotic assemblies with position uncertainty", IROS 2005. Verified by the knowledge-base agent [chhatpar-2005].
- [S9] NIST Assembly Task Board 1 laser plate, `vendors/nist-taskboard/` (hole list in `runs/scene/experiments/stl_holes.log`).
