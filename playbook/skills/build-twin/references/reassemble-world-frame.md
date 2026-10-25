# build-twin: reassemble-world-frame

The v1 skill `world-frame`, moved here unchanged by skills-v2 (archived at `playbook/archive/skills-v1/world-frame/SKILL.md`; only skill names and paths are updated to where their content now lives). v1 description: Put the cameras, the board and the tool of one REASSEMBLE recording in the robot base frame so twin renders line up with the real footage, calibrated from robot keypoints (FK) and board features and validated on held-out marks (stage B0 foundation; run before build-twin and replay-validate).

Entry point: `build-twin` (`playbook/skills/build-twin/SKILL.md`)

## Goal

For one recording, `twin/calib/<recording>.json` holds:
- every camera (`hama1`, `hama2`, `hand`) with pinhole intrinsics (fx, fy, cx, cy at 640x480), its pose in its parent frame (world for hama1/hama2, `gripper/hand` for the wrist camera) and its video/joint-log latency (`latency_frames`);
- the board pose (`NIST_Board1` frame, plate top at board z = 0): x, y, yaw from robot contacts (board-from-contacts), z from the calibrated cameras;
- the tool: `tcp.hand_offset_m`, how far the physical hand sits along its z axis from where `base.xml` puts it.

Done means the B0 checks (`validation.yaml`, unchanged since v5) pass when an independent watcher re-measures them (`measure_b0`): held-out finger-pad and board keypoints ≤ 4 px per static camera, two-view residual ≤ 2 px, wrist camera ≤ 6 px for the worse of fingers and board. Before handing over, the actor reports the same quantities on marks it held out itself (`keypoint_check`, the hand tool's leave-one-key-out). Edge overlays are diagnostic only (see Lessons: they pass a robot rendered 33 s off).

The robot base frame is FR3 link0 = MuJoCo world = mocap world (`docs/research/reassemble-data-notes.md`).

## Inputs

- Episode data in `data/reassemble/`: joints, gripper, `O_T_EE` every frame, 3 videos (30 fps, one clock).
- `twin/calib/<recording>.json` `board_contact` (from `board_from_contacts`); `runs/<brain>/<run>/board/layout_nist.json` (NIST plate placed on the contact layout, used for hole and plate-corner 3D via `twin/assets/board_nist_plate.json`).
- `twin/scene/base.xml` (FR3 + Franka hand + F/T stack). Since 2026-10-24 21:19 it mounts the hand on episode 0's fitted 47.7 mm stack (`FT_STACK` in `twin/scene/build_base.py`, read from episode 0's `tcp.hand_offset_m`), with the `tcp` site kept on `O_T_EE`. A new recording's fit therefore starts from episode 0's tool length: treat it as a prior and fit the residual. Renders apply only that residual (`twin.world.calib_hand_offset`: the recording's `tcp.hand_offset_m` minus base.xml's, 0 for episode 0).
- Mocap camera poses: marker brackets, often duplicated or swapped. A seed at most.
- Priors: hama1/hama2 are the same webcam (HAMA C-600 Pro), so share intrinsics; the wrist camera is a RealSense D435i colour stream (fx ≈ 615 at 640x480).
- The watcher's held-out frames (`runs/<brain>/<run>/b0-watch/*.json`): keep every fit frame more than 30 frames away from them.

## Tools

Harness `harness/tools/world.py`, code in `twin/camcal.py` (keypoint calibration) and `twin/world.py` (renders). Outputs in `runs/<brain>/<run_id>/world/`.

| Tool | Use |
|---|---|
| `mark_candidates` | For (frame, key) items: project the key with the saved calibration, find dark-blob candidates around it, write a zoomed crop sheet with a pixel grid and numbered blobs. You only say which blob is which key; the blob's centroid becomes the mark. Keys: `left_pad`, `right_pad` (robot, FK), `c00`…`c01` (plate corners), `hole_<part>` (board). |
| `calibrate_static_cameras` | hama1 + hama2 together: render-and-compare of the arm's black joint caps and the finger pads on frames spread over the episode, then bundle adjustment of both poses + shared f, cx, cy (+ optional k1) + latency per camera + tool offset + board plate-top height, with optional marks. 2–3 iterations, search radius 25 → 8 px. Every 3rd frame held out and scored. Writes `static_fit.json`. |
| `calibrate_hand_camera` | Wrist camera in the tool-corrected hand frame: far-field hole marks + near-field pad corners on the open/closed hand medians, f with a soft prior. Reports leave-one-key-out board error and pad-corner error on held-out open frames. Writes `hand_fit.json`. |
| `keypoint_check` | The actor's held-out score: marks never used in the fit → px per camera, two-view triangulation residual, and the same marks scored with the uncorrected twin. |
| `render_overlay`, `real_frame` | Look. `render_overlay` now uses the calibrated board, the NIST plate and the hand offset. |
| `refine_camera`, `check_world_frame`, `save_world_frame` | The old silhouette/edge pipeline and its frozen edge checks. Kept for comparison; do not use as the goal. |

Writing the calibration: `twin.camcal.write_calib(...)` keeps the replaced values under `previous`.

## Approach

**Options, with sources** (ids in `docs/knowledge/references.md`):

| Approach | What it needs | Fit here |
|---|---|---|
| Robot-supported calibration: features on the robot at many poses, calibrate on one sequence, evaluate on another [werner-2018] | features whose 3D the robot knows | **chosen**; the features are the robot's own caps and pads, 3D from FK |
| Robot keypoints + PnP, intrinsics known [dream-2020, ctrnet-2023]; one tracked reference point + FK + PnP [kalib-2024]; offline datasets: temporal PnP then render refinement [caliball-2025] | keypoint detector or tracker | the same idea; detection here is render-and-compare + a brain identifying blobs |
| Render-and-compare of the robot mask [easyhec-2023, splatting-physical-2025] | segmentation, a start in the basin | used before (2026-10-24 morning); the mask/edge loss converged to a wrong pose (Lessons) |
| Hand-eye AX = XB from motion [parkmartin-1994, tsai-lenz-1989] | relative camera motions | failed on the blurry close-range wrist footage (Lessons) |
| Reprojection-error hand-eye and time-offset estimation [koide-2019] | FK-projected points | **used** for the latency: one time offset per camera inside the same bundle adjustment |

**1. Seed.** A native look-at guess (cues below) or the previous calibration; `refine_camera` is not needed. The auto features need a start within about 25 px.

**2. Robot keypoints, automatically (render-and-compare).** For each frame, render the twin at the logged joints with the current camera. For each distinctive robot feature (black joint caps on links 3–6, finger pads), cut the rendered patch (robot pixels only) and find it in the real frame by masked ZNCC within ±R px. The observed 2D is the rendered visible centroid plus the shift; the 3D point is the centroid of the feature's visible rendered surface (depth back-projection), fixed to its link. Keep matches with score ≥ 0.7 (caps) / 0.75 (pads). Skip `blk7` (low scores). This gives about 6,800 correspondences on 595 frames in 75 s (about 4,300 pass the score cut).

**3. Board features, by eye.** Mark the four plate top corners on the temporal-median background image of each static camera (robot-free). Mark them in both cameras so they also give a two-view check. Wrist camera: mark empty holes (after the part's Remove segment ended) with `mark_candidates`; identity by eye, position = blob centroid.

**4. Bundle adjustment** (`twin.camcal.BA`). Both static cameras at once, one parameter set:
   - pose per camera; shared f, cx, cy (same webcam model); radial k1 only if held-out residuals show radial structure;
   - latency per camera: FK evaluated at frame + δ by interpolating the FK points of neighbouring frames;
   - tool offset: every point on the hand or fingers moves along the hand z axis by one shared length. The arm caps do not depend on it, so arm + pads separate it from the camera pose;
   - board plate-top height: board corners are seen by both cameras, so their z is triangulated against the robot-calibrated cameras;
   - Cauchy loss (2 px). Iterate: re-render with the new model (hand offset and latency applied), re-match with R = 8 px, refit.

**5. Wrist camera** (`calibrate_hand_camera`). Only after the tool offset and board height are known: the camera is fixed to the physical hand, so with the wrong hand length its near field (fingers) and far field (board) cannot agree. Fit pose + f (prior 615 ± 20 px, principal point at the centre) to the hole marks plus the inner/outer top corners of the pads on the open and closed hand medians; the pad-corner 3D points are re-rendered each iteration.

**6. Hold out and check.** Your own held-out marks: frames at a different phase (15k + 7) and more than 30 frames from any watcher frame. Mark pads with `mark_candidates` (blob identity by eye). Run `keypoint_check`: px per camera, two-view residual on pads seen by both cameras at the same frame, and the uncorrected-twin score. Wrist camera: leave-one-hole-out and pad corners on held-out open frames.

**7. Save** with `write_calib` (old values kept under `previous`). Record numbers with `record_event(kind="note")`; publish before/after overlays. A separate watcher runs `measure_b0`.

**Native-inference cues that worked** (first pass, 2026-10-24): which way robot +x runs in the image decides the side; the triangle robot base → board centre gives the azimuth (hama1 −26° by eye vs −25°); the arm's apparent size gives distance. On this rig the wrist image has up = +x_hand, right = +y_hand.

**Gotchas.**
- Disable MSAA before segmentation renders; keep one renderer per image size (`twin.world.get_view`). `get_view()` resets the hand offset to 0: set it again after every call.
- MuJoCo depth from `Renderer` with depth rendering on is z-depth along the optical axis (checked on the table plane).
- `w.board_pose(rec)` without the calib returns mocap; pass the calib.
- Two board heights exist. The calib `board` entry takes z from this recording's cameras; replay, starts and `measure_b0` place the plate top at `board_contact` z + `surface_z` (30.3 mm, fitted on episode 0; `twin/replay.py` `ReplayParams`). On episode 1 the two differ by 13 mm (`measure_b0` reports `twin_minus_calib_board_mm`).
- Rendered plate: the 384 mm NIST plate at the board pose plus a darker base slab down to the table, added in memory by `TwinView` (base.xml untouched, replay builds on it).

## Checks

B0 (`playbook/validation.yaml`, since v5), measured by a watcher with `measure_b0` on its own held-out frames and marks:

- `B0.kp_hama1`, `B0.kp_hama2`: median reprojection of held-out finger-pad and board keypoints ≤ 4 px.
- `B0.two_view_px`: median per-view residual after triangulating the same marked points from hama1 and hama2 ≤ 2 px.
- `B0.overlay_hand`: worse of the finger and board/part medians ≤ 6 px.
- `B0.tcp_err_mm` ≤ 2 mm (twin `tcp` site vs logged O_T_EE); `B0.board_err_mm` ≤ 3 mm.
- `B0.overlay_hama1/2` (edge distance): track only.

Actor pre-checks (not records): `keypoint_check` on your own held-out marks, and the hand tool's leave-one-key-out.

`measure_b0` computes FK on `base.xml` (episode 0's tool length) plus the recording's residual tool offset, and scores part keypoints against the board the twin uses (calib `board_contact` + `surface_z`, contact-layout holes). It does not model `latency_frames`, so moving frames carry up to about 2 px of latency error.

## Lessons

**2026-10-24 (evening), episode 0 (`2025-01-09-14-11-58`), claude stand-in actor (caller `world-frame-recal`), robot-keypoint recalibration.** Evidence: `runs/claude/run-standin-1/world/` (`static_fit.json`, `hand_fit.json`, `marks_*.json`, `beforeafter_*.png`), `twin/calib/2025-01-09-14-11-58.json`. Exploratory: episode 0 is the only calibrated recording, and the v5 targets were set on it.

*What the old method got wrong.* The silhouette/edge refinement reached about 1 px edge distance, yet the watcher measured 5.35 px (hama1) and 16 px (hama2) on held-out finger pads, 7.1 px two-view and 33 px on the wrist camera. My own held-out pad marks reproduce this: 5.62 / 15.8 px under the previous calibration. The edge check passes a robot rendered 33 s off (validate-checks Lessons), so it could not see the error. Three causes, found by the keypoint fit:
- **The twin hand is 22.9 mm too far from the flange.** With the cameras fixed by the arm caps alone, the pads were 10–14 px off; one shared tool offset of −22.9 mm (fits ranged −22.9 to −24.4 mm across settings) brings held-out auto pads to 0.8 / 1.1 px. The hand housing outline lines up only with the offset (`runs/claude/run-standin-1/world/beforeafter_*`). The physical stack is then 47.7 mm (REASSEMBLE URDF: 44.5 mm). The configured `O_T_EE` (0.174 m) lies about 23 mm beyond the pad centres. This also explains the wrist camera's near/far disagreement, and the "TCP drops 16–21 mm after first contact" conflict noted in `twin/board.py`.
- **The plate top is 30 mm above the mocap board z** (z ≈ 0.038 m): the plate sits on an aluminium frame. Plate corners triangulated by the robot-calibrated cameras sit 24–34 mm above the mocap plane; the fitted offset is +30.3 mm (±4 mm, from 8 corner marks).
- **The rendered plate was a 400 x 400 x 25 mm placeholder**, 6–12 mm off in x. The silhouette fit balanced robot against that plate.

*Results (actor's held-out numbers; watcher pending).*

| camera | previous calib, my held-out marks | new calib, my held-out marks | new, auto held-out frames |
|---|---|---|---|
| hama1 | 5.62 px (20 pad marks) | 1.75 px (p90 3.6) | arm 0.66, pads 0.81 px |
| hama2 | 15.8 px (16 pad marks) | 1.01 px (p90 1.7) | arm 0.67, pads 1.10 px |
| two-view (pads at the same frame, 15) | 4.77 px | 0.43 px | |
| two-view (plate corners, 4) | 2.3–10.5 px | 0.4–1.6 px | |
| wrist: board (leave-one-hole-out, 15 marks / 5 holes) | 33 px (watcher) | 4.89 px | fit 3.2 px |
| wrist: pad inner corners, 146 on 77 held-out open frames | | 4.47 px | |

Scoring the same new cameras with the uncorrected twin (no tool offset): 12.3 / 11.5 px, so `measure_b0` will fail `B0.kp_*` until base.xml (or the watcher's FK) applies the offset. (Resolved: base.xml carries the offset since 21:19, and the watcher's `measure_b0` gave 1.63 / 1.51 px at 22:09, `runs/status/checks.jsonl`.)

*Intrinsics.* hama1/hama2 share f = 554.0 px, principal point (317.1, 222.2) (se about 1 px). A separate fit gives f 555.9 / 556.9 and cy 208 / 231, and held-out error does not improve, so they stay shared. A free k1 (−0.06 to −0.13) did not improve held-out residuals, so the model stays pinhole (and matches `measure_b0`, which ignores distortion). Wrist camera: f = 579.2 px (prior 615 ± 20), principal point fixed at the centre; a free principal point drifts 30–45 px in cy without better held-out error.

*Latency.* The video lags the joint log by 0.78 frames (hama1) and 1.05 frames (hama2): 26–35 ms, which is about 4.5 mm at the median moving speed of 0.135 m/s. The silhouette chamfer vs time-shift test gave the same (−1.16 / −0.81 frames). It is negligible on still frames and up to 2 px on moving ones. `measure_b0` does not model it.

*Board.* Contact x, y, yaw + camera z. With the robot-calibrated cameras, the hole projections land on the real board features in both views without being fitted. The triangulated plate sides come out 374–399 mm (should be 384), so corner marks are good to about ±4 mm. Wrist hole residuals are systematic per hole (±3 mm), which is about the layout's along-pad accuracy and the board-z uncertainty.

*What a brain did vs code.* About 15 image reads: identifying blobs (which dark blob is which pad or hole) and marking 8 plate corners. Every position came from code (blob centroids, ZNCC), because reading absolute pixel coordinates off zoomed crops varied by 2–5 px between two readings of the same crop. Use the eye for identity, code for position.

*Cost.* `calibrate_static_cameras` from the previous calibration: 376 s for 3 iterations on 557 frames × 2 cameras, and it reached the same solution as the hand-driven iterations (positions within 3 mm, f 554.0 vs 554.9). `calibrate_hand_camera`: 9 s. No LLM API calls.

**2026-10-24 (morning), first pass, claude stand-in (callers `world-frame-agent` / `world-frame-watcher`).** Evidence: `runs/world/refine/*.json`, `runs/world/overlays/`.
- Native inference got the side, the up direction and the position within 8–10 cm and 4–12°: hama1 (1.45, −0.70, 0.55) vs refined (1.525, −0.718, 0.533); hama2 (1.40, 0.35, 0.60) vs (1.560, 0.268, 0.647). It left 6–8 px edge error; `refine_camera` took it to 1–2 px edge distance. That later turned out to be a wrong pose (see above).
- Depth/focal valley: hama2 with a free focal fitted 518 px, 12 cm closer, with the same loss. Silhouettes alone cannot separate focal length from distance; keypoints at many depths can.
- Mocap camera poses: Hama1/Hama2 2 mm apart in this file (duplicated). Use mocap only as a seed.
- TCP: FK + 0.174 m reproduces `O_T_EE` to 0.24 mm median. That certifies the configured EE frame, not the physical fingertip: the cameras put the pads 23 mm short of it.
- Target-free hand-eye (SIFT + essential matrix + Park–Martin, `twin.world.hand_eye_from_motion`) failed: 45 of 400 frame pairs usable, 2.4° rotation residual.
- Wrist camera with the uncorrected hand: near field alone gave pad IoU 0.73 / 0.83 but parts 51 px off; adding parts pulled f to 540 and pad IoU to 0.54. Both are symptoms of the 23 mm hand error.

**Open**
- Fix `base.xml`: done 2026-10-24 21:19, `FT_STACK = 0.0706 + hand_offset_m` (0.0477 m), `tcp` site kept on `O_T_EE` (`twin/scene/build_base.py`); B1 reran after the change (build-twin Lessons).
- `measure_b0` applies the residual tool offset (2026-10-25); `latency_frames` is still not modelled.
- Second recording: episode 1 was calibrated on 2026-10-25 by the confirmatory run (`runs/claude/run-confirm-ep1/`, reconstruct-scene Lessons); its watcher re-measure is pending.
- Wrist-camera far field rests on 15 hole marks; more holes and plate edges would tighten it. Its own latency is not fitted (still frames only).
- Board z rests on 8 corner marks (±4 mm); sub-pixel plate-edge lines would tighten it.

## Sources

[werner-2018], [dream-2020], [ctrnet-2023], [kalib-2024], [caliball-2025], [easyhec-2023], [splatting-physical-2025], [koide-2019], [parkmartin-1994], [tsai-lenz-1989], [hagemann-2021] (a low residual is not a correct calibration), [autofocus-2024]: full entries in `docs/knowledge/references.md`; synthesis in `docs/knowledge/calibration-and-localization.md`.
