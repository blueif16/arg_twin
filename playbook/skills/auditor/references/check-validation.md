# auditor: check-validation

The v1 skill `validate-checks`, moved here unchanged by skills-v2 (archived at `playbook/archive/skills-v1/validate-checks/SKILL.md`; only skill names and paths are updated to where their content now lives). v1 description: How any watcher validates a stage's checks before trusting them - re-measure on its own held-out data, prove with known errors and deliberately wrong twins what each check can detect (in physical units), cross-check with an independent measurement, and keep a confirmatory recording that is scored once. Use whenever a check passes for the first time, when a threshold is set or changed, or when an actor's number looks too good.

## Goal

For every check you are asked to watch, you can say, with evidence:

- **the re-measured value** on data you chose yourself, which the actor never saw, next to the actor's value;
- **what the check can and cannot certify**: the smallest error in physical units (mm, degrees, ms) that makes it fail at its current threshold, and which kinds of error it never notices;
- **that a deliberately wrong twin fails** (negative controls), and by how much;
- **whether the threshold should change**, as a proposal with numbers for the orchestrator. Watchers never edit `validation.yaml`.

A check counts as validated only when all four exist in `runs/<brain>/<run-id>/` and are cited in this skill's Lessons.

## Inputs

- `playbook/validation.yaml` (the check's metric text and threshold) and `playbook/splits.json` (never touch exam recordings; `harness.registry.call` refuses them anyway).
- The actor's outputs (for B0: `twin/calib/<recording>.json`). Use their numbers only. Don't read the actor's reasoning: method/status text, refine logs, or the actor skill's Lessons (rules.md: actors and watchers are separate).
- The call log `runs/calls/*.jsonl`. Read only tool **arguments**, to find which frames, episodes and points the actor touched.

## Tools

| Tool | Use |
|---|---|
| `measure_b0` | B0 checks re-measured on seeded held-out frames, kept away from every frame the fit touched, for any calibrated recording. Also scores wrist-camera keypoints you mark (hole keypoints take the twin's contact-layout hole), automatic finger-pad corners, and static-camera robot keypoints (their per-camera medians are `B0.kp_hama1/2`). Files are per recording (`measure_b0_<recording>.json`, marks in `hand_keypoints_<recording>.json` / `static_robot_keypoints_<recording>.json`; never another recording's). `frame_counts` and `thin_evidence` say how many frames each component got; `finger_radius` and `n_board_view_frames` help on short recordings. |
| `b0_known_error` | The same metrics under known camera, board and tool-length errors, plus negative controls (robot rendered at the wrong time, the other camera's calibration). Returns response curves and the smallest error each check fails at. |
| `triangulate_marks` | 3D from marks or straight edges in two calibrated cameras: two-view consistency (px), distance to model points (mm), and plate size and pose. Writes `board_camera_<recording>.json`. |
| `integrity_audit` | Stage G: exam reads and rule violations the call log and run records show, with evidence and suggested values. |
| `real_frame`, `probe_video`, `render_overlay` | Look at frames. For marking, crop and zoom yourself, and keep model projections off the image until the marks are written down. |
| `record_check`, `record_event`, `publish_media` | Report values (the harness judges), raise blockers, and show evidence. |

Outside sources: see Sources.

## Approach

**1. Re-measure, don't re-read.**
- Rebuild the metric from the words in `validation.yaml`, with your own code for the parts that decide the number (edges, boundaries, projection, aggregation). Share only data access and the renderer.
- First check that your projection matches the renderer on a known point (for example, plate corners against the rendered board mask, to within 1 px). That check found a plate rendered 16 mm larger than the calib file says.

**2. Choose held-out data yourself.**
- Reconstruct every frame the actor's tools touched: their deterministic defaults plus the frames named in logged call arguments, including frames used internally (background medians, hand medians).
- Draw a seeded, stratified sample that stays more than 1 s from all of them.
- Record the seed, the exclusion list and the frames. If the actor tuned on the only recording available, label the result exploratory and keep one untouched recording for a confirmatory run, scored once with the frozen tool and seed (Werner et al.: calibrate on one sequence, evaluate on another; mrcal: compare against an independent dataset).

**3. Known-error tests give resolution in physical units.**
- Perturb the twin by known amounts along each physical degree of freedom: camera shift per axis, camera turn per axis, board shift and turn, tool length, time offset. Use both signs, and log-spaced magnitudes from below the expected resolution to well beyond it.
- Report, per family, the smallest magnitude at which the check fails for **both** signs (the worst case). When frames are sampled, also give the magnitude at which the bootstrap 95% lower bound clears the threshold.
- Convert to physical units: px per mm at the working distance is f/Z (static cameras about 0.5 px/mm at 1.2-1.3 m, the wrist camera 2.4-3.8 px/mm at 0.16-0.23 m), and px per degree of camera turn is f·π/180 (about 10 px/deg). Depth moves are 5-10 times less visible than lateral ones.
- ISO 11843 vocabulary: the minimum detectable value belongs to the method, and the critical value comes from a blank series.

**4. Negative controls set the floor.**
- Measure the check on twins that are certainly wrong: the robot rendered 10-33 s off, the other camera's calibration, the board moved several cm.
- If a certainly wrong twin passes, the check certifies nothing at that threshold, whatever the actor's value. A threshold at most about half the "blank" (wrong-pose) value is the minimum.

**5. Cross-check with an independent quantity.**
- A pixel metric can be fooled by clutter. Measure something with different failure modes: reprojection of points whose 3D is known independently (fingertips from forward kinematics, empty holes from the layout), the two-view consistency of a calibrated camera pair (triangulate the same physical point from both, and the residual must be about the marking error), or agreement with a second source (mocap, robot contacts).
- Low reprojection error on the fit data does not prove correct extrinsics. A camera on a robot with known poses exposes the error (arXiv 2402.04686).

**6. Marking discipline (any brain).**
- Mark on clean zoomed crops with a pixel grid. Claude reads crops as images; Gemini can take the clip too.
- Identify features (which hole is which) from labels if needed, but read positions from the clean crop.
- Prefer objects whose 3D the twin defines exactly: empty holes at the plate top, fingertip pads from the model. Avoid tall parts whose height the twin doesn't have.
- Write marks to a JSON file next to the results so the next watcher can rerun them.

**7. Report.**
- `record_check` with the value exactly as defined, even if you have shown the check to be blind, plus a note giving the resolution and the cross-check.
- When the value passes but the cross-check shows the twin is wrong, add a `record_event` blocker naming the skill to revise.
- Publish the held-out overlays and the response-curve figure (not featured unless clearly better than the hero).
- Propose threshold or metric changes with numbers in your report.

## Checks

The watcher's own definition of done, which another watcher can rerun:

- `runs/<brain>/<run-id>/b0-watch/measure_b0_<recording>.json` (or the stage's equivalent) has the seed, the exclusion sources and the frames, and no frame lies within the radius of a fit frame (`frames_too_close_to_fit` is empty). `thin_evidence` is empty, or the record's note says which component had fewer frames than asked.
- The known-error JSON has every family with both signs, a figure with the threshold line, and at least two negative controls.
- Each check has a stated certify/cannot-certify line in physical units.
- Marks and edge inputs are saved as JSON, and the tools reproduce the numbers from them.
- Records carry `brain`, `run_id`, `by` and evidence paths. Nothing touches exam recordings (G.exam_reads stays 0).

## Lessons

**2026-10-24, claude stand-in watcher (`b0-watcher`), stage B0, episode 0 (exploratory: the fit and the targets were both chosen on this recording).** Evidence: `runs/claude/run-standin-1/b0-watch/` (`measure_b0.json`, `b0_known_error.json`, `b0_known_error_curves.png`, `b0_known_error_proposed.png`, `board_camera.json`, the input JSONs). Tool: `harness/tools/watch.py`.

**Re-measure: the actor's values repeat, but that proves little.**
- On 23 seeded held-out frames, ≥36 frames from all 241 frames the fit touched, the static overlay medians re-measured at 1.0 and 1.41 px (actor: 0.96 and 1.91) and the tool tip at 0.26 mm (actor: 0.6 max).
- The overlay median can only take values of the distance-transform lattice (0, 1, 1.41, 2, 2.24 …; 0.955 and 1.37 with the 3×3 mask). Two decimals of agreement there are not two decimals of precision.

**Known errors: the static overlay check (median edge distance ≤ 3 px) is nearly blind.**
- It fails only at 50 mm lateral camera shift or 2° pan/tilt (hama2: 2-5°).
- It never fails for depth shifts up to 50 mm, roll up to 5°, board shifts up to 20 mm or 2°, or tool length up to 10 mm. In hama1 a 20 mm board shift does not change the value at all.
- Negative control: the robot rendered 33 s off (a completely wrong pose) scores 2.2-3.0 px and passes.
- Cause: a white robot and board in a cluttered scene leave a Canny edge within 1-3 px of almost any outline.
- A directional chamfer (edges within ±22° of the outline normal; robot outline only; mean of distances clipped at 10 px), with its threshold at 0.5 × the wrong-pose blank, fails at 5-20 mm lateral shift and 0.3-1° turn. It still never sees the board.

**Cross-check found what the check missed.**
- Finger pads marked on held-out frames, with 3D from forward kinematics, reprojected 5.4 px off in one static camera and 16 px off in the other (34 mm at the board), although both passed the overlay check.
- Triangulating the same board points from both cameras left 7.1 px per view. A consistent pair should leave about the marking error (1-2 px).
- Twenty-five pad marks over 8 frames (`static_robot_keypoints.json`) were enough. The automatic sweeps (about 3 min per run) can only bound what the check would miss.

**Wrist camera (median keypoint px ≤ 6).**
- Resolution: 2.4-3.8 px per mm at the board (0.16-0.23 m) and 9.4 px per degree of camera turn.
- With the twin's own projections as perfect marks, the finger corners fail at 2 mm camera shift or about 0.6°. Part keypoints alone fail at 3 mm board shift in xy, 5 mm in z, 1° yaw, or about 6 mm tool length.
- As defined, the pooled median (24 finger points vs 7 part points) never fails on board or tool errors. Pool with equal weight, or report fingers and parts separately. (Adopted in v5: the worse of the two medians. `measure_b0` reports it as `value` since 2026-10-25; `pooled_median` is the old number.)
- Held-out value: 33 px (holes 35 px, pads 13/33 px) vs the actor's 8.3 px on its own fit frames.

**Tool-tip check (≤ 2 mm).**
- It responds 1:1 to tool length (error + 0.26 mm) and fails at 2 mm.
- It only certifies the twin's kinematic chain against Franka's own O_T_EE, not the physical fingertip. No camera check sees tool length.

**Geometry traps.**
- Check the renderer against your own projection first. That check found the rendered plate is 400 × 400 × 25 mm, while the calib file says 384 mm. (Fixed: renders now use the 384 mm NIST plate, `twin/assets/board_nist_plate.json`.)
- Straight edges parallel to the stereo baseline cannot be triangulated as 3D lines: the back-projected planes here were 2-3° apart. Intersect the image lines in each view and triangulate the intersection as a point.
- In the wrist render, the plate top 8-14 mm above the table looks z-fighted at range, so board outlines in that view are unreliable.

**Proposed changes.** These go to the orchestrator, not into `validation.yaml`. Base the static camera check on held-out robot keypoints (finger pads) at ≤ 4 px, or on the directional chamfer at ≤ 0.5 × wrong-pose blank. Add a two-view consistency check at ≤ 2 px per view. Report the wrist-camera fingers and parts separately. (All three adopted in validation.yaml v5.)

**2026-10-25, maintenance (caller `claude:maintenance-audit`): part keypoints must use the twin's current holes.** The wrist-camera hole marks (`hand_keypoints.json`) carried 3D from the old release-median layout (`twin/assets/board_layout.json`), which sits 1–12 mm from the contact layout the twin now uses. Scored against the calib board with those coordinates, the part median read 13.95 px; with the twin's board (`board_contact` + `surface_z`) and contact-layout holes, the same marks read 5.82 px, and the static-camera numbers did not change (episode 0, `runs/claude/run-standin-1/b0-watch/measure_b0_2025-01-09-14-11-58.json`). When the twin's layout changes, re-derive mark 3D from it; `measure_b0` now does this for marks that name a hole.

## Sources

- Werner, Harrer, Henrich, "Efficient, Precise, and Convenient Calibration of Multi-Camera Systems by Robot Automation", RAAD 2018, doi:10.1007/978-3-030-00232-9_70 (calibrate on one image sequence, evaluate on another): https://eref.uni-bayreuth.de/id/eprint/92830/
- mrcal, cross-validation: compare two independent calibrations; projection uncertainty covers only input noise, not model error: https://mrcal.secretsauce.net/tour-cross-validation.html
- "The Influence of Autofocus Lenses in the Camera Calibration Process" (near-zero reprojection error with consistently wrong extrinsics, caught with a robot-held camera of known pose): https://arxiv.org/abs/2402.04686
- "Significance of Camera Pixel Error in the Calibration Process of a Robotic Vision System", Appl. Sci. 12(13):6406, 2022 (pixel errors hurt hand-eye calibration more than joint-angle errors): https://www.mdpi.com/2076-3417/12/13/6406
- Enebuse et al., "Accuracy evaluation of hand-eye calibration techniques for vision-guided robots", PLoS ONE 2022 (accuracy depends on noise level and motion range): https://pmc.ncbi.nlm.nih.gov/articles/PMC9581431/
- MathWorks, "Evaluating the Accuracy of Single Camera Calibration" (reprojection errors plus parameter standard errors from the least-squares covariance): https://www.mathworks.com/help/vision/ug/evaluating-the-accuracy-of-single-camera-calibration.html
- Twin-S (pixel reprojection error converted to mm via depth/focal length): https://arxiv.org/abs/2211.11863
- ISO 11843-1 capability of detection (critical value vs minimum detectable value of a method): https://knowledge.bsigroup.com/products/capability-of-detection-terms-and-definitions
- Liu et al., "Fast Directional Chamfer Matching", CVPR 2010 (edge matching with orientation, robust to clutter); used for the proposed overlay metric.
- OpenCV multi-view calibration tutorial (triangulation and epipolar consistency between calibrated views): https://docs.opencv.org/5.x/d6/d36/tutorial_multiview_camera_calibration.html
