# Calibration and localization: cameras, board and tool in one frame

## Question

How well can we place the external cameras, the wrist camera, the board and the grasped part in the robot frame from REASSEMBLE? The dataset has no intrinsics, mocap poses of marker brackets, and a wrist camera with no mount model. How much pose error can insertion outcomes tolerate, and how should a calibration be validated?

Localizing the board from contact events has its own file: [contact-localization](contact-localization.md). This file covers cameras, the error budget and the in-hand pose.

## What is established, with numbers

- **Small pose errors decide insertion outcomes.**
  - FORGE's official policy (0.114 mm radial clearance peg, gear, M16 nut) goes from 97–99% to 31.8–60.4% at σ = 5 mm fixture-pose noise, and 87–100% of the failures are contact without insertion [pnd-2026].
  - On a real Franka, raising camera-extrinsic error from 5 mm/1° to 10 mm/2° roughly halves gear success [pnd-2026].
  - Pose errors "of the order of 2 mm" can make plug insertion fail [preproc-insertion-2024].
  - A vision-only (AprilTag) peg pose failed 0.5 mm insertions that tactile in-hand correction nearly always completed (10 trials per setting) [ota-2023].
- **The part moves in the hand during insertion.**
  - Misalignment torque makes the part slip in the gripper [chen-2022].
  - Tactile tracking of in-hand rotation gave 38/40 insertions at 0.1 mm clearance [gibbons-2023].
  - Tactile alignment of a slipped, tilted peg gave 90% against 5% without it, at under 0.1 mm clearance [kamijo-2023].
- **Contact-based localization reaches sub-millimetre accuracy.**
  - Force-normal path registration: 0.51 mm and 0.55° [kim-2023].
  - A Rao-Blackwellized particle filter over probe contacts, with a CAD map that carries per-feature uncertainty: 50/50 real insertions after 6–15 contacts (mean 9.8) [taguchi-2010].
  - Particle filtering over a contact C-space map cut position uncertainty by more than 95%, over 50 real lock-key trials [chhatpar-2005].
  - NIST plans probing patches to minimize time and protect the part [kumbla-2018].
- **Validate a calibration on images it was not fitted to.**
  - Werner et al. calibrated on one 75-image sequence and scored on another. A robot-carried target that never reached the image corners left about 20 px mean error after 50 images.
  - A one-off hand-held pre-calibration of the intrinsics over the full field of view brought it to about 4 px, after which extrinsics converged in about six images (Full HD webcams) [werner-2018].
- **A low residual does not make a calibration correct.**
  - Reprojection RMSE "cannot provide a full picture", and the standard covariance underestimates uncertainty under non-ideal conditions. Hagemann et al. propose a bias ratio and an expected mapping error instead [hagemann-2021].
  - A near-zero reprojection error can still hide biased parameters. Here a focus-dependent focal length was exposed by a robot-held camera at known poses [autofocus-2024].
- **Which errors hurt most.**
  - Camera position and orientation are more sensitive to principal-point offset than to lens distortion (simulation) [zhu-2009].
  - In hand-eye calibration, pixel errors in the pattern hurt more than robot joint-angle errors, and more robot poses reduce the damage [khan-2022].
- **Calibration without targets, from the robot's own body.**
  - Render-and-compare of the robot in MuJoCo MJX plus splats cut TCP error from 5.4–32.6 mm to 2.9–18.5 mm (sim-to-sim, noise 0.005–0.03 rad) [splatting-physical-2025].
  - Detecting a legged robot's own feet gave camera-height spread of ±0.12 mm (IQR, best surface). Its ground truth was only tape-measured [hoepflinger-2011].
  - Robot keypoints in the image + their 3D from forward kinematics + PnP: DREAM (single frame, known intrinsics; more frames help) [dream-2020], CtRNet (keypoints + differentiable PnP, rendered-mask self-supervision) [ctrnet-2023], Kalib (one tracked reference point on the robot) [kalib-2024], CalibAll (offline robot datasets: temporal PnP, then render refinement) [caliball-2025]. Mask render-and-compare needs a start in the basin and can stall [easyhec-2023, kalib-2024].
  - The robot-camera time offset can be found as the one that minimises the reprojection of FK-projected points [koide-2019].
- **Twins built by Google use calibrated cameras.** Positions are triangulated from known intrinsics and extrinsics; scale is never estimated from one uncalibrated image [google-argon-robotics-2026].

## What works, what doesn't, what's efficient

- **Works:**
  - held-out evaluation sequences [werner-2018];
  - intrinsics fitted to features that span the whole image [werner-2018];
  - contact-based fixture localization with a robust cost [kim-2023, taguchi-2010];
  - treating the in-hand pose as a separate per-grasp variable ([contact-localization](contact-localization.md)).
- **Doesn't work:**
  - judging calibration by its own residual [hagemann-2021, autofocus-2024];
  - robot-carried features that leave the image corners empty [werner-2018];
  - assuming the gripper pose is the part pose [chen-2022].
- **Efficient:**
  - fix or tightly prior the intrinsics, then fit extrinsics, which need few views when the intrinsics are good [werner-2018];
  - probe or use contacts until uncertainty collapses (about 10 contacts in [taguchi-2010]) instead of exhaustive mapping.

## Disagreements and open questions

- **Pixels or millimetres.** Overlay metrics in pixels (our B0) and task metrics in millimetres do not map one to one without depth and focal length. The literature reports both, and we found no standard conversion.
- **Open: a wrist camera with no mount model and no intrinsics.** Every hand-eye method above assumes a calibration pattern or known intrinsics. Render-and-compare without targets reached only 3–19 mm, and that was sim-to-sim.

## For us

- For us: we found no published tolerance for frame error versus outcome at REASSEMBLE clearances. FORGE-scale tasks (0.114 mm radial clearance) collapse at a few millimetres [pnd-2026], and our `B0.board_err_mm ≤ 3` target is in that range.
  - An open-loop replay into a board placed 3 mm off will jam where the real attempt went in, unless the replay is compliant enough to search, and search can also mask real failures.
  - Measure `B2.outcome_bacc` as a function of injected board offset (0, 0.5, 1, 2, 3 mm) before trusting a single number.
- For us: the Hama cameras are 640×480 webcams with no intrinsics, and the robot and board fill only part of each image. Expect weak distortion estimates [werner-2018].
  - Start from a pinhole prior (principal point at the image centre, focal length from the datasheet FOV), fit extrinsics plus focal length only, then free the distortion terms only if held-out residuals show structure (bias), not just size [hagemann-2021, zhu-2009].
- For us: the mocap poses track marker brackets, not optical centres, and are often swapped (data notes). Use them as an initial guess, never as truth. The board pose from contacts (1.23 mm held-out across the pads) beat the mocap pose (4.08 mm); see [contact-localization](contact-localization.md).
- For us: the wrist camera has no published recipe. Fit the hand-eye transform by render-and-compare on the fingers (always visible) and check it on part keypoints in held-out frames (`B0.overlay_hand`). Report the result as exploratory.
- For us (done 2026-10-24, world-frame skill Lessons): on episode 0, bundle adjustment of both hama cameras on robot joint caps and finger pads found through render-and-compare, with shared intrinsics, latency and tool length as parameters, gave 1.75 / 1.01 px on held-out pad marks, against 5.6 / 15.8 px for the earlier silhouette fit. It also exposed two twin errors that no edge metric saw: the hand sits 23 mm closer to the flange than base.xml, and the plate top is 30 mm above the mocap z. A low edge residual was a wrong pose here, as [hagemann-2021, autofocus-2024] warn.
- For us: a single release's tool tip scatters 6–9 mm because the peg slips in the fingers (validation.yaml v2). This matches [chen-2022, gibbons-2023]. Model the in-hand offset per grasp, and treat in-hand slip during insertion as a source of real failures the twin must be able to produce.
