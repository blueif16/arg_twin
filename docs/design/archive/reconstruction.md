# Step 1: reconstruction

> **Status: proposal under discussion. Nothing in this file is agreed yet.**

Draft v1, 2026-10-24. The goal is real-to-sim at the highest quality and efficiency: a MuJoCo twin of the REASSEMBLE cell whose renders line up with the real camera footage and whose physics reproduces the real outcomes. Training happens inside this twin ([policy-learning.md](policy-learning.md)). There is no device, so nothing goes back to a real robot (D5).

## How the work is split

1. **Plan.** Argon reads the footage and the known assets (Franka model, NIST board CAD, REASSEMBLE holder STLs). It writes two things:
   - the list of reconstruction units
   - the shared interfaces every unit builds against: frame conventions, names and units
2. **Build in parallel.** Each unit gets its own builder. Flash builds the routine units; Argon builds the hard ones, such as contact physics.
3. **Check each unit.** A separate watcher instance and the unit's code checks review each unit. The builder never grades its own work.
4. **Integrate.** An integrator assembles the units and runs the global checks:
   - a composite overlay on all three cameras
   - the replay set

   A global failure is routed back to the unit responsible, and only that unit reopens.

## Units and checks

| Unit | Built from | Check |
|---|---|---|
| **Cameras** | `hama1` and `hama2` are fixed; `hand` is on the wrist. The robot is the calibration target: render the Franka at the logged joint angles, then solve for the camera pose whose robot silhouette best overlays the real one across many frames. Solve the hand camera's flange-to-camera offset the same way | Robot silhouette IoU and edge distance in px, per camera |
| **Robot** | Menagerie Franka | Forward kinematics vs logged `ee_pose` (mm) |
| **Board base** | NIST board CAD plus REASSEMBLE holder STLs | Reprojection of board edges and holes (px) |
| **Part families** (one worker each: pegs, gears, connectors, …) | CAD / STL | Silhouette IoU at rest poses; replay outcomes for that family |
| **Appearance** | Fixed cameras: real background plates taken from frames where the area is empty. Moving hand camera: real textures projected from the calibrated views onto the CAD meshes. Nano Banana only to fill gaps | Masked photometric difference per camera |
| **Timing** | — | Video-to-proprio offset by motion cross-correlation; F/T contact onset vs visual contact |
| **Contact physics and controller** | Clearance, friction and `solref`/`solimp` per family; impedance stiffness and setpoint reconstruction ([policy-learning.md](policy-learning.md#setpoint-reconstruction-comes-first)) | Replay outcome agreement, including the real failures; F/T trace error |

Notes:

- **The robot is the calibration target.** Its geometry is known exactly and its joint angles are logged on every frame, so every frame is a calibration image. No extra marker is needed.
- **Background plates beat rendering for fixed cameras.** The two fixed cameras see the same background throughout, so the twin only renders what moves (robot, parts) and composites it over the real plate.
- **Projected textures** give the moving hand camera real appearance without a splat scan. Splats stay an optional visual layer ([argon-twin-loop.md](argon-twin-loop.md#demo-visuals)).
- **Contact needs the outcome check.** Geometry and appearance checks can all pass while pegs still slide in where the real ones jammed. The replay set, including the 516 real failures, is what catches that.

## Showcase

Real and sim overlaid on all three cameras, with the difference heatmap fading round by round as units are fixed. See [demo.md](demo.md).
