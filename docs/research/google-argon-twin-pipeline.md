# How Google's Argon demos build a digital twin

Studied 2026-10-23 from the [Google Developers Blog robotics post](https://developers.googleblog.com/gemini-4-argon-for-robotics/) (13 Oct 2026), the [DeepMind cornhole video](https://www.youtube.com/watch?v=9PICEmtfpWU) and the [DeepMind Gemini page](https://deepmind.google/models/gemini/). Google publishes **prompts only**; the harness, tools and code behind each demo are not public. Every number below is Google's own.

## The common recipe

Across all six demos, the twin is **MuJoCo XML plus Python that Argon writes**, never a scan:

1. **Start from a known robot model.** Franka FR3, Apollo, Spot, ALOHA and a Sharpa-hand bimanual robot already have MJCF models. Argon builds only the scene around the robot.
2. **Reconstruct only the task-relevant objects.** "Identify only the manipulation-relevant objects (the grasped target object and the receiving container or landmark)."
3. **Get 3D positions from calibrated cameras.** Positions come from triangulation with known intrinsics and extrinsics, either the robot's stereo head cameras or the calibration supplied with the egocentric video. Argon does not estimate scale from a single image without calibration.
4. **Keep visual geometry and collision geometry separate.** Visuals are procedural meshes. Collision uses compound primitives, or convex-decomposed meshes for concave containers such as bowls, trays and bins, so objects can rest inside them. Articulation is modeled where it matters (Spot's foot-pedal bin, the scale's parallelogram pans), and soft objects are approximated (spring-hinged panels for a plush cap).
5. **Run a verification loop against concrete checks read from simulator state.** Examples: `reward == 1.0`, lift > 2 cm, pelvis drift < 3 cm, 6/6 boxes knocked down, rendered camera views matching real ones, simulated contact forces matching logged tactile forces.
6. **Write "no sim cheating" rules into the prompt.** Objects may move only through contact. After t = 0 there are no model changes, no external forces, no welds, no qpos resets, no disabled collisions and no faked sensor readings.
7. **Present the result in an HTML viewer for a human.** Typical layout: a synced timeline, the 3D scene with contact-force arrows, and real and simulated views side by side.

## Per-demo steps

### A. Single photo → twin → 4,500 episodes → VLA (56% → 75% real success)

Input: one overhead RGB photo of a bimanual FR3 cell with 9 party props.

1. Build procedural meshes and compound MuJoCo collision primitives for each prop, matching its real size, color and resting pose (including a spring-hinged deformable cap).
2. Define a 77-task curriculum: pick-and-place, uprighting, pushing, handovers, scoop-and-push and table clearing.
3. Write a closed-loop IK trajectory generator, a scripted expert with access to privileged simulator state, under ±10 cm pose randomization.
4. Keep only episodes that reach `reward == 1.0` in physics, then export more than 4,500 of them for VLA fine-tuning.

The real-robot gain came from **simulated demonstrations plus imitation learning**, not from RL.

### B. Robot teleop episode → calibrated twin (the most rigorous recipe)

Inputs: synchronized multi-camera video, logged `observation.state` and fingertip `tactile_force`.

1. Build the robot plus the reconstructed objects, then replay the episode by driving the joints toward the logged state with joint-space PD control.
2. Tune only what is set before t = 0: object geometry, collision shapes, mass, inertia, friction, joints, appearance and initial poses. Robot commands may be adjusted only in frames near contact, and only within 0.55 rad (arm) or 0.65 rad (hand) of the logged trajectory.
3. Iterate until two objectives match:
   - **Visual:** simulated renders from head_left, left_wrist and right_wrist line up with the real frames.
   - **Physical:** simulated fingertip contact force matches the logged tactile force in timing, magnitude and direction.
4. Output a per-episode HTML viewer.

### C. Egocentric human video → humanoid twin

Inputs: the video, `candidate_sim_views.png` and **`hand_3d_keypoints.json`, which an upstream hand-pose tool supplied; Argon did not extract the keypoints**.

1. Triangulate object positions from the camera intrinsics and extrinsics.
2. Replace bounding boxes with convex-decomposed meshes.
3. Retarget the wrist and finger keypoints onto a 7-DoF arm and a 22-DoF hand, then step at 500 Hz with no weld constraints.
4. Verify force closure, lift > 2 cm and a stable final placement. Render in Blender.

### D. Cornhole video → Apollo (7 throws in a row)

The robot narrates these four steps in the DeepMind video:

1. Watch the human video and track arm swing and release, then **map the human motion onto the robot's joints and hands** (motion retargeting).
2. Use the robot's **own two eye cameras** to measure the board's 3D position, then build a physics simulation of the robot body and the board. The board's geometry came from the robot's cameras, not from the human video.
3. Practice in simulation: widen the stance and tune the swing so the hand clears the hip.
4. **On the real robot, watch each bag land and adjust the release by a few centimeters** until seven shots in a row go in.

The streak therefore included real-world trial and error; it was not zero-shot sim-to-real. Google calls it an internal test, run in a lab with the robot on a tether.

### E. Sketch → `mujoco_warp` scene

1. Parse the layout from the sketch.
2. Place the humanoid, the table and a 3-2-1 box pyramid.
3. Synthesize a four-phase 50 Hz throw (140 steps).
4. Verify balance and 6/6 knockdown, then render in Blender Cycles.

### F. Spot foot-pedal bin

1. Argon builds an **articulated** MuJoCo twin of the bin (pedal opens lid).
2. It practices tripod balance in simulation before stepping on the real pedal.
3. A 2-agent "verifier swarm" checks foot and hand alignment in the real cameras before release.

## Implications for our plan (docs/test-plan-v0.md)

- **T3:** Google never reconstructs from an uncalibrated phone video. We need intrinsics (from COLMAP or EXIF) plus a metric scale reference (an ArUco tag or a measured object) before any comparison to tape measurements means anything.
- **D3:** Google *did* retarget human motion for cornhole and for the egocentric demo, so "goal, not motion" is our hypothesis to test in T5, not established practice.
- **D2 / T4:** every Google success check reads simulator state (`reward == 1.0`, lift height, knockdown count), which matches our rule.
- **D4 / T6:** the one quantified real-robot gain came from scripted demonstrations plus imitation, which supports doing demonstrations first.
- **Recipe B** gives the strongest twin-calibration check we can copy without hardware: replay a public LeRobot episode that has joint logs and multi-camera video, then score how well the renders and contacts match.
