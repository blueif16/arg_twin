# Backlog

Parked items: worth doing later, deliberately not done now. Each one has the reason it was parked and what would bring it back.

- **Dataset adapter layer** (parked 2026-10-24, user decision)
  - **What:** a mapping from any LeRobot-format dataset to one canonical schema: joints, EE pose, gripper, F/T, cameras, actions, labels. A general scene description compiled to MJCF.
  - **Why parked:** the project demonstrates the model reconstructing from footage and recordings with its own spatial understanding plus tools, as in Google's Argon demos. Intake is the agent's job: it reads the metadata, frames and signals with generic tools and writes down what is available (`playbook/skills/reconstruct-scene/`). An engineered adapter would hide the capability we want to show.
  - **Current coupling:** 19 of 36 files in `harness/tools/` and `twin/` hardcode REASSEMBLE paths, columns or NIST geometry. Counted 2026-10-24 with a grep for `data/reassemble`, `observation.state`, `force.base`, `NIST` and the episode-0 recording id.
  - **Bring back when:** the same tools must run on a second setting at scale, beyond a one-episode probe.
- **Second-setting generality probe, run by Argon** (deferred)
  - **What:** give Argon one episode from a different setting with the same skills and tools, and show how far the reconstruct-scene loop gets.
  - **Partly started 2026-10-25:** the Claude stand-in runs the SO-101 track (stage T2).
  - **Bring back when:** the Argon key is live and the REASSEMBLE loop has run once end to end.
- **Footage-only track** (listed 2026-10-25, status: not started; user decision)
  - **What:** reconstruct scenes from human or egocentric video with no robot logs. External help: hand pose, depth, segmentation, and metric scale from known objects. Then retarget the hand to a robot, as in Google's egocentric demo.
  - **Why not started:** hard to validate. With no joints, force or calibration, the only truth is the video itself (render-and-compare) plus physical plausibility.
  - **Intent:** the system should understand any footage, not only robot logs. It appears in `validation.yaml` under `tracks`.
  - **Bring back when:** the reconstruct-scene pipeline has passed on REASSEMBLE and SO-101.
