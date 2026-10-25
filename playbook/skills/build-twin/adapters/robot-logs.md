# build-twin: robot logs + cameras

The robot is the calibration object and its contacts place the fixture. REASSEMBLE (F/T, 3 cameras) is the worked instance; SO-101 (no F/T, 2 cameras) the second.

**Order** (`references/reconstruct-scene-v1.md` §5): intake → robot and physical tool length → cameras from robot keypoints → time offset per camera → fixture pose from contacts → holes and parts → contact parameters → render and compare.

**Best practice (evidence)**
- **Cameras from forward-kinematics keypoints, scored on held-out marks** and two-view consistency; edge overlays cannot detect a wrong pose (`references/reassemble-world-frame.md`, trap `edge-overlay-blind`).
- **Fit the physical tool, not the configured EE frame:** one shared tool offset in the camera bundle adjustment, confirmed on hand keypoints (trap table in the core).
- **Fit a time offset per camera together with the poses**; one clock does not remove capture latency (`references/reassemble-world-frame.md`).
- **The fixture from the robot's own contact events** (TCP when a part sits in its hole), per recording, with a leave-one-out error in mm; mocap can be stale and CAD mirrored (`references/reassemble-board-from-contacts.md`). Per-event scatter is large (median 9.3 mm, `references/reassemble.md`), so fit across events.
- **No F/T:** triangulate with the robot-calibrated cameras plus grasp and release poses; check supplied table and base heights against the robot (SO-101's table sat 13 mm below the base, `references/so101.md`).
- **Seed Argon's calibration in code** from its normalized picks (`runs/argon/probes/b0/findings.md`).
