# read-footage: robot logs + cameras

For recordings that carry the robot's own signals (joints or TCP, gripper width, F/T or logged commands) beside the cameras. REASSEMBLE and SO-101 today.

**Best practice (evidence)**
- **Signals first, video second.** S3 (signals + video) scored segment F1 0.93, boundary 0.23 s, failure bacc 0.85 on 4 REASSEMBLE train episodes at $0.041 per recording minute; S4 (signals only) matched it within noise (`runs/argon/probes/longvideo/findings.md`).
- **Draft deterministically.** `end_state_check` then one whole-window `query_signals` give the skeleton, parts and a first outcome in about 1 s of tool time per recording (`references/reassemble.md` Lessons); the model names, checks and fills gaps.
- **Contact times from force.** Visual onset is about 0.75 s early in every condition (`runs/argon/probes/api/findings.md` §2). Release from the gripper width.
- **Location names the part** when the board or hole layout is known: `identify_part_by_hole` named every at-hole cycle without video (`references/reassemble.md`). A weak board pose corrupts the verb as well as the part, so check the layout before reading.
- **Optional 1 fps whole-video pass** (about $0.014/min) only for identity and "did it move"; never for boundaries or seat outcomes (longvideo "Recommended settings" 1).
- **Watch tool caps.** `query_signals` returns at most 25 contacts and 25 gripper events per call; query long recordings in parts (`references/reassemble.md`).

**Source differences**
- REASSEMBLE: Franka, wrist F/T, 3 cameras, segment labels with success flags → `references/reassemble.md`.
- SO-101: leader-arm commands, no F/T, 2 fixed cameras, no segment labels → outcome from replay of the logged commands in the twin (`references/so101.md`). Not run for stage A yet.
