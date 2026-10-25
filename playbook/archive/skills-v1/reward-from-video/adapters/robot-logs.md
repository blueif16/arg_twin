# reward-from-video: robot-logs input

For real recordings with robot logs and a twin built from them: REASSEMBLE (joints, wrist F/T, 3 cameras) and SO-101 (leader-arm commands, 2 cameras).

**What changes from the core**
- **The footage is real, while the state the reward reads is the twin's.** Write the state API document for the twin (part pose, tip position and axis, velocities, target, gripper), not for the camera.
- **Send the recorded-signals plot with the clip.** On REASSEMBLE the force onset and seat show in signals before pixels do (read-footage skill; `runs/argon/probes/api/findings.md`).
- **Truth** is the twin's frozen evaluator. It is not a benchmark predicate. On REASSEMBLE that is `twin/evaluator.py`: seated means depth plus pose relative to the hole, and released.
- **The gate episodes** are twin episodes, never real recordings scored by the reward:
  - practice rollouts from earlier runs, successes and failures;
  - replays of logged commands (B2).

  A reward gated on twin episodes says nothing about real outcomes beyond what B2 validates.
- **Near-miss families for insertion:**
  - depth short by 1.5 / 3 / 6 mm;
  - tilt 4 / 6 / 10°;
  - not released;
  - wrong hole.

  For pick-and-place (SO-101), use the brick 20 mm outside the box, the brick held above the box, and the wrong box.
- **Splits** follow the source's frozen splits (`playbook/splits.json`, `splits_so101.json`). Exam recordings are never footage for a reward.

**SO-101, not run yet:**
- Truth: the brick inside the reconstructed box volume and released, from replaying the logged leader-arm commands in the twin (stage T2).
- Real failures: gate episodes whose replay fails, and the T2 negative control (brick start 20 mm off).
