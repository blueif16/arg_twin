# build-twin: simulator with a state API

The benchmark's own scene is the twin (LIBERO). Nothing is reconstructed, so B0/B1 checks do not apply; the job is to document the state the reward and the actor read.

- **Write the state API document:** fields, units, frames, camera geometry, sign conventions, plus one example state at t = 0 (it binds names in the video to names in the state; `runs/argon/probes/reward/findings.md`).
- **Freeze the evaluator by hash:** the server hashes the predicate code and every BDDL and init file and refuses on a mismatch (`playgrounds/libero/README.md`; `references/libero.md`).
- **Check the cameras' orientation:** the published agentview is mirrored left-right (`runs/libero/feas-argon-high-1/report.md`).
- **Next step is `reward-from-video`** (B0 on this track), not B1.
