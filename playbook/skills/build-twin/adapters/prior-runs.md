# build-twin: prior tested runs and lessons

- **Reuse frozen calibrations only after a held-out re-check on the new recording:** the confirmatory B0 run on episode 1 used the playbook only, no episode-0 values, with pre-registered marks (`references/reconstruct-scene-v1.md` Lessons, claude:confirm-ep1).
- **Re-fit the fixture per recording**: a pose file can repeat an earlier recording's pose after the fixture moved (trap `board-stale-mocap-ep117`).
- **Read the decision log of the last run** (`runs/claude/run-standin-1/decisions.jsonl`) and start each quantity from its verdict and error, not from its value.
- **A parameter flagged "not identifiable"** stays sampled (cousins), not fixed (`sim-integrity` references).
