# replay-validate: simulator with a state API

- **Exact open-loop replay is the check:** a LIBERO success counts only if the recorded actions replay to the recorded states within 1e-6 and no invalidating flag fires (`success_at_reset`, non-finite physics, BADQACC) (`runs/orchestrator/libero-proof/skills/practice_libero.md`).
- **B2 physics questions do not arise** (the twin is the benchmark); F questions become "does the benchmark's physics allow this success" and go to the auditor (penetration over 20 mm, free body over 5 m/s are flagged).
