# practice: resumed from prior tested runs

- **Do not re-run attempt 1 to compare a new attempt N** if a frozen attempt-1 run on the same starts exists; pair against it (`practice_manifest {verify: true}` must pass first).
- **Read the auditor's last call** (advance / iterate / rollback) and test exactly the change it named (`runs/orchestrator/pilot-loop-1/audit/final_1440.md`).
- **Fresh starts for a new claim:** held-out starts already spent on a version are spent (`compile-skill` gate).
