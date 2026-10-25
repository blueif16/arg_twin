# practice: twin built from robot logs

- **What the actor sees:** pixels from the calibrated cameras (render error 1.7 / 1.0 / 4.9 px), overlays and the header are beliefs from the task, never sim truth (`references/reassemble.md` §2).
- **Episode protocol:** reset → observe → act with absolute TCP targets → finish with a claim; the env scores from sim state after holding the last command 0.5 s (`references/reassemble.md` §3).
- **Attempt N** adds `library_list`, `routine_run` and task tools; a routine call is one brain decision and counts against the frozen budget (`references/reassemble.md` §3a).
- **Result to beat:** pilot loop 1, arm 1 21/24 vs arm N 16/24 (McNemar p = 0.0625); arm N won no start arm 1 lost (`runs/orchestrator/pilot-loop-1/audit/final_1440.md`).
