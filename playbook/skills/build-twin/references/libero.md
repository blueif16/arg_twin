# build-twin: LIBERO

- The twin is LIBERO's own MuJoCo scene; install and server in `playgrounds/libero/README.md`, records `runs/libero/<run_id>/`.
- Frozen evaluator: predicate code plus every LIBERO-10/90 BDDL and init file hashed in `frozen.json`; a success counts only with an exact open-loop replay (1e-6) and no invalidating flag (`runs/orchestrator/libero-proof/skills/practice_libero.md`).
- Views: `[::-1, ::-1]` agentview is mirrored left-right; record the views mode with every episode.
