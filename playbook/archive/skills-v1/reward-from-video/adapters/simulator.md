# reward-from-video: simulator input

For a benchmark simulator with exact state and its own success predicate (LIBERO today). The twin is given, so the reward is the first thing built (B0) and B1/B2 have nothing to check.

**What changes from the core**
- **Truth** is the benchmark's predicate computed on every recorded demo state. The writer never sees the predicate code or the goal files; the watcher's gate scripts use them.
- **Demos replay exactly.** The datasets store full sim states, so every timestep of every held-out demo is scorable.
- **Near-misses are built by perturbing held-out final states** and recomputing the predicate. Only states the predicate scores as not done count. Generators, one per family:
  - joint threshold sweeps just short of done;
  - just outside the target region;
  - hovering, tilted;
  - wrong object, or wrong but similar place;
  - held but not released.
- **Real failures:** sibling-task demos (same scene, a different instruction) replayed in the target task's environment.
- **Start states:** the benchmark's fixed init states. The check must be false at every one.
- **Never use the scored split** of the benchmark (LIBERO-10) for reward writing or gating. Write and gate on the practice suite (LIBERO-90).
- **Camera conventions are part of the state API document.** Check which way the rendered image faces before naming left and right: LIBERO's published agentview is mirrored left-right.

**Benchmark predicates can be looser than intent.** Keep intent near-misses in the gate anyway, and report "stricter than the benchmark" apart from errors. Score rollouts against both the reward and the predicate (`L.reward_truth_agreement`).
