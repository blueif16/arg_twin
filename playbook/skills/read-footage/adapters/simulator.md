# read-footage: simulator demos with a state API

For a benchmark whose demos store the full sim state at every step (LIBERO). The footage is rendered from those states.

**What changes from the core**
- **The job is the task spec, not labels.** Read one demo clip and the instruction; write subtasks with pre- and postconditions in the state API's names. Times and outcomes come from the recorded states and the benchmark's predicate, so there is nothing to guess from pixels (`references/libero.md`).
- **Bind names with one example state.** Send the state API document plus the state at t = 0 with the clip; name binding from video plus one example state was right in 6/6 first writes (`runs/argon/probes/reward/findings.md`). The spec then feeds `reward-from-video`.
- **Check the camera convention before naming left and right.** LIBERO's published agentview is mirrored left-right (`runs/libero/feas-argon-high-1/report.md`).
- **Never read the scored split's demos** (LIBERO-10); write from the practice suite (LIBERO-90).
