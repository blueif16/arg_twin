# practice: LIBERO reference

The playground is described in `playgrounds/libero/README.md` (install notes: `docs/research/libero-setup-notes.md`). Records go to `runs/libero/<run_id>/`, and the tools are `libero_*` in `harness/tools/libero.py`. The checks are validation.yaml stage L.

## Boundary

| Item | Value |
|---|---|
| Actor ops | `task`, `observe`, `move_to {targets, note}`, `done {summary, hindsight}`, `give_up {reason, hindsight}`, `status` |
| `move_to` targets | Absolute world-frame x, y, z, rot6d (r00..r21) and gripper. Bounds: x and y [-0.6, 0.6], z [0.42, 1.6] |
| `move_to` motion | Interpolated at 0.01 m / 0.05 / 0.1 per 20 Hz step, for at most 10 s, through clamp and delta-limit approvers |
| Frozen evaluator | `frozen.json` (314 files) |
| Replay check | Exact open-loop replay to 1e-6 on every success |
| Flags | Penetration over 20 mm; a free body faster than 5 m/s |
| Budget | 80 LLM calls, 900 control steps, $30 on estimated cost, 16,000 output tokens per call |

## Attempt 1: the published protocol ([llm-libero-2026]; `playgrounds/libero/argon_loop.py`)

- **Calls.** Argon at high effort with stateless calls. The prompt is the upstream system template plus `BASE_DOCS`, zero-shot. Images are kept for the 2 most recent observations.
- **Limits and seeds.** Limits as above, plus 3 failing turns in a row. Env seed 142 + init, then 10 no-op steps.
- **Runner.** `harness.brains.run` can't evict images because it chains turns, so `argon_loop.py` remains the anchor runner.
- **Episodes.** 10 tasks × init 0–9 = 100. The paper's Argon high scored 43/100 (0.33–0.53) at $7.84 per episode, with 41 limit-reached. That is about 16 h per attempt at 3 sims.
- **Mirrored views.** The paper's agentview `[::-1, ::-1]` is mirrored left-right. Tasks 4 and 6, the left/right tasks, are the paper's two 0/180 tasks. `sim.py` has `views="paper"` and `views="true"`, and every record stores the mode.
- **Tool text.** The published `move_to` text says rotations are relative to the start orientation, but the adapter takes absolute rot6d. Argon believed the text and sent all-zero rot6d.

## Practice chain (`runs/argon/libero_chain/`)

| Item | Value |
|---|---|
| Tasks | 3 LIBERO-90 tasks: drawer, mug, butter |
| Starts | init 11, 27 and 43 per task, 9 in all (`sched.py`) |
| Trials | up to 4 per start; no new trial after 16:25 |
| Brain | Argon medium with chained turns, 20 calls per trial (`LC_CALLS`), medium-resolution images |
| Reward | frozen v1 code (`frozen_reward.json`) |
| Reflection | Argon medium |
| Nudge | flash, when best progress is not up by 0.02 in 6 brain calls |
| Guide | L1–L9 in `chain.py` |
| Scoring | `analyze.py` → `results/summary.json`: success@k by reward and by truth, agreement, cost, leak grep |

## Numbers (2026-10-25)

**Feasibility, attempt 1** (`runs/libero/feas-argon-high-1/report.md`): 2/6 succeeded (CI 0.04–0.78), against the paper's 31/60 on the same tasks.

| | Ours | Paper |
|---|---|---|
| $ per episode | 5.70 | 5.90 |
| Calls per episode | 45.3 | 49.5 |
| Cached input | 2.6% | |

The 4 failures: 2 false `done` on the mirrored left/right tasks, 1 degenerate rot6d, and 1 `content_blocked` input filter.

**Chain, interim at 15:40 (not scored)**

| Measure | Value |
|---|---|
| Starts / trials | 5 / 5 |
| success@1, by truth | 0.2 |
| success@1, by reward | 0.4 |
| Reward–truth agreement | 0.8 (1 reward-only success: lc-mug-i27) |
| $ per trial | 0.37 |
| Leak hits | 0 in 24 observer texts |
