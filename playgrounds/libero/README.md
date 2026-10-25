# LIBERO playground

The task loop (practice → task tools → compile → attempt N, cheaper) runs on LIBERO from 2026-10-25 (user decision): it is MuJoCo-based, has 130 language-specified tasks, and carries the only published Argon-as-policy baseline, 43% on LIBERO-10 at high effort [llm-libero-2026]. Install and gotchas: `docs/research/libero-setup-notes.md`.

Everything here is sim-only.

## Files

| File | Runs under | What it is |
|---|---|---|
| `sim.py` | `.venv-libero` | `LiberoSim`: one LIBERO env, the paper's adapter (reset, 10-D absolute EEF action, rot6d → axis-angle, observation packing), physics health, evaluator hash |
| `server.py` | `.venv-libero` | `Playground`: the boundary. Narrow JSON command API over stdio or a Unix socket, frozen predicate, per-episode budget, monitors, episode records. `freeze` writes `frozen.json` |
| `client.py` | `.venv` | `SimClient` (starts one server, strips keys from its env, single-threaded BLAS), `call_socket` |
| `argon_loop.py` | `.venv` | Argon as the policy, the published protocol over the Interactions API; one process = one episode |
| `run_feasibility.py` | `.venv` | The 10 feasibility episodes, at most 3 at once |
| `report.py` | `.venv` | Success with a Clopper–Pearson interval, per-episode cost, limit hits, vs the paper |
| `harness_tools_staged.py` | `.venv` | `libero_*` harness tool declarations, staged; move to `harness/tools/libero.py` after the pilot ends |
| `splits.json` | | Feasibility (never scored), scored (the paper's 100 episodes), smoke |
| `frozen.json` | | Hash of everything that defines success (314 files) |
| `paper_argon_episodes.json` | | The paper's per-episode Argon numbers (300 episodes, 3 efforts), parsed from its project page data |
| `constraints-libero.txt` | | Pins for `.venv-libero` |
| `smoke.py` | `.venv-libero` | Install check on all 10 LIBERO-10 tasks |

Records: `runs/libero/<run_id>/` (`episodes/<n>_tNN_iNN/{episode.json, states.npz, rollout.mp4, frames/}`, `loop/<episode>/{meta.json, system_instruction.md, tools.json, turns.jsonl, summary.json}` written by `harness/brains/record.py` with no headers, `results/`, `report.{json,md}`, `attempts.jsonl`, `episodes.jsonl`).

## The boundary

Same shape as `twin/env.py`: one process owns the simulator, and actors only send commands and receive observations.

**Actor ops:**

| Op | What it does |
|---|---|
| `task` | Returns the instruction, conventions, bounds, per-step limits and budget |
| `observe` | Returns `eef_state` (10 labelled values), `gripper_qpos`, and agentview and wrist PNG paths |
| `move_to {targets, note}` | Executes a motion (details below) |
| `done {summary, hindsight}` | Ends the episode as a claim of success; no action runs |
| `give_up {reason, hindsight}` | Ends the episode; no action runs |
| `status` | Returns the budget used |

**`move_to`** reproduces the published motion layer:
- validates the targets and the note;
- interpolates linearly at the declared per-step limit (0.01 m, 0.05 rot6d, 0.1 gripper per 20 Hz step), up to 10 s;
- passes each step through the clamp and delta-limit approvers;
- checks LIBERO's predicate after every step.

**Harness ops:**
- `reset {task_id, init_state, split}`: the episode must be in the session's split. A scored episode is attempted once per (run, actor), and every attempt is logged.
- `begin_call {reserve_usd}`: the call and spend gate. A refusal ends the episode as `limit_reached`.
- `charge {usage, usd}`, `end_limit {what}`, `ping`, `close`.

**Never sent to an actor:** object poses, sim state, the predicate or its value. On the scored and feasibility splits the outcome isn't sent either; only practice splits may set `reveal_outcome`.

**Frozen evaluator.** Success means LIBERO's own goal predicates on sim state, checked after every control step. At start, the server hashes the predicate code plus every LIBERO-10/90 BDDL and init file, and refuses to run if the hash differs from `frozen.json`. Each record stores the hash, the BDDL and init-state digests, the env seed and the per-step predicate trace.

**Budget per episode** (frozen per run): 80 LLM calls, 900 control steps, $30 spend cap (estimated real cost), and 16,000 output tokens per call including thinking. The brain runner asks `begin_call` before every model call, so the server is the source of truth.

**Monitors, run on every episode:**

Integrity:
- `success_at_reset`: the predicate already held at reset (invalidates).
- `replay`: the executed actions are replayed open-loop in a fresh env, and the predicate trace and final state must match to 1e-6. This runs on every success and invalidates on a mismatch.
- `false_claim`: `done` was called without success.
- Command whitelist: actor ops only. `render`, raw state and other ops are refused.

Plausibility:
- non-finite physics or MuJoCo BADQACC (invalidates);
- contact penetration over 20 mm (flag);
- a free body faster than 5 m/s (flag).

A success counts only if no invalidating flag is raised.

## Published protocol (extracted from the paper and its code)

Sources: `data/refs/llm-libero-2026.html` sections 2–3, plus IRVLUTD/LLM_LIBERO and robocurve/inspect-robots at the commit the paper pins (`4d35fe4b`). We read the code; we didn't run it.

| Item | Protocol |
|---|---|
| Observation | Two 256×256 RGB images: agentview (`[::-1, ::-1]`, which is the true view **mirrored left-right**; see the note under the table) and wrist (raw: mirrored top-bottom). Robot state: 12 values in two fields: `eef_state` (xyz, rot6d = the first two columns of `ee_ori_mat`, gripper command 0/1) and `gripper_qpos` (2 finger positions). Sent as one user message: state text, then the labelled images. No object poses, no success signal |
| History | Stateless calls. Each call resends the system prompt, the goal, every text observation, every tool call and note, and the feedback. Images are kept for the 2 most recent observations only (≤ 4 per call); older ones become "[2 camera frame(s) elided]" |
| Actions | `move_to(targets, note)`: absolute world-frame x, y, z, rot6d (r00..r21) and gripper, any subset; bounds x, y [-0.6, 0.6], z [0.42, 1.6]. Interpolated at 0.01 m / 0.05 / 0.1 per 20 Hz step, ≤ 10 s. Also `done(summary, hindsight)` and `give_up(reason, hindsight)`. One tool call per turn; native function calling |
| Controller | LIBERO OSC with `use_delta=False`; rot6d → orthonormalised → axis-angle; gripper 2g − 1 |
| Limits | 80 LLM calls (then `give_up` is issued on the model's behalf), 900 control steps (45 s), a per-episode spend cap (value unpublished; their Argon-high max was $25), and 3 failing turns in a row |
| Effort | High (best for Argon). Output cap per call including thinking: the site says 8,192 for high, but its Argon-high data peaks at exactly 16,000 (6 calls), so the effective cap was 16,000 |
| Episodes | 10 tasks × init states 0–9 = 100. Env seed 142 + init, then 10 no-op steps |
| Success | LIBERO's predicate after every control step. Ending early on success is the only success; `done` is never scored |
| Prompt | Upstream `_SYSTEM_TEMPLATE` + "Embodiment notes:" + `BASE_DOCS`, verbatim, zero-shot. No task notes, no demos |
| Baseline | Argon high: 43/100; $7.84 per episode (mean; median $5.41); 53.8 calls; 3.55M input tokens per episode with only 81k cached; 29 episodes at the 80-call cap and 13 `error`; 41 limit-reached in Table 1. Per-call latency mean 32.6 s; episode latency mean 29 min |

**Mirrored views in the published protocol (found 2026-10-25 14:45 by the reward probe; checked here).**
- MuJoCo returns rows bottom-up, and the agentview's image-right axis is world +y. So the paper's `[::-1, ::-1]` shows the scene mirrored left-right: LIBERO's "left plate" (y ≈ −0.30) appears on the right.
- `sim.py` now has `views="paper"` (the published mirrored views) and `views="true"` (both images `[::-1]`). `"true"` has been the default since 14:50, and every record stores the mode.
- Tasks 4 and 6 are the ones with left/right in the instruction, and they are the paper's two 0/180 tasks.
- Attempt 1 on the scored split has to pick one mode:
  - `"paper"` keeps the comparison with the published 43% exact.
  - `"true"` is the honest sensor.
  - Proposal: run `"true"` as our attempt 1 and attempt N, and report the paper's number as protocol-mirrored.

`argon_loop.py` reproduces all of this. It deviates in four places:
- It calls the **Interactions API** with `store=False`. The paper used Google's OpenAI-compatible chat endpoint. The prompt, tools, history and eviction are the same.
- Thoughts are resent as signatures only, which is what the chat wire resent.
- The spend cap is $30 on estimated real cost.
- Thought summaries are requested for the record but never resent.

## Feasibility result (10 episodes, init states never scored)

Source: `runs/libero/feas-argon-high-1/report.md`, regenerated with `.venv/bin/python playgrounds/libero/report.py --run-id feas-argon-high-1`. Status at 15:07, 2026-10-25:
- 6 of 10 episodes finished (tasks 0, 1, 4, 6, 7, 9).
- Tasks 2 and 3 started before the 15:00 cutoff and were still running. They use `views="true"`.
- Tasks 5 and 8 were not run (the deadline).

All 6 finished episodes ran with the published mirrored views.

| | ours (n=6) | paper, same 6 tasks (n=60) | paper, all (n=100) |
|---|---|---|---|
| success | 2/6 = 0.33 (95% Clopper–Pearson 0.04–0.78) | 31/60 = 0.52 | 43/100 = 0.43 (0.33–0.53) |
| limit reached | 1/6 (API input filter) | 16/60 (12 call cap, 4 error) | 41/100 |
| calls / episode | 45.3 | 49.5 | 53.8 |
| USD / episode | 5.70 | 5.90 | 7.84 |
| input / cached tokens per episode | 2.53M / 65k (2.6%) | 2.68M / 72k | 3.55M / 81k |
| output + thinking / episode | 75k | 67k | 90k |
| wall time / episode | 27.7 min (model 25.2 min) | 23.9 min | 29.2 min |

Per episode:

| Task (init) | Paper on this task | Result |
|---|---|---|
| 0 (17) | 10/10 | Embodiment fault at call 53: "Degenerate rot6d", see finding 3 |
| 1 (16) | 10/10 | Success at 32 calls, $3.09; replay exact |
| 4 (34) | 0/10 | False `done` at 62 calls; left/right task under the mirror |
| 6 (35) | 0/10 | False `done` at 44 calls; left/right task under the mirror |
| 7 (36) | 10/10 | Success at 36 calls, $2.68; replay exact |
| 9 (13) | 1/10 | API 400 `content_blocked` ("Input blocked by Gemini's filters") at call 46, on a benign robot transcript; limit reached |

Findings:
1. **Cost and call profile reproduce the paper:**
   - Within 4% of the paper's per-episode USD on the same tasks.
   - ~2.6% cached input, because image eviction rewrites history each call and so defeats implicit caching.
   - ~30 s per high-thinking call.
2. **The published views are mirror images.** The agentview is mirrored left-right and the wrist top-bottom (see above). Tasks 4 and 6 are compromised. These are the paper's two 0/180 tasks.
3. **The published tool text contradicts the embodiment.** The `move_to` description says rotation targets are "relative to the trial's start orientation (0 means the start orientation)". `BASE_DOCS` and the adapter use absolute rot6d columns. Argon believed the tool text, sent all-zero rot6d, and the run stopped on a degenerate rotation. This is a candidate cause of some of the paper's 13 Argon `error` episodes.
4. **A Gemini input safety filter** blocked a benign manipulation transcript (`content_blocked`). This is the paper's "API error" limit category.
5. **2 of the 4 failures were false `done` claims** (finished-looking scenes where LIBERO's predicate didn't hold). The paper counts 287 such claims across all its models.

Demo clip: `runs/libero/feas-argon-high-1/episodes/001_t07_i36/rollout_trueview.mp4`. Task 7, alphabet soup and cream cheese into the basket, success. It's 17.8 s with the agentview (left) and wrist (right) side by side, 1024×512, H.264, converted to the true orientation. `rollout.mp4` next to it is what Argon saw (mirrored).

## Integration plan

### Tool names (staged in `harness_tools_staged.py`)

`libero_open`, `libero_episodes`, `libero_reset`, `libero_move_to`, `libero_observe`, `libero_done`, `libero_give_up`, `libero_status` and `libero_close` are for the brain. `libero_charge` is for the brain runner only, the analogue of `env_charge_tokens`.

All of them pass `gemini_schema` unchanged; this was checked with `Tool.declaration()` in a throwaway process. `libero_move_to` returns image paths, so `harness/brains/gemini.py` `find_media` attaches them as it does today.

Two runner changes are needed before Argon can run the protocol through `harness.brains.run` instead of `argon_loop.py`. They are proposals, because `harness/brains` is frozen:
1. A stateless history mode with an image horizon. Today's runner chains `previous_interaction_id`, so every frame stays in context and the paper's eviction can't be reproduced.
2. A call gate that asks `libero_charge {check_only}` before each model call.

### Monitors that carry over from the twin

- **Integrity:**
  - Twin checks that carry over: the frozen evaluator (hash), once-only scored attempts, outcomes withheld on scored splits, false-claim counting, the actor-op whitelist.
  - New for LIBERO: open-loop replay of the executed actions (the analogue of the twin's dt/2 rerun) and the predicate-at-reset check.
  - Next, ported from `twin/monitors.py`: `xfrc_applied`/`qfrc_applied` zero, no state writes outside `mj_step`, model parameters unchanged after t = 0. These matter once privileged red-team scripts exist; actors here never get raw access.
- **Plausibility:**
  - In place: non-finite or BADQACC, penetration, and free-body speed (the teleport/energy monitors in spirit).
  - Thresholds are placeholders until the demo envelope (below) calibrates them.

### Real-data envelope

LIBERO ships 50 human teleop demos per task for all 130 tasks (`yifengzhu-hf/LIBERO-datasets`, HDF5 with full sim states; LIBERO-10 ≈ 13 GB). Because the states are stored, every demo replays exactly in our sim. From LIBERO-90 demos (practice suite only, so held-out data isn't touched), per scene family, measure:
- episode length;
- EEF speed and acceleration percentiles;
- gripper close/open timing and widths on objects;
- object max speed and penetration while handled;
- how far objects travel before success.

A counted success outside the envelope gets an `envelope` flag, as `E.envelope_violations` does for pegs. That keeps a success that flings a can into the basket out of the count. The LeRobot conversions (`lerobot/libero_10_image`, `HuggingFaceVLA/libero`, 10 fps, no sim states) are fine for imitation baselines but can't give contact or object-speed envelopes.

### Practice, task tools and compile on LIBERO-90 → LIBERO-10

- **Practice (D)** on LIBERO-90: 90 tasks × 50 init states, with outcomes revealed. All nine LIBERO-10 scenes also appear in LIBERO-90 (KITCHEN_SCENE3/4/6/8, LIVING_ROOM_SCENE1/2/5/6, STUDY_SCENE1), and 35 of the 90 tasks are set in them. The other 55 tasks are in 11 other scenes. That gives a transfer split for free. Declare a `practice` split in `splits.json` before the first practice run: practice tasks, held-out practice tasks, and the same-scene versus new-scene grouping for the transfer claim.
- **Task tools (C)** that the brain authors from practice evidence, all without privileged state:
  - Camera calibration learned by hand-eye: move the gripper to known poses and fit the agentview projection. That turns "pixel → world ray" and "where is the table" into tools. Astranomy measured calibrated geometry as the largest single gain, 19 → 73% [astranomy-2026].
  - Grasp verification from `gripper_qpos` plus the lift check.
  - A "motion until arrived" helper, because chunks end before the arm arrives (paper: 64–72% of the commanded displacement).
- **Compile (E):**
  - A sandboxed routine has no camera, so a routine owns the motion phases and the brain keeps perception. The brain picks the object and target from images; the routine then runs approach, descend, close, verify by finger gap, lift, transport, lower and release with zero brain calls. It hands back on a failed grasp check.
  - This is ASPIRE's split: a fixed perception/motion API with repair knowledge in a library, 31% vs 4% on LIBERO-Pro Long [aspire-2026].
  - The server's actor ops (`task`, `observe`, `move_to`, `done`, `give_up`, `status`) are the sandbox API as they are. Images stay out of the sandbox, which matches `twin/sandbox.py`.
  - The promotion gate is the same as for pegs: routine success on held-out practice tasks, envelope clean, monitors clean, files frozen by hash.

### What the attempt-N comparison needs

1. **Our own attempt 1 on the scored split** (init 0–9, 100 episodes), with this loop and this budget. The paper's 43% came over a different wire, so the clean A/B is ours vs ours. The paper's per-episode outcomes on the same 100 episodes are a third paired reference.
2. **Attempt N** on the same 100 episodes, with the same model, effort, observations, `move_to`, budget and prompt template. It adds only what practice produced: library routines (invoked through a tool, zero brain calls inside), the brain's task tools, and lessons. Freeze all of them by hash before the first scored episode.
3. **Statistics:**
   - Exact McNemar on the paired outcomes.
   - Success with Clopper–Pearson intervals.
   - Limit hits reported apart from failures.
   - Cost per success: calls, input/cached/output tokens, $, wall time.
   - Brain share: control steps commanded by the brain vs routines.
4. **Time and money.** 100 episodes at the paper's ~29 min each, 3 at a time, is about 16 h of wall time per attempt. Budget it, or pre-declare a smaller scored subset (for example 3 tasks × 10 init states) before any scored episode runs.

### Blockers

- `harness/`, `twin/` and `playbook/` are frozen while the Claude pilot runs. The tools stay staged, the runner changes stay proposals, and `argon_loop.py` drives Argon directly until then.
- Sandboxed routines can't see. Compile can only take the motion phases until a perception tool exists that both sides may call (proposal: the learned camera calibration plus a brain-chosen pixel).
- Wall time (above) and the shared CPU: at most 3 sims, with 15–85 s per reset.
