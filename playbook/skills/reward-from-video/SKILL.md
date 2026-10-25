---
name: reward-from-video
description: Stage B0 on any data source, and before any practice run. Turn demo footage of a task into a task spec plus `success(s, history)` and `progress(s, history)` code over the twin's state. Gate it with no model calls against held-out demos, automatic near-misses and real failures; give the writer one verification turn; freeze the passing `success` by hash as the reward and evaluator. Pick the adapter by input type (simulator, robot logs, video only) and the reference by source. Use it whenever practice needs a success check that nobody hand-wrote.
stage: build
steps: [B0]
---
## Goal

The result is a frozen reward for one task, made of:
- a spec (subtasks with pre- and postconditions);
- `success` (bool) and `progress` (0–1) code over a documented state API;
- a pass on the gate below, on data the writer never saw;
- the stored file hash.

Practice ends a trial when the reward fires. Where the source has its own check (a benchmark predicate, the frozen evaluator), that check is recorded beside the reward, never in its place.

Done when:
- the gate numbers are recorded as checks (the source's stage in validation.yaml, e.g. `L.reward_*` on LIBERO);
- the frozen file and its hash are in the run (`frozen_reward.json`);
- Lessons holds the run.

## Choose your path

| Input type | You have | Truth for the gate | Adapter | Sources (references/) |
|---|---|---|---|---|
| **simulator** | A benchmark simulator with exact state and its own success predicate; demo videos rendered from recorded states | The benchmark predicate on every recorded demo state | `adapters/simulator.md` | `libero.md` |
| **robot-logs** | Real footage plus robot logs (joints, F/T or commands), and a twin built from them (stage B) | The twin's frozen evaluator on twin episodes (replays, practice rollouts) | `adapters/robot-logs.md` | `reassemble.md` (peg twin); SO-101 is in the adapter, not run yet |
| **video-only** | Footage, no logs, no calibrated twin | None from a simulator: the demo's own end frames, plus a human who confirms 2–3 frames near the threshold | `adapters/video-only.md` | none yet (track `footage`) |

Read the adapter, then the source reference. The core below is the same for every path.

## Inputs

- **The task instruction** in words.
- **The footage.** One demo clip per task is enough. Send it at 4 fps in a user step, never inside a function result, where `fps` is ignored (`runs/argon/probes/api/findings.md` §2). For long footage, cut the demo first (skill `read-footage`).
- **The state API document** (fields, units, frames, camera geometry, sign conventions), plus one example state from the demo at t = 0. The example state binds names in the video to names in the state.
- **Never** the source's own success code, its goal definitions, the demo's final state or any held-out demo.
- **For the gate (watcher side only):**
  - held-out demos with the state at every timestep;
  - real failures;
  - the fixed start states;
  - the near-miss generators.

Splits per task, fixed before the first call:
- demo 0 is the video;
- demos 1–2 are the verification turn;
- every other demo is held out;
- the gate's near-misses come only from held-out final states, and the verification turn's only from demos 1–2.

## Tools

- **Writing.** Argon over the Interactions API at **medium** thinking with `thinking_summaries: auto`, nothing else. Medium matched high on every task after one revision, at 0.4× the cost and a third of the latency. The verification turn is chained with `previous_interaction_id`.
- **Gate scripts (no model calls),** one set per source; paths are in the references. They are not registry tools yet. Run them with `.venv/bin/python -I`, or under the simulator's own venv.
- **Freezing.** The sha256 of the code and spec JSON go in `frozen_reward.json`, and the practice server refuses a mismatch.
- **Outside:** docs/knowledge/learning-signals.md (progress and success detectors), docs/knowledge/evaluation-and-statistics.md (known-error tests for checks).

## Approach

1. **Write.** Ask for (a) the spec, (b) `progress`, (c) `success`. Ask for thresholds as margins from the demo's end state, and ask the writer to say what it cannot see. High thinking narrated measurements it could not make, such as joint values and success times read off the clip's end. Name binding from video plus one example state was right in 6/6 first writes.
2. **Expect the first write to be loose, not strict.** Thresholds guessed from pixels let "almost done" through:
   - drawer −0.11 / −0.09 m vs the truth's −0.14 m;
   - a 6 cm vs 3 cm mug radius;
   - peg depth short by 1.5–3 mm, and a 4° tilt.

   Recall was 1.0 in 10 of 12 LIBERO specs and on the peg. Precision and near-miss rejection carry the errors, on both sources.
3. **Gate with no model calls** (definitions under Checks):
   - every held-out timestep against the source's truth;
   - the near-miss families;
   - the real failures;
   - the start states.
4. **One verification turn.** Show the disagreements: the full state at the first disagreement and at the first true step, from the demos where the check disagrees most, not picked at random. Also show the near-miss families it accepted, generated from demos 1–2 only. Random feedback demos can hold no errors, as with the mug. Demo-only feedback cannot show a near-miss leak: the peg turn showed 3- and 5-state lags, never the 1.5–3 mm acceptances.
5. **Watch for truth-chasing.** A revision can drop an intent requirement to match a looser source check: butter, at high thinking, dropped "released", and 3 demos then fired early. Keep intent near-misses (held but not released, tilted, hovering) in the gate even where the source accepts them. Report those as "stricter than the source", not as errors. The opposite also happens: closing a leak can over-tighten the check (the peg v1 lost recall, 1.00 → 0.88). Re-run the whole gate after the revision, not only the families that failed.
6. **Freeze** the passing `success` by hash. Use `progress` for shaping and stall triggers only. A reward that fails the gate is not used for practice. If it is used anyway, the run says so, and the reward-vs-truth check on rollouts shows what it cost.

## Checks

The watcher runs these, never the writer:

| Check | How | Pass |
|---|---|---|
| precision, recall | every held-out demo timestep vs truth; minimum over tasks | ≥ 0.95 each |
| near-miss rejection | share of near-misses (not done by truth) the check rejects | ≥ 0.95 |
| endpoint errors | demo ends called not done + start states called done + real failures where it ever fires | 0 |
| progress | AUROC of successful finals vs near-misses; median Spearman with time to success | ≥ 0.95; ρ tracked |
| frozen | the hash in `frozen_reward.json` equals the file the practice server loads | equal |
| splits | the writer's inputs hold demo 0, demos 1–2 (verification turn) and nothing else of the task | true |
| stricter than the source | states the source accepts and the check rejects, by family | reported |

On LIBERO these are `L.reward_precision`, `L.reward_recall`, `L.reward_nearmiss_reject`, `L.reward_endpoint_errors`, `L.progress_auroc` and `L.progress_rho` (validation.yaml v17, stage L). A video-only source has no truth to score precision against: its gate is provisional (see the adapter).

## Lessons

- **2026-10-25, simulator (LIBERO-90, 3 tasks, 12 Argon calls, $0.78; `references/libero.md`).**
  - After one revision, two of three tasks pass every gate line: drawer 1.000 / 1.000 / 1.000 (P / R / near-miss rejection), butter 0.985 / ≥ 0.985 / 1.000.
  - The mug fails: 0.928 / 1.000 / 0.865.
  - About $0.08 and 65 s per task. The revised drawer check carried unchanged to another scene.
- **2026-10-25, robot logs (REASSEMBLE peg twin, 59 pilot episodes, 392 near-misses; `references/reassemble.md`).**
  - **First write:** P 0.995, R 1.000, near-miss rejection 0.625. Depth short by 1.5 and 3 mm and a 4° tilt were accepted, the same loose-threshold failure as on LIBERO.
  - **After one revision (frozen):** P 1.000 and near-miss rejection 1.000, but R 0.88. The revision over-tightened release, and 2 successes stall at progress 0.95.
  - **So gate on recall too:** fixing a near-miss leak can open a recall gap.
- **2026-10-25, first use in practice (`runs/argon/libero_chain/`).** The mug reward, which fails this gate, gave the chain its first reward-only success: lc-mug-i27 trial 1, where the reward fired and LIBERO's predicate never held. The gate predicted what the policy then found.
- **2026-10-25** (claude:libero-chain): A revision must keep demo-end recall >= 0.9 on the gate; change one condition at a time; prefer a settle/hold rule to a tighter tolerance. LIBERO-90, 2026-10-25: v1 (medium) let 3/3 off-centre mug releases (4.0-5.8 cm) count as successes that LIBERO rejects, and fired on a butter still falling. v2 tightened the mug radius to ~3.5 cm, required release, stillness and a 5-step hold all at once: near-miss rejection 1.0 but true at only 3/17 mug demo ends (10/17 demos end with a finger still on the mug) and 0 practice successes. v3 (offline only) kept v1 structure plus a hold rule with a tolerance between the two: true at 17/17 mug, 17/17 drawer, 17/18 butter demo ends, all 300 near-misses rejected. Evidence: `runs/argon/libero_chain/r2/results/gate_v2.json`, `runs/argon/libero_chain/r2/results/gate_v3.json`, `runs/argon/libero_chain/results/summary.json`.

## Sources

- `runs/argon/probes/reward/findings.md` (method, failure modes, the gate proposal), `runs/argon/peg_reward/` (port to the peg twin), `runs/argon/libero_chain/` (frozen use in practice), `runs/argon/probes/api/findings.md` (video in a user step, thinking levels).
- docs/knowledge/learning-signals.md, docs/knowledge/evaluation-and-statistics.md; test plan T4 (perceived vs true success, `docs/test-plan-v0.md`).
