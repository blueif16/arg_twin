# author-tools: reassemble

The v1 skill `author-tools`, moved here unchanged by skills-v2 (archived at `playbook/archive/skills-v1/author-tools/SKILL.md`; only skill names and paths are updated to where their content now lives). v1 description: Stage C. Design, test and version task-level tools (approach, guarded touch, search, insert, release) on top of the practice env's actor ops, from the twin's sites and frames, acting only on believed geometry, with written feedback conventions; test them with a deterministic self-test on practice starts (exact and believed information). Use it before practice (D) whenever a brain needs actions above raw end-effector poses, or when a tool's interface changes.

## Goal

A versioned set of task tools (interface `task/v1`) that a brain calls instead of raw poses. Each tool runs a whole closed-loop motion and hands control back with structured feedback, so the brain decides between calls; this beats a pre-written program, 56% vs 24% [urai-2026].

Done when:
- each tool passes its own test in the twin with exact information on randomized practice starts (`C.tool_success` ≥ 0.95, the minimum over tools);
- the same test is reported with believed information;
- nothing that passed before fails after an interface change (`C.interface_regressions` = 0);
- the tool list and versions are in `runs/<brain>/<run_id>/task_tools/manifest.json`.

## Inputs

- The env and its actor ops (`twin/env.py`; skill `sim-integrity`): `task`, `observe`, `move {pos, quat, speed ≤ 0.25 m/s, force_max ≤ 15 N, stop_force, timeout}`, `gripper`, `wait`, `status`, `finish`.
- The task's frames: the hole entry pose (+z into the hole, +y along the long side), the part geometry and clearance, the nominal grasp (`tip_in_tcp_frame_m`), and the yaw symmetry. The MuJoCo sites behind them are `<slug>_hole/entry`, `<slug>/tip` and `tcp`; the twin's evaluator reads them, the tools never do.
- Frozen practice starts (`twin/starts.py`). Each start pairs truth and belief: the believed board is the nominal contact-fit board (`starts/practice.json` `nominal_board`), and the truth is that board plus the start's `board_err`. The true in-hand pose is `inhand`.
- The real envelope (`runs/<brain>/<run_id>/envelope/envelope.json`) for force ceilings and speeds; operator strategy statistics (`runs/claude/run-standin-1/task_tools/research/operator_strategies.json`); `docs/knowledge/insertion-strategies.md`.

## Tools

| Tool | Use |
|---|---|
| `task_state` | Belief, contact memory, estimated stage. No motion. |
| `task_approach` | approach_hole + align: square the believed part axis to the hole axis (yaw by symmetry, nearest the current yaw), hover the believed tip `standoff_mm` above the believed hole, converge, then trim the steady-state lateral error. |
| `task_move_until_contact` | Guarded move along `down`/`up`/`±x`/`±y` or a hole-frame vector. After a downward stop it re-touches slowly on the commanded line and probes with a steady press. Returns the contact class, the touch point from the wrench, and the stiffness estimate. |
| `task_search` | Spiral or raster under a light press until the tip drops. |
| `task_insert` | Composite. Strategies `auto`, `press`, `spiral`, `wiggle`, `tilt`. Push to the seat with torque alignment and wiggle on a stall; optional release. |
| `task_release` | Open, retreat, check for a pull. |
| `task_tool_selftest` | The deterministic test runner (watchers rerun it). It records no checks. |

Code:
- `twin/task_tools.py`: the routines. They use numpy only and act through an actor-op object.
- `harness/tools/task.py`: the registered tools. They talk to the session daemon's socket and keep a per-episode memory in `runs/<brain>/<run_id>/task_tools/sessions/<session>.json`.
- `twin/task_selftest.py`: the test runner.

## Approach

**1. Design from the frames, not the simulator.** Write every target in the hole frame and convert with the belief:
- tip_world = TCP + R_hand · tip_tcp;
- R_hand = R_part · R_inᵀ (put the believed part axis on the hole axis).
Choose the yaw from the hole's symmetry nearest the current hand yaw that keeps joint 7 (read from the observation) 0.25 rad inside its ±3.05 rad. Joint 7 turns the hand about its own z, so a yaw change moves it one for one; ignoring it drove joint 7 past its limit (Lessons).

**2. Information honesty: the hard part of the task is not knowing where things are.** A tool may use only:
- `task()`: the nominal board, the nominal grasp, and limits;
- what the robot observes: TCP pose, F/T, gripper width, time;
- what contact taught it this episode: the surface touch height, the touch point, the drop;
- the brain's own `corrections`: `hole_shift_mm`, and `grip {dx_mm, dy_mm, grasp_from_top_mm, pivot_deg, rock_deg}`.

Never the start's offsets, the evaluator or MjData. The env already separates truth and belief: `task()` is the believed board, and the true board is that plus `board_err` (|e| median 1.28 mm, p90 4.0 mm; z σ 1.6 mm). So a perception accessor is not needed. A brain that perceives better (camera, overlays from `practice_view`) passes corrections.

"Exact information" (C.tool_success) is the same corrections filled with the truth by the self-test, from the start file (hole) and from the scene (in-hand pose, re-read before every call). It exists only in `task_tool_selftest`. Label it: exact numbers say the tools work; believed numbers say what a brain gets.

The stage a tool reports is `stage_est`, estimated from observations. The evaluator's rubric uses the TRUE hole, so returning it mid-episode would leak where the hole is. It comes only from `env_finish`, on practice starts.

**3. Feedback conventions (Show-Harness: the written conventions carry the grounding [show-harness-2026]).** Every tool returns the same shape:
- `ok` and `code`, from a closed list: OK, NO_CONTACT, UNEXPECTED_CONTACT, NOT_FOUND, JAMMED, SHALLOW, NO_SURFACE_REF, NOT_HOLDING, FORCE_LIMIT, GRIP_SLIP, EPISODE_OVER, ENV_ERROR, BAD_ARGS;
- `summary`: text with numbers;
- `tried`: the phases, their commands and why each stopped;
- `contact {state, force_N, force_hole_N, torque_Nm, touch_point_hole_mm}`;
- `depth {below_touch_mm, below_believed_surface_mm, seat_depth_mm}`;
- `stage_est`, `tip_hole_mm`, `commands`, `commands_since_drop`, `sim_s` and `memory`.

Units are mm, N, deg and m/s, in the hole frame. Each tool's description says what it does, its units and its failure codes. Declarations must pass `gemini_schema` (`harness.cli list`; `harness.brains.test_offline`).

**4. Limits in code.**
- The env caps speed (0.25 m/s) and force (15 N).
- The tools cap force again at the family's 99th-percentile real peak (13.8 N round, 12.6 N square) [forge-plus-2026].
- Defaults sit inside the real envelope where the twin allows it (Lessons).
- Mind the env's progress monitors. `progress.stuck` stalls an episode after 40 commands or 60 sim s with no true depth or stage progress, so a blind search gets about 32 moves. `progress.repeat` stalls it after 6 near-identical moves, so never re-send the same target.

**4a. Check the grip before searching.** One tip contact cannot tell an along-pad offset from an in-hand tilt: both move the touch point along the pads. A grip that turns can still be seen. Press twice, standing still (about 1.5 N, then 3 N), and compare:
- the contact point in the HAND frame (the wrench's line of action cut with the believed-tip plane);
- how far the hand sank.

A rigid grip moves neither (< 0.15 mm, < 0.1 mm). A turning grip moves the contact ≥ 0.5 mm and lets the hand sink ≥ 0.15 mm; both must hold, because a flat face's centre of pressure can wander without sinking. Then searching or pushing only turns the part further and fakes a drop. So `task_search` and `task_insert` refuse with GRIP_SLIP until the brain corrects the grip. The tool hands over a rough tilt from atan(sink / shift) and a suggested `corrections.grip`. Also watch the gripper width: fingers closing more than max(1 mm, 20%) past their width at the approach mean the part is slipping out.

**Corrections change what memory means.** Contact memory in TCP coordinates (touch height, touch point, engaged) is stale once the grip changes, so the harness forgets it when a `grip` correction arrives. Keep the progress counters (`deepest_s`, `commands_since_drop`): the env's no-progress count runs on. Only a correction that gives `grasp_from_top_mm` narrows the tip-height band.

**4b. Keep the call log complete.** A tool that drives the env must leave the same trail a brain would. Every env command goes to `runs/calls/` as an `env_*` line with the caller, session and a `parent` pointer to the tool call (`harness/tools/task.py` `SocketAPI`). Check it: `twin.practice_view.episode_log(session)` must rebuild exactly the env record's command list.

**5. Test.**
- Run `task_tool_selftest` on fixed practice starts, never held-out or red-team, in both modes and both chains (`tools`: every tool in sequence; `insert`: the composite).
- Score every tool on its own postcondition, read by the runner from the evaluator state between calls (the tools never see it). The `raw` chain is the baseline: one raw-pose descent.
- Then run `.venv/bin/python -m tests.test_sim_integrity --quick` for interface regressions.
- Log notes with `record_event(kind="note", stage="C", brain=...)`. The actor never records checks.

**6. Version.** Tool names, arguments and returned fields are an interface (rules.md). A change bumps `task/vN` in `twin/task_tools.py` `INTERFACE`, the harness module and the manifest, and is followed by a rerun of the self-test and the sim-integrity tests.

## Checks

| id | How | Pass |
|---|---|---|
| C.tool_success | `task_tool_selftest {info_modes: [exact]}` → `summary.C_tool_success_exact_min` (minimum over tools of the per-tool rate). Report the believed mode next to it | ≥ 0.95 |
| C.interface_regressions | `tests.test_sim_integrity --quick` and `harness.brains.test_offline` pass; previously passing skills still pass | 0 |
| honesty | `grep -n "mujoco\|ev\.state\|\.ep\[\|starts\.load" twin/task_tools.py harness/tools/task.py` finds nothing. The tool code imports only `twin.task_tools`, `twin.envelope.family_of`, and `twin.task_selftest` for the self-test tool; the self-test's oracle lives in `twin/task_selftest.py` only | no hits |
| manifest | `runs/<brain>/<run_id>/task_tools/manifest.json` lists every tool with its interface version | present |
| call log | for a daemon episode driven by task tools, `twin.practice_view.episode_log(session)` commands equal the env record's `commands[].req` | equal |

## Lessons

**2026-10-25, claude-opus-5.5 stand-in (caller `claude-author-tools`; prototypes in the session scratchpad, debug scripts in `runs/claude/run-standin-1/task_tools/scripts/`; self-tests in `runs/claude/run-standin-1/task_tools/selftest/`).**

- **Final self-test** (`selftest/20261025-093310`, task/v1 r3, env `twin/env.py` 2a67b284, 3 practice starts per part × 8 pegs, 3 workers, 47 min):

  | | exact information | believed information |
  |---|---|---|
  | `task_insert` alone, counted | 22/24 (0.92, CI 0.73–0.99) | 10/24 (0.42, CI 0.22–0.63) |
  | tool chain, counted | 22/24 | 10/24 |
  | one raw-pose descent (baseline), counted | 9/24 | 1/24 |
  | per tool: approach, touch, search, insert, release | 23/24, 24/24, 0/2, 23/23, 22/23 | 0/24, 23/24, 9/24, 10/11, 10/10 |
  | sim s per counted success, median (max) | 25 (43) | 39 (65) |
  | commands per counted success, median (max) | 15 (34) | 31.5 (40) |

  - `C.tool_success` as defined (the minimum over tools, exact) is 0.0, because the search ran on only 2 exact starts. It failed on square peg 3-002, where the tip touched 0.1 mm from the axis and still did not drop, probably a yaw the exact oracle cannot express (it corrects position, pivot and rock, not the part's yaw about its axis). The other exact search "drop" (round peg 1-000) stopped 0.73 mm in, short of the 1 mm the scorer asks, and the push then seated it. Per part, exact: round pegs 12/12, square pegs 10/12. The other exact miss is square peg 4-002: a seat that the dt/2 rerun dropped, so the env voided it (`numerics.dt_half`).
  - Believed `task_approach` is 0/24 by design. It hovers over the believed hole, and the scorer asks for ≤ 1 mm from the true one.
  - Believed failures: 9 NOT_FOUND (the tip further off than a 36-command spiral covers: in-hand offset plus board error, or a square part's yaw), 4 GRIP_SLIP from the grip check, 1 JAMMED.

- **The arm lags sideways while it moves, and a guarded stop freezes the lag.**
  - Descending at 25 mm/s with exact information, the TCP drifted 2.3 mm sideways (hole −y) over 11.6 mm and landed the 8 mm peg on the rim; in free space at rest the error is 0.4 mm.
  - Pressing back sideways does not undo it: the lateral spring force (K × 0.9 mm ≈ 0.45 N) is below friction (0.3 × 3 N).
  - The fix: touch fast (13 mm/s), lift 1.5 mm back onto the commanded line, converge, and touch again at 3 mm/s (about 0.3 mm of lag). With it, exact-information inserts went from 5/7 to 13/15 in the next (partial) self-test (later superseded by the env's guarded-speed cap; final numbers below).
- **Large re-orientations creep.** The env caps the rotational spring at force_max × 0.05 m, so the hand turns at about 2°/s (26° takes about 13 s). The approach waits for convergence (lateral ≤ 0.3 mm, ≤ 0.7°) and then trims the steady lateral error (about 0.25 mm) by re-aiming at goal − error, which ends within 0.05 mm.
- **Thin pegs move in the pads.** For round peg 1, the in-hand pose after the 0.25 s grasp pre-roll differs from the start file's registered pose by +2.5–3° pivot and 1.3–1.5 mm at the tip, and it keeps changing as the hand turns (+1.2° after the approach). Exact information therefore has to be re-read before each call; a t = 0 oracle left the tip 0.4–0.6 mm off against a 0.45 mm capture.
- **The touch point is only as good as the contact.** A flat tip face can touch anywhere on its face. On round peg 2, moving the touch point onto the hole axis put the tip 3.9 mm off instead of 2.5 mm, and the search missed. Learning the tip shift from the touch is now opt-in (`learn_tip`).
- **Press semantics.** `force_max` is a ceiling on |K · error|, not a press. A target 3 mm below presses only K × 3 mm ≈ 1.5 N. The probe measures K (`k_est_N_m`), and presses are commanded as force / K.
- **Searches drag.** Sliding under a press lags the target by about μ × press depth (1.2 mm at 4 mm depth), so a light press (0.8 N, 1.6 mm) is needed for small spirals. A thin peg pivoting in the grip can fake a "drop" of about 1 mm (round peg 1 believed: tilt 18.6° at the "drop", then a jam at 53°).
- **Impacts.** At 25 mm/s the stop tick read 41 N, but the monitor's 30 ms mean peaked at 4.6 N: no flag.
- **Monitors shape tools.** Six identical move targets in a row ended a debug episode as `stalled` (`progress.repeat`). The 40-command no-progress limit caps a blind search at about 32 chords, roughly a 4 mm radius at a 0.8 mm pitch. Real errors beyond that (board p90 6.2 mm in 2D, plus the in-hand offset) need better perception, not a longer search; spiral over 7–15 mm took about 100 s in [pomdp-search-2024].
- **Raw-pose baseline** (smoke run, one start per part): a single raw-pose descent to the believed seat succeeded 0/8 with believed information (all dropped or jammed) and 4/8 with exact information. The final self-test's `raw` chain gave 1/24 believed and 9/24 exact.
- **The call log must stay complete** (finding from the practice dry run, `runs/claude/run-standin-practice-1/`). The first version sent its env commands straight to the daemon socket, so `runs/calls/` held only the task tool's own line, and neither the replay recorder nor `integrity_audit` could see the moves. Now `SocketAPI` appends every move, gripper, wait, observe and status as its own `env_*` line: the same shape and caller as a registry call, plus `parent: {tool, call_id}`. The tool's result names that `call_id` and the number of lines logged. On P-round_peg_3-001 (believed, session `s20261025-020128-25296`), `practice_view.episode_log` rebuilt 24/24 commands, identical to the env's episode record, and the outcome was a counted success.
- **Stall budget, not bigger limits.** A preview of `task_insert` on round peg 1-000 used 50 commands and the env ended it as `stalled`.
  - Search and push now track progress conservatively: the TCP must go 0.5 mm deeper than ever before, and after a touch it must also be more than 2 mm below the touch height. A press can tilt a thin peg in the pads and lower the TCP while the tip stays on the surface.
  - Both stop 4 commands short of the env's 40 and return NOT_FOUND or JAMMED with the commands left. On round peg 2-002 (believed), the search returned after 33 moves with 5 commands to spare instead of stalling.
  - The settle loop waits in 2 s steps (fewer commands).
- **Thin pegs turn in the fingers.** On round peg 1 (4 mm, starts with 5.5–7.1° of pivot, believed information), searching and pushing turned the part 45–64° in the pads. The TCP went 17 mm "deep" while the tip stayed on the surface.
  - The push now starts at 6 N, the real median peak; only later rounds go to the ceiling.
  - GRIP_SLIP is reported when the hand goes more than 1.5 mm past the seat depth.
  - Neither fixes the believed round peg 1 cases: the in-hand tilt is unknown and the pads let the part turn. That needs a perceived grip correction (`corrections.grip.pivot_deg`, e.g. from the wrist view) or a regrasp.
- **What turns the part, measured** (resume run, 2026-10-25 08:30; watcher-side contact logger `runs/claude/run-standin-1/task_tools/scripts/dbg_contact.py`). On round peg 1-000 (believed), the tilt relative to the hole axis was 7.1° in the start file, 9.7° after the hand turned, 14.8° after the first touch, 16.9° after the probe and 22° within 8 search moves. Each step moved the contact along the pads (10.4 → 12.0 → 15.2 mm in the hand) and lowered the TCP; the search read the lowering as a drop.
  - The env's target filter (time constant about 0.2 s) keeps the probe's deep target pressing during the first search moves: |F| read 4.5, 4.1, 3.6, 3.0, 2.4 N on moves meant to press 0.8 N. That extra press helps turn thin pegs. Open: keep each spring ceiling near its press.
- **Tilt-then-insert does not fix in-hand tilt.** `strategy: tilt` on round peg 1 (believed): on 1-000 a false drop at 22° and a jam at 1.9 mm. On 1-001 the part ended seated, but the fingers had closed from 3.2 to 1.0 mm and the env flagged `plausibility.penetration_grasp` and `plausibility.teleport`: a success that does not count, reported by the tool as OK. Hence the width check above, so the tool now reports GRIP_SLIP instead.
- **The grip check, measured.** Between a 1.5 N and a 3 N press:
  - round peg 1-000 and 1-001 (believed, 12–17° in-hand tilt): the contact moved 1.74 and 1.70 mm in the hand and the hand sank 0.39 and 0.40 mm. The tools stopped after 12 commands with the part at 16.8° and 12.6°, instead of turning it to 64° and 44° over 43–49 commands.
  - round peg 2-000 and 2-002 (believed, rigid): 0.03–0.14 mm and 0.05–0.09 mm.
  - Over the final self-test (26 checks, `scripts/grip_check_stats.py`), 5 were flagged. Four had true tilt growth of 1.5–4.7° during the touch and probe; one is round peg 4-002 (believed), a wedged tip sliding 0.78 mm deeper into the hole with +0.3° of tilt. No check missed a tilt growth of ≥ 1°. Flat faces moved the contact up to 6.7 mm without sinking (square peg 3-002), which is why both conditions must hold.
- **Recovery with the suggested correction** (`scripts/loop_recover.py`, believed information, the "brain" applies `suggested_grip` and re-approaches):
  - 1-000: the suggestion (pivot 12.5°) cut the tilt from 16.8° to 4.9°. The search found the hole (true depth 1.7 mm, 0.21 mm off axis). The push then took the tip to 18.4 of 20 mm before the part slid 3.9 mm up in the fingers (GRIP_SLIP, not seated).
  - 1-001: the suggestion (13.3°) cut the tilt from 12.6° to 1.7°, but the search then read a 1.0 mm drop with the tip still on the surface 4.2 mm off, and the episode stalled. The stall came from forgetting `deepest_s` with the grip memory, so a re-approach counted as progress; that is fixed.
  - So the rough tilt helps, but a 4 mm peg in a 0.05 mm-clearance hole needs better than ±5°: a wrist-view estimate or a regrasp.
- **The env's guarded stop overshoots** (1.5 N requested → 5.6–10.9 N measured by the practice dry run). The env builder fixed it in env code (twin/env.py, 2026-10-25 02:25): every move with a `stop_force` is now capped at 1.5–3 mm/s (`GUARD_SPEED_TABLE`), with a peak tolerance of max(0.5, 0.2 × stop_force).
  - The tools do not compensate. They report the force at the stop and the env's measured overshoot (`tried[].force_at_stop_N`, `env_stop_overshoot_N`).
  - They adapted to the new interface. Free-space legs (approach, the lead-in to the contact band, re-aims, the retreat) are unguarded moves with an 8–15 N spring ceiling, checked after the move (|F − F_ref| ≥ 3 N counts as unexpected contact). Only the band where contact is possible is guarded.
  - Guarded moves carry no tool-side timeout. Before this change, the approach legs guarded at 3 N crawled at 2 mm/s, and the timed band timed out at 3 mm depth in the hole (an aborted self-test run, `selftest/20261025-021137-aborted-envchange`).
  - Lesson: a tool built on another agent's env must re-run its self-test whenever `twin/env.py` changes. The manifest records the env hash the tools were tested with, and each self-test's `config.json` records the hashes of the tool, env, monitor and evaluator files.
- **Guard only where contact is possible.** At 1.5–3 mm/s, a guarded leg longer than about 30 mm runs into the env's 20 s move timeout. The downward guarded leg now ends 4 mm past the depth where an untouched tip must already be in the hole (height uncertainty + 2 mm past the believed surface): about 28 mm uncorrected (≤ 17 s) and 11 mm with a known height (about 7 s). Everything above the band is an unguarded move with the post-check.
  - Don't cut the margin to the bone. With 1.5 mm instead of 4, square peg 2-002 (exact) slid 2.9 mm into its tight hole and the leg ended `settled`: the spring's last 2 mm pressed only about 1 N against the wall friction. The tip was short of the in-hole depth, so the tool reported NO_CONTACT. With 4 mm it went 6.9 mm in and seated.
- **Joint 7 decides a square part's yaw.** Joint 7 turns the hand about its own z, so a yaw change of d moves it by d. Choosing "the symmetric yaw nearest the current one" drove joint 7 to 3.052 rad on square pegs 2-002 and 4-002 (exact). That raised the invalidating `plausibility.joint_range` flag; on 4-002 it also stopped the hand 11° short of square, and the part jammed. The approach now reads joint 7 from the observation and takes the nearest yaw that keeps it 0.25 rad inside its range. Both starts then succeeded in a check run.
- **Slip is relative.** A 12 mm peg held 2 mm from its top settled 1.0 mm narrower in the fingers and stayed upright (round peg 3-001, believed). Round peg 1, slipping out, went from 3.2 to 1.0 mm. The width check now fires at max(1 mm, 20% of the width at the approach).

## Sources

Ids are in `docs/knowledge/references.md`.
- [urai-2026] decide between tool calls: 56% vs 24% for a pre-written program; tools 53% vs 18%.
- [show-harness-2026] written conventions carry the grounding.
- [forge-plus-2026] force ceilings in the fast loop, recovery chosen by the model.
- [google-argon-robotics-2026] zero-shot primitives `move_until_contact`/`place_until_contact` with `max_lag` 12 mm and `relief` 2 mm.
- [lian-2021] spiral search on the NIST board: round pegs caught almost surely, rectangles jam; 10.7 s cycles.
- [watson-2020] spiral r_max 4 mm at 2 mm/s; failures from part tilt in the fingers.
- [pomdp-search-2024] tilt-probe search 596/600 in 10 s vs spiral 509/600 in 103 s.
- [inoue-2017] fixed search patterns rated ≤ 1 mm, ≤ 1°.
- [rax-2026] corner-first tilt for 0.1 mm holes.
- [robocurve-astra-2026] raw-pose LLM control stalls at the slot (2/20).
- [llm-libero-2026] Argon 43% on LIBERO-10.
- Knowledge topic: `docs/knowledge/insertion-strategies.md`.
