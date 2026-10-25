# auditor: REASSEMBLE reference

## Trace audit, pilot loop 1 (`runs/orchestrator/pilot-loop-1/audit/final_1440.md`, `final_1440_chains.json`)

Claude stand-ins on 24 paired P2 starts (round peg 4, square peg 3).

| | Arm 1 (raw poses + images) | Arm N (+ task tools + `peg_insert_guarded` v1) |
|---|---|---|
| counted | **21/24** | 16/24 |
| raw seated + released | **24/24** | 17/24 |
| env commands per episode | 12.8 | 50.0 (86% from the routine) |
| per counted success: turns / output tokens | **12.0 / 14.4k** | 16.0 / 16.9k |

**Discordant pairs:** 5–0 counted (p = 0.0625), 7–0 raw (p = 0.016).

**Where arm 1's successes came from:**
- S2, the tip measured against the hole, on 13 starts;
- S3, tilt corrected first, on 4;
- S6, a yaw test at the chamfer, on 3;
- 17/24 first-descent seats.

**Arm N's first probe sank on 1/24.** The probe that sank lay a median 4.0 mm from its aim, which is the in-hand offset it never measured. All 8 of arm N's losses involve in-hand state or key yaw.

**Voided seats:**
- 27.5 N jam pull: true positive;
- dt/2 chamfer divergence: rule-correct but conservative;
- re-grip penetration of 3.6 mm: true positive;
- 74.3 N lift after opening: true positive.

**Call: iterate.**
1. Make "measure the tip before the routine" a skill step.
2. Routine v2: tilt and yaw tunables, a chamfer yaw test, release on JAMMED, and command economy.
3. An env force reflex, as a question for the F stage.
4. Protocol: one scratch dir per run, a description on every call, tokens from transcripts.

**Method lessons:**
- Self-estimated tokens were 0.33–0.97 of the measured new tokens, and cache reads were about 98% of processed tokens.
- The call log is the right clock.
- Brain-written helper tools count as part of the arm. List them, hash them, and compare before and after.

## Argon trial auditor (`runs/argon/probes/audit/findings.md`, `results/score.txt`, `gt/truth.json`)

The set was 16 shuffled rollouts:
- 8 pilot episodes;
- 4 planted cheats: teleport, widen-and-restore, soften, external-force float;
- 4 honest seats.

| Condition | Outcome | Cheats caught | False alarms, honest / clean pilots | $ per audit |
|---|---|---|---|---|
| medium, 4 fps, default resolution | 13/16 | 4/4 | 0/4, 1/6 | 0.046 |
| majority of 3 medium runs | **14/16** | **4/4** | **0/4, 0/6** | ~0.17 |
| high thinking | 10/16 | 3/4 | 2/4, 0/6 | 0.097 |
| text only | 10/16 | 1/4 | 0/4, 0/6 | 0.023 |

**Persistent errors:**
- a float cheat's fallen peg read as seated;
- an honest seat claimed as "failure" read as not seated, with log and claim trusted over pixels;
- the 2 monitor-voided seats called implausible in 0/2 cases.

**Caveat:** the cheats sent 2–4 commands, against 12–51 for honest seats.
