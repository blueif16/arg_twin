# Evaluation: sim as a proxy, trial counts, paired tests, held-out design

## Question

When does a result in the twin say anything about the real robot? How many trials does a claim need, how should two brains or skills be compared, how should held-out data be kept, and how do we show that a check can actually catch the errors it is meant to catch?

## What is established, with numbers

- **Sim-to-real agreement is measured across policies, with Pearson r and MMRV.** MMRV is the mean, over policies, of the largest real-performance margin of any rank inversion. SIMPLER introduced it because r measures only a linear fit and swings with noise when policies lie in a narrow range, while rank correlation ignores how big the mis-ranked gap is [simpler-2024]. Reported agreement:
  - SIMPLER r 0.855–0.976 [simpler-2024];
  - PolaRiS mean 0.90, worst scene 0.81 [polaris-2025];
  - real2sim-eval 0.901–0.944 [real2sim-eval-2025];
  - authored vs default reconstruction 0.90 vs 0.51 [recon-effects-2026].
- **Small studies cannot separate simulators by correlation.** With 10 task-policy cells × 20 trials, the r intervals for 0.90 and 0.51 overlapped. Mean score error did separate the two reconstructions (6.97 vs 17.54 points, 95% interval excluding zero) [recon-effects-2026].
- **Batch tests need a fixed N.**
  - Re-running Barnard's test as trials arrive inflates the false-positive rate. STEP is a sequential test that keeps Type-I error control and uses up to 32% fewer trials than earlier sequential tests [step-2025].
  - Real cases: 0.56 vs 0.92 was decided after 19 pairs. 0.59 vs 0.68 with 100 trials per policy returned "fail to decide". In sim, 0.00 vs 0.03 needed 225 [step-2025].
  - The guarantees assume i.i.d. trials and binary outcomes [step-2025].
- **A few paired real trials can correct many sim trials.**
  - SureSim (prediction-powered inference) took n = 60 paired real and sim evaluations plus N = 700 extra sim runs, with paired correlation 0.72. The confidence interval narrowed from 0.187 to 0.16, about 20–25% fewer hardware trials.
  - It helps only when the variance of real-minus-sim is smaller than the variance of the real outcomes [suresim-2025].
- **NIST's reliability metric** is the lowest success probability P_S still consistent with m successes in n trials at confidence CL (binomial CDF). Completion time is reported as mean, SD and 95% CI [nist-taskboard-2020].
- **Best practice for real evaluation** [empirical-science-2024]:
  - fix success criteria and a rubric in advance;
  - match initial conditions across policies;
  - interleave policies blind, within one session;
  - report counts, not only percentages;
  - describe failure modes.
  - After a lab reorganization, one policy went to 0/8 on starts it had passed before, which is why sessions and dates must be reported.
- **Matched starts reduce variance.** real2sim-eval ran sim and real from the same 16–27 grid starts and reported Clopper–Pearson intervals [real2sim-eval-2025]. PND compared arms with 30 trials per cell and reported Wilson intervals and Fisher's exact test (17 points, n = 90, p = 0.012) [pnd-2026].
- **Held-out design.**
  - Calibrate on one image sequence and score on another [werner-2018].
  - Score registration at targets not used to fit it (TRE, not FRE) [cheok-2018].
  - Keep co-training scenes disjoint from evaluation scenes [polaris-2025].
- **Graders differ.** Rule-based sim grading agreed with human grading on 28 of 30 sampled trials. The authors charge the disagreement to the reconstruction, which inflates its error [recon-effects-2026].
- **Checks that can be fooled.**
  - Reprojection residual alone misses biased calibrations [hagemann-2021, autofocus-2024].
  - Low dynamics-prediction error did not imply insertion success across model families [touching-gnn-2025].
  - Policies exploit simulator penetration unless penetrating episodes are filtered [industreal-2023].

## What works, what doesn't, what's efficient

- **Works:**
  - paired designs: the same starts for both arms, and the same recorded attempt in sim and real [real2sim-eval-2025, suresim-2025];
  - pre-registered criteria, and sequential tests when trials arrive one at a time [step-2025, empirical-science-2024].
- **Doesn't work:**
  - peeking at a batch test and adding trials [step-2025];
  - correlation over a handful of cells [recon-effects-2026];
  - percentages without counts [empirical-science-2024].
- **Efficient:**
  - large sim batches corrected by a small paired real set, but only when sim and real outcomes correlate [suresim-2025];
  - sequential stopping for large gaps [step-2025].

## Disagreements and open questions

- No published method gives valid inference about the real robot when there is no real robot and only fixed logs. SureSim needs fresh paired trials drawn from the same distribution.
- Everything above compares policies on binary success. Agreement on a single attempt, and on forces, has no standard metric.

## For us

These are our computations and inferences, not the sources' claims.

- **Trial counts.** One-sided 95% Clopper–Pearson lower bounds (our arithmetic): 20/20 gives 0.861 and 29/29 gives 0.902.
  - To claim `D.success_rate ≥ 0.9` with 0 / 1 / 2 / 3 failures we need at least 29 / 46 / 61 / 76 frozen held-out starts.
  - A Wilson 95% interval on 16/20 is [0.58, 0.92], and on 24/30 is [0.63, 0.91]: too wide to separate 0.8 from 0.9.
- **Per-attempt replay.** `B2.outcome_bacc` is a paired design: each logged attempt is replayed once with its real label.
  - The specificity half rests on the real failures in the gate set. With 10 failures, an 8/10 hit rate has a Wilson interval of [0.49, 0.94]. Use at least about 30 real failures. The data has them: 39 failed inserts in episodes 0–14 alone (data notes).
- **Comparing two brains or skill versions.**
  - Use the same frozen starts and seeds in both arms, and test the discordant pairs with an exact McNemar test.
  - If results arrive one at a time and we may stop early, use a sequential test such as STEP instead [step-2025].
  - Never add starts after looking. validation.yaml's H1/H2 pre-registration fits this.
- **Sim is not real.**
  - Every D/E number is a twin number. The only real anchor is per-attempt replay agreement on REASSEMBLE.
  - If that agreement is weak, report twin gains as twin-only. This is the SureSim condition: weak correlation means sim adds nothing about real.
- **Known-error tests.** Before trusting a check, run it on deliberately corrupted inputs it must reject, and on the uncorrupted input it must pass. Examples:
  - shift the board 1/2/3 mm;
  - swap hama1 and hama2;
  - scale K by 0.5 or 2;
  - halve the clearance.

  Record the smallest error each check detects (the validate-checks skill's detection-capability idea [iso-11843-1997, secondhand]).
- **Exam split.** `G.exam_reads = 0` and targets frozen before the confirmatory recording match [werner-2018, cheok-2018] and the PolaRiS disjoint-scene rule. Episode-0 results stay exploratory, as validation.yaml v3 says.

## Added 2026-10-25: a permissive-hole negative control can pass or fail for the wrong reason

- For us (replay diagnosis, train): B2's seated predicate requires tilt ≤ 3° whatever the clearance. In a hole widened by 1 mm per side, a fully inserted peg can rest at up to about 5.7° (2 mm over 20 mm depth), so the permissive twin's "misses" include pegs that are fully in:
  - v5b permissive: 35 such pegs, which lower its seat rate from 0.769 (depth-only) to 0.656;
  - v5a's permissive runs at 10/40/80 ms board softness: 38/55/82 such pegs.
- For us: the one "passing" control on train (v5a 40/80 ms mixture, +0.104) came from those tilted pegs. Under a depth-only predicate every control we measured is negative or near zero:
  - v5b −0.146;
  - v5b with noslip −0.127;
  - the kinematic oracle +0.006.
- For us: because 24 of 308 windows were never released, no twin can seat more than 0.922 of them. With a real seat rate of 0.776, a control of ≥ 0.10 needs the frozen twin within 0.046 of reality and the permissive twin at ≥ 0.876. That tightens the 0.10 seat-rate target by more than half.
- For us: a known-error control fits a reconstruction twin better [iso-11843-1997, secondhand]: shift the board 1–3 mm after registration and require the seat rate to fall. v5b seated 16/16, 52/64 and 24/64 at 0, 1 and 3 mm (replay-validate Lessons).
