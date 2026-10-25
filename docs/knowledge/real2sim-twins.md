# Real-to-sim twins: what fidelity buys

## Question

What does a simulated twin need so that outcomes in it (successes, real failures, forces) track the real robot well enough to practice and compare in it? What should be copied exactly and what should be a distribution?

## What is established, with numbers

- **Sim can rank policies the way the real robot does, under conditions.**
  - SIMPLER: Pearson r 0.855–0.976 and MMRV 0.000–0.111 across Google Robot and WidowX tasks, about 1,500 real episodes. Tasks were pick-and-place, move and drawer, not tight insertion [simpler-2024].
  - PolaRiS: mean r = 0.90 over six scenes, worst scene 0.81. It needs a short sim co-training step first; object physics stayed at IsaacSim defaults; the authors say complex contact is not covered [polaris-2025].
  - real2sim-eval (Gaussian-splat twin plus spring-mass physics fit from video): r 0.901–0.944 on three tasks with 16–27 matched starts per checkpoint [real2sim-eval-2025].
- **Both appearance and dynamics matter (ablations).**
  - real2sim-eval without colour alignment: r 0.529–0.805. Without physics optimisation: r 0.694–0.905. Full: 0.901–0.944 [real2sim-eval-2025].
  - SIMPLER visual matching helps only when applied to the whole scene: MMRV 0.050 with background, object and robot matched, 0.087–0.142 with any subset [simpler-2024].
  - Wrist cameras: green-screen compositing "is insufficient", so a splat is used. This is the authors' design argument, not a measured ablation [real2sim-eval-2025].
- **The control gap comes first.** SIMPLER replays real demo actions open-loop and fits the sim controller's PD gains by simulated annealing. Control loss 0.131 vs 0.267/0.432 for other gains, and MMRV 0.031 vs 0.070/0.100. With the real controller's nominal gains the sim arm missed the grasp [simpler-2024].
- **Twins can fail to reproduce real failures.** In one of ten cells the real robot scored zero on 13 of 20 trials; no sim trial scored zero in either reconstruction. Sweeping finger friction 0.8–1.4, exposure and gains did not close it; the authors blame the shared control gap [recon-effects-2026].
- **Better object reconstruction narrows the score gap, but the evidence is confounded.** Authored geometry plus metric scale plus authored mass and friction plus scene splat, against a default recipe: mean score error 6.97 vs 17.54 points, r 0.90 vs 0.51. With 10 cells × 20 trials the r intervals overlap. All factors changed at once, and it is a vendor study [recon-effects-2026].
- **An exact twin trains a brittle policy.** One task (IKEA cabinet door), 20 real trials: twin-trained policy 100% in sim, 25% real; twin with wider pose and scale randomization 55%; digital cousins 90%; twin plus cousins 95% [acdc-2024].
- **Identify only the parameters the task depends on.**
  - ASID chooses exploration that maximizes Fisher information, then identifies. Rod balancing 6/9 vs 0/9 for domain randomization; shuffleboard 7/10 vs 3/10. Identification broke down when the centre of mass sat near the rod's middle (unidentifiable) [asid-2024].
  - Intent-driven Real2Sim: a VLM picks the minimal missing parameters and writes a probing behaviour tree. One scene: 12 vs 29 actions, 33 vs 62 s. Mass 0.257 ± 0.022 kg vs true 0.254 kg; friction estimates had SD 0.11–0.20 [intent-real2sim-2026].
- **Contact-rich assembly transfers from sim through robustness, not through an exact copy.**
  - IndustReal: 83–99% real success over 600 trials, 0.5–0.6 mm clearance, no F/T sensor [industreal-2023].
  - AutoMate: 86.5% (specialists) and 84.5% (generalist) over 20 assemblies, 200 trials each [automate-2024].
  - FORGE conditions policies on a force threshold and randomizes robot, controller and part dynamics. It tolerates up to 5 mm position error, over 1,000 real trials [forge-2025].
- **Frame error, not physics, dominates precision assembly.**
  - At σ = 5 mm fixture-pose noise the official FORGE policy falls from 97–99% to 31.8–60.4% in sim. Students trained with offset-aware supervision keep 92–99% [pnd-2026].
  - On a real Franka at 5 mm: 83.3% vs 34.4% (n = 90 per arm). Raising camera-extrinsic error from 5 mm/1° to 10 mm/2° roughly halves gear success. Swapping a dark table for a light one drops peg success from 93% to 25% [pnd-2026].
- **Google's recipe B** (teleop episode to calibrated twin) is a prompt with no reported match numbers.
  - Replay drives the joints toward logged `observation.state` with joint PD.
  - Only quantities before t = 0 are tuned. Commands may change only near contact, within 0.55 rad (arm) or 0.65 rad (hand).
  - Objectives: renders match the real cameras; simulated fingertip force matches logged tactile force in timing, magnitude and direction. No sim-cheating rules apply [google-argon-robotics-2026].
- **Render-and-compare can recalibrate robot and camera poses** (MuJoCo MJX plus splats, sim-to-sim). Joint noise of 0.005/0.01/0.02/0.03 rad gave TCP errors of 5.4/10.9/21.7/32.6 mm, cut to 2.9/3.8/12.6/18.5 mm. Wrist views were weighted 0.1 against 1.0 for fixed cameras [splatting-physical-2025].

## What works, what doesn't, what's efficient

- **Works:**
  - identifying the controller response before anything else [simpler-2024];
  - matching appearance across the whole scene [simpler-2024, real2sim-eval-2025];
  - running sim and real from the same initial states [real2sim-eval-2025];
  - training over a distribution around the twin [acdc-2024, forge-2025].
- **Doesn't work:**
  - an exact twin as the only training scene [acdc-2024];
  - partial visual matching [simpler-2024];
  - expecting a well-correlated sim to reproduce rare real failures; nothing above shows it does [recon-effects-2026].
- **Efficient:**
  - identify the few parameters the outcome is sensitive to [asid-2024, intent-real2sim-2026];
  - for pick and drawer tasks, varying mass and friction moved success by ≤ 15% but kept the ranking [simpler-2024]. No such insensitivity has been shown for tight insertion.

## Disagreements and open questions

- **How much object physics matters for evaluation.** SIMPLER finds rankings insensitive to mass and friction. real2sim-eval finds physics fitting essential, but for deformables. The vendor study cannot separate physics from geometry.
- **Correlation is not calibration.** r ≥ 0.9 coexists with absolute success-rate offsets [real2sim-eval-2025, simpler-2024]. All these studies compare policies, not single attempts.
- **Open:** no published work reports a twin reproducing per-attempt real insertion failures from teleop logs, or matching F/T traces on a NIST board. Recipe B is the closest and gives no numbers.

## For us

- For us: `B2.outcome_bacc` (per-attempt replay agreement, real failures included) is stricter than any published sim-eval metric, which rank policies. Expect the twin to over-score, as in [recon-effects-2026], and treat a pass as a finding, not a given.
- For us: close the control gap first. Replay free-space segments and check tracking against the logged pose before tuning contact. This is the SIMPLER order.
- For us: copy exactly what is measured and task-critical: TCP offset, board pose, hole geometry and clearance. Treat friction, solref/solimp, stiffness K and the in-hand offset as distributions; see [contact-physics-and-sysid](contact-physics-and-sysid.md).
- For us: frame error is the main threat. A board error well above the part clearance turns a replayed success into a jam; see [calibration-and-localization](calibration-and-localization.md).
- For us: Argon reads simulator state, not pixels, so splat-level appearance is needed only for the visual-overlay checks and any image-based policy, not for state-based practice. This is our inference.

## Added 2026-10-24: keeping a twin aligned, and estimating hidden state

- **Continuous synchronization instead of open-loop replay.**
  - Real-is-Sim keeps an Embodied-Gaussians twin synchronized with the real scene at 60 Hz. It corrects object state with "fictitious visual forces" computed from photometric error, and the real robot follows the simulated robot.
  - Virtual evaluations were consistent with real ones on PushT. The authors say synchronization breaks under long occlusion or when the corrections fight the physics [real-is-sim-2025].
  - For us: correcting object state with invented forces breaks our no-cheat rules in validation. Re-syncing the robot or its command to the log is allowed (Google's recipe B allows bounded command edits near contact). NPDR re-syncs every K steps for the same reason [npdr-2021].
- **Hidden in-hand pose estimated with many parallel simulations.**
  - Many GPU simulations run the same robot commands, each with a perturbed object pose and physics parameters. A sample-based optimizer periodically keeps the states and parameters whose contact feedback best matches the real sensors. The best simulation's object pose is the estimate. Tested on 24 human teleop trajectories with tactile sensing [inhand-gpu-2020].
  - For us: this fits our unlogged in-hand pose. Run 1,024 warp worlds per attempt with sampled in-hand poses and contact parameters, and weight them by how well the simulated F/T matches the real F/T. Using only data up to a decision time keeps the estimate label-blind and causal.
