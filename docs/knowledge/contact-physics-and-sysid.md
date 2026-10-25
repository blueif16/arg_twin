# Contact physics and system identification for insertion

## Question

Can MuJoCo reproduce real NIST-board insertions, including forces and jams? Which contact and controller parameters should we identify, and which should we randomize? And how do we replay teleop logs whose commanded setpoints were never recorded?

## What is established, with numbers

- **MuJoCo can simulate tight NIST-board insertions when the geometry is set up for it.**
  - Sunderlabs report an 8 mm peg with 0.11 mm diametral clearance at 99/100 on float64 MuJoCo, and one shared policy over nine board tasks at 896/900 on MuJoCo (1,779/1,797 on their Metal port).
  - Conditions: success means within 1 mm, parts start in the gripper with exact sim state, and there is no hardware validation. A scripted controller assembled the full board in 17/30 runs. This is a vendor page, simulation only [sunderlabs-nist-2026].
  - Non-convex meshes need convex decomposition or primitives [mujoco-738-2023].
- **Defaults fail.** Yoon et al. found ODE, Bullet, Vortex and MuJoCo "very hard (if not impossible)" to use for a 50 mm peg with 0.065 mm clearance and an M48 bolt. Their own data-driven contact pipeline reached contact-force RMSE of 1.63 N (peg) and 1.01 N (bolt) against real F/T [yoon-2022]. The MuJoCo lead calls a similar engine comparison a comparison of "default values"; contacts are tunable from steel to sponge [mujoco-738-2023].
- **Contact parameters must be identified against a force sensor.** "The only way to do this is to use a force sensor and do some system identification" (MuJoCo maintainers) [mujoco-1124-2023]. From the MuJoCo docs [mujoco-docs]:
  - the direct solref format (−stiffness, −damping) "is the recommended format for system identification";
  - timeconst should be ≥ 2× the timestep;
  - rest penetration is r = a_u(1 − d)/stiffness;
  - to suppress slip, use elliptic cones, a large impratio and Newton with a small tolerance (e.g. 1e-10), and add noslip if that is not enough.
- **An identified MuJoCo model predicts motion better than wrench.**
  - Identified on 20k real steps (peg-in-slot, Franka), the model gave a 100-step position RMSE of 5.21 mm and 0.92°, and one-step F/T error of 3.54 N and 0.092 Nm.
  - A learned graph model reached 2.87 mm and 0.92 N. Adding a residual MLP to MuJoCo cut the force error to 1.12 N but worsened position to 6.63 mm.
  - Lower prediction error did not mean higher insertion success across model families [touching-gnn-2025].
- **Collision geometry sets the force.**
  - QBIT (MuJoCo, 0.1 mm clearance, 100 real runs): 2,000 spheres of 1 mm radius matched real maximum and average insertion force.
  - Scaling a CoACD decomposition to 98.9% instead of 99% changed peak force greatly.
  - Randomizing o_solref over [0.01, 0.5], o_solimp over [0.001, 0.99] and friction over [0.1, 0.7] produced a force spread covering the real one. This is shown in a figure, not as a statistic [qbit-2025].
- **Robot model identification helps forces too.** Identifying Franka inertial parameters nearly halved the external-force error in a MuJoCo sim-to-real comparison [multipanda-2026]. An RL-based MuJoCo parameter tuner made simulations 8% less dissimilar overall, but not in every scenario [balint-2023].
- **Fitting contact parameters to force error is standard practice.** Approaches include friction, contact stiffness and damping by minimizing virtual-real force error for peg-hole [wang-2024]; geometry-conditioned contact models supervised by real force [pipe-2024]; and CMA-ES for contact clustering [yoon-2022].
- **Distrust episodes that penetrate.** IndustReal drops returns of episodes with more than 1 mm interpenetration. Counting only successes below 0.5 mm, baselines stayed under 40% while filter-plus-weight reached 84.9–87.6% [industreal-2023].
- **Compliance can absorb the gap.**
  - Gain tuning trained on a generic peg reached 9/10, 7/10, 10/10 and 9/10 on 0.05 and 0.02 mm clearances, against 0–5/10 for baselines [compliance-tuning-2023].
  - FORGE policies are robust "regardless of the controller gains" after robot, controller and part randomization [forge-2025].
- **Replay without commanded targets: force-informed targets.**
  - DexForce: x_f = x_o + k_f·f, where f is the force the finger exerts. One value, k_f = 0.0045 (with k_p = 310), worked for all six tasks. Each task tolerated a range of k_f, and the ranges overlapped.
  - Policies trained on x_f averaged 76% success; on observed positions, near zero [dexforce-2025].
  - ACP builds virtual targets from a spring model on wrench filtered with a 1 s moving average. The filter adds hindsight about contacts that are about to happen [acp-2024].
  - Diffusion-based impedance learning reconstructs the equilibrium from pose and wrench, but trains on the recorded commanded pose [geiger-2025].
- **Controller law** (franka_ros Cartesian impedance example) [franka-ros-cic]:
  - τ = Jᵀ(−K·e − D·J·q̇) + nullspace + Coriolis, with D = 2√K;
  - the target is low-passed with factor 0.005 per 1 kHz tick (time constant ≈ 0.2 s, our arithmetic);
  - torque rate is capped at 1 Nm per tick;
  - default K is 200 N/m (range 0–400) and 10 Nm/rad (0–30).
- **Parameter distributions instead of point estimates.** NPDR infers a posterior over simulator parameters (10-D in a contact-rich mini-golf task) from a few real rollouts. It re-synchronizes the simulation with the real trajectory every K steps so that rollouts don't diverge [npdr-2021].

## What works, what doesn't, what's efficient

- **Works:**
  - identified contact parameters, or a calibrated spread of them, scored against a force sensor [mujoco-1124-2023, qbit-2025];
  - dense or primitive collision geometry at the mating surfaces [qbit-2025, mujoco-738-2023];
  - force-informed replay targets [dexforce-2025, acp-2024].
- **Doesn't work:**
  - default contact settings at sub-0.1 mm clearance [yoon-2022];
  - replaying measured poses as targets (near-zero success in [dexforce-2025]);
  - judging a contact model by prediction RMSE alone [touching-gnn-2025].
- **Efficient:**
  - identify geometry and frames first, then the controller, then a few contact parameters;
  - randomize friction and solref/solimp within ranges that cover the real force spread [qbit-2025, npdr-2021];
  - a single global k_f was enough in DexForce.

## Disagreements and open questions

- **MuJoCo at tight clearance.** [sunderlabs-nist-2026] and [qbit-2025] say it is workable with care; [yoon-2022] says it isn't. Both readings are consistent with the outcome depending on collision geometry and parameters. Nobody has published MuJoCo force traces matched to real NIST-board insertions.
- **Point identification or a distribution.** Identification won in [asid-2024] and [simpler-2024]; randomization won in [acdc-2024] and [forge-2025]. They answer different questions: reproducing one log versus training something that transfers.
- **Accuracy ceiling.** An identified MuJoCo model still had about 3.5 N one-step force error [touching-gnn-2025]. How peak force, which is our metric, behaves under that error is untested.

## For us

- For us: rebuild the REASSEMBLE target as x_cmd ≈ x − K⁻¹·F_base. F_base is the force on the tool, so this is DexForce's sign convention flipped. The data notes measured K: code default 500 N/m, older values 200 and 800, release jumps 390–730 N/m (`docs/research/reassemble-data-notes.md`).
  - Sample K from about 200–800 N/m rather than fixing it.
  - Smooth F as ACP does (≤ 1 s), and log how sensitive the replay outcome is to K and to the window.
- For us: in contact the arm moves about 2 mm/s (data notes: 0.07 mm per frame at p50). With D of order 10² Ns/m the damping force is about 0.2 N, against a median contact force of about 7.6 N, so the quasi-static reconstruction should hold. This is our estimate, not measured.
- For us: REASSEMBLE ran its own controller, not the franka_ros example. It has inertia-aware damping with a damping ratio, also tracks a velocity target, uses filter factor 0.005 and caps torque rate at 0.5 Nm per tick (`vendors/reassemble/catkin_reas/src/franka_reassemble/reassemble_controllers/src/cartesian_impedance_controller_damping_ratio.cpp`). Model that controller in the twin, not the example.
- For us: tune contact in this order. Elliptic cones, Newton with tolerance 1e-10, direct-format solref; timeconst ≥ 2·dt (`B1.dt_halving_agreement` already guards this).
  - Use primitives or a dense sphere set at the hole walls, not a scaled convex hull.
  - Then fit stiffness and damping so the twin's peak insertion force lands inside the spread of logged peaks per part.
- For us: `B2.force_peak_err ≤ 0.3` has no published precedent. A 3.5 N one-step error [touching-gnn-2025] is about half our median contact force. Report the error distribution per part and keep the target provisional.
- For us: keep `B1.settle_penetration_mm` and also count a replay as successful only if penetration stayed below 0.5 mm, as IndustReal does.

## Added 2026-10-24: geometry for gears and nuts; real parts aren't nominal

- **MuJoCo ships analytic SDF collision plugins for bolt, nut, gear (involute teeth), bowl and torus**, with example models `nutbolt.xml` and `gear.xml`. mujoco_warp has the SDF collision path too. A maintainer: use the plugin "only if you have your own analytic SDF"; mesh-SDF collision is still in progress and too many SDF contact points is a known issue (tune `sdf_initpoints`) [mujoco-sdf-plugins, mujoco-2801-2025].
  - For us: model the NIST gears and the M16 nut/bolt as analytic SDFs with NIST dimensions, not as meshes. Newton's nut/bolt example (SDF meshes with a MuJoCo solver) is a second reference.
- **Real receptacles are not nominal.**
  - Industrial insertions "frequently involve tight insertions where the clearance is less than 0.1 mm and can even be negative when dealing with a deformable receptacle" [compliance-tuning-2023].
  - QBIT matched real insertion forces at 0.1 mm clearance only after modelling surface roughness (spheres offset for Ra 50–100 µm). Scaling a convex decomposition from 99% to 98.9% changed peak force greatly [qbit-2025].
  - Yoon et al. identified fixed-object poses as the penetration-minimizing pose once the peg is partly inserted [yoon-2022].
  - For us: fitting an effective clearance per part, plus roughness or compliance, before t = 0 is consistent with practice. NIST nominal clearance is a starting point, not ground truth.

## Added 2026-10-25: a held part creeps in soft contacts; how much torque a grasp holds; compliant replay drift

- **Soft-contact slip is built into MuJoCo, and noslip removes it.** The noslip pass runs after the main solver and re-solves only the friction dimensions as hard constraints (R = 0), "suppress[ing] the contact slip that is inherent to soft-constraint models". The docs call it an ad-hoc correction that "can occasionally cause instabilities in models with complex multi-contact interactions". Under Newton or CG it leaves out tendon and actuator couplings [mujoco-docs-2026].
  - For us (`runs/claude/run-standin-1/replay/diag/`, train): a peg of a few grams turned in the pads at about 1°/s with zero contact force (ep 34 seg 24: 0.2° to 10° in 4.5 s, 18° by release). The turn is about the closing axis, in the pad plane; rock stays below 0.2°.
  - With `noslip_iterations` 10 the turn stopped: 0.05° at release, arrival 5.5 mm to 0.4 mm, at the same wall time.
  - Over 308 windows it halved round-peg in-hand error among missed seats (2.2 to 0.9 mm). The seat rate did not move (0.620 to 0.617), because other attempts flipped.
  - At dt/2, outcomes agree on 286/308 (92.9%), against 290/308 without noslip. That is below the 95% bar, so treat noslip as a fix for the creep artefact, not yet a validated setting.
- **How much torque a grasp holds before the part pivots: the limit surface.** For a patch with pressure p, τ_max = μ p ∫|x| dA. Force and moment together must lie inside a roughly ellipsoidal limit surface, (f/f_max)² + (τ/τ_max)² ≤ 1 [howe-cutkosky-1996].
  - For a circular patch of radius R, τ_max is (2/3) μ N R under uniform pressure and (3π/16) μ N R under Hertzian pressure [howe-cutkosky-1996, secondhand].
  - The radius of a soft-finger contact grows as N^γ, with γ between 0 and 1/3 [xydas-kao-1999, secondhand].
  - Parallel-gripper objects rotate about the grasp axis when the grip force is too low for the torsional load. Controlled pivoting works by lowering the grip force [costanzo-2023].
  - For us: a line contact of length L under uniform pressure gives τ_max = μ N L/4. That is a MuJoCo torsional coefficient of about 4 mm for μ = 1 and the twin's 16.5 mm pad; the twin uses 2 mm. This is our arithmetic, not measured on the REASSEMBLE fingers.
- **For us: in an open-loop compliant replay, hand drift is the force mismatch divided by K.** With x_d rebuilt from the log, the twin hand settles at x_log + K⁻¹(F_sim − F_real).
  - On 308 train windows, at the frame the real tip reaches the plate, lateral hand drift tracks |ΔF_lat|/K: Spearman 0.84, median ratio 1.2, direction cosine −0.91.
  - At K = 500 N/m, each newton of lateral force mismatch moves the hand 2 mm, about 10 times the peg clearance. Where the real tip caught on a hole edge and the twin tip did not, the twin hand runs on by F/K.
