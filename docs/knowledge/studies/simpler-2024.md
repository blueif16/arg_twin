# SIMPLER: evaluating real-world policies in simulation (Li et al., CoRL 2024)

`[simpler-2024]` · status: verified · local: `data/refs/simpler-2024.pdf`

Read 2026-10-24 from arXiv 2405.05941 (Li, Hsu, Gu, Pertsch, Mees, et al.). Every number is theirs. Lines starting "For us" are our reading.

## Setup

- **Policies:** RT-1 (several checkpoints), RT-1-X, RT-2-X and Octo, all trained on real data and evaluated both in real and in purpose-built SAPIEN scenes. About 1,500 real episodes.
- **Robots and tasks:** Google Robot (pick coke can, move near, open/close drawer) and WidowX (Bridge tasks).
- **Two gaps addressed:**
  - **Control:** replay real demo actions open-loop in sim, and fit the controller's PD gains by three rounds of simulated annealing on translation and rotation tracking loss.
  - **Visual:** "visual matching" green-screens the real background and textures the objects and robot. "Variant aggregation" instead averages over randomized visual variants.

## Results

- **Agreement:** r 0.855–0.976 and MMRV 0.000–0.111 per task family with visual matching. Variant aggregation was weaker on average (MMRV 0.143 vs 0.056; r 0.778 vs 0.924 in the summary table).
- **Control sysid (pick coke can):**

  | Gains | Control loss | MMRV |
  |---|---|---|
  | setting 1 | 0.267 | 0.070 |
  | setting 2 | 0.432 | 0.100 |
  | fitted | 0.131 | 0.031 |

  With the real controller's nominal gains, the open-loop replay missed the grasp.
- **Visual ablation (drawer):** green screen plus drawer and robot textures gave MMRV 0.050 and a real–sim gap of 0.136. Any subset gave 0.087–0.142 and 0.198–0.328. Matching only some parts of the scene did not help.
- **Physics sensitivity:** varying can mass, gripper friction or cabinet joint friction moved success rates by up to 15%, but MMRV stayed low and r high.
- **Engine:** the same setup in Isaac Sim gave MMRV 0.058 and r 0.919, against 0.082 and 0.923 in SAPIEN.

## Limits

- Pick-and-place, move and drawer tasks, not tight insertion. They compare policies, not single attempts.
- Physical properties were simplified on purpose.

## For us

- For us: copy the order. Match the controller by open-loop replay of logged commands first; then appearance; then physics only if outcomes depend on it.
- For us: our replay has no logged commands, only the rebuilt target x − K⁻¹F. So "control loss" for us is tracking error on free-space segments, where F ≈ 0 and the target is the logged pose. Record it as a check before any contact tuning.
- For us: the ≤ 15% physics insensitivity is for pick and drawer tasks. Do not assume it for 0.1 mm insertion.
- For us: report MMRV, not only r, whenever we compare brains or skills across twin variants.
