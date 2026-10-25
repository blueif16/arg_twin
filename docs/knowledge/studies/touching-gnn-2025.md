# Learning contact dynamics through touching, against an identified MuJoCo model (Yi et al., 2025)

`[touching-gnn-2025]` · status: verified · local: `data/refs/touching-gnn-2025.pdf`

Read 2026-10-24 from arXiv 2509.12151 (Yi, Hertzberg, Atzmueller). Every number is theirs. Lines starting "For us" are our reading. We cite it because it reports a system-identified MuJoCo contact model against real motion and F/T data for peg-in-slot.

## Setup

- A robot with a wrist F/T sensor randomly touches a fixed slot with a tool, with no task context. The data is joint encoders plus F/T.
- **Learned model:** an action-conditional graph network (Act-FIGNet) that predicts tool pose updates and reaction wrench.
- **MuJoCo baselines:** a full robot model and a simplified one. Their parameters were fitted to minimize MSE of trajectories and F/T over 20k real steps. The simplified model was the more accurate and is the one reported.
- Also tested: MuJoCo plus a learned residual MLP, and other graph-network baselines.

## Results

Real-world random-policy test set: 100-step RMSE for motion, one-step error for wrench.

| Model | Position (mm) | Orientation (°) | Force (N) | Torque (Nm) |
|---|---|---|---|---|
| Identified MuJoCo | 5.21 | 0.92 | 3.54 | 0.092 |
| MuJoCo + residual MLP | 6.63 | 1.47 | 1.12 | 0.045 |
| Act-FIGNet (theirs) | 2.87 | 0.64 | 0.918 | 0.034 |

- Against identified MuJoCo, their model is 45% and 30% lower in motion error, and 74% and 63% lower in force and torque error. MuJoCo's deficit is largest in wrench.
- **Sim peg insertion with MPC on the learned model:** up to 98% success. At 1 mm clearance it reached 88–89% after fine-tuning, matching the ground-truth oracle (86%).
- **Prediction error vs success:** an MLP ensemble had lower position error than one Act-FIGNet variant yet never completed an insertion. "Low rollout error alone is therefore" not sufficient.

## Limits

- The real data comes from random touching on one setup. The insertion results are simulated, at 1 and 5 mm clearance, much looser than the NIST board.
- The MuJoCo baseline is the authors' own identification; better collision geometry might do better.

## For us

- For us: even after identification, MuJoCo's one-step force error (3.54 N) is about half our median insertion contact force (about 7.6 N, data notes). Position matched better than force. Expect `B2.force_peak_err` to be the hardest B2 check, and keep its 0.3 target provisional until we see the per-part spread.
- For us: a residual model on top of MuJoCo fixes force but hurts motion. That is a trade-off to avoid in a twin whose outcomes must stay physical, and it would also break the no-sim-cheating rule if it injected forces.
- For us: judge the twin by outcome agreement and force agreement separately. Neither predicts the other.
