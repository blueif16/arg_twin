# DexForce: force-informed actions from kinesthetic demonstrations (Chen et al., RA-L 2025)

`[dexforce-2025]` · status: verified · local: `data/refs/dexforce-2025.pdf`

Read 2026-10-24 from arXiv 2501.10356 (Chen, Yu, Choi, Cutkosky, Bohg; Stanford), published in RA-L 2025 (doi 10.1109/LRA.2025.3568318). Every number is theirs. Lines starting "For us" are our reading.

## Setup

- A dexterous hand with fingertip F/T sensors (CoinFT) under per-finger Cartesian impedance control F = k_p(x_d − x_c) − k_v·ẋ_c, with k_p = 310 and k_v = 2.
- **Stage 1:** kinesthetic demos record the observed fingertip position x_o and the contact force f the finger exerts on the object.
- **Stage 2:** the demo is replayed by tracking force-informed targets x_f = x_o + k_f·f. During this replay, robot-only observations (wrist camera, forces) are recorded for training.
- k_f was tuned until replays reproduced the demos. One value, k_f = 0.0045, was used for all six tasks: slide cube, reorient smiley, open AirPods, grasp battery, unscrew nut and flip box.

## Results

- **Policies trained on x_f:** 76% mean success, from 90% (slide cube) to 57% (open AirPods).
- **Policies trained on observed positions x_o:** near-zero success. Commanding where the finger was does not produce the force it applied.
- **F/T in the observation** never hurt, and helped most for open AirPods, unscrew nut and slide cube.
- **k_f sweep:** k_f was stepped by 0.0001 from 0.0045 until replay failed, five demos per task, with an upper limit of 0.025. Every task tolerated a range of k_f, the ranges overlapped across tasks, and 0.0045 sat inside all of them. The authors read this as: "contact-rich tasks do not necessarily require applying exact forces".

## Limits

- Kinesthetic demos on a hand, not haptic teleop on an arm.
- Replay is on the real robot, not in simulation.
- k_f is a hand-tuned scalar, not an identified controller stiffness. 0.0045 m/N differs from 1/k_p ≈ 0.0032.

## For us

- For us: REASSEMBLE's `action` is the next measured pose, which is DexForce's x_o case. Replaying it as an impedance target should give near-zero contact force and lose force-dependent outcomes, as the data notes predict.
- For us: the fix carries over with the sign flipped. F_base is the force on the tool, so x_cmd = x − K⁻¹·F_base, i.e. k_f ≈ 1/K = 1/500 N/m = 0.002 m/N at the code default.
- For us: DexForce's sweep is the template for our sensitivity check. Sweep 1/K across about 1/800 to 1/200 m/N, record where each replayed attempt's outcome flips, and report a per-part range rather than a single fitted K.
