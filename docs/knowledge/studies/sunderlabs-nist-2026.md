# NIST board insertion learned on one laptop, MuJoCo physics (Sunderlabs, 2026)

`[sunderlabs-nist-2026]` · status: verified (vendor web page, simulation only) · local: `data/refs/sunderlabs-nist-2026.html`

Read 2026-10-24 from https://www.sunderlabs.com/results/nist-insertion (result dated 17 October 2026). Every number is theirs. Lines starting "For us" are our reading. This is a company results page, not peer-reviewed, and it reports no hardware validation.

## Setup

- **Scene:** NIST Assembly Task Board 1 with the real part CAD, a Franka arm, and parts starting in the gripper with exact simulator state. Success means within 1 mm.
- **Training:** PPO with 2,048 environments per task on one Apple M4 Max, using their Metal port of MuJoCo 3.10's contact solver, with MLX for the policy.
- **Testing:** each policy is tested on fresh held-out seeds in their port and again in float64 MuJoCo.

## Results

| Task | Training time | Result |
|---|---|---|
| 12×8 mm bar from scratch | median 9.4 min (4.0–17.5, four seeds) | 91.5–96% on 200 seeds; 94–98/100 in MuJoCo |
| 8 mm peg, 0.11 mm diametral clearance | 49 min | 194/200 in their port; 99/100 in MuJoCo |
| One shared policy, nine insertion tasks | 5.6 h | 1,779/1,797 (99%); 896/900 in MuJoCo |
| Full board (20 parts from kit trays), scripted controller | n/a | 17/30 held-out runs |

- **Physics agreement:** 1.1 M steps/s with contacts at 16,384 environments, 4.4× MuJoCo 3.10 on 14 CPU cores. Same outcome as MuJoCo on 98/100 scripted nut and BNC episodes.

## Limits (stated by them)

- Parts start in the gripper. Reset states come from a scripted controller.
- Learned policies cover 9 of the board's 20 tasks.
- Under a stricter "AlphaNIST-style" contract no policy has passed the 12% floor; we could not find what AlphaNIST is.
- Simulation only.

## For us

- For us: MuJoCo can represent the board's tight fits (0.11 mm diametral) well enough for reliable aligned insertion. This supports `B1.aligned_insert_rate = 1.0` as a reachable target. It says nothing about force accuracy or about reproducing real failures.
- For us: a scripted full-board run succeeded 17/30. Long sequences compound per-part failure even in a perfect-information sim, so per-part checks are the right unit, not whole-recording success.
- For us: 98/100 outcome agreement between two engines is a good template for `B1.dt_halving_agreement`. Report agreement as k/n with an interval, not a bare rate.
