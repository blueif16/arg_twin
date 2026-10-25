# Astranomy: GPT-6 Astra as a native robot policy (NVIDIA, 2026)

`[astranomy-2026]` · status: verified · local: `data/refs/astranomy-2026.html`

Read 2026-10-24 from the [project page](https://research.nvidia.com/labs/srl/projects/astranomy/) (Yang, Khetan, Zhang); headline numbers re-checked against the saved copy the same day. Every number is theirs. Lines starting "For us" are our reading.

## Setup

- RoboLab-120: 120 pick-and-place tasks in simulation, DROID robot (Franka), 15 Hz control, physics paused during inference. 5 seeded episodes per task; ablations on a 30-task subset.
- Inputs: left shoulder, right shoulder and wrist images at 384×216, plus robot state. Medium reasoning effort, 8,192 completion tokens per call.
- Three roles for the same model:
  - **Direct:** stateless API calls that return an action chunk (3–15 targets).
  - **Agentic:** a persistent Codex session with full history and optional tools. It outputs one EEF target at a time, each held for 1–5 steps.
  - **Orchestrated:** a WAM (Cosmos3-Nano-Policy, fine-tuned on DROID) proposes 15-step chunks. Astra accepts each one or overrides it.

## Results (RoboLab-120, SR / score)

| Policy | SR | Latency per call | Cost per successful episode |
|---|---|---|---|
| Agentic | 74.8 / 83.6 | 13.37 s | 30.9M tokens, $36.40 |
| Orchestrated | 66.5 / 79.1 | 11.50 s Astra + 2.74 s WAM | 11.0M, $17.72 |
| Direct (abs EEF) | 66.3 / 76.7 | 18.56 s | 160k, $1.96 |
| Lakeside (best RFM) | 51.0 | | |
| Cosmos3-Nano-Policy | 36.8 | 0.71 s | |
| π0.5 | 28.0 | 0.05 s | |

Inference takes 95–98% of episode wall time. The worst case is CleanUpToys: 633 calls and 158 min of inference for 188 s of task.

## Ablations (30-task subset)

- **Action representation (Direct):**
  - EEF: absolute 72.7, delta 66.0, relative 64.7
  - Joint: absolute 48.7
  - B-spline: 37.3
  - Agentic: EEF 86.0 vs joint 84.0
- **Chunk length (Direct):** 16–32 actions per call vs 3–15 means 62% fewer calls for −6 pp (66.7 vs 72.7).
- **Observation history (Direct):** current frame only 49.3; t−14, t−7, t 72.7. Longer or denser history did not help further.
- **Context history (Agentic, 23 tasks):**

  | History kept | SR |
  |---|---|
  | Full | 100 |
  | Full text + recent images | 95.7 |
  | Text only | 87.0 |
  | Recent text + recent images | 73.9 |
  | Images only | 56.5 |

- **Geometry in the prompt (Direct):**

  | What the prompt includes | SR |
  |---|---|
  | Base only | 19.3 |
  | + gripper geometry | 35.3 |
  | + camera intrinsics/extrinsics | 62.7 |
  | + both | 72.7 |

  Agentic barely moves: 83.3 → 86.0.
- **Expanded interface (Agentic):** more state, gripper and camera geometry, and execution guidance give 80.0 → 86.0 over 150 paired episodes (McNemar p = 0.108, not significant).
- **Vague instruction:** Direct −8.7 pp, Agentic −0.7 pp.
- **Video demonstration (Agentic):** 87.9 vs 90.0 for language alone, and 8.8% more control steps. It did not help.
- **Model as Direct policy:** Astra 72.7, Sol 33.3, Luna 2.0, Claude Opus 5.5 25.3.
- **Reasoning effort:** xhigh 74.5 vs medium 72.7 (Direct), at 2.6× the latency (44.9 s vs 17.1 s) and 2.2× the cost.
- **Agentic scaffolding:**
  - Tools: 88.0 vs 86.0. Tool episodes averaged 19.7 tool calls, many of them image crops.
  - Planning: none 90.0 > explicit 86.0 > discouraged 81.0.
  - Motion-error feedback: 85.9 vs 85.2. It was cited in only 1.14% of action explanations.
- **Orchestrated interventions:** Astra overrode the WAM on 14.5% of steps (4,254 blocks). By cause:
  - grasping 64.4%
  - wrong target or plan 20.2%
  - placement 11.4%
  - transport 3.7%

## For us

- Their cost is flat per episode: nothing learned in one episode makes the next one cheaper. Our claim is the opposite curve: compile decisions into skills, so tokens, calls and time per success fall. Their per-episode numbers are a reference point for an uncompiled LLM policy.
- Calibrated geometry is the biggest single lever for direct control (+53 pp). B0 calibration therefore also serves the brain's observation interface, not only the demo visuals. Google's zero-shot demo does the same with a projected world grid and a jaw reticle on the wrist image.
- History matters: images alone 56.5 vs full 100. Our Gemini runtime chains `previous_interaction_id`, which keeps the full server-side history.
- Prompt engineering moves little: explicit planning, motion feedback and tools all land within ±4 pp. Their result does not support making playbooks more prescriptive.
- Higher thinking buys about 2 pp at 2.6× the latency. This argues for choosing the thinking level per decision type rather than per run.
- In-context video demos did not help pick-and-place. Footage helps us in other ways: labels, failure taxonomy and the physics reference. That still needs measuring, not assuming.
- Grasping is 64% of the interventions, so the physical-precision part is where LLM control breaks. Insertion is harder than grasping.
- They evaluate with fixed seeds and paired tests (McNemar). Our attempt-1 vs attempt-N comparison should use the same frozen starts and a paired test.
- Claude as a direct policy scores 25.3 vs Astra's 72.7. Stand-in results say nothing about Argon's control ability; they test the harness only.
