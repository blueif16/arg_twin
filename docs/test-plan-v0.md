# Footage → Twin → Robot: Test Plan v0

*Oct 23, 2026 · Starting point: Google's Gemini 4 Argon robotics demos. Detailed demo breakdown: [research/google-argon-twin-pipeline.md](research/google-argon-twin-pipeline.md).*

## 1. What Google's demo actually shows

Google positions Argon as the **offline brain** for robotics. It can't do live control because it has no Live API; Gemini Robotics ER 2 handles live control. Argon's four demonstrated capabilities, in the order a pipeline would use them:

1. **Read the episode.** It labels video hierarchically. From a 136-second two-arm robot episode folding an origami fox, it produced 1 goal, 6 subtasks and 17 contact-level primitives, and it held the progress score flat while the robot fumbled.
2. **Score progress.** It split a 60-second ring-sorting episode into 28 segments and marked each as positive, neutral or negative progress. In effect, this is a value function.
3. **Rebuild the scene as code.**
   - From one photo of a tabletop, it wrote a calibrated MuJoCo twin and collected over 4,500 randomized episodes. Training on those episodes raised a real robot's success rate from 56% to 75%.
   - From a cornhole video, it mapped the space, worked out whole-body joint motion and wrote the simulation. An Apollo humanoid then made 7 throws in a row.
4. **Practice and revise.** It writes controller code, runs it, measures the result and rewrites the code. On an SO-101 arm, a pick-and-place task sped up over three rounds: 58.9 s → 22.9 s → 17.4 s.

**What the twin is.** Google's twin is **simulation code the model writes**: geometry, joints and camera calibration in MuJoCo. It is not a scan. AGI Factory took the opposite approach, using a 14.2M-splat scan as the stage. Our flow needs both:

- **The scan** gives what the camera sees.
- **Code and meshes** give what collides.

**Caveats.**

- Every result is Google's own.
- The cornhole run is described as an internal test.
- No success rate has been published for the Spot demo.
- The 56% → 75% gain came from **simulated demonstrations**, not from RL on rewards derived from video.

Sources: [Google Developers Blog](https://developers.googleblog.com/gemini-4-argon-for-robotics/) · [eesel breakdown](https://www.eesel.ai/blog/gemini-4-argon-for-robotics) · [AGI Factory](https://cerebralvalley.ai/e/crusoe-recursive-agents-hackathon/hackathon/gallery/50)

## 2. The flow under test

**Footage** → **task spec + repeatability** → **cell twin** → **success checks + rewards** → **policy** → *(real arm)*

Five rules, one per dispute:

| Dispute | Rule |
| --- | --- |
| D1: Splats aren't a simulator | Splats render; meshes collide. Align both to the robot's coordinate frame. |
| D2: VLM rewards get hacked | Rewards come from simulator state. Gemini *writes* them and *audits* rollout videos. It never grades frames as the reward. |
| D3: Embodiment gap | Footage gives goals and constraints, not robot motions. |
| D4: RL from scratch is risky | Demonstrations plus imitation first; RL only as fine-tuning. |
| D5: No robot means no proof | Results stay labeled "sim only" until a real arm confirms them. |

## 3. Tests

**Local** means the desktop plus API calls. **Cloud** means a rented GPU.

| # | Question | Setup | Metric | Where |
| --- | --- | --- | --- | --- |
| **T0** | Can we reproduce Google's four capabilities? | Public robot episodes and one photo of a desk cell; Argon labels, scores, writes MuJoCo, iterates a controller | Qualitative match to the demo; cost and time per call | Local |
| **T1** | Does footage become a correct task spec? | [IndustReal](https://github.com/TimSchoonbeek/IndustReal), [HA-ViD](https://iai-hrc.github.io/ha-vid); Argon outputs steps, pre/postconditions, success criteria | Step-boundary and step-order accuracy against ground truth; recall on known omissions and execution errors | Local |
| **T2** | Can it tell repeatable work from variable work? | Same datasets, repeated cycles | Per-step duration variance; whether Argon's "repeatable" ranking matches the annotations | Local |
| **T3** | How accurate is the twin's geometry? (D1) | Phone scan of a desk cell turned into splats plus collision meshes, compared with Argon's photo-to-MuJoCo scene | Dimensional error against a tape measure on known objects; whether a scripted grasp succeeds in sim on the reconstructed objects | Local if VRAM allows, else cloud |
| **T4** | Which reward source holds up? (D2) | Simple MuJoCo task, trained with (a) a reward over simulator state that Argon writes and (b) VLM-graded frames. Then plant intent violations in rollouts (throwing instead of placing) | Perceived vs true success, giving a reward-hacking rate (method from [SOLE-R1](https://arxiv.org/pdf/2603.28730)); Argon's catch rate on the planted violations | Local |
| **T5** | Goal or motion? (D3) | From one human video, (a) retarget the human's motion and (b) plan to the spec's postconditions | Sim success rate of each | Local |
| **T6** | Which learning method? (D4) | Scripted demos with full sim-state access, then imitation with [LeRobot](https://github.com/huggingface/lerobot), compared with RL from scratch at the same compute | Success rate, samples needed, wall time | Local, then cloud for larger runs |
| **T7** | Does the sim predict reality? (D5) | Real arm (SO-101 or a lab arm); several policies run in sim and in real | Correlation of sim and real success rates (method from [real2sim-eval](https://real2sim-eval.github.io/files/paper.pdf)) | Deferred until an arm is available |

Reference implementations: [Re³Sim](https://arxiv.org/html/2502.08645v3) (meshes for collision, splats for rendering) · [SplatSim](https://splatsim.github.io/assets/splatsim_paper.pdf) (policies trained only in sim, deployed to real) · [Eureka](https://arxiv.org/pdf/2310.12931) (LLM-written reward code) · [MuJoCo](https://mujoco.org)

## 4. Log for every run

- **Model and settings:** frame rate, resolution, agentic video mode on or off
- **Cost:** tokens, cost and latency
- **Split:** which inputs are train and which are held out. Don't touch held-out videos until the final run.
- **Checks:** which pass/fail checks decided the result
- **Failures:** each failure with its timestamp and a one-line cause

## 5. Order

1. **T0 → T1 → T2 → T4.** Cheap, the data already exists, and no hardware is needed.
2. **T3, T5, T6.**
3. **T7**, once an arm is available.

**For Sunday's hackathon:** T1 + T4 are the candidate core. Footage goes in; a task spec and success checks come out, with a measured reward-hacking rate.
