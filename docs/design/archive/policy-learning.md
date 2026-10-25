# Step 2: policy learning

> **Status: proposal under discussion. Nothing in this file is agreed yet.**

Draft v2, 2026-10-24. This replaces v1. Training happens inside the twin built in [reconstruction.md](reconstruction.md). Argon learns each skill first, by acting through tools. Its working strategies become fast programs, and the programs are distilled into a team of small models. There is no human in the loop, and results stay sim-only (D5).

## Who drives the arm

| Phase | What drives the arm | Output |
|---|---|---|
| **A. Argon acts** | Argon calls tools (`move_until_contact`, `spiral_search`, …). Each call runs a whole motion locally, then Argon looks at the result and decides the next call. Slow, with few attempts | Strategies that work |
| **B. Programs** | Skill programs Argon compiles from what worked in A, with parameters and built-in checks. Fast: thousands of runs with no LLM in the loop | Verified skills, and the training dataset |
| **C. Small models** | Small networks trained on B's data. 30 Hz, sensors only | The final controller team |

Argon starts with no skills. The story runs from not knowing the task, through mastered programs, to small models that reproduce the behavior.

## Rules for phases A and B

- **Mastery is measured on programs, by code.** Each skill has a bar (for example ≥95%) on frozen held-out seeds, measured from sim state. Argon's own live attempts in A don't count; the program's success rate does.
- **The actor and the watcher are separate.**
  - The watcher is a different instance, with fresh context and no access to the actor's reasoning.
  - Check code and held-out seeds are frozen before the actor starts, and the actor cannot edit them.
  - The watcher explains failures; code decides pass or fail.
- **Programs may read sim state; small models may not.** Programs see object poses with realistic noise injected. Small models see only cameras, F/T and joints. Distillation bridges that gap.
- **Real footage is the fast path to mastery.**
  - Argon writes a **strategy card** per skill from the operators' attempts: approach, alignment, contact strategy (tilt then upright, spiral, wiggle, twist), force profile, common failures and recoveries. Each claim is tied to timestamps and F/T windows.
  - A program's simulated F/T profile is compared with the real one for that skill.

## Setpoint reconstruction comes first

The dataset's `action` is the "next absolute EE target". It may simply be the next measured pose; to check, compare `action[t]` with `ee_pose[t+1]`. If it is, it understates what the operator commanded during contact. Under impedance control the commanded setpoint sits *inside* the surface, and that gap is what produces the insertion force. A model trained on measured poses learns to hover at the hole.

**Fix:** reconstruct the setpoint as x_cmd ≈ x + K⁻¹·F_ext.

- F_ext is the logged wrench. The dataset logs it in both the sensor frame and the base frame.
- K is the controller stiffness, identified during reconstruction: fit K so that replaying x_cmd reproduces the logged wrench.

**Check:** replay held-out attempts with and without reconstruction, then compare outcome agreement and F/T error. Twin replay, real demos used as student data, and the real-vs-learned comparison in the demo all depend on this step.

## How the small models are trained (phase C)

1. **Record.** Every program run logs, every 1/30 s:
   - three camera images (rendered at the calibrated poses)
   - wrist F/T
   - joint state
   - the action: EE setpoint plus gripper
   - per-skill success
2. **Supervised training.** The small network takes the latest images, F/T and joints, and predicts the next ~0.5 s of actions (an action chunk). Candidate models:
   - a flow or diffusion transformer of about 50–100M parameters
   - SmolVLA (450M), with the task label as its prompt
3. **DAgger.** Run the student in the twin. From the student's own states, ask the program what it would do, add those labels, and retrain. Repeat until the student matches the teacher's success rate.
4. **Optional RL fine-tuning** on the hardest contacts.

### Data sources

| Source | Role |
|---|---|
| Program runs | Main training data and the DAgger teacher |
| 4,035 real successful attempts, replayed in the twin with reconstructed setpoints | Extra student data in the real action distribution |
| 516 real failures, replayed in the twin | Failure states for training the monitor and recovery |
| Real footage | Strategy cards, and the yardstick for the twin |

## The team of small models

- **Skill models**, one per part family. Each trains from the states the previous skill actually leaves the board in, not from clean resets.
- **A monitor model.** It estimates success and progress from images and F/T, and triggers a retry or recovery. It trains on program runs and on labelled real footage. The value-function work lives here: Argon's hindsight labels and progress curves (Google's labeling and value-function demos, GVL), and optional advantage weighting of training data (RECAP).
- **A sequencer** compiled from the task graph Argon extracts from the footage. It is code, not a network.
- **Optional:** merge the skill models into one generalist later.

**Why the monitor matters:** board success multiplies across 68 skills. At 98% per skill with no recovery, the full board succeeds about 25% of the time. With one retry and 80% of failures recoverable, it reaches about 75%.

## Backstage checks

These numbers run underneath the loop and decide what is accepted. They are not the centrepiece of the presentation ([demo.md](demo.md)).

- **Gates.** Paired comparisons on held-out seeds. A change is kept only if it improves the target with no regression elsewhere.
- **Real-data proxies.** On real recordings from the exam split:
  - the monitor's success prediction against `segment.success`
  - pose estimates against logged EE state
  - phase recognition against the real labels
- **Planted faults.** In some runs the harness injects a known fault, for example a camera offset, action latency, a wrong F/T frame or a gripper too weak for a part. The measure is whether the watcher finds the right cause.
- **Forecast calibration.** For every change, Argon predicts the effect size; the record shows how close it came.
- **Comparisons for the record:**
  - copying the real demos only
  - a single pass with no loop
  - the loop with Flash instead of Argon
  - ablations of strategy cards, watchers and DAgger

## Specific risks

- **Numerical convergence.** Tight clearances depend on timestep and contact softness. A skill must still succeed when the timestep is halved.
- **Handoffs between skills.** Each skill model must start from where the previous model actually leaves the board.
- **Visual gap.** The small models see renders. Background plates and projected textures ([reconstruction.md](reconstruction.md)) narrow the gap; the pose-estimate proxy on real frames measures it.
- **Multi-modal strategies.** Operators solved the same insertion in different ways. A flow or diffusion head keeps those modes instead of averaging them.
