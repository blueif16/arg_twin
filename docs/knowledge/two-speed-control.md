# Two-speed control

## 1. Question

How should slow reasoning and fast control share the robot? When is the brain called, how does a fast routine hand control back, and what do synchronous vs event-driven designs cost in success, latency and tokens?

## 2. What is established

**Fast and slow loops run at very different rates.**
- Helix: a 7B VLM (System 2) at 7–9 Hz feeds a latent to an 80M visuomotor policy (System 1) at 200 Hz. System 1 also predicts its own "percentage task completion", used as a termination signal [helix-2025].
- NavGPT-3: a language-model decision every 3–19 s vs 0.78 s per VLA step [navgpt3-2026]. Astranomy: 11.5–18.6 s per Astra call vs 0.05 s for π0.5 [astranomy-2026].

**Synchronous planning idles the arm.** In a π0.5-style waypoint hierarchy on a real dual-arm robot, the planner took 1,094 ms (1,266 ms round trip) to plan a ~0.5 s action segment, so **the arms moved only 28.7% of the time**. Block-wise decoding cut planning to 125 ms and raised motion duty cycle to 65.6%. Erasing the planned endpoints did not lower success at all: the executor was ignoring the plan [fastplans-2026].

**Asynchronous and event-driven designs.**
- τ0-VLA: a background worker refreshes the current subtask about every 1 s; the ~30 Hz control loop reads the cached subtask and never waits [tau0-vla-2026].
- TypeGo (quadruped): asynchronous LLM loops, preemption and speculative skill streaming cut per-step delay 50% vs step-by-step planning and time-to-first-action 73% [typego-2026].
- NavGPT-3: checking the VLA every 16 steps raises SR from 77 (no periodic checks) to 83, but tokens go from 179k to 2.32M per episode; checking every 32 steps gives 78 SR at 1.53M tokens. The authors conclude periodic review is too costly and argue for event-triggered review [navgpt3-2026].
- SenseNova: Astra reviewing a VLA's actions four at a time saves 63.5% of tokens over per-action review [sensenova-astra-2026].
- FORGE-plus: the LLM is called exactly twice, before execution (force ceiling) and on failure (recovery choice) [forge-plus-2026].

**How often the brain intervenes once a fast layer exists.** Astra overrode a learned policy on 14.5% of steps [astranomy-2026]; in Galbot's hybrid, 14.4% of control steps came from Astra and 85.6% from π0.5 [galbot-astra-2026]. HERO ran 86.7% of successful subtasks on its distilled policy [hero-2026].

**Design of the hand-off (Hi-VLA study, MuJoCo ALOHA)** [hivla-study-2026]:
- Ending a subgoal on a success detector works best; a fixed horizon of 4–8 s is acceptable; letting the VLM predict duration works worst.
- ~10% detector error is harmless; false positives hurt most.
- Best vs naive vs flat VLA on long-horizon tasks: 67.08 vs 40.56 vs 25.30%.

**Orchestrating frozen skills.** Pigey: a VLM plans, calls a frozen TAMP planner or π0.5, verifies and recovers; LIBERO-PRO 12.8% → 53.3%; DROID reasoning probes 4.6% → 96.9%; 3–15 reasoner calls per real trial [pigey-2026]. Harness VLA uses a frozen VLA only for contact-rich phases and analytic primitives for the rest: +38.6 pp on LIBERO-Pro [harness-vla-2026].

## 3. What works, what doesn't, what's efficient

- **Works:** fast layer owns the control rate; brain called on subgoal end, failure or exception [hivla-study-2026], [rax-2026]; batch review [sensenova-astra-2026]; success-check termination [hivla-study-2026].
- **Doesn't:** a planner on the critical path [fastplans-2026]; periodic polling of the brain [navgpt3-2026]; brain-predicted durations [hivla-study-2026].
- **A weak fast layer can hurt.** NavGPT-3: adding the VLA cut Astra's turns 10.71 → 3.37 but SR fell to 75 and cost stayed at $0.469, spent supervising the VLA [navgpt3-2026]. Galbot: hybrid 92% < direct 98% on RoboLab where π0.5 alone is 36% [galbot-astra-2026].
- **Decide between calls, not in advance.** Same tools: 56% when agents decide after each call vs 24% for a pre-written program [urai-2026]. DAHLIA goes the other way and checks outcomes only after each episode to save per-step inference [dahlia-2025].

## 4. Disagreements and open questions

- **Per-step vs per-episode feedback.** URAI and Hi-VLA favour checks after each step; DAHLIA reports strong long-horizon results with episode-level feedback only. Probably task-dependent; no head-to-head.
- **What triggers a wake-up** in contact-rich work is unstudied. FORGE-plus uses failure only; NavGPT-3 proposes event triggers but tests periodic ones.
- Most results use a learned VLA as the fast layer. Code skills as the fast layer (RAX, URAI) are tested with brain calls between tool runs, not with interrupts mid-motion.

## 5. For us

- For us: our fast layer is skill code running in the twin's control loop, not a VLA. The transferable lessons: never put Argon on the control-rate path; wake it on events (subgoal success check, force over threshold, stall, skill exception), not on a timer.
- For us: end each step on a success check that reads simulator state; treat false "done" as the costliest error.
- For us: in real-time mode, report motion duty cycle (fraction of time the arm moves), as Fast Plans did. Attempt 1 should show low duty cycle; compiled attempts should approach the routine's own.
- For us: the "who's in control" timeline should report the share of control steps from Argon. Published hybrids sit around 14% when a decent fast layer exists.
