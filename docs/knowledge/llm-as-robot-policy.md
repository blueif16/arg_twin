# LLM as robot policy

## 1. Question

Can a frontier multimodal model act as the robot policy itself, choosing poses or programs each step with no task training? Where does it break, especially on contact-rich insertion, and what does a success cost?

## 2. What is established

**General results.**
- RoboLab-120 sim pick-and-place (DROID Franka, physics paused): Astra agentic 74.8%, direct 66.3%, best robot foundation model 51.0%, π0.5 28.0% [astranomy-2026].
- RoboDojo, 42 sim tasks, 2,100 trials: Astra 22.48%, above all 40 public policies; GPT-5.5 0.88%. Strong on semantic tasks, weak on contact-rich, precision, dynamic and bimanual tasks [robodojo-astra-2026].
- LIBERO-10, end-effector pose tool calls, 100 episodes per configuration: **Astra 44%, Gemini-4 Argon 43%** at high effort; Claude Opus 5.5 and GPT-6.1 Sol 33%; GPT-6 Luna 0%. Argon low → high effort 34 → 43%, at ~$7.84 per episode; 41 of Argon's 100 high-effort episodes hit the 80-call, spending or token limit (Astra: 3) [llm-libero-2026]. This is the only published Argon-as-policy number we found.
- With privileged environment-state access, an unmodified Claude Agent SDK agent reaches 84.9% on LIBERO, 85.7% on ManiSkill3 and 96% on MetaWorld [faea-2026].
- Model choice dominates. Direct policy on 30 tasks: Astra 72.7, Sol 33.3, Luna 2.0, Claude Opus 5.5 25.3 [astranomy-2026].

**LLM control on precision and contact tasks.** Ordered roughly from coarse to fine:

| Task | Result | Source |
|---|---|---|
| Block into bowl, real YAM arms | Astra 19/20, 2.5 min, 2.1k output tokens, $0.94 per run | [robocurve-astra-2026] |
| LIBERO-10 task mix | object placement relatively good; "struggling with precise spatial arrangements and complex multi-step interactions"; pooled over all models, per-task success runs from 58% (cream cheese and butter into basket) down to 0% on two multi-object tasks | [llm-libero-2026] |
| Chess-piece placement (Show-Harness probe, 3 models, 20 trials) | "planning is also not the main bottleneck; errors concentrate on fine-grained grasping and placement" | [show-harness-2026] |
| Geometry-constrained LIBERO-GeoMotion | Astra 11.1% zero-shot | [gpt-gear-2026] |
| Block-in-cup insertion, real, 5 trials | 2/5 zero-shot, 4/5 with one example | [gpt-gear-2026] |
| Puzzle piece into groove, real YAM arms | Astra 2/20, same as Fable 5.1. Stage reached: 2 seated, 8 above the groove, 1 lifted, 6 contact only, 3 no approach | [robocurve-astra-2026] |
| Plug insertion, three real platforms | "markedly harder" than pick-and-place; 2 of 3 PiPER trials ended by a joint collision fault | [sensenova-astra-2026] |
| Four-pair ring/post assembly (AutoMate parts), real YAM, agent writes code | 8/10, 37.2 min and 13.0M tokens per success | [agp-2026] |
| Pegs into 0.1 mm-clearance holes, real FR3, agent probes by touch | all three seated in ~55 min under user supervision; never run unattended | [rax-2026] |

- In Astranomy's orchestrated mode, grasping caused 64.4% of Astra's overrides of the learned policy [astranomy-2026].
- Learned policies hit the same wall. Physical Intelligence's π0.6-based VLA "performs well in the initial phases" of screw installation, zip-tie fastening, Ethernet and charger insertion "but struggles at the stage that demands the most precision", often probing near contact. Online RL on just the critical phase (RLT, 15 min to 5 h of robot data) speeds it up to 3× and lifts screw insertion from 20% to 65% [pi-rlt-2026].

**Claims we were given, checked 2026-10-24:**

| Claim | Status | What the source says |
|---|---|---|
| RoboCurve: Astra 95% block-in-bowl, 10% puzzle-in-groove; reliably got the piece above the slot, then stalled or misaligned | verified in part | Primary is Robocurve's report (Sept 4, 2026), not ai-tldr or ecosistemastartup. 19/20 and 2/20 are correct; the text says Astra "reaches the groove and stalls at the same final step Fable does". But per-trial grades show only 10/20 got above the groove (stage ≥ 3); 9/20 never lifted the piece. Not "reliably"; "misaligned" is not stated. 20-call budget, medium effort, 25% speed cap [robocurve-astra-2026] |
| LIBERO-10: Astra 44%, Argon 43%; struggled with precise spatial arrangements | verified | arXiv 2610.30051 (Espinola, Zhang, Xiang), high reasoning effort, 100 episodes each [llm-libero-2026] |
| Show-Harness: planning not the bottleneck; errors on fine-grained grasping and placement | verified | One probe: chess-piece placement, strongest three VLMs, 20 trials; numbers only in a figure [show-harness-2026] |
| PI found screw driving, cable insertion and zip-tying needed online RL fine-tuning (π*0.6 / RECAP) | verified in part | Source is PI's RLT work (March 2026), not π*0.6 / RECAP. Tasks: M3 screw installation, zip tie, Ethernet and charger insertion. PI says the base VLA struggles at the precise stage and RLT improves it; it does not say RL was the only fix. Applies to a VLA, not an LLM policy [pi-rlt-2026] |

## 3. What works, what doesn't, what's efficient

**Works (Direct policy, 30-task subset unless noted)** [astranomy-2026]:
- Calibrated geometry in the prompt: base 19.3 → + gripper 35.3 → + cameras 62.7 → both 72.7. Agentic barely moves (83.3 → 86.0).
- Absolute end-effector targets 72.7 vs delta 66.0, joint 48.7, B-spline 37.3. SenseNova agrees: directional commands +2.50, joint angles −5.84 [sensenova-astra-2026].
- Short image history: current frame only 49.3 vs three frames 72.7. Agentic: images only 56.5 vs full history 100 (23 tasks).
- Finer action units: halving Show-Harness's step from 2 cm to 1 cm raised zero-shot stacking and peg insertion from 60% to 82% [show-harness-2026].
- More thinking helps some models on LIBERO-10 (Astra 28 → 44%, Opus 5.5 9 → 33%) but not others (Opus 5: 12% → 6%) [llm-libero-2026].

**Did not help** [astranomy-2026]: explicit planning (none 90.0 > explicit 86.0 > discouraged 81.0); tools (88.0 vs 86.0); motion-error feedback (85.9 vs 85.2); video demonstration (87.9 vs 90.0); xhigh vs medium thinking on pick-and-place (74.5 vs 72.7 at 2.6× latency). One-shot demos on RoboDojo: 17.9% (image) and 12.9% (text) vs 22.9% zero-shot [robodojo-astra-2026].

**Fails to verify physical success.** Astra "can tell a task is unfinished but not which physical condition is unmet" [sensenova-astra-2026]. On LIBERO-10, models called `done` without success in 287 of 1,800 episodes [llm-libero-2026].

**Cost per success** (uncompiled, so flat across episodes):

| Source | Tokens / time / cost |
|---|---|
| [astranomy-2026] agentic / orchestrated / direct | 30.9M ($36.40) / 11.0M ($17.72) / 160k ($1.96) per success; inference 95–98% of wall time |
| [galbot-astra-2026] 50 RoboDojo instances | direct 1.13B tokens, hybrid with π0.5 625M; 30 s locomotion took 250 calls × 39.86 s |
| [sensenova-astra-2026] | ~1M tokens and ~1 h per LIBERO-Long++ episode; ~98% of tokens are cached input |
| [llm-libero-2026] | Astra and Argon at high effort ~$7–8 per episode, ~50 calls per episode |
| [agp-2026] real tasks | 5–18M tokens, 17–51 min per success |

Longer chunks cut calls: 16–32 actions per call instead of 3–15 gives 62% fewer calls for −6 pp [astranomy-2026].

## 4. Disagreements and open questions

- **Demonstrations in context.** Null or negative in [astranomy-2026] and [robodojo-astra-2026]; positive in [gpt-gear-2026] (real 10/25 → 20/25) and in SenseNova's training-set knowledge packages (SIMPLE 70.8 → 90.8%) [sensenova-astra-2026].
- **Hybrid with a learned policy.** Galbot: hybrid 48% vs direct 26% on a RoboDojo subset, but direct 98% vs hybrid 92% on RoboLab where π0.5 alone gets 36% [galbot-astra-2026].
- **No graded precision ladder exists.** Every insertion result above is a single task with n = 5–20, on different robots and harnesses. None runs an LLM policy on NIST-board-style insertion in simulation with frozen paired starts.
- Whether the stall is perception (cannot see the slot), control resolution, or missing contact feedback is not separated by any source. Show-Harness's step-size result and GPT-GEAR's tactile probe point at resolution and feedback.

## 5. For us

- For us: attempt 1 = Astranomy's agentic recipe (absolute EEF targets, full history, calibrated geometry). Score every attempt on the Robocurve stage rubric (contact, lifted, above target, seated) so a stall still shows where it happened.
- For us (H2): the sources support the first half of H2, that native LLM control drops on precision tasks, but not a graded drop. REASSEMBLE's human failure rates (place 0–7%, peg/key 24%, gear 26%, connector 27%, nut threading 44%) give the ladder; we would be first to measure an LLM along it. Word H2's mechanism as "stalls at or before the slot". Robocurve's own grades show half the puzzle trials never got above the groove.
- For us: Argon's 41% limit-reached rate on LIBERO-10 means our call and token budget can decide the baseline's score. Freeze it before attempt 1 and report limit-reached episodes separately from failures.
- For us: log tokens split into cached input, uncached input and output, plus calls and inference vs sim time. Use frozen paired starts and a paired test (Astranomy used McNemar).
- For us: physics paused during brain calls is standard in these sim evaluations; real time is a separate, harder condition.
- For us: PI's RL finding is for a VLA, but it is the same pattern: broad competence, then a critical contact phase. It argues for targeting the compile step at that phase, not the whole task.
