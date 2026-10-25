# Skill libraries and compilation

## 1. Question

How do agents turn slow reasoning that worked into reusable skills or programs (System 2 → System 1)? What does that buy in success, tokens, calls and time, and what should be stored and gated?

## 2. What is established

**Compiling decisions into programs cuts inference cost by one to two orders of magnitude** [rax-2026]:
- BEHAVIOR-1K load car: first success 123 min of inference and 254k output tokens; final episode 4 min and 4k tokens. Make toast: 17 min → 6 s.
- Real Franka: air fryer 72.3 min / 262.4k tokens by the agent vs 18 s / 2.0k tokens by the compiled program; Bandu tower 17.2 min / 88.4k vs 10 s / 2.2k. Robot time barely changes.
- A skill enters the repository only after an unsupervised successful run, stored with its evidence. Programs raise an exception when perception or planning fails and the agent steps back in.
- The 0.1 mm peg insertion was done by the agent under supervision and never compiled.

**Skill libraries of validated repairs transfer across tasks and embodiments** [aspire-2026]:
- LIBERO-90 library → unseen LIBERO-Pro Long: 31% vs 4%.
- Sim skills given as context on a real bimanual robot: tokens to first success 61.94M → 6.58M (soda can, 13/20 → 19/20), 334.92M → 81.67M (drawer, 0/20 → 11/20).
- Online on hardware (Rubik's cube, 3 episodes): tokens per face turn 11.27M → 2.01M; time per turn 17 min → 3 min.

**Experience as documents and data** :
- AGP: Astra's saved documents (settings, corrections, scripts) shorten later runs, mainly in reasoning time. Given to weaker GPT-5.6 Terra: 1/5 → 4/5, time −34%, tokens −11% [agp-2026].
- XLeRobot: synchronized action/state records cut time 68.6% vs 14% for a prose summary; reuse holds at starts displaced 10–100 cm (−58 to −63%) [xlerobot-2026].
- EmbodiedSWE: a verified solution to a similar task raises success 11% → 31%; a dissimilar one speeds early progress but ends no better (17%) [embodiedswe-2026].
- Encore: demonstrations given as evidence to read (keyframes, gripper events, full trajectory), not as training data. Frozen programs reach 96.3% vs 89.3% for the best prior agent with the same model on LIBERO-PRO; on RoboDojo tasks whose instructions leave the goal unstated, no program succeeds without demonstrations [encore-2026].

**Consolidating into a learned fast policy** (HERO, real Franka, Gemini-3-Flash, zero human demos): VLM reasoning 46.6 s per subtask, exemplar reuse 21.0 s, distilled visuomotor policy 37.9 s. Deployed, 86.7% of successful subtasks run on the distilled policy. All three levels 86.0% vs reasoning only 76.0% and distilled only 70.0%; 103/120 tasks [hero-2026].

**Outside robotics, same pattern.** HEXIS compiles agent skills into finite-state machines: +16.2 pp over skill-in-context ReAct, 38.4–88.9% fewer tokens (Qwen3.8-27B); updates accepted only after static checks and replay of all previously accepted traces [hexis-2026]. FlowEvo compiles successful workflows into callable skills and suppresses skills with negative transfer: ALFWorld 85.6%, about a third of the tokens [flowevo-2026].

**Classic baselines.** Code as Policies: LLM-written policy code over perception and control APIs [cap-2022]. Voyager: executable skill library; without it progress plateaus; removing self-verification cuts discovered items 73% [voyager-2023].

**Cognitive science.** Anderson's ACT theory: knowledge compilation moves a skill from a declarative stage (facts interpreted by general procedures, verbal rehearsal, high working-memory load) to a procedural stage. Two subprocesses: composition (collapse a sequence of steps into one) and proceduralization (embed the facts into the step). Its signs are speedup, loss of verbal rehearsal and one-step matching [anderson-1982]. This maps onto Fitts's cognitive, associative and autonomous stages [anderson-1982]; Fitts & Posner's 1967 book is the usual citation [fitts-posner-1967, secondhand].

## 3. What works, what doesn't, what's efficient

- **Works:** gate admission on reproduced success (ASPIRE uses a fresh verifier actor; RAX requires an unsupervised run); keep a hand-back path (RAX exceptions); search over whole programs when local repair stalls (ASPIRE evolutionary search +10 pp) [aspire-2026], [rax-2026].
- **Doesn't:** pre-written programs that make all later decisions (URAI 24% vs 56%) [urai-2026]. Distilled skills can lose cues: in SkillEvolBench (180 non-robot tasks), agents often used raw trajectories better than the skills distilled from them, and skill gains were unstable under frozen deployment [skillevolbench-2026].
- **Efficient:** the large savings come from code that runs without the model (RAX ~60–250×); context-only experience saves less (AGP −11% tokens; ASPIRE up to ~9× on tokens to first success).

## 4. Disagreements and open questions

- **What to store:** repair knowledge (ASPIRE), programs (RAX), documents plus scripts (AGP), synchronized action/state data (XLeRobot), raw trajectories (SkillEvolBench). No head-to-head comparison on one robot task.
- **Who compiles:** RAX's agent compiles itself; XLeRobot's routine was refactored by the researchers [xlerobot-2026].
- **Contact-rich compile is unshown.** RAX's precision insertion stayed agent-driven; HERO's distilled policy handles pick, place, push and pull only.
- AGP's refinement gains are uncontrolled by the authors' own account [agp-2026].

## 5. For us

- For us: the promotion gate is "passes unseen frozen starts with no brain calls" (our rule) plus ASPIRE's fresh-verifier reproduction and HEXIS's replay of all previously accepted traces as a regression check.
- For us: demotion = the routine raises and Argon takes over (RAX). Log every hand-back; the share of control Argon holds is our System 1 metric.
- For us: store the skill code and the raw traces it came from; SkillEvolBench says distillation loses information.
- For us: compile motion phases (approach, align, seat) before branch decisions; URAI shows decisions compiled too early lose success.
- For us: plot inference time and output tokens per success separately from sim time, as RAX's Figure 7 does. That curve is our headline.
- For us: AGP's transfer result shows saved experience carries task findings between models, which is why Argon starts from an empty learned-skill library.
