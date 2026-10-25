# Harness and tools

## 1. Question

What interface between the model and the robot raises success and lowers cost: action granularity, tools, what the model is told about its body, how observations and feedback are presented?

## 2. What is established

**Task-level tools beat raw poses, if the model still decides between calls.**
- URAI, five RoboDojo tasks, four frozen agents: agent-written tools raise aggregate success from 18.0% (direct fingertip control) to 53.0%. With the same tools, a program written in advance reaches 24% vs 56% when agents decide after each call. Three of four agents are 1.3–1.5× faster with 1.5–1.7× fewer output tokens [urai-2026].
- CaP-X, 12 models, 7 tasks including peg insertion: success rises monotonically with human-crafted abstraction, and falls as it is removed. Human experts needed 2–3 weeks per contact-rich task (peg insertion, handover) to find the right primitives [capx-2026].
- NavGPT-3 (navigation): with basic tools Astra reaches 72 SR at 325k tokens per episode; adding mapping and backtrack tools gives 83 SR with 41% fewer tokens [navgpt3-2026].

**Tell the model about its body.**
- XLeRobot elevator button (sim, 3 trials per condition): complete robot geometry plus camera info cuts task time 57.4%; synchronized images, actions and states from one success cut it 68.6%; the model's own written summary of that success only 14% [xlerobot-2026].
- Calibrated geometry raises direct-policy success 19.3 → 72.7 [astranomy-2026]. Google's zero-shot arm demo draws a calibrated 2 cm world grid and a jaw reticle on the wrist image [google-argon-robotics-2026] (prompt only, no numbers).

**Write down what each action does.** Conventions carry most of the grounding; action names are only a prior. Arbitrary symbols with conventions ≈ semantic names with conventions; arbitrary symbols without conventions succeed 1/20, with 23.3% of inferred mappings correct (Gemini-3.1 Pro, 20 trials) [show-harness-2026].

**Text beats raw images inside the loop.**
- CaP-X: execution feedback as text each turn helps every model; interleaving raw RGB each turn hurts; a visual-differencing module that describes changes in text does best [capx-2026].
- Hi-VLA study: describing the image in text helps; adding the simulator's contact information to that text helps most [hivla-study-2026].

**State goals as geometric relations; let code solve for poses.** ReKep writes each task stage as Python functions mapping 3D keypoints to a cost; a solver turns them into end-effector poses in a real-time perception-action loop, and a VLM writes the constraints from language and RGB-D [rekep-2024]. OmniManip places interaction points and directions in each object's canonical, affordance-defined frame; one loop has the VLM check rendered interactions, another tracks 6D pose during execution [omnimanip-2025].

**Expose traces, not just outcomes.** ASPIRE's per-primitive execution traces raise LIBERO-Pro macro success 14% → 62% [aspire-2026].

**Force as text, limits in code.** FORGE-plus (Isaac Lab, sim only): an LLM makes two calls, one to set a per-object force ceiling and one to pick a recovery from a fixed menu given a compact text force signature. A clamp in the fast loop enforces the ceiling. 0.4 mm-clearance gear insertion and fragile bottles: 256/256 episodes without breakage; force-signature recovery resolves 40% and 64% of injected slip failures on two grippers, while a press-harder baseline is ineffective or breaks parts [forge-plus-2026].

## 3. What works, what doesn't, what's efficient

- **Works:** tools that run a whole multi-phase motion and return [urai-2026]; structured per-call traces [aspire-2026]; robot model and calibration [xlerobot-2026]; written conventions [show-harness-2026]; text observations and contact descriptions [capx-2026], [hivla-study-2026]; empty-grasp detection (removing recovery drops Show-Harness to 72%) [show-harness-2026].
- **Twin-side tools help a little:** EmbodiedSWE gave agents a checkpoint tree of simulator states and a CMA-ES parameter-search tool; both models scored higher, "though effect sizes are relatively small" [embodiedswe-2026].
- **Doesn't (or not shown):** Astranomy's tools (mostly image crops) 88.0 vs 86.0 [astranomy-2026]. A real two-robot agentic stack was "flexible but fragile": non-deterministic, prompt-sensitive [agentic-fragile-2026].
- **Human priors still matter on hardware:** SenseNova's Tianji hand grasps 0/3 without grasp primitives, 3/3 with human-written ones [sensenova-astra-2026].
- **Efficient:** Pigey uses 3–15 reasoner calls per real trial, $0.02–0.50, 2–6 min [pigey-2026]. Google's arm prompt asks the model to pick its call granularity on turn 1: coarse 2–4 calls, hover-verify 4–7, fine servoing 8–20 [google-argon-robotics-2026].

## 4. Disagreements and open questions

- **Do tools help?** Yes in [urai-2026], [navgpt3-2026], [aspire-2026]; no in [astranomy-2026]. Our reading: helpful tools encapsulate motion or state the model cannot compute; Astranomy's tools were perception helpers on easy pick-and-place.
- **Who designs the primitives?** CaP-X shows human abstractions raise success but cap expressivity [capx-2026]; RAX uses none and has the agent build its own [rax-2026].
- **Images or text?** Most results favour text summaries, but Astranomy's agentic history ablation shows images still carry information (text only 87.0 vs full 100) [astranomy-2026].
- No source measures how tool-interface versioning affects a running skill library.

## 5. For us

- For us: build `insert(part, hole, max_force)`-style tools from the twin's task frames (our MuJoCo sites play the role of ReKep keypoints and OmniManip's canonical frames, but exact), and keep Argon choosing and parameterising each call. Compile decisions into code only after held-out checks (URAI 56 vs 24).
- For us: give Argon the MJCF, the calibrated cameras and REASSEMBLE's synchronized pose/force records as experience. XLeRobot says the synchronized records matter far more than prose notes.
- For us: each tool returns a structured trace (inputs, outputs, status, frames before and after, contact summary in words) and the conventions file is part of the tool interface, versioned with it.
- For us: report contact as text from simulator state (touching bodies, force level, slip). This is the information Astra lacks when it "cannot tell which physical condition is unmet".
- For us: force ceilings and safety limits live in code, never in the model's output (FORGE-plus pattern).
