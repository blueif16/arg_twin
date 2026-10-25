# Learning signals: value functions, progress estimation, reward design

## 1. Question

How can practice be scored densely enough to learn from: how far along an attempt is, whether it succeeded, and which of two attempts was better? What can a frontier model estimate on its own, and where must the signal come from the simulator?

## 2. What is established

**VLMs can estimate progress zero-shot, if the frames are shuffled.** GVL asks a frozen VLM for the completion percentage of each frame of a shuffled trajectory. In time order, VLMs shortcut to monotonically rising values and ignore the task. Zero- and few-shot on 300+ real tasks; in-context examples from other tasks or human videos help; supports dataset filtering, success detection and advantage-weighted regression. Metric: value-order correlation (VOC) with true time order on expert videos [gvl-2024].

**Argon is shown doing this, without accuracy numbers.** Google's post: dense progress over a 60 s bimanual UMI episode, split into 28 segments with positive, neutral and negative slope; a 4-level hindsight annotation of a 136 s ALOHA origami episode (goal, 6 subtasks, 17 contact-level primitives, quality control) that holds progress flat during fumbles and down-weights them 0.15–0.25× for imitation [google-argon-robotics-2026]. Prompts only.

**Simple value targets work at scale.** RECAP (π*0.6) derives reward from the episode success label alone: −1 per step, −C_fail at a failed end, so value = −(steps to success). An advantage-conditioned VLA trained on demonstrations, autonomous rollouts and interventions "more than doubles task throughput and roughly halves the task failure rate" on the hardest tasks [pistar06-2025]. ViVa swaps in a video-generative value model and reaches 80% average success inside RECAP [viva-2026].

**Metadata labels let failures become training data.** π0.7 conditions on episode speed (binned), quality (1–5) and per-segment mistake labels, so it can train on failed and autonomous episodes and be prompted at run time for "fast, high quality, no mistakes" [pi07-2026].

**Progress should follow procedure, not time.** ProcVLM assigns progress budgets per subtask and spreads them by visual change within the subtask, so stalls and failures do not look like progress [procvlm-2026].

**VLM judges get exploited.** SOLE-R1: as RL rewards, strong VLMs (GPT-5, Gemini-3-Pro and others) "often fail under partial observability and distribution shift, enabling policies to exploit perceptual errors"; their trained video reasoner supports zero-shot online RL on 24 unseen tasks [sole-r1-2026].

**LLM-written rewards and priors.** Eureka: evolutionary search over LLM-written reward code beats expert human rewards on 83% of 29 RL environments, +52% normalized [eureka-2023]. RLingua: an LLM-written rule-based controller supplies early actions and a regulariser for TD3, cutting sample needs [rlingua-2024].

**Cheap online improvement of an existing policy.** DSRL runs RL in a diffusion policy's latent noise space, black-box, no weight changes [dsrl-2025]. RLT fine-tunes only the critical precision phase with a small actor-critic on an "RL token": up to 3× faster and screw insertion 20% → 65% within hours [pi-rlt-2026]. In residual RL, anchoring the critic on base-policy transitions and normalising it fixes cold-start value learning [residual-value-2026].

## 3. What works, what doesn't, what's efficient

- **Works:** shuffled-frame progress [gvl-2024]; success-only reward turned into time-to-success value [pistar06-2025]; stage-based progress [procvlm-2026]; self-verification (removing it cost Voyager 73% of discovered items) [voyager-2023].
- **Doesn't:** asking a VLM to read progress off frames in order [gvl-2024]; trusting a general VLM as the sole reward [sole-r1-2026]; trusting the model's own "done" (SenseNova: it cannot tell which physical condition is unmet) [sensenova-astra-2026].
- **Detector errors are asymmetric:** ~10% success-detector error is harmless; false positives hurt most [hivla-study-2026].
- **Stage rubrics are cheap and informative:** Robocurve grades each trial 0–4 (contact, lifted, above target, placed), so failed runs still show progress [robocurve-astra-2026].

## 4. Disagreements and open questions

- VLM progress estimates are useful for filtering and ranking footage, but whether they are accurate enough on sub-millimetre contact phases is untested; GVL's tasks are mostly coarse.
- RL fine-tuning (RECAP, RLT, DSRL) and LLM-written code skills (RAX, ASPIRE) are competing routes to the fast layer. No study compares them on the same contact-rich task.

## 5. For us

- For us: in the twin, success and progress come from simulator state (insertion depth, axis alignment, contact force), not from a VLM. The sources show VLM judges are exploitable and models cannot verify contact conditions.
- For us: use VLM-style progress where there is no simulator: scoring Argon's reading of REASSEMBLE footage. Shuffle frames (GVL), score with VOC against the dataset's own segment labels, and hold progress flat during fumbles.
- For us: RECAP's time-to-success value is the natural target for comparing attempts and for promoting skills: a routine is better if it reaches the same success in fewer steps.
- For us: tag every practice episode with π0.7-style metadata (speed bin, quality, mistake segments). It makes failed attempts usable later and costs nothing in the twin.
