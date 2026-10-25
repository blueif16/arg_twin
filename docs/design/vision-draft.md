# Vision draft (v0): proposal, not agreed

> Status: a draft offer of the plan, put together from the discussion on 2026-10-24. Expect it to change as real obstacles and search findings come in. Once it is agreed, we expand it with full references into v1.

## The claim

Argon turns real robot footage into a simulated world shaped around the task, then teaches itself the task inside that world. With practice, work moves out of slow reasoning and into fast routines, so each attempt is more reliable and cheaper in tokens and time.

All the parts serve one goal: efficient task completion.

- The "policy" is Argon plus its harness: tools, compiled skills and a watcher.
- Small models are a later, optional experiment.

## How it is built

- **Fixed orchestrator.** The top level is fixed and does not change. It watches the whole loop, runs the stages, enforces the rules and decides when each stage is done. See `playbook/orchestrator.md`.
- **Playbook and skills: the main asset.** Each step of the flow is an agentic workflow driven by a skill file. A skill file states:
  - the goal
  - the tools
  - the rules
  - the definition of done
  - the lessons learned so far

  Skills give goals and checks, not step-by-step recipes. Agents work out the specifics, test them, and write back what worked. See `playbook/`.
- **Thin harness, with code only where it must be deterministic.** That covers:
  - the tool registry
  - the simulator
  - the frozen evaluator
  - the call recorder

  See `harness/` and `twin/`.
- **Swappable brain.** Claude agents stand in until the Gemini key arrives. Argon then runs the same skills through the same tools. Before Argon runs, its skill library is wiped, so its improvement is its own. Every Argon call is recorded and can be replayed.

## The loop

1. **Read the footage.**
   - Argon reads the long recordings and zooms in itself at contact moments.
   - Outputs:
     - hierarchical labels and progress
     - strategy notes (how operators tilt, wiggle and twist)
     - a failure taxonomy
   - Labels are scored against the dataset's own labels.
   - This stage also decides what the twin must get right. Insertions fail most often, and the small gear is the hardest.
   - *From:* Google's Argon robotics post, GVL.
2. **Build the twin.**
   - Shared foundations come first:
     - cameras, inferred natively from the images and then checked by overlay
     - the board frame
     - the tool-tip offset
     - a scan of the scene for the wrist camera
   - Then parallel builders take one part family each. Every part gets task frames (MuJoCo sites: entry, axis, clearance, grasp), success checks and contact parameters.
   - Each part family has its own separate watcher.
   - Precision goes only where outcomes depend on it. Each parameter carries an uncertainty, so we get a spread of worlds rather than one exact copy.
   - Checks: the overlay matches on all cameras, and replays of real attempts, including failures, match both the outcome and the force trace.
   - *From:* Google recipe B, real2sim-eval, intent-driven Real2Sim, ASID, ACDC, the laptop NIST study.
3. **Tools come from the twin.**
   - Tools are written in task terms, for example `insert(part, hole, max_force)`, and code turns the relations into poses.
   - Argon extends the toolset, and changes to the interface are versioned and gated.
   - *From:*
     - GPT-6 stalling at insertion when it has to give raw poses
     - ReKep and OmniManip
     - URAI: agent-written tools raised success from 18% to 53%
     - the XLeRobot study: a body model plus synced demos cut time by 57–69%
4. **Practice (System 2).**
   - Argon decides only at decision points, while skill code runs the fast force loop.
   - The watcher wakes Argon only when something happens.
   - Argon plans one step ahead, with branches.
   - Every run happens in two modes: physics paused while Argon thinks, and real time.
   - *From:*
     - Helix and NavGPT-3
     - the hierarchical planner study: with synchronous planning the arm moves only 28.7% of the time
     - the Hi-VLA study: end each step on success, describe contact in words
5. **Compile (System 2 → System 1).**
   - A decision that works becomes a verified skill program once it passes starts it has never seen.
   - Control goes back to Argon when a routine meets something new.
   - *From:*
     - RAX
     - URAI: deciding after each call reached 56%, a program written in advance only 24%, so Argon stays at the decision points
     - Fitts and Posner
3. **Physics questions go back to the twin.**
   - If a behaviour relies on physics the replays never checked, that becomes a reconstruction request, checked against the real recordings.
   - Strategies outside what the human data covers are flagged.

## What we prove

The evaluator is frozen and checks simulator state.

1. **The twin is faithful (reconstruction; agreed 2026-10-24).** Given only what the log lacks (the part's pose in the gripper, the stiffness setting), the twin seats the real successes and jams the real jams, with matching force, and a permissive twin cannot. Predicting outcomes open-loop is a stretch goal, reported apart: the log leaves the in-hand pose about 6 mm uncertain against 0.05–0.25 mm clearance.
2. **Argon understands the crux.** The rules it states predict which held-out real attempts fail.
3. **Self-improvement against a strong baseline (agreed 2026-10-24).** All comparisons are in the twin; there is no real arm. Attempt 1 is the strongest published uncompiled recipe (Astranomy: absolute EEF targets, full history, calibrated geometry in the prompt), run by us on the same frozen starts as attempt N, with paired tests. Their numbers come from a different benchmark; we reuse their recipe, not their numbers. The real human teleop success rate and time per part (REASSEMBLE) is a second reference line, meaningful only once B2 shows the twin reproduces real outcomes. From attempt 1 to attempt N:
   - success goes up
   - tokens and time per part go down
   - Argon's share of control shrinks
4. **Transfer.** A harness built on about 8 parts makes the other 9 cheaper from their first attempt.

## What we show (3 minutes, visual)

1. The footage and the twin snapping into alignment, with a real jam reproduced in the twin.
2. Argon's attempt 1: slow, thinking at every step, stalling at the insertion.
3. Attempt N: fast and driven by routines. A "who's in control" timeline shows Argon appearing only when something surprises it.
4. An unseen part: Argon steps back in, then fades out faster.

## Known facts and constraints

- **Gemini key.** It arrives tomorrow and is a demo key, so Argon runs must be recorded.
- **REASSEMBLE gaps.** See `docs/research/reassemble-data-notes.md`:
  - no camera intrinsics, and buggy mocap camera poses
  - the board layout differs from the NIST CAD
  - `action` is the measured next pose, so the commanded target has to be rebuilt from force
  - the tool-tip offset is 0.174 m
- **Environment.** See `docs/research/env-setup-notes.md`: MuJoCo, MuJoCo Warp and LeRobot are ready on the GPU.
- **Earlier drafts.** These are in `docs/design/archive/` and are superseded by this file.
