# What matters in hierarchical VLA agents (Hi-VLA study, 2026)

`[hivla-study-2026]` · Hu, Shridhar, Lu, Shah et al., "What Matters in Orchestrating Robot Policies", arXiv 2606.10267 · status: verified · local: `data/refs/hivla-study-2026.html`

Read 2026-10-24 from the arXiv HTML. Every number is theirs. Lines starting "For us" are our reading.

## Setup

- A VLM planner issues language subgoals to a VLA controller, framed as options: a subgoal holds until a termination condition fires. One component varied at a time.
- Main experiments in the MuJoCo ALOHA suite (short-horizon, long-horizon, reasoning tasks); a real ALOHA check. Gemini 2.5 Flash with thinking for the auxiliary queries.

## Findings

- **Planner:** turning on thinking matters a lot; VLM size matters little. **Controller:** larger, steerable VLAs matter; a VLA fine-tuned narrowly on one instruction stops following others.
- **Termination:** a success detector works best; a fixed horizon of 4–8 s is an acceptable cheaper option; letting the VLM predict execution time works worst. About 10% detector error does not hurt; higher **false-positive** rates hurt badly because the planner moves on too early.
- **Observation:** converting the image to a text description helps, especially with bounding boxes; adding privileged **contact information** from the simulator to that description helps most.
- **Memory:** more in-episode history barely matters; **cross-episode** summarised experience helps a lot.
- Combined best vs naive vs flat VLA (sim): long-horizon 67.08 vs 40.56 vs 25.30%; reasoning 80.89 vs 66.49 vs 50.90%; short-horizon 78.22 vs 69.57 vs 69.63%. Real ALOHA fruit-to-plate: 12/15, 9/15, 3/15 (the real version uses bounding boxes and no privileged contact).

## For us

- End each step on a success check, not on a timer the brain guesses. Our checks read simulator state, so they are close to their privileged detector; guard against false positives above all.
- Hand Argon contact state as words (which bodies touch, force level), computed from the twin. This was their single best observation change.
- Lessons that carry across episodes are where memory pays; raw in-episode history is not.
