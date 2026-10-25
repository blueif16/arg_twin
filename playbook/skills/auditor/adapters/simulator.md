# auditor: simulator input (benchmark with its own predicate)

**Labels.** Use stage words for the furthest stage reached: reach, grasp verified, lift, transport, place or release, done. Then give one cause:

- wrong referent (left/right, a distractor);
- grasp not verified (closed on nothing);
- arrived short (a move that stopped before its target);
- released too high, or tipped;
- false done;
- budget (calls, steps, spend);
- API (an input filter or a degenerate action);
- view convention (mirrored image).

**Extra checks.**
- **Reward-only successes.** The frozen reward fired but the benchmark predicate never held. For each one, read the clip and say which applies:
  - a reward hack (the policy found what the gate never showed);
  - the reward is stricter or looser than the benchmark by design (tilted, hovering, off-centre).

  These are the disagreements in `L.reward_truth_agreement`.
- **Truth-only successes.** The predicate held at some step but the reward never fired. That means the reward is stricter than the benchmark there; list them.
- **Replay.** A counted success has an exact open-loop replay (1e-6). Any mismatch is a finding, not a verdict for the auditor.
- **Observer leaks.** Run `runs/argon/libero_chain/analyze.py` `leak_grep` over every nudge and reflection. It checks truth names and keys, and numbers equal to an object's start coordinate.

**Trial-audit prompt.** Give the instruction, the action conventions, the per-step limits, the command chain with each `move_to` result (steps, stop reason) and the claim. Use the true-orientation clip (`rollout_trueview.mp4`, agentview + wrist) at 4 fps. Never give object poses, the predicate or its trace.
