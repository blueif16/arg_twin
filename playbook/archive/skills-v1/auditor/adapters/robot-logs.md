# auditor: robot-logs input (twin built from real recordings)

**Labels for contact tasks (peg twin).**
- Success sources:
  - S1 first aim from images;
  - S2 image aim with the tip measured;
  - S3 tilt or yaw corrected first;
  - S4 routine re-aim by touch;
  - S5 brain re-aim from probe touches;
  - S6 compliance move (yaw test, release on jam);
  - S7 recovery.
- Failure causes:
  - F1 aim error;
  - F2 tip offset not measured;
  - F3 tilt;
  - F4 yaw;
  - F5 jam handling;
  - F6 stuck stop;
  - F7 force flag;
  - F8 numerics flag (dt/2);
  - F9 verification or claim error;
  - F10 tool misuse.

**Extra checks.**
- **dt/2 flags on closed-loop successes.** An open-loop rerun can miss a chamfer capture at sub-0.1 mm clearance. Replay in memory and find the first diverging command. If dt and dt/2 agree up to a capture the brain then corrected, the verdict is "rule-correct but conservative".
- **Force flags.** Name the command and its `force_max`. A plain pull or lift of a wedged part exceeds `force_max` (14.6–74.3 N seen).
- **Envelope.** On a routine, check its own segment up to any hand-back (validation v17 E rule).

**Trial-audit prompt.** Give each command's peak |F| against the family's largest real peak (13.96 N for round pegs), the tool return codes (NOT_FOUND, JAMMED), and the warning that believed depth is nominal. Render clips from hand + hama1 at 4 fps, plus hama2 or a tight crop after release.
