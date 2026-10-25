# Insertion strategies

## 1. Question

Which classical and recent peg-in-hole strategies (guarded moves, spiral and raster search, tilt-then-insert, wiggle, torque-guided alignment, compliance settings, contact-state estimation from F/T) work, with what parameters, success rates and failure modes? What did the REASSEMBLE operators actually do? Which primitives should task-level tools offer a brain?

## 2. What is established

**Two phases: search, then insertion.** The peg-in-hole literature splits the task into finding the hole under position uncertainty and pushing the peg in once it is over the hole; most work treats only the second [xu-2019]. Whitney's quasi-static analysis names the two insertion failures, jamming (forces and moments in the wrong ratio) and wedging (two-point contact that locks the peg) [whitney-1982, secondhand via xu-2019].

**Blind search: spiral and raster.**
- The common method: press the peg on the surface and move along a fixed path (spiral, Lissajous, raster) until it drops in. It is guaranteed if the path is dense enough and slow when the error is large [pomdp-search-2024]. Chhatpar and Branicky compared tilting near the hole and spiralling towards it [chhatpar-2001, secondhand via pomdp-search-2024 and heuristic-free-search-2022].
- NIST task board, industrial spiral search (ABB force package, ATI F/T, hole error injected up to 2 mm): it widens the tolerance area for every peg type, and round pegs are "caught and aligned almost surely". It does not handle rotation error, so rectangular pegs and connectors jam, and a bump can be mistaken for the hole. Cycle 10.7 ± 4.2 s against 3.3 ± 0.1 s for position control; tuning took more than 5 h against under 0.5 h. The position-controlled arm's own accuracy was 0.14 mm and 0.04° [lian-2021].
- WRS 2018 task board (UR5, Robotiq): spiral with r_max = 4 mm at 2 mm/s, stopped by a 0.9 N·m torque or a force threshold; pulley on a shaft 13/20. All 7 failures were the part tilting in the fingers, and 12 of the 13 successes ended jammed and needed tamping. The authors note the system "does not have means of sensing tilt of the held part within the fingers" [watson-2020].
- In simulation (Drake, chamferless, 1 mm clearance, 6.7–15 mm initial error, 5 N press), spiral search succeeds 509/600 in 102.9 ± 20.5 s over a 46.8 cm path. An active strategy that tilts the peg to read the hole direction from contact succeeds 596/600 in 10.2 ± 3.2 s over 1.33 cm [pomdp-search-2024].
- In a kinematic simulation the Archimedean spiral was the best compromise. The Fermat spiral sometimes skipped over a nearby hole [maddio-2020].
- Search cost grows with the uncertainty area: path ≈ πr²/pitch, so halving the error radius cuts it 4×. The pitch has to stay under the capture width, about twice (chamfer + clearance) (for us, from the geometry).

**Tilt-then-insert.** Tilt the peg away from the hole axis, lower its leading edge into the hole, then rotate upright under pressure. Watson's tilt-in for a bearing (+0.2 mm tolerance) succeeded 13/20, with jamming as the main failure [watson-2020]. Van Wyk et al. report 0.88 for tilt-in and 0.95 for spiral search with larger 3D-printed parts [vanwyk-2018, secondhand via watson-2020]. RAX's agent entered 0.1 mm holes corner-first with a tilt, then rocked the peg upright [rax-2026]. Jin et al. classify the contact pose from tilt-then-rotate patterns [jin-2021].

**Learned and precise insertion.** For 35 mm pegs with 10–20 µm clearance, an LSTM policy trained by RL succeeded in 100/100 trials per case with 1–3 mm and 0–1.6° initial error in 3.5–4.7 s. A commercial force-sensing controller with fixed search patterns is rated for ≤ 1 mm and ≤ 1° in about 5 s [inoue-2017]. IndustReal reached 83–99% over 600 real trials, counting only successes with penetration under 0.5 mm [industreal-2023].

**Compliance and impedance settings.**
- Insertion runs on a compliant Cartesian impedance controller, and its stiffness sets what a "press" is. The REASSEMBLE recordings used the controller code's default of 500 N/m translational and 50 N·m/rad rotational stiffness with damping ratio 1, tunable live (`docs/research/reassemble-data-notes.md`). The franka_ros example controller defaults to 200 N/m (range 0–400) and 10 N·m/rad (range 0–30) [franka-ros-2024].
- At 500 N/m a 3 N press means a target 6 mm below the surface. A force ceiling on the spring caps the press but does not make one.
- Stiffness tuned per insertion helps at tight clearance. A gain-tuning policy trained on a generic peg reached 9/10, 7/10, 10/10 and 9/10 at 0.05 and 0.02 mm clearance, against 0–5/10 for the baselines [compliance-tuning-2023]. FORGE policies trained with randomized robot, controller and part dynamics are robust "regardless of the controller gains" [forge-2025]. That suggests randomizing stiffness (the twin's cousins use 200–800 N/m) can stand in for tuning it exactly (our inference).

**Keyed and square parts: orientation as well as position.** Spiral search does not correct rotation error, so rectangular pegs and connectors jam [lian-2021]. Probing with a particle filter localized a key in a lock in 50 trials, cutting the uncertainty by more than 95% [chhatpar-2005]. The REASSEMBLE operators twisted square pegs by more than 10° in 31% of successful attempts (table below), so twisting under light pressure is part of what operators do, though it does not separate outcomes (section 4). For a square peg, the yaw error is the board's yaw error plus the in-hand yaw.

**Force ceilings in code, recovery by the model.** FORGE-plus enforces a per-object force ceiling in the fast loop and lets an LLM choose recovery from a menu given a text force signature. Gear insertion at 0.4 mm clearance went 256/256 without breakage [forge-plus-2026].

**Contact-state estimation from F/T.** Contact states (no contact, one-point, two-point) are recognized by analytical models or by classifiers. On one assembly dataset, GMM reached 94.4% recognition, SVM 64.2% and boosting 60.7% [jasim-2017, secondhand via xu-2019]. Probing with particle filters cuts position uncertainty by more than 95% [chhatpar-2005].

**Google's zero-shot primitives** (Gemini 4 Argon arm demo, prompt only):
- `look_at`, `grasp`, `place`, `place_until_contact(x, y, z, max_lag=0.012, relief=0.002)`, `goto`, `move_by`, `move_until_contact(dz=-0.06, speed=0.03, max_lag=0.012, relief=0.002)`.
- Contact is detected by tracking lag (12 mm) rather than force, and the arm backs off 2 mm afterwards.
- The model picks a granularity: coarse 2–4 calls, hover-verify 4–7, fine servoing 8–20 [google-argon-robotics-2026].

**What the REASSEMBLE operators did** (TRAIN in-hand peg Insert attempts with a contact onset, 303 of them, contact onset to release; `runs/claude/run-standin-1/task_tools/research/operator_strategies.json`, made with `twin.replay` signals; medians):

| | Round, success (116) | Round, failure (28) | Square, success (122) | Square, failure (37) |
|---|---|---|---|---|
| Contact time (s) | 2.7 | 2.6 | 4.5 | 5.7 |
| Hand tilt from vertical at contact (deg) | 4.4 (43% > 5°) | 8.7 (75% > 5°) | 5.7 (53%) | 6.0 (62%) |
| Yaw twist range (deg) | 1.7 (5% > 10°) | 1.9 | 5.7 (31% > 10°) | 6.1 (35%) |
| Lateral path, extent (mm) | 21, 8.7 | 23, 9.2 | 30, 9.1 | 58, 14.2 |
| Direction reversals | 4 | 4 | 7 | 9 |
| Share searching (path > 5 mm, ≥ 2 reversals) | 87% | 82% | 93% | 100% |
| Push Fz median, peak (N) | 2.0, 5.7 | 2.2, 5.9 | 2.4, 6.5 | 2.5, 6.3 |
| Time to 2 mm of descent (s) | 1.8 | 1.7 | 2.4 | 2.9 |

- The real envelope (1st–99th percentile of successes) agrees: peak force 3.1–13.8 N round and 3.2–12.6 N square; approach speed 12–122 mm/s (median 37 and 27); contact 1.0–19 s and 0.7–27 s (twin/envelope.py, sim-integrity Lessons).
- The read-footage notes say the same on recorded video. Thin pegs go in under 6 N: lower over the hole, search briefly sideways (12 mm path, 2 reversals) at +4 N, and the peg drops in (`runs/footage/strategy_notes.json`, insert/round peg).
- The failure codes so far: F-JAM-AXIAL (a part held axially under a sustained 8–14 N pull, small gear), F-ABORT-EARLY (a short probe, then let go), F-GRASP-POSE, F-GRASP-SLIP and F-THREAD-NO-CATCH (`runs/footage/failure_taxonomy.json`).
- On train, 55 of 69 real peg failures are operator aborts rather than jams (validation.yaml v10 history), which is consistent with F-ABORT-EARLY.

## 3. What works, what doesn't, what's efficient

- **Works:**
  - A guarded descent, then a light-press search, then a push. This is what operators do (a 2–2.5 N median press, a 20–30 mm search path) and what industrial spiral search does [lian-2021, watson-2020].
  - Round pegs are forgiving: the chamfer catches the hole [lian-2021].
  - A force ceiling in code [forge-plus-2026].
  - Tilting to widen capture or to read the hole direction [pomdp-search-2024, rax-2026].
- **Doesn't:**
  - Spiral search against rotation error: rectangles jam [lian-2021].
  - Any search when the part tilts in the fingers, which is unsensed and was the cause of all 7 pulley failures [watson-2020].
  - Pressing harder to unjam [forge-plus-2026].
  - Blind search over large errors: about 100 s for 7–15 mm [pomdp-search-2024].
- **Efficient:**
  - Shrink the uncertainty before searching, by vision [lian-2021: 3D vision 100% except 4/150 on RJ45] or by contact [chhatpar-2005].
  - Search time scales with area/pitch.
  - Learned policies reach 3.5–4.7 s at µm clearance [inoue-2017].
  - Operators finish a round peg in a median of 2.7 s of contact.

## 4. Disagreements and open questions

- **Spiral or tilt?** Van Wyk reports spiral above tilt (0.95 vs 0.88) [vanwyk-2018, secondhand]; Wang et al. report active tilt search far above spiral in success and time [pomdp-search-2024]. The tasks differ: large plastic parts with known positions, against chamferless pegs with a 7–15 mm error.
- **Twist for square pegs.** 31% of successful square attempts twisted more than 10°, and so did 35% of failed ones. Twist is common in both, so it doesn't separate outcomes.
- **Tilt at contact.** Round failures started more tilted than round successes (8.7° vs 4.4° median; 75% vs 43% above 5°), consistent with jamming theory. No causal test exists.
- **In-hand pose.** No source measures in-hand tilt online with a parallel gripper and a wrist F/T alone. Watson names it as missing [watson-2020]; the twin's starts carry pivot up to 26° (p90 14.5°).
  - In our twin (2026-10-25, author-tools Lessons), one tip contact cannot separate an along-pad offset from a tilt: both move the touch point along the pads. A part that turns in the pads can still be detected. Between a 1.5 N and a 3 N press, the contact moved 1.70–1.74 mm in the hand frame and the hand sank 0.39–0.40 mm for round peg 1 at 12–17° of in-hand tilt. For rigid grips (round peg 2) the numbers were 0.03–0.14 mm and 0.05–0.09 mm. This is a twin measurement on a few starts, not a published result.

## 5. For us

- For us: offer the primitives operators and industry use (guarded touch, light-press search, push with alignment and wiggle, release) as task tools, with force ceilings from the real envelope in code (`harness/tools/task.py`, task/v1).
- For us: tools act on the believed geometry. Contact supplies what perception lacks: the surface height, the touch point and the drop. Search radius and time grow with the board error (p90 4 mm) and the in-hand offset (p90 5.4 mm), so a brain that corrects the belief, from images or touch, saves search time as the square of the error.
- For us: in-hand tilt is the failure that search cannot fix [watson-2020]. Align the hand by the wall torque once the tip is engaged, and report the residual tilt. A tilt-probe search is the next strategy to try for large errors [pomdp-search-2024].
- For us: check the grip before searching. A two-level press tells a turning grip from a rigid one, and a tool should then stop and ask for a grip correction (wrist view) or a regrasp rather than search. Searching on round peg 1 turned it from 15° to 64° in the pads, and the TCP's descent was read as a drop.
- For us: presses are spring targets. Keep the commanded depth equal to force / stiffness and the spring ceiling near the press, so a light search press stays light (author-tools Lessons).
- For us: keep tool defaults inside the real envelope (approach ≥ 12 mm/s, press 1–6 N, contact ≤ 19–27 s) or report the violation. The twin's sideways lag at speed limits how fast a precise touch can be (author-tools Lessons).
