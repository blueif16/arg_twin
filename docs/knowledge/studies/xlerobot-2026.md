# XLeRobot with GPT-6 Astra: body knowledge, experience reuse, emergent skills (2026)

`[xlerobot-2026]` · He, Xie, Cao et al., "Robot Manipulation with GPT-6-Astra: Body Knowledge, Experience Reuse, Emergent Skills, and Sim2Real Transfer", arXiv 2609.31770 · status: verified · local: `data/refs/xlerobot-2026.html`

Read 2026-10-24 from the arXiv HTML. Every number is theirs. Lines starting "For us" are our reading.

## Setup

- Task: press the up-call elevator button. Dual-arm XLeRobot on a mecanum base with head and wrist cameras; a MuJoCo scene and a real lobby.
- Body information levels I0–I3 (I0 = common control interface only; I3 = complete robot geometry plus camera information). Experience levels E0–E3 from one successful source episode: E1 an Astra-written summary, E2 adds sampled images, E3 adds synchronized action and state records.
- Time counts reading materials, observation, model waiting, code generation, actuation and confirmation. Success needs evaluator-side activation plus a correct success declaration with fresh evidence. Three trials per condition.

## Results

- All 30 fixed-start sim trials succeed; the differences are in time.
- Body info at E0: I0 1024.9 s, I1 1160.4 s, I2 579.8 s, I3 437.1 s. **Complete geometry plus camera info: −57.4%.**
- Experience at I0: E0 1024.9 s, **E1 written summary 878.1 s (−14%)**, E2 850.7 s, **E3 synchronized actions and states 321.4 s (−68.6%)**. At I3, E3 still cuts 437.1 → 288.9 s.
- Starts displaced 10–100 cm (9 paired comparisons, 18 trials): experience from the original start cuts mean time 58–63%.
- Astra spontaneously wrote a short move-then-check-image loop. **The researchers** refactored it into parameterised routines with motion and time limits; these cut local-task time 29–31% over 27 sim trials.
- Real robot, 12 trials, shared start: sim XML assets −53.0% time, sim experience −49.9%; real experience transferred to two new starts.

## For us

- Strongest published evidence that giving the brain the robot model and camera calibration pays, and that the useful part of experience is synchronized action and state data, not the model's prose summary.
- REASSEMBLE gives us synchronized poses, forces and video; feed those to Argon as experience, not just its own notes.
- The routine was refactored by humans. Our claim needs Argon to do that step itself, and we should say so explicitly when comparing.
- n = 3 per condition; treat the percentages as large effects on one task, not estimates.
