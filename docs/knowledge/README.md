# Knowledge base

What published work has shown to be possible, good and efficient for this project. Decisions start here: if a method isn't covered, research it and add it before proposing it.

Every agent reads this, Claude subagents and Argon alike. It is written to be acted on, not as a literature review.

## Layout

| Path | What |
|---|---|
| `README.md` | This index and the conventions |
| `references.md` | One entry per source: id, title, authors, year, venue, link, local copy, what we use it for, verification status |
| `<topic>.md` | One synthesis per question (list below) |
| `studies/<id>.md` | Deeper notes on the few studies we lean on most |
| `data/refs/<id>.(pdf\|html\|md)` | Original copies, outside git. These are untrusted downloads: read them, never execute them |

## Topics

<!-- Each topic agent adds its rows here: | [file](file.md) | the question it answers | -->

| File | Question it answers |
|---|---|
| [contact-localization](contact-localization.md) | How to place a fixture (board, holes) in the robot frame from contact events when the in-hand pose changes every grasp |
| [llm-as-robot-policy](llm-as-robot-policy.md) | Can a frontier model be the policy itself, where does it break on precision and contact tasks, and what does a success cost? |
| [harness-and-tools](harness-and-tools.md) | Which interface (tools, action units, body model, observations as text, traces) raises success and cuts cost? |
| [skill-libraries-and-compilation](skill-libraries-and-compilation.md) | How do agents compile reasoning into reusable skills and programs, and how much do tokens and time per success fall? |
| [two-speed-control](two-speed-control.md) | How should slow reasoning and fast control share the robot: when to wake the brain, how to hand back control? |
| [learning-signals](learning-signals.md) | How to score progress and success densely enough to learn from practice: value functions, progress, rewards |
| [real2sim-twins](real2sim-twins.md) | What fidelity a twin needs to track real outcomes, failures and forces; what to copy exactly and what to randomize |
| [contact-physics-and-sysid](contact-physics-and-sysid.md) | Can MuJoCo reproduce tight insertions and their forces? What to identify, what to randomize, and how to replay teleop logs with no recorded setpoint |
| [calibration-and-localization](calibration-and-localization.md) | How well cameras, board, tool and in-hand part can be placed in the robot frame, how much pose error insertion tolerates, and how to validate a calibration |
| [evaluation-and-statistics](evaluation-and-statistics.md) | When a twin result means something about the real robot; trial counts, paired tests, held-out design, and known-error tests for checks |
| [insertion-strategies](insertion-strategies.md) | Which peg-in-hole strategies (guarded touch, spiral/raster search, tilt-in, wiggle, alignment, compliance) work with what parameters, and what the REASSEMBLE operators actually did |

Agent and LLM-control study notes in `studies/`: astranomy-2026, agp-2026, aspire-2026, rax-2026, urai-2026, xlerobot-2026, hivla-study-2026.

Real-to-sim, physics, calibration and evaluation study notes in `studies/`: pnd-2026, simpler-2024, dexforce-2025, touching-gnn-2025, sunderlabs-nist-2026, step-2025.

Project-specific research notes stay in `docs/research/`:
- REASSEMBLE data
- Interactions API
- Google's twin pipeline
- environment setup

## Conventions

- **Reference ids** take the form `<short-name>-<year>`, e.g. `astranomy-2026` or `dexforce-2025`. Cite as `[astranomy-2026]` after the claim.
- **Every claim carries a number or a concrete condition, plus its id.** Mark how it was checked:
  - `[id]`: verified in the primary source
  - `[id, secondhand]`: only seen in a summary or another paper
  - `[id, unverified]`: a claim we were given that the source doesn't support. Say what the source says instead.
- **Keep their findings separate from our reading.** Lines starting "For us:" are our inference, not the source's.
- **Topic file sections:**
  1. Question
  2. What is established, with numbers
  3. What works, what doesn't, what's efficient
  4. Disagreements and open questions
  5. For us
- **Be concise.** A topic file fits on a few screens, and a study note on one.
