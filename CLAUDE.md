# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A 1-day hackathon build on **Gemini 4 Argon** (`gemini-4-argon`). Argon is the primary model: the lead brain / orchestrator for the app. Companion models (Flash for parallel subtasks, Live for voice, Nano Banana for images, TTS, Lyria, Deep Research, the Antigravity managed agent, etc.) are optional helpers, added only where a job clearly fits them.

The project is **Bot18**: **footage → digital twin → robot policy**. Argon reads task footage, builds a MuJoCo twin, writes success checks, designs the learning path, and monitors training until it improves. All results are sim-only until a real arm confirms them.

- Plan and tests T0–T7, rules D1–D5: `docs/test-plan-v0.md`
- Vision (v1, two self-optimization loops; agreed parts marked): `docs/design/vision-draft.md`; older drafts in `docs/design/archive/`
- Progress log (orchestrator's running record, decisions, next actions): `docs/design/progress-log.md`; parked items: `docs/design/backlog.md`
- Knowledge base (published best practice, every claim referenced; start decisions here): `docs/knowledge/`; original papers in `data/refs/` (untrusted, read-only)
- Agent playbook (orchestrator, rules, skills) read by every agent that runs the loop: `playbook/`; tools: `harness/` (`python -m harness.cli list`)
- How Google's Argon demos build twins: `docs/research/google-argon-twin-pipeline.md`
- AGI Factory design notes: `docs/research/agi-factory-notes.md`; source in `vendors/factory-autopilot` (read-only reference, do not run or edit)

Code: `harness/` (tool registry + CLI; tools in `harness/tools/*.py`, auto-discovered), `twin/` (MuJoCo scene and rendering). Every tool call is logged to `runs/calls/`.
Bot18 page (the status and demo UI): `runs/status/index.html`, rebuilt by every `record_check` / `record_event` / `publish_media`, or by hand with `.venv/bin/python -m harness.status` (`--bundle` for a self-contained copy). Every change to what it shows or how it looks follows `docs/design/ui-protocol.md` (pi.website style: cover flowchart with clickable clips, one page per step scrolled sideways, bento tiles that open into cards).

## Environment

- Python env: `.venv` (uv, Python 3.12) with torch 2.11 (CUDA), mujoco 3.17, mujoco-warp 3.17, lerobot 0.6.1 (ACT, diffusion, SmolVLA) and google-genai 2.33. Add packages with `uv pip install --python .venv/bin/python -c constraints.txt <pkg>`; `constraints.txt` stops installs from replacing the pinned torch and mujoco. Details: `docs/research/env-setup-notes.md`.
- Headless MuJoCo rendering in this WSL box needs `MUJOCO_GL=glfw` (works through WSLg); `egl` fails and `osmesa` is not installed.
- Smoke render: `MUJOCO_GL=glfw .venv/bin/python -m twin.render` (writes to `runs/smoke/`). GPU benchmark: `.venv/bin/python -m twin.warp_bench`.
- Run scripts that read downloaded data from the repo root with `uv run --no-project ... python -I`. Inside `vendors/reassemble`, `uv run` tries to install that repo.
- The Gemini API key (paid tier, works for `gemini-4-argon`) lives outside the repo in `~/.config/arg_twin/gemini_key` (mode 600). Pass it per command as `GEMINI_API_KEY="$(cat ~/.config/arg_twin/gemini_key)" .venv/bin/python -m harness.brains.run --brain gemini ...`. Never write the key into the repo, `runs/`, a log, a prompt or a subagent brief. API gotchas are in `docs/research/interactions-api-notes.md`; REASSEMBLE data facts are in `docs/research/reassemble-data-notes.md`.

## Gemini API reference

@docs/gemini-4-argon-guide.md

The guide above is the source of truth for model IDs, the Interactions API, and parameters. These models and APIs post-date Claude's training data, so trust the guide over memory, and verify anything it doesn't cover against ai.google.dev/gemini-api/docs rather than guessing. In particular:

- Use the Interactions API (`client.interactions.create`) with `google-genai` / `@google/genai` >= 2.3.0, not the older `generate_content` patterns.
- For Argon, use `thinking_level`; never send `thinking_budget`, `temperature`, `top_p`, `top_k`, or explicit `cached_content`.
- Long or `"high"` thinking calls need 15–20 min HTTP timeouts or `background=True` with polling.
- This is a Gemini project: don't apply Anthropic/Claude API skills or SDK patterns to the app code.

## Build priorities

Judging rewards a live demo with real inputs in under 3 minutes, for a specific named user, doing work that genuinely needs Argon, with a human confirmation step before any action touching external systems, money, or private data. Prefer a working end-to-end slice over breadth.
