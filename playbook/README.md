# Playbook

The playbook is what agents read to run the footage → twin → practice loop. The reader can be a Claude subagent today or Argon tomorrow, and the same files serve both. Read `orchestrator.md` for the loop and `rules.md` before touching anything.

## Layout

| Path | What it is |
|---|---|
| `orchestrator.md` | The fixed top level: stages, what to watch, when a stage is done, when to escalate |
| `rules.md` | Rules every agent follows: integrity, no sim cheating, frozen evaluation, actor/watcher split, records |
| `skills/<name>/SKILL.md` | One skill per job, shared by every data source: a source-neutral core with a "Choose your path" table |
| `skills/<name>/adapters/<input-type>.md` | Best practice per input type, with evidence: `robot-logs`, `video-only`, `simulator`, `prior-runs` |
| `skills/<name>/references/<source>.md` | Facts, conventions and findings tied to one source (`reassemble`, `so101`, `libero`, `assembly101`, ...); read with `read_skill(name, file)` |
| `archive/skills-v1/` | The v1 skills (one per REASSEMBLE step), kept read-only |
| `validation.yaml` | Every stage's checks and frozen targets, the data tracks, and the version history |
| `splits.json`, `splits_so101.json` | Frozen train / gate / exam splits for REASSEMBLE, and train / gate for SO-101 (no exam) |

## Skill file format

```markdown
---
name: <kebab-case>
description: <one line: when to use this skill>
stage: <read | build | tools | practice | compile | watch | infra>
steps: [<validation.yaml stage ids, e.g. B0, B1> | loop]
---
## Goal            what done looks like, in observable terms
## Choose your path   input type you were given -> adapters/<input-type>.md and references/<source>.md
## Inputs          files, data, prior skill outputs
## Tools           harness tools (by registry name) and outside libraries / sources
## Approach        guidance, not a fixed recipe: options, what to try first, how to tell if it works
## Per-model notes   where brains differ (Claude reads PNG frames; Argon takes video in a user step, numbers via code_execution or high thinking)
## Checks          the definition of done: checks an independent watcher can run
## Lessons         dated notes from real attempts: what worked, what failed, numbers
```

How to write skills:

- **Give goals and checks, not step-by-step recipes.** The agent running a skill works out the specifics, tries them, and records what happened under Lessons. A skill only counts as working once Lessons shows a real successful run.
- **Write for any source.** The core holds only what is true on every source; per-input-type practice goes in `adapters/`, per-source facts and source-specific Lessons detail in `references/`. Lessons stay one dated line each in the core (the page reads them there), naming the source and the reference.
- **Write for any agent.** Don't assume Claude-only or Gemini-only abilities. Where they differ, say so; for example, Claude reads frames as images while Gemini can take video clips.
- **Ground every claim** in a file, a tool result or a source. Published best practice lives in `docs/knowledge/` (topic syntheses plus `references.md`). Start from it, cite its ids under Sources, and add to it when you research something new.

## Tools

Tools live in `harness/tools/*.py` and are registered with `@tool` from `harness/registry.py`. Every brain sees the same set:

```sh
.venv/bin/python -m harness.cli list                          # declarations (JSON schema)
.venv/bin/python -m harness.cli call <tool> '<json>' --caller <agent-name>
```

- Every call is appended to `runs/calls/` so runs can be replayed.
- Tools return JSON. Images and clips are written under `runs/` and returned as file paths.
- Simulation code lives in the `twin/` package; tools wrap it.

## Validation and status

Watchers measure; the harness judges. `validation.yaml` holds every stage's checks and frozen targets.

- `record_check(check_id, value, evidence, brain, by)` records one measured number. The harness looks up the target and decides pass, fail or track itself, so never pass a verdict. When a check flips or a stage changes state, it logs the milestone or regression on its own (an integrity failure is logged as a blocker).
- `record_event(kind, text, ...)` adds a decision, blocker or note to the timeline. Milestones and regressions come automatically.
- `publish_media(stage, title, paths, brain, ...)` shows a video, images or a comparison for a stage, with its key metrics. `featured=true` pins it as the stage's hero. `make_comparison(left, right, layout)` builds real vs sim or attempt 1 vs attempt N as an H.264 mp4 (or a PNG) in side-by-side, overlay or wipe layout.
- `status_summary` returns every stage's state per brain as compact JSON.
- `integrity_audit` scans the call log and run records for the G checks (exam reads, rule violations) and suggests values; a watcher records them.
- `steps:` in a skill's frontmatter places it on the page's loop (stage ids from `validation.yaml`, or `loop` for a skill that serves every step).
- Every record carries `brain`. Claude stand-in and Argon results are never merged into one number.

The page is `runs/status/index.html`: a cover flowchart of the loop, one chapter per step (its text is in `runs/status/story.yaml`), then a ledger of every event and cost. It reloads every 30 s while Live is on. What goes where, and the writing limits for titles, captions and claims, are in `docs/design/ui-protocol.md`. Every write rebuilds it, or rebuild by hand with `.venv/bin/python -m harness.status`. Add `--bundle` to copy the linked media into `runs/status/media/`, so the folder can be zipped or published.
