---
name: brains
description: Run a task with a brain (Gemini 4 Argon over the Interactions API, the offline fake, or a Claude stand-in through harness.cli), record every turn under runs/loop/<run-id>/, and replay a run with no network and no key.
stage: infra
steps: [loop]
---
## Goal

A brain runs a task through the playbook and the registry tools, and leaves a record that a watcher can audit and the demo can replay:

- `runs/loop/<run-id>/turns.jsonl` has one line per model turn. Each line holds the request digest, every HTTP exchange with its full response, thought summaries, tool calls (args, result, media, time), tokens (input, cached, output, thought) and latency.
- `runs/loop/<run-id>/summary.json` holds the stop reason, model calls, tool calls by name, tool errors, tokens, cache hit ratio, a cost estimate, wall, model and tool time, and the final text. Next to it are `meta.json` (brain, model, config, budget, task), `system_instruction.md` and `tools.json`: exactly what was sent.
- `--replay <run-id>` runs the real SDK and the real brain loop again on the recorded responses, so the demo never depends on the key.

Every brain gets the same tools and the same system instruction: a short preamble, `orchestrator.md`, `rules.md` and the skill index.

## Inputs

- `playbook/`: the system instruction is rebuilt from it at the start of each run (`harness/brains/prompt.py`).
- Every tool registered under `harness/tools/`, declared to Gemini through `Tool.declaration()`.
- For Argon: `GEMINI_API_KEY` or `GOOGLE_API_KEY` with paid-tier access (Argon is not on the free tier). Never write the key into a file, prompt or log. Records never contain request headers.

## Tools

| Tool | Use |
|---|---|
| `list_skills` | Index: name, stage, description and lesson count for every `skills/*/SKILL.md`. Optional `stage` filter. |
| `read_skill` | Full SKILL.md plus its helper files list; `file` reads one helper file. |
| `read_playbook_file` | `README`, `orchestrator` or `rules`. |
| `append_lesson` | Adds a dated note under a skill's `## Lessons`. Needs at least one existing repo file or URL as `evidence`; rejects strings that look like keys. |
| `log_note` | Keeps a working note or decision point in the record. Use it instead of text right before a tool call. |

Code lives in `harness/brains/`: `gemini.py` (loop, config, media), `record.py` (records, replay transport), `fake.py` (offline server), `prompt.py`, `run.py` (CLI) and `test_offline.py`.

## Approach

### Run it

```sh
# Offline, no key: the real SDK and the real loop against a scripted server (fake.py)
.venv/bin/python -m harness.brains.run --brain fake --task 'Tour the skills'

# Argon. Asks before spending unless --yes; refuses with no key or a non-tty without --yes
.venv/bin/python -m harness.brains.run --brain gemini --task '<task>' --thinking-level medium \
    --max-model-calls 20 --max-usd 5 --max-wall-s 1200 [--skills build-twin | --tools a,b,c] [--yes]

# Replay: no network, no key. Recorded tool results by default; --replay-tools live re-runs them;
# --pace 1 waits the recorded latency, so it looks live in a demo
.venv/bin/python -m harness.brains.run --replay <run-id> [--pace 1] [--replay-tools live] [--record-replay]

.venv/bin/python -m harness.brains.run --summary <run-id>
.venv/bin/python -m harness.brains.test_offline
```

Each turn prints a line with latency, tokens and cached tokens, then the thought summary and each tool call with its result. The summary JSON prints at the end.

### Claude stand-in

1. **Get the prompt.** Run `.venv/bin/python -m harness.brains.run --print-system --brain claude --run-id <run-id>`. It prints the text Argon gets, with a "Calling tools" section for the CLI.
2. **Spawn the subagent.** Give a Claude subagent that prompt and the task. It calls tools with `.venv/bin/python -m harness.cli call <tool> '<json>' --caller claude:<run-id>`. Claude reads PNG grids; it cannot take mp4 clips.
3. **Record the run.** Run `.venv/bin/python -m harness.brains.run --import-calls claude:<run-id> --run-id <run-id> --task '<task>'`. It writes `turns.jsonl` and `summary.json` from `runs/calls/`, covering tool calls, errors and wall time. Tokens show as `null`: take them from the subagent's own usage report and write them into the round log.

### Budgets

Model calls, tokens, cost and wall time are checked before each model call. Tool calls and wall time are checked before each tool call. Defaults: 30 model calls, 150 tool calls, 3M tokens, $10 and 1800 s. A run ends with one of these stop reasons: `done`, `budget:<name>`, `api_error`, `status:<failed|cancelled|incomplete>` or `replay_end`.

- **Tokens** are the sum of `usage.total_tokens`. A chained run re-counts the whole context as input on every turn, so the total grows roughly with the square of the turn count. Cached tokens cost 5%. Use `--max-usd` to cap spend and `--max-tokens` as a context guard.
- **Cost** is an estimate at the introductory Argon price: $2 per 1M input tokens, $0.10 per 1M cached, $10 per 1M output plus thought. This holds through 2026-11-30; input doubles from 1 Dec (`record.PRICES`).
- **Wall time** also caps each HTTP timeout at the remaining time, with a 60 s floor. A background poll that runs out of time cancels the interaction.

### Settings (`BrainConfig`; CLI flags in brackets)

| Setting | Default | Why |
|---|---|---|
| `thinking_level` [`--thinking-level`] | `medium` | `high` for design and multi-file debugging, `low` for routing. Argon accepts low, medium and high only. |
| `max_output_tokens` | 32768 | Covers thinking plus output; use at least 32K on Argon. Above 65,536 it switches to `background=True` and polls. |
| HTTP timeout [`--http-timeout-s`] | 1200 s | The SDK default is no timeout at all. |
| `thinking_summaries` | `auto` | Puts thought summaries in the record as decision points. |
| `tool_names` [`--tools`] | all | The docs advise 10–20 active tools; `meta.json` warns above 20. |
| [`--skills a,b`] | — | Allowlist = tools named in those skills' `## Tools` sections + core tools (`run.skill_tools`). All 60 declarations cost about 68k characters per turn; one skill gives 10–16 tools. Prefer this to `--tools`. |
| `media_resolution`, `video_fps` | server default | Per media block; `video_fps` sets static sampling (the default is 1 fps). |
| `video_placement` | `function_result` | Fallback `user_input` puts clips in a user step after the results. |

### Media

Any string in a tool result that names an existing image (.png, .jpg, .webp, .gif) or video (.mp4, .mov, .webm) inside the repo is attached after the JSON text of that `function_result`, with a line naming the path:

- **Images** go inline: `{"type": "image", "data": <base64>, "mime_type": "image/png"}`. This is the documented shape for Gemini 3+ function results.
- **Videos** go inline when they are 8 MB or less. Larger ones are uploaded with the Files API, polled until ACTIVE, and sent as `{"type": "video", "uri": ...}`.
- **Caps:** 6 files per turn, 7 MB per image, and 18 MB of inline base64 per request (the inline request limit is 20 MB). Anything skipped is named, with the reason, in the result text.

Footage tool outputs on disk are 0.2–2 MB, so they fit inline. Replays don't re-attach media; the record keeps what was sent.

### Known API gotchas

From `docs/research/interactions-api-notes.md`, verified offline against google-genai 2.33.0:

- **Timeout units.** `HttpOptions.timeout` is in ms and `create(timeout=)` is in seconds. Setting `http_options.timeout` overrides the 5 s default of a custom httpx client; that was verified here.
- **Interaction-scoped settings.** `tools`, `system_instruction` and `generation_config` apply to one interaction, so the loop re-sends them every turn. The fake server rejects a turn that forgets them.
- **Banned parameters.** Never send `temperature`, `top_p`, `top_k`, `thinking_budget` or `cached_content`; `BrainConfig.validate` refuses them. The SDK silently drops `thinking_budget` but does send `temperature`.
- **`output_text` is incomplete.** It holds only the last run of text and is empty after tool calls. The loop joins every `model_output` step.
- **Malformed_Function_Call.** Free text right before a function call can trigger it; use `log_note` instead.
- **Thought signatures.** Stateful chaining (`previous_interaction_id`) handles them; never strip steps.
- **Background mode.** It needs `store=True`. Poll while the status is `in_progress` or `queued`. You can't chain to an interaction that is still running.
- **Implicit caching.** It matches on the prompt prefix: the system instruction and tools are byte-identical every turn, and the run-specific part sits at the end. Argon caches nothing under 4,096 tokens, and caches the whole previous request only when a turn adds at least 2,050 tokens. Read `cached` per turn in `turns.jsonl`.
- **Retries.** The SDK retries 408, 409, 429 and 5xx three times, within 30 s.

### Unverified until the key arrives

- Whether a sync turn that stops at tool calls reports `requires_action` or `completed`. The loop continues whenever there are `function_call` steps, so either works.
- Several `function_result` steps in one input (parallel calls). They serialize correctly, but this has not been confirmed live.
- A video inside a `function_result`. The SDK type allows it, but the docs only show images. If the server rejects it, try `--video-placement user_input`, which is also unconfirmed.
- `thinking_summaries: auto` on Argon. The docs don't say; if it is rejected, use `--thinking-summaries none`.
- Real token counts, cache hit ratio and latency. Fake-server numbers are synthetic: about 4 characters per token of the request, plus the prior context.

First steps with the key:

1. Run `scripts/smoke_gemini.py --model gemini-4-argon`.
2. Run `--brain gemini --task 'Tour the skills' --max-model-calls 4 --thinking-level low --yes`.
3. Run `--replay <that run-id>`.
4. Write what you learned under Lessons.

Argon appends lessons with `author: argon`, so diff the skill files before and after a run to see what it learned for itself.

## Checks

- `.venv/bin/python -m harness.brains.test_offline` passes 7 of 7 with networking disabled: `httpx.HTTPTransport` is patched to fail, and no key is set.
- **Per run:**
  - `summary.json` exists and `stop_reason` is not `api_error`.
  - The number of model turns in `turns.jsonl` equals `model_calls`.
  - Every turn has `usage`.
  - `grep -rE "AIza[0-9A-Za-z_-]{30}" runs/loop/<run-id>` finds nothing (a key-shaped string, not the word).
- `--replay <run-id>` reports `matches_original` true for every field and `divergences: []`.
- **Live:**
  - Each turn's `request.generation_config` shows `thinking_level` and no sampling parameters.
  - On chains over 4K tokens, `cached` is above 0 from turn 2 on.

## Lessons

- **2026-10-24** (claude:brains-builder): Offline only, no key. The real google-genai 2.33.0 SDK and GeminiBrain loop ran against the scripted server: 3 model turns, 4 tool calls (2 in parallel in one turn), results chained with previous_interaction_id, and tools, system_instruction and generation_config re-sent every turn. A replay re-served 3 of 3 exchanges; every summary field matched and there were 0 divergences. test_offline passed 7 of 7: budgets, background polling, image and mp4 attached inline, the Files API path through a stub uploader, unknown tool -> is_error, HTTP 400 -> api_error. A real probe_video result (169 KB grid PNG + 150 KB clip) attached inline in a 436 KB request. Token counts are synthetic; nothing here is checked against the live API. Evidence: `runs/loop/offline-fake-tour/summary.json`, `harness/brains/test_offline.py`.

- 2026-10-24: tool parameter schemas are restricted to Gemini's documented JSON-Schema subset in `registry.gemini_schema`: type lists become anyOf, defaults move into the description, other keywords are dropped. Unverified against the live API until the key arrives; check the first real call for 400 INVALID_ARGUMENT.
