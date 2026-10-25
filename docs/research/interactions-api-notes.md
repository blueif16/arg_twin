# Calling Argon, Flash and Flash-Lite through the Interactions API

Verified 2026-10-24 against three sources:

- The hosted docs at ai.google.dev, fetched as `.md.txt` the same day.
- The `google-gemini/cookbook` notebooks.
- The installed SDK, `google-genai` 2.33.0, under `.venv/lib/python3.12/site-packages/google/genai/` (written `genai/` below).

**No API key was available**, so nothing here was run against the live API. "Verified offline" means the request the SDK would send was captured with an `httpx.MockTransport` and canned responses were parsed: real SDK code, no network. Every claim cites a doc page or an SDK file:line. Anything unconfirmed is marked **Uncertain**.

Doc pages cited by short name:
[FC](https://ai.google.dev/gemini-api/docs/function-calling) ·
[IX](https://ai.google.dev/gemini-api/docs/interactions) ·
[VID](https://ai.google.dev/gemini-api/docs/video-understanding) ·
[IXVID](https://ai.google.dev/gemini-api/docs/interactions/video-understanding) ·
[MR](https://ai.google.dev/gemini-api/docs/media-resolution) ·
[TOK](https://ai.google.dev/gemini-api/docs/tokens) ·
[CACHE](https://ai.google.dev/gemini-api/docs/caching) ·
[GCCACHE](https://ai.google.dev/gemini-api/docs/generate-content/caching) ·
[LM](https://ai.google.dev/gemini-api/docs/latest-model) ·
[LD](https://ai.google.dev/gemini-api/docs/long-decode) ·
[BG](https://ai.google.dev/gemini-api/docs/background-execution) ·
[SO](https://ai.google.dev/gemini-api/docs/structured-output) ·
[THINK](https://ai.google.dev/gemini-api/docs/thinking) ·
[FILES](https://ai.google.dev/gemini-api/docs/files) ·
[PRICE](https://ai.google.dev/gemini-api/docs/pricing) ·
[ARGON](https://ai.google.dev/gemini-api/docs/models/gemini-4-argon) ·
[REF](https://ai.google.dev/api/interactions-api) ·
[BC](https://ai.google.dev/gemini-api/docs/interactions-breaking-changes-may-2026) ·
[CB-VID](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Video_understanding.ipynb)

## Before you start

- **Argon needs a paid key.** "Gemini 4 Argon is not available on the Free tier." [PRICE], [LM]
- **The Interactions API is in Beta.** Schemas may still change. [IX]
- **SDK version is fine.** 2.33.0 uses the current `steps` schema; any SDK 2.x does [BC]. Since 8 June 2026 the server ignores the `Api-Revision` header [BC]. In the captured requests, SDK 2.33.0 did not send that header at all. The skill's migration note says it "sets it automatically", which is not what we observed; it is harmless either way.
- **Thinking levels by model** [THINK], [ARGON]:

  | Model | Default | Levels |
  |---|---|---|
  | `gemini-4-argon` | `medium` | `low`, `medium`, `high` |
  | `gemini-3.8-flash` | `medium` | `low`, `medium`, `high` (`minimal` returns an error) |
  | `gemini-3.5-flash-lite` | `minimal` | `minimal`, `low`, `medium`, `high` |

  The skill's `references/migration.md` lists `minimal` for 3.8 Flash. The model page says that errors. Trust the model page.
- **What the SDK does with legacy parameters (verified offline).** `generation_config={"thinking_budget": ...}` is silently dropped and never sent. `temperature` is sent; the server ignores it on Argon [LM]. Unknown top-level kwargs raise `TypeError` client-side (`genai/_gaos/google_genai.py:1473`).

## Client setup and timeouts

The client-level timeout is in **milliseconds**; the per-call timeout is in **seconds**:

- `HttpOptions.timeout` is milliseconds (`genai/types.py:2730`) and is passed straight through as `timeout_ms` (`genai/_gaos/google_genai.py:206`).
- `create(..., timeout=)` is multiplied by 1000 (`genai/_gaos/basesdk.py:72-76`).

Verified offline: `http_options={"timeout": 1_200_000}` gave a 1200 s httpx timeout, and `timeout=900` on one call gave 900 s.

**The default is no timeout at all.** The SDK's own httpx client has `Timeout(None)` (verified offline), so an Argon call with no timeout set can hang forever. If you pass your own `httpx_client`, its timeout applies instead; plain httpx defaults to 5 s.

**Default retries:** 3 attempts on 408, 409, 429, 5XX and connection errors. Backoff runs 0.5 s → 8 s, capped at 30 s total (`genai/_gaos/google_genai.py:140-146`, `genai/_gaos/interactions.py:507`). Change this with `HttpOptions.retry_options` (`genai/types.py:2746`).

```python
# Verified offline against google-genai 2.33.0
from google import genai

client = genai.Client(http_options={"timeout": 20 * 60 * 1000})  # 20 min, in ms
# or for one call: client.interactions.create(..., timeout=1200)  # seconds
```

The docs agree on the length: "set HTTP client timeouts to at least 15 minutes" [LM], and "set HTTP read timeouts to 15 to 20 minutes per stream" [LD].

## 1. Function calling

**Declaring a tool.** Each tool is a plain dict with `"type": "function"`, `name`, `description` and a JSON-Schema `parameters` object [FC]; see also `genai/_gaos/types/interactions/function.py:30`. Pass the list as top-level `tools=`. Built-in tools such as `{"type": "google_search"}` can sit in the same list [FC].

**Tool choice.** `generation_config={"tool_choice": "auto" | "any" | "none" | "validated"}`, or `{"allowed_tools": {"mode": "any", "tools": [names]}}` to restrict which functions may be called [FC]; see also `genai/_gaos/types/interactions/toolchoicetype.py:25` and `allowedtools.py:28`.

**How calls come back.** The response has `steps`. Each call is a step with `type == "function_call"` carrying `id`, `name` and `arguments` (already a dict) [FC]; see also `functioncallstep.py:29`. Thought signatures live only on `thought` steps, never on function calls, and the SDK handles them for you [THINK]. `interaction.output_text` holds only the last contiguous run of text, so it is empty when the turn ended in tool calls (`genai/_gaos/google_genai.py:1380`; verified offline).

**Sending results back, stateful (recommended).** Pass `previous_interaction_id=first.id`. The `input` is a list of `function_result` steps with `call_id` (the call's `id`), `name` and `result`. `result` is a list of content blocks, or a string [FC]; see also `functionresultstep.py:56`. You can also set `is_error: true` (`functionresultstep.py:63`). The `result` may include images and other media for Gemini 3+ models [FC].

**Re-send scoped settings every turn.** `tools`, `system_instruction` and `generation_config` apply to one interaction only, so send them again each turn [IX].

**Parallel calls.** The model can return several `function_call` steps in one interaction [FC]. Send one `function_result` per `call_id`.

**Uncertain:** The docs never show more than one result in a single request. Sending all results in one `input` list serialized correctly offline, but is not confirmed live.

```python
# Request/response shapes verified offline against google-genai 2.33.0 (not run live)
import json

tools = [{
    "type": "function",
    "name": "get_joint_limits",
    "description": "Position limits of one joint, in radians.",
    "parameters": {"type": "object",
                   "properties": {"joint": {"type": "string"}},
                   "required": ["joint"]},
}]

first = client.interactions.create(
    model="gemini-4-argon",
    input="Limits for wrist and elbow?",
    tools=tools,
    generation_config={"thinking_level": "medium"},
)
calls = [s for s in first.steps if s.type == "function_call"]

final = client.interactions.create(
    model="gemini-4-argon",
    previous_interaction_id=first.id,
    tools=tools,                                     # re-send: interaction-scoped
    generation_config={"thinking_level": "medium"},  # re-send: interaction-scoped
    input=[{"type": "function_result", "call_id": c.id, "name": c.name,
            "result": [{"type": "text", "text": json.dumps(my_tool(**c.arguments))}]}
           for c in calls],
)
print(final.output_text)
```

**Stateless alternative.** Set `store=False` and resend the full history every turn: the `user_input` step, every returned step (including `thought` and `function_call`) exactly as received, then the `function_result` steps [FC]. With `store=False` you cannot use `previous_interaction_id` or `background=True` [IX].

**Statuses.** A background interaction waiting on a tool ends in `requires_action` [LD], [BG]. The status enum is in `genai/_gaos/types/interactions/interaction.py:142`.

**Uncertain:** Whether a synchronous call that stops at a tool call reports `requires_action` or `completed`. `scripts/smoke_gemini.py` prints it.

**Chaining.** You cannot chain to an interaction that is still `in_progress`; that returns 400 [BG].

**Limits.** There is no hard cap on the number of functions in the docs. The guidance is "Keep active set to 10-20 tools maximum" [FC]. Only a subset of OpenAPI schema is supported, and `any` mode may reject very large or deeply nested schemas [FC]. In the SDK, `minLength`/`maxLength`-style keywords in `parameters` are ignored (`function.py:37-43`). Text emitted right before a tool call can cause `Malformed_Function_Call`. The fix is to put working notes in a dedicated `update()` function call instead [FC].

**Retention.** Stored interactions are kept 55 days on the paid tier and 1 day on the free tier [IX].

## 2. Video input

**Getting the video in** [VID], [IXVID]:

- **Files API.** `client.files.upload(file=path)`, then poll `client.files.get(name=...)` until the state is `ACTIVE`. Pass `{"type": "video", "uri": f.uri, "mime_type": f.mime_type}`. This is recommended for anything over about 100 MB, for videos over 10 min, and for videos you reuse. Files are deleted after 48 h [FILES].
- **Inline.** `{"type": "video", "data": <base64>, "mime_type": "video/mp4"}`.
- **YouTube.** A public YouTube URL as `uri`.

**Uncertain, size limits:**
- [VID] gives the Files API 20 GB paid / 2 GB free. [FILES] says 2 GB per file and 20 GB per project.
- [VID]'s table puts inline at "< 100MB", but its text says to use the Files API above 20 MB total request size.
- Plan for at most 2 GB per file and Files API for anything above a few MB.

**Processing mode.** Set it per video item with `"processing"` [VID], [REF]; see also `genai/_gaos/types/interactions/videocontent.py:55,85`.

- `"static"` is the default. It samples frames at 1 FPS and puts them all in context. It is the only mode that accepts clipping and a custom fps.
- `"agentic"` lets the model move through the timeline itself, loading transcript, frames or audio on demand. It uses "up to 88%" fewer tokens on long-form content but may add latency on clips under 5 min [VID].
- Agentic mode adds `processing_call` and `processing_result` steps. They need no response from you, but in stateless mode you must resend them [VID].

**Uncertain, agentic mode on Argon.** [ARGON] lists agentic video understanding as "Supported (defaults to `static`)", and [LM] calls it generally available. The [VID] table lists only 3.8 Flash, 3.6 Flash and 3.5 Flash-Lite. Confirm on Argon with one live call before relying on it.

**Static options**, an object with `type: "static"` (`genai/_gaos/types/interactions/staticmediaprocessing.py:30`):

- **`start_offset` / `end_offset` are strings**, such as `"10.5s"` [REF]. The docs' Python clipping example passes the integers `1200` / `1500` [VID]. **SDK 2.33.0 rejects integers client-side**; this was verified offline with a pydantic `string_type` error. Use `"1200s"`.
- **`fps` is a float.** For long, mostly static videos the docs suggest fps below 1; for fast motion they suggest a higher fps ([VID], generateContent page).

**Uncertain, fps range.** No minimum or maximum fps is documented on any page fetched.

**Uncertain, overview page.** The Interactions overview page still says "video metadata (clipping, custom frame rate) not yet available" [IX]. The newer [VID] page, the API reference [REF] and SDK 2.33.0 all have it. That overview line looks stale.

**Resolution and tokens.** Per-item `"resolution": "low" | "medium" | "high" | "ultra_high"` (`genai/_gaos/types/interactions/mediaresolution.py:25`). It works alongside `processing` [MR].

**Uncertain, tokens per frame.** The docs disagree:

| Source | Tokens per frame | Audio | Per second of video |
|---|---|---|---|
| [VID] | 66 (low) / 258 (other) | 32 tok/s | ~100 (low) / ~300 (high) |
| [MR], Gemini 3 table | 70 (`low`, `medium` and default all equal) / 280 (`high`); `ultra_high` not available for video | 25 tok/s | not stated |
| [TOK] | not stated | not stated | 263 (static) |

Treat it as about 100 tok/s at low and about 300 tok/s at high, and measure with `usage`.

**Agentic token accounting.** Navigation reasoning counts as `total_thought_tokens`. Frames, audio and transcript loaded on demand count as `total_tool_use_tokens` [VID], [TOK].

**Several videos in one request.** This works, and each video can use a different mode [VID], [CB-VID].

**Uncertain, how many videos.** The only stated cap is "maximum of 10 videos per request" for Gemini 2.5+, and it sits in the YouTube section [VID]. The docs' best practice is "one video per prompt" for optimal results, with the text prompt after the video [VID], [IXVID].

**Audio track.** The model uses both the audio and the visual streams. In static mode audio is processed at 1 Kbps mono, and timestamps are added every second [VID]. In agentic mode the model may load audio or the transcript on demand [VID].

**Uncertain, turning audio off.** No documented switch on the Interactions API. If we don't want audio, strip it with ffmpeg before uploading.

```python
# Serialization verified offline against google-genai 2.33.0 (not run live)
import time

f = client.files.upload(file="episode_cam1.mp4")
while not f.state or f.state.name != "ACTIVE":   # pattern from [IXVID]
    time.sleep(5)
    f = client.files.get(name=f.name)

interaction = client.interactions.create(
    model="gemini-4-argon",
    input=[
        {"type": "video", "uri": f.uri, "mime_type": f.mime_type,
         "resolution": "low",
         "processing": {"type": "static", "fps": 5,
                        "start_offset": "2s", "end_offset": "10.5s"}},  # strings, not ints
        {"type": "text", "text": "List each manipulation sub-task with MM:SS timestamps."},
    ],
)
```

Refer to moments inside a video as `MM:SS` [VID].

**SDK behavior.** A plain list of content blocks is wrapped into one `user_input` step before it is sent (`genai/_gaos/google_genai.py:1473-1487`; verified offline).

## 3. Implicit caching

- **Explicit caching is not supported on the Interactions API or on Argon.** No `cached_content` and no `caches.create`. Implicit caching is on by default [CACHE], [LM].
- **Minimum prompt size for a cache hit:** 4,096 tokens for Argon and 3.8 Flash [CACHE].

  **Uncertain:** The minimum for 3.5 Flash-Lite is not listed on any caching page fetched.
- **How matching works.** "Cached content is a prefix to the prompt" [GCCACHE]. The docs' only guidance is to put large, common content at the beginning and to send requests with a similar prefix close together in time [CACHE]. It works in both stateful (`previous_interaction_id`) and stateless mode [CACHE]. `previous_interaction_id` makes hits easier [IX].

  **Uncertain:** No cache lifetime (TTL) is documented.
- **Argon agent loops.** "Gemini caches the full previous request only if the new request adds at least 2,050 tokens". Model output, thoughts and function results all count toward that. Below it, the last 2,050 tokens of the previous request are re-processed at the full input rate [CACHE].
- **Price.** Cached Argon input costs $0.10 per 1M tokens instead of $2.00, a 95% discount, through 30 Nov 2026. From 1 Dec 2026 it is $0.20 instead of $4.00 [PRICE], [LM].
- **Reading it.** Use `interaction.usage.total_cached_tokens`, plus `cached_tokens_by_modality` for a breakdown (`genai/_gaos/types/interactions/usage.py:61,76`; [CACHE]). Other fields: `total_input_tokens`, `total_output_tokens`, `total_thought_tokens`, `total_tool_use_tokens`, `total_tokens` (`usage.py:76-93`).

  **Uncertain:** The Interactions-specific caching page ([ai.google.dev/gemini-api/docs/interactions/caching](https://ai.google.dev/gemini-api/docs/interactions/caching)) still points to `usage_metadata`. That is the generateContent field name; the `Interaction` object has `usage`.

```python
# Field names verified in SDK 2.33.0 (genai/_gaos/types/interactions/usage.py)
u = interaction.usage
print(u.total_input_tokens, u.total_cached_tokens, u.total_output_tokens,
      u.total_thought_tokens, u.total_tool_use_tokens)
```

## 4. Background mode, polling and long output

**Starting one.** `background=True` returns an interaction id immediately; poll with `client.interactions.get(id=...)` [BG].

- It needs `store=True`, which is the default; `store=False` returns 400 [BG].
- It runs for at most 24 h [LD].
- Poll **while the status is in `("in_progress", "queued")`** [LM], [LD]. The polling snippet on [BG] checks only `in_progress`, which would exit early on `queued`.

**Terminal states** [LD]:

| Status | Meaning |
|---|---|
| `completed` | Finished |
| `requires_action` | A tool call is waiting |
| `incomplete` | Hit `max_output_tokens`; the text may be empty if the model was still thinking |
| `failed` | Server error, or ran past 24 h |
| `cancelled` | Stopped by `client.interactions.cancel(id=...)` |

**Output length.**

- Without background, output is capped at 65,536 tokens, and `max_output_tokens` above that returns 400 [LD], [BG].
- With `background=True` and `max_output_tokens` above 65,536, Argon generates up to 1,048,576 tokens minus the input, in 64K "hops" [LD].
- `max_output_tokens` covers thinking **and** output. Use at least 32,768 on Argon [LM]. The docs suggest 131,072 or 262,144 for hard reasoning [LM].

**Reading long output.** Each hop adds a `model_output` step. Join the text of **all** `model_output` steps; `output_text` returns only the last contiguous run [LD], [LM]; see also `genai/_gaos/google_genai.py:1380`.

**Token counts.** In long decode, `total_input_tokens` adds up every hop's input. Do not use it to estimate context size [LD].

```python
# Pattern from [LD]/[LM]; request shape verified offline (not run live)
import time

interaction = client.interactions.create(
    model="gemini-4-argon",
    input=prompt,
    background=True,
    generation_config={"thinking_level": "high", "max_output_tokens": 131072},
)
while interaction.status in ("in_progress", "queued"):
    time.sleep(5)
    interaction = client.interactions.get(id=interaction.id)

text = "".join(c.text for s in interaction.steps if s.type == "model_output"
               for c in (s.content or []) if c.type == "text")
```

**Streaming instead of polling.** Pass `stream=True`, and reconnect after a drop with `client.interactions.get(id=..., stream=True, last_event_id=...)` [BG], [LD].

## 5. Structured output

`response_format` is a **top-level** argument, not part of `generation_config`. Pass `{"type": "text", "mime_type": "application/json", "schema": {...}}`, then parse `interaction.output_text` [SO]; see also `genai/_gaos/types/interactions/textresponseformat.py:40,65`. A dict with the key `"schema"` serializes correctly (verified offline). `response_mime_type` is marked deprecated in the SDK (`createmodelinteraction.py:170`). Pydantic's `Model.model_json_schema()` works as the schema [SO].

**Supported JSON-Schema subset** [SO]:

- Types: `string`, `number`, `integer`, `boolean`, `object`, `array`, `null`.
- Keywords: `enum`, `format`, `minimum`/`maximum`, `items`, `prefixItems`, `minItems`/`maxItems` (both enforced), `required`, `additionalProperties`, `anyOf`.
- Silently dropped: `minLength`/`maxLength`, `minProperties`/`maxProperties`, `uniqueItems`, `contains`. Validate these yourself.
- Very large or deeply nested schemas, big `maxItems`, or large enums can return 400 `INVALID_ARGUMENT` [SO].

**With tools.** Structured output combines with function calling and built-in tools on Gemini 3+ models, as a preview feature [SO]. The docs' example uses `gemini-4-argon` [SO].

```python
# Serialization verified offline against google-genai 2.33.0 (not run live)
from pydantic import BaseModel

class SubTasks(BaseModel):
    steps: list[str]

interaction = client.interactions.create(
    model="gemini-3.5-flash-lite",
    input="Split 'pick peg, insert peg' into steps.",
    response_format={"type": "text", "mime_type": "application/json",
                     "schema": SubTasks.model_json_schema()},
)
result = SubTasks.model_validate_json(interaction.output_text)
```

## 6. Model listing

**Not done: no API key was set in this environment.**

- **How:** call `client.models.list()` (`genai/models.py:6864`). It returns `name`, `input_token_limit`, `output_token_limit` and `supported_actions`. Running `scripts/smoke_gemini.py --list-models` prints the same fields.
- **Expected model ids:** the SDK's model enum already includes `gemini-4-argon`, `gemini-3.8-flash`, `gemini-3.5-flash-lite` and `gemini-robotics-er-2-preview` (`genai/_gaos/types/interactions/model.py`).
- **Token limits** [ARGON] and the model pages:

  | Model | Window or input limit | Output limit |
  |---|---|---|
  | Argon | 1,048,576 shared by input and output | 65,536 synchronous; up to 1M with long decode |
  | 3.8 Flash | 1,048,576 input | 65,536 |
  | 3.5 Flash-Lite | 1,048,576 input | 65,536 |

## Smoke test

`scripts/smoke_gemini.py` makes:

- one plain call to `gemini-3.5-flash-lite`;
- one function-calling round trip, forced with `tool_choice: "any"` on the first turn, then `function_result` sent back via `previous_interaction_id`.

It prints each response's id, status, step types, function-call arguments, `output_text` and `usage`. It never prints the key.

**Status: written and compile-checked only.** It was not run because no `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set. To run it:

```sh
.venv/bin/python scripts/smoke_gemini.py --list-models
```

When it is run, paste the printed shapes into this section. They settle the open questions on `requires_action` vs `completed` and on Flash-Lite cached-token behavior.

## Gemini API skill

Installed `gemini-api-dev` from `google-gemini/gemini-skills`, commit `ee9da52195752ca83c2fca701d96bcb5a9c08d2d` (13 Oct 2026, Apache-2.0), into `.claude/skills/gemini-api-dev/`. It holds `SKILL.md` and `references/migration.md`, fetched as raw files with no npx.

Two places where it disagrees with what was verified above:

- It says the SDK sets `Api-Revision` automatically. The captured requests had no such header, and the server ignores it now anyway [BC].
- It lists `minimal` thinking for 3.8 Flash, which the model page says errors.
