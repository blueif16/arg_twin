"""Gemini brain: runs the playbook through the registry tools with the Interactions API.

It sees exactly the tools a Claude stand-in sees through harness.cli (registry declarations) and
the same system instruction (harness.brains.prompt). One run is a tool-call loop:

    create(input=task) -> function_call steps -> harness.registry.call(..., caller="argon")
    -> function_result steps (+ attached media) -> create(previous_interaction_id=...) -> ...
    until a turn has no function calls, an error, or a budget runs out.

API settings follow docs/research/interactions-api-notes.md (verified offline against
google-genai 2.33.0):
  - HttpOptions.timeout is in ms and the SDK default is no timeout; per-call timeout= is seconds.
  - tools, system_instruction and generation_config are interaction-scoped: re-sent every turn.
  - thinking_level only; temperature/top_p/top_k/thinking_budget/cached_content are never sent.
  - max_output_tokens > 65,536 needs background=True; background runs are polled while
    status is in_progress or queued.
  - usage.total_cached_tokens is read every turn (implicit caching, no cached_content).
  - Images go back inline inside the function_result (documented for Gemini 3+); videos inline
    when small, else via the Files API.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
from google import genai
from google.genai import types

from harness import registry
from harness.brains import files as _files  # noqa: F401  registers write_run_file (brain deliverables)
from harness.brains.prompt import build_system_instruction
from harness.brains.record import (RecordingTransport, ReplayExhausted, RunRecorder, cost_usd,
                                   request_digest, usage_dict)
from harness.registry import REPO

try:  # the Interactions SDK's error classes (google-genai 2.x)
    from google.genai._gaos.lib.compat_errors import APIConnectionError, APIStatusError, APITimeoutError
except ImportError:  # pragma: no cover - other SDK layouts: never treat anything as transient
    APIConnectionError = APIStatusError = APITimeoutError = ()  # type: ignore[assignment,misc]

ARGON = "gemini-4-argon"
FORBIDDEN = {"temperature", "top_p", "top_k", "thinking_budget", "cached_content"}
SYNC_OUTPUT_CAP = 65_536  # above this the call must run with background=True
IMAGE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
              ".webp": "image/webp", ".gif": "image/gif"}
VIDEO_MIME = {".mp4": "video/mp4", ".mov": "video/mov", ".webm": "video/webm",
              ".avi": "video/avi", ".mpeg": "video/mpeg", ".mpg": "video/mpg"}


class BudgetStop(Exception):
    pass


@dataclass
class Budget:
    max_model_calls: int = 30
    max_tool_calls: int = 150
    max_tokens: int = 3_000_000   # sum of usage.total_tokens; each turn re-counts the whole context
    max_usd: float | None = 10.0  # estimate from record.PRICES (Argon only)
    max_wall_s: float = 1800.0    # also caps each HTTP call's timeout


@dataclass
class BrainConfig:
    model: str = ARGON
    thinking_level: str = "medium"           # low | medium | high (Argon)
    thinking_summaries: str | None = "auto"  # thought summaries into the record; None to omit
    max_output_tokens: int = 32_768          # covers thinking + output; >= 32K on Argon
    background: bool | None = None           # None: on when max_output_tokens > 65,536
    http_timeout_s: float = 1200.0           # 20 min, per the guide for "high" thinking
    poll_interval_s: float = 5.0
    tool_names: list[str] | None = None      # allowlist; None = every registered tool
    tool_choice: str | None = None           # auto | any | none | validated; None = server default
    caller: str = "argon"                    # caller name in runs/calls/
    attach_media: bool = True
    media_resolution: str | None = None      # low | medium | high | ultra_high; None = default
    video_fps: float | None = None           # static sampling fps for attached clips; None = 1 fps default
    video_processing: str | None = None      # static | agentic (model navigates the clip); None = server default (static)
    max_media_per_turn: int = 6
    max_inline_image_bytes: int = 7_000_000
    max_inline_video_bytes: int = 8_000_000
    max_inline_total_bytes: int = 18_000_000  # inline request limit is 20 MB in total
    video_placement: str = "function_result"  # or "user_input": videos in a user step after results
    transient_retries: int = 6               # loop-level retries of a dropped connection / 5xx, on top of the
                                             # SDK's own 3 tries within 30 s; waits 15, 30, 60, 120, 120, 120 s
    max_result_chars: int = 120_000
    extra_generation_config: dict = field(default_factory=dict)
    budget: Budget = field(default_factory=Budget)

    def use_background(self) -> bool:
        return self.background if self.background is not None else self.max_output_tokens > SYNC_OUTPUT_CAP

    def validate(self) -> None:
        bad = FORBIDDEN & set(self.extra_generation_config)
        if bad:
            raise ValueError(f"never send {sorted(bad)} to Gemini 4 models; use thinking_level")
        levels = {"low", "medium", "high"} if self.model == ARGON else {"minimal", "low", "medium", "high"}
        if self.thinking_level not in levels:
            raise ValueError(f"thinking_level {self.thinking_level!r} not in {sorted(levels)} for {self.model}")
        if self.max_output_tokens > SYNC_OUTPUT_CAP and not self.use_background():
            raise ValueError("max_output_tokens > 65536 needs background=True (sync calls return 400)")
        if self.video_placement not in ("function_result", "user_input"):
            raise ValueError("video_placement must be function_result or user_input")
        if self.video_processing not in (None, "static", "agentic"):
            raise ValueError("video_processing must be static or agentic")
        if self.video_processing == "agentic" and self.video_fps:
            raise ValueError("video_fps applies to static processing only")

    def generation_config(self) -> dict:
        gc: dict[str, Any] = {"thinking_level": self.thinking_level,
                              "max_output_tokens": self.max_output_tokens}
        if self.thinking_summaries:
            gc["thinking_summaries"] = self.thinking_summaries
        if self.tool_choice:
            gc["tool_choice"] = self.tool_choice
        gc.update(self.extra_generation_config)
        assert not FORBIDDEN & set(gc)
        return gc

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BrainConfig":
        d = dict(d)
        budget = Budget(**d.pop("budget", {}))
        known = {f for f in cls.__dataclass_fields__}
        return cls(budget=budget, **{k: v for k, v in d.items() if k in known})


def make_client(transport: httpx.BaseTransport, api_key: str | None, http_timeout_s: float) -> genai.Client:
    """SDK client whose HTTP goes through `transport` (live, scripted or replay)."""
    http = httpx.Client(transport=transport, timeout=http_timeout_s)
    # HttpOptions.timeout is milliseconds and overrides the httpx client's own default.
    return genai.Client(api_key=api_key,
                        http_options=types.HttpOptions(timeout=int(http_timeout_s * 1000), httpx_client=http))


def live_tools(caller: str) -> Callable[[str, dict, str], tuple[Any, dict]]:
    def run(name: str, args: dict, call_id: str) -> tuple[Any, dict]:
        return registry.call(name, args, caller=caller), {"source": "live"}
    return run


def files_uploader(client: genai.Client, sleep: Callable = time.sleep, wait_s: float = 300):
    """Upload a video with the Files API and wait until ACTIVE. Cached per (path, mtime, size)."""
    cache: dict[tuple, tuple[str, str]] = {}

    def upload(p: Path) -> tuple[str, str]:
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
        if key not in cache:
            f = client.files.upload(file=str(p), config={"mime_type": VIDEO_MIME[p.suffix.lower()]})
            t0 = time.monotonic()
            while f.state is not None and f.state.name == "PROCESSING":
                if time.monotonic() - t0 > wait_s:
                    raise TimeoutError(f"{p.name} still PROCESSING after {wait_s}s")
                sleep(2)
                f = client.files.get(name=str(f.name))
            if (f.state is not None and f.state.name != "ACTIVE") or not f.uri:
                raise RuntimeError(f"upload of {p.name} ended in state {f.state and f.state.name}")
            cache[key] = (f.uri, f.mime_type or VIDEO_MIME[p.suffix.lower()])
        return cache[key]
    return upload


def find_media(result: Any) -> list[Path]:
    """Image/video files named anywhere in a tool result (repo-relative or absolute, inside the repo)."""
    found: list[Path] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and len(x) < 1024 and Path(x).suffix.lower() in IMAGE_MIME | VIDEO_MIME:
            try:
                p = (Path(x) if x.startswith("/") else REPO / x).resolve()
            except OSError:
                return
            if p.is_file() and p.is_relative_to(REPO) and p not in found:
                found.append(p)
    walk(result)
    return found


def step_summary(s: Any) -> dict:
    t = getattr(s, "type", "?")
    if t == "function_call":
        return {"type": t, "id": s.id, "name": s.name, "arguments": s.arguments}
    if t == "thought":
        return {"type": t, "summary": [c.text for c in (s.summary or []) if getattr(c, "text", None)],
                "signature_chars": len(s.signature or "")}
    if t == "model_output":
        return {"type": t, "content_types": [c.type for c in (s.content or [])],
                "text": "".join(c.text for c in (s.content or []) if c.type == "text" and c.text)}
    return {"type": t}


def output_text(ix: Any) -> str:
    """Text of all model_output steps (long decode adds one per 64K hop; output_text keeps only the last)."""
    return "".join(c.text for s in (ix.steps or []) if s.type == "model_output"
                   for c in (s.content or []) if c.type == "text" and c.text)


class GeminiBrain:
    def __init__(self, config: BrainConfig, client: genai.Client, recorder: RunRecorder,
                 transport: RecordingTransport | None = None, brain_name: str = "argon",
                 system_instruction: str | None = None, tool_declarations: list[dict] | None = None,
                 tool_executor: Callable[[str, dict, str], tuple[Any, dict]] | None = None,
                 uploader: Callable[[Path], tuple[str, str]] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 on_turn: Callable[[dict], None] | None = None, meta_extra: dict | None = None):
        config.validate()
        self.cfg = config
        self.client = client
        self.rec = recorder
        self.transport = transport
        self.brain_name = brain_name
        self.system_instruction = system_instruction
        self.tool_declarations = tool_declarations
        self.tool_executor = tool_executor or live_tools(config.caller)
        self.uploader = uploader
        self.sleep = sleep
        self.on_turn = on_turn
        self.meta_extra = meta_extra or {}
        self.t0 = time.monotonic()
        self.totals = {"model_calls": 0, "tool_calls": 0, "tokens": 0, "usd": 0.0}
        self.retry_log: list[dict] = []

    # ------------------------------------------------------------ budget

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def budget_stop(self, before: str) -> str | None:
        b, t = self.cfg.budget, self.totals
        if self.elapsed() >= b.max_wall_s:
            return "budget:wall_s"
        if before == "tool":
            return "budget:tool_calls" if t["tool_calls"] >= b.max_tool_calls else None
        if t["model_calls"] >= b.max_model_calls:
            return "budget:model_calls"
        if t["tokens"] >= b.max_tokens:
            return "budget:tokens"
        if b.max_usd is not None and t["usd"] >= b.max_usd:
            return "budget:usd"
        return None

    def call_timeout(self) -> float:
        return min(self.cfg.http_timeout_s, max(60.0, self.cfg.budget.max_wall_s - self.elapsed()))

    # ------------------------------------------------------------ transient errors

    def transient(self, e: Exception, kind: str) -> bool:
        """A dropped connection, DNS failure or 5xx is worth retrying. A client timeout is retried only for a
        background create or a poll (quick calls); a sync create that times out was a long generation."""
        if APITimeoutError and isinstance(e, APITimeoutError):
            return kind == "get" or self.cfg.use_background()
        if APIConnectionError and isinstance(e, APIConnectionError):
            return True
        return bool(APIStatusError and isinstance(e, APIStatusError)
                    and getattr(e, "status_code", None) in (500, 502, 503, 504))

    def with_retry(self, kind: str, fn: Callable[[], Any]) -> Any:
        """Run one API call, retrying transient errors with backoff inside the wall-time budget. Failed
        exchanges never reach the record, so a replay serves only the call that succeeded."""
        for attempt in range(self.cfg.transient_retries + 1):
            try:
                return fn()
            except Exception as e:
                wait = min(15.0 * 2 ** attempt, 120.0)
                if (attempt == self.cfg.transient_retries or not self.transient(e, kind)
                        or self.elapsed() + wait >= self.cfg.budget.max_wall_s):
                    raise
                self.retry_log.append({"call": kind, "attempt": attempt + 1, "wait_s": wait,
                                       "error": f"{type(e).__name__}: {e}"[:300]})
                self.sleep(wait)

    # ------------------------------------------------------------ one model call

    def declarations(self) -> list[dict]:
        tools = registry.load_all()
        names = sorted(tools) if self.cfg.tool_names is None else self.cfg.tool_names
        missing = [n for n in names if n not in tools]
        if missing:
            raise ValueError(f"unknown tools in allowlist: {missing}")
        return [tools[n].declaration() for n in names]

    def create(self, inp: Any, prev: str | None, system: str, decls: list[dict]) -> Any:
        kw: dict[str, Any] = {"model": self.cfg.model, "input": inp, "system_instruction": system,
                              "generation_config": self.cfg.generation_config(),
                              "timeout": self.call_timeout()}
        if decls:
            kw["tools"] = decls
        if prev:
            kw["previous_interaction_id"] = prev
        bg = self.cfg.use_background()
        if bg:
            kw["background"] = True
        ix: Any = self.with_retry("create", lambda: self.client.interactions.create(**kw))
        while bg and ix.status in ("in_progress", "queued"):
            if self.elapsed() >= self.cfg.budget.max_wall_s:
                try:
                    self.client.interactions.cancel(id=ix.id)
                finally:
                    raise BudgetStop("budget:wall_s")
            self.sleep(self.cfg.poll_interval_s)
            ix_id = ix.id
            ix = self.with_retry("get", lambda: self.client.interactions.get(id=ix_id, timeout=self.call_timeout()))
        return ix

    # ------------------------------------------------------------ tools and media

    def media_blocks(self, paths: list[Path], state: dict) -> tuple[list[dict], list[dict], list[dict]]:
        """-> (blocks for the function_result, video blocks for a user step, media records)."""
        cfg = self.cfg
        blocks, user_blocks, recs = [], [], []
        for p in paths:
            rel = str(p.relative_to(REPO))
            kind = "image" if p.suffix.lower() in IMAGE_MIME else "video"
            mime = (IMAGE_MIME | VIDEO_MIME)[p.suffix.lower()]
            size = p.stat().st_size
            b64_size = (size + 2) // 3 * 4  # what it costs in the request body
            rec = {"path": rel, "kind": kind, "bytes": size}
            block: dict[str, Any] | None = None
            if state["n"] >= cfg.max_media_per_turn:
                rec.update(mode="skipped", reason="max_media_per_turn")
            elif kind == "image":
                if size <= cfg.max_inline_image_bytes and b64_size <= state["bytes_left"]:
                    block = {"type": "image", "data": base64.b64encode(p.read_bytes()).decode(), "mime_type": mime}
                    rec["mode"] = "inline"
                else:
                    rec.update(mode="skipped", reason="image too large to send inline")
            elif size <= cfg.max_inline_video_bytes and b64_size <= state["bytes_left"]:
                block = {"type": "video", "data": base64.b64encode(p.read_bytes()).decode(), "mime_type": mime}
                rec["mode"] = "inline"
            elif self.uploader is not None:
                try:
                    uri, umime = self.uploader(p)
                    block = {"type": "video", "uri": uri, "mime_type": umime or mime}
                    rec.update(mode="uri", uri=uri)
                except Exception as e:  # upload problems go back to the brain as a note
                    rec.update(mode="skipped", reason=f"upload failed: {type(e).__name__}: {e}")
            else:
                rec.update(mode="skipped", reason="video too large to inline and no uploader")
            if block is not None:
                if cfg.media_resolution:
                    block["resolution"] = cfg.media_resolution
                if kind == "video" and cfg.video_processing == "agentic":
                    block["processing"] = "agentic"
                elif kind == "video" and cfg.video_fps:
                    block["processing"] = {"type": "static", "fps": float(cfg.video_fps)}
                elif kind == "video" and cfg.video_processing == "static":
                    block["processing"] = "static"
                state["n"] += 1
                if rec["mode"] == "inline":
                    state["bytes_left"] -= b64_size
                target = user_blocks if kind == "video" and cfg.video_placement == "user_input" else blocks
                target += [{"type": "text", "text": f"[attached {kind}: {rel}]"}, block]
            else:
                blocks.append({"type": "text", "text": f"[not attached: {rel} ({rec['reason']})]"})
            recs.append(rec)
        return blocks, user_blocks, recs

    def run_tools(self, calls: list[Any], allowed: set[str]) -> tuple[list[dict], list[dict], str | None]:
        """Execute function calls -> (next input steps, tool call records, stop reason)."""
        steps, user_blocks, recs = [], [], []
        media_state = {"n": 0, "bytes_left": self.cfg.max_inline_total_bytes}
        stop = None
        for c in calls:
            args = c.arguments or {}
            if stop is None:
                stop = self.budget_stop("tool")
            if stop:
                recs.append({"call_id": c.id, "name": c.name, "args": args, "skipped": stop})
                continue
            t = time.monotonic()
            meta: dict = {}
            if c.name not in allowed:
                result: Any = {"error": f"unknown tool {c.name!r}; available: {sorted(allowed)}"}
                meta = {"source": "brain"}
            else:
                try:
                    result, meta = self.tool_executor(c.name, args, c.id)
                except Exception as e:  # registry.call already catches; this guards replay executors
                    result, meta = {"error": f"{type(e).__name__}: {e}"}, {"source": "brain"}
            dt = time.monotonic() - t
            self.totals["tool_calls"] += 1
            is_error = isinstance(result, dict) and set(result) == {"error"}
            text = json.dumps(result, default=str)
            if len(text) > self.cfg.max_result_chars:
                text = (text[:self.cfg.max_result_chars]
                        + f" ...[truncated: {len(text)} chars; ask the tool for less]")
            content: list[dict] = [{"type": "text", "text": text}]
            media = meta.get("media", [])
            if self.cfg.attach_media and not is_error:
                blocks, ublocks, media = self.media_blocks(find_media(result), media_state)
                content += blocks
                user_blocks += ublocks
            step = {"type": "function_result", "call_id": c.id, "name": c.name, "result": content}
            if is_error:
                step["is_error"] = True
            steps.append(step)
            recs.append({"call_id": c.id, "name": c.name, "args": args, "result": result,
                         "is_error": is_error, "dt_s": round(dt, 3), "source": meta.get("source"),
                         "media": media})
        if user_blocks:
            steps.append({"type": "user_input", "content": user_blocks})
        return steps, recs, stop

    # ------------------------------------------------------------ the loop

    def run(self, task: str) -> dict:
        cfg = self.cfg
        decls = self.tool_declarations if self.tool_declarations is not None else self.declarations()
        allowed = {d["name"] for d in decls if d.get("type") == "function"}
        system = self.system_instruction or build_system_instruction(
            self.brain_name, self.rec.run_id, budget=asdict(cfg.budget))
        meta = {"brain": self.brain_name, "model": cfg.model, "task": task, "caller": cfg.caller,
                "config": cfg.to_dict(), "sdk": f"google-genai {genai.__version__}",
                "n_tools": len(decls), **self.meta_extra}
        if len(decls) > 20:
            meta["warning"] = f"{len(decls)} tools declared; the docs advise 10-20 active tools"
        self.rec.start(meta, system, decls)
        self.t0 = time.monotonic()
        inp: Any = [{"type": "text", "text": task}]
        prev: str | None = None
        stop: str | None = None
        final_text = ""
        n = 0
        while stop is None:
            stop = self.budget_stop("model")
            if stop:
                break
            n += 1
            turn: dict[str, Any] = {"turn": n, "kind": "model", "t_start": time.time()}
            t = time.monotonic()
            ix = None
            try:
                ix = self.create(inp, prev, system, decls)
            except ReplayExhausted:
                stop = "replay_end"
                break
            except BudgetStop as e:
                stop = str(e)
                turn["error"] = "cancelled: wall-time budget ran out while polling"
            except Exception as e:  # API errors end the run and are recorded
                stop = "api_error"
                turn["error"] = f"{type(e).__name__}: {e}"[:4000]
            turn["latency_s"] = round(time.monotonic() - t, 3)
            if self.retry_log:
                turn["retries"], self.retry_log = self.retry_log, []
            http = self.transport.drain() if self.transport else []
            turn["request"] = request_digest(next((h["request"] for h in http if h["method"] == "POST"
                                                   and h["path"].split("?")[0].endswith("/interactions")), None))
            turn["http"] = http
            self.totals["model_calls"] += 1
            if ix is not None:
                u = usage_dict(ix.usage)
                turn["usage"] = u
                self.totals["tokens"] += u["total"]
                self.totals["usd"] += cost_usd(cfg.model, u) or 0.0
                steps = ix.steps or []
                turn["interaction"] = {"id": ix.id, "status": ix.status,
                                       "steps": [step_summary(s) for s in steps]}
                turn["text"] = output_text(ix)
                turn["thoughts"] = [x for s in turn["interaction"]["steps"] for x in s.get("summary", [])]
                calls = [s for s in steps if s.type == "function_call"]
                if ix.status in ("failed", "cancelled"):
                    stop = f"status:{ix.status}"
                    turn["error"] = str(getattr(ix, "error", None) or "")[:4000] or None
                elif not calls:
                    stop = "done" if ix.status == "completed" else f"status:{ix.status}"
                    final_text = turn["text"]
                else:
                    t_tools = time.monotonic()
                    inp, turn["tool_calls"], stop = self.run_tools(calls, allowed)
                    turn["tools_s"] = round(time.monotonic() - t_tools, 3)
                    prev = ix.id
            else:
                turn["usage"] = usage_dict(None)
            turn["totals"] = {**self.totals, "usd": round(self.totals["usd"], 6),
                              "wall_s": round(self.elapsed(), 2)}
            self.rec.write_turn(turn)
            if self.on_turn:
                self.on_turn(turn)
        if stop == "replay_end" and self.transport is not None:
            self.transport.drain()
        return self.rec.finish(stop or "done", final_text)
