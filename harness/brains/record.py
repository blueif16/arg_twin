"""Run records and replay for brain runs.

Layout of one run, runs/loop/<run-id>/:
    meta.json               brain, model, config, budget, task, sdk version, replay_of
    system_instruction.md   the exact system instruction sent every turn
    tools.json              the exact tool declarations sent every turn
    turns.jsonl             one line per model turn (schema below), appended as the run goes
    summary.json            tokens, calls, wall time, stop reason, final text

turns.jsonl line:
    turn, t_start, latency_s (model call incl. background polling), request (summary of the body
    actually sent), http (every HTTP exchange: method, path, status, dt, request summary, full
    response JSON), interaction (id, status, step summaries), text, thoughts, tool_calls (call_id,
    name, args, result, is_error, dt_s, source, media), usage (input, cached, output, thought,
    tool_use, total), totals (running budget counters), error.

Replay re-serves the recorded HTTP responses through a fake httpx transport, so the real SDK and
the real brain loop run again with no network and no key. Tool results are re-served from the
record by default (no side effects), or re-run live with replay_tools="live".

Nothing secret is written: request headers (the API key) are never recorded, request bodies are
summarized with base64 media replaced by size and hash.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from harness.registry import CALL_LOG_DIR, REPO

LOOP_DIR = REPO / "runs" / "loop"

# USD per 1M tokens, introductory Argon pricing through 2026-11-30 (docs/gemini-4-argon-guide.md,
# interactions-api-notes.md section 3). Thought tokens are billed as output. Estimate only.
PRICES = {"gemini-4-argon": {"input": 2.00, "cached": 0.10, "output": 10.00}}

USAGE_KEYS = {"input": "total_input_tokens", "cached": "total_cached_tokens",
              "output": "total_output_tokens", "thought": "total_thought_tokens",
              "tool_use": "total_tool_use_tokens", "total": "total_tokens"}


def new_run_id(tag: str) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}-{secrets.token_hex(2)}"


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()[:16]


def usage_dict(usage: Any) -> dict:
    """Interaction.usage (SDK object or dict) -> {input, cached, output, thought, tool_use, total}."""
    if usage is None:
        return {k: 0 for k in USAGE_KEYS}
    get = usage.get if isinstance(usage, dict) else (lambda k: getattr(usage, k, None))
    return {k: int(get(v) or 0) for k, v in USAGE_KEYS.items()}


def cost_usd(model: str, u: dict) -> float | None:
    p = PRICES.get(model)
    if not p:
        return None
    fresh = max(u["input"] - u["cached"], 0)
    return round((fresh * p["input"] + u["cached"] * p["cached"]
                  + (u["output"] + u["thought"]) * p["output"]) / 1e6, 6)


# ---------------------------------------------------------------- request summaries

def _strip_media(x: Any) -> Any:
    """Copy of a request body with base64 payloads replaced by '<base64 N bytes sha>'."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k == "data" and isinstance(v, str) and (
                    "mime_type" in x or x.get("type") in ("image", "video", "audio", "document")):
                raw = len(v) * 3 // 4
                out[k] = f"<base64 {raw} bytes sha {sha(v)}>"
            elif k == "signature" and isinstance(v, str):
                out[k] = f"<signature {len(v)} chars>"
            else:
                out[k] = _strip_media(v)
        return out
    if isinstance(x, list):
        return [_strip_media(v) for v in x]
    return x


def summarize_request(method: str, path: str, content: bytes, content_type: str) -> dict:
    if not content:
        return {}
    if "json" not in content_type:
        return {"bytes": len(content), "content_type": content_type}
    try:
        body = json.loads(content)
    except ValueError:
        return {"bytes": len(content), "unparsed": True}
    if not isinstance(body, dict) or not path.split("?")[0].rstrip("/").endswith("/interactions"):
        return {"bytes": len(content), "body": _strip_media(body)}
    s: dict = {"bytes": len(content)}
    for k, v in body.items():
        if k == "system_instruction":
            s[k] = {"chars": len(v), "sha": sha(v)}
        elif k == "tools":
            s[k] = {"n": len(v), "names": [t.get("name") or t.get("type") for t in v],
                    "sha": sha(json.dumps(v, sort_keys=True))}
        elif k == "input":
            s[k] = _strip_media(v) if isinstance(v, list) else {"text": str(v)[:2000]}
        else:
            s[k] = _strip_media(v)
    return s


def request_digest(summary: dict | None) -> dict | None:
    """Compact view of a create request for turns.jsonl (the full summary stays in turn['http'])."""
    if not summary:
        return None
    d = {k: summary[k] for k in ("bytes", "model", "previous_interaction_id", "generation_config",
                                 "background", "store") if k in summary}
    d["system_instruction_sha"] = (summary.get("system_instruction") or {}).get("sha")
    d["tools"] = {k: (summary.get("tools") or {}).get(k) for k in ("n", "sha")}
    inp = summary.get("input")
    if isinstance(inp, list):
        d["input"] = []
        for st in inp:
            blocks = st.get("result") if st.get("type") == "function_result" else st.get("content")
            blocks = blocks if isinstance(blocks, list) else []
            e = {"type": st.get("type", "content")}
            if st.get("type") == "function_result":
                e.update(name=st.get("name"), call_id=st.get("call_id"), is_error=st.get("is_error", False))
            e["media"] = [b.get("type") for b in blocks if b.get("type") not in (None, "text")]
            if st.get("type") == "text":
                e["text"] = (st.get("text") or "")[:300]
            d["input"].append(e)
    else:
        d["input"] = inp
    return d


def _path_of(url: httpx.URL) -> str:
    q = "&".join(f"{k}={v}" for k, v in url.params.multi_items() if k.lower() not in ("key", "api_key"))
    return url.path + (f"?{q}" if q else "")


# ---------------------------------------------------------------- transports

class RecordingTransport(httpx.BaseTransport):
    """Wraps a real (or scripted) transport and keeps every exchange for the run record."""

    def __init__(self, inner: httpx.BaseTransport):
        self.inner = inner
        self.exchanges: list[dict] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        t0 = time.monotonic()
        path = _path_of(request.url)
        req = summarize_request(request.method, path, request.read(),
                                request.headers.get("content-type", ""))
        resp = self.inner.handle_request(request)
        content = resp.read()
        try:
            body: Any = json.loads(content) if content else None
        except ValueError:
            body = {"_raw": content[:4000].decode(errors="replace")}
        keep_headers = {k: v for k, v in resp.headers.items()
                        if k.lower().startswith("x-goog-upload") or k.lower() == "content-type"}
        # The record keeps the response without base64 media and thought signatures: background polls echo the
        # turn's input steps (every attached image again, ~2 MB per poll), and random substrings of those opaque
        # blobs can look like a key to a scanner. Replay never needs them; the SDK parses the stripped copy
        # (checked by replaying stripped copies of real runs: same summary fields, 0 divergences).
        self.exchanges.append({"method": request.method, "path": path, "status": resp.status_code,
                               "dt": round(time.monotonic() - t0, 3), "request": req,
                               "headers": keep_headers, "response": _strip_media(body)})
        # content is already decoded, so drop the encoding/length headers before re-wrapping it
        headers = [(k, v) for k, v in resp.headers.multi_items()
                   if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")]
        return httpx.Response(resp.status_code, headers=headers, content=content)

    def drain(self) -> list[dict]:
        out, self.exchanges = self.exchanges, []
        return out

    def close(self) -> None:
        self.inner.close()


class ReplayExhausted(Exception):
    """The brain asked for more model turns than the record holds."""


class ReplayDivergence(Exception):
    """The replayed brain sent a request the record does not match (method or path)."""


class ReplayTransport(httpx.BaseTransport):
    """Serves recorded /interactions responses in order. Never touches the network."""

    def __init__(self, exchanges: list[dict], pace: float = 0.0, sleep: Callable = time.sleep):
        self.queue = [e for e in exchanges if "/interactions" in e["path"]]
        self.i = 0
        self.pace = pace
        self.sleep = sleep
        self.divergences: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self.i >= len(self.queue):
            raise ReplayExhausted(f"record holds {len(self.queue)} exchanges; brain asked for more")
        rec = self.queue[self.i]
        self.i += 1
        path = _path_of(request.url)
        if (request.method, path.split("?")[0]) != (rec["method"], rec["path"].split("?")[0]):
            raise ReplayDivergence(f"exchange {self.i}: got {request.method} {path}, "
                                   f"recorded {rec['method']} {rec['path']}")
        got = summarize_request(request.method, path, request.read(),
                                request.headers.get("content-type", ""))
        want = rec.get("request") or {}
        for k in ("model", "previous_interaction_id"):
            if got.get(k) != want.get(k):
                self.divergences.append(f"exchange {self.i}: {k} {got.get(k)!r} != recorded {want.get(k)!r}")
        ids = lambda s: [x.get("call_id") for x in (s.get("input") or []) if isinstance(x, dict)
                         and x.get("type") == "function_result"]
        if isinstance(got.get("input"), list) and ids(got) != ids(want):
            self.divergences.append(f"exchange {self.i}: function_result ids {ids(got)} != recorded {ids(want)}")
        if self.pace:
            self.sleep(rec.get("dt", 0) * self.pace)
        content = json.dumps(rec["response"]).encode() if rec["response"] is not None else b""
        return httpx.Response(rec["status"], headers={"content-type": "application/json"},
                              content=content)


# ---------------------------------------------------------------- recorder

class RunRecorder:
    def __init__(self, run_id: str, root: Path = LOOP_DIR, enabled: bool = True):
        self.run_id = run_id
        self.dir = Path(root) / run_id
        self.enabled = enabled
        self.turns: list[dict] = []
        self.meta: dict = {}
        self.t0 = time.time()

    def start(self, meta: dict, system_instruction: str, tool_declarations: list[dict]) -> None:
        self.t0 = time.time()
        self.meta = {"run_id": self.run_id, "started_at": self.t0,
                     "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **meta,
                     "system_instruction_sha": sha(system_instruction),
                     "tools_sha": sha(json.dumps(tool_declarations, sort_keys=True))}
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "system_instruction.md").write_text(system_instruction)
        (self.dir / "tools.json").write_text(json.dumps(tool_declarations, indent=1))
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=1, default=str))
        (self.dir / "turns.jsonl").write_text("")

    def write_turn(self, rec: dict) -> None:
        self.turns.append(rec)
        if self.enabled:
            with open(self.dir / "turns.jsonl", "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    def finish(self, stop_reason: str, final_text: str, extra: dict | None = None) -> dict:
        s = summarize(self.turns, self.meta, wall_s=time.time() - self.t0,
                      stop_reason=stop_reason, final_text=final_text)
        s.update(extra or {})
        if self.enabled:
            (self.dir / "summary.json").write_text(json.dumps(s, indent=1, default=str))
        return s


def summarize(turns: list[dict], meta: dict, wall_s: float | None = None,
              stop_reason: str | None = None, final_text: str | None = None) -> dict:
    tok = {k: sum(t.get("usage", {}).get(k, 0) for t in turns) for k in USAGE_KEYS}
    calls = [c for t in turns for c in t.get("tool_calls", []) if "skipped" not in c]
    by_name: dict[str, int] = {}
    for c in calls:
        by_name[c["name"]] = by_name.get(c["name"], 0) + 1
    model = meta.get("model", "")
    if wall_s is None and turns:
        last = turns[-1]
        wall_s = last["t_start"] + last.get("latency_s", 0) + last.get("tools_s", 0) - turns[0]["t_start"]
    return {
        "run_id": meta.get("run_id"), "brain": meta.get("brain"), "model": model,
        "task": meta.get("task"), "replay_of": meta.get("replay_of"),
        "stop_reason": stop_reason,
        "model_calls": sum(1 for t in turns if t.get("kind", "model") == "model"),
        "tool_calls": len(calls), "tool_errors": sum(1 for c in calls if c.get("is_error")),
        "tool_calls_by_name": by_name,
        "media_attached": sum(1 for c in calls for m in c.get("media", []) if m.get("mode") != "skipped"),
        "tokens": tok,
        "cache_hit_ratio": round(tok["cached"] / tok["input"], 3) if tok["input"] else None,
        "usd_est": cost_usd(model, tok),
        "wall_s": round(wall_s or 0.0, 2),
        "model_s": round(sum(t.get("latency_s", 0) for t in turns), 2),
        "tools_s": round(sum(t.get("tools_s", 0) for t in turns), 2),
        "final_text": final_text,
    }


def load_run(run_id: str, root: Path = LOOP_DIR) -> dict:
    d = Path(root) / run_id
    if not (d / "meta.json").is_file():
        raise FileNotFoundError(f"no run record at {d}")
    turns = [json.loads(ln) for ln in (d / "turns.jsonl").read_text().splitlines() if ln.strip()]
    summ = d / "summary.json"
    return {"dir": d, "meta": json.loads((d / "meta.json").read_text()),
            "system_instruction": (d / "system_instruction.md").read_text(),
            "tools": json.loads((d / "tools.json").read_text()), "turns": turns,
            "summary": json.loads(summ.read_text()) if summ.is_file() else None}


class RecordedTools:
    """Tool executor for replay: returns the recorded result for each call_id."""

    def __init__(self, turns: list[dict]):
        self.by_id = {c["call_id"]: c for t in turns for c in t.get("tool_calls", []) if "skipped" not in c}
        self.divergences: list[str] = []

    def __call__(self, name: str, args: dict, call_id: str) -> tuple[Any, dict]:
        rec = self.by_id.get(call_id)
        if rec is None:
            self.divergences.append(f"no recorded result for call {call_id} ({name})")
            return {"error": f"replay: no recorded result for call_id {call_id}"}, {"source": "recorded"}
        if rec["name"] != name or rec.get("args") != args:
            self.divergences.append(f"call {call_id}: {name}{args} != recorded {rec['name']}{rec.get('args')}")
        return rec["result"], {"source": "recorded", "media": rec.get("media", [])}


# ---------------------------------------------------------------- Claude stand-in records

def import_cli_calls(caller: str, run_id: str, root: Path = LOOP_DIR, task: str | None = None) -> dict:
    """Build a run record from runs/calls/ entries made through harness.cli with --caller <caller>.

    A Claude stand-in calls tools via the CLI, so its tokens live in its own usage report, not here;
    the record holds its tool calls and wall time so its runs line up with Argon's.
    """
    rows = []
    for f in sorted(CALL_LOG_DIR.glob("*.jsonl")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                if r.get("caller") == caller:
                    rows.append(r)
    if not rows:
        raise ValueError(f"no calls with caller {caller!r} in {CALL_LOG_DIR}")
    rec = RunRecorder(run_id, root)
    rec.start({"brain": "claude", "model": "claude (stand-in via harness.cli)", "caller": caller,
               "task": task, "imported_from": str(CALL_LOG_DIR)}, "(set by the Claude orchestrator)", [])
    rec.t0 = rows[0]["t"]
    rec.meta["started_at"] = rec.t0
    (rec.dir / "meta.json").write_text(json.dumps(rec.meta, indent=1, default=str))
    for i, r in enumerate(rows, 1):
        rec.write_turn({"turn": i, "kind": "cli", "t_start": r["t"], "latency_s": 0, "tools_s": r["dt"],
                        "usage": {k: 0 for k in USAGE_KEYS},
                        "tool_calls": [{"call_id": f"cli-{i}", "name": r["tool"], "args": r["args"],
                                        "result": r["result"], "is_error": not r.get("ok", True),
                                        "dt_s": r["dt"], "source": "cli"}]})
    last = rows[-1]
    s = summarize(rec.turns, rec.meta, wall_s=last["t"] + last["dt"] - rows[0]["t"],
                  stop_reason="imported", final_text=None)
    s["tokens"] = None  # unknown here: take them from the Claude agent's own usage report
    (rec.dir / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    return s
