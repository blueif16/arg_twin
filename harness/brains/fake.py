"""Scripted stand-in for the Interactions API endpoint, so the real GeminiBrain + SDK run offline.

`--brain fake` runs the Gemini brain unchanged; only the HTTP transport is swapped for this server.
It answers POST /interactions, GET /interactions/{id} (background polling) and POST .../cancel with
JSON in the real response shape (steps, status, usage), and it is stricter than the real server
about our side of the contract, returning 400 when the brain:
  - sends temperature/top_p/top_k/thinking_budget/cached_content,
  - chains to an unknown or still-running interaction,
  - leaves a function_call without its function_result (or answers one that was not asked),
  - forgets to re-send tools or system_instruction on a later turn (they are interaction-scoped),
  - asks for > 65,536 output tokens without background, or background with store=false,
  - sends a malformed media block.

A script is `script(turn, ctx) -> list of steps`; ctx carries the task, the parsed function
results of this turn (ctx["results"]) and everything earlier (ctx["by_name"]), and is kept per
interaction chain. Use the helpers thought(), call() and say() to build steps.
"""

from __future__ import annotations

import base64
import copy
import json
import re
from typing import Any, Callable

import httpx

FORBIDDEN = {"temperature", "top_p", "top_k", "thinking_budget", "cached_content"}
Script = Callable[[int, dict], list[dict]]


def thought(text: str) -> dict:
    return {"type": "thought", "signature": base64.b64encode(b"fake-signature").decode(),
            "summary": [{"type": "text", "text": text}]}


def call(name: str, args: dict | None = None) -> dict:
    return {"type": "function_call", "name": name, "arguments": args or {}}


def say(text: str) -> dict:
    return {"type": "model_output", "content": [{"type": "text", "text": text}]}


def skills_tour(turn: int, ctx: dict) -> list[dict]:
    """Default script: list skills, load one skill and the rules in parallel, then report."""
    if turn == 1:
        return [thought("Start from the skill index, as the preamble says."),
                call("log_note", {"text": f"Plan: list skills, load one, read rules. Task: {ctx['task'][:200]}",
                                  "kind": "plan"}),
                call("list_skills", {})]
    if turn == 2:
        idx = ctx["by_name"].get("list_skills", {}).get("data") or {}
        names = [s["name"] for s in idx.get("skills", [])]
        ctx["pick"] = "brains" if "brains" in names else (names[0] if names else "brains")
        return [thought(f"Load skill {ctx['pick']} and rules.md in one turn."),
                call("read_skill", {"name": ctx["pick"]}),
                call("read_playbook_file", {"file": "rules"})]
    skill = ctx["by_name"].get("read_skill", {})
    rules = ctx["by_name"].get("read_playbook_file", {}).get("data") or {}
    sd = skill.get("data") or {}
    if skill.get("is_error"):
        got = f"read_skill failed: {sd.get('error')}"
    else:
        got = (f"read skill '{ctx['pick']}' ({len(sd.get('text', ''))} chars, "
               f"{len(sd.get('helpers', []))} helper files, {sd.get('lessons', 0)} lessons)")
    n = len((ctx["by_name"].get("list_skills", {}).get("data") or {}).get("skills", []))
    return [say(f"Fake brain finished. Task: {ctx['task']!r}. {n} skills in the index; {got}; "
                f"rules.md is {len(rules.get('text', ''))} chars. No live model was called.")]


class FakeInteractionsServer:
    def __init__(self, script: Script = skills_tour, model: str | None = None, background_polls: int = 1):
        self.script = script
        self.model = model
        self.background_polls = background_polls
        self.store: dict[str, dict] = {}       # id -> full interaction JSON
        self.ctx: dict[str, dict] = {}         # id -> script context after that turn
        self.polls_left: dict[str, int] = {}
        self.requests: list[dict] = []         # every create body, for tests
        self.timeouts: list[Any] = []          # httpx timeout extension of every request
        self.n_ix = 0
        self.n_call = 0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    # ------------------------------------------------------------ HTTP

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.timeouts.append(request.extensions.get("timeout"))
        path = request.url.path
        try:
            if request.method == "POST" and path.endswith("/interactions"):
                return httpx.Response(200, json=self.create(json.loads(request.content)))
            m = re.match(r".*/interactions/([^/]+)(/cancel)?$", path)
            if m and m.group(1) in self.store:
                ix_id = m.group(1)
                if m.group(2) and request.method == "POST":
                    self.store[ix_id]["status"] = "cancelled"
                    self.polls_left[ix_id] = 0
                    return httpx.Response(200, json={"id": ix_id, "status": "cancelled", "steps": []})
                if request.method == "GET":
                    if self.polls_left.get(ix_id, 0) > 0:
                        self.polls_left[ix_id] -= 1
                        return httpx.Response(200, json={"id": ix_id, "status": "in_progress",
                                                         "model": self.store[ix_id]["model"]})
                    return httpx.Response(200, json=self.store[ix_id])
            return self.error(404, f"fake server: no route for {request.method} {path}")
        except ValueError as e:
            return self.error(400, str(e))

    @staticmethod
    def error(code: int, msg: str) -> httpx.Response:
        return httpx.Response(code, json={"error": {"code": code, "message": msg,
                                                    "status": "INVALID_ARGUMENT" if code == 400 else "NOT_FOUND"}})

    # ------------------------------------------------------------ create

    def create(self, body: dict) -> dict:
        self.requests.append(body)
        gc = body.get("generation_config") or {}
        bad = FORBIDDEN & (set(gc) | set(body))
        if bad:
            raise ValueError(f"forbidden parameters sent: {sorted(bad)}")
        if not body.get("model"):
            raise ValueError("model is required")
        if (gc.get("max_output_tokens") or 0) > 65536 and not body.get("background"):
            raise ValueError("max_output_tokens > 65536 requires background=true")
        if body.get("background") and body.get("store") is False:
            raise ValueError("background requires store=true")

        prev = body.get("previous_interaction_id")
        inp = body.get("input")
        steps_in = inp if isinstance(inp, list) else [{"type": "user_input", "content": [{"type": "text", "text": str(inp)}]}]
        if prev:
            if prev not in self.store:
                raise ValueError(f"unknown previous_interaction_id {prev}")
            p = self.store[prev]
            if p["status"] in ("in_progress", "queued") or self.polls_left.get(prev, 0) > 0:
                raise ValueError("cannot chain to an interaction that is still in progress")
            ctx = copy.deepcopy(self.ctx[prev])
            if ctx["had_tools"] and not body.get("tools"):
                raise ValueError("tools missing: tools are interaction-scoped, re-send them every turn")
            if ctx["had_system"] and not body.get("system_instruction"):
                raise ValueError("system_instruction missing: it is interaction-scoped, re-send it every turn")
            asked = {s["id"] for s in p["steps"] if s["type"] == "function_call"}
            answered = [str(s.get("call_id")) for s in steps_in if s.get("type") == "function_result"]
            if sorted(answered) != sorted(asked):
                raise ValueError(f"function_result call_ids {answered} do not match function_call ids {sorted(asked)}")
        else:
            text = " ".join(c.get("text", "") for s in steps_in for c in (s.get("content") or [])
                            if s.get("type") == "user_input" and c.get("type") == "text")
            ctx = {"task": text, "turn": 0, "ctx_tokens": 0, "by_name": {},
                   "had_tools": bool(body.get("tools")), "had_system": bool(body.get("system_instruction"))}

        ctx["results"] = []
        for s in steps_in:
            if s.get("type") == "function_result":
                ctx["results"].append(self.parse_result(s))
            elif s.get("type") == "user_input" and prev:
                ctx.setdefault("user_media", []).extend(self.check_blocks(s.get("content") or []))
        for r in ctx["results"]:
            ctx["by_name"][r["name"]] = r
        ctx["turn"] += 1

        out = []
        for st in self.script(ctx["turn"], ctx):
            st = dict(st)
            if st["type"] == "function_call" and "id" not in st:
                self.n_call += 1
                st["id"] = f"call_{self.n_call}"
            out.append(st)

        new_tokens = len(json.dumps(body)) // 4
        inp_tok = ctx["ctx_tokens"] + new_tokens
        out_tok = sum(len(json.dumps(s)) for s in out if s["type"] != "thought") // 4
        th_tok = 40 * sum(1 for s in out if s["type"] == "thought")
        usage = {"total_input_tokens": inp_tok, "total_cached_tokens": ctx["ctx_tokens"],
                 "total_output_tokens": out_tok, "total_thought_tokens": th_tok,
                 "total_tokens": inp_tok + out_tok + th_tok}
        ctx["ctx_tokens"] = usage["total_tokens"]

        self.n_ix += 1
        ix_id = f"fake_ix_{self.n_ix}"
        status = "requires_action" if any(s["type"] == "function_call" for s in out) else "completed"
        ix = {"id": ix_id, "status": status, "model": self.model or body["model"], "steps": out, "usage": usage}
        self.store[ix_id] = ix
        self.ctx[ix_id] = ctx
        if body.get("background"):
            self.polls_left[ix_id] = self.background_polls
            return {"id": ix_id, "status": "in_progress", "model": ix["model"]}
        return ix

    def check_blocks(self, blocks: list[dict]) -> list[dict]:
        media = []
        for b in blocks:
            t = b.get("type")
            if t not in ("text", "image", "video", "audio", "document"):
                raise ValueError(f"unknown content block type {t!r}")
            if t != "text":
                if not (b.get("data") or b.get("uri")) or not b.get("mime_type"):
                    raise ValueError(f"{t} block needs data or uri, and mime_type")
                if b.get("data"):
                    base64.b64decode(b["data"], validate=True)
                media.append(b)
        return media

    def parse_result(self, s: dict) -> dict:
        res = s.get("result")
        blocks = [{"type": "text", "text": res}] if isinstance(res, str) else (res or [])
        if not isinstance(blocks, list):
            raise ValueError("function_result.result must be a string or a list of content blocks")
        media = self.check_blocks(blocks)
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        first = next((b.get("text", "") for b in blocks if b.get("type") == "text"), "")
        try:
            data = json.loads(first)
        except ValueError:
            data = None
        return {"name": s.get("name"), "call_id": s.get("call_id"), "is_error": bool(s.get("is_error")),
                "text": text, "data": data, "media": media}
