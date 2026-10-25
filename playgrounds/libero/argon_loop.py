"""Argon as a closed-loop LIBERO policy: the published protocol [llm-libero-2026] over the Interactions API.

Runs under the repo's .venv (google-genai); the simulator runs in a child process under .venv-libero
(client.SimClient -> server.py). One process = one episode.

What is reproduced, from the paper (data/refs/llm-libero-2026.html, sections 2-3) and its code (IRVLUTD/LLM_LIBERO
at upstream inspect-robots 4d35fe4b, read 2026-10-25, never run):
  - system prompt = upstream _SYSTEM_TEMPLATE (name 'libero_10_taskNN', budget 80) + "Embodiment notes:" + BASE_DOCS,
    verbatim; the goal as the first user message ("Goal: ..."), then one user message per observation:
    "Current observation." / "Instruction: ..." / state[eef_state] labelled / state[gripper_qpos] / approver line,
    then "camera 'agentview' (step N):" + PNG and "camera 'wrist' (step N):" + PNG
  - tools move_to(targets, note), done(summary, hindsight), give_up(reason, hindsight), descriptions verbatim
  - stateless calls: every call resends the system prompt, goal and full history; camera frames are kept for the two
    most recent observations only, older ones become "[2 camera frame(s) elided]"; the model's own reasoning is
    resent as signatures only (the chat wire resent thought signatures, not thought text)
  - one tool call per turn: the first executes, the rest get "ignored: one tool call per turn"; a turn with no tool
    call gets "Respond with exactly one tool call."; 3 failing turns in a row end the episode
  - 80 LLM calls (then give_up on the model's behalf = limit reached), 900 control steps, success checked after
    every step by LIBERO's predicate (server side), done/give_up end the episode as failures
  - reasoning effort high -> thinking_level "high"; output cap per call 16,000 tokens incl. thinking (the paper's
    Argon-high data peaks at exactly 16,000; its site says 8,192 for high, which its own data exceeds)
Deviations (stated in the report): Interactions API (stateless, store=False) instead of the OpenAI-compatible chat
endpoint; a $30 per-episode cap on estimated real cost (the paper's cap value is unpublished; its Argon-high
episodes peaked at $25); thought summaries requested for the record (not resent).

    GEMINI_API_KEY=... .venv/bin/python playgrounds/libero/argon_loop.py --run-id feas-argon-high-1 \
        --split feasibility --task 2 --init 35
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

from harness.brains.record import RecordingTransport, RunRecorder, cost_usd, request_digest, usage_dict  # noqa: E402
from playgrounds.libero.client import SimClient  # noqa: E402

ARGON = "gemini-4-argon"
FORBIDDEN = {"temperature", "top_p", "top_k", "thinking_budget", "cached_content"}
MAX_CONSECUTIVE_FAILURES = 3
RETRY_BACKOFF_S = (15, 30, 60, 120)     # paper budget.py _SERVER_ERROR_BACKOFF_S
TRANSIENT = {408, 409, 429, 500, 502, 503, 504, 520, 529}
PRICE = {"input": 2.00, "cached": 0.10, "output": 10.00}

# ---------------------------------------------------------------- verbatim protocol text
SYSTEM_TEMPLATE = (
    "You are controlling a real robot embodiment named {name!r} through tool calls. Each observation message gives "
    "you the current proprioceptive state and camera images. Work toward the user's goal in small, deliberate "
    "motions; re-check the observation after every motion. Every move tool call must include a `note`: in one or "
    "two sentences, say what you observe in the current observation and why you chose this motion. The user is "
    "watching these notes to see what you see and what you decide, so write them for a human reader. Safety "
    "approvers clamp out-of-bounds and too-fast actions below you. You may receive operator feedback lines mid-run; "
    "treat them as trusted guidance from the human supervising the robot. Respond with exactly one tool call per "
    "turn. When the goal is achieved call done; if it cannot be achieved call give_up. Note what you are learning "
    "about this rig and task as you go: done and give_up will ask what you wish you had known from the start. You "
    "have a budget of {budget} LLM calls for the whole trial.")

BASE_DOCS = (
    "Single Panda arm. move_to uses world-frame x,y,z in meters and rot6d "
    "orientation (first two rotation-matrix columns). z points up. Gripper 0=open, 1=close. "
    "The gripper is open/close only: any value above 0.5 keeps closing and any value below "
    "0.5 keeps opening, so intermediate values do not give a partial opening. The gripper "
    "entry of eef_state is the current gripper command, not the finger position: it reads "
    "1 while closed on an object even though the fingers are stopped apart. Measured finger "
    "positions are in gripper_qpos (meters, about +/-0.04 each when fully open, near 0 when "
    "closed on nothing); fingers stopped clearly apart after closing indicate something is "
    "between them. Omitting gripper from a move_to call keeps the current command. Keep "
    "orientation unchanged unless necessary. Images are upright agentview and wrist camera "
    "views. Only robot proprioception and RGB are available; no object coordinates. "
    "Intermediate target arrival must be checked from the next observation. Verify the "
    "object actually followed the lift in the next observation before assuming the grasp "
    "is secure.\n\n"
    "General manipulation guidance:\n"
    "1. Avoid collisions: before a lateral move that could intersect nearby objects or "
    "furniture, lift to a visibly safe clearance. Do not sweep through or push objects "
    "unless the task requires it. Use large moves in free space and small moves only in "
    "the last few centimeters near an object.\n"
    "2. Identify the target object carefully; similar-looking distractors may be present. "
    "Cross-check both camera views.\n"
    "3. A closed gripper does not mean a successful grasp. After closing, lift 3-5 cm and "
    "confirm in both cameras that the object rose with the gripper.\n"
    "4. Handle objects the way a person would: grasp a stable, graspable part without "
    "pushing or tipping the object, keep it in the same orientation (usually upright) "
    "from grasp to placement, and lower it close to the support surface before "
    "releasing.\n"
    "5. If the same approach fails twice, change it (grasp point, approach direction, or "
    "a small adjustment); do not repeat the same motion.\n"
    "6. After each move, confirm from the next observation that it succeeded before "
    "starting the next step; do not assume success.\n"
    "7. Prefer fewer, purposeful moves over many tiny corrections; your call budget is "
    "limited.\n"
    "8. Declare success only when the latest observation clearly shows that the requested "
    "final state has been achieved.")

HINDSIGHT = (
    "What do you know now that you wish you had known at the start of this episode? Concrete, transferable facts "
    "about this rig, task, or embodiment (camera mounting and extrinsics, table and base geometry, gripper axis and "
    "offsets, controller behavior, metric scale), written as advice to a future agent attempting the same task. Say "
    "'none' if nothing qualifies.")

LABELS = ("x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper")
LOW = (-0.6, -0.6, 0.42, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0)
HIGH = (0.6, 0.6, 1.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
BOUNDS_TEXT = "Per-dimension bounds: " + ", ".join(f"{k}: [{lo:.4g}, {hi:.4g}]" for k, lo, hi in zip(LABELS, LOW, HIGH)) + "."
MOVE_DESCRIPTION = (
    "Move to absolute Cartesian end-effector targets (meters for positions, radians for rotations, per the dimension "
    "labels). The motion is a straight line interpolated at a fixed safe speed and the result reports its step "
    "count. Unnamed dimensions hold their current value. Coordinates are absolute in the embodiment's declared "
    "frame; on multi-arm embodiments each arm uses its own base frame and axes may differ between arms depending on "
    "mounting. Rotation dimensions are absolute targets measured relative to the trial's start orientation (0 means "
    "the start orientation) and interpolate linearly without wrapping, so prefer intermediate values for large "
    "rotations. " + BOUNDS_TEXT)
NOTE_DESCRIPTION = (
    "What you observe right now in the observation (images, if any, and state), and why you chose this motion. The "
    "user reads these notes live and in the saved transcript to follow what you see and what you decide. Write for "
    "them, in one or two plain sentences.")


def tool_declarations(targets_properties: bool = False) -> list[dict]:
    targets = {"type": "object", "description": "Map of dimension name to value. Valid names: " + ", ".join(LABELS)}
    if targets_properties:   # fallback only if the API rejects a property-less object
        targets["properties"] = {k: {"type": "number"} for k in LABELS}
    return [
        {"type": "function", "name": "move_to", "description": MOVE_DESCRIPTION,
         "parameters": {"type": "object", "properties": {"targets": targets,
                                                          "note": {"type": "string", "description": NOTE_DESCRIPTION}},
                        "required": ["targets", "note"]}},
        {"type": "function", "name": "done",
         "description": "Declare the task finished. The trial ends; a scorer judges success.",
         "parameters": {"type": "object", "properties": {"summary": {"type": "string"},
                                                          "hindsight": {"type": "string", "description": HINDSIGHT}},
                        "required": ["summary", "hindsight"]}},
        {"type": "function", "name": "give_up",
         "description": "Stop trying; the task cannot be completed. The trial ends.",
         "parameters": {"type": "object", "properties": {"reason": {"type": "string"},
                                                          "hindsight": {"type": "string", "description": HINDSIGHT}},
                        "required": ["reason", "hindsight"]}},
    ]


def system_prompt(task_id: int, budget: int) -> str:
    return SYSTEM_TEMPLATE.format(name=f"libero_10_task{task_id:02d}", budget=budget) + \
        "\n\nEmbodiment notes:\n" + BASE_DOCS.strip()


# ---------------------------------------------------------------- observation messages
def obs_blocks(obs: dict) -> list[dict]:
    """upstream policy._observation_content: state as labelled text, then the labelled camera frames."""
    lines = ["Current observation.", f"Instruction: {obs['instruction']}",
             f"state[eef_state]: {obs['eef_state_text']}",
             f"state[gripper_qpos]: {[round(float(v), 4) for v in obs['gripper_qpos']]}"]
    if obs.get("approvals"):
        lines.append(obs["approvals"])
    blocks = [{"type": "text", "text": "\n".join(lines)}]
    for cam in ("agentview", "wrist"):
        p = REPO / obs["images"][cam]
        blocks.append({"type": "text", "text": f"camera {cam!r} (step {obs['step']}):"})
        blocks.append({"type": "image", "data": base64.b64encode(p.read_bytes()).decode(), "mime_type": "image/png"})
    return blocks


def evicted_view(history: list[dict], horizon: int = 2) -> list[dict]:
    """upstream policy._evicted_view: frames older than the image horizon become a text stub."""
    img_idx = [i for i, h in enumerate(history) if h["kind"] == "obs"]
    stub = set(img_idx[:-horizon]) if horizon else set()
    out = []
    for i, h in enumerate(history):
        if h["kind"] == "model":
            out.extend(h["steps"])
            continue
        st = h["step"]
        if i in stub:
            content = st["content"]
            n_img = sum(1 for b in content if b.get("type") == "image")
            keep, skip = [], set()
            for j, b in enumerate(content):
                if b.get("type") == "image":
                    skip.add(j)
                    if j > 0 and content[j - 1].get("type") == "text":
                        skip.add(j - 1)
            keep = [b for j, b in enumerate(content) if j not in skip]
            keep.append({"type": "text", "text": f"[{n_img} camera frame(s) elided]"})
            st = {"type": "user_input", "content": keep}
        out.append(st)
    return out


def resend_steps(ix) -> list[dict]:
    """The model's returned steps as input steps for the next stateless call (thoughts as signatures only)."""
    out = []
    for s in ix.steps or []:
        t = getattr(s, "type", None)
        if t == "thought":
            if getattr(s, "signature", None):
                out.append({"type": "thought", "signature": s.signature})
        elif t == "function_call":
            out.append({"type": "function_call", "id": s.id, "name": s.name, "arguments": s.arguments or {}})
        elif t == "model_output":
            txt = "".join(c.text for c in (s.content or []) if getattr(c, "type", None) == "text" and c.text)
            if txt:
                out.append({"type": "model_output", "content": [{"type": "text", "text": txt}]})
        elif t == "user_input":
            continue
        else:
            out.append(s.model_dump(mode="json", exclude_none=True, by_alias=True))
    return out


def step_summary(s) -> dict:
    t = getattr(s, "type", "?")
    if t == "function_call":
        return {"type": t, "id": s.id, "name": s.name, "arguments": s.arguments}
    if t == "thought":
        return {"type": t, "summary": [c.text for c in (s.summary or []) if getattr(c, "text", None)],
                "signature_chars": len(s.signature or "")}
    if t == "model_output":
        return {"type": t, "text": "".join(c.text for c in (s.content or []) if getattr(c, "type", None) == "text"
                                           and c.text)}
    return {"type": t}


def status_code(e: Exception) -> int | None:
    for a in ("status_code", "code", "status"):
        v = getattr(e, a, None)
        if isinstance(v, int):
            return v
    r = getattr(e, "response", None)
    v = getattr(r, "status_code", None)
    return v if isinstance(v, int) else None


# ---------------------------------------------------------------- the episode
def run_episode(a) -> dict:
    run_dir = REPO / "runs" / "libero" / a.run_id
    name = f"t{a.task:02d}_i{a.init:02d}"
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY not set")   # checked before any sim process or attempt record exists
    rec = RunRecorder(f"{a.split}-{name}", root=run_dir / "loop")
    sim = SimClient(dict(run_id=a.run_id, actor=a.actor, brain="argon", split=a.split, views=a.views,
                         budget=dict(max_llm_calls=a.max_calls, max_steps=a.max_steps, max_usd=a.max_usd,
                                     max_output_tokens_per_call=a.max_output_tokens)),
                    log_path=(run_dir / "logs").mkdir(parents=True, exist_ok=True) or run_dir / "logs" / f"{name}.server.log")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY not set")
    transport = RecordingTransport(httpx.HTTPTransport())
    http = httpx.Client(transport=transport, timeout=a.http_timeout_s)
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(a.http_timeout_s * 1000),
                                                                     httpx_client=http))
    del key
    gc = {"thinking_level": a.thinking, "max_output_tokens": a.max_output_tokens}
    if a.thinking_summaries:
        gc["thinking_summaries"] = "auto"
    assert not FORBIDDEN & set(gc)
    decls = tool_declarations(a.targets_properties)
    t0 = time.time()
    try:
        r = sim.call("reset", task_id=a.task, init_state=a.init, split=a.split)
        if "error" in r:
            raise RuntimeError(r["error"])
        obs = r["obs"]
        ep_dir = (REPO / obs["images"]["agentview"]).parent.parent
        system = system_prompt(a.task, a.max_calls)
        rec.start({"brain": "argon", "model": ARGON, "task": obs["instruction"], "protocol": "llm-libero-2026",
                   "split": a.split, "task_id": a.task, "init_state": a.init, "generation_config": gc,
                   "views": a.views,
                   "store": False, "image_horizon": 2, "budget": dict(max_llm_calls=a.max_calls,
                                                                     max_steps=a.max_steps, max_usd=a.max_usd),
                   "sdk": f"google-genai {genai.__version__}", "episode_dir": str(ep_dir.relative_to(REPO)),
                   "evaluator": sim.evaluator}, system, decls)
        history = [{"kind": "goal", "step": {"type": "user_input",
                                             "content": [{"type": "text", "text": f"Goal: {obs['instruction']}"}]}},
                   {"kind": "obs", "step": {"type": "user_input", "content": obs_blocks(obs)}}]
        totals = dict(model_calls=0, usd=0.0, input=0, cached=0, output=0, thought=0, latency_s=0.0, tool_s=0.0)
        failures = 0
        last_input = 0
        over = False
        stop = None
        n = 0
        hindsight = None
        while not over:
            reserve = ((last_input + 4096 + 2 * 1100) * PRICE["input"] + a.max_output_tokens * PRICE["output"]) / 1e6
            b = sim.call("begin_call", reserve_usd=reserve)
            if b.get("episode_over"):
                stop = f"limit:{b['limit']['what']}"
                break
            if time.time() - t0 > a.max_wall_s:
                sim.call("end_limit", what="wall_s", used=round(time.time() - t0), limit=a.max_wall_s)
                stop = "limit:wall_s"
                break
            n += 1
            turn = {"turn": n, "kind": "model", "t_start": time.time()}
            ix, err = None, None
            tc = time.monotonic()
            for attempt in range(len(RETRY_BACKOFF_S) + 1):
                try:
                    ix = client.interactions.create(model=ARGON, input=evicted_view(history), system_instruction=system,
                                                    tools=decls, generation_config=gc, store=False,
                                                    timeout=a.http_timeout_s)
                    err = None
                    break
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"[:3000]
                    code = status_code(e)
                    retry = code in TRANSIENT or (code is None and any(f" {c}" in err for c in ("429", "500", "502", "503", "504")))
                    turn.setdefault("retries", []).append(dict(attempt=attempt, code=code, error=err[:500]))
                    if not retry or attempt == len(RETRY_BACKOFF_S):
                        break
                    time.sleep(RETRY_BACKOFF_S[attempt])
            turn["latency_s"] = round(time.monotonic() - tc, 3)
            turn["http"] = transport.drain()
            post = next((h["request"] for h in turn["http"] if h["method"] == "POST"), None)
            turn["request"] = request_digest(post)
            if ix is None:
                turn["error"] = err
                turn["usage"] = usage_dict(None)
                rec.write_turn(turn)
                sim.call("end_limit", what="api_error", used=n, limit=None, note=err)
                stop = "limit:api_error"
                break
            u = usage_dict(ix.usage)
            usd = cost_usd(ARGON, u) or 0.0
            last_input = u["input"]
            sim.call("charge", usage={k: u[k] for k in ("input", "cached", "output", "thought", "total")}, usd=usd)
            totals["model_calls"] += 1
            totals["usd"] += usd
            totals["latency_s"] += turn["latency_s"]
            for k in ("input", "cached", "output", "thought"):
                totals[k] += u[k]
            steps = ix.steps or []
            turn["usage"] = u
            turn["interaction"] = {"id": ix.id, "status": ix.status, "steps": [step_summary(s) for s in steps]}
            turn["thoughts"] = [x for s in turn["interaction"]["steps"] for x in s.get("summary", [])]
            turn["text"] = "".join(s.get("text", "") for s in turn["interaction"]["steps"] if s["type"] == "model_output")
            history.append({"kind": "model", "step": None, "steps": resend_steps(ix)})
            calls = [s for s in steps if s.type == "function_call"]
            turn["tool_calls"] = []
            if not calls:
                failures += 1
                turn["no_tool_call"] = dict(status=ix.status, failures=failures)
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    what = "output_tokens" if ix.status == "incomplete" else "no_tool_call"
                    sim.call("end_limit", what=what, used=failures, limit=MAX_CONSECUTIVE_FAILURES,
                             note=f"no tool call in {failures} consecutive turns (last status {ix.status})")
                    stop = f"limit:{what}"
                    over = True
                else:
                    history.append({"kind": "nudge", "step": {"type": "user_input", "content": [
                        {"type": "text", "text": "Respond with exactly one tool call."}]}})
            else:
                closed = False
                closed_after_failure = False
                new_obs = None
                tt = time.monotonic()
                for c in calls:
                    args = c.arguments or {}
                    res = None
                    if not closed:
                        if c.name == "move_to":
                            res = sim.call("move_to", targets=args.get("targets"), note=args.get("note"))
                        elif c.name in ("done", "give_up"):
                            kw = dict(hindsight=args.get("hindsight"))
                            kw["summary" if c.name == "done" else "reason"] = args.get("summary" if c.name == "done" else "reason")
                            res = sim.call(c.name, **kw)
                            hindsight = args.get("hindsight")
                        else:
                            res = {"tool_error": f"unknown tool {c.name!r}; available: move_to, done, give_up"}
                        if "error" in res and "tool_error" not in res and not res.get("episode_over"):
                            raise RuntimeError(f"server error: {res['error']}")
                        if "tool_error" in res:
                            text = res["tool_error"]
                            failures += 1
                            closed = closed_after_failure = True
                        else:
                            text = res.get("result", "")
                            if res.get("error"):
                                text = res["error"]
                            closed = True
                            if res.get("episode_over"):
                                over = True
                                stop = stop or ("model:" + c.name if c.name in ("done", "give_up") else "env_end")
                            else:
                                new_obs = res.get("obs")
                                failures = 0
                    elif closed_after_failure:
                        text = "ignored: an earlier call in this turn failed"
                    else:
                        text = "ignored: one tool call per turn"
                    history.append({"kind": "result", "step": {"type": "function_result", "call_id": c.id,
                                                               "name": c.name,
                                                               "result": [{"type": "text", "text": text}]}})
                    turn["tool_calls"].append({"call_id": c.id, "name": c.name, "args": args, "result": text,
                                               "server": {k: v for k, v in (res or {}).items() if k != "obs"},
                                               "is_error": bool(res and "tool_error" in res)})
                turn["tools_s"] = round(time.monotonic() - tt, 3)
                totals["tool_s"] += turn["tools_s"]
                if not over and failures >= MAX_CONSECUTIVE_FAILURES:
                    sim.call("end_limit", what="tool_errors", used=failures, limit=MAX_CONSECUTIVE_FAILURES,
                             note="tool calls kept failing")
                    stop = "limit:tool_errors"
                    over = True
                if new_obs is not None and not over:
                    history.append({"kind": "obs", "step": {"type": "user_input", "content": obs_blocks(new_obs)}})
            turn["totals"] = {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in totals.items()},
                              "wall_s": round(time.time() - t0, 1)}
            rec.write_turn(turn)
        # flatten model steps into the history record for the transcript
        server_rec = json.loads((ep_dir / "episode.json").read_text()) if (ep_dir / "episode.json").exists() else {}
    finally:
        sim.close()
        http.close()
    out = dict(run_id=a.run_id, split=a.split, task_id=a.task, init_state=a.init, model=ARGON,
               views=server_rec.get("views", "paper"),
               thinking_level=a.thinking, max_output_tokens=a.max_output_tokens,
               outcome=server_rec.get("outcome"), counted_success=server_rec.get("counted_success"),
               ended_by=server_rec.get("ended_by"), limit=server_rec.get("limit"), loop_stop=stop,
               claim=(server_rec.get("claim") or {}).get("kind"), false_claim=server_rec.get("false_claim"),
               steps=server_rec.get("steps"), llm_calls=totals["model_calls"],
               tokens={k: totals[k] for k in ("input", "cached", "output", "thought")},
               usd=round(totals["usd"], 4), wall_s=round(time.time() - t0, 1),
               model_latency_s=round(totals["latency_s"], 1), sim_and_tool_s=round(totals["tool_s"], 1),
               flags=server_rec.get("flags"), replay=server_rec.get("replay"), monitor=server_rec.get("monitor"),
               tool_errors=server_rec.get("tool_errors"), hindsight=hindsight,
               episode_dir=str(ep_dir.relative_to(REPO)), loop_dir=str(rec.dir.relative_to(REPO)))
    rec.finish(stop or "done", "", extra=dict(episode=out))
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "results" / f"{name}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", default="feasibility")
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--init", type=int, required=True)
    ap.add_argument("--actor", default="argon")
    ap.add_argument("--thinking", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--max-output-tokens", type=int, default=16000)
    ap.add_argument("--max-calls", type=int, default=80)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--max-usd", type=float, default=30.0)
    ap.add_argument("--max-wall-s", type=float, default=3 * 3600)
    ap.add_argument("--http-timeout-s", type=float, default=1200.0)
    ap.add_argument("--no-thinking-summaries", dest="thinking_summaries", action="store_false")
    ap.add_argument("--targets-properties", action="store_true")
    ap.add_argument("--views", default="true", choices=["true", "paper"],
                    help="true: real camera views (default since 2026-10-25 14:50); paper: the published, mirrored "
                         "views (agentview mirrored left-right, wrist top-bottom)")
    a = ap.parse_args()
    out = run_episode(a)
    print(json.dumps({k: out[k] for k in ("task_id", "init_state", "outcome", "ended_by", "loop_stop", "steps",
                                          "llm_calls", "usd", "wall_s")}))


if __name__ == "__main__":
    main()
