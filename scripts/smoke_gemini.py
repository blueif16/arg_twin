"""Smoke test for the Gemini Interactions API (google-genai >= 2.30).

Makes two cheap calls to gemini-3.5-flash-lite and prints the response shape:
  1. one plain text call;
  2. one function-calling round trip:
     call with a tool -> run the tool locally -> send function_result back
     with previous_interaction_id -> read the final text.
With --list-models it also lists the models this API key can see.

Usage (from the repo root):
  .venv/bin/python scripts/smoke_gemini.py [--list-models] [--model MODEL]

Needs GEMINI_API_KEY or GOOGLE_API_KEY in the environment (the SDK prefers
GOOGLE_API_KEY when both are set). The key is never printed.
Exit code 0 = both calls worked, 1 = an API call failed, 2 = no key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from google import genai

DEFAULT_MODEL = "gemini-3.5-flash-lite"

JOINT_TOOL = {
    "type": "function",
    "name": "get_joint_limits",
    "description": "Return the position limits of one robot arm joint, in radians.",
    "parameters": {
        "type": "object",
        "properties": {
            "joint": {
                "type": "string",
                "enum": ["shoulder", "elbow", "wrist"],
                "description": "Joint name.",
            }
        },
        "required": ["joint"],
    },
}

FAKE_LIMITS = {
    "shoulder": {"lower": -2.9, "upper": 2.9},
    "elbow": {"lower": -1.8, "upper": 1.8},
    "wrist": {"lower": -3.0, "upper": 3.0},
}


def get_joint_limits(joint: str) -> dict[str, Any]:
    """Local stand-in for a real tool."""
    return {"joint": joint, **FAKE_LIMITS.get(joint, {"error": "unknown joint"})}


def step_shape(step: Any) -> dict[str, Any]:
    """Summarize one step without dumping signatures or large payloads."""
    out: dict[str, Any] = {"type": step.type}
    if step.type == "function_call":
        out.update(id=step.id, name=step.name, arguments=step.arguments)
    elif step.type in ("model_output", "user_input"):
        out["content_types"] = [c.type for c in (step.content or [])]
    elif step.type == "thought":
        out["has_signature"] = bool(step.signature)
        out["summary_items"] = len(step.summary or [])
    return out


def interaction_shape(interaction: Any) -> dict[str, Any]:
    usage = interaction.usage.model_dump(exclude_none=True) if interaction.usage else None
    return {
        "id": interaction.id,
        "status": interaction.status,
        "model": interaction.model,
        "steps": [step_shape(s) for s in (interaction.steps or [])],
        "output_text": interaction.output_text,
        "usage": usage,
    }


def show(label: str, data: Any) -> None:
    print(f"\n=== {label}")
    print(json.dumps(data, indent=2, default=str))


def list_models(client: genai.Client) -> None:
    rows = []
    for m in client.models.list():
        rows.append({
            "name": m.name,
            "input_token_limit": m.input_token_limit,
            "output_token_limit": m.output_token_limit,
            "supported_actions": m.supported_actions,
        })
    show(f"models.list ({len(rows)} models)", rows)


def plain_call(client: genai.Client, model: str) -> None:
    interaction = client.interactions.create(
        model=model,
        input="Reply with exactly the word: pong",
    )
    show("1. plain call", interaction_shape(interaction))
    if not interaction.output_text:
        raise RuntimeError("plain call returned no output_text")


def function_call_round_trip(client: genai.Client, model: str) -> None:
    first = client.interactions.create(
        model=model,
        input="What are the position limits of the wrist joint? Use the tool.",
        tools=[JOINT_TOOL],
        # tool_choice is interaction-scoped: "any" forces a call on this turn only.
        generation_config={"tool_choice": "any"},
    )
    show("2a. function-calling turn 1", interaction_shape(first))

    calls = [s for s in (first.steps or []) if s.type == "function_call"]
    if not calls:
        raise RuntimeError("model returned no function_call step")

    results = [
        {
            "type": "function_result",
            "call_id": call.id,
            "name": call.name,
            "result": [{"type": "text", "text": json.dumps(get_joint_limits(**call.arguments))}],
        }
        for call in calls
    ]
    second = client.interactions.create(
        model=model,
        previous_interaction_id=first.id,
        tools=[JOINT_TOOL],  # tools are interaction-scoped, so send them again
        input=results,
    )
    show("2b. function-calling turn 2 (after function_result)", interaction_shape(second))
    if not second.output_text:
        raise RuntimeError("function-calling turn 2 returned no output_text")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("No GEMINI_API_KEY or GOOGLE_API_KEY set; nothing to do.", file=sys.stderr)
        return 2

    # HttpOptions.timeout is in milliseconds. 2 minutes is plenty for Flash-Lite;
    # Argon "high" calls need 15-20 minutes or background=True.
    client = genai.Client(http_options={"timeout": 120_000})
    print(f"google-genai {genai.__version__}, model {args.model}")

    try:
        if args.list_models:
            list_models(client)
        plain_call(client, args.model)
        function_call_round_trip(client, args.model)
    except Exception as exc:  # noqa: BLE001 - report any SDK/API error and exit non-zero
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
