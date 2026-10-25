"""Tool registry shared by every brain (Claude stand-in via CLI, Argon via function calling).

A tool is a plain Python function that takes JSON-able keyword arguments and returns a
JSON-able dict. Register it with @tool in any module under harness/tools/; modules are
discovered automatically, so no shared file needs editing when a tool is added.
"""

import importlib
import json
import pkgutil
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
CALL_LOG_DIR = REPO / "runs" / "calls"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema for the arguments object
    fn: Callable[..., dict]

    def declaration(self) -> dict:
        """Function declaration in the shape the Gemini Interactions API expects."""
        return {"type": "function", "name": self.name, "description": self.description,
                "parameters": gemini_schema(self.parameters)}


GEMINI_KEYS = {"type", "description", "enum", "format", "minimum", "maximum", "items", "prefixItems",
               "minItems", "maxItems", "required", "additionalProperties", "anyOf", "properties"}


def gemini_schema(s):
    """Copy of a JSON schema restricted to Gemini's documented subset (docs/research/interactions-api-notes.md):
    a type list such as ["integer", "null"] becomes anyOf of single types, a default moves into the
    description, and other keywords are dropped."""
    if not isinstance(s, dict):
        return s
    out = {}
    for k, v in s.items():
        if k == "properties":
            out[k] = {name: gemini_schema(sub) for name, sub in v.items()}
        elif k in ("items", "additionalProperties"):
            out[k] = gemini_schema(v)
        elif k in ("anyOf", "prefixItems"):
            out[k] = [gemini_schema(x) for x in v]
        elif k in GEMINI_KEYS:
            out[k] = v
    if isinstance(out.get("type"), list):
        out["anyOf"] = [{"type": t} for t in out.pop("type")] + out.get("anyOf", [])
    if "default" in s and "default" not in out.get("description", "").lower():
        out["description"] = (out.get("description", "") + f" (default {json.dumps(s['default'])})").strip()
    return out


TOOLS: dict[str, Tool] = {}
LOAD_ERRORS: dict[str, str] = {}  # module -> import error; one broken module must not take down the rest


def tool(name: str, description: str, parameters: dict | None = None):
    def wrap(fn):
        if name in TOOLS:
            raise ValueError(f"duplicate tool name: {name}")
        TOOLS[name] = Tool(name, description, parameters or {"type": "object", "properties": {}}, fn)
        return fn
    return wrap


def load_all() -> dict[str, Tool]:
    import harness.tools as pkg
    for mod in pkgutil.iter_modules(pkg.__path__):
        try:
            importlib.import_module(f"harness.tools.{mod.name}")
        except Exception as e:
            LOAD_ERRORS[mod.name] = f"{type(e).__name__}: {e}"
    return TOOLS


SPLITS = REPO / "playbook" / "splits.json"
EXAM_CALLER = "exam-scorer"  # the only caller allowed to touch exam recordings (rules.md: frozen evaluator)


def exam_refs(args: dict) -> list[str]:
    """Exam-split episodes or recordings named in a tool's arguments."""
    if not SPLITS.exists():
        return []
    eps = json.loads(SPLITS.read_text())["episodes"]
    exam = {k for k, v in eps.items() if v["split"] == "exam"}
    exam |= {v["recording"] for v in eps.values() if v["split"] == "exam"}
    hits = []
    for key in ("episode", "episodes", "recording", "recordings"):
        vals = args.get(key)
        for v in vals if isinstance(vals, list) else [vals]:
            if v is not None and str(v) in exam:
                hits.append(str(v))
    return hits


CALLER: ContextVar = ContextVar("harness_caller", default=None)  # the caller of the tool call in progress
RECORDING_RE = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}")
EPISODE_DESCRIPTOR_KEYS = {"recording", "split", "playbook_split", "hf_split", "n_segments", "n_failed", "segments"}


def current_caller() -> str | None:
    """Caller of the registry call in progress (None outside harness.registry.call)."""
    return CALLER.get()


def exam_leaks(result_text: str, result) -> list[str]:
    """Exam ids a tool result would hand over: any exam recording id in its JSON text, or an exam episode index in a
    dict that describes a REASSEMBLE episode (an `episode` key next to a recording, split or label key). One regex
    pass, plus a walk only when the text has an "episode" key. SO-101 tools use `ep`, so their indices never match."""
    from twin.exam import exam_episodes, exam_recordings
    hits = sorted(set(RECORDING_RE.findall(result_text)) & exam_recordings())
    if '"episode"' in result_text:
        eps, stack = exam_episodes(), [result]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                ep = x.get("episode")
                if isinstance(ep, int) and not isinstance(ep, bool) and ep in eps and EPISODE_DESCRIPTOR_KEYS & x.keys():
                    hits.append(f"episode {ep}")
                stack.extend(x.values())
            elif isinstance(x, (list, tuple)):
                stack.extend(x)
    return sorted(set(hits))


def call(name: str, args: dict, caller: str = "unknown") -> dict:
    """Run a tool and append the call and its result to the run log (replayable record).

    Exam guard, two layers: arguments naming an exam episode or recording are refused before the tool runs, and a
    result that names one is withheld from any caller but exam-scorer (the tool's own files may already exist)."""
    load_all()
    from twin import exam
    t0 = time.time()
    tok_c, tok_e = CALLER.set(caller), exam.EXAM_OK.set(caller == EXAM_CALLER)
    withheld = None
    try:
        if caller != EXAM_CALLER and (hits := exam_refs(args)):
            raise PermissionError(f"exam split is held out until the final score: {hits} (playbook/splits.json)")
        if name not in TOOLS and LOAD_ERRORS:
            raise KeyError(f"unknown tool {name!r}; modules failed to load: {LOAD_ERRORS}")
        result = TOOLS[name].fn(**args)
        ok = True
    except Exception as e:  # tool errors go back to the brain as data, not crashes
        result, ok = {"error": f"{type(e).__name__}: {e}"}, False
    finally:
        CALLER.reset(tok_c)
        exam.EXAM_OK.reset(tok_e)
    text = json.dumps(result, default=str)
    if ok and caller != EXAM_CALLER and (withheld := exam_leaks(text, result)):
        result, ok = {"error": f"PermissionError: the result named exam data {withheld} and was withheld "
                               "(exam split is held out until the final score; playbook/splits.json)"}, False
        text = json.dumps(result)
    head = {"t": t0, "dt": round(time.time() - t0, 3), "caller": caller, "tool": name, "args": args, "ok": ok}
    if withheld:
        head["withheld_exam_ids"] = withheld
    CALL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CALL_LOG_DIR / f"{time.strftime('%Y%m%d')}.jsonl", "a") as f:
        f.write(json.dumps(head, default=str)[:-1] + ', "result": ' + text + "}\n")
    return result
