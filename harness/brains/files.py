"""write_run_file: lets a function-calling brain save its deliverable (a labels JSON, a report) as a file.

The Claude stand-in writes files with its own editor tools; Argon acts only through function calls, and no
harness tool writes a free-form file. This tool fills that gap and nothing more:

- it writes only under runs/argon/ (Argon's run directories), never playbook/, twin/, harness/ or data/;
- a .json file must parse, so a malformed deliverable is caught at write time, not by the watcher;
- key-shaped strings are refused (rules.md: no secrets in logs);
- it goes through harness.registry like every other tool, so each write is in runs/calls/.

It is registered from harness/brains/ (not harness/tools/), so harness.cli does not list it; the integrity
audit shows its calls under the review item tools_not_in_registry.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from harness.registry import REPO, TOOLS, tool

ROOT = REPO / "runs" / "argon"
SUFFIXES = {".json", ".jsonl", ".md", ".txt"}
MAX_CHARS = 2_000_000
SECRET_RE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}")


def write_run_file(path: str, content: str) -> dict:
    p = (Path(path) if str(path).startswith("/") else REPO / str(path)).resolve()
    if not p.is_relative_to(ROOT):
        raise PermissionError(f"write_run_file writes only under runs/argon/; got {path!r}")
    if p.suffix.lower() not in SUFFIXES:
        raise ValueError(f"suffix must be one of {sorted(SUFFIXES)}; got {p.suffix!r}")
    if not isinstance(content, str):
        raise ValueError("content must be a string (for .json, the JSON text)")
    if len(content) > MAX_CHARS:
        raise ValueError(f"content is {len(content)} chars; the limit is {MAX_CHARS}")
    if SECRET_RE.search(content):
        raise ValueError("content contains a string that looks like an API key; not written")
    if p.suffix.lower() == ".json":
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"not valid JSON, file not written: {e.msg} at line {e.lineno} column {e.colno} "
                             f"(char {e.pos})") from None
    existed = p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    data = content.encode()
    return {"path": str(p.relative_to(REPO) if p.is_relative_to(REPO) else p), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()[:16],
            "overwrote": existed}


if "write_run_file" not in TOOLS:
    tool(
        "write_run_file",
        "Write a text file under runs/argon/ (your run's deliverables, e.g. a labels JSON at the output path your "
        "task names). Overwrites the file if it exists. A .json file must be valid JSON or nothing is written and "
        "the error says where it failed. Returns path, bytes, sha256.",
        {"type": "object", "required": ["path", "content"], "properties": {
            "path": {"type": "string", "description": "repo-relative path under runs/argon/, ending .json, .jsonl, .md or .txt"},
            "content": {"type": "string", "description": "the full file text (for .json, the JSON document as text)"}}},
    )(write_run_file)
