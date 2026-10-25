"""Playbook tools: progressive disclosure of the skill library to any brain.

A brain sees a one-line index of every skill (list_skills) and loads a skill's full text, or one of
its helper files, only when it needs it (read_skill). read_playbook_file serves the three top files.
append_lesson is the one write path into skills: a dated note under the skill's Lessons section
that must cite evidence (rules.md: every claim in Lessons cites a run log, a file or a source).
log_note lets a brain record a working note or decision point as a tool call; on Gemini, free text
right before a function call can cause Malformed_Function_Call (docs/research/interactions-api-notes.md).
"""

from __future__ import annotations

import datetime as dt
import fcntl
import re
from pathlib import Path

from harness.registry import REPO, tool

PLAYBOOK = REPO / "playbook"  # tests point this at a temp copy
TOP_FILES = {"readme": "README.md", "orchestrator": "orchestrator.md", "rules": "rules.md"}
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SECRET_RE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}")
MAX_HELPER_BYTES = 200_000
MAX_NOTE_CHARS = 2000


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split '---' frontmatter from the body. Plain 'key: value' lines; values may hold colons."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    meta: dict[str, str] = {}
    key = None
    for line in text[3:end].splitlines():
        m = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if m:
            key = m.group(1)
            meta[key] = m.group(2).strip().strip("'\"")
        elif key and line.strip():  # indented continuation of a long value
            meta[key] = (meta[key] + " " + line.strip()).strip()
    body = text[end + 4:].lstrip("\n")
    return meta, body


def _skill_dirs() -> list[Path]:
    root = PLAYBOOK / "skills"
    return sorted(p.parent for p in root.glob("*/SKILL.md")) if root.is_dir() else []


def _find_skill(name: str) -> Path:
    if not NAME_RE.match(name or ""):
        raise ValueError(f"bad skill name {name!r}: use the kebab-case name from list_skills")
    d = PLAYBOOK / "skills" / name
    if (d / "SKILL.md").is_file():
        return d
    for d in _skill_dirs():  # frontmatter name may differ from the folder name
        if parse_frontmatter((d / "SKILL.md").read_text())[0].get("name") == name:
            return d
    known = [d.name for d in _skill_dirs()]
    raise FileNotFoundError(f"no skill {name!r}; known: {known}")


def _lessons_count(body: str) -> int:
    m = re.search(r"^##\s+Lessons\b.*?$(.*?)(?=^##\s|\Z)", body, re.M | re.S)
    return len(re.findall(r"^\s*[-*]\s+\S", m.group(1), re.M)) if m else 0


@tool(
    "list_skills",
    "Index of the playbook's skills: name, stage and a one-line description of when to use each. "
    "Read this first, then load a skill with read_skill before doing its job.",
    {"type": "object", "properties": {
        "stage": {"type": "string", "description": "only skills of this stage "
                  "(read | build | tools | practice | compile | watch | infra)"}}},
)
def list_skills(stage: str | None = None) -> dict:
    skills = []
    for d in _skill_dirs():
        text = (d / "SKILL.md").read_text()
        meta, body = parse_frontmatter(text)
        if stage and meta.get("stage") != stage:
            continue
        row = {"name": meta.get("name") or d.name, "stage": meta.get("stage"),
               "description": meta.get("description"), "path": _rel(d / "SKILL.md"),
               "lessons": _lessons_count(body)}
        if meta.get("name") and meta["name"] != d.name:
            row["warning"] = f"frontmatter name differs from folder {d.name!r}"
        skills.append(row)
    return {"count": len(skills), "skills": skills,
            "stages": sorted({s["stage"] for s in skills if s["stage"]})}


@tool(
    "read_skill",
    "Load one skill: its full SKILL.md text (goal, inputs, tools, approach, checks, lessons) and the "
    "list of its helper files. Pass file to read one helper file instead.",
    {"type": "object", "required": ["name"], "properties": {
        "name": {"type": "string", "description": "skill name from list_skills"},
        "file": {"type": "string", "description": "helper file path relative to the skill folder"}}},
)
def read_skill(name: str, file: str | None = None) -> dict:
    d = _find_skill(name)
    helpers = sorted(p for p in d.rglob("*") if p.is_file() and p.name != "SKILL.md"
                     and not any(part.startswith((".", "__")) for part in p.relative_to(d).parts))
    if file:
        target = (d / file).resolve()
        if not target.is_relative_to(d.resolve()) or not target.is_file():
            raise FileNotFoundError(f"no helper {file!r} in skill {name!r}; helpers: "
                                    f"{[str(p.relative_to(d)) for p in helpers]}")
        if target.stat().st_size > MAX_HELPER_BYTES:
            raise ValueError(f"{file} is {target.stat().st_size} bytes (> {MAX_HELPER_BYTES}); "
                             "open it with a tool built for that file type")
        return {"skill": name, "file": str(target.relative_to(d.resolve())), "path": _rel(target),
                "text": target.read_text(errors="replace")}
    text = (d / "SKILL.md").read_text()
    meta, body = parse_frontmatter(text)
    return {"name": meta.get("name") or d.name, "path": _rel(d / "SKILL.md"), "frontmatter": meta,
            "text": text, "lessons": _lessons_count(body),
            "helpers": [{"file": str(p.relative_to(d)), "bytes": p.stat().st_size} for p in helpers[:200]]}


@tool(
    "read_playbook_file",
    "Read one of the playbook's top files: README (layout and skill format), orchestrator (the fixed "
    "loop, stages and done-criteria) or rules (integrity, no sim cheating, records).",
    {"type": "object", "required": ["file"], "properties": {
        "file": {"type": "string", "enum": ["README", "orchestrator", "rules"]}}},
)
def read_playbook_file(file: str) -> dict:
    key = file.strip().lower().removesuffix(".md")
    if key not in TOP_FILES:
        raise ValueError(f"file must be one of README, orchestrator, rules (got {file!r})")
    p = PLAYBOOK / TOP_FILES[key]
    return {"file": TOP_FILES[key], "path": _rel(p), "text": p.read_text()}


def _check_evidence(items: list[str]) -> list[dict]:
    out = []
    for e in items:
        e = e.strip()
        if re.match(r"^https?://", e):
            out.append({"ref": e, "kind": "url"})
            continue
        path = re.split(r"[#:]", e, maxsplit=1)[0]  # allow 'file.jsonl:12' or 'file.md#section'
        p = Path(path) if path.startswith("/") else REPO / path
        out.append({"ref": e, "kind": "file", "exists": bool(path) and p.exists()})
    return out


@tool(
    "append_lesson",
    "Append a dated lesson under a skill's Lessons section: what worked or failed, with numbers. "
    "evidence must cite at least one existing repo file (run log, tool output, e.g. "
    "'runs/loop/<run-id>/turns.jsonl') or a source URL. Lessons are the only part of a skill a brain writes.",
    {"type": "object", "required": ["skill", "note", "evidence"], "properties": {
        "skill": {"type": "string", "description": "skill name from list_skills"},
        "note": {"type": "string", "description": "one paragraph: what was tried, what happened, numbers"},
        "evidence": {"type": "array", "items": {"type": "string"},
                     "description": "repo-relative paths (optionally path:line) or URLs backing the note"},
        "author": {"type": "string", "description": "who learned it, e.g. argon or claude:<agent>"}}},
)
def append_lesson(skill: str, note: str, evidence: list[str] | str, author: str = "unknown") -> dict:
    if isinstance(evidence, str):
        evidence = [evidence]
    note = " ".join((note or "").split())
    if not note:
        raise ValueError("note is empty")
    if len(note) > MAX_NOTE_CHARS:
        raise ValueError(f"note is {len(note)} chars; keep it under {MAX_NOTE_CHARS}")
    if SECRET_RE.search(note) or any(SECRET_RE.search(e) for e in evidence):
        raise ValueError("note or evidence looks like it contains a secret; rules.md forbids secrets in skill files")
    checks = _check_evidence([e for e in evidence if e and e.strip()])
    if not any(c["kind"] == "url" or c.get("exists") for c in checks):
        raise ValueError(f"evidence must include an existing repo file or a URL; got {checks}")
    author = re.sub(r"[^\w:.@/-]", "", author or "unknown")[:40] or "unknown"
    refs = ", ".join(f"`{c['ref']}`" if c["kind"] == "file" else c["ref"] for c in checks)
    entry = f"- **{dt.date.today().isoformat()}** ({author}): {note} Evidence: {refs}."

    path = _find_skill(skill) / "SKILL.md"
    with open(path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # parallel agents may append to the same skill
        lines = f.read().splitlines()
        start = next((i for i, ln in enumerate(lines) if re.match(r"^##\s+Lessons\b", ln)), None)
        if start is None:
            while lines and not lines[-1].strip():
                lines.pop()
            lines += ["", "## Lessons", "", entry]
        else:
            end = next((i for i in range(start + 1, len(lines)) if re.match(r"^##\s", lines[i])), len(lines))
            insert = end
            while insert > start + 1 and not lines[insert - 1].strip():
                insert -= 1
            block = [entry] if insert > start + 1 else ["", entry]
            if insert == end < len(lines):  # next heading directly follows: keep one blank line
                block.append("")
            lines[insert:insert] = block
        f.seek(0)
        f.write("\n".join(lines) + "\n")
        f.truncate()
    return {"ok": True, "skill": skill, "path": _rel(path), "entry": entry, "evidence": checks,
            "missing_evidence": [c["ref"] for c in checks if c["kind"] == "file" and not c["exists"]]}


@tool(
    "log_note",
    "Record a short working note, plan or decision point in the run record. It has no other effect. "
    "Use it for reasoning you want kept, instead of writing text right before a tool call.",
    {"type": "object", "required": ["text"], "properties": {
        "text": {"type": "string"},
        "kind": {"type": "string", "enum": ["note", "plan", "decision", "surprise"],
                 "description": "default note"}}},
)
def log_note(text: str, kind: str = "note") -> dict:
    return {"ok": True, "kind": kind, "chars": len(text or "")}
