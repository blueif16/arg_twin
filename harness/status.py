"""Bot18 page: runs/status/index.html, one self-contained file that shows the loop and its results.

Layout and update rules (slots, tokens, motion) are in docs/design/ui-protocol.md; follow it when adding
information or changing the template (harness/status_template.html).

    .venv/bin/python -m harness.status            # rebuild the page and print its path
    .venv/bin/python -m harness.status --bundle   # also copy every referenced file into runs/status/media/
                                                  # so the folder can be zipped or published on its own

Sources (only validation.yaml is required; everything else may be missing):
    playbook/validation.yaml      stages, checks, frozen targets, data tracks (`tracks`). The loop's steps (A-G) are
                                  shared by every track; the first track's checks live in them, another track's in a
                                  container stage (`track: so101`) whose checks each name their `step`
    runs/status/checks.jsonl      measurements, written by the record_check tool
    runs/status/events.jsonl      milestones, regressions, decisions, blockers, notes (record_event + automatic)
    runs/status/media.jsonl       videos, images and comparisons per stage (publish_media)
    runs/status/story.yaml        story (demo) mode text: per section title, claim, numbers, featured media, with
                                  optional per-track overrides; written with defaults on the first build
    playbook/skills/*/SKILL.md    does the skill exist; does Lessons show a dated real run; its description,
                                  Goal, Approach and Lessons are shown in each chapter's method sheet; its
                                  frontmatter `steps:` puts its tag on the overview (skills_index)
    runs/calls/*.jsonl            tool calls per caller, last activity
    runs/loop/*/summary.json      tokens, calls and wall time per brain run
    runs/<brain>/<run>/decisions.jsonl   reconstruction decision logs, per run
    runs/<brain>/<run>/manifest.json, library/index.json, task_tools/manifest.json
                                  frozen practice runs, compiled routines and task tool releases: with the skills'
                                  dated Lessons and the records, what each step of the two loops optimized
                                  (loop_progress; story.yaml `loops` names the loops)

Every record sits on one data track and one loop step (place): the page builds every track's view at build time
and switches between them in the browser. Judging lives here (judge, brain_view, stage states) so the record_check
tool, status_summary and the page always agree. Results are grouped by brain and never merged: a Claude stand-in
number and an Argon number are separate rows everywhere. A step's state comes only from its checks; a brain's events
and media on a step with no measured check (activity) show it on the page as in progress, not scored yet, never as a
verdict.
"""

from __future__ import annotations

import datetime as dt
import argparse
import html
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import yaml

from harness.registry import CALL_LOG_DIR, REPO

# Module-level paths so tests can point them at a temp copy.
VALIDATION = REPO / "playbook" / "validation.yaml"
SKILLS_DIR = REPO / "playbook" / "skills"
STATUS_DIR = REPO / "runs" / "status"
LOOP_DIR = REPO / "runs" / "loop"
RUNS_DIR = REPO / "runs"  # runs/<brain>/<run>/decisions.jsonl
CALLS_DIR = CALL_LOG_DIR
TEMPLATE = Path(__file__).with_name("status_template.html")
VENDOR = Path(__file__).with_name("vendor")  # GSAP 3.13.0 (gsap.com/standard-license), inlined into the page
VENDOR_JS = ("gsap.min.js", "ScrollTrigger.min.js", "ScrollToPlugin.min.js")

GATED = (">=", "<=")
DIR_SYM = {">=": "≥", "<=": "≤"}
BRAIN_LABELS = {"claude": "Claude stand-in", "argon": "Argon"}
STAGE_LABELS = {"not_started": "not started", "in_progress": "in progress", "passing": "passing",
                "failing": "failing", "regressed": "regressed"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
VIDEO_EXT = {".mp4", ".m4v", ".mov", ".webm"}
URL_RE = re.compile(r"^https?://")
MAX_HIST = 50       # points per sparkline
MAX_RECENT = 6      # records per check in the drill-down
MAX_EVENTS = 300


def checks_path() -> Path:
    return STATUS_DIR / "checks.jsonl"


def events_path() -> Path:
    return STATUS_DIR / "events.jsonl"


def page_path() -> Path:
    return STATUS_DIR / "index.html"


def media_path() -> Path:
    return STATUS_DIR / "media.jsonl"


def story_path() -> Path:
    return STATUS_DIR / "story.yaml"


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


# ------------------------------------------------------------------------------- judging

_CODES_RE = (re.compile(r"\s*\((?:stages?|steps?)\s+[A-Z]\d?(?:\s*[-–,/]\s*[A-Z]\d?)*\)"),
             re.compile(r",?\s*\bstages?\s+[A-Z]\d?(?:\s*[-–]\s*[A-Z]\d?)?\b"))


def _uncode(text: str) -> str:
    """Prose without stage codes ('full loop, stages A-G' -> 'full loop'): codes stay internal on the page."""
    for r in _CODES_RE:
        text = r.sub("", text)
    return text.strip()


def _track(t: dict) -> dict:
    """A data track from validation.yaml `tracks`: the same playbook on a different kind of recordings."""
    def prose(s: str) -> str:  # a sentence for the card: capital first letter (not for ids like 'lerobot/...'), full stop
        s = " ".join(str(s or "").split())
        s = s[:1].upper() + s[1:] if re.match(r"[a-z]+(\s|,|$)", s) else s
        return s + "." if s and s[-1] not in ".!?" else s

    name = str(t.get("name") or t["id"])
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", name)
    runs = t.get("runs") or []
    return {"id": str(t["id"]), "name": name, "title": m.group(1) if m else name,
            "detail": prose(m.group(2)) if m else "", "scope": prose(_uncode(str(t.get("scope") or ""))),
            "status": str(t.get("status") or "active"), "why": prose(t.get("why")),
            "runs": [str(r) for r in ([runs] if isinstance(runs, str) else runs) if r]}


INPUT = "input"  # the step name of the loop's start: a track's real footage


def _story_tracks() -> list[dict]:
    """Tracks story.yaml defines itself (`tracks: <id>: define: {name, scope, status, why, runs}`), for a track whose
    validation.yaml entry has to wait (a frozen run hashes validation.yaml). validation.yaml always wins over these."""
    try:
        raw = yaml.safe_load(story_path().read_text()) or {}
    except (OSError, yaml.YAMLError):
        return []
    out = []
    for tid, t in (raw.get("tracks") or {}).items():
        d = (t or {}).get("define") if isinstance(t, dict) else None
        if isinstance(d, dict):
            out.append(dict(d, id=str(tid)))
    return out


def load_validation(path: Path | None = None) -> dict:
    """Stages, checks and data tracks.

    The loop's steps are the stages without `track:` (A, B0 ... G); every data track runs through the same steps.
    The first track listed is the loop's own: its checks are those steps' checks. Another track keeps its checks in a
    container stage (`track: so101`) and each check names the loop step it belongs to (`step: B0`); a check without one
    sits on the step most of its stage's checks name. A track's view of step X is its checks placed on X.
    """
    raw = yaml.safe_load(Path(path or VALIDATION).read_text())
    tracks = [_track(t) for t in raw.get("tracks") or [] if isinstance(t, dict) and t.get("id")]
    if not tracks:  # no tracks listed: one implicit track, so every record and view still has one
        tracks = [_track({"id": "main", "name": "Recordings", "scope": "the loop"})]
    primary = tracks[0]["id"]
    warnings = []
    for t in _story_tracks():  # provisional: defined in story.yaml until validation.yaml lists it (then that wins)
        if t["id"] not in {x["id"] for x in tracks}:
            tracks.append(dict(_track(t), provisional=True))
            warnings.append(f"track {t['id']!r} is defined in story.yaml (`define:`), not in validation.yaml: "
                            "provisional, with no checks until validation.yaml lists it")
    stages = []
    for s in raw.get("stages", []):
        track = str(s.get("track") or primary)
        if track not in {t["id"] for t in tracks}:
            warnings.append(f"stage {s['id']}: track {track!r} is not under `tracks`; shown as its own track")
            tracks.append(_track({"id": track}))
        checks = [{"id": c["id"], "short": c["id"].split(".", 1)[-1], "name": c.get("name", c["id"]),
                   "metric": c.get("metric", ""), "direction": c.get("direction", "track"),
                   "target": c.get("target"), "why": c.get("why", ""), "track": track,
                   "step": str(c.get("step") or "")} for c in s.get("checks", [])]
        role = ("track" if track != primary else
                "integrity" if s.get("integrity") or s["id"] == "G" else
                "feedback" if s.get("feedback") or s["id"] == "F" else "main")
        stages.append({"id": s["id"], "name": s.get("name", s["id"]), "skill": s.get("skill"),
                       "role": role, "track": track, "checks": checks, "step": str(s.get("step") or "")})
    loop = [s["id"] for s in stages if s["role"] != "track"]
    for s in stages:
        if s["role"] != "track":  # a loop step's checks sit on it
            s["step"] = s["id"]
            for c in s["checks"]:
                c["step"] = s["id"]
            continue
        named = [c["step"] for c in s["checks"] if c["step"] in loop]
        s["step"] = (s["step"] if s["step"] in loop else max(named, key=named.count) if named
                     else loop[0] if loop else s["id"])
        for c in s["checks"]:
            if c["step"] not in loop:
                warnings.append(f"check {c['id']}: " + (f"step {c['step']!r} is not a loop step" if c["step"] else
                                                        "no `step`") + f"; placed on {s['step']}")
                c["step"] = s["step"]
    run_track = {}
    for t in tracks:
        t["primary"] = t["id"] == primary
        t["stages"] = [s["id"] for s in stages if s["track"] == t["id"]]
        t["steps"] = {x: [c["id"] for s in stages if s["track"] == t["id"] for c in s["checks"] if c["step"] == x]
                      for x in loop}
        for r in t["runs"]:
            if r in run_track and run_track[r] != t["id"]:
                warnings.append(f"run {r!r} is listed under tracks {run_track[r]!r} and {t['id']!r}; "
                                f"kept under {run_track[r]!r}")
            run_track.setdefault(r, t["id"])
    return {"version": raw.get("version"), "history": raw.get("history", []), "stages": stages,
            "tracks": tracks, "primary": primary, "loop_ids": loop, "run_track": run_track, "warnings": warnings,
            "hypotheses": [h for h in raw.get("hypotheses", []) if isinstance(h, dict)],
            "index": {c["id"]: (s, c) for s in stages for c in s["checks"]}}


def place(val: dict, stage: str | None = None, track: str | None = None, step: str | None = None,
          check_id: str | None = None, run_id: str | None = None) -> tuple[str, str | None]:
    """Where a record sits on the page: (its data track, its loop step or 'input', the track's footage).

    - A check's own track and step win (measurements and the events they trigger).
    - A track's container stage (T2) gives that track, at `step` or the step most of its checks sit on.
    - A loop step keeps its place; its track is `track`, else the run's track (validation.yaml `tracks: runs`),
      else the first track. Records written before tracks existed land on the first track.
    """
    if check_id and check_id in val["index"]:
        s, c = val["index"][check_id]
        return s["track"], c["step"]
    s = next((x for x in val["stages"] if x["id"] == stage), None) if stage else None
    good = step if step and (step in val["loop_ids"] or step == INPUT) else None
    if s and s["role"] == "track":
        return s["track"], good or s["step"]
    tid = (track if track in {t["id"] for t in val["tracks"]} else
           val["run_track"].get(str(run_id or "")) or val["primary"])
    return tid, good or (s["id"] if s else None)


def brain_key(brain: str | None) -> str:
    """Normalise a brain name: every Claude stand-in is 'claude', every Argon run is 'argon'."""
    b = re.sub(r"\s+", "-", str(brain or "").strip().lower())
    if not b:
        return ""
    if b.startswith("claude") or b in ("stand-in", "standin"):
        return "claude"
    if b.startswith("argon") or b == "gemini-4-argon":
        return "argon"
    return re.sub(r"[^a-z0-9._-]", "", b)[:32] or "unknown"


def judge(direction: str, target, value: float) -> str:
    if direction == ">=":
        return "pass" if value >= target else "fail"
    if direction == "<=":
        return "pass" if value <= target else "fail"
    return "track"


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    if float(v).is_integer() and abs(v) < 1e15:
        return f"{int(v):,}"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.3g}"


def target_text(check: dict) -> str:
    d = check["direction"]
    return f"{DIR_SYM[d]} {fmt(check['target'])}" if d in GATED else "track"


def brain_view(records: list[dict], val: dict) -> dict:
    """Records of ONE brain (chronological) -> per-check and per-stage state."""
    per: dict[str, list] = defaultdict(list)
    for r in records:
        if r.get("check_id") in val["index"]:
            per[r["check_id"]].append(r)
    checks = {}
    for cid, rs in per.items():
        state = rs[-1].get("state", "track")
        ever = any(r.get("state") == "pass" for r in rs)
        checks[cid] = {"state": state, "ever_passed": ever, "regressed": state == "fail" and ever,
                       "latest": rs[-1], "records": rs}
    stages = {s["id"]: stage_view(s, checks) for s in val["stages"]}
    return {"checks": checks, "stages": stages}


def stage_view(stage: dict, checks: dict) -> dict:
    ids = [c["id"] for c in stage["checks"]]
    gated = [c["id"] for c in stage["checks"] if c["direction"] in GATED]
    measured = [c for c in ids if c in checks]
    g = [checks[c] for c in gated if c in checks]
    if not measured:
        state = "not_started"
    elif any(v["regressed"] for v in g):
        state = "regressed"
    elif any(v["state"] == "fail" for v in g):
        state = "failing"
    elif gated and len(g) == len(gated):
        state = "passing"
    else:
        state = "in_progress"
    latest = sorted((checks[c]["latest"] for c in measured), key=lambda r: r.get("t", 0))
    return {"state": state, "passed": sum(v["state"] == "pass" for v in g), "gated": len(gated),
            "measured": len(measured), "total": len(ids),
            "failing": [c for c in gated if c in checks and checks[c]["state"] == "fail"],
            "regressed": [c for c in gated if c in checks and checks[c]["regressed"]],
            "unmeasured": [c for c in ids if c not in checks],
            "last_t": latest[-1].get("t") if latest else None,
            "by": list(dict.fromkeys(r.get("by") for r in reversed(latest) if r.get("by")))[:3]}


# ------------------------------------------------------------------------------- records

def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for ln in path.read_text(errors="replace").splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue  # a torn line never takes the page down
    return out


def by_brain(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list] = defaultdict(list)
    for r in sorted(records, key=lambda r: r.get("t", 0)):
        out[brain_key(r.get("brain")) or "unknown"].append(r)
    return dict(out)


@contextmanager
def locked():
    """Serialise writers (parallel watchers) across processes."""
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATUS_DIR / ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")


def iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


# ------------------------------------------------------------------------------- page inputs

_BUNDLE: dict | None = None  # abs path -> bundled name, set only while build(bundle=True) runs


def href(p: Path) -> str:
    """Link from runs/status/index.html to a file: relative, or a copy under runs/status/media/ when bundling."""
    p = Path(p)
    media = (STATUS_DIR / "media").resolve()
    if _BUNDLE is not None and p.is_file() and not p.resolve().is_relative_to(media):
        key = str(p.resolve())
        name = _BUNDLE.get(key)
        if not name:
            name = f"{hashlib.sha1(key.encode()).hexdigest()[:8]}-{p.name}"
            dst = media / name
            if not dst.exists() or dst.stat().st_size != p.stat().st_size or dst.stat().st_mtime < p.stat().st_mtime:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
            _BUNDLE[key] = name
        return "media/" + name
    return os.path.relpath(p, STATUS_DIR).replace(os.sep, "/")


def ref_path(ref: str) -> Path | None:
    """'runs/x.jsonl:12' or 'file.md#part' -> the file's path (None for URLs)."""
    if URL_RE.match(ref):
        return None
    path = ref.split("#", 1)[0]
    m = re.match(r"^(.*?):\d+(-\d+)?$", path)  # 'file.jsonl:12' cites a line
    path = m.group(1) if m else path
    return Path(path) if path.startswith("/") else REPO / path


def evidence_item(ref: str) -> dict:
    ref = str(ref).strip()
    p = ref_path(ref)
    if p is None:
        return {"ref": ref, "href": ref, "exists": True, "img": False}
    exists = p.exists()
    return {"ref": ref, "href": href(p) if exists else "", "exists": exists,
            "img": exists and p.suffix.lower() in IMAGE_EXT and p.is_file()}


def media_file(ref: str) -> dict:
    p = ref_path(ref)
    ext = Path(ref).suffix.lower()
    typ = "video" if ext in VIDEO_EXT else "image" if ext in IMAGE_EXT else "other"
    if p is None:
        return {"ref": ref, "href": ref, "exists": True, "type": typ}
    exists = p.is_file()
    out = {"ref": ref, "href": href(p) if exists else "", "exists": exists, "type": typ}
    if exists and typ == "video":  # a still with the same stem (cut_clip writes one) is the video's poster
        poster = next((q for q in (p.with_suffix(".jpg"), p.with_suffix(".png")) if q.is_file()), None)
        if poster:
            out["poster"] = href(poster)
    return out


EXPLAIN_DIR = "runs/explain/"  # cut_clip's folder: a published clip from here is footage, not a result


def media_place(val: dict, m: dict) -> tuple[str, str | None]:
    """A media item's (track, step); a clip cut for people (runs/explain/) published without a step is the footage."""
    track, step = place(val, m.get("stage"), m.get("track"), m.get("step"), None, m.get("run_id"))
    if not m.get("step") and str((m.get("paths") or [""])[0]).startswith(EXPLAIN_DIR):
        step = INPUT
    return track, step


def activity(val: dict) -> dict:
    """Work on record per brain, data track and loop step, scored or not: the events and media items placed there.

    brain -> track -> step -> {"events": n, "media": n, "t": latest}. It judges nothing: a step's `state` still comes
    only from its checks. The page shows a step with work but no measured check as in progress, not scored yet.
    Brains never mix: a record with no brain counts for no brain, and a track's footage (step 'input') is no step's work.
    """
    out: dict = {}

    def add(rec: dict, track: str, step: str | None, kind: str) -> None:
        b = brain_key(rec.get("brain"))
        if not b or step not in val["loop_ids"]:
            return
        a = out.setdefault(b, {}).setdefault(track, {}).setdefault(step, {"events": 0, "media": 0, "t": 0})
        a[kind] += 1
        a["t"] = max(a["t"], num(rec.get("t")) or 0)

    for e in read_jsonl(events_path()):
        add(e, *place(val, e.get("stage"), e.get("track"), e.get("step"), e.get("check_id"), e.get("run_id")), "events")
    for m in read_jsonl(media_path()):
        add(m, *media_place(val, m), "media")
    return out


def with_activity(sv: dict, a: dict | None) -> dict:
    """A step's check-based view (stage_view) plus the work on record there. `state` stays the checks' verdict, as in
    status_summary; `unscored` marks work with no measured check yet, which the page shows as ◐ not scored yet."""
    ev, md = int((a or {}).get("events") or 0), int((a or {}).get("media") or 0)
    return {**sv, "events": ev, "media": md, "act_t": (a or {}).get("t") or None,
            "unscored": not sv["measured"] and ev + md > 0}


def media_items(val: dict) -> list[dict]:
    out = []
    for m in sorted(read_jsonl(media_path()), key=lambda m: -m.get("t", 0)):
        metrics = []
        for k, v in (m.get("metrics") or {}).items():
            row = {"k": k, "v": num(v) if not isinstance(v, str) else v[:60]}
            if k in val["index"] and isinstance(row["v"], (int, float)):
                c = val["index"][k][1]
                row.update(name=c["name"], target=target_text(c), state=judge(c["direction"], c["target"], row["v"]))
            metrics.append(row)
        track, step = media_place(val, m)
        out.append({"id": m.get("id"), "t": m.get("t"), "stage": m.get("stage"), "track": track, "step": step,
                    "title": m.get("title", ""),
                    "kind": m.get("kind", "image"), "caption": m.get("caption", ""), "metrics": metrics,
                    "labels": m.get("labels") or [], "featured": bool(m.get("featured")),
                    "brain": brain_key(m.get("brain")), "run_id": m.get("run_id") or "", "by": m.get("by") or "",
                    "files": [media_file(r) for r in m.get("paths", [])]})
    return out


STORY_DEFAULT = """\
# The text of runs/status/index.html: the cover flowchart, one chapter per section, then the ledger.
# Edit freely; the page picks it up on its next rebuild (any record_check / record_event / publish_media,
# or `.venv/bin/python -m harness.status`). Rules and limits: docs/design/ui-protocol.md. Top level:
#   brand            the project name, shown as the text logo
#   title, subtitle  the cover headline and its one-sentence explanation
#   input, output    the flowchart's two ends: label (shown in capitals), note (under it), explain (below)
#   footer           the sim-only line at the end of the ledger
#   builders         the last page: names (who built it) and note (one line above the wordmark)
#   tracks           per data track (validation.yaml `tracks`; the switcher at the flowchart's start shows one at a
#                    time): label (at most 14 characters; default the track's name up to its bracket), and optional
#                    overrides of input (note, explain), output (note) and sections ({A: {title, claim, numbers, read,
#                    explain, feature}}). Everything else in this file is the first track's (REASSEMBLE) text. Another
#                    track never falls back to it: without an override a step shows its node name and a line saying
#                    how many of the track's checks sit there. Status, scope and why come from validation.yaml
# Per section (one chapter each, in this order; main sections are numbered 1, 2, 3 ... in the page):
#   stages   validation.yaml loop stages the section covers; their role (main / feedback / integrity) places the
#            node. A track's container stage (T2) is never listed: its checks name their own step
#   node    the flowchart label, at most 18 characters, shown in capitals
#   short    the label in the page navigator, one word (default: node)
#   parts    for a section with several stages: one word per stage, e.g. {B0: cameras}
#   in, out  arrow labels: what the step takes and what it hands on, at most 2 short lines of 9 characters
#            (`in` defaults to the previous main section's `out`)
#   title    the chapter headline, at most 8 words; claim: one sentence the numbers below must back up
#   read     optional paragraphs for the chapter's method sheet (Markdown: **bold**, `code`, lists)
#   numbers  up to two check ids shown big (default: each stage's first gated check)
#   feature  media id or exact title shown large (default: latest featured item, else latest item)
#   show     extra tiles: cost
#   explain  an explainer card opened from the flowchart: title, text (Markdown) and clips. A clip is
#            title, caption, tag (success / failure), paths (one mp4 per camera, played in sync) and labels.
#            Cut clean clips from a recording with the cut_clip tool.
# Skill tags under each node are not set here: each skill's SKILL.md frontmatter `steps:` places it
# (docs/design/ui-protocol.md, section 2).
# loops: the two self-optimization loops (docs/design/vision-draft.md), drawn as soft regions behind their nodes; hover
#   or focus shows a panel, a tap opens it as a sheet. Per loop:
#   id, label     the region's label (shown in capitals, at most 14 characters)
#   steps         validation.yaml loop stages it encloses
#   objective     one line, at most 56 characters (the same for every track unless a track overrides it)
#   headline      up to 5 checks, each with a short label (at most 18 characters), shown against target
#   curve         the check whose records give the region's trend glyph (default: the first headline check)
#   better        {check: lower|higher} for a tracked (no target) curve check
#   playground    what the loop runs on for this track, at most 60 characters
#   given         for a track whose source supplies the loop (a simulator's exact scene): one line, at most 60 characters
#   Another track's loop text goes under tracks: <id>: loops: <loop id>: and never falls back to this one's headline or
#   playground. What each step optimized (lessons, versions, tools, routines, frozen runs) comes from the records, the
#   skills' Lessons and the run folders, not from here.
brand: Bot18
title: From footage to a twin to a policy
subtitle: Argon reads real robot footage, builds a simulated twin shaped around the task, and teaches itself the task inside it.
input:
  label: Real footage
  note: Franka arm, force sensor, 3 cameras
  explain:
    title: The REASSEMBLE recordings
    text: |
      REASSEMBLE (Sliwowski et al., TU Wien, 2025) is a public dataset of a Franka arm, teleoperated by a person,
      taking parts out of and putting parts into the NIST Task Board 1.

      - **149 recordings, about 13 hours** at 30 fps, with 4,551 labelled demonstrations, of which 516 failed.
      - **Every recording has three cameras**: two fixed webcams (hama1, hama2) and a wrist camera, all 640×480.
        It also has an event camera, a force/torque sensor at the flange, joint states and the gripper.
      - **Every segment has a label**: a verb (Pick, Insert, Remove or Place), one of 17 parts (connectors,
        gears, pegs and a nut) and whether it succeeded.
      - **Inserting is the hard part.** In the episodes on disk, 39 of 143 inserts (27%) failed.

      Argon reads these recordings blind in step 1. In step 2 the twin has to succeed and jam where the real
      robot did. Sources: `docs/research/reassemble-data-notes.md`.
    clips:
      - title: Insert round peg 1
        caption: Episode 1, 42.6–60.8 s. The peg goes in on the first try; this is the attempt the twin replays in step 2.
        tag: success
        paths: [runs/explain/reassemble/ep001_insert_round_peg_1_hama1.mp4, runs/explain/reassemble/ep001_insert_round_peg_1_hama2.mp4, runs/explain/reassemble/ep001_insert_round_peg_1_hand.mp4]
        labels: [fixed camera 1, fixed camera 2, wrist camera]
      - title: Thread the nut, and fail
        caption: Episode 1, 72.2–95.0 s. The M16 nut ("bolt 4") does not go on. The operator picks it up again and succeeds on the next try (105–125 s).
        tag: failure
        paths: [runs/explain/reassemble/ep001_insert_bolt_4_fail_hama1.mp4, runs/explain/reassemble/ep001_insert_bolt_4_fail_hama2.mp4, runs/explain/reassemble/ep001_insert_bolt_4_fail_hand.mp4]
        labels: [fixed camera 1, fixed camera 2, wrist camera]
      - title: Pull the small gear, and fail
        caption: Episode 0, 37.7–52.0 s. The gear turns freely on its shaft but will not come off; all four small-gear attempts in this recording fail.
        tag: failure
        paths: [runs/explain/reassemble/ep000_remove_small_gear_fail_hama1.mp4, runs/explain/reassemble/ep000_remove_small_gear_fail_hama2.mp4, runs/explain/reassemble/ep000_remove_small_gear_fail_hand.mp4]
        labels: [fixed camera 1, fixed camera 2, wrist camera]
      - title: Remove the waterproof connector
        caption: Episode 0, 3.3–16.7 s. A clean remove on the first try.
        tag: success
        paths: [runs/explain/reassemble/ep000_remove_waterproof_hama1.mp4, runs/explain/reassemble/ep000_remove_waterproof_hama2.mp4, runs/explain/reassemble/ep000_remove_waterproof_hand.mp4]
        labels: [fixed camera 1, fixed camera 2, wrist camera]
tracks:
  so101:
    label: SO-101
    input:
      note: SO-101 arm, 2 cameras, no force
      explain:
        title: The SO-101 recordings
        text: |
          `lerobot/svla_so101_pickplace` is a public LeRobot dataset. A person drives an SO-101 follower arm with a
          leader arm to put a pink LEGO brick into a clear box.

          - **50 recordings, 11,939 frames** at 30 fps, about 6.6 minutes in all.
          - **Two fixed cameras**, up and side, both 640×480, with no calibration file.
          - **Six joint positions and the leader arm's six commands** for every frame.
          - **No force sensor, no CAD and no step labels**: every recording has the same task string,
            "pink lego brick into the transparent box".
          - Episodes 0–34 are for building the twin and 35–49 are the gate set, frozen on 25 Oct 2026.

          REASSEMBLE gives CAD and force but no commands; SO-101 gives commands but no CAD or force. Sources:
          `data/so101/svla_so101_pickplace/meta/info.json`, `playbook/splits_so101.json`.
    sections:
      A:
        title: No step labels to read on SO-101
        claim: Every recording has the same task string and no step labels, so there is nothing to score reading against.
      B:
        title: A twin from two cameras and joint logs
        claim: With no CAD or force sensor, the robot calibrates both cameras, and the logged commands, replayed in the twin, must land the brick in the box.
        numbers: [T2.kp_px, T2.replay_success]
      C:
        title: Task tools wait for the REASSEMBLE loop
        claim: Not started. SO-101 is a reconstruction and replay track; it learns only after the REASSEMBLE loop has run once.
      D:
        title: Practice waits for the REASSEMBLE loop
        claim: Not started. SO-101 practises only after the REASSEMBLE loop has run once.
      E:
        title: Compiling waits for the REASSEMBLE loop
        claim: Not started. Routines come from practice, and SO-101 practises only after the REASSEMBLE loop has run once.
      F:
        title: No physics questions from SO-101 yet
        claim: Questions come from practice, which has not started on SO-101.
      G:
        title: No integrity checks written for SO-101
        claim: SO-101 has no exam set, only train and gate episodes, and no integrity check is written for it yet.
    loops:
      twin:
        headline:
          T2.kp_px: camera keypoints
          T2.two_view_px: two-view px
          T2.grasp_object_mm: brick at grasp
          T2.replay_success: replay success
          T2.replay_negative_control: shifted-brick ctrl
        curve: T2.replay_success
        playground: SO-101 recordings, 2 cameras, joint logs
      task:
        playground: Not started; learns after the REASSEMBLE loop
  footage:
    label: Footage only
    input:
      note: human or egocentric video
    loops:
      twin:
        playground: Human or egocentric video, no robot logs
      task:
        playground: Not started
    sections:
      A:
        title: Read video with no robot logs
        claim: Not started. With no joints, force or calibration, the video itself is the only truth to read against.
      B:
        title: Rebuild the scene from video alone
        claim: Not started. The plan is to rebuild from video with outside hand-pose and scale cues, checked by render-and-compare and physical plausibility.
output:
  label: Real arm
  note: sim only until a real arm confirms
footer: All results are sim-only until a real arm confirms them.
builders:
  names: [Shiran Wang, Xinrui Yi]
  note: Built in a day on Gemini 4 Argon. Argon is element 18.
loops:
  - id: twin
    label: Twin loop
    steps: [A, B0, B1, B2]
    objective: The twin explains observations it was not fitted on.
    headline:
      B2.dist_seat_rate_err: seat-rate error
      B2.dist_force_ks: force KS
      B2.dist_negative_control: wrong-twin control
      A.segment_f1: segment F1
      A.success_bacc: success bacc
    playground: REASSEMBLE recordings, Franka, force, 3 cameras
  - id: task
    label: Task loop
    steps: [C, D, E]
    objective: Success in a frozen budget; cost and brain share fall.
    headline:
      D.success_rate: success
      D.tokens_per_success: tokens per success
      D.brain_share: brain share
    better: {D.tokens_per_success: lower, D.brain_share: lower}
    playground: The peg twin built in step 2; LIBERO is next
sections:
  - id: A
    stages: [A]
    node: Read footage
    short: Read
    in: recordings
    out: labelled steps
    title: Argon reads the footage
    claim: From long recordings of real assembly, Argon labels every step and tells the failed attempts from the good ones.
    numbers: [A.segment_f1, A.success_bacc]
  - id: B
    stages: [B0, B1, B2]
    node: Build twin
    short: Twin
    parts: {B0: cameras, B1: contact, B2: replay}
    out: twin
    title: It builds a twin that fails the way reality fails
    claim: The twin's cameras line up with the real ones, and replayed attempts succeed and jam where the real robot did.
    numbers: [B0.overlay_hama1, B2.outcome_bacc]
  - id: C
    stages: [C]
    node: Task tools
    short: Tools
    out: tools
    title: Tools come from the twin
    claim: Argon turns the twin's geometry into task-level tools like insert(part, hole), and each tool proves itself in the twin first.
    numbers: [C.tool_success]
  - id: D
    stages: [D]
    node: Practise
    short: Practise
    out: working moves
    title: It practises on starts it has never seen
    claim: Argon thinks only at decision points while fast routines run the contact, and success climbs with practice.
    numbers: [D.success_rate, D.tokens_per_success]
  - id: E
    stages: [E]
    node: Compile
    short: Compile
    out: routines
    title: Practice compiles into routines
    claim: Moves that work become routines that run without Argon, so every attempt costs fewer tokens and less time.
    numbers: [E.routine_success, E.transfer_cost_ratio]
    show: [cost]
  - id: F
    stages: [F]
    node: Physics questions
    short: Feedback
    in: practice
    out: answers fix the twin
    title: Physics questions go back to the twin
    claim: When practice leans on physics the replays never checked, the question goes back to the real recordings.
    numbers: [F.open_questions]
  - id: G
    stages: [G]
    node: Integrity
    short: Integrity
    in: every number
    out: verdicts
    title: Nothing was faked
    claim: The exam recordings stayed sealed, the simulator was never bent, and the harness, not the agent, judged every number.
    numbers: [G.exam_reads, G.cheat_violations]
"""


def _explain(raw) -> dict | None:
    """An explainer card: title, Markdown text and clips (each clip is one moment, one mp4 per camera)."""
    if not isinstance(raw, dict):
        return None
    clips = []
    for c in raw.get("clips") or []:
        if not isinstance(c, dict):
            continue
        paths = c.get("paths") or ([c["path"]] if c.get("path") else [])
        files = [media_file(str(x)) for x in paths]
        if files:
            clips.append({"title": str(c.get("title") or ""), "caption": str(c.get("caption") or ""),
                          "tag": str(c.get("tag") or ""), "labels": [str(x) for x in c.get("labels") or []],
                          "files": files})
    return {"title": str(raw.get("title") or ""), "text_html": md_html(str(raw.get("text") or "")), "clips": clips}


def _end(raw, default_label: str) -> dict:
    raw = raw if isinstance(raw, dict) else {"label": raw} if raw else {}
    return ({k: str(raw.get(k) or "") for k in ("label", "out", "note")}
            | {"label": str(raw.get("label") or default_label), "explain": _explain(raw.get("explain"))})


def _builders(raw) -> dict:
    raw = raw if isinstance(raw, dict) else {"names": raw} if raw else {}
    names = raw.get("names") or []
    return {"names": [str(n) for n in (names if isinstance(names, list) else [names])], "note": str(raw.get("note") or "")}


def load_story(val: dict) -> dict:
    path = story_path()
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(STORY_DEFAULT)
    error = None
    try:
        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw.get("sections"), list):
            raise ValueError("story.yaml needs a 'sections' list")
    except Exception as e:  # a typo in the story must not take the status page down
        error, raw = f"{rel(path)}: {e}; showing the default story", yaml.safe_load(STORY_DEFAULT)
    by_id = {s["id"]: s for s in val["stages"]}
    loop_ids = [s["id"] for s in val["stages"] if s["role"] != "track"]
    first = {s["id"]: (next((c for c in s["checks"] if c["direction"] in GATED), None) or (s["checks"] or [None])[0])
             for s in val["stages"]}
    warnings, sections = [], []
    for sec in raw["sections"]:
        if not isinstance(sec, dict):
            continue
        listed = sec.get("stages") or [sec.get("id")]
        stages = [x for x in listed if x in loop_ids]
        if moved := [x for x in listed if x in by_id and by_id[x]["role"] == "track"]:
            # a track's container stage is never a step: its checks name their own step and show in that track's view
            warnings.append(f"section {sec.get('id')}: {moved} belong to data track "
                            f"{by_id[moved[0]]['track']!r}; their checks show on their own steps in its view")
            if not stages:
                continue
        nums = [n for n in (sec.get("numbers") or []) if n in val["index"] and val["index"][n][0]["role"] != "track"][:2]
        nums = nums or [first[x]["id"] for x in stages if first.get(x)][:2]
        node = str(sec.get("node") or (by_id[stages[0]]["name"] if stages else sec.get("id") or ""))
        parts = sec.get("parts") if isinstance(sec.get("parts"), dict) else {}
        sections.append({"id": str(sec.get("id") or "-".join(stages)), "stages": stages,
                         "role": by_id[stages[0]]["role"] if stages else "main", "node": node,
                         "short": str(sec.get("short") or node), "parts": {str(k): str(v) for k, v in parts.items()},
                         "explain": _explain(sec.get("explain")),
                         "in": str(sec.get("in") or ""), "out": str(sec.get("out") or ""),
                         "title": str(sec.get("title") or ""), "claim": str(sec.get("claim") or ""),
                         "read_html": md_html(str(sec.get("read") or "")),
                         "numbers": nums, "feature": sec.get("feature"), "show": list(sec.get("show") or [])})
    labels, over = {}, {}  # per data track: the switcher label and text overrides; the tracks come from validation.yaml
    for tid, t in (raw.get("tracks") or {}).items() if isinstance(raw.get("tracks"), dict) else []:
        if str(tid) not in {x["id"] for x in val["tracks"]}:
            warnings.append(f"tracks: {tid!r} is not a data track in validation.yaml")
            continue
        over[str(tid)] = t if isinstance(t, dict) else {}
        if label := str((t.get("label") if isinstance(t, dict) else t) or "").strip():
            labels[str(tid)] = label
    inp, outp = _end(raw.get("input"), "Real footage"), _end(raw.get("output"), "Real arm")
    loops = _loops(raw.get("loops"), val, sections, warnings)
    cards = _cards(raw.get("cards"), {t["id"] for t in val["tracks"]}, {s["id"] for s in sections}, warnings)
    return {"brand": str(raw.get("brand") or "Bot18"), "title": str(raw.get("title") or ""),
            "subtitle": str(raw.get("subtitle") or ""), "footer": str(raw.get("footer") or ""), "cards": cards,
            "frame_note": str(raw.get("frame_note") or ""),
            "builders": _builders(raw.get("builders")), "input": inp, "output": outp,
            "sections": sections, "track_labels": labels,
            "loops": [{k: lp[k] for k in ("id", "label", "steps", "objective")} for lp in loops],
            "views": track_views(val, sections, inp, outp, labels, over, warnings, loops),
            "warnings": warnings, "error": error, "path": rel(path)}


# story.yaml `cards`: a few named notes the presenter opens from the cover (and from a step page when they name a step):
# "Not proven yet", a comparison with a published method. Each: id, label (the link, at most 28 characters), title, text
# (Markdown), tracks (default every track), step (optional section id). Pending numbers are written as pending, never guessed.
def _cards(raw, track_ids: set, sec_ids: set, warnings: list) -> list[dict]:
    out = []
    for i, c in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(c, dict) or not c.get("id"):
            warnings.append(f"cards[{i}]: needs an id")
            continue
        tr = c.get("tracks")
        tracks = [str(x) for x in ([tr] if isinstance(tr, str) else tr or [])]
        if bad := [x for x in tracks if x not in track_ids]:
            warnings.append(f"cards.{c['id']}.tracks: {bad} are not data tracks")
        step = str(c.get("step") or "")
        if step and step not in sec_ids:
            warnings.append(f"cards.{c['id']}.step: {step!r} is not a story section")
        label = str(c.get("label") or c.get("title") or c["id"])
        if len(label) > 28:
            warnings.append(f"cards.{c['id']}.label: longer than 28 characters")
        out.append({"id": str(c["id"]), "label": label, "title": str(c.get("title") or label),
                    "kicker": str(c.get("kicker") or ""), "text_html": md_html(str(c.get("text") or "")),
                    "tracks": [x for x in tracks if x in track_ids], "step": step if step in sec_ids else "",
                    "cover": c.get("cover", True) is not False,
                    "figures": [media_file(str(x)) for x in ([c["figure"]] if isinstance(c.get("figure"), str) else c.get("figure") or [])]})
    return out


def _first_check(val: dict, ids: list[str]) -> str | None:
    gated = [c for c in ids if val["index"][c][1]["direction"] in GATED]
    return (gated or ids or [None])[0]


# ------------------------------------------------------------------------------- the two self-optimization loops
# story.yaml `loops` names the twin loop and the task loop: the steps each encloses, its objective, the checks that
# measure it (headline), the check whose records draw its learning curve (curve) and what it runs on (playground). The
# overview draws each as a soft region behind its nodes; hovering or focusing it shows a panel. Per data track, like a
# section's text: another track never falls back to the first track's headline checks or playground.
# docs/design/ui-protocol.md, sections 1, 3, 5 and 6.
LOOP_LIMITS = {"label": 14, "objective": 56, "playground": 60, "given": 60, "headline": 18}
MAX_HEADLINE = 5


def _headline_items(raw) -> list[tuple[str, str]]:
    """`headline` as a mapping {check id: short label}, a list of ids, or a list of {id: label}: [(id, label)], in order."""
    if isinstance(raw, dict):
        return [(str(k), str(v or "")) for k, v in raw.items()]
    out: list[tuple[str, str]] = []
    for x in raw if isinstance(raw, list) else [raw] if raw else []:
        if isinstance(x, dict):
            out += [(str(k), str(v or "")) for k, v in x.items()]
        elif x:
            out.append((str(x), ""))
    return out


def _loops(raw, val: dict, sections: list, warnings: list) -> list[dict]:
    """The loop definitions: id, label, steps (loop stages, in loop order), objective, and the first track's raw text."""
    out = []
    shown = {x for s in sections for x in s["stages"]}
    for i, lp in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(lp, dict) or not lp.get("id"):
            warnings.append(f"loops[{i}]: needs an id")
            continue
        lid, steps = str(lp["id"]), [str(x) for x in lp.get("steps") or []]
        if bad := [x for x in steps if x not in val["loop_ids"] or x not in shown]:
            warnings.append(f"loops.{lid}.steps: {bad} are not loop steps on the page")
        steps = [x for x in val["loop_ids"] if x in steps and x in shown]
        if not steps:
            continue
        for k in ("label", "objective"):
            if len(str(lp.get(k) or "")) > LOOP_LIMITS[k]:
                warnings.append(f"loops.{lid}.{k}: longer than {LOOP_LIMITS[k]} characters")
        out.append({"id": lid, "label": str(lp.get("label") or lid), "steps": steps,
                    "objective": str(lp.get("objective") or ""), "raw": lp})
    return out


def _loop_view(lp: dict, raw: dict, t: dict, val: dict, where: str, warnings: list) -> dict:
    """One loop as one track shows it: its objective, headline checks [[id, label]], curve check, playground, and
    `given` (a loop the track's source supplies, e.g. a simulator's exact scene: no checks, a line saying so).
    Headline checks must be the track's own checks on the loop's steps; without any, the track's checks there, gated
    first, up to five."""
    mine = [c for x in lp["steps"] for c in t["steps"].get(x, [])]
    items = _headline_items(raw.get("headline"))
    if bad := [cid for cid, _ in items if cid not in mine]:
        warnings.append(f"{where}.headline: {bad} are not {t['id']} checks on the loop's steps")
    items = [(cid, lab) for cid, lab in items if cid in mine][:MAX_HEADLINE]
    if not items and "headline" not in raw:
        gated = [c for c in mine if val["index"][c][1]["direction"] in GATED]
        items = [(c, "") for c in (gated + [c for c in mine if c not in gated])][:MAX_HEADLINE]
    for cid, lab in items:
        if len(lab) > LOOP_LIMITS["headline"]:
            warnings.append(f"{where}.headline.{cid}: label longer than {LOOP_LIMITS['headline']} characters")
    head = [[cid, lab or val["index"][cid][1]["short"].replace("_", " ")] for cid, lab in items]
    curve = str(raw.get("curve") or "")
    if curve and curve not in mine:
        warnings.append(f"{where}.curve: {curve} is not a {t['id']} check on the loop's steps")
    better = raw.get("better") if isinstance(raw.get("better"), dict) else {}
    for k in ("playground", "given"):
        if len(str(raw.get(k) or "")) > LOOP_LIMITS[k]:
            warnings.append(f"{where}.{k}: longer than {LOOP_LIMITS[k]} characters")
    return {"objective": str(raw.get("objective") or lp["objective"]), "headline": head,
            "curve": curve if curve in mine else head[0][0] if head else None,
            "better": {str(k): str(v) for k, v in better.items() if str(k) in mine and str(v) in ("lower", "higher")},
            "playground": str(raw.get("playground") or ""), "given": str(raw.get("given") or ""), "checks": len(mine)}


def track_views(val: dict, sections: list, inp: dict, outp: dict, labels: dict, over: dict, warnings: list,
                loops: list | None = None) -> dict:
    """Each data track's text for the page: its footage end, its output end, and per section title, claim, numbers,
    method text, explainer and featured media. The first track's text is story.yaml's top level. Another track uses
    its `tracks: <id>:` overrides and never falls back to the first track's claims: without one, a section shows its
    node name and a line saying how many of the track's checks sit on it."""
    views = {}
    sec_ids = {s["id"] for s in sections}
    for t in val["tracks"]:
        o = over.get(t["id"]) or {}
        label = labels.get(t["id"]) or t["title"]
        so = o.get("sections") if isinstance(o.get("sections"), dict) else {}
        if bad := sorted(str(k) for k in so if str(k) not in sec_ids):
            warnings.append(f"tracks.{t['id']}.sections: {bad} are not story sections")
        secs = {}
        for sec in sections:
            ov = so.get(sec["id"]) if isinstance(so.get(sec["id"]), dict) else {}
            mine = [c for x in sec["stages"] for c in t["steps"].get(x, [])]  # this track's checks on the section
            if t["primary"]:
                base = {k: sec[k] for k in ("title", "claim", "numbers", "read_html", "explain", "feature")}
            else:
                n = len(mine)
                base = {"title": sec["node"][:1].upper() + sec["node"][1:].lower(),
                        "claim": f"{n} check{'' if n == 1 else 's'} on this step for {label}." if n
                        else f"No checks on this step yet for {label}.", "read_html": "", "explain": None, "feature": None,
                        "numbers": [x for x in (_first_check(val, t["steps"].get(s, [])) for s in sec["stages"]) if x][:2]}
            nums = [str(x) for x in ov.get("numbers") or []]
            if bad := [x for x in nums if x not in mine]:
                warnings.append(f"tracks.{t['id']}.sections.{sec['id']}.numbers: {bad} are not "
                                f"{t['id']} checks on this step")
            nums = [x for x in nums if x in mine][:2]
            secs[sec["id"]] = {"title": str(ov.get("title") or base["title"]), "claim": str(ov.get("claim") or base["claim"]),
                               "numbers": nums or base["numbers"],
                               "read_html": md_html(str(ov["read"])) if ov.get("read") else base["read_html"],
                               "explain": _explain(ov["explain"]) if ov.get("explain") else base["explain"],
                               "feature": ov.get("feature") or base["feature"], "checks": len(mine),
                               **({} if t["primary"] else _step_labels(ov, f"tracks.{t['id']}.sections.{sec['id']}", warnings))}
        oi = o.get("input") if isinstance(o.get("input"), dict) else {}
        oo = o.get("output") if isinstance(o.get("output"), dict) else {}
        if t["primary"]:  # the first track's ends are story.yaml's top level, never overridden here
            oi, oo = {k: v for k, v in oi.items() if k not in ("label", "out")}, {k: v for k, v in oo.items() if k != "label"}
        ol = o.get("loops") if isinstance(o.get("loops"), dict) else {}
        if bad := sorted(str(k) for k in ol if str(k) not in {lp["id"] for lp in loops or []}):
            warnings.append(f"tracks.{t['id']}.loops: {bad} are not loops in story.yaml")
        lv = {}
        for lp in loops or []:  # the first track's loop text is the loop's own; another track's comes only from its override
            raw = lp["raw"] if t["primary"] else ol.get(lp["id"]) if isinstance(ol.get(lp["id"]), dict) else {}
            where = f"loops.{lp['id']}" if t["primary"] else f"tracks.{t['id']}.loops.{lp['id']}"
            lv[lp["id"]] = _loop_view(lp, raw, t, val, where, warnings)
        views[t["id"]] = {
            "label": label,
            "input": {"label": str(oi.get("label") or inp["label"]), "out": str(oi.get("out") or inp["out"]),
                      "note": str(oi.get("note") or (inp["note"] if t["primary"] else t["title"] + " recordings")),
                      "explain": _explain(oi["explain"]) if oi.get("explain") else inp["explain"] if t["primary"] else None},
            "output": {**outp, "label": str(oo.get("label") or outp["label"]), "note": str(oo.get("note") or outp["note"]),
                       "explain": outp["explain"] if t["primary"] else None},
            "sections": secs, "loops": lv,
            "anchor": _anchor(o.get("anchor"), t, {s["id"] for s in sections}, f"tracks.{t['id']}.anchor", warnings)}
    return views


# Another track can run the same loop steps as a different chain (LIBERO: read demos, reward from video, tools, trials with
# feedback, compile later). Its sections may then rename the step on the flowchart and in the navigator: node, short,
# in, out; `parts` renames a several-stage step's parts or, set to `none`, shows the step as one; `planned` marks a step
# the track will run later (drawn like a planned skill). The first track's labels never change.
NODE_LIMIT = 18


def _step_labels(ov: dict, where: str, warnings: list) -> dict:
    out = {k: str(ov[k]).strip() for k in ("node", "short", "in", "out") if ov.get(k) is not None and str(ov[k]).strip()}
    if len(out.get("node", "")) > NODE_LIMIT:
        warnings.append(f"{where}.node: longer than {NODE_LIMIT} characters")
    if " " in out.get("short", ""):
        warnings.append(f"{where}.short: one word")
    if "parts" in ov:
        p = ov["parts"]
        out["parts"] = {str(k): str(v) for k, v in p.items()} if isinstance(p, dict) else {}
        out["one_step"] = not isinstance(p, dict)  # `parts: none`: the step's stages show as one step on this track
    if ov.get("planned"):
        out["planned"] = "planned" if ov["planned"] is True else str(ov["planned"])
    if ov.get("pending"):  # a result due soon on a step with nothing on record yet: shown as words, never as a state
        out["pending"] = str(ov["pending"])
    return out


# A track can name an external anchor: a published number its loop is measured against, from docs/knowledge/, and our
# own run of the same protocol, counted live from that run's episode log (one JSON line per finished episode, `outcome`).
# It sits on the frame like the integrity chip; its card holds the comparison. It never reads as our result: the
# published number is the paper's, and our run's count says "so far" and "not scored" until a check is recorded.
def _anchor(raw, t: dict, sec_ids: set, where: str, warnings: list) -> dict | None:
    if not isinstance(raw, dict):
        return None
    pub = raw.get("published") if isinstance(raw.get("published"), dict) else {}
    ours = raw.get("ours") if isinstance(raw.get("ours"), dict) else {}
    step = str(raw.get("step") or "")
    if step and step not in sec_ids:
        warnings.append(f"{where}.step: {step!r} is not a story section")
    run = {}
    if ours.get("episodes"):
        p = ref_path(str(ours["episodes"]))
        rows = read_jsonl(p) if p and p.is_file() else []
        if p and not p.is_file():
            warnings.append(f"{where}.ours.episodes: {ours['episodes']} not found")
        run = {"n": len(rows), "successes": sum(1 for r in rows if str(r.get("outcome")) == "success"),
               "t": p.stat().st_mtime if p and p.is_file() else None}
    srcs = [str(x) for x in ([raw.get("sources")] if isinstance(raw.get("sources"), str) else raw.get("sources") or [])]
    return {"label": str(raw.get("label") or "External anchor"), "step": step,
            "published": {k: str(pub.get(k) or "") for k in ("value", "text", "ref", "detail")},
            "ours": {"text": str(ours.get("text") or ""), "brain": str(ours.get("brain") or ""),
                     "run_id": str(ours.get("run_id") or ""), "planned": int(ours.get("planned") or 0) or None,
                     "detail": str(ours.get("detail") or ""), **run},
            "sources": [evidence_item(x) for x in srcs]}


# ------------------------------------------------------------------------------- Markdown for the method sheets

_PATH_RE = re.compile(r"^((?:runs|docs|playbook|harness|twin)/[\w./@+-]+?)(?::\d+(?:-\d+)?)?$")
_LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")


def _code(c: str) -> str:
    """A code span; a repo path that exists becomes a link (never while bundling: that would copy every cited file)."""
    m = _PATH_RE.match(c.strip())
    if m and _BUNDLE is None and (REPO / m.group(1)).exists():
        link = os.path.relpath(REPO / m.group(1), STATUS_DIR).replace(os.sep, "/")
        return f'<a href="{html.escape(link)}" target="_blank" rel="noopener"><code>{html.escape(c)}</code></a>'
    return f"<code>{html.escape(c)}</code>"


def md_inline(text: str) -> str:
    codes: list[str] = []

    def stash(m):
        codes.append(_code(m.group(1)))
        return f"\x00{len(codes) - 1}\x00"

    t = html.escape(re.sub(r"`([^`]+)`", stash, text), quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', t)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], t)


def md_html(text: str) -> str:
    """The small Markdown subset the playbook uses: paragraphs, nested lists, headings, tables, fences."""
    out: list[str] = []
    para: list[str] = []
    lists: list[list] = []  # open lists: [indent, tag]
    lines = text.splitlines()

    def flush():
        if para:
            out.append("<p>" + md_inline(" ".join(para)) + "</p>")
            para.clear()

    def close(to: int = -1):
        while lists and lists[-1][0] > to:
            out.append(f"</li></{lists.pop()[1]}>")

    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            flush(), close()
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(lines[i + 1:j])) + "</code></pre>")
            i = j + 1
            continue
        if s.startswith("|"):
            flush(), close()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                head = "".join(f"<th>{md_inline(c)}</th>" for c in rows[0])
                body = "".join("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
                out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
            continue
        h = re.match(r"^(#{1,6})\s+(.*)$", s)
        m = _LIST_RE.match(ln)
        if not s:
            flush()
        elif h:
            flush(), close()
            out.append(f"<h5>{md_inline(h.group(2))}</h5>")
        elif m:
            flush()
            ind, tag = len(m.group(1)), "ol" if m.group(2)[0].isdigit() else "ul"
            if lists and ind > lists[-1][0]:
                out.append(f"<{tag}><li>")
                lists.append([ind, tag])
            else:
                close(ind)
                if lists:
                    out.append("</li><li>")
                else:
                    out.append(f"<{tag}><li>")
                    lists.append([ind, tag])
            out.append(md_inline(m.group(3)))
        elif lists and (ln[:1].isspace() or not para):
            if ln[:1].isspace():
                out.append(" " + md_inline(s))
            else:
                close()
                para.append(s)
        else:
            para.append(s)
        i += 1
    flush(), close()
    return "".join(out)


def _section(text: str, name: str) -> str:
    m = re.search(rf"^##\s+{name}\b.*?$(.*?)(?=^##\s|\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else ""


def _frontmatter(text: str) -> dict:
    """A SKILL.md's frontmatter. Descriptions often hold ': ', which YAML rejects, so plain 'key: value' lines are the
    fallback (as in harness/tools/playbook.py), with '[a, b]' read as a list."""
    fm = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not fm:
        return {}
    try:
        d = yaml.safe_load(fm.group(1))
        if isinstance(d, dict):
            return d
    except yaml.YAMLError:
        pass
    out: dict = {}
    for ln in fm.group(1).splitlines():
        m = re.match(r"^([A-Za-z_][\w-]*):[ \t]*(.*?)[ \t]*$", ln)
        if m:
            v = m.group(2)
            out[m.group(1)] = ([x.strip(" '\"") for x in v[1:-1].split(",") if x.strip(" '\"")]
                               if v.startswith("[") and v.endswith("]") else v)
    return out


def skill_doc(path: Path) -> dict:
    """Description, Goal, Approach and Lessons of a SKILL.md, as HTML for the chapter's method sheet."""
    text = path.read_text(errors="replace")
    return {"description": str(_frontmatter(text).get("description") or ""), "goal": md_html(_section(text, "Goal")),
            "approach": md_html(_section(text, "Approach")), "lessons": md_html(_section(text, "Lessons"))}


_TOD = {"morning": 1, "midday": 2, "noon": 2, "afternoon": 3, "evening": 4, "night": 5}
_WHO_RE = re.compile(r"claude|argon|gemini|stand-in|watcher|agent|:", re.I)
_RUN_RE = re.compile(r"\bruns/([\w.-]+)/([\w.-]+)")  # a cited run folder: runs/<brain>/<run_id>/...


def _plain(s: str) -> str:
    """Markdown to one plain line: no code spans, links, emphasis or parentheticals."""
    s = re.sub(r"`[^`]*`", "", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s).replace("**", "").replace("*", "")
    while (t := re.sub(r"\([^()]*\)", "", s)) != s:
        s = t
    s = re.sub(r"\s{2,}", " ", re.sub(r"\s+([,.;:])", r"\1", s))
    return s.strip(" ,.;:-–—")


def _sentence(s: str, n: int = 150) -> str:
    m = re.match(r"(.+?[.!?])(?:\s|$)", s)
    s = (m.group(1) if m else s).rstrip(".")
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


def _lead_sentence(s: str, n: int = 150) -> str:
    """The first sentence of at least four words, after any leading date ('2026-10-25, SO-101 track, continued. (4)
    Supplied-looking priors were wrong again: ...' -> 'Supplied-looking priors were wrong again: ...')."""
    s = re.sub(r"^\s*20\d\d-\d\d-\d\d\b[\s,;:.-]*", "", s)
    for part in re.findall(r".+?(?:[.!?](?=\s|$)|$)", s):
        if len(part.split()) >= 4:
            return _sentence(part.strip(), n)
    return _sentence(s, n)


def skill_lessons(text: str) -> list[dict]:
    """Dated lessons under a SKILL.md's ## Lessons, in file order.

    A lesson is a block that starts at a dated line: '- **2026-10-24** (author): note' from append_lesson, or a dated
    bold paragraph header with sub-bullets. It shows a real run if the block cites a file under runs/. Its headline is
    the note's first sentence, or the block's first bold sentence, or what the header says after the date and author.
    """
    m = re.search(r"^##\s+Lessons\b.*?$(.*?)(?=^##\s|\Z)", text, re.M | re.S)
    section = m.group(1) if m else ""
    starts = list(re.finditer(r"^[ \t]*(?:[-*][ \t]+)?\**[ \t]*(20\d\d-\d\d-\d\d)\b(.*)$", section, re.M))
    out = []
    for i, d in enumerate(starts):
        block, head = section[d.start():starts[i + 1].start() if i + 1 < len(starts) else len(section)], d.group(2)
        paren = re.match(r"\**[ \t]*\(([^)]{1,40})\)", head)
        tod = _TOD.get(paren.group(1).strip().lower(), 0) if paren else 0
        author = paren.group(1).strip() if paren and not tod and "," not in paren.group(1) else None
        if not author:
            c = re.match(r"\**[ \t]*(?:\([^)]*\))?[ \t]*,[ \t]*([^,*():]{1,40})", head)
            author = c.group(1).strip() if c and _WHO_RE.search(c.group(1)) else None
        note = re.match(r"\**[ \t]*\([^)]{1,40}\)[ \t]*:[ \t]*(.*)", head)  # append_lesson's form
        headline = _lead_sentence(_plain(re.split(r"\bEvidence:", note.group(1))[0])) if note else ""
        if not headline:
            bold = (_plain(b) for b in re.findall(r"\*\*(.+?)\*\*", block[len(d.group(0)):]))
            headline = next((_sentence(b) for b in bold if len(b.split()) >= 4), "")
        if not headline:
            gist = _plain(re.split(r"\bEvidence:", head.replace(author or "\0", "", 1))[0])
            headline = _sentence(gist) if len(gist.split()) >= 2 else ""
        out.append({"date": d.group(1), "tod": tod, "pos": d.start(), "author": author, "headline": headline,
                    "real_run": bool(re.search(r"\bruns/", block)), "head": head[:200],
                    "runs": list(dict.fromkeys(_RUN_RE.findall(block)))})
    for i, x in enumerate(out):  # an untimed lesson is later than the timed ones above it on its date (append order)
        x["tod"] = x["tod"] or max([o["tod"] for o in out[:i] if o["date"] == x["date"]] or [0])
    return out


def latest_lesson(lessons: list[dict]) -> dict | None:
    return max(lessons, key=lambda x: (x["date"], x["tod"], x["pos"])) if lessons else None


def _skill_paths() -> dict[str, Path]:
    """Skill name (frontmatter `name`, else the folder) -> its SKILL.md."""
    out: dict[str, Path] = {}
    for p in sorted(SKILLS_DIR.glob("*/SKILL.md")) if SKILLS_DIR.is_dir() else []:
        out.setdefault(str(_frontmatter(p.read_text(errors="replace")).get("name") or p.parent.name), p)
    return out


def skill_status(name: str | None) -> dict | None:
    if not name:
        return None
    path = _skill_paths().get(name) or SKILLS_DIR / name / "SKILL.md"
    if not path.is_file():
        return {"name": name, "status": "missing", "lessons": 0}
    text = path.read_text(errors="replace")
    dated = skill_lessons(text)
    entries = dated or re.findall(r"^\s*[-*]\s+\S", _section(text, "Lessons"), re.M)
    real = [d for d in dated if d["real_run"]]
    last = latest_lesson(dated)
    return {"name": name, "status": "tested" if real else "untested", "lessons": len(entries),
            "dated": len(dated), "real_runs": len(real), "href": href(path), "doc": skill_doc(path),
            "last_lesson": last["date"] if last else None, "last_author": last["author"] if last else None,
            "last_real_run": max(d["date"] for d in real) if real else None}


# ------------------------------------------------------------------------------- skill tags on the overview

# Where a skill sits: `steps:` in its SKILL.md frontmatter (validation.yaml stage ids, or `loop` for a skill that
# serves every step). Without `steps:`, its coarse `stage:` word places it. A skill that validation.yaml names as a
# stage's `skill` always sits at that stage; if it has no SKILL.md yet, it shows there as planned.
SKILL_STAGE_WORDS = {"read": "A", "build": "B", "tools": "C", "practice": "D", "compile": "E", "feedback": "F",
                     "watch": "G", "integrity": "G"}
CHECK_ID_RE = re.compile(r"\b([A-Z]\d?\.[a-z][a-z0-9_]*)\b")


def _sources_count(text: str) -> int | None:
    """Entries under ## Sources (list items, table rows, or distinct [id-2025] references); None without the section."""
    sec = _section(text, "Sources")
    if not sec:
        return None
    rows = [r for r in re.findall(r"^\|.*\|[ \t]*$", sec, re.M) if not re.fullmatch(r"\|[\s:|-]+\|", r.strip())]
    n = len(re.findall(r"^(?:[-*+]|\d+[.)])[ \t]+\S", sec, re.M)) + max(0, len(rows) - 1)
    return max(n, len(set(re.findall(r"\[([a-z][\w.+-]*-\d{4}[a-z]?)\]", sec))))


def skills_index(val: dict) -> tuple[dict, list[str]]:
    """Every playbook skill, written or planned, with what its card shows; plus warnings for the build output."""
    ids = [s["id"] for s in val["stages"]]
    owners: dict[str, list] = defaultdict(list)  # skill -> stages validation.yaml names it for
    for s in val["stages"]:
        if s.get("skill"):
            owners[s["skill"]].append(s["id"])
    out, warn = {}, []
    for name, path in _skill_paths().items():
        text = path.read_text(errors="replace")
        fm = _frontmatter(text)
        raw = fm.get("steps")
        if raw is None:
            word = str(fm.get("stage") or "").strip().lower()
            raw = ["loop"] if word == "infra" else [x for x in ids if x[:1] == SKILL_STAGE_WORDS.get(word)]
        raw = [str(x) for x in ([raw] if isinstance(raw, (str, int)) else raw if isinstance(raw, list) else [])]
        if bad := [x for x in raw if x not in ids and x != "loop"]:
            warn.append(f"{rel(path)}: steps {bad} are not stages in validation.yaml")
        steps = [x for x in ids if x in raw or x in owners.get(name, [])]
        if not steps and "loop" not in raw:
            warn.append(f"{rel(path)}: on no step yet; add `steps: [...]` to its frontmatter")
        checks_md = _section(text, "Checks")
        known = [c["id"] for x in owners.get(name, []) for s in val["stages"] if s["id"] == x for c in s["checks"]]
        known += [c for c in CHECK_ID_RE.findall(checks_md) if c in val["index"]]
        dated = skill_lessons(text)
        last = latest_lesson(dated)
        out[name] = {"name": name, "planned": False, "steps": steps, "loop": "loop" in raw, "owns": owners.get(name, []),
                     "description": str(fm.get("description") or ""), "goal_html": md_html(_section(text, "Goal")),
                     "checks": list(dict.fromkeys(known)), "checks_html": md_html(checks_md),
                     "lesson": {k: last[k] for k in ("date", "author", "headline", "real_run")} if last else None,
                     "lessons": len(dated) or len(re.findall(r"^\s*[-*]\s+\S", _section(text, "Lessons"), re.M)),
                     "tested": any(d["real_run"] for d in dated), "sources": _sources_count(text),
                     "href": href(path), "path": rel(path)}
    for name, sts in owners.items():
        if name not in out:  # named by validation.yaml, not written yet
            out[name] = {"name": name, "planned": True, "steps": sts, "loop": False, "owns": sts, "description": "",
                         "goal_html": "", "checks_html": "", "lesson": None, "lessons": 0, "tested": False,
                         "sources": None, "href": "", "path": rel(SKILLS_DIR / name / "SKILL.md"),
                         "checks": [c["id"] for s in val["stages"] if s["id"] in sts for c in s["checks"]]}
    return out, warn


def section_skills(sec: dict, skills: dict, val: dict) -> list[str]:
    """The skills at one story section: each stage's own skill first, in stage order, then the others."""
    own = [s["skill"] for x in sec["stages"] for s in val["stages"] if s["id"] == x and s.get("skill") in skills]
    rest = sorted((k for k, v in skills.items() if set(v["steps"]) & set(sec["stages"]) and k not in own),
                  key=lambda k: (min(sec["stages"].index(x) for x in skills[k]["steps"] if x in sec["stages"]), k))
    return list(dict.fromkeys(own + rest))


# ------------------------------------------------------------------------------- what each loop step optimized
# For the loop panels: per brain, data track and loop step, the dated lessons its skills gained, the versions its
# records name, the task tool releases, compiled routines and frozen practice runs in the run folders. It judges
# nothing; brains and tracks never mix.
_HINT_RE = re.compile(r"(?<![a-z0-9])([a-z]\d)(?![0-9])", re.I)  # a stage code in a caller name: claude:b2-replay
_BRAIN_RE = re.compile(r"\b(claude|argon)\b", re.I)
_TOOL_VER_RE = re.compile(r"(?<![/\w.-])([a-z][a-z0-9_-]*/v\d+)\b(?:\s*\(?\s*(r\d+)\b\)?)?")  # task/v1 (r3)
_PARAMS_VER_RE = re.compile(r"\bparams_v(\d+)([a-z]?)\b")
ATTEMPTS = {"baseline": "attempt 1", "attempt_1": "attempt 1", "attempt_n": "attempt N"}


def _ts(s) -> float:
    try:
        return dt.datetime.strptime(str(s)[:19], "%Y-%m-%d %H:%M:%S" if len(str(s)) >= 19 else "%Y-%m-%d %H:%M").timestamp()
    except ValueError:
        return 0.0


def _test_run(run_dir: Path) -> bool:
    """A run its manifest calls a test (attempt or note: 'compile tool test', 'harness test of ...'): not real work."""
    try:
        m = json.loads((run_dir / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(re.search(r"\btest\b", f"{m.get('attempt') or ''} {m.get('note') or ''}", re.I))


def _skill_stage(val: dict, skill: str, default: str) -> str | None:
    """The loop step validation.yaml gives a skill (compile-skill -> E), else the default if it is a loop step."""
    own = [s["id"] for s in val["stages"] if s.get("skill") == skill and s["role"] != "track"]
    return own[0] if own else default if default in val["loop_ids"] else None


def _step_versions(text: str, codes: set[str]) -> list[str]:
    """Versions a record names for its own step: right after the stage code ('B2 v4', 'B2 round 6'; a code with a digit
    may have a few words between, as in 'B2 formulation (b), v5b'), or a params_vN file. Validation versions
    ('validation v16') and tool releases ('task/v1') are never counted."""
    out = []
    for code in sorted(c for c in codes if c):
        gap = "[^.:;\n]{0,24}?" if re.search(r"\d", code) else "\\s*"
        for m in re.finditer(rf"\b{re.escape(code)}\b(?P<gap>{gap})(?<![/\w-])(?:v(?P<v>\d+)(?P<s>[a-z]?)|round\s+(?P<r>\d+))\b",
                             text):
            if "validation" not in m.group("gap").lower():
                out.append("v" + (m.group("v") + m.group("s") if m.group("v") else m.group("r")))
    return out + ["v" + a + b for a, b in _PARAMS_VER_RE.findall(text)]


def _lesson_place(val: dict, k: dict, lesson: dict) -> tuple[str, list[str], str | None]:
    """A lesson's (brain, tracks, step). brain: the first brain its header names (author or caller), else the brain
    folder of the runs it cites; '' when neither (no brain's lesson). tracks: of the runs it cites (validation.yaml
    `runs`; other runs are the first track's), else the first track. step: a stage code its header names among the
    skill's steps (claude:b2-replay -> B2), else the stage validation.yaml names the skill for, else its first step."""
    m = _BRAIN_RE.search(lesson["head"][:160])
    cited = [(brain_key(d), r) for d, r in lesson["runs"] if brain_key(d) in ("claude", "argon")]
    brain = brain_key(m.group(1)) if m else cited[0][0] if cited else ""
    tracks = list(dict.fromkeys(val["run_track"].get(r) or val["primary"] for _, r in cited)) or [val["primary"]]
    by_id = {s["id"]: s for s in val["stages"]}

    def steps_of(ids):  # a track's container stage (T2) serves the steps its checks name
        out = []
        for x in ids:
            s = by_id.get(x)
            out += ([c["step"] for c in s["checks"]] or [s["step"]]) if s and s["role"] == "track" else [x]
        return [x for x in val["loop_ids"] if x in out]

    serves, head = steps_of(k["steps"]), lesson["head"][:160]
    named = [h.upper() for h in _HINT_RE.findall(head)]  # stage codes, then stages whose own skill the header names
    named += [s["id"] for s in val["stages"] if s.get("skill") and s["skill"] != k["name"] and s["skill"] in head]
    hint = [x for x in steps_of(named) if x in serves]
    owns = steps_of(k["owns"])
    return brain, tracks, (hint or owns or serves or [None])[0]


def loop_progress(val: dict, skills: dict) -> dict:
    """brain -> track -> loop step -> {lessons, lesson, params, tools, routines, runs}: what the step optimized.

    - lessons: dated lessons of the skills at the step (lesson placement in _lesson_place), and the latest one
    - params: versions its records name (_step_versions), in the order they first appear
    - tools: task tool releases: `<interface>/vN rM` in its records, and runs/<brain>/<run>/task_tools/manifest.json
    - routines: compiled routines in runs/<brain>/<run>/library/index.json (an imported copy counts once; a run whose
      manifest calls it a test, such as a harness test of the library tools, is left out), on the compile step
    - runs: frozen practice runs (runs/<brain>/<run>/manifest.json, attempt 1 or N, not dry runs), on the practice step
    """
    out: dict = {}

    def slot(b: str, t: str, x: str | None) -> dict | None:
        if not b or x not in val["loop_ids"]:
            return None
        return out.setdefault(b, {}).setdefault(t, {}).setdefault(
            x, {"lessons": 0, "lesson": None, "params": [], "tools": [], "routines": [], "runs": []})

    for name, k in skills.items():
        if k.get("planned"):
            continue
        for ls in skill_lessons((REPO / k["path"]).read_text(errors="replace")):
            brain, tracks, step = _lesson_place(val, k, ls)
            for t in tracks:
                s = slot(brain, t, step)
                if s is None:
                    continue
                s["lessons"] += 1
                key = (ls["date"], ls["tod"], ls["pos"])
                if s["lesson"] is None or key >= s["lesson"]["key"]:
                    s["lesson"] = {"date": ls["date"], "headline": ls["headline"], "skill": name, "key": key}
    recs = []  # (t, brain, track, step, stage, text) for every record, oldest first
    for e in read_jsonl(events_path()):
        tr, st = place(val, e.get("stage"), e.get("track"), e.get("step"), e.get("check_id"), e.get("run_id"))
        recs.append((num(e.get("t")) or 0, brain_key(e.get("brain")), tr, st, e.get("stage"), str(e.get("text") or "")))
    for c in read_jsonl(checks_path()):
        tr, st = place(val, c.get("stage"), None, None, c.get("check_id"), c.get("run_id"))
        recs.append((num(c.get("t")) or 0, brain_key(c.get("brain")), tr, st, c.get("stage"), str(c.get("note") or "")))
    for m in read_jsonl(media_path()):
        tr, st = media_place(val, m)
        recs.append((num(m.get("t")) or 0, brain_key(m.get("brain")), tr, st, m.get("stage"),
                     f"{m.get('title') or ''}. {m.get('caption') or ''}"))
    tool_seen: dict = defaultdict(list)  # (brain, track, step) -> [(t, release)]
    for t0, b, tr, st, stage, text in sorted(recs, key=lambda r: r[0]):
        s = slot(b, tr, st)
        if s is None or not text:
            continue
        for v in _step_versions(text, {st, str(stage or st)}):
            if v not in s["params"]:
                s["params"].append(v)
        for name, r in _TOOL_VER_RE.findall(text):
            tool_seen[(b, tr, st)].append((t0, f"{name} {r}".strip()))
    tools_step = _skill_stage(val, "author-tools", "C")
    for p in sorted(RUNS_DIR.glob("*/*/task_tools/manifest.json")) if RUNS_DIR.is_dir() else []:
        try:
            m = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        run, b = p.parent.parent.name, brain_key(p.parent.parent.parent.name)
        if m.get("interface"):
            rev = re.search(r"\brevision\s+(\d+)", str(m.get("revision") or ""))
            tr = val["run_track"].get(run) or val["primary"]
            tool_seen[(b, tr, tools_step)].append((_ts(m.get("updated")) or p.stat().st_mtime,
                                                   f"{m['interface']} r{rev.group(1)}" if rev else str(m["interface"])))
    for (b, tr, st), seen in tool_seen.items():
        s = slot(b, tr, st)
        if s is not None:  # a release with a revision supersedes the bare interface name it extends
            names = list(dict.fromkeys(x for _, x in sorted(seen)))
            s["tools"] = [x for x in names if not any(y != x and y.startswith(x + " ") for y in names)]
    lib_step, prac_step = _skill_stage(val, "compile-skill", "E"), _skill_stage(val, "practice", "D")
    routines: dict = {}
    for p in sorted(RUNS_DIR.glob("*/*/library/index.json")) if RUNS_DIR.is_dir() else []:
        try:
            idx = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        run, b = p.parent.parent.name, brain_key(p.parent.parent.parent.name)
        if _test_run(p.parent.parent):
            continue
        tr = val["run_track"].get(run) or val["primary"]
        for name, r in (idx.get("routines") or {}).items():
            for ver, v in ((r or {}).get("versions") or {}).items():
                key, row = (b, tr, f"{name}@{ver}"), {"name": str(name), "version": str(ver), "status": str(v.get("status") or ""),
                                                      "parts": [str(x) for x in v.get("parts") or []], "run_id": run,
                                                      "t": _ts(v.get("created")), "copy": bool(v.get("imported_from"))}
                if key not in routines or (routines[key]["copy"] and not row["copy"]):
                    routines[key] = row
    for (b, tr, _), row in sorted(routines.items(), key=lambda kv: kv[1]["t"]):
        if (s := slot(b, tr, lib_step)) is not None:
            s["routines"].append({k: row[k] for k in ("name", "version", "status", "parts", "run_id")})
    for p in sorted(RUNS_DIR.glob("*/*/manifest.json")) if RUNS_DIR.is_dir() else []:
        try:
            m = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        att = ATTEMPTS.get(str(m.get("attempt") or ""))
        if not att or m.get("dry_run"):
            continue
        run, b = p.parent.name, brain_key(p.parent.parent.name)
        if (s := slot(b, val["run_track"].get(run) or val["primary"], prac_step)) is not None:
            s["runs"].append({"run_id": run, "attempt": att, "t": _ts(m.get("frozen_at")),
                              "note": _sentence(str(m.get("note") or ""), 140)})
    for per in out.values():
        for steps in per.values():
            for s in steps.values():
                if s["lesson"]:
                    s["lesson"].pop("key", None)
                s["runs"].sort(key=lambda r: r["t"])
    return out


def calls_summary() -> dict:
    callers: dict[str, dict] = {}
    for f in sorted(CALLS_DIR.glob("*.jsonl")) if CALLS_DIR.is_dir() else []:
        for r in read_jsonl(f):
            c = callers.setdefault(str(r.get("caller", "unknown")), {"calls": 0, "errors": 0, "last_t": 0,
                                                                   "tools": defaultdict(int)})
            c["calls"] += 1
            c["errors"] += 0 if r.get("ok", True) else 1
            c["last_t"] = max(c["last_t"], r.get("t", 0))
            c["tools"][r.get("tool", "?")] += 1
    rows = [{"caller": k, "brain": brain_key(k) if brain_key(k) in ("claude", "argon") else "",
             "calls": v["calls"], "errors": v["errors"], "last_t": v["last_t"],
             "top": sorted(v["tools"].items(), key=lambda kv: -kv[1])[:3]} for k, v in callers.items()]
    rows.sort(key=lambda r: -r["last_t"])
    return {"callers": rows, "last_t": rows[0]["last_t"] if rows else None}


def loop_summary(val: dict | None = None) -> dict:
    """Per data track, then per brain: runs, tokens, model and tool calls, wall time, est. USD, and a per-run series.
    A run belongs to the track that lists it (validation.yaml `tracks: runs`), else to the first track."""
    val = val or load_validation()
    out: dict[str, dict] = {t["id"]: {} for t in val["tracks"]}
    for d in sorted(LOOP_DIR.glob("*/")) if LOOP_DIR.is_dir() else []:
        try:
            meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").is_file() else {}
            summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").is_file() else None
        except (json.JSONDecodeError, OSError):
            continue
        if not meta and not summ:
            continue
        brain = brain_key(meta.get("brain") or (summ or {}).get("brain")) or "unknown"
        tid = val["run_track"].get(d.name) or val["primary"]
        b = out[tid].setdefault(brain, {"runs": 0, "running": 0, "replays": 0, "tokens": 0, "tokens_runs": 0,
                                        "model_calls": 0, "tool_calls": 0, "wall_s": 0.0, "usd": 0.0, "series": []})
        if meta.get("replay_of") or (summ or {}).get("replay_of"):
            b["replays"] += 1  # a replay re-serves recorded tokens: never count them twice
            continue
        b["runs"] += 1
        t = meta.get("started_at") or d.stat().st_mtime
        if summ is None:
            b["running"] += 1
            continue
        tok = summ.get("tokens")
        total = tok.get("total") if isinstance(tok, dict) else None
        if total is not None:
            b["tokens"] += int(total)
            b["tokens_runs"] += 1
        b["model_calls"] += int(summ.get("model_calls") or 0)
        b["tool_calls"] += int(summ.get("tool_calls") or 0)
        b["wall_s"] += float(summ.get("wall_s") or 0)
        b["usd"] += float(summ.get("usd_est") or 0)
        b["series"].append({"t": t, "run_id": d.name, "tokens": total, "wall_s": summ.get("wall_s"),
                            "calls": (summ.get("model_calls") or 0) + (summ.get("tool_calls") or 0)})
    for per in out.values():
        for b in per.values():
            b["series"] = sorted(b["series"], key=lambda s: s["t"])[-30:]
            b["usd"] = round(b["usd"], 4)
            b["wall_s"] = round(b["wall_s"], 1)
    return out


def decision_logs(val: dict) -> dict:
    """Reconstruction decision logs (runs/<brain>/<run>/decisions.jsonl, harness/tools/decisions.py) per data track.
    A run belongs to the track that lists it in validation.yaml (`tracks: runs`), else to the track its records'
    stages belong to."""
    out: dict[str, list] = {t["id"]: [] for t in val["tracks"]}
    for p in sorted(RUNS_DIR.glob("*/*/decisions.jsonl")) if RUNS_DIR.is_dir() else []:
        rs = read_jsonl(p)
        if not rs:
            continue
        run, brain = p.parent.name, brain_key(p.parent.parent.name)
        tids = [place(val, r.get("stage"))[0] for r in rs]
        tid = val["run_track"].get(run) or max(tids, key=tids.count)
        steps = defaultdict(int)
        for r in rs:
            steps[str(r.get("step") or "other").replace("_", " ")] += 1
        out[tid].append({"run_id": run, "brain": brain, "n": len(rs), "href": href(p), "path": rel(p),
                         "steps": sorted(steps.items(), key=lambda kv: (-kv[1], kv[0])),
                         "native": sum((r.get("native") or {}).get("source") == "native" for r in rs),
                         "open": sum(r.get("verdict") == "open" for r in rs),
                         "last_t": max((r.get("t") or 0) for r in rs)})
    for rows in out.values():
        rows.sort(key=lambda r: -r["last_t"])
    return out


def num(v):
    """JSON.parse rejects NaN/Infinity: a hand-edited record must not break the page."""
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def page_record(r: dict) -> dict:
    return {"t": r.get("t"), "value": num(r.get("value")), "state": r.get("state"), "by": r.get("by"),
            "run_id": r.get("run_id") or "", "note": r.get("note") or "", "version": r.get("validation_version"),
            "evidence": [evidence_item(e) for e in r.get("evidence", [])]}


def collect() -> dict:
    val = load_validation()
    records = read_jsonl(checks_path())
    results = {}
    groups, act = by_brain(records), activity(val)
    for brain in list(groups) + sorted(set(act) - set(groups)):  # a brain with work on record but no measurement yet too
        rs = groups.get(brain, [])
        view = brain_view(rs, val)
        checks = {}
        for cid, cv in view["checks"].items():
            recs = cv["records"]
            checks[cid] = {"state": cv["state"], "regressed": cv["regressed"], "ever_passed": cv["ever_passed"],
                           "hist": [[r.get("t"), num(r.get("value")), r.get("state")] for r in recs[-MAX_HIST:]],
                           "recent": [page_record(r) for r in reversed(recs[-MAX_RECENT:])], "n": len(recs)}
        # every track's view of the loop: its checks placed on each step, judged with this brain's records, plus this
        # brain's events and media there (n counts measurements, act the work on record)
        tracks = {}
        for t in val["tracks"]:
            own = {c for ids in t["steps"].values() for c in ids}
            mine = act.get(brain, {}).get(t["id"], {})
            tracks[t["id"]] = {"n": sum(r.get("check_id") in own for r in rs),
                               "act": sum(a["events"] + a["media"] for a in mine.values()), "stages": {
                x: with_activity(stage_view({"id": x, "checks": [val["index"][c][1] for c in ids]}, view["checks"]),
                                 mine.get(x)) for x, ids in t["steps"].items()}}
        results[brain] = {"checks": checks, "n": len(rs), "tracks": tracks}
    events, per_track = [], defaultdict(int)
    for e in sorted(read_jsonl(events_path()), key=lambda e: -e.get("t", 0)):
        track, step = place(val, e.get("stage"), e.get("track"), e.get("step"), e.get("check_id"), e.get("run_id"))
        per_track[track] += 1
        if per_track[track] > MAX_EVENTS:
            continue
        events.append({"t": e.get("t"), "kind": e.get("kind", "note"), "text": e.get("text", ""),
                       "stage": e.get("stage"), "track": track, "step": step, "check_id": e.get("check_id"),
                       "brain": brain_key(e.get("brain")),
                       "by": e.get("by") or "", "run_id": e.get("run_id") or "", "auto": bool(e.get("auto")),
                       "evidence": [evidence_item(x) for x in e.get("evidence", [])]})
    stages = []
    for s in val["stages"]:
        stages.append({**{k: s[k] for k in ("id", "name", "skill", "role", "track", "step")},
                       "checks": [{**c, "target_text": target_text(c)} for c in s["checks"]],
                       "skill_status": skill_status(s["skill"])})
    brains = ["claude", "argon"] + sorted(set(results) - {"claude", "argon"})
    skills, skill_warnings = skills_index(val)
    story = load_story(val)
    for sec in story["sections"]:
        sec["skills"] = section_skills(sec, skills, val)
    tracks = [{**t, "label": story["views"][t["id"]]["label"]} for t in val["tracks"]]
    hyps = [{**h, "track": str(h.get("track") or val["primary"])} for h in val["hypotheses"]]
    views = story.pop("views")
    now = time.time()
    return {"built_t": now, "built": iso(now), "version": val["version"], "history": val["history"],
            "hypotheses": hyps, "tracks": tracks, "primary": val["primary"],
            "track_warnings": val["warnings"] + story["warnings"],
            "brains": brains, "brain_labels": {b: BRAIN_LABELS.get(b, b) for b in brains},
            "stages": stages, "results": results, "events": events, "cost": loop_summary(val),
            "decisions": decision_logs(val), "media": media_items(val), "story": story, "views": views,
            "skills": skills, "loop_skills": sorted(k for k, v in skills.items() if v["loop"]),
            "skill_warnings": skill_warnings, "progress": loop_progress(val, skills),
            "calls": calls_summary(),
            "counts": {"checks": len(records), "events": len(events)},
            "paths": {"validation": rel(VALIDATION), "checks": rel(checks_path()), "events": rel(events_path()),
                      "media": rel(media_path())}}


def render(data: dict) -> str:
    blob = json.dumps(data, default=str, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    blob = blob.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    vendor = "\n".join((VENDOR / f).read_text() for f in VENDOR_JS if (VENDOR / f).is_file())
    page = TEMPLATE.read_text().replace("/*__DATA__*/{}", blob, 1)
    return page.replace("/*__VENDOR__*/", vendor.replace("</script", "<\\/script"), 1)


def build(bundle: bool = False) -> Path:
    """Rebuild runs/status/index.html (atomic replace, safe with parallel writers).

    bundle=True also copies every file the page links to into runs/status/media/ and links the copies,
    so runs/status/ is self-contained (zip it or publish it). The next plain rebuild links the originals again.
    """
    global _BUNDLE
    out = page_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    _BUNDLE = {} if bundle else None
    try:
        html = render(collect())
    finally:
        copied, _BUNDLE = len(_BUNDLE or {}), None
    tmp = out.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(html)
    os.replace(tmp, out)
    build.bundled = copied
    return out


# ------------------------------------------------------------------------------- summary for brains

def summary(brain: str | None = None, stage: str | None = None) -> dict:
    val = load_validation()
    groups, act = by_brain(read_jsonl(checks_path())), activity(val)
    want = [brain_key(brain)] if brain else (sorted(set(groups) | set(act)) or ["claude", "argon"])
    out = {"validation_version": val["version"], "page": rel(page_path()), "brains": {}}
    for b in want:
        view = brain_view(groups.get(b, []), val)
        stages = {}
        for s in val["stages"]:
            if stage and s["id"] != stage:
                continue
            sv = view["stages"][s["id"]]
            row = {"name": s["name"], "state": sv["state"], "pass": f"{sv['passed']}/{sv['gated']}"}
            # `state` is the checks' verdict only; `activity` adds this brain's events and media on the step, and
            # `unscored_work` marks work with no measured check yet (the page's "in progress, not scored yet")
            mine = act.get(b, {}).get(s["track"], {})
            if s["role"] != "track" and (a := mine.get(s["id"])):
                row["activity"] = {"events": a["events"], "media": a["media"]}
                if not sv["measured"]:
                    row["unscored_work"] = True
            elif s["role"] == "track" and mine:  # a track's container: its work per loop step
                tv = {x: stage_view({"id": x, "checks": [val["index"][c][1] for c in ids]}, view["checks"])
                      for t in val["tracks"] if t["id"] == s["track"] for x, ids in t["steps"].items()}
                row["activity"] = {x: {"events": a["events"], "media": a["media"]}
                                   | ({"unscored_work": True} if not (tv.get(x) or {}).get("measured") else {})
                                   for x, a in mine.items()}
            if s["role"] == "track":  # another data track's container: each check names its loop step
                row["track"] = s["track"]
            meas = {}
            for c in s["checks"]:
                cv = view["checks"].get(c["id"])
                if cv:
                    meas[c["id"]] = {"value": cv["latest"].get("value"), "state":
                                     "regressed" if cv["regressed"] else cv["state"], "target": target_text(c)}
                    if s["role"] == "track":
                        meas[c["id"]]["step"] = c["step"]
            if meas:
                row["checks"] = meas
            if sv["unmeasured"] and sv["measured"]:
                row["unmeasured"] = sv["unmeasured"]
            stages[s["id"]] = row
        out["brains"][b] = {"label": BRAIN_LABELS.get(b, b), "records": len(groups.get(b, [])), "stages": stages}
    ev = sorted(read_jsonl(events_path()), key=lambda e: -e.get("t", 0))
    ev = [e for e in ev if e.get("kind") != "note" and (not brain or brain_key(e.get("brain")) in ("", brain_key(brain)))]
    out["recent_events"] = [{"when": e.get("ts"), "kind": e.get("kind"), "stage": e.get("stage"),
                             "track": place(val, e.get("stage"), e.get("track"), e.get("step"), e.get("check_id"),
                                            e.get("run_id"))[0],
                             "brain": brain_key(e.get("brain")), "text": e.get("text")} for e in ev[:6]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild runs/status/index.html")
    ap.add_argument("--bundle", action="store_true", help="copy every linked file into runs/status/media/ "
                    "and link the copies, so the folder can be zipped or published on its own")
    a = ap.parse_args()
    out = build(bundle=a.bundle)
    data = collect()
    states = {b: {tid: {x: v["state"] for x, v in tv["stages"].items() if v["total"]}
                  for tid, tv in r["tracks"].items() if tv["n"]} for b, r in data["results"].items()}
    states = {b: v for b, v in states.items() if v}
    unscored = {b: {tid: [x for x, v in tv["stages"].items() if v["unscored"]] for tid, tv in r["tracks"].items()
                    if any(v["unscored"] for v in tv["stages"].values())} for b, r in data["results"].items()}

    def track_line(t: dict) -> str:
        steps = [x for x, ids in t["steps"].items() if ids]
        media = [m for m in data["media"] if m["track"] == t["id"]]
        return (f"{t['status']}{', the first track' if t['primary'] else ''}; "
                + (f"checks on {', '.join(steps)}" if steps else "no checks yet")
                + f"; {len(media)} media ({sum(m['step'] == INPUT for m in media)} footage), "
                + f"{sum(e['track'] == t['id'] for e in data['events'])} events"
                + (f"; runs {', '.join(t['runs'])}" if t["runs"] else ""))

    info = {"page": rel(out), "checks": data["counts"]["checks"], "events": data["counts"]["events"],
            "media": len(data["media"]), "states": states or "not measured yet",
            "skills": {s["id"]: s["skills"] for s in data["story"]["sections"]} | {"loop": data["loop_skills"]},
            "tracks": {t["id"]: track_line(t) for t in data["tracks"]},
            "loops": {lp["id"]: {"steps": lp["steps"], "headline": {t: [c for c, _ in v["loops"][lp["id"]]["headline"]]
                                                                    for t, v in data["views"].items()}}
                      for lp in data["story"]["loops"]}}
    if unscored := {b: v for b, v in unscored.items() if v}:  # steps with work on record, no measured check yet
        info["unscored_work"] = unscored
    if data["skill_warnings"]:
        info["skill_warnings"] = data["skill_warnings"]
    if data["track_warnings"]:
        info["track_warnings"] = data["track_warnings"]
    if a.bundle:
        info["bundled_files"] = build.bundled
        info["bundle_dir"] = rel(STATUS_DIR / "media")
    if data["story"]["error"]:
        info["story_error"] = data["story"]["error"]
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
