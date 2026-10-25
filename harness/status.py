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

Every record sits on one data track and one loop step (place): the page builds every track's view at build time
and switches between them in the browser. Judging lives here (judge, brain_view, stage states) so the record_check
tool, status_summary and the page always agree. Results are grouped by brain and never merged: a Claude stand-in
number and an Argon number are separate rows everywhere.
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
        track, step = place(val, m.get("stage"), m.get("track"), m.get("step"), None, m.get("run_id"))
        if not m.get("step") and str((m.get("paths") or [""])[0]).startswith(EXPLAIN_DIR):
            step = INPUT  # a clip cut for people, published without a step: the track's footage
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
  footage:
    label: Footage only
    input:
      note: human or egocentric video
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
    return {"brand": str(raw.get("brand") or "Bot18"), "title": str(raw.get("title") or ""),
            "subtitle": str(raw.get("subtitle") or ""), "footer": str(raw.get("footer") or ""),
            "builders": _builders(raw.get("builders")), "input": inp, "output": outp,
            "sections": sections, "track_labels": labels,
            "views": track_views(val, sections, inp, outp, labels, over, warnings),
            "warnings": warnings, "error": error, "path": rel(path)}


def _first_check(val: dict, ids: list[str]) -> str | None:
    gated = [c for c in ids if val["index"][c][1]["direction"] in GATED]
    return (gated or ids or [None])[0]


def track_views(val: dict, sections: list, inp: dict, outp: dict, labels: dict, over: dict, warnings: list) -> dict:
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
                               "feature": ov.get("feature") or base["feature"], "checks": len(mine)}
        oi = o.get("input") if isinstance(o.get("input"), dict) else {}
        oo = o.get("output") if isinstance(o.get("output"), dict) else {}
        views[t["id"]] = {
            "label": label,
            "input": {"label": inp["label"], "out": inp["out"],
                      "note": str(oi.get("note") or (inp["note"] if t["primary"] else t["title"] + " recordings")),
                      "explain": _explain(oi["explain"]) if oi.get("explain") else inp["explain"] if t["primary"] else None},
            "output": {**outp, "note": str(oo.get("note") or outp["note"]), "explain": outp["explain"] if t["primary"] else None},
            "sections": secs}
    return views


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
        headline = _sentence(_plain(re.split(r"\bEvidence:", note.group(1))[0])) if note else ""
        if not headline:
            bold = (_plain(b) for b in re.findall(r"\*\*(.+?)\*\*", block[len(d.group(0)):]))
            headline = next((_sentence(b) for b in bold if len(b.split()) >= 4), "")
        if not headline:
            gist = _plain(re.split(r"\bEvidence:", head.replace(author or "\0", "", 1))[0])
            headline = _sentence(gist) if len(gist.split()) >= 2 else ""
        out.append({"date": d.group(1), "tod": tod, "pos": d.start(), "author": author, "headline": headline,
                    "real_run": bool(re.search(r"\bruns/", block))})
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
    for brain, rs in by_brain(records).items():
        view = brain_view(rs, val)
        checks = {}
        for cid, cv in view["checks"].items():
            recs = cv["records"]
            checks[cid] = {"state": cv["state"], "regressed": cv["regressed"], "ever_passed": cv["ever_passed"],
                           "hist": [[r.get("t"), num(r.get("value")), r.get("state")] for r in recs[-MAX_HIST:]],
                           "recent": [page_record(r) for r in reversed(recs[-MAX_RECENT:])], "n": len(recs)}
        # every track's view of the loop: its checks placed on each step, judged with this brain's records
        tracks = {}
        for t in val["tracks"]:
            own = {c for ids in t["steps"].values() for c in ids}
            tracks[t["id"]] = {"n": sum(r.get("check_id") in own for r in rs), "stages": {
                x: stage_view({"id": x, "checks": [val["index"][c][1] for c in ids]}, view["checks"])
                for x, ids in t["steps"].items()}}
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
            "skill_warnings": skill_warnings,
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
    groups = by_brain(read_jsonl(checks_path()))
    want = [brain_key(brain)] if brain else (sorted(groups) or ["claude", "argon"])
    out = {"validation_version": val["version"], "page": rel(page_path()), "brains": {}}
    for b in want:
        view = brain_view(groups.get(b, []), val)
        stages = {}
        for s in val["stages"]:
            if stage and s["id"] != stage:
                continue
            sv = view["stages"][s["id"]]
            row = {"name": s["name"], "state": sv["state"], "pass": f"{sv['passed']}/{sv['gated']}"}
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
            "tracks": {t["id"]: track_line(t) for t in data["tracks"]}}
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
