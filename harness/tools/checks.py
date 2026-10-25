"""Validation tools: watchers report measurements, the harness judges them (playbook/validation.yaml).

record_check   a measured value for one check in validation.yaml. The harness looks up the check,
               computes pass / fail / track itself and appends to runs/status/checks.jsonl with the
               validation.yaml version. When the check's pass state flips, or its stage's overall state
               changes, it appends the matching event to runs/status/events.jsonl on its own.
record_event   a milestone, regression, decision, blocker or note for the timeline.
publish_media  a video, images or a comparison for a stage's showcase (runs/status/media.jsonl).
status_summary compact JSON of every stage's state per brain, for brains to read.

Every record carries its data track and loop step (validation.yaml `tracks`; status.place): a check's come from
the check, an event's or a media item's from `stage` (a track's container such as T2 names the track), `track`, the
run's track, or the first track, at `step` or the stage itself. Every writer rebuilds runs/status/index.html
(harness/status.py). Results are kept per brain: a Claude stand-in result is never counted as Argon's.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

from harness import status as st
from harness.registry import REPO, tool

URL_RE = st.URL_RE
EVENT_KINDS = ("milestone", "regression", "decision", "blocker", "note")
SECRET_RE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}")
MAX_TEXT = 1000
NAME_RE = re.compile(r"[^\w:.@/+-]")


def _clean_name(s: str, field: str, required: bool = True) -> str:
    s = NAME_RE.sub("", str(s or "").strip().replace(" ", "-"))[:64]
    if required and not s:
        raise ValueError(f"{field} is required")
    return s


def _evidence(items, need_one: bool) -> tuple[list[str], list[str]]:
    if isinstance(items, str):
        items = [items]
    refs, missing = [], []
    for e in items or []:
        e = str(e).strip()
        if not e:
            continue
        if e.startswith(str(REPO) + "/"):
            e = e[len(str(REPO)) + 1:]  # store repo-relative paths
        if SECRET_RE.search(e):
            raise ValueError("evidence looks like it contains a secret (rules.md: no secrets in records)")
        refs.append(e)
        if not st.evidence_item(e)["exists"]:
            missing.append(e)
    if need_one and len(missing) == len(refs):
        raise ValueError("evidence must include at least one existing file under the repo (e.g. a plot, "
                         f"overlay or run log under runs/) or a URL; got {refs or 'nothing'}")
    return refs, missing


def _text(s: str, field: str) -> str:
    s = " ".join(str(s or "").split())
    if len(s) > MAX_TEXT:
        raise ValueError(f"{field} is {len(s)} chars; keep it under {MAX_TEXT}")
    if SECRET_RE.search(s):
        raise ValueError(f"{field} looks like it contains a secret (rules.md: no secrets in records)")
    return s


def _rebuild() -> str | None:
    try:
        st.build()
        return None
    except Exception as e:  # a page error never loses a record
        return f"{type(e).__name__}: {e}"


def _placement(val: dict, stage: str | None, track: str | None, step: str | None, run_id: str,
               allow_input: bool = False) -> dict:
    """Validate where a record goes and return {stage, track, step} as stored. Raises on unknown or conflicting
    values, so a wrong call fails instead of landing on the wrong track."""
    stages = {s["id"]: s for s in val["stages"]}
    tracks = [t["id"] for t in val["tracks"]]
    steps = val["loop_ids"] + ([st.INPUT] if allow_input else [])
    stage, track, step = (str(x or "").strip() or None for x in (stage, track, step))
    if stage and stage not in stages:
        raise ValueError(f"unknown stage {stage!r}; known: {list(stages)}")
    if track and track not in tracks:
        raise ValueError(f"unknown track {track!r}; known: {tracks}")
    if step and step not in steps:
        raise ValueError(f"unknown step {step!r}; known: {steps}")
    s = stages.get(stage)
    if s and s["role"] == "track" and track and track != s["track"]:
        raise ValueError(f"stage {stage} belongs to track {s['track']!r}, not {track!r}")
    if s and s["role"] != "track" and step and step not in (stage, st.INPUT):
        raise ValueError(f"stage {stage} is a loop step itself; to place a record on step {step}, pass "
                         f"step={step!r} alone, or with track= or a track's stage (e.g. T2)")
    tid, where = st.place(val, stage, track, step, None, run_id)
    if s is None and step and step != st.INPUT:
        s = stages.get(step)
    if tid != val["primary"] and (s is None or s["track"] != tid):
        # another track's record keeps that track's container stage, so readers of `stage` alone never mistake it
        # for the first track's (T2 for so101); a track without one keeps the loop step
        own = [x for x in val["stages"] if x["track"] == tid]
        s = next((x for x in own if any(c["step"] == where for c in x["checks"])), own[0] if own else s)
    return {"stage": s["id"] if s else None, "track": tid, "step": where}


def _auto_event(val: dict, before: dict, after: dict, rec: dict) -> tuple[str, str] | None:
    """The one event (kind, text) a new record implies, or None. Stage-level news wins over check-level."""
    stage, check = val["index"][rec["check_id"]]
    sid, cid, tgt = stage["id"], rec["check_id"], st.target_text(check)
    old = before["checks"].get(cid)
    old_c, new_c = (old["state"] if old else None), rec["state"]
    os_, ns = before["stages"][sid]["state"], after["stages"][sid]
    v, prev = st.fmt(rec["value"]), (st.fmt(old["latest"]["value"]) if old else None)
    score = f", {ns['passed']}/{ns['gated']} pass" if ns["gated"] else ""

    if stage["role"] == "integrity" and new_c == "fail" and old_c != "fail":
        return "blocker", f"Integrity: {cid} = {v}, must be {tgt}. Stop and report (rules.md)."
    if ns["state"] != os_ and ns["state"] == "passing":
        return "milestone", f"{sid} {stage['name']} passes: all {ns['gated']} checks meet target (last: {check['short']} {v} {tgt})."
    if ns["state"] != os_ and ns["state"] == "regressed":
        return "regression", f"{sid} {stage['name']} regressed: {check['short']} {v} misses {tgt}" + (f" (was {prev})." if prev else ".")
    if old_c == "pass" and new_c == "fail":
        return "regression", f"{cid} regressed: {prev} → {v}, target {tgt}. Stage {sid} {st.STAGE_LABELS[ns['state']]}{score}."
    if new_c == "pass" and old_c != "pass":
        again = " again" if old and old["ever_passed"] else ""
        return "milestone", f"{cid} passes{again}: {v} {tgt}. Stage {sid} {st.STAGE_LABELS[ns['state']]}{score}."
    if ns["state"] != os_:
        what = (f"first measured: {st.STAGE_LABELS[ns['state']]}" if os_ == "not_started"
                else f"{st.STAGE_LABELS[os_]} → {st.STAGE_LABELS[ns['state']]}")
        return "note", f"{sid} {stage['name']} {what}{score} ({cid} = {v}, {tgt})."
    return None


@tool(
    "record_check",
    "Record one measured value for a check in playbook/validation.yaml (e.g. B0.overlay_hama1). "
    "Watchers measure with domain tools and report the number here; the harness compares it with the frozen "
    "target and decides pass/fail itself, so never pass a verdict. Milestones and regressions are logged "
    "automatically when a check or stage changes state. Returns the verdict and the stage state.",
    {"type": "object", "required": ["check_id", "value", "evidence", "brain", "by"], "properties": {
        "check_id": {"type": "string", "description": "check id from validation.yaml, e.g. A.segment_f1"},
        "value": {"type": "number", "description": "the measured value, in the check's metric units"},
        "evidence": {"type": "array", "items": {"type": "string"},
                     "description": "repo-relative paths (plots, overlays, run logs under runs/) or URLs "
                                    "backing the value; at least one must exist"},
        "brain": {"type": "string", "description": "brain whose work was measured: claude (stand-in) or argon"},
        "by": {"type": "string", "description": "name of the watcher that measured it"},
        "run_id": {"type": "string", "description": "runs/loop/<run-id> the result belongs to, if any"},
        "note": {"type": "string", "description": "one line: how it was measured, sample size, caveats"}}},
)
def record_check(check_id: str, value: float, evidence: list[str] | str, brain: str, by: str,
                 run_id: str = "", note: str = "") -> dict:
    val = st.load_validation()
    if check_id not in val["index"]:
        prefix = str(check_id).split(".")[0]
        near = [c for c in val["index"] if c.startswith(prefix + ".")] or list(val["index"])
        raise ValueError(f"unknown check {check_id!r}; known: {near}")
    if isinstance(value, bool):
        raise ValueError("value must be a number, not a boolean")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"value must be a number; got {value!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"value must be finite; got {value}")
    value = int(value) if value.is_integer() and abs(value) < 1e15 else value
    bkey = st.brain_key(brain)
    if not bkey:
        raise ValueError("brain is required: 'claude' for a stand-in run, 'argon' for Argon")
    refs, missing = _evidence(evidence, need_one=True)
    stage, check = val["index"][check_id]
    now = time.time()
    rec = {"t": now, "ts": st.iso(now), "check_id": check_id, "stage": stage["id"], "track": stage["track"],
           "step": check["step"], "value": value,
           "direction": check["direction"], "target": check["target"],
           "state": st.judge(check["direction"], check["target"], value),
           "brain": bkey, "run_id": _clean_name(run_id, "run_id", False), "by": _clean_name(by, "by"),
           "note": _text(note, "note"), "evidence": refs, "validation_version": val["version"]}

    with st.locked():
        mine = [r for r in st.read_jsonl(st.checks_path()) if st.brain_key(r.get("brain")) == bkey]
        before = st.brain_view(mine, val)
        after = st.brain_view(mine + [rec], val)
        st.append_jsonl(st.checks_path(), rec)
        auto = _auto_event(val, before, after, rec)
        event = None
        if auto:
            event = {"t": now, "ts": rec["ts"], "kind": auto[0], "text": auto[1], "stage": stage["id"],
                     "track": rec["track"], "step": rec["step"], "check_id": check_id, "brain": bkey, "run_id": rec["run_id"], "by": rec["by"],
                     "evidence": refs, "auto": True, "validation_version": val["version"]}
            st.append_jsonl(st.events_path(), event)
    sv = after["stages"][stage["id"]]
    cv = after["checks"][check_id]
    out = {"ok": True, "check_id": check_id, "value": value, "target": st.target_text(check),
           "state": "regressed" if cv["regressed"] else rec["state"], "brain": bkey,
           "track": rec["track"], "step": rec["step"],
           "validation_version": val["version"],
           "stage": {"id": stage["id"], "state": sv["state"], "pass": f"{sv['passed']}/{sv['gated']}",
                     "unmeasured": sv["unmeasured"]},
           "event": {"kind": event["kind"], "text": event["text"]} if event else None,
           "page": st.rel(st.page_path())}
    if missing:
        out["missing_evidence"] = missing
    err = _rebuild()
    if err:
        out["page_error"] = err
    return out


@tool(
    "record_event",
    "Add an entry to the status timeline: a milestone, regression, decision, blocker or note. "
    "Use it for news the checks don't capture (a decision taken, a blocker hit, a skill promoted). "
    "Check flips and stage changes are already logged automatically by record_check. "
    "The entry lands on one data track's step: a track's own stage (T2 is SO-101) names its track; "
    "otherwise pass track (default: the run's track, else REASSEMBLE).",
    {"type": "object", "required": ["kind", "text"], "properties": {
        "kind": {"type": "string", "enum": list(EVENT_KINDS)},
        "text": {"type": "string", "description": "one or two sentences, with numbers where they exist"},
        "stage": {"type": "string", "description": "stage id from validation.yaml (A, B0, ... G, or a track's stage "
                  "such as T2), if any"},
        "track": {"type": "string", "description": "data track id from validation.yaml tracks (reassemble, so101, "
                  "footage); default: the stage's track, else the run's track, else the first track"},
        "step": {"type": "string", "description": "loop step it belongs to (A, B0, B1, B2, C, D, E, F, G) when stage is "
                 "a track's stage like T2 or is omitted; default: the stage itself, or the step most of T2's checks sit on"},
        "evidence": {"type": "array", "items": {"type": "string"}, "description": "repo-relative paths or URLs"},
        "brain": {"type": "string", "description": "claude (stand-in) or argon, when the event belongs to one brain"},
        "run_id": {"type": "string"},
        "by": {"type": "string", "description": "who records it (agent name)"}}},
)
def record_event(kind: str, text: str, stage: str | None = None, evidence: list[str] | None = None,
                 brain: str = "", run_id: str = "", by: str = "", track: str = "", step: str = "") -> dict:
    if kind not in EVENT_KINDS:
        raise ValueError(f"kind must be one of {list(EVENT_KINDS)}; got {kind!r}")
    text = _text(text, "text")
    if not text:
        raise ValueError("text is empty")
    val = st.load_validation()
    run_id = _clean_name(run_id, "run_id", False)
    where = _placement(val, stage, track, step, run_id)
    refs, missing = _evidence(evidence, need_one=False)
    now = time.time()
    ev = {"t": now, "ts": st.iso(now), "kind": kind, "text": text, "stage": where["stage"], "track": where["track"],
          "step": where["step"], "check_id": None, "brain": st.brain_key(brain), "run_id": run_id,
          "by": _clean_name(by, "by", False), "evidence": refs, "auto": False,
          "validation_version": val["version"]}
    with st.locked():
        st.append_jsonl(st.events_path(), ev)
    out = {"ok": True, "kind": kind, "stage": ev["stage"], "track": ev["track"], "step": ev["step"],
           "brain": ev["brain"], "page": st.rel(st.page_path())}
    if missing:
        out["missing_evidence"] = missing
    err = _rebuild()
    if err:
        out["page_error"] = err
    return out


MEDIA_KINDS = ("video", "image", "compare")


@tool(
    "publish_media",
    "Show a visual result on the Bot18 page (runs/status/index.html): a video, images, or a comparison "
    "(e.g. real vs sim from make_comparison, or two stills shown with a wipe slider). Each chapter shows its stage's "
    "latest featured item large (else its latest item) with its caption and metrics; the rest go in the chapter's reel. "
    "Videos that browsers can't play are re-encoded to H.264 automatically. Title and caption limits: docs/design/ui-protocol.md. "
    "The item lands on one data track's step: a track's own stage (T2 is SO-101) names its track, or pass track and step. "
    "Real footage for a track (clean clips, not results) goes to step 'input', the start of the loop.",
    {"type": "object", "required": ["title", "paths", "brain"], "properties": {
        "stage": {"type": "string", "description": "stage id from validation.yaml (A, B0, B1, B2, C, D, E, F, G, or a "
                  "track's stage such as T2); needed unless step is given"},
        "track": {"type": "string", "description": "data track id from validation.yaml tracks (reassemble, so101, "
                  "footage); default: the stage's track, else the run's track, else the first track"},
        "step": {"type": "string", "description": "loop step it belongs to (A, B0, B1, B2, C, D, E, F, G), or 'input' "
                 "for the track's real footage; default: the stage itself, or the step most of T2's checks sit on"},
        "title": {"type": "string", "description": "a few words, e.g. 'hama1: real vs twin overlay'"},
        "paths": {"type": "array", "items": {"type": "string"},
                  "description": "repo-relative files: the main video or image first, then supporting stills"},
        "kind": {"type": "string", "enum": list(MEDIA_KINDS), "description": "default: video if the first file is a video, else image"},
        "caption": {"type": "string", "description": "one sentence: what the viewer is looking at"},
        "metrics": {"type": "object", "description": "key numbers shown beside it, e.g. {'B0.overlay_hama1': 0.88, 'frames': 40}; "
                    "keys that are check ids are judged against their target"},
        "labels": {"type": "array", "items": {"type": "string"}, "description": "for a two-image compare: the two side names, default real / sim"},
        "featured": {"type": "boolean", "description": "pin as the stage's hero item (default false)"},
        "brain": {"type": "string", "description": "claude (stand-in) or argon: whose run produced it"},
        "run_id": {"type": "string"},
        "by": {"type": "string", "description": "who publishes it (agent name)"}}},
)
def publish_media(title: str, paths: list[str] | str, brain: str, stage: str = "", kind: str | None = None,
                  caption: str = "", metrics: dict | None = None, labels: list[str] | None = None,
                  featured: bool = False, run_id: str = "", by: str = "", track: str = "", step: str = "") -> dict:
    from harness.tools.compare import IMAGE_EXT, VIDEO_EXT, ensure_playable  # ffmpeg only when publishing

    val = st.load_validation()
    if not (stage or step):
        raise ValueError("stage or step is required, e.g. stage='B0', or stage='T2' for SO-101, "
                         "or track='so101', step='input' for its footage")
    run_id = _clean_name(run_id, "run_id", False)
    where = _placement(val, stage, track, step, run_id, allow_input=True)
    title = _text(title, "title")[:120]
    if not title:
        raise ValueError("title is required")
    bkey = st.brain_key(brain)
    if not bkey:
        raise ValueError("brain is required: 'claude' for a stand-in run, 'argon' for Argon")
    refs, missing = _evidence(paths, need_one=False)
    if missing or not refs:
        raise ValueError(f"paths must be existing repo files; missing: {missing or 'no paths given'}")
    files, notes = [], []
    for r in refs:
        if URL_RE.match(r):
            files.append(r)
            continue
        p = Path(r) if r.startswith("/") else REPO / r
        if p.suffix.lower() in VIDEO_EXT:
            q = ensure_playable(p)
            if q != p:
                notes.append(f"{r} re-encoded to H.264 for browsers: {st.rel(q)}")
                r = st.rel(q)
        files.append(r)
    first = Path(files[0]).suffix.lower()
    kind = kind or ("video" if first in VIDEO_EXT else "image")
    if kind not in MEDIA_KINDS:
        raise ValueError(f"kind must be one of {list(MEDIA_KINDS)}")
    if kind == "video" and first not in VIDEO_EXT:
        raise ValueError("kind 'video' needs a video file first in paths")
    if kind == "image" and first not in IMAGE_EXT and first != ".svg":
        raise ValueError("kind 'image' needs an image file first in paths")
    clean_metrics = {}
    for k, v in (metrics or {}).items():
        k = _text(k, "metric name")[:60]
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise ValueError(f"metric {k!r} must be a number or short text")
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"metric {k!r} must be finite")
        clean_metrics[k] = _text(v, f"metric {k}")[:60] if isinstance(v, str) else v
    if len(clean_metrics) > 8:
        raise ValueError("keep metrics to the 8 numbers that matter")
    now = time.time()
    item = {"t": now, "ts": st.iso(now), "id": f"m{int(now * 1000):x}", "stage": where["stage"],
            "track": where["track"], "step": where["step"], "title": title, "kind": kind,
            "paths": files, "caption": _text(caption, "caption"), "metrics": clean_metrics,
            "labels": [_text(x, "label")[:30] for x in (labels or [])][:2], "featured": bool(featured),
            "brain": bkey, "run_id": run_id, "by": _clean_name(by, "by", False),
            "validation_version": val["version"]}
    with st.locked():
        st.append_jsonl(st.media_path(), item)
    out = {"ok": True, "id": item["id"], "stage": item["stage"], "track": item["track"], "step": item["step"],
           "kind": kind, "paths": files, "featured": item["featured"], "page": st.rel(st.page_path())}
    if notes:
        out["notes"] = notes
    err = _rebuild()
    if err:
        out["page_error"] = err
    return out


@tool(
    "status_summary",
    "Compact JSON of every stage's state per brain (not started / in progress / passing / failing / "
    "regressed), each measured check's latest value against its target, and recent key events. "
    "Read it to decide what to work on next.",
    {"type": "object", "properties": {
        "brain": {"type": "string", "description": "only this brain (claude or argon); default every brain with records"},
        "stage": {"type": "string", "description": "only this stage id"}}},
)
def status_summary(brain: str | None = None, stage: str | None = None) -> dict:
    return st.summary(brain=brain, stage=stage)
