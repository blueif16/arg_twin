"""Decision-point log for scene reconstruction (playbook/skills/reconstruct-scene).

At every reconstruction step the brain writes one record: what it looked at, its native hypothesis
(with numbers) before any precision tool, the tool it then called and what came back, the error in
physical units before and after, and what it decided. The log is the evidence for the claim
"the model's own look set the structure; tools supplied the precision".

log_decision      append one schema-checked record to runs/<brain>/<run_id>/decisions.jsonl
decision_summary  read a run's records back: per-step counts, the before/after table, and
                  optionally a native-vs-tools chart (PNG) for publish_media

Cheap by design: no simulator, no page rebuild, matplotlib only when a chart is asked for.
Records are append-only; an optional `id` must be unique in the file, so a backfill can be rerun safely.
"""

from __future__ import annotations

import fcntl
import json
import math
import re
import time
from pathlib import Path

from harness import status as st
from harness.registry import REPO, TOOLS, tool

RUNS_DIR = REPO / "runs"  # tests point this at a temp dir
SCHEMA = "decision/v1"

STEPS = ("intake", "robot_tool", "time_sync", "cameras", "object_geometry", "object_pose", "controller",
         "commands", "contact", "hidden_state", "validation", "integrity")
NATIVE_SOURCES = ("native", "prior", "log", "none")
VERDICTS = ("adopt", "reject", "revise", "escalate", "open")
# physical units, with the factor to the canonical unit used for charts (mm, deg, ms)
UNITS = {"mm": ("mm", 1.0), "cm": ("mm", 10.0), "m": ("mm", 1000.0),
         "deg": ("deg", 1.0), "rad": ("deg", 180.0 / math.pi),
         "ms": ("ms", 1.0), "s": ("ms", 1000.0), "frames": ("frames", 1.0),
         "px": ("px", 1.0), "N": ("N", 1.0), "Nm": ("Nm", 1.0), "N/m": ("N/m", 1.0), "kg": ("kg", 1.0),
         "fraction": ("fraction", 1.0), "count": ("count", 1.0)}
NON_PHYSICAL = ("fraction", "count")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
SECRET_RE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}")
MAX_TEXT = 600


def _text(s, field: str, required: bool = True, limit: int = MAX_TEXT) -> str:
    s = " ".join(str(s or "").split())
    if required and not s:
        raise ValueError(f"{field} is required")
    if len(s) > limit:
        raise ValueError(f"{field} is {len(s)} chars; keep it under {limit}")
    if SECRET_RE.search(s):
        raise ValueError(f"{field} looks like it contains a secret (rules.md: no secrets in records)")
    return s


def _num(v, field: str, allow_none: bool = True):
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{field} must be a finite number{' or null' if allow_none else ''}; got {v!r}")
    return float(v)


def _values(d, field: str) -> dict:
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError(f"{field} must be an object of name -> number")
    out = {}
    for k, v in d.items():
        k = _text(k, f"{field} key", limit=60)
        if isinstance(v, (list, tuple)):
            out[k] = [_num(x, f"{field}.{k}", False) for x in v][:8]
        else:
            out[k] = _num(v, f"{field}.{k}", False)
    if len(out) > 12:
        raise ValueError(f"keep {field} to the 12 numbers that matter")
    return out


def _refs(items, field: str, need_existing: bool) -> tuple[list[str], list[str]]:
    if isinstance(items, str):
        items = [items]
    refs, missing = [], []
    for e in items or []:
        e = str(e).strip()
        if not e:
            continue
        if e.startswith(str(REPO) + "/"):
            e = e[len(str(REPO)) + 1:]
        if SECRET_RE.search(e):
            raise ValueError(f"{field} looks like it contains a secret")
        refs.append(e)
        if not st.evidence_item(e)["exists"]:
            missing.append(e)
    if not refs:
        raise ValueError(f"{field} needs at least one repo path or URL")
    if need_existing and len(missing) == len(refs):
        raise ValueError(f"{field} must include at least one existing repo file or a URL; missing: {missing}")
    return refs, missing


def run_dir(brain: str, run_id: str) -> Path:
    b = st.brain_key(brain)
    if not b:
        raise ValueError("brain is required: 'claude' for a stand-in run, 'argon' for Argon")
    r = re.sub(r"[^\w.@+-]", "", str(run_id or "").strip())[:64]
    if not r:
        raise ValueError("run_id is required")
    return RUNS_DIR / b / r


def log_path(brain: str, run_id: str) -> Path:
    return run_dir(brain, run_id) / "decisions.jsonl"


def read_records(brain: str, run_id: str) -> list[dict]:
    p = log_path(brain, run_id)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


@tool(
    "log_decision",
    "Record one reconstruction decision point (skill reconstruct-scene): what you looked at, your native "
    "hypothesis with numbers BEFORE any precision tool, the tool you then called and its result, the error in "
    "physical units before and after, and what you decided. Appends a schema-checked line to "
    "runs/<brain>/<run_id>/decisions.jsonl. Call it once per step and quantity (camera pose, tool length, board "
    "pose, latency, object size, controller, contact parameters, in-hand pose, ...). If no tool was needed, "
    "omit tool and say why in decision. Before calling the precision tool, write the native estimate as a "
    "pre-registration (verdict open, no tool, no error); the call log's order then proves it came first.",
    {"type": "object",
     "required": ["brain", "run_id", "stage", "step", "quantity", "saw", "native", "result",
                  "verdict", "decision", "evidence"],  # error may be omitted: intake or pre-registration
     "properties": {
         "brain": {"type": "string", "description": "claude (stand-in) or argon"},
         "run_id": {"type": "string"},
         "stage": {"type": "string", "description": "validation.yaml stage id this feeds (A, B0, B1, B2, ...)"},
         "step": {"type": "string", "enum": list(STEPS)},
         "quantity": {"type": "string", "description": "what is being reconstructed, e.g. 'hama1 extrinsics', 'tool length'"},
         "saw": {"type": "array", "items": {"type": "string"},
                 "description": "repo paths of what you looked at: frames, grids, plots, signal JSON"},
         "native": {"type": "object", "required": ["source", "estimate"], "properties": {
             "source": {"type": "string", "enum": list(NATIVE_SOURCES),
                        "description": "native = your own estimate from images, signals or reasoning; prior = a "
                                       "supplied value (mocap, config, datasheet, simulator default); log = read "
                                       "straight from the recording; none = no guess"},
             "estimate": {"type": "string", "description": "the hypothesis in words, with numbers and units"},
             "values": {"type": "object", "description": "the hypothesis as name -> number (SI or the error unit)"}}},
         "tool": {"type": "object", "description": "omit when no tool was used (say why in decision)", "properties": {
             "name": {"type": "string", "description": "harness tool name, or a library function"},
             "call": {"type": "string", "description": "call reference, e.g. runs/calls/20261024.jsonl:120, or key args"}}},
         "result": {"type": "object", "required": ["summary"], "properties": {
             "summary": {"type": "string", "description": "what came back, with numbers"},
             "values": {"type": "object"}}},
         "error": {"type": "object", "description": "required except for step intake, or for a pre-registration "
                   "(verdict open, no tool) written before the tool runs", "required": ["metric", "unit", "before", "after", "reference"], "properties": {
             "metric": {"type": "string", "description": "what the error measures"},
             "unit": {"type": "string", "enum": list(UNITS)},
             "before": {"type": "number", "description": "error of the native hypothesis (omit if not measurable)"},
             "after": {"type": "number", "description": "error after the tool (omit if not measured yet)"},
             "reference": {"type": "string", "description": "what the error is measured against (held-out marks, contacts, ...)"},
             "heldout": {"type": "boolean", "description": "after is measured on data the fit never used"},
             "mm_per_px": {"type": "number", "description": "for px errors: mm per px at the working distance"}}},
         "verdict": {"type": "string", "enum": list(VERDICTS)},
         "decision": {"type": "string", "description": "what you decided and why, one or two sentences"},
         "evidence": {"type": "array", "items": {"type": "string"}, "description": "repo paths or URLs backing the numbers"},
         "label": {"type": "string", "description": "short chart label, at most 32 characters (default: quantity)"},
         "skill": {"type": "string", "description": "the specific skill whose method you followed (world-frame, ...)"},
         "check_ids": {"type": "array", "items": {"type": "string"}, "description": "validation check ids this affects"},
         "id": {"type": "string", "description": "optional stable slug, unique in this run's log"},
         "by": {"type": "string", "description": "agent name"},
         "backfilled": {"type": "boolean", "description": "true when written after the fact from records"},
         "decided_on": {"type": "string", "description": "for backfills: date of the original decision (YYYY-MM-DD)"}}},
)
def log_decision(brain: str, run_id: str, stage: str, step: str, quantity: str, saw, native: dict,
                 result: dict, verdict: str, decision: str, evidence, error: dict | None = None,
                 tool: dict | None = None,
                 label: str = "", skill: str = "", check_ids: list[str] | None = None, id: str = "", by: str = "",
                 backfilled: bool = False, decided_on: str = "") -> dict:
    val = st.load_validation()
    stage_ids = [s["id"] for s in val["stages"]]
    if stage not in stage_ids:
        raise ValueError(f"unknown stage {stage!r}; known: {stage_ids}")
    if step not in STEPS:
        raise ValueError(f"step must be one of {list(STEPS)}; got {step!r}")
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {list(VERDICTS)}; got {verdict!r}")
    path = log_path(brain, run_id)

    if not isinstance(native, dict):
        raise ValueError("native must be an object {source, estimate, values?}")
    nsrc = native.get("source")
    if nsrc not in NATIVE_SOURCES:
        raise ValueError(f"native.source must be one of {list(NATIVE_SOURCES)}")
    nest = _text(native.get("estimate"), "native.estimate", required=nsrc != "none")
    nvals = _values(native.get("values"), "native.values")
    if nsrc != "none" and not nvals and not re.search(r"\d", nest):
        raise ValueError("native hypothesis needs numbers: put them in native.estimate or native.values")

    if tool is not None:
        if not isinstance(tool, dict) or not str(tool.get("name") or "").strip():
            raise ValueError("tool must be null or an object {name, call?}")
        tname = _text(tool.get("name"), "tool.name", limit=80)
        tool = {"name": tname, "call": _text(tool.get("call"), "tool.call", required=False, limit=200),
                "registered": tname in TOOLS}

    if not isinstance(result, dict):
        raise ValueError("result must be an object {summary, values?}")
    result = {"summary": _text(result.get("summary"), "result.summary"),
              "values": _values(result.get("values"), "result.values")}

    # error may be null for an intake inventory, or for a pre-registration: the native estimate written down
    # (verdict open, no tool yet) before the precision tool runs. The call log's order shows it came first.
    pre_registration = verdict == "open" and tool is None
    warnings = []
    if error is None and (step == "intake" or pre_registration):
        err = None
    else:
        if not isinstance(error, dict):
            raise ValueError("error must be an object {metric, unit, before, after, reference}; it may be null "
                             "only for step intake or a pre-registration (verdict open, tool null)")
        unit = error.get("unit")
        if unit not in UNITS:
            raise ValueError(f"error.unit must be one of {list(UNITS)}; got {unit!r}")
        err = {"metric": _text(error.get("metric"), "error.metric", limit=200), "unit": unit,
               "before": _num(error.get("before"), "error.before"), "after": _num(error.get("after"), "error.after"),
               "reference": _text(error.get("reference"), "error.reference", limit=300),
               "heldout": bool(error.get("heldout", False))}
        if err["before"] is None and err["after"] is None:
            raise ValueError("error needs before or after (or both); use null only for the one not measured")
        if "mm_per_px" in error and error["mm_per_px"] is not None:
            if unit != "px":
                raise ValueError("mm_per_px only applies to px errors")
            err["mm_per_px"] = _num(error["mm_per_px"], "error.mm_per_px", False)
        if unit == "px" and "mm_per_px" not in err:
            warnings.append("px error without mm_per_px: give the conversion at the working distance so it reads in mm")
        if unit in NON_PHYSICAL:
            warnings.append(f"unit {unit!r} is not physical; use it only for rates and counts (validation, integrity)")

    saw_refs, saw_missing = _refs(saw, "saw", need_existing=False)
    ev_refs, ev_missing = _refs(evidence, "evidence", need_existing=True)
    checks = [_text(c, "check_id", limit=60) for c in (check_ids or [])]
    known = {c["id"] for s in val["stages"] for c in s.get("checks", [])}
    bad = [c for c in checks if c not in known]
    if bad:
        raise ValueError(f"unknown check ids {bad}")
    rid = str(id or "").strip()
    if rid and not ID_RE.match(rid):
        raise ValueError("id must be a lowercase slug (a-z, 0-9, . _ -), at most 64 chars")
    if decided_on and not re.match(r"^\d{4}-\d{2}-\d{2}$", decided_on):
        raise ValueError("decided_on must be YYYY-MM-DD")

    now = time.time()
    rec = {"schema": SCHEMA, "t": now, "ts": st.iso(now), "id": rid or None,
           "brain": st.brain_key(brain), "run_id": path.parent.name, "stage": stage, "step": step,
           "quantity": _text(quantity, "quantity", limit=120),
           "label": _text(label, "label", required=False, limit=32) or None, "saw": saw_refs,
           "native": {"source": nsrc, "estimate": nest, "values": nvals}, "tool": tool, "result": result,
           "error": err, "verdict": verdict, "decision": _text(decision, "decision"),
           "evidence": ev_refs, "skill": _text(skill, "skill", required=False, limit=64) or None,
           "check_ids": checks, "by": re.sub(r"[^\w:.@/+-]", "", str(by or ""))[:64],
           "backfilled": bool(backfilled), "decided_on": decided_on or None,
           "validation_version": val["version"]}

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # parallel agents may log to the same run
        f.seek(0)
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
        if rid:
            for ln in lines:
                try:
                    if json.loads(ln).get("id") == rid:
                        raise ValueError(f"id {rid!r} already logged in {st.rel(path)}; records are append-only")
                except json.JSONDecodeError:
                    continue
        rec["seq"] = len(lines) + 1
        f.seek(0, 2)
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    out = {"ok": True, "seq": rec["seq"], "id": rec["id"], "path": st.rel(path), "step": step,
           "change": {k: err[k] for k in ("unit", "before", "after")} if err else None}
    if saw_missing:
        out["saw_missing"] = saw_missing
    if ev_missing:
        out["missing_evidence"] = ev_missing
    if warnings:
        out["warnings"] = warnings
    return out


def canonical(err: dict) -> tuple[str, float | None, float | None]:
    """(canonical unit, before, after): px with mm_per_px becomes mm; s -> ms, rad -> deg, cm/m -> mm."""
    unit = err.get("unit")
    if unit == "px" and err.get("mm_per_px"):
        f = err["mm_per_px"]
        return "mm", (err["before"] * f if err.get("before") is not None else None), \
            (err["after"] * f if err.get("after") is not None else None)
    cu, f = UNITS.get(unit, (unit, 1.0))
    return cu, (err["before"] * f if err.get("before") is not None else None), \
        (err["after"] * f if err.get("after") is not None else None)


def _chart(rows: list[dict], out: Path, title: str, subtitle: str = "") -> str:
    """Dumbbell per quantity: native estimate (or supplied prior) -> after tools, one log axis per unit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    units = [u for u in ("mm", "deg", "ms") if any(r["unit"] == u for r in rows)]
    if not units:
        raise ValueError("no records with both errors in mm, deg or ms to chart")
    n = [sum(r["unit"] == u for r in rows) for u in units]
    row_in, axis_in, head_in, width_in = 0.5, 0.95, 1.25, 8.4
    fig_h = head_in + sum(k * row_in + axis_in for k in n) - 0.25
    fig = plt.figure(figsize=(width_in, fig_h))
    c_native, c_tool, c_line, ink, ink2 = "#86b6ef", "#184f95", "#c3c2b7", "#29271f", "#625c4b"
    unit_label = {"mm": "error, mm (log scale)", "deg": "error, degrees (log scale)", "ms": "error, ms (log scale)"}
    left, right = 3.05 / width_in, 0.97
    y_top = fig_h - head_in
    for u, k in zip(units, n):
        h = k * row_in
        ax = fig.add_axes((left, (y_top - h) / fig_h, right - left, h / fig_h))
        y_top -= h + axis_in
        rs = sorted([r for r in rows if r["unit"] == u], key=lambda r: r["before"])
        for y, r in enumerate(rs):
            ax.plot([r["after"], r["before"]], [y, y], color=c_line, lw=2.5, zorder=1, solid_capstyle="round")
            prior = r["native_source"] != "native"
            ax.scatter([r["before"]], [y], s=90, zorder=3, marker="o", linewidth=2.2,
                       facecolor="white" if prior else c_native, edgecolor=c_native if prior else "white")
            held = bool(r.get("heldout"))
            ax.scatter([r["after"]], [y], s=90 if held else 80, zorder=3, marker="o" if held else "D",
                       facecolor=c_tool, edgecolor="white", linewidth=1.5)
            ax.annotate(f"{r['before']:.3g}", (r["before"], y), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=10.5, color=ink2)
            ax.annotate(f"{r['after']:.3g}", (r["after"], y), xytext=(-9, 0), textcoords="offset points",
                        va="center", ha="right", fontsize=10.5, color=ink)
        ax.set_yticks(range(len(rs)))
        ax.set_yticklabels([r["label"] for r in rs], fontsize=11, color=ink)
        ax.set_xscale("log")
        lo = min(min(r["after"], r["before"]) for r in rs)
        hi = max(max(r["after"], r["before"]) for r in rs)
        ax.set_xlim(lo / 4.0, hi * 3.5)
        ax.set_ylim(-0.6, len(rs) - 0.4)
        ax.set_xlabel(unit_label[u], fontsize=10.5, color=ink2)
        ax.grid(axis="x", which="major", color="#e5e4dc", lw=0.9)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#d7cfb6")
        ax.tick_params(axis="y", length=0, pad=6)
        ax.tick_params(axis="x", colors=ink2, labelsize=10)
    fig.text(0.012, 1 - 0.22 / fig_h, title, ha="left", va="top", fontsize=13.5, color=ink, weight="bold")
    if subtitle:
        fig.text(0.012, 1 - 0.55 / fig_h, subtitle, ha="left", va="top", fontsize=10.5, color=ink2)
    handles = [Line2D([], [], marker="o", ls="", markersize=9, markerfacecolor=c_native, markeredgecolor="white",
                      label="native estimate"),
               Line2D([], [], marker="o", ls="", markersize=9, markerfacecolor="white", markeredgecolor=c_native,
                      markeredgewidth=2.2, label="supplied prior"),
               Line2D([], [], marker="o", ls="", markersize=9, markerfacecolor=c_tool, markeredgecolor="white",
                      label="after tools, held-out"),
               Line2D([], [], marker="D", ls="", markersize=8, markerfacecolor=c_tool, markeredgecolor="white",
                      label="after tools, fit or spread")]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=10.5, labelcolor=ink,
               bbox_to_anchor=(0.004, 1 - 0.8 / fig_h), handletextpad=0.3, columnspacing=1.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return st.rel(out)


@tool(
    "decision_summary",
    "Read a run's reconstruction decision log back (runs/<brain>/<run_id>/decisions.jsonl): records per step, "
    "and the before -> after error table in physical units. chart=true also writes a native-vs-tools PNG "
    "(one row per quantity with both errors in mm, degrees or ms; px errors are converted with mm_per_px) "
    "ready for publish_media.",
    {"type": "object", "required": ["brain", "run_id"], "properties": {
        "brain": {"type": "string"}, "run_id": {"type": "string"},
        "step": {"type": "string", "description": "only this step"},
        "chart": {"type": "boolean", "description": "write the native-vs-tools chart (default false)"},
        "chart_ids": {"type": "array", "items": {"type": "string"},
                      "description": "only these record ids in the chart, in any order (default: all with both errors)"},
        "title": {"type": "string"}, "subtitle": {"type": "string"}}},
)
def decision_summary(brain: str, run_id: str, step: str | None = None, chart: bool = False,
                     chart_ids: list[str] | None = None, title: str = "", subtitle: str = "") -> dict:
    recs = read_records(brain, run_id)
    if step:
        recs = [r for r in recs if r.get("step") == step]
    by_step: dict[str, int] = {}
    table = []
    for r in recs:
        by_step[r.get("step")] = by_step.get(r.get("step"), 0) + 1
        e = r.get("error") or {}
        cu, b, a = canonical(e)
        table.append({"seq": r.get("seq"), "id": r.get("id"), "step": r.get("step"), "quantity": r.get("quantity"),
                      "native_source": (r.get("native") or {}).get("source"),
                      "tool": (r.get("tool") or {}).get("name"), "unit": cu, "before": b, "after": a,
                      "label": r.get("label") or (r.get("quantity") or "")[:40],
                      "heldout": e.get("heldout"), "verdict": r.get("verdict")})
    out = {"path": st.rel(log_path(brain, run_id)), "n": len(recs), "by_step": by_step,
           "backfilled": sum(1 for r in recs if r.get("backfilled")), "table": table}
    if chart:
        rows = [t for t in table if t["before"] is not None and t["after"] is not None
                and t["before"] > 0 and t["after"] > 0 and t["unit"] in ("mm", "deg", "ms")]
        if chart_ids:
            want = {c: i for i, c in enumerate(chart_ids)}
            rows = [t for t in rows if t["id"] in want]
        b = st.brain_key(brain)
        png = run_dir(brain, run_id) / "reconstruct" / "native_vs_tools.png"
        out["chart"] = _chart(rows, png, title or "Native estimate vs native estimate plus tools",
                              subtitle or f"{st.BRAIN_LABELS.get(b, b)}, {run_id}")
        out["chart_rows"] = len(rows)
    return out
