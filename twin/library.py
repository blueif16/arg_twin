"""Learned-routine library for stage E, COMPILE (System 2 -> System 1; skill playbook/skills/compile-skill).

A routine is a parameterized skill program (one checker-clean skill.py plus a frozen params.json) compiled from a
brain's practice successes. It runs in the skill sandbox against envapi with ZERO brain calls, either standalone
(it finishes the episode itself: skill_run, promotion_gate) or invoked from a brain episode (routine_run: it never
finishes, it hands control back with a code).

Layout (task findings: they stay with their brain and are never handed to another brain's run):

  runs/<brain>/<run_id>/library/
    index.json                       routines -> versions, status, active version
    events.jsonl                     add / smoke / gate / promote / demote / stale, one line each
    invocations.jsonl                every routine_run from a brain episode (sim time under the routine, commands, code)
    sessions/<session>.json          per-episode routine memory (surface touch, hand frame, references)
    <name>/<version>/skill.py        the program (frozen; hashed in meta.json)
    <name>/<version>/params.json     defaults, per-part overrides, the tunable keys a brain may override
    <name>/<version>/meta.json       sources, decision structure, parts, hand-back codes, hashes, smoke result
    <name>/<version>/gate/*.json     every promotion_gate result (pilot | scored | review)

Status: candidate (added, smoke-tested) -> piloted (pilot gate on practice starts passed) -> promoted (scored gate on
frozen held-out starts and cousins passed, run by a watcher) -> demoted (failed later) | stale (its code, params or the
evaluator changed since the gate). A demoted or stale version is never invoked; a fix is a new version.

This module imports no simulator code; the harness tools in harness/tools/compile.py run the episodes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np

from twin import REPO_DIR

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,48}$")
STATUSES = ("candidate", "piloted", "promoted", "demoted", "stale", "rejected")
INVOCABLE = ("candidate", "piloted", "promoted")
HANDBACK_CODES = {
    "OK": "seated (estimated from TCP travel past the surface touch) and, if asked, released",
    "NO_CONTACT": "a guarded descent ran its full distance without touching: the part or board is not where any prior "
                  "puts it",
    "UNEXPECTED_CONTACT": "touched something where free space was expected, or the surface is outside the height prior",
    "NOT_FOUND": "the probe pattern was used up without the tip dropping in: the hole is outside the routine's search "
                 "area (a perception decision for the brain)",
    "JAMMED": "the push stalled short of the seat after the wiggles",
    "NOT_HOLDING": "the gripper is open or empty",
    "ENVELOPE_TIME": "another probe would take the episode past the real duration envelope (twin/envelope.py)",
    "BUDGET": "the routine's own command or sim-time budget is used up",
    "PULLED": "the retreat after opening met a force: the part may have come with the fingers",
    "EPISODE_OVER": "the env ended the episode (stall, budget)",
    "ENV_ERROR": "the env refused a command",
    "BAD_ARGS": "a parameter the brain passed is not tunable or out of range",
    "SMOKE": "smoke run only (setup, no motion)",
}


# ================================================================================================ paths, hashes
def lib_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "library"


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO_DIR)) if p.is_relative_to(REPO_DIR) else str(p)


def run_of(path: Path) -> tuple[str, str] | None:
    """(brain, run_id) of a path under runs/<brain>/<run_id>/, else None."""
    try:
        parts = Path(path).resolve().relative_to(REPO_DIR / "runs").parts
    except ValueError:
        return None
    return (parts[0], parts[1]) if len(parts) >= 2 else None


def same_brain(path: Path, brain: str, what: str):
    """Task findings never cross brains (orchestrator.md): refuse a source from another brain's runs."""
    r = run_of(path)
    if r is None:
        raise PermissionError(f"{what} {rel(path)} must live under runs/{brain}/<run_id>/ (a run of the same brain)")
    if r[0] != brain:
        raise PermissionError(f"{what} {rel(path)} belongs to brain {r[0]!r}; a {brain!r} library never reads another "
                              "brain's runs (orchestrator.md: task findings stay with their brain)")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _event(run_dir: Path, kind: str, **kw):
    d = lib_dir(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "events.jsonl", "a") as fh:
        fh.write(json.dumps(dict(t=_now(), kind=kind, **kw), default=str) + "\n")


def load_index(run_dir: Path) -> dict:
    f = lib_dir(run_dir) / "index.json"
    return json.loads(f.read_text()) if f.exists() else dict(routines={}, created=_now())


def save_index(run_dir: Path, idx: dict):
    d = lib_dir(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    idx["updated"] = _now()
    (d / "index.json").write_text(json.dumps(idx, indent=1))


# ================================================================================================ decision structure
def _traj_at(traj: np.ndarray, t: float) -> np.ndarray:
    i = int(np.searchsorted(traj[:, 0], t))
    return traj[min(max(i, 0), len(traj) - 1)]


def trace_phases(rec: dict, engage_mm: float = 2.0) -> dict:
    """The decision structure of one finished episode, read from its command log and 30 Hz trajectory only.
    Every command becomes a phase: hover, wait, touch (the first guarded stop: the surface reference), lift, shift,
    probe_rim / probe_in (a guarded descent stopping above / more than engage_mm below the touch height), push,
    release, retreat. World z is up; the hole axis points down on this board."""
    traj = np.asarray(rec["trajectory"]["rows"], float)
    out, z_touch, released, prev_t, first_move = [], None, False, 0.0, True
    for c in rec["commands"]:
        req, res = c["req"], (c.get("result") or {})
        t = float(c["t"])
        p0 = _traj_at(traj, prev_t)[1:4]
        p1 = _traj_at(traj, t)[1:4]
        prev_t = t
        op = req.get("op")
        ph = dict(t=round(t, 2), op=op)
        if op == "gripper":
            ph["phase"] = "release" if req.get("action") in ("open", "width") else "grip"
            released = released or ph["phase"] == "release"
        elif op == "wait":
            ph["phase"] = "wait"
        elif op == "move":
            tgt = np.asarray(req["pos"], float)
            d = tgt - p0
            dz, dxy = 1000 * float(d[2]), 1000 * float(np.linalg.norm(d[:2]))
            guarded = req.get("stop_force") is not None
            ph.update(speed_mm_s=round(1000 * float(req.get("speed", 0.02)), 1), force_max_N=req.get("force_max"),
                      stop_force_N=req.get("stop_force"), stopped=res.get("stopped"), dz_cmd_mm=round(dz, 2),
                      dxy_cmd_mm=round(dxy, 2))
            if first_move:
                ph["phase"] = "hover"
            elif released:
                ph["phase"] = "retreat"
            elif dz < -1.0 and dxy < max(2.0, abs(dz)):
                if z_touch is None:
                    if res.get("stopped") == "stop_force":
                        z_touch = float(p1[2])
                        ph["phase"] = "touch"
                    else:
                        ph["phase"] = "descend"
                else:
                    depth = 1000 * (z_touch - float(p1[2]))
                    ph["depth_mm"] = round(depth, 2)
                    prev = out[-1]["phase"] if out else ""
                    if depth > engage_mm:
                        ph["phase"] = "push" if prev in ("probe_in", "push") else "probe_in"
                    else:
                        ph["phase"] = "probe_rim"
            elif dz > 1.0 and dxy < max(2.0, dz):
                ph["phase"] = "lift"
            elif dxy >= 0.3:
                ph["phase"] = "shift"
                ph["shift_mm"] = round(dxy, 2)
            else:
                ph["phase"] = "adjust"
            first_move = False
        else:
            ph["phase"] = op
        out.append(ph)
    seq = [p["phase"] for p in out if p["phase"] not in ("wait", "adjust", "grip")]
    # canonical structure: positioning before the first touch is 'approach'; everything between the first touch and
    # the LAST entry (rim probes, lifts, shifts, entries that were lifted out again) is one 'search'; a push continues
    # the entry; repeats collapse
    canon: list[str] = []

    def put(x):
        if not canon or canon[-1] != x:
            canon.append(x)
    i_t = seq.index("touch") if "touch" in seq else None
    last_in = max([i for i, x in enumerate(seq) if x == "probe_in"], default=None)
    head = seq if i_t is None else seq[:i_t]
    for x in head:
        put(x if x in ("hover", "release", "retreat") else "approach")
    if i_t is not None:
        put("touch")
        if last_in is not None and last_in > i_t:
            if seq[i_t + 1:last_in]:
                put("search")
            put("probe_in")
            for x in seq[last_in + 1:]:
                if x != "push":
                    put(x if x in ("release", "retreat") else "search")
        elif seq[i_t + 1:]:
            for x in seq[i_t + 1:]:
                put(x if x in ("release", "retreat") else "search")
    n_probe = sum(1 for p in out if p["phase"] in ("probe_rim", "probe_in"))
    return dict(phases=out, sequence=seq, structure=" > ".join(canon), probes=n_probe,
                rim_probes=sum(1 for p in out if p["phase"] == "probe_rim"),
                outcome=rec.get("outcome"), counted=rec.get("counted"))


def extract(records: list[dict], min_successes: int = 3) -> dict:
    """Is a decision pattern ready to compile? Same canonical structure over >= min_successes counted successes, plus
    the parameter ranges the brain used per phase (speeds, force ceilings, stop forces, lifts and shifts)."""
    rows = []
    for rec in records:
        tp = trace_phases(rec)
        rows.append(dict(start_id=rec["start_id"], part=rec["part"], outcome=rec["outcome"], counted=rec["counted"],
                         structure=tp["structure"], probes=tp["probes"], rim_probes=tp["rim_probes"],
                         n_commands=rec["n_commands"], sim_s=rec["sim_s"], phases=tp["phases"]))
    succ = [r for r in rows if r["counted"]]
    by: dict[str, list[str]] = {}
    for r in succ:
        by.setdefault(r["structure"], []).append(r["start_id"])
    best = max(by.items(), key=lambda kv: len(kv[1])) if by else (None, [])
    params: dict[str, dict] = {}
    for r in succ:
        if r["structure"] != best[0]:
            continue
        for p in r["phases"]:
            for k in ("speed_mm_s", "force_max_N", "stop_force_N", "shift_mm", "dz_cmd_mm"):
                v = p.get(k)
                if isinstance(v, (int, float)):
                    params.setdefault(p["phase"], {}).setdefault(k, []).append(float(v))
    ranges = {ph: {k: dict(min=round(min(v), 3), median=round(float(np.median(v)), 3), max=round(max(v), 3), n=len(v))
                   for k, v in d.items()} for ph, d in params.items()}
    ready = len(best[1]) >= min_successes
    return dict(
        episodes=[{k: v for k, v in r.items() if k != "phases"} for r in rows], structures=by,
        common_structure=best[0], n_common=len(best[1]), ready=ready,
        why=(f"{len(best[1])} counted successes share '{best[0]}'" + ("" if ready else
             f"; need {min_successes} (compile-skill: repeated successes with the same decision structure)")),
        parameter_ranges=ranges, phases={r["start_id"]: r["phases"] for r in rows})


# ================================================================================================ versions
def _version_dir(run_dir: Path, name: str, version: str) -> Path:
    return lib_dir(run_dir) / name / version


def add_version(run_dir: Path, name: str, src: Path, meta: dict) -> dict:
    """Copy a checked program (skill.py + params.json) into the library as the next version, frozen by hash."""
    if not NAME_RE.match(name):
        raise ValueError(f"routine name {name!r}: lowercase, digits and _ only, 3-49 characters")
    src = Path(src)
    for f in ("skill.py", "params.json"):
        if not (src / f).is_file():
            raise FileNotFoundError(f"{rel(src / f)} missing: a routine is skill.py plus params.json")
    extra = sorted(p.name for p in src.iterdir() if p.is_file() and p.name not in ("skill.py", "params.json"))
    params = json.loads((src / "params.json").read_text())
    if not isinstance(params.get("defaults"), dict):
        raise ValueError("params.json needs a 'defaults' object (and optional 'per_part', 'tunable')")
    idx = load_index(run_dir)
    r = idx["routines"].setdefault(name, dict(versions={}, active=None))
    n = 1 + max([int(v[1:]) for v in r["versions"]] or [0])
    version = f"v{n}"
    d = _version_dir(run_dir, name, version)
    d.mkdir(parents=True, exist_ok=False)
    shutil.copy2(src / "skill.py", d / "skill.py")
    params["version"] = version
    params["routine"] = name
    (d / "params.json").write_text(json.dumps(params, indent=1))
    files = {f: sha256(d / f) for f in ("skill.py", "params.json")}
    meta = dict(meta, routine=name, version=version, created=_now(), files=files, ignored_files=extra,
                status="candidate", handback_codes=HANDBACK_CODES)
    (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    r["versions"][version] = dict(status="candidate", created=meta["created"], parts=meta.get("parts"),
                                  files=files, path=rel(d))
    save_index(run_dir, idx)
    _event(run_dir, "add", routine=name, version=version, files=files, by=meta.get("compiled_by"))
    return dict(routine=name, version=version, dir=rel(d), files=files)


def entry(run_dir: Path, name: str, version: str | None = None) -> dict:
    """A version's meta with its frozen-file check. version None = the active one (latest promoted, else latest
    piloted, else latest candidate)."""
    idx = load_index(run_dir)
    r = idx["routines"].get(name)
    if r is None:
        raise KeyError(f"no routine {name!r} in {rel(lib_dir(run_dir))} (library_list shows what exists)")
    if version is None:
        version = active_version(r)
        if version is None:
            raise KeyError(f"routine {name!r} has no invocable version (all demoted, stale or rejected)")
    if version not in r["versions"]:
        raise KeyError(f"routine {name!r} has no version {version!r}: {sorted(r['versions'])}")
    d = _version_dir(run_dir, name, version)
    meta = json.loads((d / "meta.json").read_text())
    now = {f: sha256(d / f) for f in meta["files"]}
    changed = [f for f, h in meta["files"].items() if now.get(f) != h]
    return dict(meta=meta, dir=d, version=version, status=r["versions"][version]["status"],
                frozen=dict(ok=not changed, changed=changed))


def active_version(r: dict) -> str | None:
    vs = r["versions"]
    order = sorted(vs, key=lambda v: int(v[1:]))
    for want in ("promoted", "piloted", "candidate"):
        cands = [v for v in order if vs[v]["status"] == want]
        if cands:
            return cands[-1]
    return None


def set_status(run_dir: Path, name: str, version: str, status: str, why: str, evidence: list | None = None,
               by: str | None = None, extra: dict | None = None):
    if status not in STATUSES:
        raise ValueError(f"status {status!r} not in {STATUSES}")
    idx = load_index(run_dir)
    v = idx["routines"][name]["versions"][version]
    old = v["status"]
    v["status"] = status
    v.setdefault("history", []).append(dict(t=_now(), from_=old, to=status, why=why, evidence=evidence or [], by=by))
    if extra:
        v.update(extra)
    idx["routines"][name]["active"] = active_version(idx["routines"][name])
    save_index(run_dir, idx)
    d = _version_dir(run_dir, name, version)
    meta = json.loads((d / "meta.json").read_text())
    meta["status"] = status
    (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    _event(run_dir, {"promoted": "promote", "demoted": "demote", "stale": "stale"}.get(status, "status"),
           routine=name, version=version, from_status=old, to=status, why=why, evidence=evidence or [], by=by)
    return dict(routine=name, version=version, from_status=old, status=status)


# ================================================================================================ invocations
def log_invocation(run_dir: Path, row: dict):
    d = lib_dir(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "invocations.jsonl", "a") as fh:
        fh.write(json.dumps(dict(t_wall=time.time(), **row), default=str) + "\n")


def session_memory(run_dir: Path, session: str, start_id: str | None, update: dict | None = None) -> dict:
    """Per-episode routine memory (surface touch, hand frame, references), reset when the episode changes."""
    f = lib_dir(run_dir) / "sessions" / f"{session}.json"
    mem = json.loads(f.read_text()) if f.exists() else {}
    if mem.get("start_id") != start_id:
        mem = dict(start_id=start_id, memory={})
    if update is not None:
        mem["memory"] = update
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(mem, indent=1))
    return mem["memory"]


def usage(run_dir: Path) -> dict:
    """Who was in control, per brain episode: sim time under routines vs the episode's sim time (D.brain_share =
    1 - routine share), and brain decisions spent on routine calls. Standalone routine episodes (skill_run, gates)
    have brain_share 0 by construction."""
    f = lib_dir(run_dir) / "invocations.jsonl"
    inv = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
    eps = {}
    ef = Path(run_dir) / "env" / "episodes.jsonl"
    if ef.exists():
        for line in ef.read_text().splitlines():
            r = json.loads(line)
            eps[(r["session"], r["start_id"])] = r
    per: dict = {}
    for r in inv:
        k = (r.get("session"), r.get("start_id"))
        e = per.setdefault(k, dict(session=k[0], start_id=k[1], routine_calls=0, routine_sim_s=0.0, codes=[],
                                   routines=set()))
        e["routine_calls"] += 1
        e["routine_sim_s"] += max(0.0, float(r.get("sim_t1") or 0) - float(r.get("sim_t0") or 0))
        e["codes"].append(r.get("code"))
        e["routines"].add(f"{r.get('routine')}@{r.get('version')}")
    rows = []
    for k, e in per.items():
        ep = eps.get(k)
        sim = float(ep["sim_s"]) if ep else None
        share = None if not sim else round(1.0 - min(1.0, e["routine_sim_s"] / sim), 3)
        rows.append(dict(e, routines=sorted(e["routines"]), routine_sim_s=round(e["routine_sim_s"], 2),
                         episode_sim_s=sim, outcome=ep["outcome"] if ep else "running", counted=ep["counted"] if ep else None,
                         brain_share=share))
    return dict(episodes=rows, n_invocations=len(inv),
                note="brain_share = 1 - (sim s under routine_run calls) / episode sim s; each routine_run is one brain "
                     "decision; standalone routine episodes have brain_share 0")


# ================================================================================================ statistics
def clopper_pearson(k: int, n: int, conf: float = 0.95) -> dict:
    from scipy.stats import beta
    if n == 0:
        return dict(k=0, n=0, rate=None, ci95=None)
    a = 1 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return dict(k=k, n=n, rate=round(k / n, 4), ci95=[round(lo, 4), round(hi, 4)])


def e_targets() -> dict:
    """E and G targets from validation.yaml as they stand now (the gate records which version it used)."""
    import yaml
    v = yaml.safe_load((REPO_DIR / "playbook" / "validation.yaml").read_text())
    out = dict(version=v["version"])
    for st in v["stages"]:
        if st["id"] in ("E", "G"):
            for c in st["checks"]:
                out[c["id"]] = dict(direction=c["direction"], target=c.get("target"))
    return out


def routine_result_from_stderr(path: Path) -> dict | None:
    """The routine's own report (the last ROUTINE_RESULT line it printed to stderr)."""
    p = Path(path)
    if not p.exists():
        return None
    res = None
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("ROUTINE_RESULT "):
            try:
                res = json.loads(line[len("ROUTINE_RESULT "):])
            except json.JSONDecodeError:
                pass
    return res


def _fmt(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x
