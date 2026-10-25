"""Stage E, COMPILE: System 2 -> System 1 (skill playbook/skills/compile-skill).

What a brain worked out in practice becomes a routine that runs with ZERO brain calls: a parameterized skill program
(checker-clean skill.py + frozen params.json) in the skill sandbox, talking only envapi. The library is a task finding:
runs/<brain>/<run_id>/library/, never read from another brain's run (twin/library.py).

  library_extract   the decision structure of a practice run's successes (phases from the command log), the parameter
                    ranges the brain used, and whether a pattern is ready to compile (>= 3 successes, same structure)
  library_add       add a routine version: static check, sandbox smoke run (boots, reads its params, speaks the env
                    protocol), frozen by hash; status candidate
  library_list      routines, versions, status, gate results, frozen check; usage=true adds who was in control per
                    brain episode (D.brain_share) from the routine_run log
  routine_run       invoke a library routine inside a brain's live episode: it runs sandboxed with zero brain calls and
                    hands control back with a code (OK, NOT_FOUND, JAMMED, ...); logged for D.brain_share and calls
  promotion_gate    the deterministic gate. pilot: practice starts + practice-start x cousin pairs (anyone; never
                    promotes). scored: frozen held-out starts + the frozen cousin pairs (watchers only; promotes or
                    rejects). review: demotes or marks stale a version whose later record, code or evaluator changed.
                    It never records checks: it suggests E/G values for a watcher's record_check.
"""

from __future__ import annotations

import json
import os
import select
import time
from pathlib import Path

from harness.registry import REPO, current_caller, tool

_RUN = {"brain": {"type": "string", "default": "claude", "description": "claude (stand-in) or argon"},
        "run_id": {"type": "string", "default": "run-compile-pilot-1", "description": "runs/<brain>/<run_id>/: the "
                   "run whose library/ holds the routines"}}
INVOKE_OPS = ("task", "observe", "status", "move", "gripper", "wait")   # an invoked routine never finishes the episode
WATCHERS = ("watcher", "evaluator", "exam-scorer")
NEXT_HINT = {
    "OK": "seated by the routine's own estimate; confirm with env_observe if you need to, then env_finish with your claim",
    "NOT_FOUND": "the hole is outside the routine's search area: look (env_observe_image) or reason about where it is, "
                 "then call routine_run again with center_mm (believed hole frame, mm) or act yourself",
    "JAMMED": "the part stalled short of the seat: decide whether to lift and retry, re-aim, or act yourself",
    "ENVELOPE_TIME": "the routine stopped before leaving the real duration envelope; decide whether to continue",
    "NO_CONTACT": "no surface where any prior puts it: check the grasp and the board before anything else",
    "UNEXPECTED_CONTACT": "touched something unexpected: look before moving again",
    "NOT_HOLDING": "the part is not in the gripper",
    "PULLED": "the retreat met a force after opening: check whether the part came with the fingers",
    "BUDGET": "the routine used its own budget; decide what next",
    "EPISODE_OVER": "the env ended the episode",
}


def _run_dir(brain: str, run_id: str) -> Path:
    from harness.tools.env import _name
    return REPO / "runs" / _name(brain, "brain") / _name(run_id, "run_id")


def _repo(p: str) -> Path:
    q = (REPO / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
    if not q.is_relative_to(REPO / "runs"):
        raise PermissionError(f"{p}: library sources live under runs/<brain>/<run_id>/")
    if not q.exists():
        raise FileNotFoundError(p)
    return q


def _rel(p) -> str:
    from twin.library import rel
    return rel(p)


# ================================================================================================ the pump
def _pump(tmp: Path, entry: str, send, wall_s: float, stderr_path: Path, cpu_s: int = 600) -> dict:
    """Run a prepared sandbox workspace and relay its envapi requests through send(req) -> reply. Only INVOKE_OPS
    pass. Returns the routine's result.json (or its last ROUTINE_RESULT line) and pump stats."""
    from twin import sandbox as SB
    from twin.library import routine_result_from_stderr
    proc, mode = SB.popen(tmp, entry, cpu_s=cpu_s, stderr_path=stderr_path)
    t0, n_req, refused, why = time.time(), 0, 0, None
    try:
        while True:
            left = wall_s - (time.time() - t0)
            if left <= 0:
                why = "wall_timeout"
                break
            r, _, _ = select.select([proc.stdout], [], [], min(left, 5.0))
            if not r:
                if proc.poll() is not None:
                    why = f"exit {proc.returncode}"
                    break
                continue
            line = proc.stdout.readline()
            if not line:
                why = f"exit {proc.wait()}"
                break
            n_req += 1
            try:
                req = json.loads(line) if len(line) <= 65536 else None
            except json.JSONDecodeError:
                req = None
            if not isinstance(req, dict):
                rep = dict(error="bad request")
            elif req.get("op") not in INVOKE_OPS:
                refused += 1
                rep = dict(error=f"op {req.get('op')!r} is not allowed for an invoked routine (allowed: "
                                 f"{list(INVOKE_OPS)}); the brain finishes the episode")
            else:
                rep = send(req)
            try:
                proc.stdin.write((json.dumps(rep, default=float) + "\n").encode())
                proc.stdin.flush()
            except BrokenPipeError:
                why = "program closed its input"
                break
    finally:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        proc.wait()
    res = None
    f = tmp / "ws" / "result.json"
    if f.exists():
        try:
            res = json.loads(f.read_text())
        except json.JSONDecodeError:
            res = None
    if res is None:
        res = routine_result_from_stderr(stderr_path)
    return dict(result=res, sandbox=mode, requests=n_req, refused_ops=refused, program_end=why,
                wall_s=round(time.time() - t0, 2))


INVOKE_LOGGED = {"move": "env_move", "gripper": "env_gripper", "wait": "env_wait", "observe": "env_observe",
                 "status": "env_status"}   # the read-only `task` op has no env_* tool and is not logged


def _logged_relay(brain: str, run_id: str, session: str, caller: str, parent: dict):
    """send(req) for _pump: relay an invoked routine's request to the session daemon AND append it to the call log as
    its own env_* line, in the shape harness.registry.call writes for env_move / env_gripper / env_wait / env_observe /
    env_status (same caller, session, brain and run_id) plus `parent` = the routine_run call it belongs to (the
    convention task tools use, author-tools 4b). Without these lines the image shadow (twin/practice_view.py
    episode_log) could not replay the routine's commands, so env_observe_image refused for the rest of the episode,
    and integrity_audit could not see them. Returns (send, stats)."""
    from harness import registry
    from harness.tools.env import _call
    stats = dict(n_logged=0)

    def send(req: dict) -> dict:
        t0 = time.time()
        r = _call(brain, run_id, session, req)
        name = INVOKE_LOGGED.get(req.get("op"))
        if name:
            args = {"session": session, **{k: v for k, v in req.items() if k != "op"}, "brain": brain,
                    "run_id": run_id}
            ok = "error" not in r
            line = {"t": t0, "dt": round(time.time() - t0, 3), "caller": caller, "tool": name, "args": args,
                    "ok": ok, "result": r if ok else {"error": r["error"]}, "parent": parent}
            registry.CALL_LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(registry.CALL_LOG_DIR / f"{time.strftime('%Y%m%d')}.jsonl", "a") as fh:
                fh.write(json.dumps(line, default=str) + "\n")
            stats["n_logged"] += 1
        return r
    return send, stats


def _prepare(version_dir: Path, invoke: dict) -> tuple[Path, str]:
    from twin import sandbox as SB
    tmp, entry = SB.prepare(version_dir)
    for junk in ("meta.json",):
        (tmp / "ws" / junk).unlink(missing_ok=True)
    gate = tmp / "ws" / "gate"
    if gate.exists():
        import shutil
        shutil.rmtree(gate)
    (tmp / "ws" / "invoke.json").write_text(json.dumps(invoke))
    return tmp, entry


# ================================================================================================ extract
@tool("library_extract",
      "Stage E (compile). Read the decision structure of a practice run's finished episodes from their command logs "
      "and trajectories only: every command becomes a phase (hover, touch, lift, shift, probe_rim, probe_in, push, "
      "release, retreat), repeated lift/shift/probe loops fold into 'search', and the brain's parameters (speeds, force "
      "ceilings, stop forces, shifts) are collected per phase. ready=true when at least min_successes counted successes "
      "share one structure: the pattern to compile. Also lists how many images each episode used (decisions that needed "
      "perception stay with the brain unless contact can replace them).",
      {"type": "object", "properties": {
          **_RUN,
          "source_run": {"type": "string", "description": "runs/<brain>/<run_id> of the practice run to read (same brain)"},
          "start_ids": {"type": "array", "items": {"type": "string"}, "description": "episodes to read (default: all "
                        "brain episodes of the source run)"},
          "min_successes": {"type": "integer", "description": "default 3"}}})
def library_extract(source_run: str, brain: str = "claude", run_id: str = "run-compile-pilot-1",
                    start_ids: list | None = None, min_successes: int = 3) -> dict:
    from twin.library import extract, same_brain
    src = _repo(source_run)
    same_brain(src, brain, "source_run")
    recs = []
    for line in (src / "env" / "episodes.jsonl").read_text().splitlines():
        r = json.loads(line)
        a = str(r.get("actor", ""))
        if r.get("privileged") or a.startswith(("selftest", "test-", "obs-", "skill:", "routine:", "library-")):
            continue
        if start_ids and r["start_id"] not in start_ids:
            continue
        recs.append(json.loads((REPO / r["file"]).read_text()))
    if not recs:
        raise ValueError(f"no brain episodes in {source_run} for {start_ids or 'any start'}")
    out = extract(recs, min_successes=min_successes)
    led = src / "practice" / "ledger.json"
    if led.exists():
        rows = {e["start_id"]: e for e in json.loads(led.read_text()).get("episodes", [])}
        for e in out["episodes"]:
            r = rows.get(e["start_id"]) or {}
            e["images"] = r.get("images")
            e["decisions"] = r.get("decisions")
    out["source_run"] = _rel(src)
    d = _run_dir(brain, run_id) / "library" / "extracts"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{src.name}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps(out, indent=1, default=str))
    out.pop("phases")
    out["file"] = _rel(f)
    return out


# ================================================================================================ add
@tool("library_add",
      "Stage E (compile). Add a routine version to this run's library: a workspace dir with skill.py (envapi only, "
      "passes skill_check) and params.json ({defaults, per_part, tunable}). Runs the static check, copies the files into "
      "runs/<brain>/<run_id>/library/<name>/v<N>/ frozen by hash, records where it came from (source episodes and their "
      "decision structure), and runs a sandbox smoke test on one practice start (the program boots, reads its params and "
      "speaks the env protocol; no motion, nothing recorded). The new version is a candidate until promotion_gate.",
      {"type": "object", "required": ["name", "source", "parts", "when"], "properties": {
          **_RUN,
          "name": {"type": "string", "description": "routine name, lowercase with _ (e.g. peg_insert_guarded)"},
          "source": {"type": "string", "description": "repo path of the workspace dir (skill.py + params.json), under "
                     "runs/<brain>/"},
          "parts": {"type": "array", "items": {"type": "string"}, "description": "parts it was compiled for"},
          "when": {"type": "string", "description": "when to apply it, in one or two sentences"},
          "source_run": {"type": "string", "description": "runs/<brain>/<run_id> holding the source episodes"},
          "source_starts": {"type": "array", "items": {"type": "string"},
                            "description": "start ids of the successes it was compiled from"},
          "dev_starts": {"type": "array", "items": {"type": "string"},
                         "description": "practice starts the compiler ran while writing it (excluded from the pilot)"},
          "notes": {"type": "string"},
          "smoke": {"type": "boolean", "description": "run the sandbox smoke test (default true)"}}})
def library_add(name: str, source: str, parts: list, when: str, brain: str = "claude",
                run_id: str = "run-compile-pilot-1", source_run: str | None = None, source_starts: list | None = None,
                dev_starts: list | None = None, notes: str = "", smoke: bool = True) -> dict:
    from twin import library as L
    from twin.evaluator import code_hash
    from twin.sandbox import check_dir
    rd = _run_dir(brain, run_id)
    src = _repo(source)
    L.same_brain(src, brain, "source")
    chk = check_dir(src)
    if not chk["ok"]:
        return dict(ok=False, static_check=chk, note="rejected by the static checker; nothing was added")
    structure = None
    if source_run:
        sr = _repo(source_run)
        L.same_brain(sr, brain, "source_run")
        recs = []
        for line in (sr / "env" / "episodes.jsonl").read_text().splitlines():
            r = json.loads(line)
            if source_starts and r["start_id"] not in source_starts:
                continue
            if r.get("privileged") or str(r.get("actor", "")).startswith(("skill:", "routine:", "selftest", "test-")):
                continue
            rec = json.loads((REPO / r["file"]).read_text())
            rec["_file"] = r["file"]
            recs.append(rec)
        ex = L.extract(recs) if recs else None
        structure = None if ex is None else dict(
            common_structure=ex["common_structure"], n_common=ex["n_common"], ready=ex["ready"], why=ex["why"],
            episodes=[{k: e[k] for k in ("start_id", "part", "outcome", "counted", "structure", "probes")}
                      for e in ex["episodes"]],
            files=[r["_file"] for r in recs], parameter_ranges=ex["parameter_ranges"])
    meta = dict(compiled_by=current_caller() or "unknown", brain=brain, run_id=run_id, parts=list(parts), when=when,
                notes=notes, source=dict(dir=_rel(src), run=source_run, starts=source_starts or [],
                                         structure=structure),
                dev_starts=dev_starts or [], static_check=dict(ok=True, sha256=chk["sha256"]),
                interface=dict(ops=list(INVOKE_OPS) + ["finish (standalone only)"], envapi="twin/sandbox.py ENVAPI_SOURCE"),
                evaluator_code_hash=code_hash())
    v = L.add_version(rd, name, src, meta)
    out = dict(ok=True, **v, static_check=dict(ok=True, sha256=chk["sha256"]), structure=structure and {
        k: structure[k] for k in ("common_structure", "n_common", "ready", "why")})
    if smoke:
        sm = _smoke(rd, brain, run_id, name, v["version"], parts[0])
        out["smoke"] = sm
        vd = L.entry(rd, name, v["version"])["dir"]
        meta2 = json.loads((vd / "meta.json").read_text())
        meta2["smoke"] = sm
        (vd / "meta.json").write_text(json.dumps(meta2, indent=1, default=str))
        if not sm.get("ok"):
            L.set_status(rd, name, v["version"], "rejected", f"sandbox smoke test failed: {sm.get('why')}",
                         by=current_caller())
            out["status"] = "rejected"
        else:
            out["status"] = "candidate"
    return out


def _smoke(rd: Path, brain: str, run_id: str, name: str, version: str, part: str) -> dict:
    from twin import library as L
    from twin import starts as S
    from twin.env import PracticeEnv
    e = L.entry(rd, name, version)
    st, _ = S.load("practice", rd / "starts")
    sid = sorted(k for k, s in st.items() if s["part"] == part)[0]
    env = PracticeEnv(brain, run_id, "practice", "c000", actor=f"library-smoke:{name}@{version}", record=False,
                      session=f"smoke-{name}-{version}")
    env.reset(sid)
    tmp, entry = _prepare(e["dir"], dict(mode="invoke", smoke=True))
    errp = L.lib_dir(rd) / name / version / "smoke.stderr"
    try:
        pm = _pump(tmp, entry, lambda req: env.command(req, allowed=INVOKE_OPS), 120.0, errp)
    finally:
        from twin import sandbox as SB
        SB.cleanup(tmp)
        if env.ep is not None and not env.ep["done"]:
            env.ep["done"] = True     # nothing recorded (record=False); the smoke never scores
    res = pm["result"] or {}
    ok = res.get("code") == "SMOKE" and pm["refused_ops"] == 0
    return dict(ok=ok, start_id=sid, code=res.get("code"), why=res.get("why"), sandbox=pm["sandbox"],
                requests=pm["requests"], refused_ops=pm["refused_ops"], commands=res.get("commands"),
                stderr=_rel(errp))


# ================================================================================================ list
@tool("library_list",
      "Stage E (compile). The routines in this run's library: versions, status (candidate, piloted, promoted, demoted, "
      "stale, rejected), parts, when to apply, tunable parameters, the latest gate results and whether the frozen files "
      "are unchanged. usage=true adds, per brain episode that invoked a routine, the sim time under routines, the codes "
      "they returned and the brain share of control (1 - routine sim s / episode sim s).",
      {"type": "object", "properties": {**_RUN, "routine": {"type": "string"},
                                        "usage": {"type": "boolean", "description": "default false"}}})
def library_list(brain: str = "claude", run_id: str = "run-compile-pilot-1", routine: str | None = None,
                 usage: bool = False) -> dict:
    from twin import library as L
    rd = _run_dir(brain, run_id)
    idx = L.load_index(rd)
    out = dict(library=_rel(L.lib_dir(rd)), routines={})
    for name, r in sorted(idx["routines"].items()):
        if routine and name != routine:
            continue
        vs = {}
        for v in sorted(r["versions"], key=lambda x: int(x[1:])):
            e = L.entry(rd, name, v)
            m = e["meta"]
            P = json.loads((e["dir"] / "params.json").read_text())
            gates = sorted((e["dir"] / "gate").glob("*.json")) if (e["dir"] / "gate").exists() else []
            last = {}
            for g in gates:
                gj = json.loads(g.read_text())
                last[gj["mode"]] = dict(file=_rel(g), at=gj.get("at"), verdict=gj.get("verdict"),
                                        summary=gj.get("summary"))
            vs[v] = dict(status=e["status"], frozen=e["frozen"], parts=m.get("parts"), when=m.get("when"),
                         tunable=P.get("tunable"), created=m.get("created"), smoke=(m.get("smoke") or {}).get("ok"),
                         source=dict(run=m["source"].get("run"), starts=m["source"].get("starts"),
                                     structure=(m["source"].get("structure") or {}).get("common_structure")),
                         gates=last, history=idx["routines"][name]["versions"][v].get("history", []), dir=_rel(e["dir"]))
        out["routines"][name] = dict(active=L.active_version(r), versions=vs)
    if usage:
        out["usage"] = L.usage(rd)
    return out


# ================================================================================================ invoke
@tool("routine_run",
      "Stage E (compile), for a brain inside a live episode: hand control to a library routine. It runs sandboxed with "
      "ZERO brain calls through the same actor ops you have (no finish), and hands control back with a code: OK (seated "
      "by its own estimate, released unless release=false), NOT_FOUND (hole outside its search area), JAMMED, "
      "ENVELOPE_TIME, NO_CONTACT, UNEXPECTED_CONTACT, NOT_HOLDING, BUDGET, EPISODE_OVER, BAD_ARGS. Returns the code, why, "
      "the phases and probes it ran (hole frame, mm), sim s and commands used. Pass center_mm (your estimate of the true "
      "hole centre in the believed hole frame, mm) and tunable params to steer it. Every command it sends is logged "
      "as its own env_* call (parent: this call), so env_observe_image still renders the episode afterwards. Logged "
      "for D.brain_share and calls per success.",
      {"type": "object", "required": ["session", "routine"], "properties": {
          **_RUN,
          "session": {"type": "string", "description": "session id from env_open (an episode must be running)"},
          "routine": {"type": "string", "description": "routine name (library_list)"},
          "version": {"type": "string", "description": "default: the active version (promoted, else piloted, else "
                      "candidate)"},
          "center_mm": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                        "description": "your estimate of the true hole centre, believed hole frame [x, y] mm"},
          "params": {"type": "object", "description": "tunable overrides only (library_list shows them)"}}})
def routine_run(session: str, routine: str, brain: str = "claude", run_id: str = "run-compile-pilot-1",
                version: str | None = None, center_mm: list | None = None, params: dict | None = None) -> dict:
    from harness.tools.env import _call
    from twin import library as L
    from twin import sandbox as SB
    rd = _run_dir(brain, run_id)
    e = L.entry(rd, routine, version)
    if not e["frozen"]["ok"]:
        L.set_status(rd, routine, e["version"], "stale", f"frozen files changed: {e['frozen']['changed']}",
                     by="routine_run")
        raise PermissionError(f"{routine}@{e['version']} files changed since it was added: marked stale; add a new "
                              "version")
    if e["status"] not in L.INVOCABLE:
        raise PermissionError(f"{routine}@{e['version']} is {e['status']}: not invocable")
    st = _call(brain, run_id, session, {"op": "status"})
    if "error" in st or st.get("episode_over") or not st.get("start_id"):
        raise RuntimeError(f"no running episode in session {session}: {st.get('error') or st.get('ended_by')}")
    o0 = _call(brain, run_id, session, {"op": "observe"})
    mem = L.session_memory(rd, session, st["start_id"])
    inv = dict(mode="invoke", params=params or {}, memory=mem, center_mm=center_mm)
    tmp, entry = _prepare(e["dir"], inv)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{os.urandom(2).hex()}"
    errp = L.lib_dir(rd) / "invocations" / f"{session}_{stamp}.stderr"
    caller = current_caller() or "unknown"
    parent = dict(tool="routine_run", call_id=f"routine_run-{time.time_ns()}", caller=caller)
    send, relay = _logged_relay(brain, run_id, session, caller, parent)
    try:
        pm = _pump(tmp, entry, send, 900.0, errp)
    finally:
        SB.cleanup(tmp)
    res = pm["result"] or dict(code="ENV_ERROR", why=f"the routine returned no result ({pm['program_end']})")
    st1 = _call(brain, run_id, session, {"op": "status"})
    t1 = st1.get("t") if st1.get("t") is not None else o0.get("t")
    if isinstance(res.get("memory"), dict):
        L.session_memory(rd, session, st["start_id"], res["memory"])
    parts = e["meta"].get("parts") or []
    row = dict(session=session, start_id=st["start_id"], part=st.get("part"), routine=routine, version=e["version"],
               status=e["status"], code=res.get("code"), sim_t0=o0.get("t"), sim_t1=t1,
               commands=res.get("commands"), requests=pm["requests"], refused_ops=pm["refused_ops"],
               outside_parts=st.get("part") not in parts, caller=current_caller(), stderr=_rel(errp),
               call_id=parent["call_id"], env_calls_logged=relay["n_logged"])
    L.log_invocation(rd, row)
    probes = [{k: p.get(k) for k in ("aim", "depth_mm", "stopped", "touch")} for p in res.get("probes") or []]
    return dict(routine=routine, version=e["version"], status=e["status"], code=res.get("code"),
                handback=res.get("code") != "OK", why=res.get("why"), depth_mm=res.get("depth_mm"),
                released=res.get("released"), probes=probes,
                phases=[{k: v for k, v in p.items() if k != "cmds"} for p in res.get("phases") or []],
                sim_s=None if t1 is None or o0.get("t") is None else round(float(t1) - float(o0["t"]), 2),
                commands=res.get("commands"), brain_calls=0, outside_parts=row["outside_parts"],
                episode_over=bool(st1.get("episode_over")), next=NEXT_HINT.get(res.get("code"), "decide"),
                call_log=dict(call_id=parent["call_id"], env_calls_logged=relay["n_logged"]))


# ================================================================================================ gate
def _pick_cousins(rd: Path, n: int) -> list[str]:
    """A small stratified pick over the frozen cousins: (dt, solver) combos in turn, alternating the lowest and the
    highest stiffness within each combo, so a pilot sees every solver/timestep and both ends of K."""
    from twin import cousins as C
    cs = [c for c in C.load(rd / "cousins")["by_id"].values() if not c.get("nominal")]
    combos: dict = {}
    for c in cs:
        combos.setdefault((c["timestep"], c["solver"]), []).append(c)
    order, k = [], 0
    while len(order) < n and any(combos.values()):
        for key in sorted(combos):
            lst = sorted(combos[key], key=lambda c: c["k_trans"])
            if not lst:
                continue
            c = lst[0] if k % 2 == 0 else lst[-1]
            combos[key] = [x for x in combos[key] if x["id"] != c["id"]]
            order.append(c["id"])
            if len(order) >= n:
                break
        k += 1
    return order


def _env3(feats: list[dict], part: str, env: dict) -> dict:
    """Routine-level envelope verdict on the three features validation.yaml's E.envelope_violations names (force
    peaks, approach speeds, durations), next to the full six-feature verdict of twin/envelope.py check_routine."""
    from twin.envelope import check_episode, family_of
    keep = {"peak_force_N", "approach_speed_mm_s", "insert_duration_s", "onset"}
    n = len(feats)
    counts: dict = {}
    for f in feats:
        for v in {(v["feature"], v["side"]) for v in check_episode(f, part, env) if v["feature"] in keep}:
            counts[f"{v[0]}:{v[1]}"] = counts.get(f"{v[0]}:{v[1]}", 0) + 1
    reasons = [f"{k} in {c}/{n} episodes (> 10%)" for k, c in sorted(counts.items()) if n and c / n > 0.10]
    fmax = env["envelope"].get(family_of(part), {}).get("peak_force_any_max_N")
    hard = [f["peak_force_N"] for f in feats if f.get("peak_force_N") is not None and fmax and f["peak_force_N"] > fmax]
    if hard:
        reasons.append(f"{len(hard)} episode(s) with peak force above the largest real peak {fmax} N")
    return dict(violates=bool(reasons), reasons=reasons, counts=counts, n_episodes=n)


def _run_jobs(rd: Path, brain: str, run_id: str, prog: Path, jobs: list, set_name: str, actor: str,
              out_dir: Path, workers: int, wall_s: float = 900.0) -> list[dict]:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from harness.tools.env import frozen_budget
    from harness.tools.integrity import _job
    from twin import starts as S
    from twin.library import routine_result_from_stderr
    budget = frozen_budget(brain, run_id, None)
    stamp = out_dir.name
    args = [dict(brain=brain, run_id=run_id, start_set=set_name, cousin=c, actor=actor, start_id=s, budget=budget,
                 program=str(prog), wall_s=wall_s, stderr_dir=str(out_dir), session=f"{prog.parent.name}-{prog.name}-"
                 f"{stamp}-{i:03d}") for i, (s, c) in enumerate(jobs)]
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(args) or 1)), mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(_job, args))
    st, _ = S.load(set_name, rd / "starts")
    for r, a in zip(rows, args):
        r["cousin"] = a["cousin"]
        r["part"] = st.get(a["start_id"], {}).get("part")
        rr = routine_result_from_stderr(out_dir / f"{a['start_id']}_{a['cousin']}.stderr") or {}
        r["routine"] = {k: rr.get(k) for k in ("code", "why", "commands", "sim_s", "contact_s")} | dict(
            probes=len(rr.get("probes") or []))
    return rows


def _handed_back(r: dict) -> bool:
    """The routine handed this episode back to the brain (any code but OK). In a standalone gate episode it then ends
    the episode itself, claiming failure. A row without a code (the routine crashed) is not a hand-back."""
    code = (r.get("routine") or {}).get("code")
    return code is not None and code != "OK"


def _routine_features(r: dict) -> dict | None:
    """Envelope features of the routine's own segment of one episode (validation.yaml v17 E.envelope_violations): the
    whole episode when the routine kept it (code OK), its segment up to the hand-back when it handed back. Only the
    brain's actions after a hand-back are excluded.

    A gate episode is standalone (twin/env.py run_program: only the routine acts, and after a hand-back it ends the
    episode itself), so its whole record is the routine's segment. A record in which a brain acted after the hand-back
    (brain_after_handback) must carry envelope.routine_features (twin/envelope.py features_until, cut at the hand-back);
    without them it has no routine segment to check and is reported as unsegmented, never silently passed."""
    env = r.get("envelope") or {}
    if env.get("routine_features"):
        return env["routine_features"]
    if _handed_back(r) and r.get("brain_after_handback"):
        return None
    return env.get("features") or None


def _summ(rows: list[dict], part: str, env: dict | None) -> dict:
    """Per-part gate summary. Envelope (validation.yaml v17 E.envelope_violations): twin/envelope.py check_routine on
    the routine's own segment of every episode, kept to the end or up to its hand-back (_routine_features); only the
    brain's later actions are excluded. The handed-back episodes' segments are also shown alone (envelope_handbacks)."""
    from harness.tools.integrity import _summarize
    from twin.envelope import check_routine
    s = _summarize(rows, part, env)
    if env:
        keep = ("violates", "reasons", "counts", "n_episodes")
        back = [r for r in rows if _handed_back(r)]

        def verdict(rs):
            feats = [f for f in (_routine_features(r) for r in rs) if f]
            if not feats:
                return dict(violates=False, reasons=[], counts={}, n_episodes=0)
            v = check_routine(feats, part, env)
            return {k2: v[k2] for k2 in keep}
        unseg = [r.get("start_id") for r in back if _routine_features(r) is None]
        s["envelope"] = dict(verdict(rows), rule="v17: the routine's segment of every episode (kept to the end, or up "
                             "to its hand-back); the brain's later actions excluded", n_handed_back=len(back),
                             unsegmented=unseg)
        s["envelope_handbacks"] = dict(verdict(back), note="the hand-back episodes' routine segments alone; already "
                                       "counted in envelope")
    k = sum(1 for r in rows if r.get("counted"))
    codes: dict = {}
    for r in rows:
        c = (r.get("routine") or {}).get("code")
        codes[c] = codes.get(c, 0) + 1
    s["routine_codes"] = codes
    s["brain_calls"] = 0
    s["per_success"] = dict(sim_s=round(sum(r.get("sim_s") or 0 for r in rows) / k, 2) if k else None,
                            wall_s=round(sum(r.get("wall_s") or 0 for r in rows) / k, 1) if k else None,
                            env_commands=round(sum(r.get("n_commands") or 0 for r in rows) / k, 1) if k else None,
                            brain_calls=0.0 if k else None, tokens=0 if k else None)
    if env:
        feats = [f for f in (_routine_features(r) for r in rows) if f]
        s["envelope_validation_yaml_3"] = _env3(feats, part, env) if feats else None   # the v15 three-feature view
    inval = [f for r in rows for f in r.get("flags", []) if f.get("invalidates")]
    s["g"] = dict(counted_with_invalidating_flag=sum(1 for r in rows if r.get("counted") and any(
                      f.get("invalidates") for f in r.get("flags", []))),
                  integrity_flags=sum(1 for f in inval if str(f.get("monitor", "")).startswith("integrity")),
                  invalidating_flags=len(inval), success_flagged=s.get("success_flagged", 0),
                  false_claims=s.get("false_claims", 0))
    return s


@tool("promotion_gate",
      "Stage E (compile). The deterministic promotion gate for one routine version; the watcher reruns it. "
      "mode=pilot (anyone, never promotes): practice starts not used while compiling, on the nominal twin, plus "
      "practice-start x cousin pairs (every solver/timestep, both stiffness ends); a pass marks it piloted. "
      "mode=scored (callers watcher*/evaluator* only): the frozen held-out starts (heldout_n per part) and the frozen "
      "(held-out start, cousin) pairs, each attempted once; promotes on E.routine_success, E.cousins_success (one-sided "
      "95% Clopper-Pearson lower bounds vs validation.yaml), E.envelope_violations (six features on the "
      "routine's own segment of every episode, nominal and cousins: kept to the end or up to its hand-back, the "
      "brain's later actions excluded; incl. the family's B2), G monitors "
      "clean and unchanged frozen files, else rejects. mode=review: demotes or marks stale a piloted/promoted version "
      "whose files, evaluator or later record changed. Never records checks: it returns suggested values for a watcher.",
      {"type": "object", "required": ["routine"], "properties": {
          **_RUN,
          "routine": {"type": "string"},
          "version": {"type": "string", "description": "default: the active version"},
          "mode": {"type": "string", "enum": ["pilot", "scored", "review"], "description": "default pilot"},
          "parts": {"type": "array", "items": {"type": "string"}, "description": "default: the routine's parts"},
          "pilot_starts": {"type": "integer", "description": "pilot: practice starts per part on c000 (default 12)"},
          "pilot_cousins": {"type": "integer", "description": "pilot: practice-start x cousin pairs per part (default 8)"},
          "heldout_n": {"type": "integer", "enum": [29, 46], "description": "scored: held-out starts per part, "
                        "pre-declared (default 29)"},
          "workers": {"type": "integer", "description": "parallel episodes (default 6)"},
          "plan_only": {"type": "boolean", "description": "list the jobs without running anything (default false)"}}})
def promotion_gate(routine: str, brain: str = "claude", run_id: str = "run-compile-pilot-1",
                   version: str | None = None, mode: str = "pilot", parts: list | None = None,
                   pilot_starts: int = 12, pilot_cousins: int = 8, heldout_n: int = 29, workers: int = 6,
                   plan_only: bool = False) -> dict:
    from twin import cousins as C
    from twin import library as L
    from twin import starts as S
    from twin.envelope import load as load_env
    from twin.evaluator import code_hash
    from twin.sandbox import check_dir
    from harness.tools.integrity import _b2_status
    rd = _run_dir(brain, run_id)
    e = L.entry(rd, routine, version)
    v, meta = e["version"], e["meta"]
    who = current_caller() or ""
    if mode == "review":
        return _review(rd, routine, e, who)
    if mode not in ("pilot", "scored"):
        raise ValueError("mode: pilot | scored | review")
    if mode == "scored" and not who.startswith(WATCHERS):
        raise PermissionError(f"the scored gate spends frozen held-out starts once: callers {WATCHERS} only "
                              f"(caller {who!r}). Actors run mode=pilot.")
    if mode == "scored" and e["status"] != "piloted" and not plan_only:
        raise PermissionError(f"{routine}@{v} is {e['status']}: the scored gate takes piloted versions only")
    parts = parts or meta.get("parts") or []
    chk = check_dir(e["dir"] / "skill.py")
    frozen = dict(files=e["frozen"], static_check=chk["ok"], evaluator_code_hash=code_hash(),
                  evaluator_unchanged_since_add=code_hash() == meta.get("evaluator_code_hash"),
                  starts_sha=(rd / "starts" / "manifest.sha256").read_text().strip()[:16],
                  cousins_sha=(rd / "cousins" / "manifest.sha256").read_text().strip()[:16],
                  envelope_sha=(rd / "envelope" / "envelope.sha256").read_text().strip()[:16])
    targets = L.e_targets()
    sha8 = meta["files"]["skill.py"][:8]
    actor = f"routine:{routine}@{v}:{sha8}"
    exclude = set(meta.get("dev_starts") or []) | set((meta.get("source") or {}).get("starts") or [])
    jobs_c0, jobs_cz = [], []
    if mode == "pilot":
        st, _ = S.load("practice", rd / "starts")
        cz = _pick_cousins(rd, pilot_cousins)
        for p in parts:
            ids = [s for s in sorted(k for k, x in st.items() if x["part"] == p) if s not in exclude]
            jobs_c0 += [(s, "c000") for s in ids[:pilot_starts]]
            jobs_cz += [(s, c) for s, c in zip(ids[:pilot_cousins], cz)]
        set_name = "practice"
    else:
        st, _ = S.load("heldout", rd / "starts")
        pairs = C.load(rd / "cousins")["eval_pairs"]
        for p in parts:
            ids = sorted(k for k, x in st.items() if x["part"] == p)
            jobs_c0 += [(s, "c000") for s in ids[:heldout_n]]
            jobs_cz += [tuple(x) for x in pairs.get(p, [])]
        set_name = "heldout"
    plan = dict(routine=routine, version=v, mode=mode, actor=actor, set=set_name, parts=parts,
                n_nominal=len(jobs_c0), n_cousins=len(jobs_cz), excluded_starts=sorted(exclude),
                nominal=jobs_c0, cousins=jobs_cz, frozen=frozen)
    if plan_only:
        return dict(plan=plan, note="plan only: nothing was run")
    if not (frozen["files"]["ok"] and chk["ok"]):
        raise PermissionError(f"{routine}@{v} is not frozen (changed {frozen['files']['changed']}, static check "
                              f"{chk['ok']}): add a new version")
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{os.urandom(2).hex()}"
    out_dir = e["dir"] / "gate" / f"{mode}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = _run_jobs(rd, brain, run_id, e["dir"], jobs_c0 + jobs_cz, set_name, actor, out_dir, workers)
    try:
        env = load_env(rd / "envelope" / "envelope.json")
    except FileNotFoundError:
        env = None
    nom = [r for r in rows if r["cousin"] == "c000" and (r["start_id"], "c000") in set(jobs_c0)]
    cz_rows = [r for r in rows if (r["start_id"], r["cousin"]) in set(jobs_cz)]
    per = {p: dict(nominal=_summ([r for r in nom if r["part"] == p], p, env),
                   cousins=_summ([r for r in cz_rows if r["part"] == p], p, env)) for p in parts}
    tgt_rs = targets["E.routine_success"]["target"]
    tgt_cz = targets["E.cousins_success"]["target"]
    b2 = _b2_status(brain)
    crit = {}
    for p, s in per.items():
        n_cp, c_cp = s["nominal"]["counted_success"], s["cousins"]["counted_success"]
        env_full = (s["nominal"].get("envelope") or {})
        env_cz = (s["cousins"].get("envelope") or {})
        crit[p] = dict(
            routine_success=dict(rate=n_cp.get("rate"), lower_one_sided_95=n_cp.get("lower_one_sided_95"),
                                 n=n_cp.get("n"),
                                 passed=(n_cp.get("rate") or 0) >= tgt_rs if mode == "pilot"
                                 else (n_cp.get("lower_one_sided_95") or 0) >= tgt_rs),
            cousins_success=dict(rate=c_cp.get("rate"), lower_one_sided_95=c_cp.get("lower_one_sided_95"),
                                 n=c_cp.get("n"),
                                 passed=(c_cp.get("rate") or 0) >= tgt_cz if mode == "pilot"
                                 else (c_cp.get("lower_one_sided_95") or 0) >= tgt_cz),
            envelope_full=dict(violates=env_full.get("violates"), reasons=env_full.get("reasons"),
                               n_episodes=env_full.get("n_episodes"), n_handed_back=env_full.get("n_handed_back")),
            envelope_cousins=dict(violates=env_cz.get("violates"), reasons=env_cz.get("reasons"),
                                  n_episodes=env_cz.get("n_episodes"), n_handed_back=env_cz.get("n_handed_back")),
            envelope_validation_yaml_3=s["nominal"].get("envelope_validation_yaml_3"),
            b2_family=dict(state=b2.get("state"), passed=b2.get("state") == "pass", note=b2.get("note")),
            g_clean=all(x["g"]["counted_with_invalidating_flag"] == 0 and x["g"]["integrity_flags"] == 0
                        for x in s.values()))
    ok_rs = all(c["routine_success"]["passed"] for c in crit.values())
    ok_cz = all(c["cousins_success"]["passed"] for c in crit.values())
    ok_env = all(c["envelope_full"]["violates"] is False and c["envelope_cousins"]["violates"] is not True
                 for c in crit.values())
    ok_b2 = all(c["b2_family"]["passed"] for c in crit.values())
    ok_g = all(c["g_clean"] for c in crit.values())
    ok_frozen = frozen["files"]["ok"] and chk["ok"]
    if mode == "pilot":
        passed = ok_rs and ok_cz and ok_env and ok_g and ok_frozen
        verdict = "piloted" if passed else "not ready"
    else:
        passed = ok_rs and ok_cz and ok_env and ok_b2 and ok_g and ok_frozen
        verdict = "promoted" if passed else "rejected"
    min_rs = min((c["routine_success"]["rate"] or 0) for c in crit.values()) if crit else None
    min_cz = min((c["cousins_success"]["rate"] or 0) for c in crit.values()) if crit else None
    n_succ = sum(1 for r in rows if r.get("counted"))
    summary = dict(
        nominal=f"{sum(1 for r in nom if r.get('counted'))}/{len(nom)} counted",
        cousins=f"{sum(1 for r in cz_rows if r.get('counted'))}/{len(cz_rows)} counted",
        min_routine_success=min_rs, min_cousins_success=min_cz, envelope_ok=ok_env, b2_ok=ok_b2, g_clean=ok_g,
        frozen=ok_frozen, brain_calls=0,
        per_success=dict(sim_s=round(sum(r.get("sim_s") or 0 for r in rows) / n_succ, 2) if n_succ else None,
                         wall_s=round(sum(r.get("wall_s") or 0 for r in rows) / n_succ, 1) if n_succ else None,
                         env_commands=round(sum(r.get("n_commands") or 0 for r in rows) / n_succ, 1) if n_succ else None,
                         brain_calls=0, tokens=0))
    res = dict(routine=routine, version=v, mode=mode, at=time.strftime("%Y-%m-%d %H:%M:%S"), caller=who,
               validation_version=targets["version"], targets={k: targets[k] for k in targets if k.startswith("E.")},
               plan={k: plan[k] for k in ("actor", "set", "parts", "n_nominal", "n_cousins", "excluded_starts")},
               cousins_used=sorted({c for _, c in jobs_cz}), frozen=frozen, per_part=per, criteria=crit,
               verdict=verdict, summary=summary, wall_s=round(time.time() - t0, 1),
               rows=[{k: r.get(k) for k in ("start_id", "cousin", "part", "outcome", "counted", "max_stage", "claim",
                                            "false_claim", "n_commands", "sim_s", "wall_s", "routine", "file")}
                     | dict(flags=[f["monitor"] for f in r.get("flags", [])],
                            envelope_violations=[x["feature"] + ":" + x["side"] for x in
                                                 ((r.get("envelope") or {}).get("violations") or [])])
                     for r in rows])
    res["suggested_checks"] = (
        dict(note="pilot values on practice starts: not E checks (those are defined on held-out starts). Log them with "
                  "record_event(kind='note', stage='E').")
        if mode == "pilot" else dict(
            **{"E.routine_success": dict(value=min_rs, evidence=[_rel(out_dir / "gate.json")],
                                         note="min over parts; one-sided 95% CP lower bounds in criteria")},
            **{"E.cousins_success": dict(value=min_cz, evidence=[_rel(out_dir / "gate.json")])},
            **{"E.envelope_violations": dict(value=0 if ok_env and ok_b2 else 1, evidence=[_rel(out_dir / "gate.json")])},
            **{"G.exploit_flags": dict(value=sum(x["g"]["counted_with_invalidating_flag"] for s in per.values()
                                                 for x in s.values()), evidence=[_rel(out_dir / "gate.json")])}))
    (out_dir / "gate.json").write_text(json.dumps(res, indent=1, default=float))
    g = e["dir"] / "gate" / f"{mode}_{stamp}.json"
    g.write_text(json.dumps({k: res[k] for k in res if k != "rows"} | dict(detail=_rel(out_dir / "gate.json")),
                            indent=1, default=float))
    if mode == "pilot" and passed and e["status"] == "candidate":
        L.set_status(rd, routine, v, "piloted", "pilot gate passed on practice starts and cousins",
                     evidence=[_rel(g)], by=who, extra=dict(gate_evaluator_hash=code_hash()))
    elif mode == "scored":
        L.set_status(rd, routine, v, "promoted" if passed else "rejected",
                     "scored gate passed" if passed else f"scored gate failed: rs {ok_rs} cz {ok_cz} env {ok_env} "
                     f"b2 {ok_b2} g {ok_g} frozen {ok_frozen}", evidence=[_rel(g)], by=who,
                     extra=dict(gate_evaluator_hash=code_hash()))
    else:
        L._event(rd, "gate", routine=routine, version=v, mode=mode, verdict=verdict, file=_rel(g), by=who)
    res.pop("rows")
    res["file"] = _rel(g)
    res["detail"] = _rel(out_dir / "gate.json")
    return res


def _review(rd: Path, routine: str, e: dict, who: str) -> dict:
    """Demotion and staleness. A piloted or promoted version is
    - stale when its files changed, or the evaluator code changed since its gate (results are not comparable);
    - demoted when its later record says it no longer works: in brain episodes, >= 5 invocations whose OK-rate has a
      two-sided 95% Clopper-Pearson upper bound below the E.routine_success target, or any later gate whose nominal
      or cousin rate has an upper bound below the target, or any counted success of it with an invalidating flag."""
    from twin import library as L
    from twin.evaluator import code_hash
    idx = L.load_index(rd)
    actions = []
    tgt = L.e_targets()["E.routine_success"]["target"]
    for v, info in sorted(idx["routines"][routine]["versions"].items(), key=lambda kv: int(kv[0][1:])):
        if info["status"] not in ("piloted", "promoted"):
            continue
        ent = L.entry(rd, routine, v)
        why = []
        if not ent["frozen"]["ok"]:
            actions.append(L.set_status(rd, routine, v, "stale", f"files changed: {ent['frozen']['changed']}", by=who))
            continue
        gh = info.get("gate_evaluator_hash")
        if gh and gh != code_hash():
            actions.append(L.set_status(rd, routine, v, "stale", "evaluator code changed since the gate "
                                        f"({gh} -> {code_hash()}): re-gate", by=who))
            continue
        f = L.lib_dir(rd) / "invocations.jsonl"
        inv = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
        mine = [r for r in inv if r.get("routine") == routine and r.get("version") == v]
        if len(mine) >= 5:
            cp = L.clopper_pearson(sum(1 for r in mine if r.get("code") == "OK"), len(mine))
            if cp["ci95"][1] < tgt:
                why.append(f"brain-episode invocations: OK {cp['k']}/{cp['n']}, upper bound {cp['ci95'][1]} < {tgt}")
        gd = ent["dir"] / "gate"
        for g in sorted(gd.glob("*.json")) if gd.exists() else []:
            gj = json.loads(g.read_text())
            for p, s in (gj.get("per_part") or {}).items():
                for kind in ("nominal", "cousins"):
                    cs = s[kind]["counted_success"]
                    if cs.get("n") and L.clopper_pearson(cs["k"], cs["n"])["ci95"][1] < tgt:
                        why.append(f"{g.name} {p} {kind}: {cs['k']}/{cs['n']} upper bound below {tgt}")
                    if s[kind]["g"]["counted_with_invalidating_flag"]:
                        why.append(f"{g.name} {p} {kind}: counted success with an invalidating flag")
        if why:
            actions.append(L.set_status(rd, routine, v, "demoted", "; ".join(why), by=who))
    return dict(routine=routine, mode="review", actions=actions,
                note="demoted and stale versions are never invoked; a fix is a new version (library_add)")
