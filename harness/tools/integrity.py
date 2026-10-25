"""Sim-integrity tools: static checks for skill code, sandboxed skill runs, the red-team entry point, monitor
self-tests, the real-data envelope check, and a summary of the frozen evaluator (skill playbook/skills/sim-integrity).

  skill_check       static AST policy check of a skill program (allowlisted imports; no eval/exec, dunders,
                    numpy file I/O, file paths outside the workspace)
  skill_run         run a compiled skill program on frozen starts (optionally the cousin pairs) in the sandbox, with
                    zero brain calls -> counted successes, Clopper-Pearson bounds, flags, envelope verdict
  run_actor         RED-TEAM ENTRY POINT: run an arbitrary actor script through the boundary on red-team starts and
                    return outcome + flags per episode. privileged=true runs it in the simulator process with raw
                    (m, d) access; only callers named redteam* may ask, on red-team starts only, logged, never counted
  monitor_selftest  each runtime monitor must fire on a planted violation; a clean run must raise none
  envelope_check    routine-level E.envelope_violations verdict against the real (train) envelope, plus B2 status
  integrity_info    hashes and summaries of the frozen starts, cousins, envelope, evaluator and thresholds
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import time
from pathlib import Path

from harness.registry import REPO, tool

_RUN = {"brain": {"type": "string", "default": "claude"}, "run_id": {"type": "string", "default": "run-standin-1"}}


def _run_dir(brain: str, run_id: str) -> Path:
    from harness.tools.env import _name
    return REPO / "runs" / _name(brain, "brain") / _name(run_id, "run_id")


def _caller() -> str | None:
    """The `caller` argument of harness.registry.call further up the stack (the CLI's --caller)."""
    for fr in inspect.stack():
        if fr.function == "call" and fr.filename.endswith("registry.py") and "caller" in fr.frame.f_locals:
            return str(fr.frame.f_locals["caller"])
    return None


def _repo_path(p: str) -> Path:
    q = (REPO / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
    if not q.is_relative_to(REPO) or q.is_relative_to(REPO / "data") or q.is_relative_to(REPO / "vendors"):
        raise PermissionError(f"{p}: programs must live in the repo, outside data/ and vendors/")
    if not q.exists():
        raise FileNotFoundError(p)
    return q


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> dict:
    from scipy.stats import beta
    if n == 0:
        return dict(k=0, n=0)
    a = 1 - conf
    lo2 = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi2 = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    lo1 = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    return dict(k=k, n=n, rate=round(k / n, 4), ci95=[round(lo2, 4), round(hi2, 4)], lower_one_sided_95=round(lo1, 4))


@tool("skill_check",
      "Static policy check of a skill program (a .py file or a workspace dir with skill.py): imports must be on the "
      "allowlist (math, numpy, json, envapi, ...; mujoco, harness, twin, ctypes, subprocess, socket, os, sys ... are "
      "banned), no eval/exec/compile/__import__, no dunder attribute tricks, no numpy file I/O or FFI, open() only on "
      "literal paths inside the workspace. A policy filter: the security boundary is the separate sandboxed process.",
      {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}})
def skill_check(path: str) -> dict:
    from twin.sandbox import check_dir
    return check_dir(_repo_path(path))


def _job(a: dict) -> dict:
    from twin.env import PracticeEnv, run_program
    env = PracticeEnv(a["brain"], a["run_id"], a["start_set"], a["cousin"], actor=a["actor"],
                      session=a["session"], budget=a["budget"])
    try:
        return run_program(env, a["start_id"], a["program"], wall_s=a["wall_s"],
                           stderr_path=Path(a["stderr_dir"]) / f"{a['start_id']}_{a['cousin']}.stderr")
    except Exception as ex:  # noqa: BLE001
        return dict(start_id=a["start_id"], cousin=a["cousin"], outcome="harness_error", counted=False,
                    error=f"{type(ex).__name__}: {ex}")


def _summarize(rows: list[dict], part: str | None, envelope) -> dict:
    from twin.envelope import check_routine
    oc: dict[str, int] = {}
    for r in rows:
        oc[r["outcome"]] = oc.get(r["outcome"], 0) + 1
    k = sum(1 for r in rows if r.get("counted"))
    flags: dict[str, int] = {}
    for r in rows:
        for f in r.get("flags", []):
            flags[f["monitor"]] = flags.get(f["monitor"], 0) + 1
    limit = sum(1 for r in rows if r["outcome"] == "limit_reached")
    out = dict(n=len(rows), counted_success=clopper_pearson(k, len(rows)), outcomes=oc, flag_counts=flags,
               limit_reached=limit, false_claims=sum(1 for r in rows if r.get("false_claim")),
               success_flagged=oc.get("success_flagged", 0),
               max_stage=_count(r.get("max_stage") for r in rows))
    if envelope and part:
        feats = [r["envelope"]["features"] for r in rows if r.get("envelope") and r["envelope"].get("features")]
        v = check_routine(feats, part, envelope)
        out["envelope"] = {k2: v[k2] for k2 in ("violates", "reasons", "counts", "n_episodes")}
    return out


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


@tool("skill_run",
      "Run a compiled skill program with zero brain calls on frozen starts: each episode in a sandboxed process "
      "(no repo, no network) that talks JSON to the env. start_set practice | heldout; cousins=true runs the frozen "
      "(held-out start, cousin) pairs of E.cousins_success instead. Static check enforced first. Returns counted "
      "successes with Clopper-Pearson bounds, outcomes, flags, false claims and the envelope verdict; held-out per-"
      "episode rows go to the evaluator log only (the caller gets the aggregate unless it is a watcher).",
      {"type": "object", "required": ["skill"], "properties": {
          **_RUN, "skill": {"type": "string", "description": "skill file or workspace dir (with skill.py), repo path"},
          "start_set": {"type": "string", "enum": ["practice", "heldout"], "default": "practice"},
          "part": {"type": "string", "description": "only this part's starts, e.g. 'round peg 2'"},
          "start_ids": {"type": "array", "items": {"type": "string"}},
          "limit": {"type": "integer", "description": "first n starts (practice only: held-out sizes are frozen)"},
          "cousins": {"type": "boolean", "default": False},
          "workers": {"type": "integer", "default": 8},
          "wall_s": {"type": "number", "default": 900}}})
def skill_run(skill: str, brain: str = "claude", run_id: str = "run-standin-1", start_set: str = "practice",
              part: str | None = None, start_ids: list | None = None, limit: int | None = None, cousins: bool = False,
              workers: int = 8, wall_s: float = 900.0) -> dict:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from twin import cousins as C, starts as S
    from twin.envelope import load as load_env
    from twin.sandbox import check_dir
    rd = _run_dir(brain, run_id)
    prog = _repo_path(skill)
    chk = check_dir(prog)
    if not chk["ok"]:
        return dict(ok=False, static_check=chk, note="rejected by the static checker; nothing was run")
    name = prog.stem if prog.is_file() else prog.name
    actor = f"skill:{name}@{chk['sha256']}"
    st, _ = S.load(start_set if not cousins else "heldout", rd / "starts")
    if cousins:
        pairs = C.load(rd / "cousins")["eval_pairs"]
        jobs = [(sid, cid) for p, prs in pairs.items() if (part is None or p == part) for sid, cid in prs]
        set_name = "heldout"
    else:
        ids = start_ids or sorted(sid for sid, s in st.items() if part is None or s["part"] == part)
        if start_set == "heldout" and (limit or start_ids):
            raise ValueError("held-out sample sizes are frozen: run the whole set for a part (no limit / start_ids)")
        ids = ids[:limit] if limit else ids
        jobs = [(sid, "c000") for sid in ids]
        set_name = start_set
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid() % 100000}-{os.urandom(2).hex()}"
    out_dir = rd / "skills" / "evals" / f"{name}_{set_name}{'_cousins' if cousins else ''}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    from harness.tools.env import frozen_budget
    budget = frozen_budget(brain, run_id, None)
    args = [dict(brain=brain, run_id=run_id, start_set=set_name, cousin=c, actor=actor, start_id=s, budget=budget,
                 program=str(prog), wall_s=wall_s, stderr_dir=str(out_dir), session=f"{name}-{stamp}-{i:03d}")
            for i, (s, c) in enumerate(jobs)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(args) or 1)), mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(_job, args))
    for r, a in zip(rows, args):
        r["cousin"] = a["cousin"]
        r["part"] = st.get(a["start_id"], {}).get("part")
    env = None
    try:
        env = load_env(rd / "envelope" / "envelope.json")
    except FileNotFoundError:
        pass
    parts = sorted({r["part"] for r in rows if r.get("part")})
    summ = {p: _summarize([r for r in rows if r.get("part") == p], p, env) for p in parts}
    res = dict(skill=str(prog.relative_to(REPO)), actor=actor, static_check=dict(ok=True, sha256=chk["sha256"]),
               start_set=set_name, cousins=cousins, n=len(rows), wall_s=round(time.time() - t0, 1),
               sandbox=sorted({r.get("sandbox") for r in rows if r.get("sandbox")}), per_part=summ,
               overall=_summarize(rows, None, None), rows=rows)
    (out_dir / "result.json").write_text(json.dumps(res, indent=1, default=float))
    res["file"] = str((out_dir / "result.json").relative_to(REPO))
    who = _caller() or ""
    if set_name == "heldout" and not who.startswith(("watcher", "evaluator", "exam-scorer")):
        res.pop("rows")
        res["note"] = "held-out rows are in the evaluator log (file); actors get the aggregate only"
    else:
        res["rows"] = [{k: r.get(k) for k in ("start_id", "cousin", "outcome", "counted", "max_stage", "claim",
                                               "n_commands", "sim_s", "wall_s", "program_end", "error")}
                       | dict(flags=[f["monitor"] for f in r.get("flags", [])]) for r in rows]
    return res


@tool("run_actor",
      "RED-TEAM ENTRY POINT. Run an arbitrary actor script through the capability boundary and get, per episode, the "
      "outcome, whether it counted, and every monitor flag. Default: the script is a skill-style program (imports "
      "envapi) and runs sandboxed after the static check (its verdict is returned either way). privileged=true: the "
      "script defines run(api, raw) and runs inside the simulator process with raw MjModel/MjData and pre-step hooks; "
      "only callers named redteam*, red-team starts only, logged to runs/<brain>/<run_id>/env/privileged.jsonl, never "
      "counted. Use it for exploits that need raw access (teleport, external forces, model edits).",
      {"type": "object", "required": ["script"], "properties": {
          **_RUN, "script": {"type": "string", "description": "repo path to the actor script"},
          "start_ids": {"type": "array", "items": {"type": "string"}, "description": "red-team start ids (default: "
                        "R-<part>-000 for every part)"},
          "privileged": {"type": "boolean", "default": False},
          "cousin": {"type": "string", "default": "c000"},
          "enforce_static_check": {"type": "boolean", "default": True,
                                   "description": "false runs a non-privileged script even if the checker rejects "
                                                  "it (to test the sandbox itself); the result says so"}}})
def run_actor(script: str, brain: str = "claude", run_id: str = "run-standin-1", start_ids: list | None = None,
              privileged: bool = False, cousin: str = "c000", enforce_static_check: bool = True) -> dict:
    from twin import starts as S
    from twin.env import PracticeEnv, run_privileged, run_program
    from twin.sandbox import check_dir
    rd = _run_dir(brain, run_id)
    path = _repo_path(script)
    st, _ = S.load("redteam", rd / "starts")
    ids = start_ids or sorted(sid for sid in st if sid.endswith("-000"))
    bad = [s for s in ids if s not in st]
    if bad:
        raise PermissionError(f"run_actor uses red-team starts only (R-*): {bad}")
    who = _caller() or "unknown"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else check_dir(path)["sha256"]
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid() % 100000}-{os.urandom(2).hex()}"
    out = dict(script=str(path.relative_to(REPO)), sha256=sha, caller=who, privileged=privileged, cousin=cousin,
               episodes=[])
    if privileged:
        if not who.startswith("redteam"):
            raise PermissionError(f"privileged mode is for red-team callers only (caller {who!r}; use --caller redteam-...)")
        log = rd / "env" / "privileged.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as fh:
            fh.write(json.dumps(dict(t=time.time(), caller=who, script=out["script"], sha256=sha, starts=ids)) + "\n")
        env = PracticeEnv(brain, run_id, "redteam", cousin, actor=f"redteam:{sha}", privileged=True,
                          session=f"redteam-{stamp}")
        for sid in ids:
            out["episodes"].append(run_privileged(env, sid, path))
        out["note"] = "privileged: raw access, never counted; flags show what the monitors caught"
        return out
    chk = check_dir(path)
    out["static_check"] = chk
    if not chk["ok"] and enforce_static_check:
        out["note"] = "rejected by the static checker before running (a catch at the static layer)"
        return out
    env = PracticeEnv(brain, run_id, "redteam", cousin, actor=f"redteam-actor:{sha}", session=f"redteam-{stamp}")
    for sid in ids:
        r = run_program(env, sid, path, stderr_path=rd / "env" / "sessions" / env.session / f"{sid}.stderr")
        out["episodes"].append(r)
    out["note"] = ("sandboxed actor; static check " + ("passed" if chk["ok"] else "FAILED but enforcement was off"))
    return out


@tool("monitor_selftest",
      "Sanity test of every runtime monitor: a planted violation per monitor (privileged, on a red-team start) must "
      "raise that monitor's flag, and a clean run must raise no invalidating flag. Not the exploit suite (that is the "
      "red team's, via run_actor).",
      {"type": "object", "properties": {**_RUN, "cases": {"type": "array", "items": {"type": "string"}}}})
def monitor_selftest(brain: str = "claude", run_id: str = "run-standin-1", cases: list | None = None) -> dict:
    from tests.monitor_sanity import CASES, dt_half_logic, run_case
    rd = _run_dir(brain, run_id)
    names = cases or list(CASES)
    rows = [run_case(n, rd) for n in names]
    if not cases:
        rows.append(dt_half_logic())
    return dict(ok=all(r["ok"] for r in rows), n=len(rows), n_ok=sum(r["ok"] for r in rows),
                cases=[{k: r.get(k) for k in ("case", "ok", "expect", "fired", "script_error")} for r in rows])


@tool("envelope_check",
      "E.envelope_violations for one routine: compare its episodes' force peak, approach and insertion speed, "
      "insertion duration and wiggle with the real successful attempts of the part's family (TRAIN recordings, "
      "1st-99th percentile), and report the B2 reconstruction status the check also requires. Give a skill_run "
      "result file or episode record files.",
      {"type": "object", "required": ["part"], "properties": {
          **_RUN, "part": {"type": "string"},
          "result_file": {"type": "string", "description": "skill_run result.json"},
          "episode_files": {"type": "array", "items": {"type": "string"}}}})
def envelope_check(part: str, brain: str = "claude", run_id: str = "run-standin-1", result_file: str | None = None,
                   episode_files: list | None = None) -> dict:
    from twin.envelope import check_routine, family_of, load
    rd = _run_dir(brain, run_id)
    env = load(rd / "envelope" / "envelope.json")
    feats = []
    if result_file:
        res = json.loads(_repo_path(result_file).read_text())
        feats += [r["envelope"]["features"] for r in res.get("rows", []) if r.get("part") == part
                  and r.get("envelope") and r["envelope"].get("features")]
    for f in episode_files or []:
        rec = json.loads(_repo_path(f).read_text())
        if rec.get("part") == part and rec.get("envelope", {}).get("features"):
            feats.append(rec["envelope"]["features"])
    if not feats:
        raise ValueError(f"no episodes of {part!r} with envelope features in the given files")
    v = check_routine(feats, part, env)
    b2 = _b2_status(brain)
    if b2.get("state") != "pass":
        v["reasons"].append(f"B2 reconstruction does not pass for this family (latest B2.outcome_bacc: {b2})")
        v["violates"] = True
    fam = family_of(part)
    return dict(part=part, family=fam, violates=v["violates"], reasons=v["reasons"], counts=v["counts"],
                n_episodes=v["n_episodes"], b2=b2, envelope=env["envelope"].get(fam, {}).get("features"),
                quantiles=env["quantiles"])


def _b2_status(brain: str) -> dict:
    f = REPO / "runs" / "status" / "checks.jsonl"
    last = None
    if f.exists():
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("check_id") == "B2.outcome_bacc" and r.get("brain") == brain:
                last = r
    if last is None:
        return dict(state="not measured", note="B2 is not recorded per part family yet")
    return dict(state=last.get("state"), value=last.get("value"), note="B2 is not recorded per part family yet")


@tool("integrity_info",
      "The frozen evaluator of a run: start sets (counts, sha256), cousins (ranges, placeholders), the real envelope "
      "per family, evaluator code hash, monitor thresholds, API limits, sandbox availability and the frozen budget.",
      {"type": "object", "properties": _RUN})
def integrity_info(brain: str = "claude", run_id: str = "run-standin-1") -> dict:
    from twin import env as E
    from twin.evaluator import EVALUATOR_VERSION, STAGES, code_hash
    from twin.monitors import THRESHOLDS
    from twin.sandbox import ALLOWED_IMPORTS, namespaces_available
    rd = _run_dir(brain, run_id)

    def j(p):
        return json.loads(p.read_text()) if p.exists() else None
    sm, cm, ev = j(rd / "starts" / "manifest.json"), j(rd / "cousins" / "manifest.json"), j(rd / "envelope" / "envelope.json")
    cs = j(rd / "cousins" / "cousins.json")
    out = dict(
        starts=None if sm is None else dict(files=sm["files"], counts=sm["counts"], frozen_at=sm["frozen_at"],
                                            heldout_subset=sm["heldout_subset"], nominal_board=sm["nominal_board"]["source"],
                                            manifest_sha256=(rd / "starts" / "manifest.sha256").read_text().strip()),
        cousins=None if cm is None else dict(n=cm["n"], combos=cm["combos"], placeholders=cm["placeholders"],
                                             ranges=cs["ranges"] if cs else None, sha256=cm["sha256"]),
        envelope=None if ev is None else {k: dict(n_success=v["n_success"], peak_force_any_max_N=v["peak_force_any_max_N"],
                                                  ranges={f: [q.get("lo"), q.get("hi")] for f, q in v["features"].items()})
                                          for k, v in ev["envelope"].items() if "_" in k},
        evaluator=dict(version=EVALUATOR_VERSION, code_hash=code_hash(), stages=list(STAGES)),
        monitors=THRESHOLDS,
        api_limits=dict(speed_cap=E.SPEED_CAP, ang_speed_cap=E.ANG_SPEED_CAP, force_cap=E.FORCE_CAP,
                        grip_force=E.GRIP_FORCE, move_max_s=E.MOVE_MAX_S, wait_max_s=E.WAIT_MAX_S),
        sandbox=dict(namespaces=namespaces_available(), allowed_imports=sorted(ALLOWED_IMPORTS)),
        budget=j(rd / "env" / "budget.json"))
    return out
