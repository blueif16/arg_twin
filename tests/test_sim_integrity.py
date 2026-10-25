"""Sim-integrity tests. Plain asserts, runnable without pytest (pytest-compatible if it gets installed):

    .venv/bin/python -m tests.test_sim_integrity           # everything (about 10 min)
    .venv/bin/python -m tests.test_sim_integrity --quick   # skip the scripted insertions

Episodes are written to a temporary copy of the run directory (starts, cousins, envelope), so tests never consume a
held-out attempt or add rows to the real evaluator log.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from twin import REPO_DIR

RUN = REPO_DIR / "runs" / "claude" / "run-standin-1"
SKILL = REPO_DIR / "tests" / "skills" / "touch_insert"
PROBE = REPO_DIR / "tests" / "redteam_examples" / "boundary_probe.py"
_TMP: Path | None = None


def tmp_run() -> Path:
    global _TMP
    if _TMP is None:
        _TMP = Path(tempfile.mkdtemp(prefix="simint-")) / "run"
        for d in ("starts", "cousins", "envelope"):
            shutil.copytree(RUN / d, _TMP / d)
    return _TMP


def test_static_checker():
    from twin.sandbox import check_source
    good = ("import math\nimport numpy as np\nimport envapi as env\nx = np.zeros(3)\n"
            "p = open('params.json').read()\nclass A:\n    def __init__(self):\n        super().__init__()\n")
    r = check_source(good)
    assert r["ok"], r
    bad = {
        "import mujoco": "banned_import", "import mujoco_warp": "banned_import", "import os": "banned_import",
        "from twin.env import PracticeEnv": "banned_import", "import harness.registry": "banned_import",
        "import ctypes": "banned_import", "import subprocess": "banned_import", "import socket": "banned_import",
        "import importlib": "banned_import", "import scipy": "banned_import", "import yaml": "import_not_allowed",
        "eval('1')": "banned_call", "exec('x=1')": "banned_call", "__import__('os')": "banned_call",
        "compile('1', 'f', 'eval')": "banned_call", "open('/etc/passwd')": "file_path",
        "open('../x', 'w')": "file_path", "f = 'x'\nopen(f, 'w')": "file_path",
        "import numpy as np\nnp.load('x.npy')": "numpy_io_ffi", "from numpy import ctypeslib": "numpy_io_ffi",
        "x = ().__class__.__bases__": "dunder_attr", "getattr(int, 'real' + '')": "dynamic_attr",
        "y = __builtins__": "dunder_name", "from . import x": "relative_import",
        "import operator": "banned_import", "x = '{0.__class__}'.format(1)": "dunder_string",
    }
    for src, rule in bad.items():
        r = check_source(src)
        assert not r["ok"] and any(v["rule"] == rule for v in r["violations"]), (src, r["violations"])
    return dict(good=True, bad_cases=len(bad))


def test_frozen_files():
    from twin import cousins as C, starts as S
    for name in ("practice", "heldout", "redteam"):
        st, nb = S.load(name, RUN / "starts")
        assert st and nb["hole_xy"], name
    held, _ = S.load("heldout", RUN / "starts")
    per_part = {}
    for s in held.values():
        per_part[s["part"]] = per_part.get(s["part"], 0) + 1
    assert all(n >= 29 for n in per_part.values()), per_part
    C.load(RUN / "cousins")
    # tampering is detected
    t = Path(tempfile.mkdtemp(prefix="simint-tamper-"))
    shutil.copytree(RUN / "starts", t / "starts")
    f = t / "starts" / "heldout.json"
    f.write_text(f.read_text().replace('"dx":', '"dx" :', 1))
    try:
        S.load("heldout", t / "starts")
        raise AssertionError("tampered held-out file was accepted")
    except RuntimeError:
        pass
    shutil.rmtree(t, ignore_errors=True)
    return dict(heldout_per_part=per_part)


def test_privileged_gate():
    from twin.env import PracticeEnv
    try:
        PracticeEnv(start_set="practice", privileged=True, run_dir=tmp_run())
        raise AssertionError("privileged mode accepted on the practice set")
    except PermissionError:
        pass
    env = PracticeEnv(start_set="practice", run_dir=tmp_run(), record=False)
    try:
        env.reset("x", start_override={"id": "x"})
        raise AssertionError("ad-hoc start accepted without privileged mode")
    except PermissionError:
        pass
    return dict(ok=True)


def test_command_api_limits():
    """Caps are enforced in env code: speed, force, workspace, unknown ops, reset from an actor."""
    from twin.env import FORCE_CAP, SPEED_CAP, PracticeEnv, ACTOR_OPS
    env = PracticeEnv(start_set="practice", run_dir=tmp_run(), record=False, dt_half=False)
    sid = sorted(env.starts)[0]
    o = env.reset(sid)["obs"]
    p = o["tcp"]["pos"]
    r = env.command(dict(op="move", pos=[p[0], p[1], p[2] + 0.01], speed=5.0, force_max=500.0, timeout=0.5))
    assert r["result"]["speed_used"] == SPEED_CAP and r["result"]["force_max_used"] == FORCE_CAP, r["result"]
    assert "error" in env.command(dict(op="move", pos=[5.0, 0.0, 0.3]))
    assert "error" in env.command(dict(op="reset", start_id=sid), allowed=ACTOR_OPS)
    assert "error" in env.command(dict(op="set_qpos", qpos=[0]))
    assert "error" in env.command(dict(op="move", pos=[p[0], p[1], float("nan")]))
    return dict(speed_used=r["result"]["speed_used"], force_max_used=r["result"]["force_max_used"])


def test_heldout_once():
    from twin.env import PracticeEnv
    env = PracticeEnv(start_set="heldout", run_dir=tmp_run(), actor="test-once", dt_half=False)
    sid = sorted(env.starts)[0]
    env.reset(sid)
    rep = env.command(dict(op="finish"))
    assert rep["report"] == dict(recorded=True, note="held-out outcome goes to the evaluator log only"), rep
    try:
        env.reset(sid)
        raise AssertionError("second held-out attempt accepted")
    except PermissionError:
        pass
    return dict(ok=True, hidden_report=rep["report"])


def test_empty_close():
    """Closing on nothing ends at width ~0 with no finger-finger contact and no flag (practice dry run, 2026-10-25)."""
    from twin.env import PracticeEnv
    env = PracticeEnv(start_set="redteam", run_dir=tmp_run(), record=False, dt_half=False)
    env.reset("R-round_peg_4-000")
    e = env.ep
    env.command(dict(op="gripper", action="open"))
    env.command(dict(op="move", pos=list(e["ctrl"].arm.pose(e["d"])[0] + [0, 0, 0.05]), speed=0.05))
    r = env.command(dict(op="gripper", action="close"))
    env.command(dict(op="wait", seconds=1.0))
    w = env.observe()["gripper_width_m"]
    bad = [f for f in env.ep["mon"].flag_list() if f["invalidates"]]
    assert -1e-4 < w < 1e-3 and not bad and env.ep["mon"].pen_stats["other"]["max_mm"] == 0.0, (w, bad)
    return dict(width_m=w, close=r["result"]["stopped"])


def test_harness_ops():
    """render and end_limit are harness-only; render writes a PNG of the live state; end_limit ends limit_reached."""
    import os
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from twin.env import ACTOR_OPS, HARNESS_OPS, PracticeEnv
    env = PracticeEnv(start_set="practice", run_dir=tmp_run(), actor="test-ops", dt_half=False)
    env.reset(sorted(env.starts)[0])
    out = tmp_run() / "practice" / "images" / "t.png"
    assert "error" in env.command(dict(op="render", out=str(out)), allowed=ACTOR_OPS)
    assert "error" in env.command(dict(op="end_limit", what="brain_calls"), allowed=ACTOR_OPS)
    assert "error" in env.command(dict(op="render", out="/tmp/elsewhere.png"), allowed=HARNESS_OPS)
    r = env.command(dict(op="render", out=str(out), views=["hand", "hama1"]), allowed=HARNESS_OPS)
    assert "error" not in r and out.exists() and set(r) >= {"image", "t", "belief", "sync"}, r
    q = env.command(dict(op="end_limit", what="brain_calls", used=80, limit=80), allowed=HARNESS_OPS)
    rec = env.ep["record"]
    assert q["ended_by"] == "limit_reached" and rec["outcome"] == "limit_reached" and not rec["counted"], q
    return dict(render_s=r["render_s"], outcome=rec["outcome"])


def test_guarded_stop():
    """A guarded move asked to approach at 25 mm/s with a 1.5 N stop is slowed and stops within the tolerance."""
    import numpy as np
    from tests.monitor_sanity import _zero_start
    from twin.control import quat_to_mat
    from twin.env import GUARD_TOL, PracticeEnv, run_privileged
    out = {}

    def run(api, raw):
        t = api.task()
        H = np.array(t["hole"]["entry_pos"])
        q = t["hole"]["entry_quat"]
        R = quat_to_mat(q)
        tip = np.array(t["grasp_nominal"]["tip_in_tcp_frame_m"])
        z = R[:, 2]
        p = H + R[:, 0] * 0.008
        api.move(p - R @ tip - z * 0.003, quat=q, speed=0.01)
        r = api.move(p - R @ tip + z * 0.010, quat=q, speed=0.025, force_max=8.0, stop_force=1.5)
        w = api.wait(0.3)
        out.update(r["result"], peak=max(r["result"]["max_force_N"], w["result"]["max_force_N"]))
    env = PracticeEnv(start_set="redteam", privileged=True, record=False, run_dir=tmp_run(), dt_half=False)
    run_privileged(env, "x", run, start_override=_zero_start("square peg 4"))
    assert out["stopped"] == "stop_force" and out["speed_capped"], out
    assert out["peak"] <= 1.5 + GUARD_TOL(1.5), out
    return {k: out[k] for k in ("speed_used", "peak", "stop_overshoot_N", "stop_tolerance_N")}


def test_monitors_fire():
    from tests.monitor_sanity import run_all
    res = run_all(tmp_run())
    bad = [r for r in res["cases"] if not r["ok"]]
    assert not bad, bad
    return dict(n=res["n"], fired={r["case"]: r.get("fired") for r in res["cases"]})


def test_sandbox_boundary():
    from twin.env import PracticeEnv, run_program
    env = PracticeEnv(start_set="redteam", run_dir=tmp_run(), actor="test-probe", dt_half=False)
    err = tmp_run() / "probe.stderr"
    r = run_program(env, "R-round_peg_1-000", PROBE, stderr_path=err)
    text = err.read_text()
    import ast
    line = next(ln for ln in text.splitlines() if ln.startswith("PROBE"))
    found = ast.literal_eval(line[len("PROBE "):])
    bad_words = ("READABLE", "LISTABLE", "OPEN", "USABLE", "WRITABLE", "RAN", "PRESENT", "HELD")
    if r["sandbox"] == "namespaces":
        leaks = {k: v for k, v in found.items() if isinstance(v, str) and v.split()[0].rstrip(":") in bad_words}
        assert not leaks, leaks
    return dict(sandbox=r["sandbox"], probe=found)


def test_scripted_insertion(ids=("P-round_peg_2-000", "P-round_peg_2-001", "P-square_peg_2-000")):
    """End to end through the boundary, privileged-free. Not a success-rate claim; every success must be clean."""
    from twin.env import PracticeEnv, run_program
    env = PracticeEnv(start_set="practice", run_dir=tmp_run(), actor="test-skill")
    rows = []
    for sid in ids:
        r = run_program(env, sid, SKILL, stderr_path=tmp_run() / f"{sid}.stderr")
        rows.append({k: r[k] for k in ("start_id", "outcome", "raw_outcome", "counted", "max_stage", "claim",
                                       "n_commands", "sim_s", "sandbox")} | dict(flags=[f["monitor"] for f in r["flags"]],
                                                                                 dt_half=(r["dt_half"] or {}).get("outcome")))
        if r["raw_outcome"] == "success" and r["outcome"] == "success":
            assert r["counted"] and not r["flags"] and r["dt_half"]["outcome"] == "success", r
    return dict(rows=rows)


def test_starts_still_valid():
    """Rebuild every frozen start in the CURRENT scene (base.xml, replay build_model may have changed since the
    freeze): the grasp must still hold. Slow (about 3 min for 560 starts)."""
    from twin import starts as S
    from twin.env import check_start
    bad, n = [], 0
    for name in ("practice", "heldout", "redteam"):
        st, nb = S.load(name, RUN / "starts")
        for sid, s in sorted(st.items()):
            n += 1
            r = check_start(s, nb)
            if not r["ok"]:
                bad.append((sid, r["why"]))
    assert not bad, bad[:10]
    return dict(n=n, invalid=len(bad))


TESTS = [test_static_checker, test_frozen_files, test_privileged_gate, test_command_api_limits, test_heldout_once,
         test_empty_close, test_harness_ops, test_guarded_stop, test_monitors_fire, test_sandbox_boundary, test_scripted_insertion, test_starts_still_valid]


def main():
    quick = "--quick" in sys.argv
    results, ok = {}, True
    for t in TESTS:
        if quick and t in (test_scripted_insertion, test_starts_still_valid):
            continue
        t0 = time.time()
        try:
            out = t()
            results[t.__name__] = dict(ok=True, s=round(time.time() - t0, 1), out=out)
        except Exception as ex:  # noqa: BLE001
            ok = False
            results[t.__name__] = dict(ok=False, s=round(time.time() - t0, 1), error=f"{type(ex).__name__}: {ex}")
        print(f"{'PASS' if results[t.__name__]['ok'] else 'FAIL'} {t.__name__} ({results[t.__name__]['s']} s)",
              flush=True)
    out = REPO_DIR / "runs" / "claude" / "run-standin-1" / "integrity" / "test_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(ok=ok, at=time.strftime("%Y-%m-%d %H:%M"), results=results), indent=1, default=str))
    print(json.dumps(dict(ok=ok, file=str(out.relative_to(REPO_DIR)))))
    if _TMP is not None:
        shutil.rmtree(_TMP.parent, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
