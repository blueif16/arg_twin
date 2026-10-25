"""Run Argon attempt 1 on the 10 reserved feasibility episodes (splits.json), at most 3 at once.

Each episode is one argon_loop.py process with its own LIBERO server process, so at most 3 sim processes run.
The key is passed in by the caller's environment only (GEMINI_API_KEY=...), never written.

    GEMINI_API_KEY="$(cat ~/.config/arg_twin/gemini_key)" .venv/bin/python playgrounds/libero/run_feasibility.py \
        --run-id feas-argon-high-1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# expected-long tasks first (the paper's Argon-high per-task success: 4, 6 at 0/10; 9, 3, 5, 8 at 1-2/10)
ORDER = (4, 6, 9, 3, 5, 8, 2, 0, 1, 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--tasks", default=None, help="comma list of task ids (default: all 10)")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="extra args for argon_loop.py after --")
    a = ap.parse_args()
    if a.parallel > 3:
        raise SystemExit("at most 3 LIBERO sim processes at once (shared CPU)")
    sp = json.loads((HERE / "splits.json").read_text())["splits"]["feasibility"]["episodes"]
    eps = {e["task_id"]: e["init_state"] for e in sp}
    want = [int(t) for t in a.tasks.split(",")] if a.tasks else list(ORDER)
    queue = [(t, eps[t]) for t in ORDER if t in want]
    logs = REPO / "runs" / "libero" / a.run_id / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    extra = [x for x in a.extra if x != "--"]
    running: dict = {}
    t0 = time.time()
    while queue or running:
        while queue and len(running) < a.parallel:
            t, i = queue.pop(0)
            f = open(logs / f"t{t:02d}_i{i:02d}.loop.log", "a")
            p = subprocess.Popen([str(REPO / ".venv" / "bin" / "python"), str(HERE / "argon_loop.py"),
                                  "--run-id", a.run_id, "--split", "feasibility", "--task", str(t), "--init", str(i),
                                  *extra], stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO))
            running[p] = (t, i, f, time.time())
            print(f"[{time.time() - t0:7.0f}s] start task {t} init {i}", flush=True)
        time.sleep(5)
        for p in list(running):
            if p.poll() is not None:
                t, i, f, ts = running.pop(p)
                f.close()
                print(f"[{time.time() - t0:7.0f}s] end task {t} init {i}: exit {p.returncode} after "
                      f"{time.time() - ts:.0f}s", flush=True)
    print(f"all done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
