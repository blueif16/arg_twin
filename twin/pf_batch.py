"""Batch driver for twin/pf.py: python -m twin.pf_batch --worker i --nworkers k --out DIR [--n 1024]
Runs the stratified train set (all real failures + an equal number of seeded real successes) through the frozen
and the permissive twin; one JSON line per attempt and mode. Train only (splits.json filter in train_attempts)."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def attempt_set(seed: int = 0):
    from twin.replay import train_attempts
    att = train_attempts()
    fails = [a for a in att if not a["success"]]
    succ = [a for a in att if a["success"]]
    random.Random(seed).shuffle(succ)
    return fails + succ[:len(fails)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default="runs/claude/run-standin-1/replay/params_v3.json")
    ap.add_argument("--registered", action="store_true", help="variant 3c-reg (reconstruction only)")
    ap.add_argument("--max-runs", type=int, default=10, help="exit after this many runs (frees GPU memory; rerun to continue)")
    a = ap.parse_args()
    from twin.replay import ReplayParams
    from twin import pf
    rp = ReplayParams.from_dict(json.load(open(a.params))["params"])
    prior = json.loads(pf.POP_PRIOR.read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"worker{a.worker}.jsonl"
    done = set()
    for ff in out.glob("worker*.jsonl"):
        for line in ff.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r or "index" in r.get("error", "").lower():
                    done.add((r["episode"], r["segment_index"], r["permissive"]))
    att = attempt_set()
    mine = [x for i, x in enumerate(att) if i % a.nworkers == a.worker]
    runs = 0
    for x in mine:
        for perm in (False, True):
            key = (x["episode"], x["segment_index"], perm)
            if key in done:
                continue
            t = time.time()
            try:
                r = pf.run_attempt(x["episode"], x["segment_index"], rp, prior, n=a.n, seed=x["episode"] * 1000 + x["segment_index"],
                                   permissive=perm, registered=a.registered)
            except Exception as e:
                r = dict(episode=x["episode"], segment_index=x["segment_index"], permissive=perm, error=f"{type(e).__name__}: {e}")
            with open(f, "a") as fh:
                fh.write(json.dumps(r, default=float) + "\n")
            print(key, round(time.time() - t, 1), r.get("reconstruction", {}).get("p_seat"), flush=True)
            runs += 1
            if "error" in r or runs >= a.max_runs:
                return
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
