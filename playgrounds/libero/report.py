"""Summarise an Argon run on LIBERO and compare it with the published Argon-high campaign [llm-libero-2026].

    .venv/bin/python playgrounds/libero/report.py --run-id feas-argon-high-1

Reads runs/libero/<run>/results/*.json (argon_loop.py) and playgrounds/libero/paper_argon_episodes.json (the
paper's per-episode numbers). Writes runs/libero/<run>/report.json and report.md.
Categories (the paper's Table 1): success; failure = model stop (done / give_up) or the 900-step horizon;
limit reached = the 80-call cap, the spend cap, an output-token cut-off (no tool call 3 turns running) or an API
error. Embodiment faults and repeated tool errors are listed apart (the paper's 'error' status).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from scipy.stats import beta

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# tasks whose instruction names left/right: LIVING_ROOM_SCENE5 (left/right plates), SCENE6 (right of the plate).
# Under views="paper" the agentview is mirrored left-right, so these episodes are compromised (reward probe 14:45).
LR_TASKS = {4, 6}
PAPER = dict(success_rate=0.43, usd_per_episode=7.84, limit_rate=0.41, calls_per_episode=53.76,
             source="[llm-libero-2026] Table 1 row 2 (Gemini-4 Argon, high); per-episode means from the site data")


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def category(r: dict) -> str:
    eb, lim = r.get("ended_by"), (r.get("limit") or {}).get("what")
    if eb == "success":
        return "success"
    if eb in ("done", "give_up", "step_limit"):
        return "failure"
    if eb == "limit_reached" and lim in ("llm_calls", "usd", "output_tokens", "no_tool_call", "api_error", "wall_s"):
        return "limit"
    return "error"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 3) if xs else None


def paper_stats(tasks: set[int] | None = None) -> dict:
    d = json.loads((HERE / "paper_argon_episodes.json").read_text())["episodes"]
    eps = [e for e in d if e["effort"] == "high" and (tasks is None or e["task_id"] in tasks)]
    n = len(eps)
    lim = sum(e["status"] in ("incomplete_budget",) for e in eps)
    err = sum(e["status"] == "error" for e in eps)
    return dict(n=n, successes=sum(e["success"] for e in eps), calls=mean([e["api_attempts"] for e in eps]),
                usd=mean([e["estimated_usd"] for e in eps]), input_tokens=mean([e["input_tokens"] for e in eps]),
                cached_tokens=mean([e["cached_tokens"] for e in eps]),
                output_tokens=mean([e["output_tokens"] for e in eps]), latency_s=mean([e["latency_s"] for e in eps]),
                steps=mean([e["steps"] for e in eps]), call_cap_hits=lim, error_status=err,
                per_task={t: f"{sum(e['success'] for e in eps if e['task_id'] == t)}/"
                             f"{sum(1 for e in eps if e['task_id'] == t)}" for t in sorted({e['task_id'] for e in eps})})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    run = REPO / "runs" / "libero" / a.run_id
    rows = [json.loads(p.read_text()) for p in sorted((run / "results").glob("*.json"))]
    n = len(rows)
    k = sum(1 for r in rows if r.get("outcome") == "success")
    k_counted = sum(1 for r in rows if r.get("counted_success"))
    cats = {c: sum(1 for r in rows if category(r) == c) for c in ("success", "failure", "limit", "error")}
    lo, hi = clopper_pearson(k, n) if n else (None, None)
    per = []
    for r in sorted(rows, key=lambda r: r["task_id"]):
        tok = r.get("tokens") or {}
        per.append(dict(task=r["task_id"], init=r["init_state"], views=r.get("views", "paper"),
                        lr_referent=r["task_id"] in LR_TASKS, outcome=r["outcome"], ended_by=r["ended_by"],
                        limit=(r.get("limit") or {}).get("what"), category=category(r), claim=r.get("claim"),
                        false_claim=r.get("false_claim"), steps=r.get("steps"), calls=r.get("llm_calls"),
                        input_tokens=tok.get("input"), cached_tokens=tok.get("cached"),
                        output_tokens=(tok.get("output") or 0) + (tok.get("thought") or 0),
                        usd=r.get("usd"), wall_s=r.get("wall_s"), model_s=r.get("model_latency_s"),
                        flags=[f["monitor"] for f in (r.get("flags") or [])],
                        replay_ok=(r.get("replay") or {}).get("ok")))
    ours = dict(n=n, successes=k, counted_successes=k_counted, success_rate=round(k / n, 3) if n else None,
                ci95_clopper_pearson=[round(lo, 3), round(hi, 3)] if n else None, categories=cats,
                limit_rate=round(cats["limit"] / n, 3) if n else None,
                limit_kinds={w: sum(1 for p in per if p["limit"] == w) for w in sorted({p["limit"] for p in per if p["limit"]})},
                calls=mean([p["calls"] for p in per]), input_tokens=mean([p["input_tokens"] for p in per]),
                cached_tokens=mean([p["cached_tokens"] for p in per]),
                output_tokens=mean([p["output_tokens"] for p in per]), usd=mean([p["usd"] for p in per]),
                usd_total=round(sum(p["usd"] or 0 for p in per), 3), wall_s=mean([p["wall_s"] for p in per]),
                model_s=mean([p["model_s"] for p in per]), steps=mean([p["steps"] for p in per]),
                false_claims=sum(1 for p in per if p["false_claim"]))
    paper_all = paper_stats()
    paper_ci = clopper_pearson(paper_all["successes"], paper_all["n"])
    tasks = {p["task"] for p in per}
    paper_same = paper_stats(tasks)
    rep = dict(run_id=a.run_id, ours=ours, per_episode=per, paper=dict(PAPER, **paper_all,
                                                                       ci95_clopper_pearson=[round(x, 3) for x in paper_ci]),
               paper_same_tasks=paper_same)
    (run / "report.json").write_text(json.dumps(rep, indent=1))
    L = [f"# {a.run_id}: Argon (high) on LIBERO-10, published protocol", "",
         f"Success {k}/{n} = {ours['success_rate']} (95% Clopper-Pearson {ours['ci95_clopper_pearson']}); "
         f"paper 43/100 (CI {[round(x, 3) for x in paper_ci]}).", "",
         "| | ours (n=%d) | paper Argon high (n=100) |" % n, "|---|---|---|",
         f"| success rate | {ours['success_rate']} | 0.43 |",
         f"| limit reached | {ours['limit_rate']} {ours['limit_kinds']} | 0.41 (29 call cap, 12-13 error/cut-off) |",
         f"| calls / episode | {ours['calls']} | {paper_all['calls']} |",
         f"| input tokens / ep | {ours['input_tokens']} | {paper_all['input_tokens']} |",
         f"| cached input / ep | {ours['cached_tokens']} | {paper_all['cached_tokens']} |",
         f"| output+thinking / ep | {ours['output_tokens']} | {paper_all['output_tokens']} |",
         f"| USD / episode | {ours['usd']} | {paper_all['usd']} |",
         f"| wall s / episode | {ours['wall_s']} (model {ours['model_s']}) | {paper_all['latency_s']} (latency) |",
         f"| control steps / ep | {ours['steps']} | {paper_all['steps']} |", "",
         "| task | init | views | outcome | ended by | calls | in tok | cached | out tok | USD | wall s | flags |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in per:
        vw = p["views"] + (" (mirrored; L/R task: compromised)" if p["views"] == "paper" and p["lr_referent"]
                           else " (mirrored)" if p["views"] == "paper" else "")
        L.append(f"| {p['task']} | {p['init']} | {vw} | {p['outcome']} | {p['ended_by']}{'/' + p['limit'] if p['limit'] else ''}"
                 f" | {p['calls']} | {p['input_tokens']} | {p['cached_tokens']} | {p['output_tokens']} | {p['usd']} |"
                 f" {p['wall_s']} | {', '.join(p['flags'])} |")
    ps = paper_same
    L += ["", f"Paper Argon-high on the same tasks {sorted(tasks)} (init 0-9, n={ps['n']}): success "
              f"{ps['successes']}/{ps['n']} = {round(ps['successes'] / ps['n'], 3) if ps['n'] else None}, "
              f"{ps['calls']} calls, ${ps['usd']}/episode, {ps['call_cap_hits']} at the 80-call cap + "
              f"{ps['error_status']} error, latency {ps['latency_s']} s, input {ps['input_tokens']} "
              f"(cached {ps['cached_tokens']}), output {ps['output_tokens']}.",
          f"Paper Argon-high successes per task (init 0-9): {paper_all['per_task']}"]
    (run / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
