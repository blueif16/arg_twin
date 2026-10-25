"""Summarise twin/pf_batch results: distribution metrics (validation v10 B2), per-attempt track metrics with
recording-level CV for the only tuned quantity (the P(seat) threshold), force/onset errors, posterior spread, N.

    python -m twin.pf_report --dir runs/claude/run-standin-1/replay/pf/v4_stratified
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_TRAIN = dict(success=239, failure=69)  # full train population of in-hand peg windows (stratified set reweights to it)


def load(d: Path):
    rows = []
    for f in sorted(d.glob("worker*.jsonl")):
        rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    ok = [r for r in rows if "error" not in r]
    return ok, [r for r in rows if "error" in r]


def wks(a, wa, b, wb):
    """Weighted two-sample KS distance."""
    xs = np.sort(np.unique(np.r_[a, b]))
    def cdf(x, w, t):
        return np.array([w[x <= v].sum() for v in t]) / w.sum()
    return float(np.max(np.abs(cdf(np.asarray(a), np.asarray(wa), xs) - cdf(np.asarray(b), np.asarray(wb), xs))))


def pop_weights(R):
    ns = sum(r["real_success"] for r in R)
    nf = len(R) - ns
    return np.array([N_TRAIN["success"] / ns if r["real_success"] else N_TRAIN["failure"] / nf for r in R])


def dist_metrics(R, rng, nboot=1000):
    w = pop_weights(R)
    real = np.array([r["real_success"] for r in R], float)
    p = np.array([r["reconstruction"]["p_seat"] for r in R])
    err = abs(np.sum(w * p) / w.sum() - np.sum(w * real) / w.sum())
    boots = []
    for _ in range(nboot):
        i = rng.integers(0, len(R), len(R))
        boots.append(np.sum(w[i] * (p[i] - real[i])) / w[i].sum())
    pk_sim = np.array([r["reconstruction"]["peak_N_draw"] for r in R])
    pk_real = np.array([r["real_peak_N"] for r in R])
    ks = wks(pk_sim, w, pk_real, w)
    by_part = {}
    for part in sorted({r["part"] for r in R}):
        ii = [k for k, r in enumerate(R) if r["part"] == part]
        by_part[part] = dict(sim=round(float(np.sum(w[ii] * p[ii]) / w[ii].sum()), 3), real=round(float(np.sum(w[ii] * real[ii]) / w[ii].sum()), 3), n=len(ii))
    return dict(seat_rate_sim=round(float(np.sum(w * p) / w.sum()), 3), seat_rate_real=round(float(np.sum(w * real) / w.sum()), 3),
                seat_rate_err=round(float(err), 3), diff_ci95=[round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
                force_ks=round(ks, 3), by_part=by_part)


def cv_bacc(R, key, k=5):
    """Threshold on P(seat) chosen on 4/5 of recordings, scored on the held-out 1/5 (folds by episode)."""
    eps = sorted({r["episode"] for r in R})
    fold = {e: i % k for i, e in enumerate(eps)}
    grid = np.linspace(0, 1, 101)
    pred = {}
    chosen = []
    def bacc(rows, th):
        P = [r for r in rows if r["real_success"]]
        N = [r for r in rows if not r["real_success"]]
        tpr = np.mean([r[key]["p_seat"] >= th for r in P]) if P else np.nan
        tnr = np.mean([r[key]["p_seat"] < th for r in N]) if N else np.nan
        return (tpr + tnr) / 2
    for f in range(k):
        fit = [r for r in R if fold[r["episode"]] != f]
        th = float(max(grid, key=lambda t: bacc(fit, t)))
        chosen.append(th)
        for r in R:
            if fold[r["episode"]] == f:
                pred[(r["episode"], r["segment_index"])] = r[key]["p_seat"] >= th
    P = [r for r in R if r["real_success"]]
    N = [r for r in R if not r["real_success"]]
    tp = sum(pred[(r["episode"], r["segment_index"])] for r in P)
    tn = sum(not pred[(r["episode"], r["segment_index"])] for r in N)
    full_th = float(max(grid, key=lambda t: bacc(R, t)))
    return dict(tpr=round(tp / len(P), 3), tnr=round(tn / len(N), 3), bacc=round((tp / len(P) + tn / len(N)) / 2, 3),
                tp=f"{tp}/{len(P)}", tn=f"{tn}/{len(N)}", thresholds_by_fold=chosen, threshold_all_train=full_th)


def summary(R):
    rng = np.random.default_rng(0)
    out = dict(n=len(R), dist=dist_metrics(R, rng), recon=cv_bacc(R, "reconstruction"), pred=cv_bacc(R, "prediction"))
    c = [r for r in R if r["real_peak_N"] > 2.0]
    ab = [abs(r["reconstruction"]["peak_N_draw"] - r["real_peak_N"]) for r in c]
    rel = [abs(r["reconstruction"]["peak_N_draw"] - r["real_peak_N"]) / r["real_peak_N"] for r in c]
    on = [r["onset_err_s_draw"] for r in R if r.get("onset_err_s_draw") is not None]
    out["force"] = dict(peak_abs_err_N_median=round(float(np.median(ab)), 2), peak_rel_err_median=round(float(np.median(rel)), 3),
                        onset_err_s_median=round(float(np.median(on)), 3) if on else None)
    post = {}
    for mode in ("prediction", "reconstruction"):
        post[mode] = {k: round(float(np.median([r[mode]["posterior"][k]["std"] / max(r[mode]["posterior"][k]["prior_std"], 1e-12) for r in R])), 3)
                      for k in ("dx", "dy", "g", "rock", "pivot", "friction", "widen")}
        post[mode]["ess_median"] = round(float(np.median([r[mode]["ess"] for r in R])), 1)
        post[mode]["beta_median"] = float(np.median([r[mode]["beta"] for r in R]))
        post[mode]["half_split_abs_diff_p_seat_median"] = round(float(np.median([abs(np.diff(r[mode]["p_seat_halves"])[0]) for r in R])), 4)
    out["posterior_std_over_prior_std_median"] = post
    out["prior_seat_rate_mean"] = round(float(np.mean([r["seat_rate_prior"] for r in R])), 3)
    out["wall_s_median"] = float(np.median([r["wall"]["total_s"] for r in R]))
    out["nan_worlds_total"] = int(sum(r["nan_worlds"] for r in R))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    ok, err = load(Path(a.dir))
    res = {}
    for perm in (False, True):
        R = [r for r in ok if r["permissive"] == perm]
        if R:
            res["permissive" if perm else "frozen"] = summary(R)
    if "frozen" in res and "permissive" in res:
        res["dist_negative_control"] = round(res["permissive"]["dist"]["seat_rate_err"] - res["frozen"]["dist"]["seat_rate_err"], 3)
    res["errors"] = [e.get("error") for e in err]
    Path(a.dir, "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
