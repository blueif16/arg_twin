"""Real-data envelope: what real, successful human insertions look like, per part family (E.envelope_violations).

Without a real arm, the real check is the real data (validation.yaml E.envelope_violations, rules.md Grounding:
"a strategy faster, harder or different from anything in the real data gets flagged"). For every in-hand Insert
attempt in the TRAIN split (playbook/splits.json; gate and exam are never read here) this module computes the
same features it computes for a twin episode, with the same code, at the same 30 Hz rate:

  peak_force_N        max |F - F_ref| while the part is held (F_ref = median of the first 0.5 s; F on the tool,
                      base frame: REASSEMBLE force.base / the twin's wrist sensor)
  approach_speed_mm_s peak TCP speed in the 1.5 s before contact onset (speed over 0.1 s spans)
  insert_speed_mm_s   peak TCP speed between contact onset and release
  insert_duration_s   release - onset
  wiggle_mm           2 x RMS of the horizontal TCP motion between onset and release, after removing a linear trend
  wiggle_deg          2 x RMS of the hand-axis angle from its mean between onset and release
  contact onset: |F - F_ref| > 2 N (twin/replay.py CONTACT_N); release: fingers open 3 mm past the holding width.

The envelope of a family is the [q_lo, q_hi] quantile range (default 1st-99th percentile) of each feature over
real successful attempts. `check_episode` flags a twin episode outside it (side: above = faster/harder/longer,
below = slower/softer/shorter); `check_routine` turns a set of episodes into the routine-level verdict.

    .venv/bin/python -I -c "import sys; sys.path.insert(0, '.'); from twin.envelope import main; main()"
writes runs/<brain>/<run_id>/envelope/envelope.json (train only).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from twin import REPO_DIR

HOLD_MAX = 0.075      # finger sum below this = holding something (twin/replay.py)
RELEASE_OPEN = 0.003  # fingers this much wider than the holding width = released (twin/replay.py)
CONTACT_N = 2.0       # contact onset threshold on |F - F_ref| (twin/replay.py)
REF_FRAMES = 15       # F_ref = median of the first 0.5 s at 30 Hz (twin/replay.py attempt_info)
FPS = 30.0
APPROACH_S = 1.5      # approach window before onset (the replay window lead)
SPAN = 3              # speed over 3-frame (0.1 s) spans: suppresses 30 Hz pose noise
FEATURES = ("peak_force_N", "approach_speed_mm_s", "insert_speed_mm_s", "insert_duration_s", "wiggle_mm",
            "wiggle_deg")
DEFAULT_Q = (1.0, 99.0)
DEFAULT_DIR = REPO_DIR / "runs" / "claude" / "run-standin-1" / "envelope"


def family_of(part: str) -> str:
    """'round peg 3' -> 'round_peg' (twin/parts.py families)."""
    return part.rsplit(" ", 1)[0].replace(" ", "_") if part[-1:].isdigit() else part.replace(" ", "_")


def _quat_z(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


def features(t, pos, quat, F, grip) -> dict:
    """Features of one attempt from 30 Hz arrays: t (n), pos (n,3) TCP m, quat (n,4) wxyz, F (n,3) N on the tool,
    grip (n) finger sum m. Identical for real segments and twin episodes."""
    t = np.asarray(t, float)
    pos = np.asarray(pos, float)
    quat = np.asarray(quat, float)
    F = np.asarray(F, float)
    g = np.asarray(grip, float)
    n = len(t)
    out = {k: None for k in FEATURES}
    out.update(n=n, onset_s=None, release_s=None, held=False, released=False)
    if n < 4:
        out["why"] = "too short"
        return out
    hold = g < HOLD_MAX
    if not hold[: max(1, n // 3)].any():
        out["why"] = "no part in hand in the first third (a push)"
        return out
    out["held"] = True
    w = float(np.median(g[hold]))
    first = int(np.argmax(hold))
    opened = np.flatnonzero(g > w + RELEASE_OPEN)
    opened = opened[opened > first]
    rel = int(opened[0]) - 1 if len(opened) else n - 1
    out["released"] = bool(len(opened))
    F_ref = np.median(F[:REF_FRAMES], axis=0)
    dF = np.linalg.norm(F[: rel + 1] - F_ref, axis=1)
    on = np.flatnonzero(dF > CONTACT_N)
    onset = int(on[0]) if len(on) else None
    out["peak_force_N"] = round(float(dF.max()), 3)
    out["release_s"] = round(float(t[rel] - t[0]), 3)
    # speed over SPAN-frame spans, assigned to the span's first frame
    dt = float(np.median(np.diff(t))) if n > 1 else 1 / FPS
    if n > SPAN:
        sp = 1000 * np.linalg.norm(pos[SPAN:] - pos[:-SPAN], axis=1) / (SPAN * dt)
    else:
        sp = np.zeros(0)
    if onset is None:
        out["why"] = f"no contact onset (|F - F_ref| never above {CONTACT_N} N while held)"
        if len(sp):
            out["approach_speed_mm_s"] = round(float(sp[: max(1, rel - SPAN + 1)].max()), 2)
        return out
    out["onset_s"] = round(float(t[onset] - t[0]), 3)
    a0 = max(0, onset - int(round(APPROACH_S / dt)))
    seg = sp[a0:max(a0 + 1, onset - SPAN + 1)]
    out["approach_speed_mm_s"] = round(float(seg.max()), 2) if len(seg) else None
    ins = sp[onset:max(onset + 1, rel - SPAN + 1)]
    out["insert_speed_mm_s"] = round(float(ins.max()), 2) if len(ins) else None
    out["insert_duration_s"] = round(float(t[rel] - t[onset]), 3)
    k = np.arange(onset, rel + 1)
    if len(k) >= 4:
        tt = t[k] - t[k].mean()
        res = []
        for ax in (0, 1):
            y = pos[k, ax]
            A = np.c_[np.ones_like(tt), tt]
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            res.append(y - A @ coef)
        out["wiggle_mm"] = round(float(2 * 1000 * np.sqrt(np.mean(np.sum(np.square(res), axis=0)))), 3)
        zs = np.array([_quat_z(q) for q in quat[k]])
        zm = zs.mean(0)
        zm /= np.linalg.norm(zm)
        ang = np.degrees(np.arccos(np.clip(zs @ zm, -1, 1)))
        out["wiggle_deg"] = round(float(2 * np.sqrt(np.mean(ang ** 2))), 3)
    else:
        out["wiggle_mm"] = 0.0
        out["wiggle_deg"] = 0.0
    return out


# ----------------------------------------------------------------------------------------------- real data
def real_attempts(family: str = "peg") -> list[dict]:
    """Features of every in-hand Insert attempt in TRAIN recordings (splits.json filter in twin.replay)."""
    from twin.replay import segments, signals, split_of, train_attempts
    rows = []
    for a in train_attempts(family):
        ep = a["episode"]
        if split_of(ep) != "train":  # belt and braces: train_attempts already filters
            continue
        s = signals(ep)
        seg = next(g for g in segments(ep) if g["index"] == a["segment_index"])
        sl = slice(seg["a"], seg["b"])
        f = features(s["t"][sl], s["ee"][sl, :3], s["ee"][sl, 3:7], s["F"][sl], s["grip"][sl].sum(1))
        f.update(episode=ep, segment_index=a["segment_index"], part=a["part"], family=family_of(a["part"]),
                 success=bool(a["success"]))
        rows.append(f)
    return rows


def _quantiles(vals: list[float], q=DEFAULT_Q) -> dict:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return dict(n=0)
    return dict(n=int(len(v)), lo=round(float(np.percentile(v, q[0])), 3), hi=round(float(np.percentile(v, q[1])), 3),
                p5=round(float(np.percentile(v, 5)), 3), p50=round(float(np.percentile(v, 50)), 3),
                p95=round(float(np.percentile(v, 95)), 3), min=round(float(v.min()), 3), max=round(float(v.max()), 3))


def build(out_dir: Path = DEFAULT_DIR, q=DEFAULT_Q) -> dict:
    t0 = time.time()
    rows = real_attempts("peg")
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["family"], []).append(r)
        groups.setdefault(r["part"], []).append(r)
    env = {}
    for key, rs in sorted(groups.items()):
        ok = [r for r in rs if r["success"]]
        env[key] = dict(
            n_success=len(ok), n_failure=len(rs) - len(ok),
            features={f: _quantiles([r[f] for r in ok], q) for f in FEATURES},
            # the plausibility monitor's force bound: the largest peak in ANY real attempt (success or failure)
            peak_force_any_max_N=round(float(max(r["peak_force_N"] for r in rs if r["peak_force_N"] is not None)), 3),
            no_onset_success=sum(1 for r in ok if r["onset_s"] is None))
    out = dict(
        generated_by="twin/envelope.py build", generated_at=time.strftime("%Y-%m-%d %H:%M"),
        source="REASSEMBLE train split only (playbook/splits.json), in-hand peg Insert attempts (twin.replay."
               "train_attempts: fingers closed in the first third); features from twin/envelope.py features()",
        quantiles=list(q), features=list(FEATURES), n_attempts=len(rows),
        definitions=__doc__.split("Real-data envelope:")[1].split("The envelope of a family")[0].strip(),
        envelope=env, wall_s=round(time.time() - t0, 1))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "attempts.json").write_text(json.dumps(rows, indent=0))
    body = json.dumps(out, indent=1, sort_keys=True)
    (out_dir / "envelope.json").write_text(body)
    (out_dir / "envelope.sha256").write_text(hashlib.sha256(body.encode()).hexdigest() + "\n")
    return out


def load(path: Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_DIR / "envelope.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: build it with twin.envelope.build() (train data only)")
    return json.loads(p.read_text())


# ----------------------------------------------------------------------------------------------- checks
def check_episode(feat: dict, part: str, env: dict | None = None, use: str = "family") -> list[dict]:
    """Violations of one episode's features against the real envelope of its family (or part)."""
    env = env or load()
    key = family_of(part) if use == "family" else part
    e = env["envelope"].get(key)
    if e is None:
        return [dict(feature="family", side="unknown", reason=f"no real envelope for {key!r}")]
    out = []
    for f in FEATURES:
        v = feat.get(f)
        r = e["features"].get(f, {})
        if v is None or not r.get("n"):
            continue
        if v > r["hi"]:
            out.append(dict(feature=f, value=v, side="above", lo=r["lo"], hi=r["hi"], real_max=r["max"],
                            reason=f"{f} {v} above the real {env['quantiles'][1]:g}th percentile {r['hi']} "
                                   f"(real successes of {key}, n={r['n']}; real max {r['max']})"))
        elif v < r["lo"]:
            out.append(dict(feature=f, value=v, side="below", lo=r["lo"], hi=r["hi"], real_min=r["min"],
                            reason=f"{f} {v} below the real {env['quantiles'][0]:g}th percentile {r['lo']} "
                                   f"(real successes of {key}, n={r['n']}; real min {r['min']})"))
    if feat.get("held") and feat.get("onset_s") is None:
        out.append(dict(feature="onset", side="none", reason="no contact onset: the insertion never pressed "
                                                             f"{CONTACT_N} N, unlike every real attempt with an onset"))
    return out


def check_routine(episodes: list[dict], part: str, env: dict | None = None, frac: float = 0.10) -> dict:
    """Routine-level verdict (E.envelope_violations counts routines). A routine violates if any episode's peak force
    exceeds the largest real peak of the family, or more than `frac` of its episodes fall outside the [lo, hi] range
    on the same feature. `episodes` = list of feature dicts (episode['features'] from the env)."""
    env = env or load()
    key = family_of(part)
    e = env["envelope"].get(key)
    per = [check_episode(f, part, env) for f in episodes]
    n = len(episodes)
    counts: dict[str, int] = {}
    for vs in per:
        for v in {(v["feature"], v["side"]) for v in vs}:
            counts[f"{v[0]}:{v[1]}"] = counts.get(f"{v[0]}:{v[1]}", 0) + 1
    reasons = [f"{k} in {c}/{n} episodes (> {frac:.0%})" for k, c in sorted(counts.items()) if n and c / n > frac]
    if e is not None:
        fmax = e.get("peak_force_any_max_N")
        hard = [f.get("peak_force_N") for f in episodes if f.get("peak_force_N") is not None and fmax is not None
                and f["peak_force_N"] > fmax]
        if hard:
            reasons.append(f"{len(hard)} episode(s) with peak force above the largest real peak {fmax} N "
                           f"(max {max(hard)} N)")
    return dict(part=part, family=key, n_episodes=n, violates=bool(reasons), reasons=reasons, counts=counts,
                per_episode=per)


def main():
    out = build()
    for k, v in out["envelope"].items():
        f = v["features"]
        print(f"{k:13s} n={v['n_success']:3d}/{v['n_success'] + v['n_failure']:3d}  "
              + "  ".join(f"{name}=[{f[name].get('lo')},{f[name].get('hi')}]" for name in FEATURES)
              + f"  Fmax_any={v['peak_force_any_max_N']}")


if __name__ == "__main__":
    main()
