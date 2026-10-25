"""Digital cousins: randomized variants of the twin for E.cousins_success, with ranges from measured uncertainty.

Solver exploits are narrow in parameter space while real strategies are robust [acdc-2024, forge-2025]; a routine is
promoted only if it survives the cousins with zero brain calls (validation.yaml E.cousins_success). A cousin is a
physics variant of the twin. E.cousins_success runs a routine on frozen evaluation pairs (held-out start j, cousin j):
the board pose and the in-hand pose vary through the paired held-out start, which draws them from the measured
distributions (twin/starts.py: train leave-one-out board errors, train Remove-grasp in-hand poses), and the physics
varies through the cousin. Keeping geometry in the start (not re-drawn per cousin) means every geometric draw was
checked once for a physically valid grasp, and no draw is counted twice. Cousin c000 is the nominal twin.

| parameter | range | source |
|---|---|---|
| board pose | via the paired held-out start: train leave-one-out across-pad errors; z sigma | twin/starts.py |
| in-hand pose | via the paired held-out start: one TRAIN Remove grasp (dx, dy, pivot, rock) + grasp prior | twin/starts.py |
| stiffness K | 200-800 N/m uniform; K_rot through REASSEMBLE's pairs (200,10) (500,50) (800,75) Nm/rad | validation.yaml, compliance_param.cfg |
| part-board friction | 0.1-0.5 uniform | PLACEHOLDER (see below) |
| pad friction | 0.6-1.2 uniform | PLACEHOLDER |
| contact softness | solref timeconst 2-8 ms, solimp width 0.25-1.0 mm | PLACEHOLDER |
| timestep | 1 ms, 0.5 ms (stratified) | rules.md dt/2 |
| solver | Newton, CG (stratified) | validation.yaml |

PLACEHOLDER: the replay agent's frozen params (runs/claude/run-standin-1/replay/params.json, provenance
"pad_friction/pad_torsion/friction/solref/solimp") say these are NOT identifiable from the replays yet, so no range
"covering the real force spread" exists. The placeholder brackets the B1 values (friction 0.3 = twin/parts.py
0.30 +- 2 sigma 0.10; pad friction 1.0; timeconst 4 ms and width 0.5 mm, each x/2 and x2), within QBIT's randomized
ranges that did cover a real spread (friction 0.1-0.7, solref 0.01-0.5 in its own units) [qbit-2025]. Replace it when
replay-validate reports a force-spread-covering range, and refreeze in a NEW run directory.

    .venv/bin/python -c "from twin.cousins import main; main()"   # writes runs/claude/run-standin-1/cousins/
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from twin import REPO_DIR

OUT_DIR = REPO_DIR / "runs" / "claude" / "run-standin-1" / "cousins"
SEED = 20261024_4
N_PER_COMBO = 8                     # 4 (dt, solver) combos x 8 = 32 cousins (>= 29 for a >= 0.9 claim, 0 failures)
TIMESTEPS = (0.001, 0.0005)
SOLVERS = ("newton", "cg")
K_PAIRS = ((200.0, 10.0), (500.0, 50.0), (800.0, 75.0))  # REASSEMBLE compliance_param.cfg (current + commented)
RANGES = dict(
    k_trans=dict(lo=200.0, hi=800.0, unit="N/m", source="validation.yaml E.cousins_success; data notes K 200-800",
                 placeholder=False),
    friction=dict(lo=0.10, hi=0.50, unit="-", source="PLACEHOLDER: twin/parts.py friction 0.30 +- 2 sigma (0.10); "
                  "replay params.json: not identifiable yet", placeholder=True),
    pad_friction=dict(lo=0.6, hi=1.2, unit="-", source="PLACEHOLDER: replay pad_friction 1.0 (scan 0.2-1.0 seated "
                      "2-4/48 everywhere: not identifiable)", placeholder=True),
    solref_tc=dict(lo=0.002, hi=0.008, unit="s", source="PLACEHOLDER: B1 timeconst 4 ms x/2 and x2 (>= 2 dt at "
                   "dt <= 1 ms, MuJoCo docs)", placeholder=True),
    solimp_width=dict(lo=0.00025, hi=0.0010, unit="m", source="PLACEHOLDER: B1 solimp width 0.5 mm x/2 and x2",
                      placeholder=True),
)


def k_rot_for(k: float) -> float:
    ks, kr = zip(*K_PAIRS)
    return float(np.interp(k, ks, kr))


def _lhs(rng, n: int, d: int) -> np.ndarray:
    """Latin hypercube in [0, 1)^d."""
    u = (rng.random((n, d)) + np.arange(n)[:, None]) / n
    for j in range(d):
        u[:, j] = u[rng.permutation(n), j]
    return u


def nominal() -> dict:
    """c000: the nominal twin (twin/replay.py ReplayParams defaults), geometry from the start."""
    from twin.replay import ReplayParams
    rp = ReplayParams()
    return dict(id="c000", nominal=True, timestep=rp.timestep, solver="newton", k_trans=rp.k_trans, k_rot=rp.k_rot,
                friction=rp.friction, pad_friction=rp.pad_friction, solref_tc=rp.solref_tc,
                solimp_width=float(rp.solimp[2]))


def make(n_per_combo: int = N_PER_COMBO, seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = [nominal()]
    keys = list(RANGES)
    i = 1
    for dt in TIMESTEPS:
        for solver in SOLVERS:
            u = _lhs(rng, n_per_combo, len(keys))
            for row in u:
                c = dict(id=f"c{i:03d}", nominal=False, timestep=dt, solver=solver)
                for k, x in zip(keys, row):
                    r = RANGES[k]
                    c[k] = float(r["lo"] + x * (r["hi"] - r["lo"]))
                c["k_rot"] = k_rot_for(c["k_trans"])
                out.append(c)
                i += 1
    return out


def apply(start: dict, cousin: dict | None) -> dict:
    """Episode = the start's geometry + the cousin's physics (see module doc)."""
    c = cousin or nominal()
    ep = dict(start)
    ep["cousin"] = c["id"]
    ep["physics"] = {k: c[k] for k in ("timestep", "solver", "k_trans", "k_rot", "friction", "pad_friction",
                                       "solref_tc", "solimp_width")}
    return ep


def pairs(cousins: list[dict], heldout_ids_by_part: dict[str, list[str]]) -> dict[str, list[list[str]]]:
    """Frozen evaluation pairs for E.cousins_success: cousin j (j >= 1) with the part's j-th held-out start."""
    rand = [c["id"] for c in cousins if not c["nominal"]]
    return {p: [[ids[j % len(ids)], cid] for j, cid in enumerate(rand)] for p, ids in heldout_ids_by_part.items()}


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze(out_dir: Path = OUT_DIR, starts_dir: Path | None = None) -> dict:
    from twin.starts import OUT_DIR as STARTS_DIR, load as load_starts
    out_dir = Path(out_dir)
    man_f = out_dir / "manifest.json"
    if man_f.exists():
        raise FileExistsError(f"{man_f} exists: cousins are frozen. A change needs a new run directory.")
    out_dir.mkdir(parents=True, exist_ok=True)
    starts_dir = Path(starts_dir) if starts_dir else STARTS_DIR
    cs = make()
    held, _ = load_starts("heldout", starts_dir)
    by_part: dict[str, list[str]] = {}
    for sid, s in sorted(held.items()):
        by_part.setdefault(s["part"], []).append(sid)
    pr = pairs(cs, by_part)
    f = out_dir / "cousins.json"
    f.write_text(json.dumps(dict(cousins=cs, eval_pairs=pr, ranges=RANGES, k_pairs=K_PAIRS, seed=SEED), indent=0))
    man = dict(frozen_at=time.strftime("%Y-%m-%d %H:%M"), frozen_by="twin/cousins.py freeze (sim-integrity stand-in)",
               file=f.name, sha256=sha256_file(f), n=len(cs), seed=SEED,
               combos=[f"dt={dt} {s}" for dt in TIMESTEPS for s in SOLVERS], n_per_combo=N_PER_COMBO,
               placeholders=[k for k, r in RANGES.items() if r["placeholder"]],
               eval_pairs_per_part=len(next(iter(pr.values()))) if pr else 0,
               starts_manifest_sha256=sha256_file(starts_dir / "manifest.json"),
               generator=dict(file="twin/cousins.py", sha256=sha256_file(Path(__file__))))
    man_f.write_text(json.dumps(man, indent=1))
    (out_dir / "manifest.sha256").write_text(sha256_file(man_f) + "\n")
    return man


def load(out_dir: Path = OUT_DIR) -> dict:
    out_dir = Path(out_dir)
    man = json.loads((out_dir / "manifest.json").read_text())
    if sha256_file(out_dir / "manifest.json") != (out_dir / "manifest.sha256").read_text().split()[0]:
        raise RuntimeError("cousins manifest.json does not match manifest.sha256")
    f = out_dir / man["file"]
    if sha256_file(f) != man["sha256"]:
        raise RuntimeError(f"{man['file']} does not match its frozen sha256")
    body = json.loads(f.read_text())
    body["by_id"] = {c["id"]: c for c in body["cousins"]}
    return body


def main():
    man = freeze()
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
