"""E gate hand-back rule (validation.yaml v17 E.envelope_violations). Plain asserts, runnable without pytest:

    .venv/bin/python -m tests.test_e_handback

A handed-back episode still has the routine's own segment, up to the hand-back, checked against the real envelope;
only the brain's actions after the hand-back are excluded. Uses the frozen REASSEMBLE envelope read-only
(runs/claude/run-standin-1/envelope/envelope.json); writes nothing.
"""

from __future__ import annotations

import numpy as np

from twin import REPO_DIR

ENV_PATH = REPO_DIR / "runs" / "claude" / "run-standin-1" / "envelope" / "envelope.json"
PART = "round peg 4"


def _env() -> dict:
    from twin.envelope import load
    return load(ENV_PATH)


def _inside(env: dict) -> dict:
    """Features in the middle of every real range of the part's family."""
    from twin.envelope import FEATURES, family_of
    r = env["envelope"][family_of(PART)]["features"]
    out = {f: round((r[f]["lo"] + r[f]["hi"]) / 2, 3) for f in FEATURES}
    return out | dict(held=True, released=True, onset_s=5.0)


def _row(start: str, code: str, feats: dict, **extra) -> dict:
    return dict(start_id=start, outcome="success" if code == "OK" else "failure", counted=code == "OK", flags=[],
                routine=dict(code=code), envelope=dict(features=feats), **extra)


def test_handback_segment_counts():
    """A gate hand-back whose own segment pressed harder than any real attempt makes the routine violate (v16 let it
    escape, as the 17.3 N pilot segment did)."""
    from harness.tools.compile import _summ
    env = _env()
    fmax = env["envelope"]["round_peg"]["peak_force_any_max_N"]
    ok = _inside(env)
    hard = ok | dict(peak_force_N=round(fmax + 3.3, 3))
    rows = [_row(f"H-{i}", "OK", ok) for i in range(9)] + [_row("H-9", "JAMMED", hard)]
    s = _summ(rows, PART, env)
    assert s["envelope"]["violates"], s["envelope"]
    assert s["envelope"]["n_episodes"] == 10 and s["envelope"]["n_handed_back"] == 1
    assert any("largest real peak" in x for x in s["envelope"]["reasons"]), s["envelope"]["reasons"]
    assert s["envelope_handbacks"]["violates"] and s["envelope_handbacks"]["n_episodes"] == 1
    clean = _summ(rows[:9] + [_row("H-9", "JAMMED", ok)], PART, env)
    assert not clean["envelope"]["violates"], clean["envelope"]


def test_brain_actions_after_handback_excluded():
    """When a brain acted after the hand-back, only the routine's segment (envelope.routine_features) is judged; a
    record without it is reported as unsegmented, not passed or failed on the brain's actions."""
    from harness.tools.compile import _summ
    env = _env()
    fmax = env["envelope"]["round_peg"]["peak_force_any_max_N"]
    ok = _inside(env)
    brain_hard = ok | dict(peak_force_N=round(fmax + 5, 3))   # whole episode: the brain pressed too hard later
    seg = _row("H-9", "NOT_FOUND", brain_hard, brain_after_handback=True)
    seg["envelope"]["routine_features"] = ok
    rows = [_row(f"H-{i}", "OK", ok) for i in range(9)] + [seg]
    s = _summ(rows, PART, env)
    assert not s["envelope"]["violates"], s["envelope"]
    assert s["envelope"]["n_episodes"] == 10 and s["envelope"]["unsegmented"] == []
    noseg = _row("H-9", "NOT_FOUND", brain_hard, brain_after_handback=True)
    s2 = _summ(rows[:9] + [noseg], PART, env)
    assert not s2["envelope"]["violates"] and s2["envelope"]["n_episodes"] == 9
    assert s2["envelope"]["unsegmented"] == ["H-9"], s2["envelope"]


def test_features_until_cuts_at_handback():
    """features_until ignores samples after the hand-back: a force spike the brain causes later is not the routine's."""
    from twin.envelope import features, features_until
    n = 300                                     # 10 s at 30 Hz
    t = np.arange(n) / 30.0
    pos = np.c_[np.zeros(n), np.zeros(n), 0.2 - 0.002 * np.arange(n) / 30.0]
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    F = np.zeros((n, 3))
    F[60:150, 2] = 6.0                          # the routine's contact: 6 N from 2 s to 5 s
    F[200:240, 2] = 30.0                        # the brain's push after the hand-back at 6 s
    grip = np.full(n, 0.03)                     # held throughout
    whole = features(t, pos, quat, F, grip)
    seg = features_until(t, pos, quat, F, grip, t_end=6.0)
    assert whole["peak_force_N"] >= 29.9, whole
    assert abs(seg["peak_force_N"] - 6.0) < 1e-6 and seg["segment_end_s"] == 6.0, seg
    assert features_until(t, pos, quat, F, grip, None)["peak_force_N"] == whole["peak_force_N"]


def main() -> int:
    tests = [test_handback_segment_counts, test_brain_actions_after_handback_excluded, test_features_until_cuts_at_handback]
    bad = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as ex:
            bad += 1
            print(f"FAIL {fn.__name__}: {ex}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
