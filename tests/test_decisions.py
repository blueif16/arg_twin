"""Decision-log tests (harness/tools/decisions.py). Plain asserts, runnable without pytest:

    .venv/bin/python -m tests.test_decisions

Records go to a temporary runs directory, never to the real run logs.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from harness.registry import load_all

load_all()
from harness.tools import decisions as dec  # noqa: E402

EVID = "playbook/skills/world-frame/SKILL.md"


def _good(**over) -> dict:
    a = {"brain": "claude", "run_id": "test-run", "stage": "B0", "step": "robot_tool", "quantity": "tool length",
         "saw": ["runs/world/ft_stack_hama1.png"],
         "native": {"source": "prior", "estimate": "pads at the configured O_T_EE, offset 0 mm"},
         "tool": {"name": "calibrate_static_cameras"},
         "result": {"summary": "shared tool offset -22.9 mm", "values": {"offset_mm": -22.9}},
         "error": {"metric": "hand position along its z axis", "unit": "mm", "before": 22.9, "after": 1.5,
                   "reference": "bundle adjustment on arm caps"},
         "verdict": "adopt", "decision": "use the fitted offset", "evidence": [EVID]}
    a.update(over)
    return a


def _expect_error(args: dict, fragment: str):
    try:
        dec.log_decision(**args)
    except ValueError as e:
        assert fragment in str(e), (fragment, str(e))
        return
    raise AssertionError(f"expected ValueError containing {fragment!r}")


def main():
    old = dec.RUNS_DIR
    dec.RUNS_DIR = Path(tempfile.mkdtemp(prefix="decisions-"))
    try:
        r = dec.log_decision(**_good(id="tool-length"))
        assert r["ok"] and r["seq"] == 1, r
        p = dec.RUNS_DIR / "claude" / "test-run" / "decisions.jsonl"
        rec = json.loads(p.read_text().splitlines()[0])
        assert rec["schema"] == "decision/v1" and rec["tool"]["registered"] is True, rec
        _expect_error(_good(id="tool-length"), "already logged")
        _expect_error(_good(native={"source": "native", "estimate": "about right"}), "needs numbers")
        _expect_error(_good(error={"metric": "x", "unit": "furlong", "before": 1, "after": 0, "reference": "y"}),
                      "error.unit")
        _expect_error(_good(error={"metric": "x", "unit": "mm", "before": None, "after": None, "reference": "y"}),
                      "before or after")
        _expect_error(_good(evidence=["runs/does/not/exist.json"]), "existing repo file")
        _expect_error(_good(stage="Z9"), "unknown stage")
        _expect_error(_good(step="vibes"), "step must be")
        _expect_error(_good(check_ids=["B0.nope"]), "unknown check ids")
        r2 = dec.log_decision(**_good(step="cameras", quantity="hama1 pose", tool=None,
                                      error={"metric": "held-out pad reprojection", "unit": "px", "before": 5.35,
                                             "after": 1.63, "reference": "watcher marks", "mm_per_px": 2.19}))
        assert r2["seq"] == 2 and "warnings" not in r2, r2
        r3 = dec.log_decision(**_good(step="validation", quantity="negative control",
                                      error={"metric": "bacc drop", "unit": "fraction", "before": -0.03,
                                             "after": None, "reference": "permissive twin"}))
        assert any("not physical" in w for w in r3["warnings"]), r3
        r4 = dec.log_decision(**_good(step="intake", quantity="inventory", tool=None, error=None,
                                      native={"source": "none", "estimate": ""}))
        assert r4["seq"] == 4 and "warnings" not in r4, r4
        _expect_error(_good(error=None), "error must be an object")
        pre = _good(step="cameras", quantity="hama2 pose", tool=None, error=None, verdict="open",
                    native={"source": "native", "estimate": "look-at (1.40, 0.35, 0.60) m"},
                    result={"summary": "pending: calibrate_static_cameras"}, decision="pre-registration")
        r5 = dec.log_decision(**pre)
        assert r5["seq"] == 5 and r5["change"] is None, r5
        s = dec.decision_summary("claude", "test-run", chart=True)
        assert s["n"] == 5 and s["by_step"] == {"robot_tool": 1, "cameras": 2, "validation": 1, "intake": 1}, s
        assert s["chart_rows"] == 2 and (dec.REPO / s["chart"]).exists() or Path(s["chart"]).exists(), s
        row = next(t for t in s["table"] if t["step"] == "cameras")
        assert row["unit"] == "mm" and abs(row["before"] - 5.35 * 2.19) < 1e-9, row
        print("decisions: all tests passed")
    finally:
        dec.RUNS_DIR = old


if __name__ == "__main__":
    main()
