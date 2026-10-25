"""Estimate a Claude stand-in's tokens per practice episode and write runs/<brain>/<run_id>/practice/tokens.json.

A stand-in cannot read its own usage, so this is an ESTIMATE, labelled as such (playbook/skills/practice §8):
  input per decision  = the tool results it read (JSON chars / 4) + images (Claude: width x height / 750 tokens)
  output per decision = OUT_PER_DECISION (visible reasoning + tool call text; thinking is not separable here)
  chained re-read     = what a fresh chained brain (Argon) would re-read: a PREFIX_TOKENS system/tool/skill prefix
                        plus the episode's growing history on every turn (sum over turns), reported separately.
Decisions come from practice_ledger (calls chained within 1 s are one turn).

    .venv/bin/python scripts/standin_tokens.py claude run-standin-practice-1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_PER_DECISION = 600      # tokens: a few sentences of reasoning plus the call; assumption, not measured
PREFIX_TOKENS = 25_000      # system instruction + 18 tool declarations + the practice skill (harness/tools/practice.py)


def img_tokens(path: Path) -> int:
    from PIL import Image
    try:
        w, h = Image.open(path).size
    except Exception:  # noqa: BLE001
        return 0
    return int(w * h / 750)


def main(brain: str, run_id: str) -> dict:
    sys.path.insert(0, str(REPO))
    from harness.tools.practice import _calls_for, practice_ledger
    rd = REPO / "runs" / brain / run_id
    led = practice_ledger(run_id, brain)
    rows = {r["start_id"]: r for r in json.loads((rd / "practice" / "ledger.json").read_text())["episodes"]}
    calls = _calls_for({r["session"] for r in rows.values()})
    out = {}
    for sid, r in rows.items():
        mine = [c for c in calls if (c.get("args") or {}).get("session") == r["session"]]
        # this episode's window: from its reset to its finish
        i0 = next(i for i, c in enumerate(mine) if c["tool"] == "env_reset" and c["args"].get("start_id") == sid)
        i1 = next((i for i, c in enumerate(mine) if i > i0 and c["tool"] == "env_finish"), len(mine) - 1)
        win = mine[i0:i1 + 1]
        per_turn = []
        for c in win:
            t = len(json.dumps(c.get("result"), default=str)) // 4
            img = (c.get("result") or {}).get("image")
            if img:
                t += img_tokens(REPO / img)
            per_turn.append(t)
        n_dec = r["decisions"] or len(win)
        inp = sum(per_turn)
        outp = OUT_PER_DECISION * n_dec
        # a fresh chained brain re-reads prefix + history each turn
        hist, reread = 0, 0
        step = (inp + outp) / max(n_dec, 1)
        for _ in range(n_dec):
            reread += PREFIX_TOKENS + hist
            hist += step
        out[sid] = dict(tokens=int(inp + outp), decisions=n_dec, source="estimate (stand-in): results/4 + images "
                        f"w*h/750 + {OUT_PER_DECISION}/decision output; not measured",
                        input_new=int(inp), output=int(outp), chained_reread_estimate=int(reread + outp),
                        images=sum(1 for c in win if (c.get("result") or {}).get("image")), watcher_tokens=0)
    f = rd / "practice" / "tokens.json"
    f.write_text(json.dumps(out, indent=1))
    return dict(file=str(f.relative_to(REPO)), episodes=out, ledger_aggregates=led["aggregates"])


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], sys.argv[2]), indent=1, default=str))
