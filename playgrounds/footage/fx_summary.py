"""fx-summary: collect the footage front-end run into runs/footage/summary.json (python -I)."""
import json
from pathlib import Path
R = Path("/home/ran/arg_twin/runs/footage")
def j(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None
out = {}
for s in ("put-bowl-in-microwave", "hot3d-clip-002066", "pov-setting-the-table"):
    rd, wr, b = j(R / "argon_read" / s / "read.json"), j(R / "scenes" / s / "write.json"), j(R / "scenes" / s / "build/build.json")
    rw, g, gs, pe = j(R / "scenes" / s / "reward.json"), j(R / "scenes" / s / "gate.json"), j(R / "scenes" / s / "gate_strict.json"), j(R / "scenes" / s / "placement_err.json")
    op, opp, opg, ope = j(R / "oneprompt" / s / "call.json"), j(R / "oneprompt" / s / "parse.json"), j(R / "oneprompt" / s / "gate_on_our_states.json"), j(R / "oneprompt" / s / "placement_err.json")
    ours_usd = sum(x for x in [rd and rd["total_usd"], wr and wr["usd"], rw and rw["usage"]["usd"]] if x)
    out[s] = {"read": rd and {"task": rd.get("final", {}).get("task"), "twin": rd.get("final", {}).get("twin_supported"),
                              "usd": rd["total_usd"], "latency_s": rd["total_latency_s"]},
              "bddl": wr and {"parse_ok": wr["parse_ok"], "repairs": len(wr["calls"]) - 1, "usd": wr["usd"],
                              "goal": (wr.get("parse_v1") or wr.get("parse_v0") or {}).get("goal"),
                              "objects": (wr.get("parse_v1") or wr.get("parse_v0") or {}).get("objects")},
              "build": b and {"libero_check": b["libero_check"], "timing": b["timing"], "total_s": b["total_s"]},
              "reward": rw and {"usd": rw["usage"]["usd"], "latency_s": rw["latency_s"]},
              "gate": g and {"agree": g.get("agree"), "history": g.get("history"), "error": g.get("error"),
                             "rows": [(r["state"], r["libero"], r["reward"]) for r in g.get("rows", [])]},
              "gate_strict": gs and gs.get("agree"),
              "placement_err": pe and {"mean_abs_err_cm": pe["mean_abs_err_cm"], "rows": pe["rows"]},
              "ours_usd": round(ours_usd, 4),
              "oneprompt": op and {"usd": op["usage"]["usd"], "latency_s": op["latency_s"], "parse_ok": opp and opp.get("ok"),
                                   "goal": opp and opp.get("goal"), "objects": opp and opp.get("objects"),
                                   "gate": opg and {"agree": opg.get("agree"), "rows": [(r["state"], r["libero"], r["reward"], r["error"]) for r in opg.get("rows", [])]},
                                   "placement_err": ope and ope["mean_abs_err_cm"]}}
(R / "summary.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
