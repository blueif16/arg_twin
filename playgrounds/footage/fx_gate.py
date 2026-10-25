"""fx-gate: no-model-call gate. Runs an Argon-written success(s, history) on the saved scene states and compares it with
LIBERO's own check on the same states. .venv/bin/python -I playgrounds/footage/fx_gate.py <reward.py> <states.json> [name]"""
import json, math, sys
import numpy as np
code_p, states_p = sys.argv[1], sys.argv[2]
states = json.loads(open(states_p).read())
ns = {"math": math, "np": np, "numpy": np, "__builtins__": __builtins__}
res = {"reward": code_p, "states": states_p, "rows": []}
try:
    exec(compile(open(code_p).read(), code_p, "exec"), ns)
    fn = ns["success"]
except Exception as e:
    res["error"] = f"load: {e!r}"; print(json.dumps(res)); sys.exit(0)
reset = states["reset"]["state"]
# A real episode that ends in a closed-door goal passed through an open-door state; give it as history when it exists
# (states["nm_door_open"] = objects placed, door open). The reset state alone is the strict variant (--strict).
strict = "--strict" in sys.argv
mid = [] if strict or "nm_door_open" not in states else [states["nm_door_open"]["state"]]
res["history"] = "reset only" if not mid else "reset, then the open-door state"
for k, v in states.items():
    hist = [] if k == "reset" else ([reset] if k == "nm_door_open" else [reset] + mid)
    try:
        r = bool(fn(v["state"], hist)); err = None
    except Exception as e:
        r, err = None, repr(e)[:200]
    res["rows"].append({"state": k, "libero": v["libero"], "reward": r, "agree": r == v["libero"], "error": err})
n = len(res["rows"]); a = sum(x["agree"] for x in res["rows"])
res["agree"] = f"{a}/{n}"
res["pass"] = a == n
print(json.dumps(res))
