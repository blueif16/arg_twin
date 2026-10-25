"""fx-parse: offline BDDL check (no sim). .venv-libero, python -I. Prints JSON {ok, error, objects, fixtures, goal}."""
import json, os, sys
os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/ran/arg_twin/runs/libero/config")
os.environ["MUJOCO_GL"] = "glx"
res = {"ok": False}
try:
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    from libero.libero.envs.objects import OBJECTS_DICT
    import libero.libero.envs.bddl_base_domain as bd
    from libero.libero.envs.problems import *  # noqa  registers problem classes
    p = robosuite_parse_problem(sys.argv[1])
    res.update(objects=p["objects"], fixtures=p["fixtures"], goal=p["goal_state"], init=p["initial_state"],
               problem=p["problem_name"], language=" ".join(p["language_instruction"]))
    errs = []
    if p["problem_name"].lower() not in bd.TASK_MAPPING:
        errs.append(f"problem name {p['problem_name']} not in {sorted(bd.TASK_MAPPING)}")
    for cat in list(p["objects"]) + list(p["fixtures"]):
        if cat not in OBJECTS_DICT and cat not in ("kitchen_table", "living_room_table", "study_table", "table"):
            errs.append(f"unknown category {cat}")
    regions = set(p["regions"])
    names = {n for v in list(p["objects"].values()) + list(p["fixtures"].values()) for n in v}
    for st in p["initial_state"] + p["goal_state"]:
        for a in st[1:]:
            if a in names or a in regions:
                continue
            if not any(a == f"{r_t}_{r}" for r, rv in p["regions"].items() for r_t in [rv["target"]]):
                errs.append(f"unknown name {a} in {st}")
    res["ok"] = not errs
    res["error"] = "; ".join(errs) or None
except Exception as e:
    res["error"] = f"{type(e).__name__}: {e}"
print(json.dumps(res, default=str))
