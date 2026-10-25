"""fx-hot3d-err: twin placement error vs HOT3D ground truth (scoring only; GT never shown to Argon).
  .venv/bin/python -I playgrounds/footage/fx_hot3d_err.py <scene.bddl> [<states.json>]
Frame-free: compares pairwise distances (region centres; 3-D GT object centres) for the mapped objects.
Mapping by LIBERO category: tomato_sauce <- can_tomato_sauce, desk_caddy <- holder_gray (the small box),
milk <- carton_milk, orange_juice <- carton_oj, alphabet_soup <- can_parmesan. The can's goal region <- its GT end pose."""
import itertools, json, math, re, sys
from pathlib import Path
G = Path("/home/ran/arg_twin/data/hot3d-clips")
mi = json.loads((G / "object_models/models_info.json").read_text())
name = {k: v["name"] for k, v in mi.items()}
def gt(frame):
    d = json.loads((G / f"extracted/clip-002066/{frame:06d}.objects.json").read_text())
    return {name[k]: v[0]["T_world_from_object"]["translation_xyz"] for k, v in d.items()}
g0, g1 = gt(0), gt(149)
MAP = {"tomato_sauce": "can_tomato_sauce", "desk_caddy": "holder_gray", "milk": "carton_milk",
       "orange_juice": "carton_oj", "alphabet_soup": "can_parmesan"}
txt = Path(sys.argv[1]).read_text()
regions = {}
for m in re.finditer(r"\((\w+)\s*\(:target kitchen_table\)\s*\(:ranges \(\s*\(([-\d.\s]+)\)", txt):
    x0, y0, x1, y1 = map(float, m.group(2).split())
    regions[m.group(1)] = ((x0 + x1) / 2, (y0 + y1) / 2)
init = dict(re.findall(r"\(On (\w+?)_\d+ kitchen_table_(\w+)\)", txt.split("(:goal")[0]))
goal = dict(re.findall(r"\(On (\w+?)_\d+ kitchen_table_(\w+)\)", txt.split("(:goal")[1])) if "(:goal" in txt else {}
twin, ref = {}, {}
for cat, reg in init.items():
    if cat in MAP and reg in regions:
        twin[cat + "@start"] = regions[reg]; ref[cat + "@start"] = g0[MAP[cat]]
for cat, reg in goal.items():
    if cat in MAP and reg in regions:
        twin[cat + "@end"] = regions[reg]; ref[cat + "@end"] = g1[MAP[cat]]
rows = []
for a, b in itertools.combinations(sorted(twin), 2):
    dt, dg = math.dist(twin[a], twin[b]), math.dist(ref[a], ref[b])
    rows.append({"pair": f"{a}-{b}", "twin_cm": round(dt * 100, 1), "gt_cm": round(dg * 100, 1), "abs_err_cm": round(abs(dt - dg) * 100, 1)})
res = {"bddl": sys.argv[1], "mapped": sorted(twin), "unmapped_gt": sorted(set(g0) - {MAP[k.split('@')[0]] for k in twin}),
       "rows": rows, "mean_abs_err_cm": round(sum(r["abs_err_cm"] for r in rows) / len(rows), 1) if rows else None}
print(json.dumps(res))
