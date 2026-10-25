"""fx-build: build a custom-BDDL LIBERO scene, render it, and save gate states with LIBERO's check value.
  .venv-libero/bin/python -I playgrounds/footage/fx_build.py <scene.bddl> <out_dir>
States (sim truth; used only by the no-model-call gate, never shown to Argon):
  reset            settled start (negative)
  goal             every goal predicate made true by a privileged teleport / joint set, then 40 settle steps
  nm_offset        the moved object 0.10 m toward the camera from its goal spot, articulations as at reset
  nm_hover         (On goals) the object 0.10 m above its goal spot, articulations as goal, not settled
  nm_door_open     (Close goals) object(s) as goal, the articulation left open
"""
import json, os, sys, time
from pathlib import Path
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k] = "1"
try:
    import setproctitle; setproctitle.setproctitle("fx-build")
except Exception:
    pass
sys.path.insert(0, "/home/ran/arg_twin/playgrounds/footage")
t0 = time.time(); T = {}
import numpy as np
from PIL import Image
from fx_lstate import TaskEnv
bddl, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
t = time.time(); te = TaskEnv(str(bddl)); T["build_reset_s"] = round(time.time() - t, 1)
env, e = te.env, te.e
NOOP = [0.0] * 6 + [-1.0]
def settle(n):
    for _ in range(n):
        env.step(NOOP)
settle(10)
reset_flat = te.flat()
Image.fromarray(te.render(size=512)).save(out / "agentview.png")
goals = [list(g) for g in e.parsed_problem["goal_state"]]
site_names = {te.mj.mj_id2name(te.m, te.mj.mjtObj.mjOBJ_SITE, i) for i in range(te.m.nsite)}

def fixture_of(name):
    for f in te.fixtures:
        if name == f or name.startswith(f + "_"):
            return f
    return None

def set_art(fx, region, mode):
    """mode: 'close' | 'open'. Region-specific joint if the fixture has several (cabinet drawers)."""
    js = te.fixture_joints.get(fx, [])
    if region and len(js) > 1:
        key = region.replace(fx + "_", "").replace("_region", "")
        js = [j for j in js if key in j[0]] or js
    for j, adr, kind, rg in js:
        lo, hi = rg
        if mode == "close":
            te.d.qpos[adr] = 0.0 if lo <= 0.0 <= hi else (hi if abs(hi) < abs(lo) else lo)
        else:
            te.d.qpos[adr] = lo if abs(lo) > abs(hi) else hi
    te.sim.forward()

def goal_spot(a, b):
    if b in site_names:
        p = te.d.site_xpos[te.mj.mj_name2id(te.m, te.mj.mjtObj.mjOBJ_SITE, b)].copy()
        if fixture_of(b):
            return p, "in"
        p[2] = te.d.xpos[te.body_id[a]][2] + 0.005   # a table region: keep the object's resting height
        return p, "on_region"
    lo, hi = te.aabb(b)
    alo, _ = te.aabb(a)
    p = te.d.xpos[te.body_id[b]].copy()
    p[2] = hi[2] + (te.d.xpos[te.body_id[a]][2] - alo[2]) + 0.005
    return p, "on_body"

def make(variant):
    te.load(reset_flat)
    moved = []
    arts = []
    for g in goals:
        op = g[0].lower()
        if op in ("on", "in") and g[1] in te.objects:
            moved.append((g[1], g[2]))
        elif op in ("open", "close"):
            arts.append((op, g[1]))
    if variant == "nm_door_open" and not any(op == "close" for op, _ in arts):
        return None
    if variant == "nm_hover" and not any(goal_spot(a, b)[1] in ("on_body", "on_region") for a, b in moved):
        return None
    if variant in ("goal", "nm_hover", "nm_door_open"):
        for op, x in arts:
            fx = fixture_of(x)
            if fx:
                set_art(fx, x if x != fx else None, "open" if (variant == "nm_door_open" and op == "close") else op)
    for a, b in moved:
        p, kind = goal_spot(a, b)
        if kind == "in":
            p = p - np.array([0, 0, 0.04])
        if variant == "nm_offset":
            p = p + np.array([0.10, 0, 0.0])
            p[2] = max(p[2], te.d.xpos[te.body_id[a]][2])
        if variant == "nm_hover":
            p = p + np.array([0, 0, 0.10])
        te.set_qpos_free(a, pos=p)
    if variant == "nm_hover":
        te.sim.forward()
    else:
        settle(40)
    return {"state": te.api_state(50, -1.0), "libero": te.truth(), "flat": te.flat().tolist()}

res = {"bddl": str(bddl), "goals": goals, "objects": te.objects, "fixtures": te.fixtures}
te.load(reset_flat)
states = {"reset": {"state": te.api_state(0, -1.0), "libero": te.truth(), "flat": reset_flat.tolist()}}
t = time.time()
for v in ("goal", "nm_offset", "nm_hover", "nm_door_open"):
    try:
        s = make(v)
    except Exception as ex:
        res.setdefault("variant_errors", {})[v] = repr(ex); s = None
    if s is not None:
        states[v] = s
        if v == "goal":
            Image.fromarray(te.render(size=512)).save(out / "agentview_goal.png")
T["states_s"] = round(time.time() - t, 1)
res["libero_check"] = {k: v["libero"] for k, v in states.items()}
res["timing"] = T; res["total_s"] = round(time.time() - t0, 1)
(out / "states.json").write_text(json.dumps({k: {"state": v["state"], "libero": v["libero"]} for k, v in states.items()}))
np.savez_compressed(out / "flat_states.npz", **{k: np.asarray(v["flat"]) for k, v in states.items()})
(out / "build.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
te.close()
