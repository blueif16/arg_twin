"""fx-bddl-load: build a LIBERO scene from a custom BDDL, render agentview, step, and evaluate the goal.

Runs under .venv-libero. Usage:
  MUJOCO_GL=glx .venv-libero/bin/python -I playgrounds/footage/fx_bddl_load.py BDDL OUT_DIR [--mode custom|override]
--mode custom: load BDDL as is (placements sampled from its regions).
--mode override: BDDL is an existing LIBERO-90 file; after reset, the goal object's free joint is moved to OBJ_XY.
After the honest check, a privileged 'goal teleport' places the first obj_of_interest on the second to prove the
predicate can turn True (test only; never an actor op).
"""
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/ran/arg_twin/runs/libero/config")
os.environ["MUJOCO_GL"] = os.environ.get("LIBERO_MUJOCO_GL", "glx")
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k] = "1"
try:
    import setproctitle; setproctitle.setproctitle("fx-bddl-load")
except Exception:
    pass
import numpy as np
from PIL import Image

t0 = time.time(); T = {}
bddl, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "custom"
from libero.libero.envs import OffScreenRenderEnv
T["import_s"] = round(time.time() - t0, 1)

t = time.time()
env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
T["build_s"] = round(time.time() - t, 1)
t = time.time(); env.seed(142); obs = env.reset(); T["reset_s"] = round(time.time() - t, 1)
inner = env.env
names = list(inner.obj_of_interest)
if mode == "override":
    xy = json.loads(os.environ.get("OBJ_XY", "[0.08, -0.18]"))
    obj = inner.objects_dict[names[0]]
    jn = obj.joints[-1]
    q = inner.sim.data.get_joint_qpos(jn).copy(); q[0], q[1] = xy[0], xy[1]; q[2] += 0.02
    inner.sim.data.set_joint_qpos(jn, q); inner.sim.forward()
t = time.time()
for _ in range(10):
    obs, r, done, info = env.step([0.0] * 6 + [-1.0])
T["10_steps_s"] = round(time.time() - t, 2)
img = obs["agentview_image"][::-1]  # true view (rows bottom-up in raw)
Image.fromarray(img).save(out / "agentview_reset.png")
def poses():
    return {n: [round(float(v), 4) for v in inner.sim.data.body_xpos[inner.obj_body_id[n]]] for n in list(inner.objects_dict) + list(getattr(inner, "fixtures_dict", {}))}
res = {"bddl": str(bddl), "mode": mode, "obj_of_interest": names, "poses_after_settle": poses(),
       "success_after_settle": bool(env.check_success()), "joint_names": [j for j in inner.sim.model.joint_names if "microwave" in j or "cabinet" in j],
       "init_state_dim": int(np.asarray(env.get_sim_state()).size)}
np.save(out / "init_state_0.npy", np.asarray(env.get_sim_state()))
# privileged goal teleport (test of the predicate only)
a, b = names[0], names[1]
ja = inner.objects_dict[a].joints[-1]
site = os.environ.get("FX_TARGET_SITE")
pb = (inner.sim.data.site_xpos[inner.sim.model.site_name2id(site)].copy() - [0, 0, 0.04]) if site else inner.sim.data.body_xpos[inner.obj_body_id[b]].copy()
q = inner.sim.data.get_joint_qpos(ja).copy(); q[0], q[1], q[2] = pb[0], pb[1], pb[2] + 0.06
inner.sim.data.set_joint_qpos(ja, q)
cj = os.environ.get("FX_CLOSE_JOINT_OF")  # fixture whose (non-free) articulation joint is set to 0 = closed
if cj:
    fx = inner.get_object(cj) if hasattr(inner, "get_object") else inner.fixtures_dict[cj]
    jn = [j for j in inner.sim.model.joint_names if j.startswith(cj) and "joint0" not in j]
    res["closed_joints"] = jn
    for j in jn:
        inner.sim.data.set_joint_qpos(j, 0.0)
inner.sim.forward()
def goal_preds():
    try:
        return [[str(g), bool(inner._eval_predicate(g))] for g in inner.parsed_problem["goal_state"]]
    except Exception as e:
        return repr(e)
res["goal_preds_before_steps"] = goal_preds()
t = time.time(); trace = []
for i in range(40):
    obs, r, done, info = env.step([0.0] * 6 + [-1.0]); trace.append(bool(env.check_success()))
T["40_steps_s"] = round(time.time() - t, 2)
Image.fromarray(obs["agentview_image"][::-1]).save(out / "agentview_teleported.png")
res["goal_preds_final"] = goal_preds()
res.update({"success_trace_after_teleport": trace, "success_final": trace[-1], "poses_final": poses(),
            "timing": T, "total_s": round(time.time() - t0, 1)})
(out / "result.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
env.close()
