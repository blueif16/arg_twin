"""Stage C self-test for the task tools (task/v1): C.tool_success, deterministic, rerunnable by a watcher.

What runs
---------
Fixed PRACTICE starts only (never held-out or red-team): the first `per_part` starts (sorted ids) of each part. Each
start is run as separate episodes for every (information mode, chain):
  modes   exact     the tools get the TRUE hole entry (frozen practice start file: board_err) and the TRUE in-hand
                    pose as `corrections`, the in-hand pose read by this runner from the scene before every tool call
                    and once more inside task_approach after the hand has turned (the `perceive` hook): thin pegs
                    pivot in the pads as the hand turns, so a t = 0 reading goes stale. This is "exact information" as validation.yaml C.tool_success defines
                    it. The tools still act only through the actor ops.
          believed  no corrections: the nominal belief from task() (what a brain has).
  chains  tools     task_approach -> task_move_until_contact -> task_search (if not engaged) -> task_insert (push) ->
                    task_release -> finish. Every tool is scored on its own postcondition.
          insert    task_insert(strategy auto, release) -> finish. Scored on the counted outcome. In exact mode a
                    task_approach comes first, so the in-hand pose is re-read after the re-orientation.
          raw       baseline: a single raw-pose descent (env move of the believed tip to the seat depth from a hover
                    above the believed hole), open, retreat, finish. Not a tool; it shows what the tools replace.

How tools are scored (the watcher side)
---------------------------------------
Between tool calls this runner reads the frozen evaluator's state (twin/evaluator.py Evaluator.state: tip depth and
lateral offset relative to the TRUE hole). The tools never see it: they run against an actor-op adapter
(PracticeEnv.command with allowed=ACTOR_OPS, the same ops a sandboxed skill gets).
  task_approach            ok and the true tip ends <= 1.0 mm from the true hole axis
  task_move_until_contact  ok (or NO_CONTACT with in_hole) and the contact class agrees with the truth:
                           surface -> the tip is not in the hole and within 2 mm of the surface height;
                           in_hole -> the tip is >= 1 mm into the hole and within the lateral tolerance
  task_search              ok and the tip is truly in the hole (>= 1 mm deep, within the lateral tolerance)
  task_insert              ok (estimated seated) and the part is truly seated (twin/predicates.py), before release
  task_release             the episode's outcome is a COUNTED success (released, at rest, no invalidating flag,
                           dt/2 rerun agrees), over episodes where the part was truly seated before release
  insert chain             counted success
Episodes are recorded by the env as actor `test-task-tools-<mode>-<chain>` (filtered out of scoring by the
sim-integrity rule for test-* actors).
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import numpy as np

from twin import REPO_DIR

PEGS = [f"round peg {i}" for i in range(1, 5)] + [f"square peg {i}" for i in range(1, 5)]
APPROACH_LAT_MM = 1.0


# ------------------------------------------------------------------------------------------------ helpers
class LocalAPI:
    """Actor ops on an in-process env, nothing else (ACTOR_OPS: task, observe, move, gripper, wait, status, finish)."""

    def __init__(self, env):
        self._env = env
        from twin.env import ACTOR_OPS
        self._ops = ACTOR_OPS

    def _c(self, op, **kw):
        r = self._env.command(dict(op=op, **{k: v for k, v in kw.items() if v is not None}), allowed=self._ops)
        if "error" in r:
            raise RuntimeError(r["error"])
        return r

    def task(self):
        return self._c("task")

    def observe(self):
        return self._c("observe")

    def status(self):
        return self._c("status")

    def move(self, pos, quat=None, speed=None, force_max=None, stop_force=None, timeout=None):
        return self._c("move", pos=[float(x) for x in pos], quat=None if quat is None else [float(x) for x in quat],
                       speed=speed, force_max=force_max, stop_force=stop_force, timeout=timeout)

    def gripper(self, action, width=None):
        return self._c("gripper", action=action, width=width)

    def wait(self, seconds):
        return self._c("wait", seconds=float(seconds))

    def finish(self, claim=None):
        return self._c("finish", claim=claim)


def exact_corrections(task: dict, start: dict, nominal: dict, env=None) -> dict:
    """Test oracle (watcher side, before the first command): the true hole entry from the start's board_err, and the
    true in-hand pose. With `env`, the in-hand pose is read from the scene at t = 0, after the grasp pre-roll: thin
    pegs re-seat in the pads during the pre-roll (round peg 1: +2.5-3 deg pivot, 1.3-1.5 mm tip shift against the
    start file's registered pose, Lessons). Without it, the start file's registered pose (rock 0: the pads squeeze it
    out)."""
    from twin import starts as S
    from twin.task_tools import PAD_TO_TCP
    H, Rw = S.entry_pose(start["part"], nominal, start["board_err"])
    sh = Rw.T @ (H - np.asarray(task["hole"]["entry_pos"], float)) * 1000
    ih = start["inhand"]
    grip = dict(dx_mm=1000 * ih["dx"], dy_mm=1000 * ih["dy"], grasp_from_top_mm=1000 * ih["grasp_from_top"],
                pivot_deg=math.degrees(ih["pivot"]), rock_deg=0.0)
    if env is not None:
        import mujoco
        e = env.ep
        m, d = e["m"], e["d"]
        t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{e['aux']['slug']}/tip")
        x, _, R = e["ctrl"].arm.pose(d)
        tip = R.T @ (d.site_xpos[t] - x)
        a = R.T @ d.site_xmat[t].reshape(3, 3)[:, 2]
        pivot = math.atan2(a[0], a[2])
        rock = math.atan2(-a[1], math.hypot(a[0], a[2]))
        L = float(task["part_geometry"]["length_mm"]) / 1000
        lg = (tip[2] + PAD_TO_TCP) / a[2]
        grip = dict(dx_mm=1000 * (tip[0] - lg * a[0]), dy_mm=1000 * (tip[1] - lg * a[1]),
                    grasp_from_top_mm=1000 * (L - lg), pivot_deg=math.degrees(pivot), rock_deg=math.degrees(rock))
    return dict(hole_shift_mm=[float(x) for x in sh], grip=grip)


def truth(env) -> dict:
    """Watcher-side: evaluator state of the tip relative to the TRUE hole (never passed to the tools)."""
    e = env.ep
    st = e["ev"].state(e["d"])
    from twin.predicates import numeric
    tol = numeric(e["m"], f"{e['aux']['slug']}/tol")
    return dict(depth_mm=round(1000 * st["depth"], 3), lateral_mm=round(1000 * st["lateral"], 3),
                tilt_deg=round(st["tilt_deg"], 3), seated=bool(st["seated"]), engaged=bool(st["engaged"]),
                held=bool(st["held"]), touching_board=bool(st["touching_board"]),
                lateral_tol_mm=round(1000 * float(tol[1]), 3))


def in_hole(t) -> bool:
    return t["depth_mm"] >= 1.0 and t["lateral_mm"] <= t["lateral_tol_mm"]


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list:
    if n == 0:
        return [None, None]
    try:
        from scipy.stats import beta
        lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
        hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
        return [round(lo, 3), round(hi, 3)]
    except Exception:  # noqa: BLE001
        return [None, None]


# ------------------------------------------------------------------------------------------------ one episode
def _brief(r: dict) -> dict:
    return {k: r.get(k) for k in ("ok", "code", "summary", "stage_est", "commands", "sim_s")} | dict(
        contact=r.get("contact", {}).get("state"), depth=r.get("depth"), tried=[(p["phase"], p["commands"],
                                                                                  p["stopped"]) for p in r.get("tried", [])])


def run_job(job: dict) -> dict:
    from twin.env import PracticeEnv
    from twin import task_tools as TT
    t0 = time.time()
    mode, chain, sid = job["mode"], job["chain"], job["start_id"]
    env = PracticeEnv(job["brain"], job["run_id"], "practice", job["cousin"], actor=f"test-task-tools-{mode}-{chain}",
                      session=f"selftest-{job['stamp']}-{job['worker_tag']}", dt_half=job["dt_half"])
    out = dict(start_id=sid, mode=mode, chain=chain, tools={}, error=None)
    try:
        rep = env.reset(sid)
        start = env.ep["start"]
        out.update(part=start["part"], board_err_mm=round(1000 * math.hypot(start["board_err"]["dx"],
                                                                           start["board_err"]["dy"]), 2),
                   pivot_deg=round(math.degrees(start["inhand"]["pivot"]), 2),
                   inhand_dx_mm=round(1000 * start["inhand"]["dx"], 2))
        api = LocalAPI(env)
        corr = exact_corrections(rep["task"], start, env.nominal, env) if mode == "exact" else None
        belief = TT.believed(rep["task"], corr)
        mem: dict = {}
        cap = job["force_cap"]

        def perceive():
            return TT.believed(rep["task"], exact_corrections(rep["task"], start, env.nominal, env))

        def call(name, fn):
            b, pc = belief, None
            if mode == "exact":     # perfect perception at every call: the in-hand pose drifts as the hand moves
                b, pc = perceive(), perceive
            tt = TT.TaskTools(api, b, mem, perceive=pc)
            try:
                r = fn(tt)
            except RuntimeError as ex:
                tt.err = str(ex)
                r = tt.report(name, False, "ENV_ERROR", str(ex))
            tr = truth(env) if not env.ep["done"] else None
            out["tools"][name] = dict(_brief(r), truth=tr)
            return r, tr

        claim = "failure"
        if chain == "tools":
            r, tr = call("task_approach", lambda tt: tt.approach(standoff_mm=12.0))
            out["tools"]["task_approach"]["success"] = bool(r["ok"] and tr and tr["lateral_mm"] <= APPROACH_LAT_MM)
            if r["ok"]:
                r, tr = call("task_move_until_contact", lambda tt: tt.move_until_contact("down", distance_mm=12.0 + 2 *
                                                                                        belief["axial_sigma_mm"] + 5,
                                                                                        max_force=min(8.0, cap)))
                cls = r.get("contact", {}).get("state")
                good = False
                if tr is not None and (r["ok"] or cls == "in_hole"):
                    if cls == "in_hole":
                        good = in_hole(tr)
                    elif cls == "surface":
                        good = (not in_hole(tr)) and abs(tr["depth_mm"]) <= 2.0
                out["tools"]["task_move_until_contact"]["success"] = bool(good)
                proceed = r["ok"] or cls == "in_hole"
                if proceed and not mem.get("engaged"):
                    rad, pitch = (1.5, 0.5) if mode == "exact" else (None, None)
                    r, tr = call("task_search", lambda tt: tt.search("spiral", radius_mm=rad, pitch_mm=pitch,
                                                                     max_force=cap))
                    out["tools"]["task_search"]["success"] = bool(r["ok"] and tr and in_hole(tr))
                    proceed = r["ok"]
                if proceed:
                    r, tr = call("task_insert", lambda tt: tt.insert("auto", max_force=min(10.0, cap)))
                    out["tools"]["task_insert"]["success"] = bool(r["ok"] and tr and tr["seated"])
                    seated_before = bool(tr and tr["seated"])
                    if r["ok"]:
                        r, tr = call("task_release", lambda tt: tt.release())
                        out["tools"]["task_release"]["seated_before"] = seated_before
                        claim = "success" if r["ok"] else "failure"
        elif chain == "insert":
            if mode == "exact":     # refresh the exact in-hand pose after the big re-orientation (see call())
                call("task_approach", lambda tt: tt.approach(standoff_mm=12.0))
                out["tools"]["task_approach"]["success"] = None
            r, tr = call("task_insert", lambda tt: tt.insert("auto", max_force=min(10.0, cap), release=True))
            claim = "success" if r["ok"] else "failure"
        elif chain == "raw":
            claim = _raw_descent(api, belief, TT)
        if not env.ep["done"]:
            api.finish(claim=claim)
        rec = env.ep["record"]
        out.update(outcome=rec["outcome"], raw_outcome=rec["raw_outcome"], counted=bool(rec["counted"]),
                   claim=rec["claim"], false_claim=rec["false_claim"], max_stage=rec["evaluation"]["max_stage"],
                   final=rec["evaluation"]["final"], flags=[f["monitor"] for f in rec["monitors"]["flags"]],
                   envelope_violations=[f"{v['feature']}:{v['side']}" for v in (rec["envelope"]["violations"] or [])],
                   envelope_features={k: rec["envelope"]["features"].get(k) for k in
                                      ("peak_force_N", "approach_speed_mm_s", "insert_speed_mm_s",
                                       "insert_duration_s", "wiggle_mm", "wiggle_deg")} if rec["envelope"]["features"] else None,
                   n_commands=rec["n_commands"], sim_s=rec["sim_s"], dt_half=rec["dt_half"],
                   episode_file=env.ep["record_info"].get("file"))
        if "task_release" in out["tools"]:
            t = out["tools"]["task_release"]
            t["success"] = bool(out["counted"])
        if chain == "insert":
            out["tools"]["task_insert"]["success"] = bool(out["counted"])
    except Exception as ex:  # noqa: BLE001
        out["error"] = f"{type(ex).__name__}: {ex}"
        out["trace"] = traceback.format_exc(limit=4)
        try:
            if env.ep is not None and not env.ep["done"]:
                env._end_episode("error")
        except Exception:  # noqa: BLE001
            pass
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def _raw_descent(api, belief, TT) -> str:
    """Baseline: what a brain sending raw poses does without task tools: hover over the believed hole, one straight
    move of the believed tip to the seat depth (10 N ceiling), open, retreat."""
    o = api.observe()
    H, Re, tip_tcp, _ = TT._B(belief)
    Rh = TT.hand_rotation(belief, o["tcp"]["quat"])
    q = TT.mat_to_quat(Rh).tolist()
    hover = H + Re @ np.array([0, 0, -0.012]) - Rh @ tip_tcp
    p = np.asarray(o["tcp"]["pos"], float)
    api.move((p + Re[:, 2] * min(0.0, float((hover - p) @ Re[:, 2]))).tolist(), q, speed=0.03, force_max=15.0)
    api.move(hover.tolist(), q, speed=0.03, force_max=15.0)
    for _ in range(6):
        api.wait(1.0)
    seat = float(belief["hole"]["seat_depth_mm"]) / 1000
    api.move((H + Re @ np.array([0, 0, seat]) - Rh @ tip_tcp).tolist(), q, speed=0.02, force_max=10.0)
    api.gripper("open")
    p = np.asarray(api.observe()["tcp"]["pos"], float)
    api.move((p - Re[:, 2] * 0.03).tolist(), q, speed=0.03, force_max=10.0)
    return "success"


# ------------------------------------------------------------------------------------------------ the run
def run(brain="claude", run_id="run-standin-1", parts=None, per_part=6, info_modes=("exact", "believed"),
        chains=("tools", "insert"), workers=12, cousin="c000", dt_half=True, out_dir: Path | None = None) -> dict:
    from twin import starts as S
    from harness.tools.task import force_cap
    run_dir = REPO_DIR / "runs" / brain / run_id
    starts, _ = S.load("practice", run_dir / "starts")
    parts = list(parts or PEGS)
    per_part = max(1, min(int(per_part), 20))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_dir) if out_dir else run_dir / "task_tools" / "selftest" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for part in parts:
        ids = sorted(s for s, v in starts.items() if v["part"] == part)[:per_part]
        for sid in ids:
            for mode in info_modes:
                for chain in chains:
                    jobs.append(dict(brain=brain, run_id=run_id, start_id=sid, mode=mode, chain=chain, cousin=cousin,
                                     dt_half=dt_half, stamp=stamp, force_cap=force_cap(part, run_dir),
                                     worker_tag=f"{len(jobs):03d}"))
    cfg = dict(brain=brain, run_id=run_id, parts=parts, per_part=per_part, info_modes=list(info_modes),
               chains=list(chains), cousin=cousin, dt_half=dt_half, n_jobs=len(jobs), interface="task/v1",
               start_set="practice", started=stamp)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    t0 = time.time()
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(max(1, int(workers)), maxtasksperchild=4) as pool, open(out_dir / "results.jsonl", "w") as fh:
        for r in pool.imap_unordered(run_job, jobs):
            results.append(r)
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
    summary = summarize(results)
    summary.update(config=cfg, wall_s=round(time.time() - t0, 1),
                   dir=str(out_dir.relative_to(REPO_DIR)) if out_dir.is_relative_to(REPO_DIR) else str(out_dir))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def summarize(results: list[dict]) -> dict:
    tools = ("task_approach", "task_move_until_contact", "task_search", "task_insert", "task_release")
    out = {}
    for mode in sorted({r["mode"] for r in results}):
        rs = [r for r in results if r["mode"] == mode]
        m = {}
        for chain in sorted({r["chain"] for r in rs}):
            cr = [r for r in rs if r["chain"] == chain]
            per_tool = {}
            for t in tools:
                xs = [r["tools"][t] for r in cr if t in r["tools"] and r["tools"][t].get("success") is not None]
                if t == "task_release":
                    xs = [x for x in xs if x.get("seated_before")]
                if not xs:
                    continue
                k = sum(1 for x in xs if x["success"])
                codes: dict[str, int] = {}
                for x in xs:
                    if not x["success"]:
                        codes[x["code"] or "?"] = codes.get(x["code"] or "?", 0) + 1
                per_tool[t] = dict(success=k, n=len(xs), rate=round(k / len(xs), 3), ci95=clopper_pearson(k, len(xs)),
                                   failure_codes=codes)
            k = sum(1 for r in cr if r.get("counted"))
            n = len(cr)
            by_part = {}
            for p in sorted({r.get("part") for r in cr if r.get("part")}):
                pr = [r for r in cr if r.get("part") == p]
                by_part[p] = f"{sum(1 for r in pr if r.get('counted'))}/{len(pr)}"
            viol: dict[str, int] = {}
            for r in cr:
                for v in r.get("envelope_violations") or []:
                    viol[v] = viol.get(v, 0) + 1
            fails = [dict(start=r["start_id"], outcome=r.get("outcome"), board_err_mm=r.get("board_err_mm"),
                          pivot_deg=r.get("pivot_deg"), error=r.get("error"),
                          last_tool=next((f"{t}:{r['tools'][t].get('code')}" for t in reversed(tools)
                                          if t in r["tools"]), None)) for r in cr if not r.get("counted")]
            m[chain] = dict(episodes=n, counted_success=k, rate=round(k / n, 3) if n else None,
                            ci95=clopper_pearson(k, n), by_part=by_part, per_tool=per_tool,
                            flags=sum(len(r.get("flags") or []) for r in cr),
                            invalidating_or_flagged=[r["start_id"] for r in cr if r.get("outcome") == "success_flagged"],
                            false_claims=sum(1 for r in cr if r.get("false_claim")),
                            envelope_violations=viol, failures=fails,
                            median_commands=float(np.median([r.get("n_commands") or 0 for r in cr])) if cr else None,
                            median_sim_s=float(np.median([r.get("sim_s") or 0 for r in cr])) if cr else None,
                            errors=[r["start_id"] for r in cr if r.get("error")])
        out[mode] = m
    # C.tool_success as defined: min over tools of success with exact information
    ex = out.get("exact", {})
    rates = []
    for chain in ex.values():
        for t, v in chain["per_tool"].items():
            rates.append((v["rate"], t))
    out["C_tool_success_exact_min"] = min(rates) if rates else None
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(run(per_part=int(sys.argv[1]) if len(sys.argv) > 1 else 2), indent=1, default=float)[:4000])
