"""Smoke test of the LIBERO install (run with .venv-libero; one sim process, sequential).

For every LIBERO-10 task: reset one init state (49, unassigned in splits.json), run a scripted absolute move
(lift 3 cm, then close and open the gripper), render the agentview and wrist images, read the success predicate,
and time each part. Writes runs/libero/smoke/{summary.json, task<NN>_{agentview,wrist}.png, strip.png}.

    .venv-libero/bin/python playgrounds/libero/smoke.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import sim as S  # noqa: E402

OUT = S.REPO / "runs" / "libero" / "smoke"
INIT = 49


def main():
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    lib = S.LiberoSim()
    table = S.task_table()
    rows, tiles = [], []
    for t in table:
        r = dict(task_id=t["task_id"], language=t["language"], n_init_states=t["n_init_states"])
        a = time.perf_counter()
        o = lib.reset(t["task_id"], INIT)
        r["reset_s"] = round(time.perf_counter() - a, 2)
        e0 = o["eef_state"].copy()
        r["success_at_reset"] = lib.success()
        # scripted absolute move: lift 3 cm at the declared 0.01 m/step limit, keep orientation and gripper command
        tgt = e0.copy()
        tgt[2] += 0.03
        a = time.perf_counter()
        n = int(np.ceil(0.03 / S.MAX_STEP[2]))
        for k in range(1, n + 1):
            o = lib.step(e0 + (tgt - e0) * k / n)
        for _ in range(10):                                   # hold so the OSC settles
            o = lib.step(tgt)
        lift = float(o["eef_state"][2] - e0[2])
        close = tgt.copy()
        close[9] = 1.0
        for _ in range(10):
            o = lib.step(close)
        q_closed = o["gripper_qpos"].tolist()
        for _ in range(10):
            o = lib.step(tgt)
        r["step_s_per_step"] = round((time.perf_counter() - a) / lib.steps, 4)
        r.update(lift_m=round(lift, 4), gripper_qpos_closed=[round(v, 4) for v in q_closed],
                 gripper_qpos_open=[round(float(v), 4) for v in o["gripper_qpos"]],
                 steps=lib.steps, success_after=lib.success(), eef_state_start=[round(float(v), 4) for v in e0],
                 health=lib.physics_health(), bddl_sha=lib.bddl_sha, init_state_sha=lib.init_state_sha)
        for cam in ("agentview", "wrist"):
            Image.fromarray(o["images"][cam]).save(OUT / f"task{t['task_id']:02d}_{cam}.png")
        tiles.append(np.concatenate([o["images"]["agentview"], o["images"]["wrist"]], axis=0))
        rows.append(r)
        print(json.dumps(r), flush=True)
    lib.close()
    Image.fromarray(np.concatenate(tiles, axis=1)).save(OUT / "strip.png")
    summary = dict(init_state=INIT, tasks=rows, evaluator=S.evaluator_hash(), total_s=round(time.perf_counter() - t0, 1))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(dict(ok=True, total_s=summary["total_s"], evaluator=summary["evaluator"])))


if __name__ == "__main__":
    main()
