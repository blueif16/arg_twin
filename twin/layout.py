"""Board layout from REASSEMBLE data: where each part's hole is, in the board frame.

The board frame is the mocap `NIST_Board1` frame stored per recording in
data/reassemble-tudata/poses/<recording>_poses.json ([x,y,z], [qx,qy,qz,qw]; mocap world == robot base
frame, docs/research/reassemble-data-notes.md). A scene places the board with ONE pose: this frame in
the robot/world frame.

Method (one event per successful `Insert <part>.` segment, train split only by default):
  * the gripper must hold something in the first third of the segment (finger sum < 75 mm);
  * the release frame is the last frame before the fingers open > 3 mm beyond the holding width
    (or the last frame if they never open);
  * the EE pose (`observation.state.ee_pose`, O_T_EE = 0.174 m tool point, w-first quaternion) at
    release is mapped into that recording's board frame.
Hole xy = per-part median; sigma = 1.4826 * MAD / sqrt(n) per axis, floored at 1 mm. Key holes take
their long-axis yaw from the gripper yaw (the fingers close along EE y and the data show keys gripped
across their long side), snapped to the nearest 90 deg. A 4x4 grid (pitch free) is fitted to the eight
pegs as a cross-check and to place the plate outline (NIST cells sit 41.4 mm from the plate edge).

    .venv/bin/python -m twin.layout            # -> twin/assets/board_layout.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from twin import REPO_DIR

DATA = REPO_DIR / "data" / "reassemble"
POSES = REPO_DIR / "data" / "reassemble-tudata" / "poses"
OUT = REPO_DIR / "twin" / "assets" / "board_layout.json"
FPS = 30
HOLD_MAX = 0.075  # finger sum below this = holding something (open is 0.080)
RELEASE_OPEN = 0.003  # fingers this much wider than the holding width = released
NIST_EDGE_TO_CELL = 0.0414  # NIST plate: first hole column/row centre from the plate edge (plate STL)
NIST_PLATE = 0.384
# grid cells of the eight pegs (column i along +x, row j along -y), read off the medians
PEG_CELLS = {"round peg 1": (0, 2), "round peg 2": (0, 1), "round peg 3": (3, 0), "round peg 4": (3, 3),
             "square peg 1": (2, 2), "square peg 2": (1, 3), "square peg 3": (2, 0), "square peg 4": (0, 0)}


def slug(part: str) -> str:
    """'round peg 1' -> 'round_peg_1', 'D-SUB' -> 'dsub' (MJCF name stem)."""
    return re.sub(r"[^a-z0-9]+", "_", part.lower().replace("-", "")).strip("_")


def _quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def insert_events(split: str | None = "train") -> list[dict]:
    import pandas as pd

    tasks = pd.read_parquet(DATA / "meta/tasks.parquet").reset_index()
    tasks = dict(zip(tasks["task_index"].astype(int), tasks["task"]))
    from twin import exam
    splits = json.loads((DATA / "meta/splits.json").read_text())
    events = []
    for f in sorted((DATA / "data").glob("chunk-*/file-*.parquet")):
        df = pd.read_parquet(f)
        for ei, ep in df.groupby("episode_index"):
            info = splits[str(int(ei))]
            if split and info["split"] != split:
                continue
            if exam.is_exam(int(ei)):  # a layout is never fitted on exam episodes (playbook/splits.json)
                continue
            pf = POSES / f"{info['recording']}_poses.json"
            if not pf.exists():
                continue
            b = json.loads(pf.read_text())["NIST_Board1"]
            tb, Rb = np.array(b[0]), _quat_xyzw_to_R(b[1])
            ep = ep.sort_values("frame_index")
            si = ep["segment.index"].to_numpy()
            cut = np.flatnonzero(np.diff(si) != 0) + 1
            pose = np.stack(ep["observation.state.ee_pose"].to_numpy()).astype(float)
            grip = np.stack(ep["observation.state.gripper_position"].to_numpy()).astype(float).sum(1)
            for a, e in zip(np.r_[0, cut], np.r_[cut, len(si)]):
                task = tasks[int(ep["task_index"].iloc[a])]
                if not task.startswith("Insert ") or not bool(ep["segment.success"].iloc[a]):
                    continue
                g = grip[a:e]
                hold = g < HOLD_MAX
                if not hold[: max(1, len(g) // 3)].any():
                    continue  # pushed with open fingers: no part in hand
                w = float(np.median(g[hold]))
                first_hold = int(np.argmax(hold))
                opened = np.flatnonzero(g > w + RELEASE_OPEN)
                opened = opened[opened > first_hold]
                rel = a + (int(opened[0]) - 1 if len(opened) else len(g) - 1)
                p = Rb.T @ (pose[rel, :3] - tb)
                qw, qx, qy, qz = pose[rel, 3:7]
                Rbe = Rb.T @ _quat_xyzw_to_R([qx, qy, qz, qw])
                events.append(dict(episode=int(ei), recording=info["recording"], part=task[7:-1],
                                   t_release=float(rel - a) / FPS, xyz=p.tolist(),
                                   ee_z_axis=Rbe[:, 2].round(5).tolist(),
                                   tilt_deg=float(np.degrees(np.arccos(np.clip(-Rbe[2, 2], -1, 1)))),
                                   ee_yaw_deg=float(np.degrees(np.arctan2(Rbe[1, 0], Rbe[0, 0]))),
                                   grip_width_mm=1000 * w))
    return events


def _mad(v):
    v = np.asarray(v)
    return float(np.median(np.abs(v - np.median(v))))


def summarise(events: list[dict]) -> dict:
    parts = {}
    for part in sorted({e["part"] for e in events}):
        ev = [e for e in events if e["part"] == part]
        xyz = np.array([e["xyz"] for e in ev])
        n = len(ev)
        sig = [float(max(1.4826 * _mad(xyz[:, k]) / np.sqrt(n), 0.001)) for k in (0, 1)]
        # key long axis = finger closing axis = EE y => yaw(EE x) + 90, modulo 180 (gripper symmetric)
        long_yaw = np.mod(np.array([e["ee_yaw_deg"] for e in ev]) + 90.0, 180.0)
        # circular median on the doubled angle
        ang = np.radians(2 * long_yaw)
        mean_long = float(np.mod(np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2, 180))
        parts[part] = dict(
            slug=slug(part), n=n, episodes=sorted({e["episode"] for e in ev}),
            xy=[round(float(np.median(xyz[:, 0])), 5), round(float(np.median(xyz[:, 1])), 5)],
            xy_sigma=[round(s, 5) for s in sig],
            xy_mad_mm=[round(1000 * _mad(xyz[:, 0]), 2), round(1000 * _mad(xyz[:, 1]), 2)],
            tcp_z_release=round(float(np.median(xyz[:, 2])), 4),
            tcp_z_release_mad_mm=round(1000 * _mad(xyz[:, 2]), 2),
            tilt_deg_median=round(float(np.median([e["tilt_deg"] for e in ev])), 2),
            grip_width_mm_median=round(float(np.median([e["grip_width_mm"] for e in ev])), 2),
            long_axis_yaw_deg_measured=round(mean_long, 1),
            long_axis_yaw_deg=float(90.0 * np.round(mean_long / 90.0) % 180.0),
        )
    # grid cross-check on the pegs: p = t + i*u + j*v (u, v free 2-vectors)
    grid = None
    names = [k for k in PEG_CELLS if k in parts]
    if len(names) >= 4:
        A, bvec = [], []
        for k in names:
            i, j = PEG_CELLS[k]
            A += [[1, 0, i, 0, j, 0], [0, 1, 0, i, 0, j]]
            bvec += parts[k]["xy"]
        sol, *_ = np.linalg.lstsq(np.array(A, float), np.array(bvec), rcond=None)
        t, u, v = sol[:2], sol[2:4], sol[4:6]
        res = {}
        for k in names:
            i, j = PEG_CELLS[k]
            res[k] = [round(1000 * float(x), 2) for x in np.array(parts[k]["xy"]) - (t + i * u + j * v)]
        rms = float(np.sqrt(np.mean([r[0] ** 2 + r[1] ** 2 for r in res.values()])))
        # plate outline: cell (0,0) is NIST_EDGE_TO_CELL from the -x edge and from the +y edge
        x0 = float(t[0] - NIST_EDGE_TO_CELL)
        y1 = float(t[1] + NIST_EDGE_TO_CELL)
        grid = dict(origin=[round(float(x), 5) for x in t], step_i=[round(float(x), 5) for x in u],
                    step_j=[round(float(x), 5) for x in v],
                    pitch_mm=[round(1000 * float(np.linalg.norm(u)), 1), round(1000 * float(np.linalg.norm(v)), 1)],
                    angle_i_deg=round(float(np.degrees(np.arctan2(u[1], u[0]))), 2),
                    residual_mm=res, residual_rms_mm=round(rms, 2),
                    plate_xy_min=[round(x0, 4), round(y1 - NIST_PLATE, 4)],
                    plate_size=[NIST_PLATE, NIST_PLATE])
    return dict(parts=parts, grid=grid)


def board_z_stats() -> dict:
    zs = []
    for f in sorted(POSES.glob("*_poses.json")):
        zs.append(json.loads(f.read_text())["NIST_Board1"][0][2])
    zs = np.array(zs)
    return dict(n=len(zs), z_p5=round(float(np.percentile(zs, 5)), 4), z_p50=round(float(np.median(zs)), 4),
                z_p95=round(float(np.percentile(zs, 95)), 4))


def build_layout(split: str | None = "train") -> dict:
    ev = insert_events(split)
    s = summarise(ev)
    return dict(
        generated_by="twin/layout.py", generated_at=time.strftime("%Y-%m-%d %H:%M"),
        frame="board = mocap NIST_Board1 frame (per-recording poses JSON); metres; z up",
        split=split, n_events=len(ev), episodes=sorted({e["episode"] for e in ev}),
        method="TCP (O_T_EE, 0.174 m) at gripper release in successful Insert segments, mapped into the "
               "recording's mocap board frame; median per part",
        mocap_board_world_z=board_z_stats(), **s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", help="train | test | all")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    lay = build_layout(None if a.split == "all" else a.split)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(lay, indent=1))
    g = lay["grid"]
    print(f"{lay['n_events']} insert events from episodes {lay['episodes']} -> {a.out.relative_to(REPO_DIR)}")
    for k, v in lay["parts"].items():
        print(f"  {k:14s} n={v['n']} xy={v['xy']} sigma_mm={[round(1000 * float(s), 1) for s in v['xy_sigma']]} "
              f"long_yaw={v['long_axis_yaw_deg_measured']} grip={v['grip_width_mm_median']}mm")
    if g:
        print(f"grid pitch {g['pitch_mm']} mm, angle {g['angle_i_deg']} deg, residual rms {g['residual_rms_mm']} mm, "
              f"plate min {g['plate_xy_min']}")


if __name__ == "__main__":
    main()
