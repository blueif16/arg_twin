"""Inspect a local subset of the REASSEMBLE LeRobot v3 port (robot-lev/reassemble).

Reads only data files (parquet, json, mp4); never imports anything from the data dir.

Usage (from the repo root):
  uv run --no-project --with pyarrow --with pandas --with numpy \
      python -I scripts/inspect_reassemble.py data/reassemble
Optional:
  --poses-dir data/reassemble-tudata/poses   board/camera poses from the TU Wien record
  --frames                                    also save hand/hama1/hama2 stills
                                              (add: --with av --with pillow)
  --frame-time SEC                            episode-0 time for the stills (default: auto)
  --contact-n N                               contact threshold on |force.base| (default 5 N)
  --stiffness K                               assumed translational stiffness N/m (default 500)
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 30
PLAYBOOK_SPLITS = Path(__file__).resolve().parent.parent / "playbook" / "splits.json"
ACTION_RE = re.compile(r"^(Pick|Insert|Remove|Place) (.+)\.$")


def stack(df, col):
    return np.stack(df[col].to_numpy()).astype(np.float64)


def quat_angle_deg(qa, qb):
    """Angle between unit quaternions (any order, same convention), degrees."""
    dot = np.abs(np.sum(qa * qb, axis=1)).clip(0, 1)
    return np.degrees(2 * np.arccos(dot))


def quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def q(a, ps=(50, 90, 99)):
    a = np.asarray(a)
    if a.size == 0:
        return "n/a"
    return " ".join(f"p{p}={np.percentile(a, p):.3g}" for p in ps) + f" max={a.max():.3g}"


def hdr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ----------------------------------------------------------------------------------------
def section_tasks(root):
    hdr("1. Tasks (meta/tasks.parquet)")
    t = pd.read_parquet(root / "meta/tasks.parquet").reset_index()
    t = t.sort_values("task_index")
    verbs, parts = Counter(), Counter()
    for _, r in t.iterrows():
        m = ACTION_RE.match(r["task"])
        if m:
            verbs[m.group(1)] += 1
            parts[m.group(2)] += 1
        else:
            verbs[r["task"]] += 1
    print(f"{len(t)} task labels:")
    for _, r in t.iterrows():
        print(f"  {r['task_index']:2d}  {r['task']}")
    print(f"\naction types: {dict(verbs)}")
    print(f"{len(parts)} parts (each seen with {set(parts.values())} verbs):")
    for p in sorted(parts):
        print(f"  {p}")
    return dict(zip(t["task_index"], t["task"]))


def exam_episodes() -> set:
    """Exam episodes (playbook/splits.json): never read before the final score, so every scan here drops them."""
    if not PLAYBOOK_SPLITS.exists():
        return set()
    return {int(k) for k, v in json.loads(PLAYBOOK_SPLITS.read_text())["episodes"].items() if v["split"] == "exam"}


def section_episodes(root):
    hdr("2. Episodes (meta/episodes, meta/splits.json, meta/stats.json; exam episodes left out)")
    e = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    e = e[~e["episode_index"].isin(exam_episodes())].reset_index(drop=True)
    splits = json.loads((root / "meta/splits.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    secs = e["length"] / FPS
    print(f"episodes={len(e)} frames={e['length'].sum()} ({e['length'].sum() / FPS / 60:.0f} min) "
          f"fps={info['fps']}")
    print(f"episode length s: min={secs.min():.0f} median={secs.median():.0f} max={secs.max():.0f}")
    print(f"splits: {Counter(v['split'] for v in splits.values())}")
    nseg = np.array([int(np.ravel(v)[0]) + 1 for v in e["stats/segment.index/max"]])
    succ_frac = np.array([float(np.ravel(v)[0]) for v in e["stats/segment.success/mean"]])
    print(f"segments per episode (stats segment.index max+1): total={nseg.sum()} "
          f"min={nseg.min()} median={int(np.median(nseg))} max={nseg.max()}")
    w = e["length"].to_numpy()
    print(f"frame-weighted share of frames inside successful segments: "
          f"{np.sum(succ_frac * w) / w.sum():.3f} (meta has no per-segment failure counts)")
    print(f"episodes with every frame in a successful segment: {(succ_frac >= 0.9999).sum()}")
    for i in (0, 1):
        r = e.iloc[i]
        print(f"episode {i}: recording={splits[str(i)]['recording']} split={splits[str(i)]['split']} "
              f"length={r['length']} ({r['length'] / FPS:.1f} s) data file={r['data/file_index']}")
        for cam in ("hand", "hama1", "hama2"):
            k = f"videos/observation.images.{cam}"
            print(f"   {cam}: file-{r[k + '/file_index']:03d}.mp4  "
                  f"{r[k + '/from_timestamp']:.3f}-{r[k + '/to_timestamp']:.3f} s")
    return e, splits


def segments_of(ep_df, tasks):
    """Runs of constant segment.index within one episode."""
    si = ep_df["segment.index"].to_numpy()
    ts = ep_df["timestamp"].to_numpy().astype(np.float64)
    cut = np.flatnonzero(np.diff(si) != 0) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(si)]
    out = []
    for a, b in zip(starts, ends):
        out.append(dict(seg=int(si[a]), task=tasks[int(ep_df["task_index"].iloc[a])],
                        t0=ts[a], t1=ts[b - 1] + 1 / FPS, success=bool(ep_df["segment.success"].iloc[a]),
                        a=a, b=b))
    return out


def section_episode0(d, tasks):
    hdr("3. Episode 0 segment list (derived from per-frame segment.index runs)")
    ep = d[d.episode_index == 0].sort_values("frame_index")
    segs = segments_of(ep, tasks)
    print(f"episode 0: {len(ep)} frames = {len(ep) / FPS:.1f} s, {len(segs)} segments")
    print("  seg  start_s   end_s   dur_s  ok    task")
    for s in segs:
        print(f"  {s['seg']:3d} {s['t0']:8.2f} {s['t1']:7.2f} {s['t1'] - s['t0']:7.2f}  "
              f"{'Y' if s['success'] else 'N'}    {s['task']}")
    mono = all(segs[i + 1]["seg"] == segs[i]["seg"] + 1 for i in range(len(segs) - 1))
    print(f"segment indices contiguous and increasing: {mono}")
    return segs


def section_local_counts(d, tasks):
    hdr("4. Segment / failure counts in the local data file (episodes in data file-000)")
    rows = []
    for ei, ep in d.groupby("episode_index"):
        segs = segments_of(ep.sort_values("frame_index"), tasks)
        for s in segs:
            m = ACTION_RE.match(s["task"])
            rows.append(dict(ep=ei, verb=m.group(1) if m else "No action", ok=s["success"]))
    r = pd.DataFrame(rows)
    print(f"episodes {r.ep.min()}-{r.ep.max()}: {len(r)} segments, {(~r.ok).sum()} failed")
    print(r.groupby("verb").ok.agg(n="size", failed=lambda x: int((~x).sum())).to_string())


def section_ft(root, d):
    hdr("5. Force / torque ranges")
    s = json.loads((root / "meta/stats.json").read_text())
    for k in ("observation.force", "observation.force.base", "observation.torque", "observation.torque.base"):
        st = s[k]
        print(f"{k} (all 149 episodes, per axis x,y,z):")
        for f in ("min", "q01", "q50", "q99", "max", "std"):
            print(f"   {f:4s} " + " ".join(f"{v:8.2f}" for v in st[f]))
    fb = stack(d, "observation.force.base")
    fr = stack(d, "observation.force")
    idle = d["task_index"].to_numpy() == 17
    mag = np.linalg.norm(fb, axis=1)
    print(f"\nlocal file |force.base| N: all {q(mag)}")
    print(f"   during 'No action.' frames: {q(mag[idle])}")
    print(f"   raw observation.force mean during 'No action.' (gravity+bias not removed): "
          f"{fr[idle].mean(0).round(2)}")


def section_setpoint(d, contact_n, stiffness):
    hdr("6. Setpoint question: action[t][:7] vs observation.state.ee_pose")
    exact_next, exact_last, grip_next = [], [], []
    rows = []
    for ei, ep in d.groupby("episode_index"):
        ep = ep.sort_values("frame_index")
        pose = stack(ep, "observation.state.ee_pose")
        act = stack(ep, "action")
        grip = stack(ep, "observation.state.gripper_position")
        state = stack(ep, "observation.state")
        fb = stack(ep, "observation.force.base")
        exact_next.append(np.abs(act[:-1, :7] - pose[1:]).max())
        exact_last.append(np.abs(act[-1, :7] - pose[-1]).max())
        grip_next.append(np.abs(act[:-1, 7:] - grip[1:]).max())
        dp = act[:-1, :3] - pose[:-1, :3]
        rows.append(dict(dp=dp, dang=quat_angle_deg(act[:-1, 3:7], pose[:-1, 3:7]),
                         f=fb[:-1], v=state[:-1, 30:33], z=pose[:-1, 2]))
    print(f"max |action[t][:7] - ee_pose[t+1]| over all t<T-1, all local episodes: {max(exact_next):.3g}")
    print(f"max |action[T-1][:7] - ee_pose[T-1]| (last frame repeats): {max(exact_last):.3g}")
    print(f"max |action[t][7:9] - gripper_position[t+1]|: {max(grip_next):.3g}")
    dp = np.concatenate([r["dp"] for r in rows])
    dang = np.concatenate([r["dang"] for r in rows])
    f = np.concatenate([r["f"] for r in rows])
    v = np.concatenate([r["v"] for r in rows])
    fm = np.linalg.norm(f, axis=1)
    c = fm > contact_n
    dpm = np.linalg.norm(dp, axis=1) * 1000
    print(f"\naction[t] - ee_pose[t] (= one 33 ms step of measured motion); contact = |force.base| > {contact_n} N")
    print(f"frames: all={len(dp)} contact={c.sum()} ({100 * c.mean():.1f}%) free={(~c).sum()}")
    for name, m in (("all", np.ones_like(c)), ("contact", c), ("free", ~c)):
        print(f"  {name:8s} |dpos| mm: {q(dpm[m])}")
        print(f"  {name:8s} dangle deg: {q(dang[m])}")
    vdt = v / FPS
    print(f"\ncheck dp == EE velocity * dt (state[30:33]/30): median |dp - v*dt| = "
          f"{np.median(np.linalg.norm(dp - vdt, axis=1)) * 1000:.2f} mm "
          f"(median |dp| = {np.median(dpm):.2f} mm)")
    # does the step track force like a compliance offset would?
    fc, dc = f[c], dp[c]
    C, *_ = np.linalg.lstsq(fc, dc, rcond=None)
    pred = fc @ C
    r2 = 1 - ((dc - pred) ** 2).sum(0) / ((dc - dc.mean(0)) ** 2).sum(0)
    corr = [np.corrcoef(fc[:, i], dc[:, i])[0, 1] for i in range(3)]
    print(f"contact frames: least-squares dp ~ C*F  R^2 per axis = {np.round(r2, 3)}, "
          f"corr(F_i, dp_i) = {np.round(corr, 3)}")
    with np.errstate(divide="ignore"):
        print(f"   implied 'stiffness' 1/diag(C) N/m = {np.round(1 / np.diag(C), 0)} (meaningless if R^2~0)")
    expect = fm[c] / stiffness * 1000
    print(f"if the commanded equilibrium were recorded, a K={stiffness:.0f} N/m impedance would put it "
          f"|F|/K away: {q(expect)} mm, vs observed |dp| in contact {q(dpm[c])} mm")
    release_events(d)


def release_events(d):
    """Rough stiffness from contact releases: when |F| collapses, an impedance-controlled
    arm jumps towards its (unrecorded) equilibrium by about F/K along -F."""
    print("\ncontact-release events (|F| > thr for 3 frames, then < 30% within 2 frames, arm still "
          "before; jump = max motion along -F over the next 0.4 s):")
    for fth in (5.0, 8.0):
        ks, fs, js = [], [], []
        for _, ep in d.groupby("episode_index"):
            ep = ep.sort_values("frame_index")
            f = stack(ep, "observation.force.base")
            p = stack(ep, "observation.state.ee_pose")[:, :3]
            fm = np.linalg.norm(f, axis=1)
            last = -99
            for t in range(3, len(ep) - 15):
                if t - last < 15 or fm[t - 3:t].min() <= fth or fm[t + 1] >= 0.3 * fm[t - 1]:
                    continue
                u = -f[t - 1] / fm[t - 1]
                pre = (p[t - 1] - p[t - 4]) @ u / 3
                jump = ((p[t:t + 12] - p[t - 1]) @ u).max()
                if jump > 0.002 and abs(pre) < 0.0005:
                    ks.append(fm[t - 1] / jump)
                    fs.append(fm[t - 1])
                    js.append(jump * 1000)
                    last = t
        if ks:
            print(f"  thr {fth:.0f} N: {len(ks)} events, median |F| {np.median(fs):.1f} N, median jump "
                  f"{np.median(js):.1f} mm, K=|F|/jump p25/p50/p75 = "
                  f"{np.percentile(ks, 25):.0f}/{np.median(ks):.0f}/{np.percentile(ks, 75):.0f} N/m")
        else:
            print(f"  thr {fth:.0f} N: no events")


def section_board(d, tasks, splits, poses_dir):
    hdr("7. Board-frame check: lowest EE point in the second half of each successful 'Insert X' "
        "segment, in the mocap board frame vs the raw robot frame")
    if poses_dir is None or not Path(poses_dir).is_dir():
        print("skipped (no --poses-dir)")
        return
    pts = {}
    for ei, ep in d.groupby("episode_index"):
        rec = splits[str(ei)]["recording"]
        pf = Path(poses_dir) / f"{rec}_poses.json"
        if not pf.exists():
            continue
        b = json.loads(pf.read_text())["NIST_Board1"]  # [[x,y,z],[qx,qy,qz,qw]]
        tb, Rb = np.array(b[0]), quat_xyzw_to_R(np.array(b[1]))
        ep = ep.sort_values("frame_index")
        pose = stack(ep, "observation.state.ee_pose")
        for s in segments_of(ep, tasks):
            if s["task"].startswith("Insert") and s["success"]:
                seg = pose[s["a"] + (s["b"] - s["a"]) // 2:s["b"], :3]
                p = seg[np.argmin(seg[:, 2])]
                pts.setdefault(s["task"], []).append((ei, Rb.T @ (p - tb), p))

    def mad_xy(a):  # median xy distance to the per-object median, mm
        return np.median(np.linalg.norm(a[:, :2] - np.median(a[:, :2], 0), axis=1)) * 1000

    print("object               n  median board-frame xyz (m)  xy spread mm: board frame | robot frame")
    sb, sr = [], []
    for k in sorted(pts):
        a = np.array([x[1] for x in pts[k]])
        r = np.array([x[2] for x in pts[k]])
        sb.append(mad_xy(a))
        sr.append(mad_xy(r))
        print(f"  {k:20s} {len(a):2d}  {np.round(np.median(a, 0), 3)}  {sb[-1]:6.1f} | {sr[-1]:5.0f}")
    print(f"median over objects: board frame {np.median(sb):.1f} mm vs robot frame {np.median(sr):.0f} mm "
          f"(episodes with a poses file: {sorted({x[0] for v in pts.values() for x in v})})")


def section_frames(root, e, frame_time, out_dir, d):
    hdr("8. Still frames (episode 0)")
    import av  # noqa: optional dependency

    ep = d[d.episode_index == 0].sort_values("frame_index")
    if frame_time is None:
        ins = ep["task_index"].map(lambda i: TASKS[i].startswith(("Insert", "Remove"))).to_numpy()
        fm = np.linalg.norm(stack(ep, "observation.force.base"), axis=1)
        fm[~ins] = -1
        frame_time = float(ep["timestamp"].iloc[int(np.argmax(fm))])
    fi = int(round(frame_time * FPS))
    row = ep.iloc[fi]
    print(f"t={frame_time:.3f} s (frame {fi}), task='{TASKS[int(row['task_index'])]}', "
          f"ee_pose={stack(ep.iloc[[fi]], 'observation.state.ee_pose')[0].round(3)}, "
          f"force.base={stack(ep.iloc[[fi]], 'observation.force.base')[0].round(1)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    r = e.iloc[0]
    for cam in ("hand", "hama1", "hama2"):
        k = f"videos/observation.images.{cam}"
        path = root / f"videos/observation.images.{cam}/chunk-{r[k + '/chunk_index']:03d}/file-{r[k + '/file_index']:03d}.mp4"
        target = r[k + "/from_timestamp"] + fi / FPS
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            c.seek(int(max(target - 1.0, 0) / st.time_base), stream=st, backward=True)
            got = None
            for fr in c.decode(st):
                if fr.time is not None and fr.time >= target - 0.5 / FPS:
                    got = fr
                    break
        name = out_dir / f"ep000_f{fi:05d}_{cam}.png"
        got.to_image().save(name)
        print(f"  {cam}: decoded pts time {got.time:.3f} (target {target:.3f}) -> {name}")


TASKS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--poses-dir", type=Path, default=None)
    ap.add_argument("--contact-n", type=float, default=5.0)
    ap.add_argument("--stiffness", type=float, default=500.0)
    ap.add_argument("--frames", action="store_true")
    ap.add_argument("--frame-time", type=float, default=None)
    a = ap.parse_args()
    root = a.root
    TASKS.update(section_tasks(root))
    e, splits = section_episodes(root)
    d = pd.read_parquet(root / "data/chunk-000/file-000.parquet")
    d = d[~d["episode_index"].isin(exam_episodes())]
    section_episode0(d, TASKS)
    section_local_counts(d, TASKS)
    section_ft(root, d)
    section_setpoint(d, a.contact_n, a.stiffness)
    section_board(d, TASKS, splits, a.poses_dir)
    if a.frames:
        section_frames(root, e, a.frame_time, root / "frames", d)


if __name__ == "__main__":
    main()
