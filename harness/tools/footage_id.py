"""Stage A helpers: name a part by WHERE it goes and judge an attempt from its end state (skill: read-footage).

Label-free by construction: these tools read only robot signals (footage._frames, which never loads label columns),
the recording's mocap board pose (as a starting guess) and the rigid hole layout fitted on TRAIN recordings
(runs/claude/run-standin-1/board/layout_contact.json, skill board-from-contacts). They never call board_fit's
extract_events (that one reads segment labels), so they are safe for blind reads of gate recordings.

Method, measured on train recordings (playbook/skills/read-footage Lessons, 2026-10-25; evaluation
runs/footage/refine-20261025/eval_train.json):
  * gripper cycles: every close (finger sum < 75 mm) ... open is one part handled; < 1 mm for < 10 s with no contact
    = closed on nothing.
  * board pose: mocap NIST_Board1 pose, refined label-free: each cycle's grasp point and seat point (deepest TCP in
    the 3 s before release) is assigned to the nearest layout hole within 20 mm and the pose is refitted with the
    point-to-line model of twin/board_fit (tight across the finger pads, loose along them), 4 rounds. When fewer than
    a quarter of the events land on holes, a label-free RANSAC over event and hole pairs is tried instead.
  * where: a grasp or seat within 25 mm (xy) of a hole is AT that hole and names the part (train: Remove grasps p95
    12 mm, Pick grasps p5 101 mm from the nearest hole). Hold width (GRIP_MM) only breaks ties.
  * attempts: while a part is held, every visit near the target hole that ends in a retreat (z up > retreat_mm above
    the visit's lowest point, or leaving sideways) is one failed Insert in the dataset's convention.
  * end state: a Remove succeeded only if the part was carried > carried_mm from its hole; after a release at a hole,
    an open-hand return near a peg, a regrasp that lifts the part out (any part) or in place (connectors), or the same
    part inserted again later, mark that Insert failed.

Tools:
  identify_part_by_hole  per gripper cycle in a window: grasp/release location, ranked holes, the part they name
  end_state_check        per gripper cycle in a window: attempts, release, what follows; verdict per attempt with the
                         rule that fired and its train precision; a draft segment list in dataset conventions
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from harness.registry import REPO, tool

FPS = 30
LAYOUT_FILE = REPO / "runs" / "claude" / "run-standin-1" / "board" / "layout_contact.json"
POSES = REPO / "data" / "reassemble-tudata" / "poses"

# Frozen after tuning on train recordings (read-footage Lessons 2026-10-25). Distances mm, times s.
PARAMS = dict(
    hold_mm=75.0,          # finger sum below this = holding (fully open is 80)
    nothing_mm=1.0,        # hold width below this = closed on nothing ...
    nothing_max_s=10.0,    # ... if the close also lasts less than this with no contact > 5 N
    min_close_s=0.4,       # shorter closes are ignored
    at_hole_mm=25.0,       # grasp/release within this xy distance of a hole = at that hole
    assign_mm=20.0,        # board refinement: event-to-hole assignment gate
    near_target_mm=35.0,   # insert attempt: TCP within this of the target hole
    retreat_mm=35.0,       # attempt ends when z rises this far above the attempt's lowest point while holding
    lift_fast_mm=8.0,      # lift onset: first 0.5 s window with z up more than this
    free_lift_mm=15.0,     # a regrasp counts as "lifted" when z rises this much while holding
    carried_mm=80.0,       # a Remove succeeded only if the part is released this far from its hole (carried away)
    regrasp_mm=45.0,       # a grasp this close to the hole just released re-takes that part (it may have toppled)
    revisit_mm=10.0,       # open-hand return: drop from the running max near the hole after rising
    revisit_rad_mm=65.0,   # ... within this xy distance of the hole (one open finger sits ~40 mm off axis)
    revisit_top_mm=40.0,   # ... and back down to within this height above the release
    dip_mm=30.0,           # aborted approach: an open-hand dip this deep (down and up again) with no close
    dip_hole_mm=20.0,      # ... at a hole (xy) = an aborted Remove of that hole's part
    first_end_s=1.05,      # Pick/Remove ends this long after the lift onset (dataset convention, train median)
    retreat_end_s=0.5,     # a failed attempt ends this long after its retreat starts to climb
    release_end_s=0.35,    # Insert/Place ends this long after the middle of the gripper opening
    nothing_end_s=-1.0,    # a close on nothing ends this long after the middle of the re-opening
    unseated_end_s=0.9,    # a peg release followed by an open-hand return ends this long after the opening middle
)

# Hold width per part (mm): median, MAD, p10, p90, n over successful Pick/Remove grasps of the 89 train recordings
# (gripper finger sum while closed; scratch analysis 2026-10-25, read-footage Lessons). Round and square pegs of the
# same index overlap, and several parts have a second grasp mode (gear hubs, keys across the short side, D-SUB and
# waterproof sideways), which p10/p90 capture. Location names the part; width only breaks ties near several holes.
GRIP_MM = {
    "BNC": [12.4, 0.2, 12.0, 12.7, 77], "D-SUB": [15.5, 0.2, 15.3, 38.7, 76], "Ethernet": [7.8, 0.6, 6.9, 10.1, 69],
    "USB": [5.4, 2.8, 2.3, 9.4, 63], "bolt 4": [22.6, 0.2, 22.4, 24.3, 69], "large gear": [59.4, 0.3, 32.9, 59.7, 74],
    "medium gear": [39.5, 0.4, 28.0, 39.9, 71], "round peg 1": [2.7, 0.9, 1.5, 3.5, 68],
    "round peg 2": [6.8, 0.5, 5.9, 7.4, 67], "round peg 3": [11.0, 0.3, 10.3, 11.3, 72],
    "round peg 4": [14.9, 0.3, 14.4, 15.3, 63], "small gear": [19.6, 0.4, 14.0, 19.9, 63],
    "square peg 1": [3.2, 0.3, 2.5, 3.6, 64], "square peg 2": [7.4, 0.3, 6.5, 7.6, 62],
    "square peg 3": [11.1, 0.3, 7.5, 11.5, 73], "square peg 4": [15.1, 0.3, 9.6, 15.5, 69],
    "waterproof": [31.4, 0.4, 31.0, 62.0, 83]}


def _grip_cost(w: float, part: str, table: dict | None = None) -> float:
    """How far (in robust sigmas) a hold width is from a part's train widths; the p10/p90 modes cost +1."""
    med, mad, p10, p90, _ = (table or GRIP_MM)[part]
    sig = max(1.4826 * mad, 0.5)
    return float(min(abs(w - med) / sig, 1 + abs(w - p10) / sig, 1 + abs(w - p90) / sig))


def _grip_guess(w: float, table: dict | None = None, n: int = 3) -> list[dict]:
    t = table or GRIP_MM
    c = sorted((round(_grip_cost(w, p, t), 2), p) for p in t)
    return [dict(part=p, grip_cost=v) for v, p in c[:n]]

# ------------------------------------------------------------------------------------------------ data
def _sig(ep: int) -> dict:
    """Label-free per-frame signals of one episode (exam refused by footage._frames)."""
    from harness.tools import footage as ft
    from scipy.spatial.transform import Rotation
    f = ft._frames(int(ep))
    st = f["observation.state"]
    R = Rotation.from_quat(st[:, 26:30], scalar_first=True).as_matrix()
    pad = R[:, :2, 0]
    pad = pad / np.maximum(np.linalg.norm(pad, axis=1, keepdims=True), 1e-9)
    xax = R[:, :, 0]
    F = f["observation.force.base"]
    return dict(t=f["t"], grip=(st[:, 21] + st[:, 22]) * 1000.0, pos=st[:, 23:26] * 1000.0, pad=pad,
                F=F, Fm=np.linalg.norm(F, axis=1), yaw=np.degrees(np.unwrap(np.arctan2(xax[:, 1], xax[:, 0]))))


@lru_cache(maxsize=1)
def _train_layout() -> dict:
    """The train contact layout (board frame, mm). Refuses a layout that used any non-train recording."""
    lay = json.loads(LAYOUT_FILE.read_text())
    spl = {int(k): v["split"] for k, v in json.loads((REPO / "playbook/splits.json").read_text())["episodes"].items()}
    bad = [e for e in lay["episodes"] if spl.get(int(e)) != "train"]
    if bad:
        raise RuntimeError(f"layout {LAYOUT_FILE} was fitted on non-train episodes {bad}")
    return lay


def _holes(layout: dict) -> tuple[list[str], np.ndarray]:
    names = sorted(layout["parts"])
    return names, np.array([[1000.0 * v for v in layout["parts"][n]["xy"]] for n in names])


def _mocap(ep: int):
    """(x mm, y mm, yaw rad, z mm, sane) of the recording's mocap NIST_Board1 pose, or None."""
    rec = json.loads((REPO / "playbook/splits.json").read_text())["episodes"][str(int(ep))]["recording"]
    f = POSES / f"{rec}_poses.json"
    if not f.exists():
        return None
    pos, (x, y, z, w) = json.loads(f.read_text())["NIST_Board1"]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z)]])
    r22 = 1 - 2 * (x * x + y * y)
    return (1000 * pos[0], 1000 * pos[1], float(np.arctan2(R[1, 0], R[0, 0])), 1000 * pos[2],
            bool(np.degrees(np.arccos(np.clip(r22, -1, 1))) < 5.0))


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _rot(y):
    c, s = np.cos(y), np.sin(y)
    return np.array([[c, -s], [s, c]])


# ------------------------------------------------------------------------------------------------ cycles
def _cycles(s: dict, P: dict) -> list[dict]:
    """Gripper cycles: a close-hold-open interval. a = fingers start closing, c = finished closing, rel = last frame
    before opening, b = open again (frame indices)."""
    g = s["grip"]
    out = []
    for a, b in _runs(g < P["hold_mm"]):
        if (b - a) / FPS < P["min_close_s"]:
            continue
        seg = g[a:b]
        q = len(seg) // 4
        w = float(np.median(seg[q:max(q + 1, 3 * q)]))
        c = a + int(np.argmax(seg < w + 0.5)) if (seg < w + 0.5).any() else a
        op = np.flatnonzero(seg > w + 3.0)
        op = op[op > c - a]
        rel = a + (int(op[0]) - 1 if len(op) else len(seg) - 1)
        out.append(dict(a=int(a), c=int(c), rel=int(max(rel, c)), b=int(b), w=round(w, 2)))
    return out


def _seat_frame(s, cy):
    lo = max(cy["c"], cy["rel"] - 3 * FPS)
    return lo + int(np.argmin(s["pos"][lo:cy["rel"] + 1, 2]))


# ------------------------------------------------------------------------------------------------ board
def _refine(events, Hb, x0, gate_mm, iters=4):
    from scipy.optimize import least_squares
    x = np.array(x0, float)
    n_in = 0
    for _ in range(iters):
        Hw = Hb @ _rot(x[2]).T + x[:2]
        A = []
        for p, dv in events:
            dist = np.linalg.norm(p[None] - Hw, axis=1)
            k = int(np.argmin(dist))
            if dist[k] < gate_mm:
                A.append((p, dv, k))
        n_in = len(A)
        if n_in < 3:
            break
        Pp = np.array([a[0] for a in A])
        D = np.array([a[1] for a in A])
        K = np.array([a[2] for a in A])

        def fun(x):
            v = Pp - (Hb[K] @ _rot(x[2]).T + x[:2])
            n = np.c_[-D[:, 1], D[:, 0]]
            return np.r_[(v * n).sum(1) / 1.0, (v * D).sum(1) / 6.0]

        x = least_squares(fun, x, loss="cauchy", f_scale=2.0, x_scale=[10, 10, 0.01]).x
    return x, n_in


def _ransac(events, Hb, tol=8.0):
    """Label-free 2D rigid start from event and hole pairs (used when the mocap start explains few events)."""
    pts = np.array([p for p, _ in events])
    if len(pts) < 3:
        return None, 0
    best = (0, None)
    nh = len(Hb)
    dH = np.linalg.norm(Hb[:, None] - Hb[None], axis=2)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dp = np.linalg.norm(pts[j] - pts[i])
            if dp < 25:
                continue
            for a in range(nh):
                for b in range(nh):
                    if a == b or abs(dH[a, b] - dp) > 12:
                        continue
                    yaw = np.arctan2(*(pts[j] - pts[i])[::-1]) - np.arctan2(*(Hb[b] - Hb[a])[::-1])
                    t = pts[i] - _rot(yaw) @ Hb[a]
                    Hw = Hb @ _rot(yaw).T + t
                    n = int((np.linalg.norm(pts[:, None] - Hw[None], axis=2).min(1) < tol).sum())
                    if n > best[0]:
                        best = (n, np.r_[t, yaw])
    return best[1], best[0]


def _board(ep: int, s: dict, cys: list, layout: dict, P: dict) -> dict:
    names, Hb = _holes(layout)
    ev = []
    for cy in cys:
        if cy["w"] < P["nothing_mm"]:
            continue
        for fr in (cy["c"], _seat_frame(s, cy)):
            ev.append((s["pos"][fr, :2].copy(), s["pad"][fr].copy()))
    mc = _mocap(ep)
    src = "mocap+contacts"
    x0 = np.r_[mc[0], mc[1], mc[2]] if mc and mc[4] else None
    x, n_in = (_refine(ev, Hb, x0, P["assign_mm"]) if x0 is not None else (None, 0))
    if x is None or n_in < max(3, 0.25 * len(ev)):
        xr, nr = _ransac(ev, Hb)
        if xr is not None and nr > n_in:
            x, n_in = _refine(ev, Hb, xr, P["assign_mm"])
            src = "ransac+contacts"
    if x is None:
        x, src = (x0, "mocap only") if x0 is not None else (np.zeros(3), "none")
    Hw = Hb @ _rot(x[2]).T + x[:2]
    shift = None if x0 is None else float(np.median(np.linalg.norm(Hw - (Hb @ _rot(x0[2]).T + x0[:2]), axis=1)))
    return dict(names=names, Hw=Hw, x=x, src=src, n_events=len(ev), n_assigned=n_in,
                mocap_shift_mm=None if shift is None else round(shift, 1),
                z_mocap_mm=None if mc is None else mc[3])


def _rank(board: dict, p, pad, n=3) -> list[dict]:
    """Holes ranked by xy distance from TCP p, with across-pad / along-pad components (mm)."""
    v = p[None] - board["Hw"]
    d = np.linalg.norm(v, axis=1)
    nrm = np.array([-pad[1], pad[0]])
    order = np.argsort(d)[:n]
    return [dict(part=board["names"][k], xy_mm=round(float(d[k]), 1), across_pad_mm=round(float(abs(v[k] @ nrm)), 1),
                 along_pad_mm=round(float(v[k] @ pad), 1)) for k in order]


# ------------------------------------------------------------------------------------------------ analysis
def _lift_onset(s, i0, i1, P):
    """First frame in [i0, i1) where z rises more than lift_fast_mm within 0.5 s (None if none)."""
    z = s["pos"][:, 2]
    h = FPS // 2
    for i in range(i0, max(i0, i1 - h)):
        if z[i + h] - z[i] > P["lift_fast_mm"]:
            # walk back to where the rise starts (z within 1 mm of the local min)
            j = i
            while j > i0 and z[j - 1] < z[j] - 0.05:
                j -= 1
            return j
    return None


def _attempts(s, i0, i1, hole, P):
    """Insert attempts while holding in [i0, i1): visits near `hole` that end in a retreat. Each attempt: start (first
    frame near), low (deepest frame near the hole), retreat (frame where z first exceeds low + retreat_mm, or where
    the TCP leaves 2 x near_target_mm sideways; None if still going at i1) and rise (where that retreat's climb
    starts). After a retreat, a new attempt needs the TCP back down near the previous working height."""
    pos = s["pos"]
    z = pos[:, 2]
    dist = np.linalg.norm(pos[i0:i1, :2] - hole, axis=1)
    near = dist < P["near_target_mm"]
    att, cur, floor = [], None, None
    for k in range(i1 - i0):
        i = i0 + k
        if cur is None:
            if near[k] and (floor is None or z[i] < floor + P["retreat_mm"] / 2):
                cur = dict(start=i, low=i)
            continue
        if near[k] and z[i] < z[cur["low"]]:
            cur["low"] = i
        if z[i] > z[cur["low"]] + P["retreat_mm"] or dist[k] > 2 * P["near_target_mm"]:
            j = i
            while j > cur["low"] and z[j - 1] > z[cur["low"]] + 2.0:
                j -= 1
            cur.update(retreat=i, rise=j)
            floor = z[cur["low"]]
            att.append(cur)
            cur = None
    if cur is not None:
        cur.update(retreat=None, rise=None)
        att.append(cur)
    return att


def _revisit(s, hole, rel, b, nxt_a, P):
    """After the release: after rising >= 15 mm above the release height, does the open hand come back down by more
    than revisit_mm from its running max within revisit_rad_mm (xy) of the hole, before the next close?"""
    pos = s["pos"]
    z = pos[b:nxt_a, 2]
    if len(z) < 3:
        return None
    up = np.flatnonzero(z > pos[rel, 2] + 15.0)
    if not len(up):
        return None
    i0 = up[0]
    zz = z[i0:]
    near = np.linalg.norm(pos[b + i0:nxt_a, :2] - hole, axis=1) < P["revisit_rad_mm"]
    # the return must come back down to within revisit_top_mm of the release height (a pass high above is no return)
    low_enough = zz < pos[rel, 2] + P["revisit_top_mm"]
    drop = np.where(near & low_enough, np.maximum.accumulate(zz) - zz, 0.0)
    j = int(np.argmax(drop))
    if drop[j] <= P["revisit_mm"]:
        return None
    lo = b + i0 + j
    # the return ends when the hand rises again (or at the next close)
    k = lo
    while k + 1 < nxt_a and pos[k + 1, 2] < pos[lo, 2] + 5.0:
        k += 1
    return dict(low=int(lo), end=int(k), drop_mm=round(float(drop[j]), 1),
                peak_F=round(float(s["Fm"][b + i0:lo + 1].max()), 1),
                xy_mm=round(float(np.linalg.norm(pos[lo, :2] - hole)), 1))


def _open_dips(s, i0, i1, P):
    """Open-hand dips in [i0, i1): the hand comes down >= dip_mm and goes back up >= dip_mm without closing.
    Returns the frames of the dip bottoms."""
    z = s["pos"][i0:i1, 2]
    if len(z) < FPS:
        return []
    out, k = [], 0
    hi = z[0]
    lo_i = None
    while k < len(z):
        if lo_i is None:
            hi = max(hi, z[k])
            if z[k] < hi - P["dip_mm"]:
                lo_i = k
        else:
            if z[k] < z[lo_i]:
                lo_i = k
            elif z[k] > z[lo_i] + P["dip_mm"]:
                out.append(i0 + lo_i)
                hi, lo_i = z[k], None
        k += 1
    return out


def _family(part):
    return "gear" if "gear" in part else "peg" if "peg" in part else "nut" if part == "bolt 4" else "connector"


def analyse(ep: int, layout: dict | None = None, params: dict | None = None) -> dict:
    """Everything the two tools report, for the whole episode (label-free)."""
    P = dict(PARAMS, **(params or {}))
    s = _sig(ep)
    lay = layout or _train_layout()
    cys = _cycles(s, P)
    bd = _board(ep, s, cys, lay, P)
    names, Hw = bd["names"], bd["Hw"]
    pos = s["pos"]
    n = len(s["t"])

    def at(fr):
        r = _rank(bd, pos[fr, :2], s["pad"][fr])
        return r, (r[0]["part"] if r[0]["xy_mm"] < P["at_hole_mm"] else None)

    out = []
    for ci, cy in enumerate(cys):
        nxt_a = cys[ci + 1]["a"] if ci + 1 < len(cys) else n
        held_F = float(s["Fm"][cy["c"]:cy["rel"] + 1].max())
        # closed on nothing: < 1 mm and short with no contact (a USB plug can be held at 0.9 mm for 30 s of insertion)
        nothing = cy["w"] < P["nothing_mm"] and ((cy["b"] - cy["a"]) / FPS < P["nothing_max_s"] and held_F < 5.0)
        g_rank, g_hole = at(cy["c"])
        lift = _lift_onset(s, cy["c"], cy["rel"] + 1, P)
        z_rise = float(pos[cy["c"]:cy["rel"] + 1, 2].max() - pos[cy["c"], 2])
        carried = None
        if g_hole is not None:
            carried = float(np.linalg.norm(pos[cy["rel"], :2] - Hw[names.index(g_hole)]))
        freed = (not nothing) and (carried is None or carried > P["carried_mm"])
        r = dict(i=ci, close=cy["a"], closed=cy["c"], release=cy["rel"], open=cy["b"], hold_mm=cy["w"],
                 nothing=nothing, grasp_rank=g_rank, grasp_hole=g_hole, lift=lift, z_rise_mm=round(z_rise, 1),
                 carried_mm=None if carried is None else round(carried, 1), freed=freed, attempts=[], release_rank=None,
                 release_hole=None, revisit=None, next_grasp_same=False, grasp_xy=pos[cy["c"], :2].round(1).tolist(),
                 release_xy=pos[cy["rel"], :2].round(1).tolist(),
                 grip_guess=_grip_guess(cy["w"], P.get("grip_table")) if not nothing else None)
        if not nothing:
            # target hole while held: where the TCP gets lowest near any hole after the lift (or at the grasp)
            h0 = lift if lift is not None else cy["c"]
            seat = _seat_frame(s, cy)
            r["release_rank"], r["release_hole"] = at(seat)
            # holes the held part visited: lowest z and closest approach to each
            visits = {}
            for i in range(h0, cy["rel"] + 1, 3):
                d = np.linalg.norm(Hw - pos[i, :2], axis=1)
                for k in np.flatnonzero(d < P["near_target_mm"]):
                    zmin, dmin = visits.get(int(k), (np.inf, np.inf))
                    visits[int(k)] = (min(zmin, pos[i, 2]), min(dmin, d[k]))
            target, r["target_why"] = r["release_hole"], "released at the hole"
            if target is None and visits:
                # failed attempts only: the visited hole that is lowest, closest and fits the hold width best
                z0 = min(v[0] for v in visits.values())
                cost = {k: (v[0] - z0) / 15.0 + v[1] / 15.0 + _grip_cost(cy["w"], names[k], P.get("grip_table"))
                        for k, v in visits.items()}
                kbest = min(cost, key=cost.get)
                target = names[kbest]
                r["target_why"] = "visited while held (lowest, closest, width fits): " + ", ".join(
                    f"{names[k]} {c:.1f}" for k, c in sorted(cost.items(), key=lambda kv: kv[1])[:3])
            if target is not None and target != g_hole:
                hk = Hw[names.index(target)]
                att = _attempts(s, h0, cy["rel"] + 1, hk, P)
                for a in att:
                    a["Fpeak"] = round(float(s["Fm"][a["start"]:(a["retreat"] or cy["rel"]) + 1].max()), 1)
                    a["Fz_mean"] = round(float(s["F"][a["start"]:(a["retreat"] or cy["rel"]) + 1, 2].mean()), 1)
                    a["z_low_mm"] = round(float(pos[a["low"], 2]), 1)
                r["attempts"] = att
                r["target"] = target
            if r["release_hole"] is not None:
                r["revisit"] = _revisit(s, Hw[names.index(r["release_hole"])], cy["rel"], cy["b"], nxt_a, P)
                if ci + 1 < len(cys):
                    nc = cys[ci + 1]
                    dn = float(np.linalg.norm(pos[nc["c"], :2] - Hw[names.index(r["release_hole"])]))
                    r["next_grasp_same"] = dn < P["regrasp_mm"]
                    r["next_grasp_lift_mm"] = round(float(pos[nc["c"]:nc["rel"] + 1, 2].max() - pos[nc["c"], 2]), 1)
        # aborted approaches before this close: open-hand dips at a hole (not the hole just released) or low dips
        # off the board, ending at least 1 s before this close starts
        prev_b = cys[ci - 1]["b"] if ci else 0
        prev_hole = out[-1]["release_hole"] if out else None
        r["aborted"] = []
        for lo in _open_dips(s, prev_b, max(prev_b, cy["a"] - FPS), P):
            rk = _rank(bd, pos[lo, :2], s["pad"][lo], n=1)[0]
            if prev_hole is not None and np.linalg.norm(pos[lo, :2] - Hw[names.index(prev_hole)]) < P["revisit_rad_mm"]:
                continue  # an open-hand press or return after a release
            hole = rk["part"] if rk["xy_mm"] < P["dip_hole_mm"] else None
            if hole is None and pos[lo, 2] > pos[cy["c"], 2] + 30.0:
                continue  # not down at working height
            k = lo
            while k + 1 < cy["a"] and pos[k + 1, 2] < pos[lo, 2] + P["dip_mm"]:
                k += 1
            r["aborted"].append(dict(low=int(lo), up=int(k), hole=hole, z_mm=round(float(pos[lo, 2]), 1)))
        out.append(r)
    return dict(episode=int(ep), duration=n / FPS, board=bd, cycles=out, sig=s, params=P)


# ------------------------------------------------------------------------------------------------ draft
def _insert_outcome(r: dict, fam: str, P: dict) -> tuple[bool, str]:
    """Success of an Insert whose part was released at its hole, from what happens next."""
    if r["next_grasp_same"] and (fam == "connector" or r.get("next_grasp_lift_mm", 0) > P["free_lift_mm"]):
        return False, "regrasped at the same hole" + ("" if fam == "connector" else " and lifted")
    if r["revisit"] and fam == "peg":
        return False, "open-hand return to the hole after release (peg not seated)"
    return True, "released at the hole"


def _name(cys: list, ci: int) -> tuple[str | None, str]:
    """(part, basis) of a cycle: its hole, else the same object in an adjacent cycle, else the hold-width guess."""
    r = cys[ci]
    for key, basis in (("grasp_hole", "grasped at its hole"), ("release_hole", "released at its hole"),
                       ("target", "holes visited while held, tie broken by hold width")):
        if r.get(key):
            return r[key], basis
    if r["nothing"]:
        return None, "closed on nothing"
    for cj in (ci + 1, ci - 1):
        if 0 <= cj < len(cys) and not cys[cj]["nothing"]:
            o = cys[cj]
            a, b = (r["release_xy"], o["grasp_xy"]) if cj > ci else (r["grasp_xy"], o["release_xy"])
            if np.hypot(a[0] - b[0], a[1] - b[1]) < 30.0 and abs(o["hold_mm"] - r["hold_mm"]) < 2.5:
                nm = o.get("grasp_hole") or o.get("release_hole") or o.get("target")
                if nm:
                    return nm, f"same object as cycle {cj} (grasped where this one released, same width)"
    if r["grip_guess"]:
        return r["grip_guess"][0]["part"], "hold width only (weak: confirm in the hand camera)"
    return None, "unknown"


def _verdicts(A: dict) -> list[dict]:
    """Turn the analysis into dataset-style segments: start, end, task, success and the rule that decided it."""
    P = A["params"]
    s, cys = A["sig"], A["cycles"]
    n = len(s["t"])
    segs = []
    last_insert = {}           # hole -> index in segs of the Insert that released a part there
    prev = (None, None)        # (kind of the previous cycle: insert / remove / other, its hole)

    def name(ci):
        return _name(cys, ci)[0]

    def add(end, verb, part, ok, rule):
        segs.append(dict(end=int(min(n - 1, end)), verb=verb, part=part, success=ok, rule=rule))
        return len(segs) - 1

    for ci, r in enumerate(cys):
        nxt = cys[ci + 1] if ci + 1 < len(cys) else None
        open_mid = (r["release"] + r["open"]) // 2
        rel_end = open_mid + int(P["release_end_s"] * FPS)
        rv = r["revisit"]
        # gears and connectors: an open-hand press after the release belongs to the same Insert
        press_end = rv["end"] + int(P["release_end_s"] * FPS) if rv else rel_end
        gh = r["grasp_hole"]
        if prev[0] == "insert" and gh is None and not r["nothing"]:
            hp = A["board"]["Hw"][A["board"]["names"].index(prev[1])]
            if np.hypot(r["grasp_xy"][0] - hp[0], r["grasp_xy"][1] - hp[1]) < P["regrasp_mm"]:
                gh = prev[1]  # re-taking the part just released next to its hole
        # open-hand dips with no close (r["aborted"]) are reported, not drafted: on held-out train recordings they
        # caught no failure and added 4 false segments (read-footage Lessons 2026-10-25)
        regrasp = gh is not None and prev == ("insert", gh) and gh in last_insert
        if r["nothing"]:
            if regrasp and last_insert[gh] == len(segs) - 1:
                segs[-1]["end"] = rel_end  # a missed regrasp while inserting: same Insert
                continue
            part = gh or (nxt is not None and name(ci + 1)) or "?"
            add(open_mid + int(P["nothing_end_s"] * FPS), "Remove" if gh else "Pick", part, False, "closed on nothing")
            continue
        fam = _family(name(ci) or "")
        if regrasp:
            j = last_insert[gh]
            taken_out = (r["carried_mm"] or 0) > P["carried_mm"] or r["z_rise_mm"] > P["free_lift_mm"]
            if taken_out:
                segs[j].update(success=False, rule="regrasped and lifted out of the hole")
            elif fam == "connector":
                segs[j].update(success=False, rule="regrasped in place (connector not seated)")
                ok, rule = _insert_outcome(r, fam, P)
                last_insert[gh] = add(press_end, "Insert", gh, ok, "regrasp in place; " + rule)
                prev = ("insert", gh)
                continue
            else:
                ok, rule = _insert_outcome(r, fam, P)
                end = press_end if fam != "peg" else rel_end
                if j == len(segs) - 1:  # gears, nuts, pegs: one Insert continues through a regrasp in place
                    segs[j].update(end=end, success=ok, rule="regrasp in place; " + rule)
                else:
                    last_insert[gh] = add(end, "Insert", gh, ok, "regrasp in place; " + rule)
                prev = ("insert", gh)
                continue
        first = "Pick" if (gh is None or regrasp) else "Remove"
        part = name(ci) or "?"
        if first == "Remove" and not r["freed"]:
            add(rel_end, "Remove", part, False, "released near its hole (not carried away)")
            prev = ("remove", gh)
            continue
        lift = r["lift"] if r["lift"] is not None else r["closed"]
        if first == "Remove":
            last_insert.pop(gh, None)
        add(lift + int(P["first_end_s"] * FPS), first, part, True,
            "regrasped and lifted" if regrasp else "lifted while holding")
        for k, a in enumerate(r["attempts"]):
            last = k == len(r["attempts"]) - 1
            if a["retreat"] is not None and not (last and r["release_hole"] == r.get("target")):
                add(a["rise"] + int(P["retreat_end_s"] * FPS), "Insert", part, False,
                    "retreat while holding (attempt %d)" % (k + 1))
        rh = r["release_hole"]
        if rh is not None and (rh != gh or regrasp):
            ok, rule = _insert_outcome(r, fam, P)
            j = last_insert.get(rh)
            if j is not None and segs[j]["success"]:
                # the same part goes into the same hole again later (no Remove in between): the first one failed
                segs[j].update(success=False, rule="inserted again later (" + segs[j]["rule"] + ")")
            unseated = rv and fam == "peg" and not ok and not r["next_grasp_same"]
            end = open_mid + int(P["unseated_end_s"] * FPS) if unseated else press_end if fam != "peg" else rel_end
            last_insert[rh] = add(end, "Insert", part, ok, rule)
            prev = ("insert", rh)
            if unseated:
                add(rv["end"], "Insert", part, True, "open-hand press/nudge")
        else:
            add(rel_end, "Place", part, True, "released off the board")
            prev = ("other", None)
    # starts: contiguous, the first at the first motion
    v = np.linalg.norm(np.gradient(s["pos"], axis=0), axis=1) * FPS
    mv = np.flatnonzero(v > 10.0)
    t0 = int(mv[0]) if len(mv) else 0
    out, start = [], t0
    for g in segs:
        e = max(g["end"], start + 1)
        out.append(dict(start=round(start / FPS, 2), end=round(e / FPS, 2),
                        task=f"{g['verb']} {g['part']}.", success=g["success"], rule=g["rule"]))
        start = e
    return out


def draft(ep: int, layout: dict | None = None, params: dict | None = None) -> list[dict]:
    return _verdicts(analyse(ep, layout, params))


# ------------------------------------------------------------------------------------------------ tools
# Train precision of each drafting rule: of the segments the rule decided that matched a dataset segment (IoU >= 0.5),
# how many agreed on success, over the 89 train recordings (60 tuning + 29 held-out; layout and widths without the held-out
# ones). Source: scratch evaluation 2026-10-25, read-footage Lessons.
RULE_TRAIN = {
    "lifted while holding": "1151/1162 success right",
    "released off the board": "555/567 success right",
    "released at the hole": "493/506 success right",
    "open-hand press/nudge": "38/38 success right",
    "regrasp in place": "22/25 success right (gears, nuts, pegs: one Insert through the regrasp)",
    "regrasped and lifted": "7/8 success right",
    "retreat while holding": "76/76 failures right",
    "closed on nothing": "61/69 failures right",
    "released near its hole": "45/51 failures right",
    "open-hand return to the hole after release": "39/51 failures right (lowest: confirm in the hand camera)",
    "regrasped and lifted out of the hole": "16/19 failures right",
    "regrasped in place": "3/3 failures right (connectors)",
    "inserted again later": "3/4 failures right",
    "regrasped at the same hole": "0/1 failures right (too rare to trust: confirm)",
}


def _rule_note(rule: str) -> str | None:
    for k, v in RULE_TRAIN.items():
        if rule.startswith(k) or rule.split("; ")[-1].startswith(k):
            return v
    return None


def _s(i):
    return None if i is None else round(int(i) / FPS, 2)


def _in(r, t0, t1):
    return (r["open"] / FPS >= t0) and (r["close"] / FPS <= t1)


def _board_summary(bd):
    return dict(source=bd["src"], events=bd["n_events"], events_on_holes=bd["n_assigned"],
                mocap_shift_mm=bd["mocap_shift_mm"],
                note="Label-free: mocap NIST_Board1 pose refined so this recording's own grasp and seat points sit on "
                     "the train hole layout. Few events on holes (< 3) means the mocap pose is used as is.")


def _where(hole, rank):
    """Compact location: the hole it is at (with distances and the runner-up), or off the board."""
    if hole:
        return dict(at_hole=hole, xy_mm=rank[0]["xy_mm"], across_pad_mm=rank[0]["across_pad_mm"],
                    along_pad_mm=rank[0]["along_pad_mm"], runner_up=f"{rank[1]['part']} {rank[1]['xy_mm']} mm")
    return dict(at_hole=None, nearest=f"{rank[0]['part']} {rank[0]['xy_mm']} mm")


def _window(A, t0, t1):
    t0 = 0.0 if t0 is None else float(t0)
    t1 = A["duration"] if t1 is None else float(t1)
    return t0, t1


@tool(
    "identify_part_by_hole",
    "Name the part of every gripper cycle in a window from WHERE it is grasped or released: the TCP at the grasp "
    "and at the seat (deepest point in the 3 s before release) mapped onto the board's hole layout (fitted on train "
    "recordings; this recording's board pose comes from mocap refined by its own contacts, label-free). A grasp or "
    "release within 25 mm of a hole is AT that hole and names the part (train: 99% right on successful inserts and "
    "removes; table grasps sit >= 90 mm from every hole). Failed attempts are named from the holes visited while held, "
    "with the hold width as tie-break; table-only cycles from the adjacent cycle that handles the same object, else "
    "from hold width alone (weak). Reads signals only, never labels.",
    {"type": "object", "required": ["episode"], "properties": {
        "episode": {"type": "integer"},
        "t0": {"type": "number", "description": "window start s (default 0)"},
        "t1": {"type": "number", "description": "window end s (default the episode end)"}}},
)
def identify_part_by_hole(episode: int, t0: float | None = None, t1: float | None = None) -> dict:
    A = analyse(int(episode))
    t0, t1 = _window(A, t0, t1)
    cys = A["cycles"]
    rows = []
    for ci, r in enumerate(cys):
        if not _in(r, t0, t1):
            continue
        part, basis = _name(cys, ci)
        row = dict(cycle=ci, close_s=_s(r["close"]), closed_s=_s(r["closed"]), release_s=_s(r["release"]),
                   open_s=_s(r["open"]), hold_mm=r["hold_mm"], closed_on_nothing=r["nothing"],
                   grasp=_where(r["grasp_hole"], r["grasp_rank"]),
                   seat=None if r["release_rank"] is None else _where(r["release_hole"], r["release_rank"]),
                   part=part, basis=basis)
        if r.get("target") and not r["release_hole"]:
            row["attempt_target"] = dict(part=r["target"], why=r.get("target_why"))
        if part and part in GRIP_MM and not r["nothing"]:
            med, _, p10, p90, _ = GRIP_MM[part]
            c = _grip_cost(r["hold_mm"], part)
            row["width_check"] = dict(train_width_mm=dict(median=med, p10=p10, p90=p90), cost=round(c, 1),
                                      agrees=c < 4.0)
            if c >= 4.0:
                row["width_check"]["width_suggests"] = r["grip_guess"]
        rows.append(row)
    return dict(episode=int(episode), window=[round(t0, 2), round(t1, 2)], board=_board_summary(A["board"]),
                cycles=rows,
                how_to_read="Trust 'grasped/released at its hole' over the hand camera and over hold width: at-hole "
                            "names matched the train labels on 1186 of 1194 cycles, and on 13 of the 15 where the width "
                            "disagreed (cost >= 4). Where hole and width agree against a label, suspect the label "
                            "(3 square-peg cases on train). 'hold width only' must be confirmed in the hand camera.")


@tool(
    "end_state_check",
    "Judge every gripper cycle in a window from end-state physics, label-free, and draft dataset-style segments. Per "
    "cycle: where the part was grasped and released (at a hole or off the board), lift onset, how far a removed part "
    "was carried, insert attempts while held (dips near the target hole, each ended by a retreat = one failed Insert in "
    "the dataset's convention), what follows a release at a hole (open-hand return or press, regrasp at the same hole "
    "and whether it lifts the part out, the same part inserted again later), and open-hand dips with no close. The "
    "draft gives start/end/task/success per segment with the rule that decided it and that rule's train precision. "
    "Train (89 recordings, 29 held out): segment F1 0.96, success balanced accuracy 0.89, part accuracy 0.96 on the "
    "held-out ones. Reads signals only, never labels.",
    {"type": "object", "required": ["episode"], "properties": {
        "episode": {"type": "integer"},
        "t0": {"type": "number", "description": "window start s (default 0)"},
        "t1": {"type": "number", "description": "window end s (default the episode end)"},
        "draft": {"type": "boolean", "description": "include the drafted segments (default true)"}}},
)
def end_state_check(episode: int, t0: float | None = None, t1: float | None = None, draft: bool = True) -> dict:
    A = analyse(int(episode))
    t0, t1 = _window(A, t0, t1)
    cys = A["cycles"]
    pos = A["sig"]["pos"]
    rows = []
    for ci, r in enumerate(cys):
        if not _in(r, t0, t1):
            continue
        part, basis = _name(cys, ci)
        row = dict(cycle=ci, close_s=_s(r["close"]), release_s=_s(r["release"]), open_s=_s(r["open"]),
                   part=part, basis=basis, hold_mm=r["hold_mm"], closed_on_nothing=r["nothing"],
                   grasp="at hole " + r["grasp_hole"] if r["grasp_hole"] else "off the board",
                   lift_onset_s=_s(r["lift"]), z_rise_mm=r["z_rise_mm"])
        if r["grasp_hole"]:
            row["carried_from_hole_mm"] = r["carried_mm"]
            row["carried_away"] = r["freed"]
        if not r["nothing"]:
            row["release"] = "at hole " + r["release_hole"] if r["release_hole"] else "off the board"
            row["release_z_mm"] = round(float(pos[r["release"], 2]), 1)
        if r["attempts"]:
            row["attempts"] = [dict(start_s=_s(a["start"]), low_s=_s(a["low"]), climb_s=_s(a.get("rise")),
                                    peak_N=a["Fpeak"], Fz_mean_N=a["Fz_mean"], z_low_mm=a["z_low_mm"])
                               for a in r["attempts"]]
            row["attempt_target"] = r.get("target")
        if r["release_hole"]:
            rv = r["revisit"]
            row["after_release"] = dict(
                open_hand_return=None if not rv else dict(low_s=_s(rv["low"]), end_s=_s(rv["end"]), drop_mm=rv["drop_mm"],
                                                          peak_N=rv["peak_F"], xy_from_hole_mm=rv["xy_mm"]),
                next_grasp_at_same_hole=r["next_grasp_same"], next_grasp_lift_mm=r.get("next_grasp_lift_mm"))
        if r["aborted"]:
            row["open_hand_dips_before"] = [dict(low_s=_s(d["low"]), up_s=_s(d["up"]), at_hole=d["hole"], z_mm=d["z_mm"])
                                            for d in r["aborted"]]
        rows.append(row)
    out = dict(episode=int(episode), window=[round(t0, 2), round(t1, 2)], board=_board_summary(A["board"]),
               cycles=rows)
    if draft:
        segs = _verdicts(A)
        out["draft"] = [dict(g, rule_train=_rule_note(g["rule"])) for g in segs if g["end"] > t0 and g["start"] < t1]
        out["draft_note"] = ("Dataset conventions: Pick/Remove ends ~1 s after the lift onset; Insert/Place ends ~0.35 s "
                             "after the middle of the opening; each retreat while holding ends a failed Insert; a peg "
                             "released and then touched again by the open hand = failed Insert + a new successful Insert "
                             "(the nudge); a gear/nut regrasped in place stays one Insert; an open-hand press on a gear "
                             "or connector belongs to its Insert. Idle before the first motion is not drafted.")
    return out
