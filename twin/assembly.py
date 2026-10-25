"""Assemble a full scene: base.xml (FR3 + hand + cameras) + the task board + parts -> one MJCF file.

    build_scene(parts=["round peg 1"], board_pose={"recording": "2025-01-09-15-27-49"})
    check_scene(xml)    compile, settle under gravity, penetrations, part drift, predicates, dt/2 re-run
    render_scene(xml)   PNGs from the scene cameras

Outputs go to runs/scene/<name>/: scene.xml (self-contained apart from the Menagerie meshes, which
are referenced by relative path), params.json (every parameter used, with sigma and source).

Board pose = pose of the board frame (mocap NIST_Board1 frame) in the world (FR3 base) frame:
  {"pos": [x,y,z], "quat": [w,x,y,z]} | {"pos": .., "quat_xyzw": [..]} | {"recording": "<timestamp>"}
The last reads the recording's raw mocap pose; world-frame calibration (twin/calib, twin/world.py)
may supply a better one.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np

from twin import BASE_XML, REPO_DIR, SCENE_DIR
from twin.board import CONTACT, add_board, board_spec, seated_body_pose
from twin.layout import POSES
from twin.parts import add_part_body, load_layout, part_spec, resolve

OUT_ROOT = REPO_DIR / "runs" / "scene"
DEFAULT_RECORDING = "2025-01-09-15-27-49"  # episode 1 (train split), first local episode with inserts
# base.xml runs at 2 ms. With 0.05 mm clearances, 2 ms lets sub-0.03 mm rim overlaps push through (probe,
# 2026-10-24); 1 ms and 0.5 ms agree. Scenes with board parts default to 1 ms.
INSERT_TIMESTEP = 0.001


def resolve_board_pose(board_pose: dict | None) -> dict:
    bp = dict(board_pose or {"recording": DEFAULT_RECORDING})
    if "recording" in bp:
        f = POSES / f"{bp['recording']}_poses.json"
        b = json.loads(f.read_text())["NIST_Board1"]
        x, y, z, w = b[1]
        return dict(pos=list(b[0]), quat=[w, x, y, z], source=f"mocap NIST_Board1 in {f.name}")
    if "quat_xyzw" in bp:
        x, y, z, w = bp["quat_xyzw"]
        bp["quat"] = [w, x, y, z]
    q = np.asarray(bp.get("quat", [1, 0, 0, 0]), float)
    return dict(pos=list(map(float, bp["pos"])), quat=(q / np.linalg.norm(q)).tolist(),
                source=bp.get("source", "given"))


def _compose(pos_a, quat_a, pos_b, quat_b):
    """World pose of b given a (world) and b (in a)."""
    import mujoco
    p = np.zeros(3)
    q = np.zeros(4)
    mujoco.mju_mulPose(p, q, np.asarray(pos_a, float), np.asarray(quat_a, float),
                       np.asarray(pos_b, float), np.asarray(quat_b, float))
    return p.tolist(), q.tolist()


def _fix_xml(xml: str, out_dir: Path) -> str:
    # Menagerie mesh paths in base.xml are relative to twin/scene/; re-root them at the output dir.
    def rel(mo):
        p = Path(mo.group(2))
        absp = p if p.is_absolute() else (SCENE_DIR / p).resolve()
        return f'{mo.group(1)}"{os.path.relpath(absp, out_dir)}"'
    xml = re.sub(r'(<mesh [^>]*file=)"([^"]+)"', rel, xml)
    # MuJoCo 3.17 to_xml drops mass="0" density="0" from fr3's collision default (see build_base.py)
    xml = re.sub(r'(<default class="collision">\s*<geom type="mesh" group="3")/>',
                 r'\1 mass="0" density="0"/>', xml)
    return xml


def build_scene(parts=None, board_pose: dict | None = None, part_init: str | dict = "seated",
                sample_seed: int | None = None, timestep: float | None = None, name: str | None = None,
                out_root: Path = OUT_ROOT) -> dict:
    import mujoco

    names = resolve(parts)
    layout = load_layout()
    board = board_spec(layout)
    specs = [part_spec(n, layout) for n in names]
    if sample_seed is not None:
        rng = np.random.default_rng(sample_seed)
        board = board.sampled(rng)
        specs = [p.sampled(rng) for p in specs]
    pose = resolve_board_pose(board_pose)

    spec = mujoco.MjSpec.from_file(str(BASE_XML))
    # The board lies on the table: a peg cannot stop below the table top. Lowest plate-top corner vs table.
    warnings = []
    table = spec.geom("table_top")
    table_top = float(table.pos[2] + table.size[2]) if table is not None else None
    if table_top is not None:
        (x0, y0), (sx, sy) = board.plate_xy_min, board.plate_size
        corners = [[x, y, board.v("surface_z")] for x in (x0, x0 + sx) for y in (y0, y0 + sy)]
        zmin = min(_compose(pose["pos"], pose["quat"], c, [1, 0, 0, 0])[0][2] for c in corners)
        room = zmin - table_top
        if board.v("seat_depth") > room + 1e-4:
            if sample_seed is not None:
                from dataclasses import replace as _r
                board.params["seat_depth"] = _r(board.params["seat_depth"], value=max(room, 0.002))
                warnings.append(f"sampled seat_depth clamped to {1000 * max(room, 0.002):.2f} mm: plate top is "
                                f"only {1000 * room:.2f} mm above the table at this board pose")
            else:
                warnings.append(f"seat_depth {1000 * board.v('seat_depth'):.2f} mm > plate-top height above the "
                                f"table {1000 * room:.2f} mm: seated parts will hit the table (check board pose z, "
                                f"surface_z, seat_depth)")
    placeholder = spec.body("task_board")
    if placeholder is not None:
        spec.delete(placeholder)
    spec.option.timestep = float(timestep or INSERT_TIMESTEP)
    mat = "board_mat" if spec.material("board_mat") is not None else None
    _, info = add_board(spec, spec.worldbody, specs, board, pose["pos"], pose["quat"], material=mat)

    inits = {}
    for p in specs:
        mode = part_init.get(p.name, part_init.get(p.slug, "seated")) if isinstance(part_init, dict) else part_init
        if mode == "none":
            continue
        if isinstance(mode, dict):
            wpos, wquat = mode["pos"], mode.get("quat", [1, 0, 0, 0])
        else:
            depth = None if mode == "seated" else -0.010 if mode == "above" else None
            if mode not in ("seated", "above"):
                raise ValueError(f"part_init must be seated | above | none | {{pos, quat}}, got {mode!r}")
            bpos, bquat = seated_body_pose(p, board, depth=depth)
            wpos, wquat = _compose(pose["pos"], pose["quat"], bpos, bquat)
        add_part_body(spec, spec.worldbody, p, wpos, wquat, free=True, contact=CONTACT)
        inits[p.slug] = mode if isinstance(mode, str) else "custom"

    m0 = spec.compile()
    # keyframe: keep the robot part, append the parts' free-joint poses
    for key in spec.keys:
        q = list(key.qpos)[:9] + m0.qpos0[9:].tolist()
        key.qpos = q
        key.qvel = [0.0] * m0.nv
    ref = spec.compile()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = name or f"{'_'.join(p.slug for p in specs) if len(specs) <= 2 else f'{len(specs)}parts'}_{stamp}"
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / "scene.xml"
    header = (f"<!-- GENERATED by twin/assembly.py build_scene at {stamp}: base.xml + board + {names}. "
              f"Do not edit; rebuild. -->\n")
    xml_path.write_text(header + _fix_xml(spec.to_xml(), out_dir))
    m2 = mujoco.MjModel.from_xml_path(str(xml_path))
    for attr in ("body_mass", "geom_size", "geom_pos", "site_pos", "body_pos", "numeric_data"):
        a, b = getattr(ref, attr), getattr(m2, attr)
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-7):
            raise RuntimeError(f"round-trip mismatch in {attr} for {xml_path}")

    params = dict(built_at=stamp, warnings=warnings, parts=names, board_pose=pose, part_init=inits, sample_seed=sample_seed,
                  timestep=float(m2.opt.timestep), layout_file="twin/assets/board_layout.json",
                  layout_generated_at=layout.get("generated_at"), board=board.describe(), board_info=info,
                  part_specs={p.slug: p.describe() for p in specs})
    (out_dir / "params.json").write_text(json.dumps(params, indent=1, default=float))
    return dict(xml=str(xml_path.relative_to(REPO_DIR)), params=str((out_dir / "params.json").relative_to(REPO_DIR)),
                nbody=m2.nbody, ngeom=m2.ngeom, nq=m2.nq, nu=m2.nu, timestep=float(m2.opt.timestep),
                board_pose=pose, parts={p.slug: inits.get(p.slug, "none") for p in specs},
                holes=info["holes"], n_plate_tiles=info["n_plate_tiles"], warnings=warnings)


# --------------------------------------------------------------------------------------- check
def _load(xml):
    import mujoco
    xml = Path(xml)
    if not xml.is_absolute():
        xml = REPO_DIR / xml
    m = mujoco.MjModel.from_xml_path(str(xml))
    return m, xml


def _reset(m, d):
    import mujoco
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "reassemble_start")
    if k >= 0:
        mujoco.mj_resetDataKeyframe(m, d, k)
    else:
        mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)


def _penetrations(m, d, tol: float) -> list[dict]:
    import mujoco
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist < -tol:
            g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
            g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
            b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1])
            b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2])
            out.append(dict(geoms=[g1, g2], bodies=[b1, b2], depth_mm=round(-1000 * float(c.dist), 4)))
    return sorted(out, key=lambda r: -r["depth_mm"])


def _free_bodies(m):
    import mujoco
    out = {}
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            b = m.jnt_bodyid[j]
            out[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)] = b
    return out


def _settle(m, settle_s: float, pen_tol: float):
    import mujoco
    from twin.predicates import all_states
    d = mujoco.MjData(m)
    _reset(m, d)
    fb = _free_bodies(m)
    x0 = {n: d.xpos[b].copy() for n, b in fb.items()}
    pre = all_states(m, d)
    max_pen, worst = 0.0, []
    nan = False
    steps = int(round(settle_s / m.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qacc).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            nan = True
            break
        if d.ncon:
            dmin = -float(d.contact.dist[: d.ncon].min())
            if dmin > max_pen:
                max_pen, worst = dmin, _penetrations(m, d, pen_tol)
    drift = {n: round(1000 * float(np.linalg.norm(d.xpos[b] - x0[n])), 4) for n, b in fb.items()}
    post = all_states(m, d)
    final_pen = -float(d.contact.dist[: d.ncon].min()) if d.ncon else 0.0
    return d, dict(nan=nan, steps=steps, final_penetration_mm=round(1000 * max(final_pen, 0.0), 4),
                   final_worst_contacts=_penetrations(m, d, pen_tol)[:8],
                   max_penetration_mm=round(1000 * max_pen, 4), worst_contacts=worst[:8],
                   drift_mm=drift, states_before=pre, states_after=post,
                   final_xpos={n: d.xpos[b].tolist() for n, b in fb.items()})


def list_sites(m, d, groups=None) -> list[dict]:
    import mujoco
    out = []
    for i in range(m.nsite):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
        if groups is not None and m.site_group[i] not in groups:
            continue
        R = d.site_xmat[i].reshape(3, 3)
        out.append(dict(name=nm, pos=[round(float(v), 5) for v in d.site_xpos[i]],
                        z_axis=[round(float(v), 4) for v in R[:, 2]], y_axis=[round(float(v), 4) for v in R[:, 1]]))
    return out


def check_scene(xml, settle_s: float = 1.0, halved_timestep: bool = True, penetration_tol_mm: float = 0.1,
                drift_tol_mm: float = 0.5) -> dict:
    import mujoco
    m, path = _load(xml)
    d = mujoco.MjData(m)
    _reset(m, d)
    tol = penetration_tol_mm / 1000
    initial = _penetrations(m, d, tol)
    sites = list_sites(m, d, groups=[4])
    _, run = _settle(m, settle_s, tol)
    res = dict(xml=str(path.relative_to(REPO_DIR)), compiled=True, nbody=m.nbody, ngeom=m.ngeom, nq=m.nq,
               timestep=float(m.opt.timestep), initial_penetrations=initial[:8], settle=run, sites=sites)
    problems = []
    if initial:
        problems.append(f"{len(initial)} initial contacts deeper than {penetration_tol_mm} mm")
    if run["nan"]:
        problems.append("NaN/instability during settle")
    if run["final_penetration_mm"] > penetration_tol_mm:
        problems.append(f"penetration {run['final_penetration_mm']} mm after settling")
    warnings = []
    if run["max_penetration_mm"] > penetration_tol_mm:
        warnings.append(f"transient penetration {run['max_penetration_mm']} mm during settle (impacts; contact "
                        f"timeconst sets it, not the timestep)")
    for n, dr in run["drift_mm"].items():
        was = run["states_before"].get(n, {}).get("seated")
        if was and dr > drift_tol_mm:
            problems.append(f"{n} drifted {dr} mm while seated")
        if was and not run["states_after"].get(n, {}).get("seated"):
            problems.append(f"{n} no longer seated after settling")
    if halved_timestep:
        m.opt.timestep = m.opt.timestep / 2
        _, run2 = _settle(m, settle_s, tol)
        diffs = {n: round(1000 * float(np.linalg.norm(np.subtract(run["final_xpos"][n], run2["final_xpos"][n]))), 4)
                 for n in run["final_xpos"]}
        agree = {n: run["states_after"][n]["seated"] == run2["states_after"][n]["seated"]
                 for n in run["states_after"]}
        res["halved_timestep"] = dict(timestep=float(m.opt.timestep), final_pos_diff_mm=diffs,
                                      seated_agrees=agree, max_penetration_mm=run2["max_penetration_mm"],
                                      final_penetration_mm=run2["final_penetration_mm"],
                                      nan=run2["nan"])
        if run2["nan"] or not all(agree.values()) or any(v > drift_tol_mm for v in diffs.values()):
            problems.append("halved-timestep run disagrees")
    for k in ("final_xpos",):
        run.pop(k, None)
    res["ok"] = not problems
    res["problems"] = problems
    res["warnings"] = warnings
    return res


# -------------------------------------------------------------------------------------- render
def render_scene(xml, cameras=("board_top", "overview", "hama1", "hama2", "hand"), width: int = 640,
                 height: int = 480, show_sites: bool = True, settle_s: float = 0.0, out_dir=None) -> dict:
    os.environ.setdefault("MUJOCO_GL", "glfw")
    import mujoco
    from PIL import Image

    m, path = _load(xml)
    d = mujoco.MjData(m)
    _reset(m, d)
    for _ in range(int(round(settle_s / m.opt.timestep))):
        mujoco.mj_step(m, d)
    out_dir = Path(out_dir) if out_dir else path.parent / "render"
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = mujoco.MjvOption()
    if show_sites:
        opt.sitegroup[4] = 1
    files, imgs = {}, []
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, height)
    with mujoco.Renderer(m, height, width) as r:
        for cam in cameras:
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam) < 0:
                files[cam] = "missing camera"
                continue
            r.update_scene(d, camera=cam, scene_option=opt)
            img = r.render().copy()
            f = out_dir / f"{cam}.png"
            Image.fromarray(img).save(f)
            files[cam] = str(f.relative_to(REPO_DIR))
            imgs.append(img)
    if imgs:
        Image.fromarray(np.concatenate(imgs, axis=1)).save(out_dir / "strip.png")
        files["strip"] = str((out_dir / "strip.png").relative_to(REPO_DIR))
    return dict(xml=str(path.relative_to(REPO_DIR)), images=files, t=float(d.time))
