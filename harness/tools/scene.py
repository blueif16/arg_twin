"""Stage B tools: build the task board and parts, check them, render them, probe the holes.

Skills: playbook/skills/mujoco (how to work with MuJoCo here), playbook/skills/build-twin (what to
build and how to know it is right). Sim code lives in twin/ (layout, parts, board, predicates, probe,
assembly); these wrappers only validate arguments and write outputs under runs/scene/.

Tools:
  scene_parts      catalog: dimensions, clearances, hole positions, every parameter with sigma + source
  board_layout     re-derive hole positions in the board frame from REASSEMBLE insert events
  scene_build      parts + board pose -> MJCF (runs/scene/<name>/scene.xml + params.json)
  scene_check      compile, settle under gravity, penetrations, drift, predicates, dt/2 re-run
  scene_render     PNGs from scene cameras (board_top, overview, hama1, hama2, hand, ...)
  insertion_probe  scripted push of one part into its hole at lateral offsets -> seated / jammed table
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

from harness.registry import REPO, tool

OUT = REPO / "runs" / "scene"
_POSE_SCHEMA = {"type": "object", "description":
                "Board frame (mocap NIST_Board1 frame) in the world/FR3-base frame. One of "
                "{pos:[x,y,z], quat:[w,x,y,z]}, {pos, quat_xyzw:[x,y,z,w]} or {recording:'<timestamp>'} "
                "(raw mocap pose of that recording). Default: recording 2025-01-09-15-27-49 (episode 1)."}
_PARTS_SCHEMA = {"description": "Dataset part names ('round peg 1'), slugs ('round_peg_1'), or 'pegs' / "
                 "'round' / 'square'. Built families: round peg 1-4, square peg 1-4.",
                 "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}


@tool("scene_parts",
      "Catalog of buildable board parts: peg and hole dimensions, per-side clearance, chamfer, grasp offset, "
      "hole position and key orientation in the board frame, and every parameter with value, sigma, unit, "
      "source and whether it is outcome-sensitive. Also lists planned part families and what they need.",
      {"type": "object", "properties": {"parts": _PARTS_SCHEMA}})
def scene_parts(parts="pegs") -> dict:
    from twin.board import CONTACT, board_spec
    from twin.parts import FUTURE_FAMILIES, load_layout, part_spec, resolve
    lay = load_layout()
    return dict(board=board_spec(lay).describe(), contact=CONTACT,
                parts={n: part_spec(n, lay).describe() for n in resolve(parts)},
                layout=dict(file="twin/assets/board_layout.json", generated_at=lay.get("generated_at"),
                            split=lay.get("split"), episodes=lay.get("episodes"), method=lay.get("method"),
                            grid=lay.get("grid")),
                planned_families=FUTURE_FAMILIES)


@tool("board_layout",
      "Re-derive hole positions in the board frame from REASSEMBLE data: the TCP pose at gripper release in "
      "every successful Insert segment, mapped into that recording's mocap board frame, median per part. "
      "Returns per-part xy, sigma, n, key long-axis yaw, grip width, and a 4x4 grid fit cross-check. "
      "write=true overwrites twin/assets/board_layout.json (every scene uses it).",
      {"type": "object", "properties": {
          "split": {"type": "string", "enum": ["train", "test", "all"], "default": "train",
                    "description": "Dataset (HF) split to read. Keep 'train' unless the orchestrator froze other splits. Exam episodes (playbook/splits.json) are always left out."},
          "write": {"type": "boolean", "default": False}}})
def board_layout(split: str = "train", write: bool = False) -> dict:
    from twin.layout import OUT as LAYOUT_JSON, build_layout
    lay = build_layout(None if split == "all" else split)
    if write:
        LAYOUT_JSON.write_text(json.dumps(lay, indent=1))
        lay["written"] = str(LAYOUT_JSON.relative_to(REPO))
    return lay


@tool("scene_build",
      "Build a scene MJCF: base.xml (FR3 + Franka hand + cameras + table) with the placeholder board replaced "
      "by the task board (plate, exact primitive holes for the chosen parts, task-frame sites) and the parts as "
      "free bodies. Sites: '<slug>_hole/entry' (+z into the hole), '<slug>_hole/bottom', '<slug>/tip', "
      "'<slug>/grasp' (+z along insertion, +y finger-closing axis). sample_seed draws one world from the "
      "parameter uncertainty (null = nominal). Returns the xml path, params.json path and hole summary.",
      {"type": "object", "properties": {
          "parts": _PARTS_SCHEMA,
          "board_pose": _POSE_SCHEMA,
          "part_init": {"description": "'seated' (in its hole), 'above' (aligned, tip 10 mm above the entry), "
                        "'none' (holes only), or a map part -> one of those or {pos, quat} in world",
                        "anyOf": [{"type": "string"}, {"type": "object"}], "default": "seated"},
          "sample_seed": {"type": ["integer", "null"], "default": None},
          "timestep": {"type": ["number", "null"], "default": None, "description": "s; default 0.001 (2 ms lets sub-0.03 mm rim overlaps push through)"},
          "name": {"type": ["string", "null"], "default": None, "description": "output folder under runs/scene/"}}})
def scene_build(parts="pegs", board_pose=None, part_init="seated", sample_seed=None, timestep=None, name=None) -> dict:
    from twin.assembly import build_scene
    return build_scene(parts=parts, board_pose=board_pose, part_init=part_init, sample_seed=sample_seed,
                       timestep=timestep, name=name)


@tool("scene_check",
      "Check a scene MJCF: compiles; contacts deeper than the tolerance at t=0; settles under gravity with the "
      "arm holding its keyframe; reports max penetration (with the worst geom pairs), each free part's drift, "
      "the insertion predicates (depth, lateral, tilt, seated) before and after, and repeats the settle at "
      "half the timestep. ok=false lists the problems. Use list_sites=true to get every task site's pose.",
      {"type": "object", "properties": {
          "xml": {"type": "string", "description": "path from scene_build (repo-relative or absolute)"},
          "settle_s": {"type": "number", "default": 1.0},
          "halved_timestep": {"type": "boolean", "default": True},
          "penetration_tol_mm": {"type": "number", "default": 0.1},
          "drift_tol_mm": {"type": "number", "default": 0.5},
          "list_sites": {"type": "boolean", "default": False}},
       "required": ["xml"]})
def scene_check(xml: str, settle_s: float = 1.0, halved_timestep: bool = True, penetration_tol_mm: float = 0.1,
                drift_tol_mm: float = 0.5, list_sites: bool = False) -> dict:
    from twin.assembly import check_scene
    res = check_scene(xml, settle_s, halved_timestep, penetration_tol_mm, drift_tol_mm)
    if not list_sites:
        res["sites"] = [s["name"] for s in res["sites"]]
    return res


@tool("scene_render",
      "Render a scene MJCF from named cameras to PNG (runs/scene/<name>/render/). 'board_top' looks straight "
      "down at the plate (board frame x right, y up); task sites (group 4) are drawn when show_sites=true. "
      "Optionally step the physics first (settle_s).",
      {"type": "object", "properties": {
          "xml": {"type": "string"},
          "cameras": {"type": "array", "items": {"type": "string"},
                      "default": ["board_top", "overview", "hama1", "hama2", "hand"]},
          "width": {"type": "integer", "default": 640}, "height": {"type": "integer", "default": 480},
          "show_sites": {"type": "boolean", "default": True},
          "settle_s": {"type": "number", "default": 0.0}},
       "required": ["xml"]})
def scene_render(xml: str, cameras=("board_top", "overview", "hama1", "hama2", "hand"), width: int = 640,
                 height: int = 480, show_sites: bool = True, settle_s: float = 0.0) -> dict:
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from twin.assembly import render_scene
    return render_scene(xml, list(cameras), width, height, show_sites, settle_s)


def _apply_overrides(part, board, overrides: dict):
    for k, v in (overrides or {}).items():
        if k in part.params:
            part.params[k] = replace(part.params[k], value=float(v), source=f"override {v}")
        elif k in board.params:
            board.params[k] = replace(board.params[k], value=float(v), source=f"override {v}")
        else:
            raise ValueError(f"unknown parameter {k!r}; part: {list(part.params)}, board: {list(board.params)}")


@tool("insertion_probe",
      "Sanity-test a hole: a scripted Cartesian rig holds the part rigidly at its grasp site, starts with the tip "
      "3 mm above the entry offset sideways by each d in offsets_mm, and pushes along the hole axis with a "
      "force-limited servo (push_force). lateral='stiff' = rigid sideways; 'compliant' = 500 N/m (REASSEMBLE "
      "impedance default) or a number in N/m. rot_k (Nm/rad) adds wrist compliance; tilt_deg holds the part "
      "crooked. Outcomes: seated | jammed_at_rim | jammed_in_hole | partial | nan. offsets_mm='auto' brackets "
      "the expected thresholds (clearance for stiff, clearance + chamfer for compliant) and reports "
      "geometry_ok. halved_timestep re-runs every case at dt/2 and reports agreement. overrides sets catalog "
      "parameters (e.g. {chamfer_frac: 0, hole_d: 0.0042, seat_depth: 0.02}).",
      {"type": "object", "properties": {
          "part": {"type": "string"},
          "offsets_mm": {"anyOf": [{"type": "string", "enum": ["auto"]}, {"type": "array", "items": {"type": "number"}}],
                         "default": "auto"},
          "offset_dir_deg": {"type": "number", "default": 0.0, "description": "direction from the entry x axis"},
          "lateral": {"anyOf": [{"type": "string", "enum": ["stiff", "compliant"]}, {"type": "number"}],
                      "default": "stiff"},
          "rot_k": {"type": "number", "default": 0.0}, "tilt_deg": {"type": "number", "default": 0.0},
          "push_force": {"type": "number", "default": 10.0, "description": "N; REASSEMBLE insert contact Fz median ~7.6 N"},
          "speed": {"type": "number", "default": 0.02, "description": "m/s along the axis"},
          "timestep": {"type": "number", "default": 0.001, "description": "s; 2 ms over-penetrates sub-0.03 mm rim overlaps"},
          "halved_timestep": {"type": "boolean", "default": True},
          "cone": {"type": "string", "enum": ["pyramidal", "elliptic"], "default": "pyramidal"},
          "sample_seed": {"type": ["integer", "null"], "default": None},
          "overrides": {"type": "object", "default": {}},
          "render": {"type": "boolean", "default": False}},
       "required": ["part"]})
def insertion_probe(part: str, offsets_mm="auto", offset_dir_deg: float = 0.0, lateral="stiff", rot_k: float = 0.0,
                    tilt_deg: float = 0.0, push_force: float = 10.0, speed: float = 0.02, timestep: float = 0.001,
                    halved_timestep: bool = True, cone: str = "pyramidal", sample_seed=None, overrides=None,
                    render: bool = False) -> dict:
    import numpy as np
    from twin.board import board_spec
    from twin.parts import MM, load_layout, part_spec, resolve
    from twin.probe import run_probe

    name = resolve(part)[0]
    lay = load_layout()
    p, b = part_spec(name, lay), board_spec(lay)
    if sample_seed is not None:
        rng = np.random.default_rng(sample_seed)
        b, p = b.sampled(rng), p.sampled(rng)
    _apply_overrides(p, b, overrides or {})
    import math
    cx, cy = (v / MM for v in p.radial_clearance)
    th = math.radians(offset_dir_deg)
    # clearance along the offset direction (rectangle: the side the offset reaches first)
    c = min(cx / abs(math.cos(th)) if abs(math.cos(th)) > 1e-9 else math.inf,
            cy / abs(math.sin(th)) if abs(math.sin(th)) > 1e-9 else math.inf)
    if p.family == "round_peg":
        c = cx
    ch = p.v("chamfer_frac") * max(p.section) / MM
    compliant = lateral != "stiff"
    thresh = c + ch if compliant else c
    if offsets_mm == "auto":
        # the 'beyond' offset overlaps the rim by >= 0.04 mm: smaller overlaps sit inside the contact
        # penetration (~0.02-0.04 mm at 10 N) and give timestep-dependent jam depths (Lessons, build-twin)
        offsets_mm = sorted({0.0, round(0.8 * thresh, 4), round(thresh + max(0.25 * thresh, 0.04), 4), 3.0})
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = OUT / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{p.slug}_{lateral if isinstance(lateral, str) else f'kp{lateral:g}'}_{stamp}"
    rows, traces = [], {}
    for off in offsets_mm:
        rpath = str(out_dir / f"{tag}_off{off:g}.png") if render else None
        r = run_probe(p, b, offset_mm=float(off), offset_dir_deg=offset_dir_deg, timestep=timestep,
                      push_force=push_force, speed=speed, lateral=lateral, rot_k=rot_k, tilt_deg=tilt_deg,
                      cone=cone, render_path=rpath)
        traces[off] = r.pop("trace")
        row = {k: r[k] for k in ("offset_mm", "outcome", "final_depth_mm", "final_lateral_mm", "final_tilt_deg",
                                 "final_axial_resist_N", "max_penetration_mm", "wall_s")}
        if rpath:
            row["render"] = str(Path(rpath).relative_to(REPO))
        if tilt_deg == 0 and rot_k == 0:
            row["expected"] = "seated" if off < thresh else "jammed"
            row["matches_expected"] = (r["outcome"] == "seated") == (off < thresh)
        if halved_timestep:
            r2 = run_probe(p, b, offset_mm=float(off), offset_dir_deg=offset_dir_deg, timestep=timestep / 2,
                           push_force=push_force, speed=speed, lateral=lateral, rot_k=rot_k, tilt_deg=tilt_deg,
                           cone=cone)
            row["half_dt"] = dict(outcome=r2["outcome"], final_depth_mm=r2["final_depth_mm"],
                                  max_penetration_mm=r2["max_penetration_mm"])
            row["half_dt_agrees"] = (r2["outcome"] == r["outcome"]
                                     and abs(r2["final_depth_mm"] - r["final_depth_mm"]) < 0.5)
        rows.append(row)
    summary = dict(part=name, lateral=lateral, rot_k=rot_k, tilt_deg=tilt_deg, push_force=push_force, speed=speed,
                   timestep=timestep, cone=cone, sample_seed=sample_seed, overrides=overrides or {},
                   clearance_mm=[round(x / MM, 4) for x in p.radial_clearance], chamfer_mm=round(ch, 3),
                   seat_depth_mm=round(b.v("seat_depth") / MM, 3), plate_thickness_mm=round(b.v("plate_thickness") / MM, 3),
                   expected_seat_threshold_mm=round(thresh, 4) if tilt_deg == 0 and rot_k == 0 else None,
                   rows=rows)
    if all("matches_expected" in r for r in rows):
        summary["geometry_ok"] = all(r["matches_expected"] for r in rows)
    if halved_timestep:
        summary["half_dt_agrees"] = all(r["half_dt_agrees"] for r in rows)
    jf = out_dir / f"{tag}.json"
    jf.write_text(json.dumps(dict(**summary, traces={str(k): v for k, v in traces.items()},
                                  trace_cols=["t", "depth_mm", "lateral_mm", "tilt_deg", "resist_N"]), indent=1))
    summary["json"] = str(jf.relative_to(REPO))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for off, tr in traces.items():
            a = np.array(tr)
            ax[0].plot(a[:, 0], a[:, 1], label=f"{off:g} mm")
            ax[1].plot(a[:, 0], a[:, 4], label=f"{off:g} mm")
        ax[0].axhline(b.v("seat_depth") / MM, color="k", ls=":", lw=0.8)
        ax[0].set(xlabel="t [s]", ylabel="tip depth below entry [mm]", title=f"{name}: depth ({lateral})")
        ax[1].set(xlabel="t [s]", ylabel="board pushes back along axis [N]", title="axial resist force")
        ax[0].legend(title="offset", fontsize=8)
        fig.tight_layout()
        pf = out_dir / f"{tag}.png"
        fig.savefig(pf, dpi=80)
        plt.close(fig)
        summary["plot"] = str(pf.relative_to(REPO))
    except Exception as e:  # plotting is a convenience
        summary["plot_error"] = str(e)
    return summary
