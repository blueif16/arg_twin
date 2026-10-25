"""Scene card -> MJCF, written with MjSpec (scene as code).

- Support surfaces become static boxes whose top is the fitted plane (z = 0 in the twin world).
- Objects become free bodies at their t0 pose: primitives (box, cylinder, sphere), a convex hull of the
  fused points, or a procedural concave shape (bowl) decomposed into convex parts with CoACD.
- Every object gets sites (centre, top) and its mass and friction from the card's physics priors.
- Contact parameters follow playbook/skills/mujoco (fixed timeconst so the dt/2 check compares the same
  physics): solref [0.004, 1], solimp [0.95, 0.99, 0.0005, 0.5, 2], margin 0, condim 3.
- The first camera of the trajectory is added as camera `cam0` (OpenCV pose converted to MuJoCo's).
- Randomisation ranges are written beside the XML (twin_ranges.json); the XML holds the prior means.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .common import CACHE, R_to_quat_wxyz, absp, quat_wxyz_to_R, read_json, rel, write_json

SOLREF = [0.004, 1.0]
SOLIMP = [0.95, 0.99, 0.0005, 0.5, 2.0]
DT = 0.002


def bowl_mesh(radius: float, height: float, wall: float = 0.006, n: int = 32):
    """Procedural open bowl (lathe of a quarter-ellipse shell), watertight."""
    import trimesh
    ts = np.linspace(0, np.pi / 2, 10)
    outer = np.stack([radius * np.sin(ts), height * (1 - np.cos(ts))], 1)  # r, z from bottom
    inner = np.stack([(radius - wall) * np.sin(ts[::-1]), wall + (height - wall) * (1 - np.cos(ts[::-1]))], 1)
    prof = np.vstack([[[0, 0]], outer[1:], inner[:-1], [[0, wall]]])
    verts, faces = [], []
    m = len(prof)
    for j in range(n):
        a = 2 * np.pi * j / n
        for r, z in prof:
            verts.append([r * np.cos(a), r * np.sin(a), z])
    for j in range(n):
        j2 = (j + 1) % n
        for i in range(m - 1):
            a, b, c, d = j * m + i, j * m + i + 1, j2 * m + i + 1, j2 * m + i
            faces += [[a, b, c], [a, c, d]]
    mesh = trimesh.Trimesh(np.array(verts), np.array(faces), process=True)
    mesh.merge_vertices()
    mesh.fix_normals()
    mesh.apply_translation([0, 0, -height / 2])
    return mesh


def coacd_parts(mesh, threshold: float = 0.08, tag: str = "") -> list[Path]:
    """CoACD convex decomposition, cached by mesh content. Returns obj paths."""
    import coacd
    import trimesh
    h = hashlib.sha256(np.ascontiguousarray(mesh.vertices).tobytes() + np.ascontiguousarray(mesh.faces).tobytes()
                       + str(threshold).encode()).hexdigest()[:16]
    d = CACHE / "coacd" / h
    if d.exists() and any(d.glob("part_*.obj")):
        return sorted(d.glob("part_*.obj"))
    d.mkdir(parents=True, exist_ok=True)
    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold, seed=0)
    out = []
    for i, (v, f) in enumerate(parts):
        p = d / f"part_{i:03d}.obj"
        trimesh.Trimesh(v, f).export(p)
        out.append(p)
    write_json(d / "meta.json", {"n_parts": len(out), "threshold": threshold, "tag": tag})
    return out


def _cam_quat_mj(T_wc: np.ndarray):
    R_gl = np.asarray(T_wc)[:3, :3] @ np.diag([1, -1, -1])
    return R_to_quat_wxyz(R_gl)


def build(card: dict, out_xml: str, overrides: dict | None = None) -> dict:
    """Write the twin XML for a scene card. overrides: {object_id: {pos_delta, size_scale, yaw_deg}} for
    known-error tests. Returns paths and a summary."""
    import mujoco
    ov = overrides or {}
    spec = mujoco.MjSpec()
    spec.modelname = "recon_twin"
    spec.option.timestep = DT
    spec.option.gravity = card.get("physics_priors", {}).get("gravity", [0, 0, -9.81])
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 640
    wb = spec.worldbody
    wb.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    ranges = {"objects": {}, "surfaces": {}}
    # surfaces
    for s in card.get("surfaces", []) or [{"id": "support", "category": "table", "extent_m": [1.2, 0.8],
                                            "centre_xy": [0.4, 0.0], "friction": {"value": 0.6, "range": [0.4, 0.9]}}]:
        ex = s.get("extent_m", [1.2, 0.8])
        cx, cy = s.get("centre_xy", [0.4, 0.0])
        g = wb.add_geom(name=f"surf_{s['id']}", type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[ex[0] / 2, ex[1] / 2, 0.025], pos=[cx, cy, -0.025],
                        rgba=[0.6, 0.45, 0.3, 1])
        g.friction = [s.get("friction", {}).get("value", 0.6), 0.005, 0.0001]
        g.solref = SOLREF
        g.solimp = SOLIMP
        ranges["surfaces"][s["id"]] = {"friction": s.get("friction", {}).get("range", [0.4, 0.9])}
    # objects
    meshes_used = []
    for o in card.get("objects", []):
        oo = ov.get(o["id"], {})
        pos = np.asarray(o["pose_t0"]["pos"], float) + np.asarray(oo.get("pos_delta", [0, 0, 0]), float)
        R = quat_wxyz_to_R(o["pose_t0"]["quat_wxyz"])
        if oo.get("yaw_deg"):
            a = np.radians(oo["yaw_deg"])
            Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            R = Rz @ R
        size = np.asarray(o["size_m"], float) * float(oo.get("size_scale", 1.0))
        body = wb.add_body(name=f"obj_{o['id']}", pos=pos, quat=R_to_quat_wxyz(R))
        if o.get("movable", True):
            body.add_freejoint(name=f"free_{o['id']}")
        phys = o.get("physics", {})
        mass = float(phys.get("mass_kg", {}).get("value", 0.3))
        fr = float(phys.get("friction", {}).get("value", 0.6))
        shape = o.get("shape", "box")
        geoms = []
        if shape == "cylinder":
            r = (size[0] + size[1]) / 4
            geoms.append(body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, size[2] / 2, 0]))
        elif shape == "sphere":
            geoms.append(body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[size.mean() / 2, 0, 0]))
        elif shape == "bowl":
            mesh = bowl_mesh((size[0] + size[1]) / 4, size[2])
            parts = coacd_parts(mesh, tag=o["id"])
            for i, p in enumerate(parts):
                mn = f"m_{o['id']}_{i}"
                spec.add_mesh(name=mn, file=str(p), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX)
                geoms.append(body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn))
            meshes_used += [rel(p) for p in parts]
        elif shape == "hull" and o.get("hull_obj"):
            mn = f"m_{o['id']}"
            spec.add_mesh(name=mn, file=str(absp(o["hull_obj"])), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX)
            geoms.append(body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn))
            meshes_used.append(o["hull_obj"])
        else:
            geoms.append(body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size / 2)))
        # mass split across geoms by default density then rescaled via body mass
        for g in geoms:
            g.friction = [fr, 0.005, 0.0001]
            g.solref = SOLREF
            g.solimp = SOLIMP
            g.condim = 3
            g.rgba = [0.2, 0.5, 0.8, 1] if o.get("movable", True) else [0.5, 0.5, 0.5, 1]
        body.add_site(name=f"site_{o['id']}_centre", pos=[0, 0, 0], size=[0.004, 0, 0])
        body.add_site(name=f"site_{o['id']}_top", pos=[0, 0, size[2] / 2], size=[0.004, 0, 0])
        body.mass = mass  # explicit; inertia from geoms is rescaled at compile (inertiafromgeom auto)
        body.explicitinertial = False
        ranges["objects"][o["id"]] = {"mass_kg": phys.get("mass_kg", {}).get("range"),
                                      "friction": phys.get("friction", {}).get("range"),
                                      "restitution": phys.get("restitution", {}).get("range"),
                                      "size_scale": [0.9, 1.1], "pos_jitter_m": 0.01}
    # camera
    traj = card.get("camera", {}).get("all_T_world_from_cam") or []
    if traj:
        T0 = np.asarray(traj[0])
        K = np.asarray(card["camera"]["K"])
        H = card["frames"].get("height", 512)
        fovy = float(np.degrees(2 * np.arctan((H / 2) / K[1, 1])))
        wb.add_camera(name="cam0", pos=T0[:3, 3], quat=_cam_quat_mj(T0), fovy=fovy)
    spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
    m = spec.compile()
    # rescale body masses to the prior (inertia from geometry, mass from the card)
    for o in card.get("objects", []):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obj_{o['id']}")
        want = float(o.get("physics", {}).get("mass_kg", {}).get("value", 0.3))
        if m.body_mass[bid] > 0:
            f = want / m.body_mass[bid]
            spec.bodies[bid].mass = want
            spec.bodies[bid].inertia = (m.body_inertia[bid] * f).tolist()
            spec.bodies[bid].ipos = m.body_ipos[bid].tolist()
            spec.bodies[bid].iquat = m.body_iquat[bid].tolist()
            spec.bodies[bid].explicitinertial = True
    m = spec.compile()
    xml = spec.to_xml()
    p = absp(out_xml)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(xml)
    # reload check
    m2 = mujoco.MjModel.from_xml_path(str(p))
    write_json(p.with_name(p.stem + "_ranges.json"), ranges)
    return {"xml": rel(p), "ranges": rel(p.with_name(p.stem + "_ranges.json")), "nbody": int(m2.nbody),
            "ngeom": int(m2.ngeom), "meshes": meshes_used, "dt": DT}
