"""Task board in its own frame, built from MuJoCo primitives so tight holes are exact.

Board frame = the mocap `NIST_Board1` frame (twin/layout.py). A scene places the whole board with one
pose (board frame -> world). Inside the board body:

  * plate: top surface at z = surface_z, `plate_thickness` thick, outline from the layout grid fit.
    It is tiled with boxes around one square "hole block" per part in the scene; holes of parts not
    in the scene are not cut.
  * hole walls: a convex polygon hole (round = regular N-gon whose flats sit exactly at the hole
    radius; key = rectangle) made of one box per edge. Each box's inner face lies on the edge line
    and it extends outward past the block corners, so the union is solid everywhere except the hole.
  * base: a slab whose top is `seat_depth` below the surface (what the peg bottoms on), plus four
    standoffs. The real stack under the PVC plate is unknown; seat_depth carries that uncertainty.
  * sites (group 4): '<slug>_hole/entry' at the hole centre on the surface, +z INTO the hole (the
    insertion direction), +y along the hole's long side; '<slug>_hole/bottom' at seat_depth.
  * custom numerics read by twin/predicates.py: '<slug>/seat_depth', '<slug>/clearance' (x, y per
    side), '<slug>/tol' (depth m, lateral m, tilt deg).
  * camera 'board_top' looking straight down at the plate centre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from twin.parts import MM, P, PartSpec, load_layout

# Contact parameters shared by board and parts. timeconst is fixed (not 2*dt) so the halved-timestep
# check compares the same physics. Board friction is low on purpose: MuJoCo takes the max of the two
# geoms, so each part's own `friction` parameter sets the part-board pair.
CONTACT = dict(solref=[0.004, 1.0], solimp=[0.95, 0.99, 0.0005, 0.5, 2.0], margin=0.0, condim=3,
               torsional=0.005, rolling=0.0001, board_friction=0.05)

# Wall tops sit 20 um below the plate top so overlapping coplanar faces do not z-fight in renders.
WALL_DROP = 2e-5

SITE_TOL = dict(depth=1.0 * MM, lateral_extra=0.2 * MM, tilt_deg=3.0)


@dataclass
class BoardSpec:
    params: dict[str, P] = field(default_factory=dict)
    plate_xy_min: tuple[float, float] = (-0.0102, -0.3833)
    plate_size: tuple[float, float] = (0.384, 0.384)
    hole_ngon: int = 32
    block_margin: float = 0.006
    contact: dict = field(default_factory=lambda: dict(CONTACT))

    def v(self, k: str) -> float:
        return self.params[k].value

    def sampled(self, rng: np.random.Generator) -> "BoardSpec":
        ps = {k: p.sample(rng) for k, p in self.params.items()}
        # a peg cannot stop inside a through-hole: seat depth >= plate thickness
        ps["seat_depth"] = replace(ps["seat_depth"], value=max(ps["seat_depth"].value, ps["plate_thickness"].value))
        return replace(self, params=ps)

    def describe(self) -> dict:
        return dict(plate_xy_min=list(self.plate_xy_min), plate_size=list(self.plate_size),
                    hole_ngon=self.hole_ngon, contact=self.contact,
                    params={k: p.as_dict() for k, p in self.params.items()})


def board_spec(layout: dict | None = None) -> BoardSpec:
    layout = layout or load_layout()
    g = layout.get("grid") or {}
    zs = layout.get("mocap_board_world_z", {})
    params = dict(
        surface_z=P(0.0303, 0.004, "m",
                    "plate top above the mocap board origin: +30.3 mm (+-4), plate corners triangulated by the "
                    "robot-calibrated static cameras (twin/calib board entry, world-frame Lessons 2026-10-24; the "
                    "plate sits on an aluminium frame). Contacts agree: with the corrected hand (pads 22.9 mm above "
                    "O_T_EE), insert onset timing puts the plate top + part length below O_T_EE at 48.4 mm, i.e. "
                    "30.3 mm with the pads 9 mm below the part top (replay-validate Lessons)", False),
        plate_thickness=P(0.009, 0.003, "m", "NIST acrylic 0.354 in (assembly_instructions_v4.docx); REASSEMBLE "
                          "uses PVC of unknown thickness", True, 0.004, 0.020),
        seat_depth=P(0.020, 0.003, "m", "depth at which a peg stops below the plate top: Remove pull-outs in "
                     "episode 0 travel 17-23 mm while the pull force lasts (TCP rise; independent of the TCP offset); "
                     "with the plate on a frame, the part passes the 9 mm plate and stops on what lies beneath",
                     True, 0.012, 0.030),
        base_thickness=P(0.010, 0.0, "m", "cosmetic"),
    )
    return BoardSpec(params=params, plate_xy_min=tuple(g.get("plate_xy_min", (-0.0102, -0.3833))),
                     plate_size=tuple(g.get("plate_size", (0.384, 0.384))))


# ----------------------------------------------------------------------------------- geometry
def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def entry_frame(part: PartSpec) -> tuple[np.ndarray, np.ndarray]:
    """Rotation (columns x_e, y_e, z_e) of the hole entry frame in the board frame; and its xy axes."""
    psi = math.radians(part.long_axis_yaw_deg)
    y_e = np.array([math.cos(psi), math.sin(psi), 0.0])
    z_e = np.array([0.0, 0.0, -1.0])
    x_e = np.cross(y_e, z_e)
    return np.column_stack([x_e, y_e, z_e]), np.array([x_e[:2], y_e[:2]])


def mat_to_quat(R: np.ndarray) -> list[float]:
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q.tolist()


def hole_edges(part: PartSpec, ngon: int) -> list[tuple[np.ndarray, float]]:
    """(outward unit normal in board xy, distance from hole centre) for each wall."""
    _, (ax, ay) = entry_frame(part)
    hx, hy = part.hole_section
    if part.family == "round_peg":
        r = hx / 2
        out = []
        for k in range(ngon):
            th = 2 * math.pi * k / ngon
            out.append((np.array([math.cos(th), math.sin(th)]), r))
        return out
    return [(ax, hx / 2), (-ax, hx / 2), (ay, hy / 2), (-ay, hy / 2)]


def hole_circumradius(part: PartSpec, ngon: int) -> float:
    hx, hy = part.hole_section
    if part.family == "round_peg":
        return hx / 2 / math.cos(math.pi / ngon)
    return math.hypot(hx, hy) / 2


def _tile(x0, x1, y0, y1, blocks):
    """Axis-aligned boxes covering [x0,x1]x[y0,y1] minus the blocks (each (bx0,bx1,by0,by1))."""
    xs = sorted({x0, x1, *[min(max(v, x0), x1) for b in blocks for v in b[:2]]})
    ys = sorted({y0, y1, *[min(max(v, y0), y1) for b in blocks for v in b[2:]]})
    strips = []  # (xa, xb, ya, yb)
    for j in range(len(ys) - 1):
        ya, yb = ys[j], ys[j + 1]
        if yb - ya < 1e-9:
            continue
        run = None
        for i in range(len(xs) - 1):
            xa, xb = xs[i], xs[i + 1]
            if xb - xa < 1e-9:
                continue
            cx, cy = (xa + xb) / 2, (ya + yb) / 2
            inside = any(b[0] <= cx <= b[1] and b[2] <= cy <= b[3] for b in blocks)
            if inside:
                if run:
                    strips.append((*run, ya, yb))
                run = None
            else:
                run = (run[0], xb) if run else (xa, xb)
        if run:
            strips.append((*run, ya, yb))
    # merge vertically adjacent strips with the same x extent
    merged = []
    for s in sorted(strips, key=lambda s: (s[0], s[1], s[2])):
        if merged and abs(merged[-1][0] - s[0]) < 1e-9 and abs(merged[-1][1] - s[1]) < 1e-9 \
                and abs(merged[-1][3] - s[2]) < 1e-9:
            merged[-1] = (s[0], s[1], merged[-1][2], s[3])
        else:
            merged.append(s)
    return merged


def add_board(spec, parent, parts: list[PartSpec], board: BoardSpec, pos=(0, 0, 0), quat=(1, 0, 0, 0),
              name: str = "board", material: str | None = None):
    """Add the board body under `parent` (mjsBody). Returns (body, info dict)."""
    import mujoco

    c = board.contact
    zs = board.v("surface_z")
    t = board.v("plate_thickness")
    seat = board.v("seat_depth")
    tb = board.v("base_thickness")
    x0, y0 = board.plate_xy_min
    x1, y1 = x0 + board.plate_size[0], y0 + board.plate_size[1]
    rgba = [0.86, 0.86, 0.84, 1.0]
    geom_kw = dict(friction=[c["board_friction"], c["torsional"], c["rolling"]], solref=c["solref"],
                   solimp=c["solimp"], condim=c["condim"], margin=c["margin"])
    if material:
        geom_kw["material"] = material
    else:
        geom_kw["rgba"] = rgba

    body = parent.add_body(name=name, pos=list(map(float, pos)), quat=list(map(float, quat)))
    body.add_site(name=f"{name}/origin", pos=[0, 0, 0], size=[0.004, 0, 0], rgba=[0, 0, 1, 1], group=4)
    body.add_site(name=f"{name}/center", pos=[(x0 + x1) / 2, (y0 + y1) / 2, zs], size=[0.004, 0, 0],
                  rgba=[0, 0, 1, 1], group=4)

    blocks, reach = [], []
    holes_info = {}
    for part in parts:
        hx, hy = part.hole_xy
        a = hole_circumradius(part, board.hole_ngon) + board.block_margin
        blk = (hx - a, hx + a, hy - a, hy + a)
        if blk[0] < x0 or blk[1] > x1 or blk[2] < y0 or blk[3] > y1:
            raise ValueError(f"hole block of {part.name} leaves the plate outline")
        blocks.append(blk)
        W = a * math.sqrt(2) + 0.001
        edges = hole_edges(part, board.hole_ngon)
        for k, (n, d) in enumerate(edges):
            T = a * math.sqrt(2) - d + 0.001
            cxy = np.array([hx, hy]) + (d + T / 2) * n
            body.add_geom(name=f"{part.slug}_hole/wall{k:02d}", type=mujoco.mjtGeom.mjGEOM_BOX,
                          pos=[float(cxy[0]), float(cxy[1]), zs - t / 2 - WALL_DROP / 2],
                          size=[T / 2, W, t / 2 - WALL_DROP / 2],
                          quat=_yaw_quat(math.atan2(n[1], n[0])), **geom_kw)
            reach.append((part.name, float(np.hypot(d + T, W))))
        r_out = max(r for n_, r in reach if n_ == part.name)
        if min(hx - x0, x1 - hx, hy - y0, y1 - hy) < r_out:
            raise ValueError(f"hole walls of {part.name} would stick out of the plate outline")
        R, _ = entry_frame(part)
        q = mat_to_quat(R)
        body.add_site(name=f"{part.slug}_hole/entry", pos=[hx, hy, zs], quat=q, size=[0.003, 0, 0],
                      rgba=[1, 0.6, 0, 1], group=4)
        body.add_site(name=f"{part.slug}_hole/bottom", pos=[hx, hy, zs - seat], quat=q, size=[0.002, 0, 0],
                      rgba=[1, 0.6, 0, 1], group=4)
        cx, cy = part.radial_clearance
        spec.add_numeric(name=f"{part.slug}/seat_depth", data=[seat], size=1)
        spec.add_numeric(name=f"{part.slug}/clearance", data=[cx, cy], size=2)
        spec.add_numeric(name=f"{part.slug}/tol", data=[SITE_TOL["depth"], max(cx, cy) + SITE_TOL["lateral_extra"],
                                                         SITE_TOL["tilt_deg"]], size=3)
        holes_info[part.slug] = dict(xy=[hx, hy], block_half=a, walls=len(edges),
                                     radial_clearance_mm=[round(cx / MM, 4), round(cy / MM, 4)])

    # walls reach beyond their block: make sure they stay inside the plate and away from other holes
    for i, p in enumerate(parts):
        for j, q2 in enumerate(parts):
            if i < j:
                dist = math.dist(p.hole_xy, q2.hole_xy)
                rmax = max(r for n_, r in reach if n_ in (p.name, q2.name))
                if dist < rmax + hole_circumradius(q2, board.hole_ngon) + 0.002:
                    raise ValueError(f"holes {p.name} and {q2.name} too close for the wall layout ({dist:.3f} m)")

    tiles = _tile(x0, x1, y0, y1, blocks)
    for k, (xa, xb, ya, yb) in enumerate(tiles):
        body.add_geom(name=f"{name}/plate{k:03d}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[(xa + xb) / 2, (ya + yb) / 2, zs - t / 2], size=[(xb - xa) / 2, (yb - ya) / 2, t / 2],
                      **geom_kw)
    # base slab (peg stop) and standoffs
    body.add_geom(name=f"{name}/base", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[(x0 + x1) / 2, (y0 + y1) / 2, zs - seat - tb / 2],
                  size=[(x1 - x0) / 2, (y1 - y0) / 2, tb / 2], **geom_kw)
    gap = seat - t
    if gap > 1e-4:
        for k, (sx, sy) in enumerate([(x0 + 0.017, y0 + 0.017), (x1 - 0.017, y0 + 0.017),
                                      (x0 + 0.017, y1 - 0.017), (x1 - 0.017, y1 - 0.017)]):
            body.add_geom(name=f"{name}/standoff{k}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                          pos=[sx, sy, zs - t - gap / 2], size=[0.005, gap / 2, 0], **geom_kw)
    cam_pos = [(x0 + x1) / 2, (y0 + y1) / 2, zs + 0.55]
    body.add_camera(name="board_top", pos=cam_pos, xyaxes=[1, 0, 0, 0, 1, 0], fovy=45.0)
    info = dict(n_plate_tiles=len(tiles), holes=holes_info, surface_z=zs, plate_thickness=t,
                seat_depth=seat, plate_xy=[[x0, y0], [x1, y1]])
    return body, info


def seated_body_pose(part: PartSpec, board: BoardSpec, depth: float | None = None,
                     lateral: tuple[float, float] = (0.0, 0.0)) -> tuple[list[float], list[float]]:
    """Body pose (board frame) that puts the part's tip `depth` along the entry z (default: seated),
    shifted by `lateral` (m) along the entry x/y axes."""
    R, _ = entry_frame(part)
    depth = board.v("seat_depth") if depth is None else depth
    entry = np.array([*part.hole_xy, board.v("surface_z")])
    tip = entry + R @ np.array([lateral[0], lateral[1], depth])
    centre = tip - R[:, 2] * (part.length / 2)
    return centre.tolist(), mat_to_quat(R)
