"""Part catalog for the board twin: geometry, hole fits, task frames and per-parameter uncertainty.

Every number is a `P` (value, sigma, unit, source, sensitive). `sensitive=True` marks parameters that
change task outcomes (clearance, chamfer, friction, seat depth); those get primitive-exact geometry and
are what a spread of worlds should vary. Everything else is cosmetic or only matters for replays.

Conventions (shared with twin/board.py, twin/predicates.py and playbook/skills/mujoco):
  * part slug: 'round peg 1' -> 'round_peg_1' (twin.layout.slug)
  * part body frame: origin at mid-length, +z along the insertion direction (towards the tip)
  * sites: '<slug>/tip' at the leading end, '<slug>/grasp' where the fingers hold it; both have
    +z along the insertion direction and +y along the finger-closing axis (= key long side)
  * keys: long side `a` along local y, short side `b` along local x

Families built so far: round_peg (1-4), square_peg (1-4). See FUTURE_FAMILIES for the rest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

import numpy as np

from twin import REPO_DIR
from twin.layout import OUT as LAYOUT_JSON, slug

MM = 1e-3


@dataclass(frozen=True)
class P:
    value: float
    sigma: float = 0.0
    unit: str = "m"
    source: str = ""
    sensitive: bool = False
    lo: float | None = None  # hard bounds after sampling
    hi: float | None = None

    def sample(self, rng: np.random.Generator) -> "P":
        if self.sigma <= 0:
            return self
        v = self.value + float(np.clip(rng.normal(), -2, 2)) * self.sigma
        if self.lo is not None:
            v = max(v, self.lo)
        if self.hi is not None:
            v = min(v, self.hi)
        return replace(self, value=v)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != ""}


SRC_NIST_STL = "vendors/nist-taskboard/taskboard_1_stl (peg STLs, mm; chamfer 0.1*max side on one end)"
SRC_NIST_DRILL = "NIST assembly_instructions_v4.docx tools list: 4.1/8.1/12.2/16.5 mm drill bits for round pegs"
SRC_NIST_PLATE = "GMC_Laser_Plate_Virtual.STL hole sections at mid-thickness (key holes are laser cut, not drilled)"
SRC_PAPER_SIZES = "REASSEMBLE arXiv 2502.05086 App. B: peg sizes 1-4 = 4/8/12/16 mm, keys 4, 8x7, 12x8, 16x10"
SRC_GRIP = "twin/layout.py: grip width at release grows 1.6/6.1/11.0/14.8 mm with peg index (train split)"
SRC_PICK = ("pad centre below the part's top end: zero crossing of the signed contact-onset error over train "
            "replays with the corrected hand (pads 22.9 mm above the logged O_T_EE) and the camera plate top (+30.3 mm): "
            "9 mm (scan 3-15 mm: -0.63 s at 3 mm, 0.0 s at 9 mm, +0.77 s at 15 mm; "
            "runs/claude/run-standin-1/replay/scan_grasp_height_v2.json). Per-attempt spread ~+-5 mm. NOTE: the "
            "'<slug>/grasp' site is where the PAD CENTRE goes; the logged TCP (O_T_EE) lies 22.9 mm beyond it")

ROUND_D = {1: 4.0, 2: 8.0, 3: 12.0, 4: 16.0}  # mm
ROUND_HOLE = {1: 4.1, 2: 8.1, 3: 12.2, 4: 16.5}  # mm, drill bit
KEY_AB = {1: (4.0, 4.0), 2: (8.0, 7.0), 3: (12.0, 8.0), 4: (16.0, 10.0)}  # mm, long a x short b
KEY_HOLE = {1: (4.11, 4.11), 2: (8.157, 7.148), 3: (12.178, 8.137), 4: (16.252, 10.212)}  # mm, plate STL

FUTURE_FAMILIES = {
    "gear": "small/medium/large gear (module 1, 20/40/60 teeth, 10 mm bore) on 10 mm shafts; Gear_*.STL in "
            "board coordinates; mujoco.sdf.gear plugin exists (CPU only, not in MuJoCo Warp)",
    "bolt_4": "M16 nut on a bolt; needs threads (mujoco.sdf.nut/bolt plugins, CPU only) or a twist-free "
              "approximation (nut as hex prism with a cylindrical bore over a cylinder)",
    "connector": "BNC, D-SUB, Ethernet, USB, waterproof: shells + key features; NIST male/female STLs; "
                 "Ethernet/USB holders were redesigned by REASSEMBLE and are not public",
}


@dataclass
class PartSpec:
    name: str
    family: str  # round_peg | square_peg
    index: int
    params: dict[str, P] = field(default_factory=dict)
    hole_xy: tuple[float, float] = (0.0, 0.0)  # board frame, from layout
    hole_xy_sigma: tuple[float, float] = (0.0, 0.0)
    long_axis_yaw_deg: float = 0.0  # board frame yaw of the hole's (and key's) long side

    @property
    def slug(self) -> str:
        return slug(self.name)

    def v(self, key: str) -> float:
        return self.params[key].value

    # --- derived geometry (metres)
    @property
    def length(self) -> float:
        return self.v("length")

    @property
    def section(self) -> tuple[float, float]:
        """(along local x, along local y) full widths of the peg cross-section."""
        if self.family == "round_peg":
            d = self.v("peg_d")
            return d, d
        return self.v("peg_b"), self.v("peg_a")

    @property
    def hole_section(self) -> tuple[float, float]:
        if self.family == "round_peg":
            d = self.v("hole_d")
            return d, d
        return self.v("hole_b"), self.v("hole_a")

    @property
    def radial_clearance(self) -> tuple[float, float]:
        """Per-side clearance along local x and y."""
        (px, py), (hx, hy) = self.section, self.hole_section
        return (hx - px) / 2, (hy - py) / 2

    @property
    def grasp_z(self) -> float:
        """Grasp site z in the body frame (origin mid-length, +z to the tip)."""
        return -self.length / 2 + self.v("grasp_from_top")

    def sampled(self, rng: np.random.Generator) -> "PartSpec":
        ps = {k: p.sample(rng) for k, p in self.params.items()}
        # keep the hole at least 10 um larger than the peg so the part can always go in
        if self.family == "round_peg":
            ps["hole_d"] = replace(ps["hole_d"], value=max(ps["hole_d"].value, ps["peg_d"].value + 2e-5))
        else:
            for h, pk in (("hole_a", "peg_a"), ("hole_b", "peg_b")):
                ps[h] = replace(ps[h], value=max(ps[h].value, ps[pk].value + 2e-5))
        xy = tuple(float(c + np.clip(rng.normal(), -2, 2) * s) for c, s in zip(self.hole_xy, self.hole_xy_sigma))
        return replace(self, params=ps, hole_xy=xy)

    def describe(self) -> dict:
        cx, cy = self.radial_clearance
        return dict(name=self.name, slug=self.slug, family=self.family, hole_xy=list(self.hole_xy),
                    hole_xy_sigma=list(self.hole_xy_sigma), long_axis_yaw_deg=self.long_axis_yaw_deg,
                    radial_clearance_mm=[round(cx / MM, 4), round(cy / MM, 4)],
                    params={k: p.as_dict() for k, p in self.params.items()})


def load_layout() -> dict:
    if not LAYOUT_JSON.exists():
        raise FileNotFoundError(f"{LAYOUT_JSON} missing: run .venv/bin/python -m twin.layout")
    return json.loads(LAYOUT_JSON.read_text())


def _common(chamfer_frac_src: str) -> dict[str, P]:
    return dict(
        length=P(0.050, 0.0002, "m", SRC_NIST_STL),
        density=P(7850.0, 785.0, "kg/m3", "steel ground rod / key stock (MISUMI RGOCG, KET); not weighed"),
        friction=P(0.30, 0.10, "-", "steel on PVC, textbook range 0.2-0.5; not measured", True, 0.05, 1.0),
        grasp_from_top=P(0.009, 0.005, "m", SRC_PICK, False, 0.002, 0.030),
        chamfer_frac=P(0.10, 0.05, "-", chamfer_frac_src, True, 0.0, 0.15),
    )


def part_spec(name: str, layout: dict | None = None) -> PartSpec:
    """Catalog entry for one part (exact dataset name, e.g. 'round peg 3')."""
    layout = layout or load_layout()
    fam, idx = name.rsplit(" ", 1)[0].replace(" ", "_"), int(name.rsplit(" ", 1)[1]) if name[-1].isdigit() else 0
    ch_src = SRC_NIST_STL + "; REASSEMBLE pegs not checked (NIST calls the chamfer optional)"
    if fam == "round_peg" and idx in ROUND_D:
        ps = _common(ch_src)
        ps["peg_d"] = P(ROUND_D[idx] * MM, 0.01 * MM, "m", SRC_PAPER_SIZES + "; " + SRC_GRIP, True)
        ps["hole_d"] = P(ROUND_HOLE[idx] * MM, 0.05 * MM, "m",
                         SRC_NIST_DRILL + "; REASSEMBLE PVC board keeps 'original tolerances' (paper III-A)", True)
    elif fam == "square_peg" and idx in KEY_AB:
        ps = _common(ch_src)
        a, b = KEY_AB[idx]
        ha, hb = KEY_HOLE[idx]
        ps["peg_a"] = P(a * MM, 0.03 * MM, "m", SRC_PAPER_SIZES + "; long side gripped (" + SRC_GRIP + ")", True)
        ps["peg_b"] = P(b * MM, 0.03 * MM, "m", SRC_PAPER_SIZES, True)
        ps["hole_a"] = P(ha * MM, 0.05 * MM, "m", SRC_NIST_PLATE, True)
        ps["hole_b"] = P(hb * MM, 0.05 * MM, "m", SRC_NIST_PLATE, True)
    else:
        raise ValueError(f"no catalog entry for {name!r}; built families: round peg 1-4, square peg 1-4. "
                         f"Planned: {list(FUTURE_FAMILIES)}")
    lp = layout["parts"].get(name)
    if lp is None:
        raise ValueError(f"{name!r} not in {LAYOUT_JSON.name}; regenerate it with twin.layout")
    yaw = lp["long_axis_yaw_deg"] if fam == "square_peg" else 0.0
    return PartSpec(name=name, family=fam, index=idx, params=ps, hole_xy=tuple(lp["xy"]),
                    hole_xy_sigma=tuple(lp["xy_sigma"]), long_axis_yaw_deg=float(yaw))


PEG_NAMES = [f"round peg {i}" for i in range(1, 5)] + [f"square peg {i}" for i in range(1, 5)]


def resolve(names: list[str] | str | None) -> list[str]:
    """Accept dataset names, slugs or family shortcuts ('pegs', 'round', 'square')."""
    if names is None or names == "pegs" or names == ["pegs"]:
        return list(PEG_NAMES)
    if isinstance(names, str):
        names = [names]
    out = []
    by_slug = {slug(n): n for n in PEG_NAMES}
    for n in names:
        if n in ("round", "round_peg", "round pegs"):
            out += PEG_NAMES[:4]
        elif n in ("square", "square_peg", "square pegs"):
            out += PEG_NAMES[4:]
        elif n in PEG_NAMES:
            out.append(n)
        elif n in by_slug:
            out.append(by_slug[n])
        else:
            raise ValueError(f"unknown part {n!r}; known: {PEG_NAMES}")
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------------------------------ MJCF
def _peg_vertices(spec: PartSpec, nseg: int = 48) -> np.ndarray:
    """Convex hull points of the peg in its body frame (chamfer at the +z/tip end)."""
    L = spec.length
    sx, sy = spec.section
    c = spec.v("chamfer_frac") * max(sx, sy)
    z0, z1 = -L / 2, L / 2
    if spec.family == "round_peg":
        r = sx / 2
        th = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
        ring = lambda rad, z: np.c_[rad * np.cos(th), rad * np.sin(th), np.full(nseg, z)]  # noqa: E731
        return np.vstack([ring(r, z0), ring(r, z1 - c), ring(r - c, z1)])
    hx, hy = sx / 2, sy / 2
    rect = lambda ax, ay, z: np.array([[ax, ay, z], [-ax, ay, z], [-ax, -ay, z], [ax, -ay, z]])  # noqa: E731
    return np.vstack([rect(hx, hy, z0), rect(hx, hy, z1 - c), rect(hx - c, hy - c, z1)])


PEG_RGBA = {"round_peg": [0.62, 0.64, 0.68, 1.0], "square_peg": [0.45, 0.47, 0.52, 1.0]}


def add_part_body(spec_root, parent, part: PartSpec, pos, quat, free: bool, contact: dict,
                  name: str | None = None):
    """Add the part as a body under `parent` (mjsBody). Returns the body.

    Collision: a convex mesh when chamfered (MuJoCo hulls it; multiccd gives multi-point contact),
    else a primitive cylinder/box. Sites '<slug>/tip' and '<slug>/grasp' follow the module conventions.
    """
    import mujoco

    s = part.slug
    body = parent.add_body(name=name or s, pos=list(map(float, pos)), quat=list(map(float, quat)))
    if free:
        body.add_freejoint(name=f"{s}/free")
    L = part.length
    sx, sy = part.section
    chamfer = part.v("chamfer_frac") * max(sx, sy)
    common = dict(density=part.v("density"), rgba=PEG_RGBA[part.family],
                  friction=[part.v("friction"), contact["torsional"], contact["rolling"]],
                  solref=contact["solref"], solimp=contact["solimp"], condim=contact["condim"],
                  margin=contact["margin"])
    if chamfer > 1e-6:
        mesh = spec_root.add_mesh(name=f"{s}_hull")
        mesh.uservert = _peg_vertices(part).ravel().tolist()
        body.add_geom(name=f"{s}/collision", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{s}_hull", **common)
    elif part.family == "round_peg":
        body.add_geom(name=f"{s}/collision", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[sx / 2, L / 2, 0],
                      **common)
    else:
        body.add_geom(name=f"{s}/collision", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx / 2, sy / 2, L / 2],
                      **common)
    body.add_site(name=f"{s}/tip", pos=[0, 0, L / 2], size=[0.002, 0, 0], rgba=[1, 0.2, 0.2, 1], group=4)
    body.add_site(name=f"{s}/grasp", pos=[0, 0, part.grasp_z], size=[0.002, 0, 0], rgba=[0.2, 1, 0.2, 1],
                  group=4)
    return body


if __name__ == "__main__":
    lay = load_layout()
    for n in PEG_NAMES:
        d = part_spec(n, lay).describe()
        print(f"{n:13s} hole_xy={d['hole_xy']} yaw={d['long_axis_yaw_deg']} clearance_mm={d['radial_clearance_mm']}")
    print(f"layout: {LAYOUT_JSON.relative_to(REPO_DIR)}")
