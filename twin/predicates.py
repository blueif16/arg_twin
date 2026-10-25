"""Success predicates for board parts, read from simulator state only.

Everything needed comes from the compiled model, so an independent watcher can evaluate a scene from
its MJCF alone (no catalog import):
  sites    '<slug>_hole/entry' (+z into the hole), '<slug>/tip' (+z along the part's insertion axis)
  numerics '<slug>/seat_depth' [m], '<slug>/clearance' [x, y per side, m],
           '<slug>/tol' [depth tol m, lateral tol m, tilt tol deg]

  depth   = (tip - entry) . z_entry          > 0 means the tip is below the surface
  lateral = |(tip - entry) - depth z_entry|   distance of the tip from the hole axis
  tilt    = angle(z_tip, z_entry)
  seated  = depth >= seat_depth - tol_depth and lateral <= tol_lateral and tilt <= tol_tilt
  engaged = depth > 2 mm and lateral <= tol_lateral (in the hole, not yet seated)
"""

from __future__ import annotations

import math

import numpy as np


def _id(m, kind, name):
    import mujoco
    return mujoco.mj_name2id(m, kind, name)


def numeric(m, name: str) -> np.ndarray | None:
    import mujoco
    i = _id(m, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    if i < 0:
        return None
    a, n = m.numeric_adr[i], m.numeric_size[i]
    return np.array(m.numeric_data[a:a + n])


def part_slugs(m) -> list[str]:
    import mujoco
    out = []
    for i in range(m.nnumeric):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_NUMERIC, i)
        if nm and nm.endswith("/seat_depth"):
            s = nm[: -len("/seat_depth")]
            if _id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}/tip") >= 0:
                out.append(s)
    return out


def contact_force_on(m, d, body_a: int, body_b: int) -> np.ndarray:
    """Total contact force (world frame, N) that body_b exerts on body_a."""
    import mujoco
    f6 = np.zeros(6)
    tot = np.zeros(3)
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if {b1, b2} != {body_a, body_b} or c.efc_address < 0:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        f_on_2 = c.frame.reshape(3, 3).T @ f6[:3]
        tot += f_on_2 if b2 == body_a else -f_on_2
    return tot


def insertion_state(m, d, slug: str, board_body: str = "board", part_body: str | None = None) -> dict:
    import mujoco
    e = _id(m, mujoco.mjtObj.mjOBJ_SITE, f"{slug}_hole/entry")
    t = _id(m, mujoco.mjtObj.mjOBJ_SITE, f"{slug}/tip")
    if e < 0 or t < 0:
        raise KeyError(f"sites for {slug} missing (need {slug}_hole/entry and {slug}/tip)")
    seat = float(numeric(m, f"{slug}/seat_depth")[0])
    tol = numeric(m, f"{slug}/tol")
    ze = d.site_xmat[e].reshape(3, 3)[:, 2]
    zt = d.site_xmat[t].reshape(3, 3)[:, 2]
    dp = d.site_xpos[t] - d.site_xpos[e]
    depth = float(dp @ ze)
    lateral = float(np.linalg.norm(dp - depth * ze))
    tilt = float(math.degrees(math.acos(float(np.clip(zt @ ze, -1, 1)))))
    out = dict(depth=depth, lateral=lateral, tilt_deg=tilt, seat_depth=seat,
               seated=bool(depth >= seat - tol[0] and lateral <= tol[1] and tilt <= tol[2]),
               engaged=bool(depth > 0.002 and lateral <= tol[1]))
    pb = _id(m, mujoco.mjtObj.mjOBJ_BODY, part_body or slug)
    bb = _id(m, mujoco.mjtObj.mjOBJ_BODY, board_body)
    if pb >= 0 and bb >= 0:
        f = contact_force_on(m, d, pb, bb)
        out["board_force"] = f.tolist()
        out["axial_resist"] = float(-f @ ze)  # board pushing the part back out of the hole (N)
    return out


def all_states(m, d, board_body: str = "board") -> dict[str, dict]:
    return {s: insertion_state(m, d, s, board_body) for s in part_slugs(m)}
