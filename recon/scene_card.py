"""The scene card: one JSON that says what is in the video, where, how sure we are, and where each number came from.

Frame conventions: the card's world ("twin world") is metric, gravity-aligned, z up, origin on the support
surface below the first camera; x is the first camera's forward direction projected onto the support.
Cameras are OpenCV (x right, y down, z forward); poses are T_world_from_cam.

Every quantity carries a `source` and, where it is estimated, a `confidence` in [0, 1] or a sigma.
Physical parameters that video cannot measure carry a prior value and a randomisation range.
"""
from __future__ import annotations

SCHEMA = "recon.scene_card/v0"

# What the card must hold; nested keys are checked one level down.
REQUIRED = {
    "schema": str,
    "source": dict,
    "frames": dict,
    "camera": dict,
    "metric_scale": dict,
    "world": dict,
    "surfaces": list,
    "objects": list,
    "hands": list,
    "actions": list,
    "contacts": list,
    "physics_priors": dict,
    "provenance": list,
}
REQUIRED_CAMERA = ["K", "K_source", "trajectory", "all_T_world_from_cam"]
REQUIRED_OBJECT = ["id", "category", "shape", "size_m", "pose_t0", "track", "confidence", "movable", "physics"]
REQUIRED_SCALE = ["value", "anchor", "sources", "consensus"]

# Category priors: typical size (longest dimension, m) usable as a scale anchor when tight, plus mass and
# friction ranges for randomisation. Values are generic household figures, not fitted to any dataset; the
# range is what the twin samples from. `anchor_ok` marks sizes tight enough (< ~15% spread) to anchor scale.
CATEGORY_PRIORS = {
    "mug":          {"size_m": 0.12, "size_range": [0.09, 0.15], "anchor_ok": False, "shape": "cylinder",
                     "mass_kg": 0.30, "mass_range": [0.15, 0.45], "friction": [0.4, 0.8]},
    "cup":          {"size_m": 0.10, "size_range": [0.07, 0.14], "anchor_ok": False, "shape": "cylinder",
                     "mass_kg": 0.15, "mass_range": [0.02, 0.35], "friction": [0.4, 0.8]},
    "can":          {"size_m": 0.122, "size_range": [0.115, 0.13], "anchor_ok": True, "shape": "cylinder",
                     "mass_kg": 0.36, "mass_range": [0.015, 0.40], "friction": [0.3, 0.6]},
    "bottle":       {"size_m": 0.24, "size_range": [0.18, 0.32], "anchor_ok": False, "shape": "cylinder",
                     "mass_kg": 0.5, "mass_range": [0.03, 1.1], "friction": [0.3, 0.7]},
    "pitcher":      {"size_m": 0.24, "size_range": [0.17, 0.30], "anchor_ok": False, "shape": "cylinder",
                     "mass_kg": 0.6, "mass_range": [0.25, 1.5], "friction": [0.4, 0.8]},
    "bowl":         {"size_m": 0.20, "size_range": [0.12, 0.30], "anchor_ok": False, "shape": "cylinder",
                     "mass_kg": 0.35, "mass_range": [0.1, 0.8], "friction": [0.4, 0.8]},
    "plate":        {"size_m": 0.26, "size_range": [0.20, 0.30], "anchor_ok": False, "shape": "cylinder",
                     "mass_kg": 0.45, "mass_range": [0.1, 0.9], "friction": [0.3, 0.7]},
    "keyboard":     {"size_m": 0.43, "size_range": [0.28, 0.47], "anchor_ok": False, "shape": "box",
                     "mass_kg": 0.7, "mass_range": [0.35, 1.2], "friction": [0.5, 1.0]},
    "mouse":        {"size_m": 0.115, "size_range": [0.10, 0.13], "anchor_ok": True, "shape": "box",
                     "mass_kg": 0.09, "mass_range": [0.06, 0.14], "friction": [0.2, 0.5]},
    "phone":        {"size_m": 0.15, "size_range": [0.14, 0.165], "anchor_ok": True, "shape": "box",
                     "mass_kg": 0.18, "mass_range": [0.14, 0.24], "friction": [0.3, 0.7]},
    "marker":       {"size_m": 0.14, "size_range": [0.12, 0.155], "anchor_ok": True, "shape": "cylinder",
                     "mass_kg": 0.015, "mass_range": [0.008, 0.025], "friction": [0.3, 0.6]},
    "book":         {"size_m": 0.24, "size_range": [0.18, 0.30], "anchor_ok": False, "shape": "box",
                     "mass_kg": 0.5, "mass_range": [0.15, 1.2], "friction": [0.4, 0.8]},
    "box":          {"size_m": 0.20, "size_range": [0.05, 0.50], "anchor_ok": False, "shape": "box",
                     "mass_kg": 0.3, "mass_range": [0.05, 2.0], "friction": [0.4, 0.8]},
    "board":        {"size_m": 0.35, "size_range": [0.20, 0.50], "anchor_ok": False, "shape": "box",
                     "mass_kg": 0.6, "mass_range": [0.2, 1.5], "friction": [0.4, 0.8]},
    "fruit":        {"size_m": 0.08, "size_range": [0.05, 0.11], "anchor_ok": False, "shape": "sphere",
                     "mass_kg": 0.15, "mass_range": [0.05, 0.3], "friction": [0.4, 0.8]},
    "utensil":      {"size_m": 0.18, "size_range": [0.14, 0.22], "anchor_ok": False, "shape": "box",
                     "mass_kg": 0.04, "mass_range": [0.015, 0.08], "friction": [0.3, 0.6]},
    "holder":       {"size_m": 0.12, "size_range": [0.06, 0.20], "anchor_ok": False, "shape": "box",
                     "mass_kg": 0.15, "mass_range": [0.05, 0.5], "friction": [0.4, 0.8]},
    "object":       {"size_m": 0.15, "size_range": [0.03, 0.50], "anchor_ok": False, "shape": "box",
                     "mass_kg": 0.3, "mass_range": [0.02, 2.0], "friction": [0.3, 0.9]},
}
# Map free-text prompts to a prior category (first substring match wins).
CATEGORY_ALIASES = [
    ("mug", "mug"), ("cup", "cup"), ("can", "can"), ("bottle", "bottle"), ("pitcher", "pitcher"),
    ("jug", "pitcher"), ("kettle", "pitcher"), ("bowl", "bowl"), ("plate", "plate"), ("keyboard", "keyboard"),
    ("mouse", "mouse"), ("phone", "phone"), ("marker", "marker"), ("pen", "marker"), ("book", "book"),
    ("box", "box"), ("board", "board"), ("tray", "board"), ("apple", "fruit"), ("orange", "fruit"),
    ("banana", "fruit"), ("fruit", "fruit"), ("spoon", "utensil"), ("fork", "utensil"), ("knife", "utensil"),
    ("spatula", "utensil"), ("holder", "holder"), ("stand", "holder"),
]
SUPPORT_PRIORS = {"table": {"friction": [0.4, 0.9], "height_m": [0.65, 0.80]},
                  "counter": {"friction": [0.4, 0.9], "height_m": [0.85, 0.95]},
                  "floor": {"friction": [0.5, 1.0], "height_m": [0.0, 0.0]}}
# Adult hand: palm width across index-to-little MCP joints (m). Used as a scale anchor (VideoMimic-style
# body-size anchoring); population spread is about +-10%.
HAND_PALM_WIDTH_M = {"value": 0.080, "range": [0.070, 0.090]}


def prior_for(category: str) -> tuple[str, dict]:
    c = (category or "").lower()
    if c in CATEGORY_PRIORS:
        return c, CATEGORY_PRIORS[c]
    for sub, cat in CATEGORY_ALIASES:
        if sub in c:
            return cat, CATEGORY_PRIORS[cat]
    return "object", CATEGORY_PRIORS["object"]


def physics_for(category: str) -> dict:
    cat, p = prior_for(category)
    return {"prior_category": cat, "mass_kg": {"value": p["mass_kg"], "range": p["mass_range"]},
            "friction": {"value": round(sum(p["friction"]) / 2, 3), "range": p["friction"]},
            "restitution": {"value": 0.1, "range": [0.0, 0.3]},
            "source": "category prior (generic household values; not measured from video)"}


def validate(card: dict) -> list[str]:
    """Return a list of problems (empty = valid)."""
    errs = []
    for k, t in REQUIRED.items():
        if k not in card:
            errs.append(f"missing {k}")
        elif not isinstance(card[k], t):
            errs.append(f"{k} should be {t.__name__}")
    if card.get("schema") != SCHEMA:
        errs.append(f"schema should be {SCHEMA}")
    for k in REQUIRED_CAMERA:
        if k not in card.get("camera", {}):
            errs.append(f"camera.{k} missing")
    for k in REQUIRED_SCALE:
        if k not in card.get("metric_scale", {}):
            errs.append(f"metric_scale.{k} missing")
    for i, o in enumerate(card.get("objects", [])):
        for k in REQUIRED_OBJECT:
            if k not in o:
                errs.append(f"objects[{i}].{k} missing")
        if "size_m" in o and (len(o["size_m"]) != 3 or min(o["size_m"]) <= 0):
            errs.append(f"objects[{i}].size_m must be 3 positive numbers")
    for i, h in enumerate(card.get("hands", [])):
        if h.get("side") not in ("left", "right"):
            errs.append(f"hands[{i}].side must be left/right")
    return errs


def empty_card() -> dict:
    return {"schema": SCHEMA, "source": {}, "frames": {}, "camera": {}, "metric_scale": {}, "world": {},
            "surfaces": [], "objects": [], "hands": [], "actions": [], "contacts": [],
            "physics_priors": {"gravity": [0, 0, -9.81], "categories": {}}, "provenance": []}
