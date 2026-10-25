"""MuJoCo digital twin of the REASSEMBLE cell (Franka FR3 + NIST Assembly Task Board #1)."""

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parent
SCENE_DIR = PKG_DIR / "scene"
BASE_XML = SCENE_DIR / "base.xml"
CAMERAS = ("hand", "hama1", "hama2")
