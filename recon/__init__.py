"""recon: source-agnostic video -> scene card -> physics twin (Loop 1, v0).

Layout:
  common.py        paths, JSON io, hashing, cache, geometry helpers
  intake.py        any video (path or URL) -> intake.json (what the source offers)
  frames.py        sample frames; undistort (fisheye624 / opencv) when calibration is supplied
  perception/      camera (MegaSaM, DA3), metric depth (MoGe-2), masks (SAM 3), hands (MediaPipe lift)
  objects.py       primitives fitted from masks + metric depth + cameras
  scene_card.py    the scene card schema and builder
  twin_builder.py  scene card -> MJCF (MjSpec)
  gates.py         label-free rejection gates and their known-error tests
  gt_hot3d.py      HOT3D-Clips ground-truth adapter and scorers (scoring only)
  cli.py           JSON in / JSON out CLI over every tool (`python -m recon.cli list|call`)
  pipeline.py      the scripted pipeline
  argon_loop.py    Argon as the brain over the same tools (Interactions API)

Perception tools run in .venv-recon (see docs/research/recon-setup-notes.md); the Argon loop runs in .venv.
"""

__version__ = "0.1.0"
