# build-twin: video only, single image or sketch

No robot to calibrate against and no contacts. Not scored on a real task yet (track `footage`).

- **Scale never from the eye alone.** Google's demos never estimated metric scale from one uncalibrated image (`docs/research/google-argon-twin-pipeline.md`). Use supplied calibration, a target, SfM, or a known-size object, and carry its error into every sigma.
- **Pipeline:** `recon/` (intake → frames → scene card → twin builder; `recon/README.md`); its HOT3D ground-truth check is `recon/gt_hot3d.py`.
- **Physics checks are the only contact evidence:** force closure, lift, stable placement on the retargeted motion (`references/reconstruct-scene-v1.md` §3).
- **Depth along the viewing direction is ambiguous** with views from one side; Argon's files were 13–24 mm low on REASSEMBLE for this reason (`runs/argon/probes/scene/findings.md`). Add a view or a contact-free constraint (resting on the table) before trusting z.
- **Handedness:** allow and test det = −1 when mapping a layout; a prior for det +1 put holes 150–300 mm off (same).
- **Human confirmation:** a person confirms the render against 2 frames before the scene is used (the project's confirmation step).
