# recon: any video → scene card → physics twin (Loop 1, v0, stopped early)

Status, 2026-10-25: **stopped when the hackathon scope was cut. Nothing has run end to end and there are no GT numbers yet.** None of this code has been executed or tested.

What exists (written, not run):
- `common.py`: JSON io, a content-hashed tool cache (`runs/recon/cache/`), Sim(3)/SE(3) pose alignment that stays well posed on short head-mounted clips, mask RLE.
- `intake.py`: a file, frame folder or URL (yt-dlp into `data/recon-downloads/`), probed with ffprobe, with optional extra streams (calibration, depth, poses, robot log). Writes `intake.json`: present/absent per stream, what must be estimated, honest limits.
- `frames.py`: frame sampling, plus undistortion of Aria fisheye624 (or OpenCV) footage to an upright pinhole when a calibration stream is supplied.
- `scene_card.py`: schema v0 (camera K and trajectory, metric scale with its sources, surfaces, objects with size, pose track and confidence, hands, actions, contacts) and category priors (size anchors, mass and friction ranges).
- `twin_builder.py`: scene card → MJCF with MjSpec. Primitives, convex hull, and a procedural bowl decomposed with CoACD; contact parameters from playbook/skills/mujoco; a `_ranges.json` written beside the MJCF.
- `gt_hot3d.py`: a HOT3D-Clips adapter (exports a clip as video + calibration, GT kept apart). Scorers: camera ATE (Sim3 and SE3 at our scale), focal, scale (depth ratio on GT-mesh pixels), object centre and size, wrist.

Data and environment:
- 3 HOT3D Aria train clips: `data/hot3d-clips/`, ids 001849, 002066, 002210.
- `.venv-recon` (Python 3.11) with transformers 5.5.4, mujoco 3.17, mediapipe and opencv. Its torch was pulled up to 2.14.1+cu130; re-pin to 2.8.0+cu128 with `.venv-recon/constraints.txt` before building MegaSaM.
- Sources for MegaSaM (with its checkpoint), MoGe and DA3 are cloned under `.venv-recon/src/`.
- SAM 3 gated weights are accessible with the cached HF token.

Not done: perception wrappers (MegaSaM, DA3, MoGe-2, SAM 3, hands), objects.py, gates.py, cli.py, pipeline, Argon loop, random-video run.

Next steps:
1. Re-pin torch, then build MegaSaM's CUDA extensions (sm_89, nvcc 12.8; torch_scatter needed).
2. Write the wrappers and the JSON CLI.
3. Run the scripted pipeline on the 3 clips and test the gates with known-error injections.
4. Argon loop.

HaWoR needs MANO files, which need registration. Use a MediaPipe + depth lift for hands until the user decides.
