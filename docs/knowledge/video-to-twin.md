# Video to twin: scene, objects, hands and physics from arbitrary footage

## Question

Given an arbitrary video (YouTube, a robot dataset, a phone; egocentric or third-person) with no robot logs, no calibration and no CAD:

- What can current tools recover, and how accurately: camera, metric geometry, objects and articulation, hands and contact, physical parameters?
- What runs on this machine (RTX 4070 SUPER, 12 GB)?
- How can the resulting twin be checked without labels, and which ground-truth (GT) benchmarks can calibrate those checks?

## What is established, with numbers

### 1. Camera and geometry from monocular video

- **MegaSaM** (deep visual SLAM with mono-depth priors; code Apache 2.0).
  - Sintel ATE 0.018 with known focal and 0.023 without. Prior and concurrent methods (DROID-SLAM, CasualSAM, MonST3R and others) score 0.030–0.110. On 12 in-the-wild videos: 0.004. Trajectories are normalised to unit length [megasam-2025].
  - Video depth abs-rel on Sintel: 0.21, against 0.37 for Depth Anything V2. Camera time is 0.7–1.0 s per frame on an A100 [megasam-2025].
  - Known failures: moving objects filling the image, object motion colinear with camera motion, zoom, and strong radial distortion [megasam-2025].
  - On egocentric HOT3D it had the lowest relative error of six trackers: RPE-T 3.19 mm, RPE-R 0.08°. But its arc-length ratio was 0.716, so its own metric scale was about 28% short (SE(3) alignment, no scale fit). Across the six trackers the ratio ranged 0.47–1.09 [mint-2026].
- **MASt3R-SLAM.** Runs at 15 FPS on an RTX 4090.
  - ATE on TUM RGB-D: 0.030 m with intrinsics, 0.060 m without. On EuRoC: 0.041 vs 0.164 m [mast3r-slam-2025].
  - It was trained on pinhole images and degrades with lens distortion, so fisheye video (Aria) must be undistorted first [mast3r-slam-2025].
  - Its checkpoints inherit non-commercial terms from their training data (`data/refs/mast3r-slam-readme-2026.md`).
- **Feed-forward models.**
  - **VGGT**, 10 frames: camera AUC@30 of 85.3 (Re10K) and 88.2 (CO3D) in 0.2 s.
    - Baselines: DUSt3R 67.7 / 76.7 in 7 s; MASt3R 76.4 / 81.8 in 9 s; COLMAP+SuperPoint/SuperGlue 45.2 / 25.3 in 15 s.
    - Peak memory on an H100: 5.58 GB at 20 frames, 11.41 GB at 50 [vggt-2025].
    - A May 2026 fix fits 2–3× more frames in the same memory. Only the gated VGGT-1B-Commercial checkpoint allows commercial use (`data/refs/vggt-readme-2026.md`).
  - **VGGT-Ω** (CVPR 2026 oral), on Sintel against the prior best (MegaSaM):
    - camera AUC@3° 40.0 vs 22.5;
    - depth δ1.25 93.5 vs 74.1;
    - 50× faster [vggt-omega-2026].
    - The authors later warned that the released 1B checkpoint's benchmark rows may be inflated by contamination [vggt-omega-2026, secondhand: learnopencv guide].
  - **Depth Anything 3.** It beats VGGT by 35.7% on camera pose and 23.6% on geometry, on its own benchmark [da3-2025].
    - Apache 2.0: Small, Base, Metric-Large and Mono-Large.
    - CC BY-NC 4.0: the Large and Giant any-view models and the metric Nested model.
    - DA3-Streaming runs long video in under 12 GB (`data/refs/da3-readme-2026.md`).
  - **Egocentric comparison** on HOT3D [egoxperience-2026]. VGGT and VGGT-Ω were scaled per sequence to DA3 Nested depth, so the comparison is not fully matched:

    | Model | Normalised ATE | Depth AbsRel |
    |---|---|---|
    | VGGT-Ω | 0.0018 | 0.10 |
    | DA3 Nested | 0.0036 | 0.08 |
    | VGGT | 0.0064 | 0.22 |
    | MapAnything | 0.0321 | 0.32 |
    | CUT3R | 0.0449 | 0.27 |

- **Metric scale from one image.**
  - MoGe-2 metric point maps, averaged over 7 datasets: relative error 8.19%, δ1 93.6.
    - Baselines: UniDepth V2 10.1% / 91.9; Depth Pro 13.7% / 81.9.
    - Metric depth error 15.7%, against 29.9% for Depth Anything V2.
    - 29 ms and 0.96 GB per frame on an A100.
    - The authors warn that metric-scale ambiguity causes deviations out of distribution [moge2-2025].
  - MapAnything regresses metric geometry and takes intrinsics, poses or depth as optional inputs [mapanything-2026, abstract].
- **Scale error dominates world error.** On HOT3D, camera ATE is 3.4–3.8 mm when scale is aligned to GT. With scale taken from metric depth it rises to 14.6–43.6 mm [hawor-2025]:

  | Scale source | ATE (mm) |
  |---|---|
  | DROID + ZoeDepth | 43.6 |
  | DROID + Depth Anything V2 | 27.5 |
  | DROID + Metric3D v2 | 21.1 |
  | HaWoR's hand-masked fit | 14.6 |

- **COLMAP on egocentric kitchen video** (EPIC-Fields) [epic-fields-2023].
  - Hands are in 95% of frames.
  - Frames with ≥ 0.9 homography overlap were dropped, 82% of all frames.
  - A video counted as reconstructed when ≥ 70% of its frames registered. That happened for 90% of videos on the first pass and 96% after a stricter retry.
  - Cost: about 2,264 compute hours for 671 videos averaging 9 min.
  - Failures were short videos over large areas, and low light.
  - There was no GT: quality was judged by reprojection error (0.87 px mean) and held-out PSNR.
- **A human as the scale anchor.** VideoMimic fits the scene scale jointly with an SMPL human trajectory, over 123 phone videos [videomimic-2025].
  - It uses MegaSaM for textured scenes and MonST3R/Align3R for textureless ones.
  - On SLOPER4D, scene Chamfer was 0.75 vs 10.66 for TRAM.

### 2. Objects and articulation

- **Segmentation.**
  - SAM 2 runs at 43.8 FPS (Hiera-B+) and 30.2 FPS (Hiera-L) on an A100. It needs 3× fewer user interactions than earlier video methods [sam2-2024].
  - SAM 3 takes text noun phrases and doubles accuracy on promptable concept segmentation [sam3-2025]. It has about 0.9B parameters and gated weights (HF `facebook/sam3`).
- **Single-image 3D (SAM 3D Objects).**
  - Shape on SA-3DAO: F1@0.01 of 0.234, against 0.148 for TRELLIS and 0.140–0.157 for Hunyuan3D 2.0/2.1. Layout on ADT: 3D IoU 0.497, ADD-S@0.1 0.767 [sam3d-2025].
  - It predicts one object at a time and "isn't trained to reason about physical interactions, such as contact, physical stability, interpenetration" [sam3d-2025].
  - It needs ≥ 32 GB of VRAM (`data/refs/sam3d-setup-2026.md`).
- **Tracking needs depth.** BundleSDF (track and reconstruct an unknown object) and FoundationPose (pose of a novel object) both take RGB-D [bundlesdf-2023, foundationpose-2024].
- **Failure rates on internet video** (Something-Something v2, Video2Policy) [video2policy-2025]:

  | Tool | Failure rate |
  |---|---|
  | Grounding DINO | 60% |
  | FoundationPose | 55% |
  | InstantMesh | 35% |
  | SAM 2 | 15% |
  | Average | 42% |

  - Relative sizes came out mostly right; absolute sizes were noisy.
  - A policy trained at the estimated sizes fell from 97% to 83% success when the objects were resized to their true sizes.
- **Articulation.**
  - **Articulate-Anything** (a VLM actor-critic over mesh retrieval) [articulate-anything-2025]:
    - 75% success on PartNet-Mobility, against 8.7–12.2% for URDFormer and Real2Code;
    - in-the-wild mesh Chamfer 0.070 with generated meshes and 0.110 with retrieved ones, against 0.109 for Real2Code (which uses DUSt3R) and 0.229 for URDFormer.
  - **FunREC** (egocentric RGB-D with known intrinsics). Revolute-axis error was 5.3–12.4°, against 26–38° for BundleSDF and 60–70° for ArtGS. It exports URDF/USD, but takes mass and inertia from a VLM query with no check [funrec-2026].
  - **PhysX-Anything** (one image to a physical asset). Absolute-scale error was 0.30 vs 43.44 for PhysXGen, and kinematic score 0.83. It reports no mass or density metric, and its MuJoCo test is qualitative only [physx-anything-2025].
- **Simulation-ready meshes.**
  - MuJoCo's `inertia="exact"` "requires a well-oriented, watertight mesh and will error otherwise". The default `legacy` overcounts volume for non-convex meshes, and `convex` is recommended [mujoco-xmlref-2026].
  - DeepSimHO made hand-object meshes simulatable in MuJoCo with CoACD convex decomposition and composite-rigid-body inertia [deepsimho-2023].
  - No benchmark of hole rate or simulation stability for generated meshes was found.

### 3. Hands and contact from egocentric video

- **Per-frame hands.** WiLoR vs HaMeR, PA-MPJPE (alignment removes scale and position):
  - FreiHAND: 5.5 vs 6.0 mm;
  - HO3D v2: 7.5 vs 7.7 mm.

  WiLoR's detector runs at 138–175 FPS on an RTX 4090. Both use weak perspective, so hand depth is not metric [wilor-2025]. Licence: CC-BY-NC-ND plus MANO's licence (`data/refs/wilor-readme-2026.md`).
- **World-space hands.** On HOT3D, HaWoR scores W-MPJPE 33.2 mm. HaMeR, WiLoR, HandDGP or HMP paired with SLAM score 119–156 mm [hawor-2025].
  - Its pipeline: a hand-masked DROID-SLAM, Metric3D for scale, and an infiller for hands that leave the view. About 40 ms per frame. Licence: CC-BY-NC-ND [hawor-2025].
  - On a newer egocentric benchmark, WiLoR + SLAM scores 328.8 mm W-MPJPE, against 113.2 mm for EgoX [egoxperience-2026].
- **Hand-object pose.** Generalising to in-the-wild egocentric RGB "remains challenging". HOPformer reaches 82.4% success on ARCTIC and introduces EPIC-Contact: 2.3K in-the-wild clips with 3D contact annotations [hopformer-2026, abstract].
- **Contact and force are weak.** EgoPHI, on ARCTIC in distribution [egophi-2026]:
  - hand-contact F1 0.19, IoU 0.11;
  - hand-force MAE 4.0 N.

  Out of distribution (H2O), contact F1 drops to 0.053.

### 4. Physical parameters you can't see

- **VLM priors.**
  - Mass: Gemini 2.5 Pro and 3.1 Pro reach min-ratio error (MnRE) 0.72–0.81 on about 300 real objects [physquantagent-2026]. MnRE = min(m/m̂, m̂/m), so 0.78 means a typical factor of 1.28 off.
    - Visual prompts add 0.01–0.03.
    - Scale estimation fails on transparent objects.
  - Mass via NeRF2Physics on ABO-500: MnRE 0.552, against 0.306 for LLaVA. Friction on 6 points: MnRE 0.736 vs 0.692 for GPT-4V [nerf2physics-2024].
  - Before refinement, VLM mass guesses (Intern-S1, Gemini 3 Pro) were off by 12–300% on the four objects reported. Refining against a wrist F/T sensor brought six objects to 1.5–7.2% [rigpi-2026].
  - PhysX-Anything's authors avoided VLM judgement of physical properties because they found it unreliable [physx-anything-2025].
- **What published pipelines do with mass and friction:**
  - RialTo uses one default mass and friction for all objects [rialto-2024];
  - Video2Policy randomises object size and weight, plus action noise 0.02 and a 0.01–0.02 s delay [video2policy-2025];
  - FunREC queries a VLM [funrec-2026];
  - Real2Render2Real drops dynamics entirely [r2r2r-2025].
- **System identification from video alone.**
  - real2sim-eval fits spring-mass physics from video, but for deformables [real2sim-eval-2025].
  - Identification that matched true values used robot probing with force or contact sensing [asid-2024, intent-real2sim-2026, rigpi-2026].
  - No published monocular, real-video identification of rigid-body mass and friction with a measured error was found.

### 5. Validation without labels: how well label-free scores track truth

- **Reprojection error favours methods built like its reference.** On 7-Scenes the SfM pseudo-GT has lower reprojection error than the depth-SLAM one (1.25 vs 1.74 px on Chess). Yet the method ranking reverses between them: Active Search ranks last against depth-SLAM pseudo-GT (68.7% at 5 cm/5°) and first against SfM (98.5%) [brachmann-2021].
- **Photometric match does not imply geometry.** On DTU, 3DGS renders its training views at 35.76 dB PSNR with Chamfer 1.96; 2DGS reaches 34.52 dB with Chamfer 0.80 [2dgs-2024]. Separately, video-model "visual realism does not imply physical understanding" [physics-iq-2025, abstract].
- **Alignment-invariant and smoothness scores hide world error.**
  - On HOT3D, WiLoR-SLAM has the best baseline PA-MPJPE (6.00 mm) but a W-MPJPE of 151.7 mm.
  - HMP-SLAM and HaWoR have the same jerk (Accel 5.50 vs 5.41), yet W-MPJPE of 119.4 vs 33.2 mm [hawor-2025].
- **Physical plausibility does not rank accuracy** [deepsimho-2023]:
  - On DexYCB, methods within 1.07–1.25 cm hand error pass the 200 ms MuJoCo stability test 16.8–32.9% of the time.
  - On HO3D, the most accurate object estimate (corner error 4.79 cm) passes 11.4% of the time, while a physics-optimised method with the worst error (5.74 cm) passes 17.0%.
  - Even ground truth penetrates by 0.91 cm. Test frames whose GT did not stay stable in simulation had to be filtered out.
  - Simulated stability must be read together with penetration, since colliding forces can balance [deepsimho-2023, citing Hasson et al. 2019].
- **Mask plus depth agreement can select good estimates, but not always.**
  - C-3PO's certificate requires two checks. 3D: ≥ 90% of observed points lie within ε of the posed model. 2D: the detected mask is covered by the rendered mask.
  - On six T-LESS objects, ADD-S was 0.79–1.00 on the certified subset vs 0.38–0.97 on all outputs.
  - On some of the full 30 T-LESS objects the certified subset was worse than average: ADD-S 0.33 vs 0.51 and 0.40 vs 0.68 [c3po-ensemble-2023].
  - It needs RGB-D and CAD.
- **A learned confidence score** reached Spearman −0.73 to −0.84 against ADD error, but on synthetic noise only [cobra-2024].
- **Our own runs.** The edge-overlay metric passed a robot rendered at the wrong time; only held-out keypoints and two-view checks caught it (`playbook/skills/reconstruct-scene/SKILL.md` Known traps; [evaluation-and-statistics](evaluation-and-statistics.md) known-error tests).

### 6. Ground-truth benchmarks

| Dataset | What it gives | Licence | Size, and a small subset |
|---|---|---|---|
| HOT3D [hot3d-2025] | 833 min, 19 subjects, 33 rigid objects; Aria and Quest 3. GT: headset poses (mm-accurate VIO/SLAM), hands (MANO, UmeTrack), object 6DoF from mocap with 3 mm markers, object meshes | Sequences CC BY-SA; hand data CC BY-NC-SA; models CC BY-SA, not for sale | HOT3D-Clips: 3,832 clips of 150 frames (5 s), 762 GB on HF `bop-benchmark/hot3d`, one tar per clip, about 0.2 GB each. Clips include cameras, object poses, masks and MANO |
| ADT [adt-2023] | 236 sequences in 2 spaces (the paper says 200); 398 objects (74 dynamic). GT: device and object 6DoF, depth, segmentation, eye gaze, body pose. Object models P50 / P80 error 0.69 / 4.7 mm; whole scene 4.7 / 20.5 mm | Meta research-only, no redistribution | About 3.5 TB without MPS. A 10 s sample is about 500 MB; per-sequence download (`--sequence_names`) |
| ARCTIC [arctic-2023] | 2.1M frames, 8 third-person views + 1 egocentric; MANO, SMPL-X, articulated object meshes and poses, contact | Non-commercial | Full images 649 GB, cropped 116 GB, raw sequences 215 MB; `download_dry_run.sh` fetches a mini version |
| HOI4D [hoi4d-2022] | 2.4M egocentric RGB-D frames, 4,000+ sequences, 800 instances in 16 categories; 3D hands, category-level object pose, meshes, segmentation. FunREC used its laptops, cabinets and safes as an articulation benchmark [funrec-2026] | not checked | not checked |
| DexYCB [dexycb-2021] | Third-person grasps: 582K frames, 20 YCB objects [deepsimho-2023], 10 subjects, object models, camera calibration | CC BY-NC 4.0 | 119 GB, or 12 GB per subject |
| Ego-Exo4D [egoexo4d-2024] | 1,286 h, 740 participants; MPS camera trajectories, point clouds, gaze, 3D hand and body pseudo-GT; object poses not listed | Licence agreement, about 2 days to approve | 38 TB in all (12 TB recommended set); per-take download with `--uids` |
| EPIC-Fields [epic-fields-2023] | SfM cameras for 671 EPIC-KITCHENS videos (pseudo-GT, see §5) plus VISOR masks | Non-commercial | not checked |
| ScanNet++ [scannetpp-2023] | Static indoor scenes: laser scans, DSLR, iPhone RGB-D (1,000+ scenes in v2) | By application | not checked |

### 7. End-to-end video-to-sim systems

- **Google's Argon egocentric demo** (prompt only, no numbers) [google-argon-robotics-2026]:
  - Inputs: the video, candidate sim views, and hand keypoints from an upstream tool.
  - Objects are triangulated from given intrinsics and extrinsics.
  - Checks: force closure, a lift > 2 cm and stable placement, at 500 Hz with no welds.
- **RialTo.** Phone scans plus articulation added in a GUI, about 15 min of active work per scene. Trained from about 15 demos, it averaged 91% / 77% / 75% under pose change, distractors and disturbances. Behaviour cloning on the same 15 demos: 25% / 11% / 5%. It uses default mass and friction [rialto-2024].
- **Video2Policy** (SSv2 internet videos).
  - Simulation success averaged 88% over 12 tasks, against 71% for Eureka.
  - A generalist policy scored 47% on a real Franka (mouse, cup, bread).
  - 42% of reconstruction steps failed [video2policy-2025].
- **Real2Render2Real** (a phone scan plus one human video; kinematic rendering, no physics).
  - Speed: 51 demos/min on one RTX 4090, against 1.7 by teleop.
  - With 1,000 generated demos it was comparable to 150 teleop demos on 5 tasks (15 trials per cell), but no task met the formal equivalence test.
  - It covers prehensile, quasi-static tasks only [r2r2r-2025].
- **EgoEngine.**
  - Inputs: Aria Gen2 hand poses, FoundationStereo depth, SAM 2 and FoundationPose with a GT object mesh. Trajectories are refined in simulation.
  - Real success: 0.35–0.70, against 0.25–0.80 for teleop and 0.00–0.10 for raw human video.
  - Its authors call twin construction the bottleneck [egoengine-2026].
- **Robustness beats exactness** [acdc-2024]. A twin-only policy scored 25% real; digital cousins scored 90%.

## What works, what doesn't, what's efficient

- **Works:**
  - Camera from dynamic handheld video with SLAM-plus-priors (MegaSaM) or new feed-forward models (VGGT-Ω, DA3) [megasam-2025, mint-2026, egoxperience-2026].
  - Hand masking during SLAM [hawor-2025, epic-fields-2023].
  - Text-prompted masks [sam3-2025].
  - Scale from metric depth fitted on static, non-hand pixels [hawor-2025].
  - A twin as the centre of a randomised distribution [acdc-2024, rialto-2024].
- **Doesn't work:**
  - Trusting any single metric-scale source. Self-estimated scale was off by 9–53% on HOT3D, and per-frame metric depth error is 16–30% (§1) [mint-2026, moge2-2025].
  - Single-image 3D as a physics asset without decomposition and checks [sam3d-2025, mujoco-xmlref-2026].
  - Contact or force from egocentric RGB as truth (F1 ≤ 0.27) [egophi-2026].
  - VLM mass as a measurement (factor of 1.3–4 off) [physquantagent-2026, rigpi-2026].
- **Efficient:**
  - Feed-forward camera and depth in seconds [vggt-2025, vggt-omega-2026].
  - Kinematic-only data generation when the task is quasi-static [r2r2r-2025].
  - Reconstructing only the task objects [google-argon-robotics-2026].

## Disagreements and open questions

- **Hand-error protocols disagree.** HaWoR reports PA-MPJPE 4.79 mm on HOT3D [hawor-2025]; MINT's protocol gives HaWoR 66.0 mm PA-MPJPE-p on HOT3D [mint-2026]. Compare tools only under one protocol of our own.
- **Benchmark leaders are unsettled.** VGGT-Ω leads Sintel and HOT3D camera error, but its 1B rows are flagged for possible contamination [vggt-omega-2026, secondhand]. MegaSaM leads HOT3D relative error in another paper [mint-2026].
- **Open:**
  - No study measures how well label-free checks (mask IoU, reprojection, plausibility) track true twin error on real egocentric video.
  - No study validates video-only rigid-body mass or friction against measured values.

## For us

- For us: on footage, the reconstruct-scene "video only" row needs a scale anchor. Without one, expect 10–50% scale error [mint-2026], enough to break any placement or grasp check.
- For us: plausibility and smoothness can only reject a twin, never certify it [deepsimho-2023, hawor-2025]. Any check we adopt must first be calibrated against GT on a benchmark subset.
- For us: Argon's role is the native look (which objects matter, what structure, which anchor) and judging disagreements between tools. Tools give the numbers. This matches Google's demo C, which used upstream hand keypoints [google-argon-robotics-2026].

## Implications for a general reconstruct-scene skill

### Recommended tool stack for this machine

The GPU is an RTX 4070 SUPER with 12 GB. On 2026-10-25, 6.9 GB was in use by in-flight runs, so plan heavy jobs for after those runs. These are our recommendations, not tested here.

| Step | Primary | Second opinion or fallback | Install notes |
|---|---|---|---|
| Camera and depth, dynamic handheld clip | MegaSaM (Apache 2.0) | DA3-Base or Small (Apache 2.0; Streaming < 12 GB); VGGT-1B-Commercial (gated; about 50 frames in 12 GB before the 2026 memory fix) | MegaSaM was tested with torch 2.0.1 and CUDA 11.8. Give it its own env (`uv venv` under `vendors/`), not the pinned `.venv` |
| Long, mostly static clip | MASt3R-SLAM (non-commercial weights; undistort first) | `pycolmap` with frame filtering as in EPIC-Fields | `uv pip install --python .venv/bin/python -c constraints.txt pycolmap` |
| Metric scale | MoGe-2 per frame (weights cached), fitted on static, non-hand pixels | A known-size object; hand size or human height (VideoMimic); DA3-Metric-Large | `moge` package from github `microsoft/MoGe`, on demand |
| Masks | SAM 3 by text prompt (`Sam3VideoModel`, installed in transformers 5.5.4; gated weights) | SAM 2 (`Sam2VideoModel`, installed) | weights on demand |
| Point tracks | CoTracker3 (weights cached) | — | code on demand |
| Hands | HaWoR (world-space; bundles the WiLoR detector, masked DROID-SLAM and Metric3D) | WiLoR per frame on our own cameras | MANO registration; CC-BY-NC-ND; tested with torch 1.13 / 2.0, so a separate env |
| Object geometry | Primitives fitted from masks and metric depth (the skill's code-first rule) | Hunyuan3D-2mini or TRELLIS mesh, rescaled, CoACD collision, `inertia="convex"` | SAM 3D Objects does not fit (≥ 32 GB) |
| Object pose over time | Render-and-compare against masks plus CoTracker tracks | FoundationPose fed MoGe-2 depth (a substitution nobody has validated; 55% failure on SSv2) | — |
| Articulation | Argon-native actor-critic over primitives (Articulate-Anything style), axis fitted to tracked part motion | — | — |
| Physics | Argon category prior → wide randomisation; kinematic cues where visible | — | — |

Kinematic cues: for a free slide, deceleration a gives μ ≈ a/g; bounce height gives restitution. This is our physics reasoning, not a published result.

### Label-free checks to use (each a gate, each calibrated on GT before adoption)

1. **Static reprojection.** Reproject CoTracker tracks on static, non-hand pixels through the recovered cameras and depth. Report px and mm at the working distance. Necessary, not sufficient [brachmann-2021].
2. **Cross-method camera agreement.** Compare two estimators with different priors (e.g. MegaSaM vs DA3) after Sim(3) alignment. Disagreement flags failure. Methods that share a prior can agree while both are wrong [brachmann-2021].
3. **Scale consensus.** Require ≥ 2 independent scale sources (metric depth, a known-size object, hand or human size, table height) to agree within a declared tolerance, e.g. 10%. Scale is the largest error term [hawor-2025, mint-2026].
4. **Held-out mask and depth certificate.**
   - Render each object onto frames the fit did not use.
   - Require mask coverage and a 90th-percentile point distance within ε (the C-3PO form).
   - Expect it to pass wrong poses of symmetric or textureless objects [c3po-ensemble-2023].
5. **Physical gates.** Objects at rest settle by < 2 mm. Penetration stays below a bound. Retargeted grasps hold for 200 ms (displacement < 1 cm), with penetration reported alongside. Use as pass/fail only [deepsimho-2023].
6. **Temporal gate.** Jitter and acceleration bounds, as rejection only [hawor-2025].
7. **Known-error tests for every check.**
   - Inject the error, e.g. scale ±10%, an object moved 2 cm, rotated 15°, frames offset in time.
   - The check must fail on the corrupted input and pass on the clean one ([evaluation-and-statistics](evaluation-and-statistics.md)).

### GT benchmarks to calibrate the checks

For each check, report the Spearman correlation between check score and true error over the subset, and adopt only checks that track truth.

1. **HOT3D-Clips (Aria).** 20 clips, about 4 GB, 5 s each. Gives camera, MANO hands, object poses and meshes. Calibrates checks 1–5 and hand world error. Caveats: fisheye RGB plus monochrome streams; hand data is non-commercial.
2. **ADT.** The 500 MB sample, then 3–5 sequences. Gives GT depth, object poses and scene geometry. Calibrates scale consensus, depth and render-and-compare.
3. **ARCTIC mini** (dry-run download), or one DexYCB subject (12 GB). Gives articulated objects, contact and third-person views. Calibrates articulation-axis error and the physical gates.

### Honest limits: what video alone cannot recover

- **Mass, inertia, stiffness, damping, and the forces at contact.** Only priors are possible: VLM mass is a factor of about 1.3 off on average and up to 4× on some objects [physquantagent-2026, rigpi-2026]. Friction is recoverable only from visible sliding.
- **Metric scale without an anchor**: 9–53% error on HOT3D (§1) [mint-2026].
- **Unseen geometry.** Back sides, interiors, thin, transparent and reflective parts are hallucinated or missed [sam3d-2025, moge2-2025, physquantagent-2026].
- **Grip force, in-hand slip, and contact location.** Contact F1 ≤ 0.27 even in distribution [egophi-2026].
- **Joint limits, springs and friction in articulations**, beyond the range shown.
- **Zoom, strong distortion, and scenes dominated by motion** break camera recovery [megasam-2025].
- **A twin that matches the video can still be physically wrong.** Expect it to over-score and to miss real failures, as on REASSEMBLE ([real2sim-twins](real2sim-twins.md), [recon-effects-2026]). Train over a distribution, not the point estimate [acdc-2024].
