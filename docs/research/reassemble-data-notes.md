# REASSEMBLE data notes

Written 2026-10-24 from a local subset of the Hugging Face port plus the original record and code. Every number below comes from a file or page listed under Sources. "Measured" means we computed it from the downloaded data with `scripts/inspect_reassemble.py` or a short check in the same environment.

## What is on disk

| Path | What | Source |
|---|---|---|
| `data/reassemble/meta/` | all HF meta: `info.json`, `tasks.parquet`, `episodes/chunk-000/file-000.parquet`, `stats.json`, `splits.json` | [S1] |
| `data/reassemble/data/chunk-000/file-000.parquet` | 47 MB, episodes 0–14 (148,444 frames) | [S1] |
| `data/reassemble/videos/observation.images.{hand,hama1,hama2}/chunk-000/file-000.mp4` | 199 / 159 / 129 MB, AV1, each starts with episodes 0 and 1 | [S1] |
| `data/reassemble/frames/` | 3 stills, episode 0, frame 2117 | made by the script |
| `data/reassemble-tudata/` | `poses.zip` (153 per-recording JSONs, extracted to `poses/`), `splits.zip`, `README.txt` from the original record | [S2] |
| `vendors/reassemble/` | shallow clone of the authors' repo (reference only) | [S3] |
| `vendors/reassemble-lerobot-port/` | shallow clone of the HF port's conversion code (reference only) | [S4] |
| `vendors/nist-taskboard/` | NIST Task Board 1 CAD, STL, laser-plate and connector-housing zips, assembly instructions | [S6] |

Free disk before download: 301 GB. Total on disk afterwards: about 680 MB (data 513 MB, vendors 167 MB including the extracted NIST STLs).

Rerun: `uv run --no-project --with pyarrow --with pandas --with numpy --with av --with pillow python -I scripts/inspect_reassemble.py data/reassemble --poses-dir data/reassemble-tudata/poses --frames`. Run it from the repo root with `--no-project`: running `uv run` inside `vendors/reassemble` makes uv try to install that repo's `pyproject.toml`.

## Dataset shape

- 149 episodes, one per recording, 1,433,406 frames at 30 fps (796 min). Episode length 23–683 s, median 296 s. Splits are 111 train, 37 test and 1 unassigned. [S1 meta]
- All sensors are resampled to the hand-camera clock by nearest neighbour, with no interpolation. [S4 `port_reassemble.py` `_resampled_state`, S1 README]
- `observation.state` (36) is the joint positions [0:7], joint velocities [7:14], joint efforts [14:21], gripper [21:23], EE pose [23:30] and EE velocity [30:36]. [S4 `STATE_PARTS`] Measured: the slices equal the named sub-features exactly.
- The EE pose is `x, y, z, qw, qx, qy, qz` in the robot base frame. [S2 README; the authors' RLDS converter `h5_to_rlds.py` line 706 also reads it as w-first] Measured: the first frame's quaternion is (−0.009, 0.999, 0.026, 0.040), which is w≈0 with the gripper pointing down.
- The EE pose is Franka's `O_T_EE`. [S3 `record_teleop/scripts/franka_to_geometry_messages.py`] Measured: running Franka DH forward kinematics on `joint_position` and comparing with `ee_pose` gives a fixed flange-to-EE offset of 0.174 m along flange z, rotated −45° about z (std 0.06 mm and 0.02° over 2,969 frames). This is longer than the 0.1034 m default Franka hand TCP [S3 `franka_robot_FT.xacro` `tcp_xyz`] because the F/T sensor stack sits in between. The repo's URDF stack adds up to about 0.148 m.
  - **Corrected 2026-10-24: 0.174 m is the configured EE frame, not the fingertips.**
    - A bundle adjustment of both static cameras on about 4,300 forward-kinematics points puts the physical F/T stack at 47.7 mm. That places the finger pads about 23 mm short of `O_T_EE`.
    - This agrees with the URDF (stack 44.5 mm, about 0.148 m to the TCP). The earlier "trust the data value" mistook the configured EE frame for the physical tool.
    - Use `O_T_EE` for commands and logs. Since 2026-10-24 21:19, `twin/scene/base.xml` mounts the hand on episode 0's fitted stack (47.7 mm) and keeps the `tcp` site on `O_T_EE`. Another recording's `tcp.hand_offset_m` in `twin/calib/` minus base.xml's is its residual (`twin.world.calib_hand_offset`).
- Gripper: two finger positions, 0.04 m each when open and 0.001 m when closed. [measured]

## Tasks (69 labels)

There are 4 verbs × 17 parts = 68 labels, plus `No action.` (task 17). The paper calls that label "Idle". [S1 `tasks.parquet`, S5 §IV-A]

- Verbs: Pick, Insert, Remove, Place. In this data, `Remove` means pull out of the board and `Place` means put down on the table. Episode 0 is all Remove/Place pairs.
- 17 parts: BNC, D-SUB, Ethernet, USB, waterproof (5 connectors); small gear, medium gear, large gear; bolt 4; round peg 1–4; square peg 1–4.
- "bolt 4" is the paper's M16 nut, which it also calls "nut 4". The other three nuts (M4, M8, M12) were left out. [S5 §III-A, App. C]
- The NIST parts behind these names [S6 STL zip and file names, mm]: round pegs are 4/8/12/16 mm × 50 mm. The "square" pegs are keys of 4×4, 8×7, 12×8 and 16×10 mm × 50 mm. Gears are module 1 with 20/40/60 teeth (about 22/42/62 mm across). Which of pegs 1–4 maps to which size is not stated anywhere we found.
- Full label list: `meta/tasks.parquet` (the script prints it).

## Episode 0

Episode 0 is recording `2025-01-09-14-11-58` (HF test split; train-only in `playbook/splits.json`, because its labels are printed here), 11,643 frames = 388.1 s. [S1 `splits.json`, episodes meta] It has 37 segments with indices 0–36 in order. Times are derived from runs of `segment.index`.

| seg | start s | end s | ok | task |
|---|---|---|---|---|
| 0 | 0.00 | 3.30 | Y | No action. |
| 1 | 3.30 | 16.70 | Y | Remove waterproof. |
| 2 | 16.70 | 22.50 | Y | Place waterproof. |
| 3 | 22.50 | 32.67 | Y | Remove D-SUB. |
| 4 | 32.67 | 37.67 | Y | Place D-SUB. |
| 5 | 37.67 | 51.97 | **N** | Remove small gear. |
| 6 | 51.97 | 81.37 | **N** | Remove small gear. |
| 7 | 81.37 | 95.00 | Y | Remove round peg 4. |
| 8 | 95.00 | 101.57 | Y | Place round peg 4. |
| 9 | 101.57 | 120.70 | Y | Remove square peg 1. |
| 10 | 120.70 | 128.77 | Y | Place square peg 1. |
| 11 | 128.77 | 141.10 | Y | Remove USB. |
| 12 | 141.10 | 145.70 | Y | Place USB. |
| 13 | 145.70 | 158.33 | Y | Remove large gear. |
| 14 | 158.33 | 163.83 | Y | Place large gear. |
| 15 | 163.83 | 174.00 | Y | Remove medium gear. |
| 16 | 174.00 | 179.33 | Y | Place medium gear. |
| 17 | 179.33 | 204.23 | **N** | Remove small gear. |
| 18 | 204.23 | 215.87 | **N** | Remove Ethernet. |
| 19 | 215.87 | 225.30 | Y | Remove Ethernet. |
| 20 | 225.30 | 231.37 | Y | Place Ethernet. |
| 21 | 231.37 | 243.80 | Y | Remove square peg 2. |
| 22 | 243.80 | 250.30 | Y | Place square peg 2. |
| 23 | 250.30 | 262.87 | Y | Remove round peg 2. |
| 24 | 262.87 | 268.97 | Y | Place round peg 2. |
| 25 | 268.97 | 280.20 | Y | Remove square peg 4. |
| 26 | 280.20 | 286.00 | Y | Place square peg 4. |
| 27 | 286.00 | 295.67 | Y | Remove square peg 3. |
| 28 | 295.67 | 300.57 | Y | Place square peg 3. |
| 29 | 300.57 | 309.17 | Y | Remove round peg 3. |
| 30 | 309.17 | 314.90 | Y | Place round peg 3. |
| 31 | 314.90 | 336.63 | Y | Remove BNC. |
| 32 | 336.63 | 342.90 | Y | Place BNC. |
| 33 | 342.90 | 357.23 | Y | Remove bolt 4. |
| 34 | 357.23 | 362.00 | Y | Place bolt 4. |
| 35 | 362.00 | 386.27 | **N** | Remove small gear. |
| 36 | 386.27 | 388.10 | Y | No action. |

Five segments failed: the small gear four times and Ethernet once.

Caveat: in gaps between annotated segments, the port carries the previous segment's label forward. Each end time above therefore runs to the next segment's start, not to the annotated end. The exact start/end times and the low-level skill labels (Grasp, Lift, Approach, Align, Release, Twist, Push, Pull, Nudge) exist only in the original HDF5. [S4 `_active_segment`, `_read_segments`; S5 §III-D]

## Segment and failure counts

- Dataset-wide from meta: summing `stats/segment.index/max + 1` over episodes gives 4,849 segments (3–55 per episode, median 37), with `No action.` included. [S1 episodes meta]
  - The paper reports 4,551 demonstrations and 4,035 successes, so 516 failed, over 781 min. [S5 abstract, S2 README] 4,849 − 4,551 = 298 = 2 × 149, which fits roughly one idle segment at each end of every recording. This is an inference, not checked against the HDF5.
- Meta has no per-segment failure counts. It only gives the frame-weighted share of frames inside successful segments: 0.852. [S1 `stats.json`]
- Measured in the local file (episodes 0–14): 499 segments, of which 31 are `No action.` and 468 are actions. 65 of the actions failed:

| verb | segments | failed |
|---|---|---|
| Insert | 143 | 39 |
| Pick | 135 | 13 |
| Remove | 96 | 12 |
| Place | 94 | 1 |

Insert fails most often (27%), and failed inserts are retried without a new Pick.

## Force / torque

- `observation.force` and `observation.torque` are the raw sensor-frame readings, with gravity and bias not removed. Measured: during idle the force averages (2.1, −0.2, −13.6) N, which is the tool weight. [S1 README "measured F/T"; measured]
- `observation.force.base` and `observation.torque.base` are gravity- and bias-compensated, then rotated into the robot base frame. [S1 README; S3 `aidin_ros/scripts/force_2_base.py`, `aidin_ros/launch/ft_calibrated.launch`]
  - Sign: this is the external force acting on the tool. Measured: in Insert contact frames F_z > 0 in 100% of frames (mean +7.6 N, the board pushing back). In Remove contact frames F_z averages −7.6 N.
- Sensor: AIDIN AFT200-D80-C on the flange. [S5 §III-B] Source rates are about 1.7 kHz raw and about 0.5 kHz base, down-sampled to 30 fps in the port. [S1 README]
- Dataset-wide ranges, per axis x / y / z [S1 `stats.json`]:

| feature | min | q01 | q50 | q99 | max |
|---|---|---|---|---|---|
| force.base N | −32.9 / −19.9 / −29.5 | −2.3 / −1.8 / −4.0 | 0.2 / −0.2 / −0.2 | 2.0 / 2.3 / 7.5 | 18.9 / 24.4 / 72.2 |
| torque.base Nm | −14.5 / −29.6 / −8.8 | −2.7 / −4.8 / −1.1 | −0.8 / −0.6 / −0.1 | 2.1 / 2.9 / 1.4 | 11.4 / 12.1 / 7.6 |
| force (raw) N | −30.3 / −10.9 / −88.1 | −0.2 / −2.6 / −21.1 | 2.1 / −0.1 / −13.4 | 5.0 / 2.5 / −9.6 | 22.2 / 12.3 / 16.9 |
| torque (raw) Nm | −2.5 / −6.5 / −2.7 | −0.3 / −0.4 / −0.2 | 0.1 / 0.1 / 0.0 | 0.6 / 0.7 / 0.2 | 1.9 / 5.3 / 2.6 |

torque.base includes the r × F moment about the base origin, which is why it is much larger than the sensor-frame torque. [S3 `force_2_base.py`]

- Contact threshold: 5 N on |force.base|. This is the teleop's own default `follower_contact_force_threshold`. [S3 `reassemble_controllers/cfg/teleop_param.cfg`] Measured: idle |force.base| has p50 0.45 N and p90 0.76 N. 10.3% of local frames are above 5 N, and 72% of those are in Insert segments.
- Known bad F/T: the last action of `2025-01-11-14-45-48` (local episode 7), `-15-27-19` and `-15-35-08`. [S2 README]

## Setpoint question: action vs measured pose

**Answer: `action[t][:7]` is exactly the measured `ee_pose[t+1]`. It is not a commanded target.**

- The port builds it that way: `action = next absolute [ee_pose, gripper]; final frame repeats last`. [S4 `port_reassemble.py` lines 209–211] The authors' own RLDS converter does the same ("Actions are the next pose and gripper position"). [S3 `scripts/conversions/h5_to_rlds.py` lines 833–835]
- Measured over all 148,429 frame pairs in episodes 0–14: max |action[t][:7] − ee_pose[t+1]| = 0, max |action[t][7:9] − gripper[t+1]| = 0, and the last frame repeats its own pose.
- The commanded target was never recorded. The impedance controller takes its target on the `equilibrium_pose` topic [S3 `cartesian_impedance_controller_damping_ratio.cpp`], and that topic is not in the rosbag list. [S3 `record_teleop/launch/record.launch`] The haptic device pose is not in the list either. The original HDF5 has only measured state. [S2 README]

Gap between action[t] and ee_pose[t], which is one 33 ms step of measured motion (measured):

| frames | share | \|Δp\| mm p50 / p90 / p99 / max | Δangle ° p50 / p90 / p99 / max |
|---|---|---|---|
| all | 100% | 0.77 / 4.3 / 8.2 / 31 | 0.14 / 0.64 / 1.6 / 4.6 |
| contact (\|F\| > 5 N) | 10.3% | 0.07 / 0.32 / 1.3 / 31 | 0.13 / 0.78 / 1.4 / 2.5 |
| free | 89.7% | 0.93 / 4.5 / 8.3 / 13 | 0.14 / 0.62 / 1.6 / 4.6 |

- Δp equals the recorded EE velocity × 1/30 s (median residual 0.13 mm), so the step is just motion.
- It does not track force. In contact, a least-squares fit of Δp on F_base gives R² of 0.001 / 0.003 / 0.000 per axis, and corr(F_i, Δp_i) is 0.02 / 0.05 / −0.10. The "implied stiffness" from this fit (2e5–7e5 N/m) means nothing.
- What is missing: under a Cartesian impedance law the static target sits at x_cmd ≈ x − K⁻¹F_base (with F_base as the external force on the tool). At the controller's default K = 500 N/m that gap is |F|/K = 15 mm at the median of contact frames (p90 24 mm, p99 34 mm). That is about 200× the gap the action labels show.
- Stiffness actually used: the code default is 500 N/m translational, 50 Nm/rad rotational and damping ratio 1. It is tunable live through dynamic reconfigure, and commented-out older values are 200/10 and 800/75. [S3 `reassemble_controllers/cfg/compliance_param.cfg`] The paper gives no numbers. [S5 §III-C]
  - Rough check from the data (measured): when contact suddenly releases, the arm jumps toward its unseen target. |F| divided by the jump gives K ≈ 730 N/m (median of 20 events above 5 N, IQR 460–1,400) or ≈ 390 N/m (3 events above 8 N). That is the right order of magnitude for 500 N/m but too noisy to fit.
- What this means for us: replaying these actions as impedance targets in sim gives near-zero contact force during insertion. To train force-aware policies, either track action as a desired pose with a stiff controller, or rebuild the target as x − F_base/K with an assumed K of about 500 N/m.

## Camera calibration

- **Intrinsics: none anywhere.**
  - The HF `info.json` has only codec and size (AV1, 480×640, 30 fps). [S1]
  - No `camera_info` topic was recorded. [S3 `record.launch`]
  - The HDF5 layout has no calibration fields. [S2 README]
  - The paper describes no intrinsic calibration. [S5 §III-B]
  - Hardware: hand camera is an Intel RealSense D435i, colour stream at 640×480/30 fps through `realsense2_camera` with depth off [S3 `record.launch`, S5 §III-B]. Its RGB FOV is often quoted as 69.4° × 42.5° for 16:9 (secondary source [S9]). The 640×480 factory intrinsics are per unit and were not saved. External cameras are two HAMA C-600 Pro webcams via `usb_cam`, 640×480 YUYV [S3 `record.launch`, S5]. Event camera is a DAVIS346, 260×346 [S1, S2].
- **Extrinsics: external cameras only, from motion capture.**
  - `poses.zip` in the TU Wien record has one JSON per recording (153 files, covering all 149 HF episodes). Each gives `Hama1`, `Hama2`, `DAVIS346` and `NIST_Board1` as `[x,y,z]` in metres and `[qx,qy,qz,qw]` in the mocap world frame, measured once at the start of each recording. [S2 README, S5 §III-B/D]
  - The poses track marker brackets, not optical centres. [S5 §III-B]
  - The repo's code to write these files is commented out. [S3 `record_manager.py` `calibrate_poses`]
- **Hand camera: no pose anywhere.** The repo has no camera link in the URDF/xacro and no mount model. [S3, searched for realsense/camera/calib] The paper only says "wrist-mounted". [S5 §III-B]
  - Our read of the stills (not documented): a white plate clamped between the F/T sensor and the Franka hand sticks out to one side, and the camera looks down past the hand. The fingertips sit at the bottom centre of the hand image (`data/reassemble/frames/ep000_f02117_*.png`). It needs hand-eye calibration from images.
- **The mocap world frame matches the robot base frame (measured).**
  - Method: map the lowest EE point of each successful Insert into the board frame using that recording's `NIST_Board1` pose.
  - Result: the same object lands within 7 mm (median over objects) across 9 recordings. Board yaw ranged from −180° to +161° and positions were up to 0.9 m apart. In raw robot coordinates the same points spread 182 mm.
  - This also confirms the quaternion order `[qx,qy,qz,qw]`.
- **Problem: Hama1/Hama2 poses are often wrong.** In 67 of 153 files the two cameras sit within 5 cm of each other but with different orientations, and in 11 more they are swapped relative to the majority layout. Episode 0 is one of them (2 mm apart). Its hama1 and hama2 stills clearly show two different viewpoints, so at least one pose is wrong. Check each recording's camera pose against the images before using it. DAVIS346 poses are stable (std ≤ 7 mm). [measured from S2 `poses/`]

## Geometry assets

From the REASSEMBLE repo [S3]; everything listed, nothing else exists:

| What | Path (under `vendors/reassemble/`) | Units, size |
|---|---|---|
| Peg holders (3D-printed, REASSEMBLE-specific) | `3d_models/square_peg_holder.STL`, `3d_models/circular_peg_holder.STL` | mm, 50 × 25 × 90 each |
| Cable holder | `3d_models/cable_holder.STL` | mm, 30 × 30 × 30 |
| F/T sensor AFT200-D80-C | `catkin_reas/src/aidin_description/meshes/AFTsensor.stl` (also `thick/AFT200-D80-C v3.stl`) | m, 102 × 78 × 21.5 mm |
| Flange and hand mounts | `.../aidin_description/meshes/optimized/{FlangeMount,HandMount}.stl` (used), plus `AFT*mount.stl` and `thick/*` variants | m |
| Old copies | `catkin_reas/src/aidin_ros/meshes/*_old.stl` | mm (not m) |
| Sensor stack URDF | `.../aidin_description/sensor/aidin_sensor_optimized.xacro`, attached in `.../reassemble_description/robots/common/franka_robot_FT.xacro` (+5 mm, then sensor +30.5 mm, hand +9 mm, yaw −135°) | m |
| Franka arm and hand | `.../reassemble_description/meshes/{visual/*.dae, collision/*.stl}` with `robots/{panda,fr3}/*.xacro` | m (DAE `<unit meter="1">`) |

Not in the repo: the board, the redesigned Ethernet/USB holders, the camera mounts and the hand-camera bracket. The paper says the modified parts are "on the project website" [S5 §III-A], but the site links only the dataset and code [S7]. The REASSEMBLE board has a matte white PVC base instead of acrylic. [S5 §III-A]

NIST Assembly Task Board 1 [S6]. Downloaded to `vendors/nist-taskboard/` because everything is under 200 MB:

| File | Size | Contents |
|---|---|---|
| `taskboard_1_cad.zip` | 52.6 MB | SolidWorks only (76 SLDPRT, 7 SLDASM incl. `GMC_Competition_Task_Board_Virtual.SLDASM`), no STEP |
| `taskboard_1_stl.zip` (extracted to `taskboard_1_stl/`) | 12.7 MB | 34 STL, **mm**. Plate `GMC_Laser_Plate_Virtual.STL` is 384 × 384 × 9 mm. Gears, connectors and nuts sit in board coordinates; pegs and keys sit at their own origin |
| `connector_housings.zip` | 49 KB | `Ethernet_Housing.STL`, `DB_Holder.STL` (mm) |
| `gmc_laser_plate.zip` | 148 KB | SVG / AI / SLDDRW cut file |
| `assembly_instructions_v4.docx` | 2.8 MB | build instructions |

- Licence: no licence file in the zips. NIST's site says its information "may be distributed or copied" with credit requested. NIST software is not subject to US copyright. [S8]
- **Corrected 2026-10-24: the board is the NIST laser plate, mirrored.**
  - An earlier 2D rigid fit of NIST STL part centres left 4–22 cm residuals because it did not allow a mirror.
  - With one mirrored rigid transform, all 17 contact-estimated holes land on NIST laser-plate features at 0.76 mm RMS. Source: `board_contact_layout` tool; the method is in `playbook/skills/board-from-contacts/`.
  - The physical cause of the mirror is unknown.
  - Peg N is 4N mm: round pegs are 4/8/12/16 mm and the keys are 4×4 / 8×7 / 12×8 / 16×10. This comes from hole positions and grip widths.
- Mocap `NIST_Board1` z is not the plate top: on episode 0 the plate top sits 30.3 mm (±4) above it (plate corners triangulated by the robot-calibrated cameras; `playbook/skills/world-frame` Lessons).
- Mocap board poses can be stale. Gate episode 117's is 82 mm and 31° off, and contacts match the board of a recording made 7 minutes later. Check each recording's board against its contacts.

## Still frames

`data/reassemble/frames/ep000_f02117_{hand,hama1,hama2}.png`: episode 0, t = 70.567 s (frame 2117), during "Remove small gear." (a failed try). This is the frame with peak |force.base| in a Remove segment: EE at (0.723, −0.032, 0.048) m, F_base (1.8, 3.7, −13.2) N. All three decoded at exactly the target time. The cameras share one 30 fps grid, so one frame index means one time. [S1, S4]

## Surprises

1. `action` is just the next measured pose, so the dataset holds no commanded setpoint and no stiffness.
2. Hama1/Hama2 mocap poses coincide or swap in about half the recordings.
3. Episode 0 is pure disassembly (Remove/Place only), with no Pick/Insert.
4. Insert fails 27% of the time locally.
5. The logged TCP (`O_T_EE`) sits 0.174 m from the flange, but the physical stack matches the URDF (about 0.148 m to the TCP): `O_T_EE` lies about 23 mm beyond the finger pads (corrected 2026-10-24, see Dataset shape).

## Sources

- [S1] Hugging Face `robot-lev/reassemble`, README and `meta/*`: https://huggingface.co/datasets/robot-lev/reassemble (local copy `data/reassemble/`)
- [S2] TU Wien record 10.48436/0ewrv-8cb44, `README.txt`, `poses.zip`, `splits.zip` (CC-BY-4.0): https://researchdata.tuwien.ac.at/records/0ewrv-8cb44 (local copy `data/reassemble-tudata/`)
- [S3] https://github.com/TUWIEN-ASL/REASSEMBLE (MIT), shallow clone at `vendors/reassemble/`
- [S4] https://github.com/lvjonok/reassemble-lerobot-port, `scripts/port_reassemble.py`, clone at `vendors/reassemble-lerobot-port/`
- [S5] Sliwowski et al., "REASSEMBLE", arXiv 2502.05086: https://arxiv.org/abs/2502.05086 (HTML version read for §III–IV and appendices)
- [S6] NIST Assembly Performance Metrics and Test Methods, Task Board #1 downloads: https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/assembly (files under https://www.nist.gov/system/files/documents/…)
- [S7] Project page: https://tuwien-asl.github.io/REASSEMBLE_page/
- [S8] NIST copyright and disclaimers: https://www.nist.gov/copyrights-disclaimers
- [S9] D435i RGB FOV (secondary): https://www.cnx-software.com/2018/11/14/intel-realsense-d435i-stereo-depth-camera-6-dof-tracking/
