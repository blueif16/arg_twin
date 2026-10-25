# Footage → LIBERO scene: scout report (2026-10-25, 16:06 PDT)

Everything is sim-only. Code is in `playgrounds/footage/`, outputs are in `runs/footage/`, and clips are in `data/web-footage/<slug>/` (untrusted, read with `python -I`).

## 1. Scene path: works, about 20–45 s per new scene

A custom BDDL loads directly. Nothing has to be registered, and no init file or LIBERO suite entry is needed. All three tests passed on the first try, with the chain's 5 sims running at the same time:

| Test | Build | Reset | Total incl. render + 50 steps | Goal at reset | Goal after privileged teleport |
|---|---|---|---|---|---|
| Custom: red mug, plate, black bowl, new layout (`bddl/fx_kitchen_put_the_red_mug_on_the_plate.bddl`) | 17.4 s | 15.3 s | 39.5 s | False | True (from step 2, stays True) |
| Custom: white bowl and microwave, door open, goal `In heating_region` + `Close` (`bddl/fx_kitchen_put_the_white_bowl_in_the_microwave_and_close_it.bddl`) | 9.9 s | 8.4 s | 21.5 s | False | True (both predicates) |
| Fallback: LIBERO-90 `LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate` with the mug's free joint moved to (0.08, −0.20) | 20.6 s | 20.8 s | 43.6 s | False | True |

Renders: `runs/footage/fx_scene_*/agentview_{reset,teleported}.png`. The footage-and-twin pair is `runs/footage/side_by_side/microwave_side_by_side.png`.

### Recipe (`.venv-libero`, `MUJOCO_GL=glx`)

```
.venv-libero/bin/python -I playgrounds/footage/fx_bddl_load.py <file.bddl> <out_dir> [--mode override]
#   env: FX_TARGET_SITE=microwave_1_heating_region  FX_CLOSE_JOINT_OF=microwave_1   (for In/Close goals)
```

What the script does:
- Sets `LIBERO_CONFIG_PATH=runs/libero/config` and `MUJOCO_GL=glx`, with BLAS threads at 1.
- Calls `OffScreenRenderEnv(bddl_file_name=<any path>, camera_heights=256, camera_widths=256)`, then `env.seed(142)`, `env.reset()` and 10 no-op steps.
- Saves `obs["agentview_image"][::-1]` (the true view), runs `env.check_success()` and saves `env.get_sim_state()` as `init_state_0.npy`.
- Test only, never an actor op: teleports the goal object onto its target body or site, closes the fixture joint, steps 40 times and logs the success trace plus each goal predicate.

### BDDL rules (learned from writing two)

- The problem name must be one of LIBERO's 6 problem classes, looked up lowercased: `LIBERO_Kitchen_Tabletop_Manipulation`, `..._Living_Room_...`, `..._Study_...`, `LIBERO_Tabletop_Manipulation`, `..._Coffee_Table_...` and `..._Floor_...`. The class fixes the table: the kitchen table top is at z ≈ 0.90, the living room's at z ≈ 0.43.
- Each region is `(:ranges ((x_min y_min x_max y_max)))` in metres in the table frame, with a `(:yaw_rotation ((a b)))`. In `:init` a region is named `<target>_<region>`, for example `kitchen_table_plate_init_region`.
- A region on a fixture (`heating_region`, `top_side`, a cabinet's `top_region`) has to be declared in `:regions` with `(:target microwave_1)`, and is then referenced as `microwave_1_heating_region`.
- Objects are written `name_1 - category`, and the category must be in `OBJECTS_DICT` (list below). Fixtures are listed under `:fixtures`. Articulated state goes in `:init`, for example `(Open microwave_1)`.
- Goals: `On`, `In`, `Open`, `Close`, `TurnOn`, `TurnOff`, joined with `And`.
- **Frame:** the robot base is at −x and the agentview camera sits at +x looking back. In the true view (`[::-1]`), world +y is on the image's right. If the footage camera is mapped to the agentview camera, a layout sketch written as "x toward the camera, y to the camera's right" drops straight into region ranges. This is the frame my Argon prompt asks for.

### Gotchas

1. No init-state file exists for a custom BDDL. `reset()` samples placements in the regions, seeded by `env.seed`. Save `get_sim_state()` after settling and replay it with `set_init_state`.
2. Keep regions apart. The microwave is about 0.3 m wide, so put it at |y| ≥ 0.25 and keep other objects ≥ 0.15 m from it. Overlapping regions make the placement sampler retry or fail.
3. The frozen evaluator and `playgrounds/libero/server.py` only know libero_10/90 BDDLs: the `frozen.json` hash covers those files only, and `reset` takes a suite `task_id`. Practice on a custom scene would need a server `reset_bddl` op that hashes the custom file. That change isn't needed for a run that only builds the front end.
4. Looks differ: LIBERO's microwave is black and the footage's is white. Wood and lighting are fixed per problem class. The twin is semantic, not photometric.
5. A fresh process pays about 1 s of imports. Build and reset scale with object count and CPU load (8–21 s each here). Memory is about 2 GB per process.
6. The robosuite "No private macro file" warnings are harmless.

### LIBERO assets for kitchen tabletop scenes (from `OBJECTS_DICT`)

**Graspable objects:**
- Bowls, plates and mugs: `akita_black_bowl`, `white_bowl`, `plate`, `porcelain_mug`, `red_coffee_mug`, `white_yellow_mug`, `glazed_rim_porcelain_ramekin`
- Cookware and trays: `moka_pot`, `chefmate_8_frypan`, `basket`, `wooden_tray`, `white_storage_box`, `desk_caddy`, `dining_set_group`
- Packaged food: `ketchup`, `milk`, `orange_juice`, `butter`, `cream_cheese`, `alphabet_soup`, `tomato_sauce`, `bbq_sauce`, `salad_dressing`, `new_salad_dressing`, `mayo`, `macaroni_and_cheese`, `chocolate_pudding`, `cookies`, `popcorn`
- Produce: `corn`, `cherries`
- Other: `wine_bottle`, `black_book`, `yellow_book`

**Fixtures:**

| Fixture | What it offers |
|---|---|
| `kitchen_table` | the table |
| `microwave` | Open/Close door, `heating_region`, `top_side` |
| `wooden_cabinet` | 3 drawers (`top_region`, `middle_region`, `bottom_region`), `top_side` |
| `white_cabinet`, `short_cabinet`, `slide_cabinet`, `short_fridge` | storage |
| `flat_stove` | TurnOn/TurnOff, cook region |
| `wine_rack`, `wooden_shelf`, `wooden_two_layer_shelf`, `rack`, `bowl_drainer` | shelving and racks |
| `faucet`, `basin_faucet`, `window`, `target_zone` | other |

**No asset:** cutlery, napkins, drinking glasses, kettles or flasks, cans other than the listed foods, and anything deformable.

## 2. Footage

| Slug | URL | Licence | Clip | Task | LIBERO mapping |
|---|---|---|---|---|---|
| `put-bowl-in-microwave` | https://www.youtube.com/watch?v=L6v9HiSvgFA (Life Skills) | standard YouTube | 11 s, 720p, static camera at counter height | open the microwave, put the white bowl in, close it | **Strong.** `microwave` and `white_bowl`; the goal is the exact LIBERO predicate pair, and the BDDL is already built and checked |
| `pov-setting-the-table` | https://www.youtube.com/watch?v=lAFxDo21Ebs (Blue Apple Learning) | standard YouTube | 51 s, 720p, top-down POV, static | plate, napkin, fork, knife, spoon, glass | **Partial.** Only `plate` maps; a glass becomes a mug at best; cutlery and napkin have no asset. Good for an honest "partial twin" verdict |
| `microwave-video-model` | https://www.youtube.com/watch?v=gMhgxeCGwMo (Julie Sindelar) | standard YouTube | 32 s, 720p, third-person, static | plate of food into the microwave, close, start, take it out, carry it to a table | **Medium.** `microwave` and `plate`; the food is deformable; it repeats the microwave scene |
| `put-away-plates` | https://www.youtube.com/watch?v=Wf8VBx-pXE8 (Kara Dillon) | standard YouTube | 40 s, handheld and shaky | stack plates into a cabinet | **Weak.** `plate` and a cabinet; heavy camera motion; stacking in a cabinet isn't LIBERO-like |
| `origami-crane` | https://www.youtube.com/watch?v=5AtvzGUe9no (Ventuno Art) | standard YouTube | 0–60 s trim | paper folding | dropped per coordinator (handled in `runs/argon/hindsight/`) |

- Each folder has `video.mp4`, `video.info.json` and `source.json` (URL, title, uploader, licence, duration).
- None of the usable clips is Creative Commons. The only CC hit, "Morning Coffee POV", turned out to be an AI-generated montage, so I deleted it.

**HOT3D (`data/hot3d-clips/extracted/`, 150 frames = 5 s each, egocentric fisheye, ground-truth 6-DoF object poses):**

| Clip | Objects | Motion | Verdict |
|---|---|---|---|
| **clip-002066** | `can_tomato_sauce`, `carton_milk`, `carton_oj`, `can_parmesan`, `flask`, `holder_gray` on a wooden table | Tomato-sauce can picked up and moved (0.30 m max, 0.087 m net). Flask moved 0.355 m (no asset). Milk and OJ static | **Usable.** Maps to `tomato_sauce`, `milk`, `orange_juice` (parmesan stands in as `alphabet_soup`). Lets us measure twin placement error against ground truth, using pairwise inter-object distances (frame-free) |
| clip-002210 | vase, bowl, bamboo plate, tomato can | only the vase moves (no asset) | static layout only |
| clip-001849 | office: keyboard, mouse, phone | — | no |

Frames are JPG sequences, so they need `ffmpeg -framerate 30 -i %06d.image_214-1.jpg` to become an mp4 first.

## 3. Argon's quick read (`put-bowl-in-microwave`)

Script: `playgrounds/footage/fx_argon_read.py`. It runs on `gemini-4-argon` through the Interactions API with `thinking_level` medium:
- **Call 1:** the Files API video at static 1 fps, asking for draft steps and up to 2 zoom windows.
- **Call 2:** chained with `previous_interaction_id`, sending a 3×3 frame grid of the requested window and asking for the final JSON.

Record: `runs/footage/argon_read/put-bowl-in-microwave/read.json`.

**Result:**
- Task: "open the microwave, put the white bowl inside it, and close the microwave".
- Steps:
  - open, 1.8–3.5 s
  - place bowl, 3.8–5.8 s
  - close, 6.0–7.6 s
  - all marked succeeded.
- Objects: `microwave` (fixture), `white_bowl` (manipulated). The green crate and white basket were mapped to `basket` as distractors; the cardboard box has no asset.
- Layout sketch: microwave at the centre, bowl to its right (+y 0.35), crate to its left. This matches the frames.
- Success conditions: `In(white_bowl, microwave_heating_region)` and `Close(microwave)`. These are LIBERO's own predicates and the same goal as my hand-written BDDL.
- `twin_supported`: yes (0.95). It noted honestly that the real door opens with a push button.

**Cost:**

| Step | Time | Tokens | Cost |
|---|---|---|---|
| Upload | 9.0 s | — | — |
| Call 1 | 8.7 s | 1,329 in / 217 out | $0.0048 |
| Call 2 | 9.8 s | 3,166 in / 653 out | $0.0129 |
| **Total** | **27.5 s** | | **$0.018** |

A 50 s clip should land around $0.03–0.06.

## 4. Origami

Dropped per the coordinator; another agent covers it in `runs/argon/hindsight/`.

## 5. Recommendation for 16:20 → 16:50 (front end only, no practice)

**Footage:**
1. `put-bowl-in-microwave`: the strong case, with the read and BDDL already proven.
2. HOT3D `clip-002066`: the measurable case, giving twin placement error against ground truth.
3. `pov-setting-the-table`: the honest "partial" case.
4. Optional: `microwave-video-model` (third-person, plate into microwave), to show viewpoint robustness.

**Route:** custom BDDL. Fall back to a LIBERO-90 scene with overridden poses only if a BDDL fails to parse or build twice.

**Per footage, in parallel except the sim step:**

1. **Read** with `fx_argon_read.py` (~30 s, ~$0.02–0.06).
2. **Scene:** one Argon call turns `read.json` into a BDDL. Give it my two BDDLs as examples, the asset list and the rules above. Validate the parse offline first with no sim (`libero.libero.envs.bddl_utils.robosuite_parse_problem`), then allow one repair retry with the error text.
3. **Build and render** with `fx_bddl_load.py`, **sequentially, one sim at a time** (20–45 s each, ≈ 3 min for 4). Then `ffmpeg hstack` a footage frame with `agentview_reset.png`.
   - **Dump the gate states in the same process.** For each scene, save one reset state (negative), one goal-teleport state (positive) and 2–3 near-miss states, each with LIBERO's check value:
     - object 5 cm off the target;
     - object hovering 5 cm above it;
     - door left open.
   - These states let step 4 run with no sim and no model call. This dump still needs about 15 min of code: reuse the chain's state-dict format (`runs/argon/libero_chain/chain_server.py`, read-only import or copy) so the chain's reward functions and gate apply unchanged.
4. **Reward and gate:** Argon writes the reward from the video. The gate scores it on the dumped states against LIBERO's check, so it agrees on positives, rejects near-misses and makes zero model calls. HOT3D also gets the placement error: pairwise distances in the twin's sampled layout versus ground truth, in cm.

All sim builds can finish before 16:35, when the slots go to the baseline arm.

**Measurable results per footage:**
- scene built (yes/no, seconds)
- predicate False at reset and True at goal
- reward-gate agreement with LIBERO (positive, negative, near-miss)
- HOT3D placement error
- twin verdict (yes / partial / no, with reasons)
- $ and latency

**Risks:**

| Risk | Detail and mitigation |
|---|---|
| Memory | The chain runs 5 sims × ~2 GB until 16:35. Keep builds sequential, and stop if available memory drops under 2.6 GB (the chain's own gate) |
| Gate-state dump not written yet | About 15 min of code. If it slips, ship steps 1–3 plus the read verdicts, and run the reward gate only on the microwave scene |
| Region collisions in Argon-written BDDLs | The microwave footprint is large. The offline parse catches syntax errors but not collisions; a placement failure shows up only at build time, so allow 1 retry |
| HOT3D input | Fisheye and only 5 s, with 2 of 6 objects lacking assets; the twin will be partial. Ground truth is in world frame, so compare relative distances, not absolute positions |
| Licences | Standard YouTube licences only, so don't redistribute the clips outside the demo |
| Practice on custom scenes | Not possible without a server op that hashes the custom BDDL (`playgrounds/libero/` is off-limits), so no practice numbers on web footage this round |
