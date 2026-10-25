# reward-from-video: LIBERO reference

Evidence: `runs/argon/probes/reward/findings.md`, `results/summary.json`; the practice use is in `runs/argon/libero_chain/`.

## State API (`runs/argon/probes/reward/scripts/lstate.py`)

- **Per object:** pos, quat, up_axis, linear and angular velocity, world bounding box.
- **Per fixture:** pose, bounding box, and joints as `{q, type, range}`. A drawer opens toward negative q; the top drawer counts as open at q ≤ −0.14.
- **Contacts:** object-level contact pairs with the table, arm, hand and each finger.
- **Gripper:** eef pose, finger qpos, width, command, contacts per finger.
- **Scene:** `table_top_z`.
- **Camera:** the agentview's image-right is world +y (`cam_xmat` column 0 = [0, 1, 0]). MuJoCo reads pixels bottom-up, so `[::-1]` alone is the true view.

## Data and splits

- **Fetched data:** 20–21 demos per task, fetched by HTTP range reads into `data/libero90-demos/` (states, actions and rewards only; untrusted, read with `-I`).
- **Demo roles:** demo 0 (agentview, 512 px, 4 fps) is the video, demos 1–2 are the verification turn, and the other 17–18 are held out.
- **Real failures:** 10 demos of a sibling task per scene (bottom drawer, red mug on the right plate, milk in the basket).
- **Cross-scene check:** 10 demos of the drawer instruction in KITCHEN_SCENE2.
- **Starts:** LIBERO's 50 fixed init states.

## Gate scripts

All in `runs/argon/probes/reward/scripts/`:

| Script | Job |
|---|---|
| `fetch_demos.py` | Fetch the demo data |
| `prep.py`, `prep_sibling.py` | Build the near-miss and real-failure sets |
| `argon_reward.py` | Write the reward (Argon) |
| `feedback.py` | The verification turn (Argon) |
| `evaluate.py`, `eval_sibling.py`, `summarise.py` | The gate (no model calls) |

Specs are in `specs/`, scores in `results/<task>/eval__*.json`, and records with media hashed in `calls/`.

## Numbers (2026-10-25)

| Task | Version | P | R | Near-miss reject | AUROC |
|---|---|---|---|---|---|
| drawer (KITCHEN_SCENE1, top drawer) | medium first / revised | 0.686 / 1.000 | 1.000 / 1.000 | 0.667 / 1.000 | 0.83 / 1.00 |
| drawer | high first / revised | 0.589 / 1.000 | 1.000 / 1.000 | 0.500 / 1.000 | 0.75 / 1.00 |
| mug (LIVING_ROOM_SCENE5, left plate) | both levels, first = revised | 0.928 | 1.000 | 0.865 (17/126 accepted, all 4.5 cm off centre) | 0.93 |
| butter (LIVING_ROOM_SCENE2, basket) | medium first / revised | 1.000 / 0.985 | 0.934 / 1.000 | 1.000 / 1.000 | 1.00 |
| butter | high first / revised | 1.000 / 0.985 | 0.788 / 0.985 | 1.000 / 1.000 | 1.00 |

- **Endpoint checks:** all 12 specs are true at 100% of held-out demo ends and false at all 50 init states, with 0 false fires in 4,123 sibling steps.
- **Cross-scene:** the revised drawer specs, unchanged in KITCHEN_SCENE2, scored 1.000 / 1.000 / 1.000.
- **Where LIBERO's predicate is looser than intent:**
  - a mug tilted 75° on the plate is true in 12/20;
  - butter hovering 4 cm inside the basket volume is true in 21/21;
  - a mug still grasped but touching the plate within 3 cm is true.
- **Cost:** a medium write is $0.035–0.040 and 32–37 s, and a revision $0.034–0.060. High costs 2.5× as much and takes 3× as long.

## Frozen for practice

`runs/argon/libero_chain/frozen_reward.json` holds the medium v1 code of all three tasks, sha256-pinned (14:58). `chain_server.py` refuses to run on a mismatch. The mug v1 fails the L gate (P 0.928, near-miss 0.865) and was frozen anyway for the demo chain. Report every mug result with that caveat.
