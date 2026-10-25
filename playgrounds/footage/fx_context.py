"""Shared context for the footage -> LIBERO front end: asset list, BDDL rules, examples, sim-state API doc."""
import json, re, time
from pathlib import Path

REPO = Path("/home/ran/arg_twin")
HERE = REPO / "playgrounds" / "footage"
M = "gemini-4-argon"

OBJECTS = ("akita_black_bowl white_bowl plate porcelain_mug red_coffee_mug white_yellow_mug glazed_rim_porcelain_ramekin "
           "moka_pot chefmate_8_frypan basket wooden_tray white_storage_box desk_caddy ketchup milk orange_juice butter "
           "cream_cheese alphabet_soup tomato_sauce bbq_sauce salad_dressing new_salad_dressing mayo macaroni_and_cheese "
           "chocolate_pudding cookies popcorn corn cherries wine_bottle black_book yellow_book").split()
FIXTURES = {"kitchen_table": "the table (always present as fixture 'kitchen_table - kitchen_table')",
            "microwave": "door (Open/Close); regions heating_region, top_side",
            "wooden_cabinet": "3 drawers: regions top_region, middle_region, bottom_region (Open/Close each), top_side",
            "white_cabinet": "drawers top_region, middle_region, bottom_region", "flat_stove": "TurnOn/TurnOff; cook_region",
            "short_fridge": "door", "wooden_shelf": "shelf", "wine_rack": "rack", "bowl_drainer": "rack"}

RULES = f"""LIBERO BDDL rules (follow exactly):
- Problem name: LIBERO_Kitchen_Tabletop_Manipulation, domain robosuite, one (:language ...) line.
- Fixtures: always `kitchen_table - kitchen_table`; others as `microwave_1 - microwave` etc.
- Objects: `<category>_1 - <category>` with category ONLY from this list: {', '.join(OBJECTS)}.
  Fixture categories: {json.dumps(FIXTURES)}.
  Objects/fixtures with no asset are left out (say so in a comment line starting with ';').
- Regions on the table: (name (:target kitchen_table) (:ranges ((x_min y_min x_max y_max))) (:yaw_rotation ((a a)))),
  metres, table frame: origin at table centre, robot base at x=-0.66 reaching toward +x, the scene camera at x=+0.66
  looking back at the robot; +y is the camera's right. Usable area x in [-0.25, 0.20], y in [-0.35, 0.35].
  Region boxes 0.02-0.05 m wide. Keep object regions >= 0.12 m apart; a microwave/cabinet needs |y| >= 0.25 and
  >= 0.15 m clearance; fixtures face the robot with yaw 0 (microwave) or pi (wooden_cabinet).
- Regions on a fixture are declared as (heating_region (:target microwave_1)) and referenced as microwave_1_heating_region.
- :init uses (On obj_1 kitchen_table_<region_name>) for every object and fixture; articulated start state as
  (Open microwave_1) or (Open wooden_cabinet_1_top_region).
- :obj_of_interest lists the manipulated object then its target.
- :goal is (And ...) of On(a, b) | In(a, fixture_region) | Open(x) | Close(x) | TurnOn(x) | TurnOff(x).
Map the footage camera to the scene camera: what is left/right/near/far in the footage becomes -y/+y/+x/-x."""

EXAMPLES = "\n\n".join(p.read_text() for p in sorted((HERE / "bddl").glob("fx_kitchen_*.bddl"))[:2])

def api_doc() -> str:
    s = (REPO / "runs/argon/probes/reward/scripts/argon_reward.py").read_text()
    doc = s[s.index('API_DOC = """') + 13: s.index('"""', s.index('API_DOC = """') + 13)]
    return doc.replace("\\\n", "")

def extract(text: str, tag: str) -> str | None:
    m = re.search(r"```" + tag + r"\s*\n(.*?)```", text, re.S)
    return m.group(1).strip() if m else None

def cost(u):
    i = getattr(u, "total_input_tokens", 0) or 0
    o = (getattr(u, "total_output_tokens", 0) or 0) + (getattr(u, "total_thought_tokens", 0) or 0)
    c = getattr(u, "total_cached_tokens", 0) or 0
    return {"input": i, "cached": c, "output_incl_thought": o, "usd": round(((i - c) * 2 + c * 0.1 + o * 10) / 1e6, 4)}

def upload(client, video: Path):
    f = client.files.upload(file=str(video), config={"mime_type": "video/mp4"})
    while f.state is not None and f.state.name == "PROCESSING":
        time.sleep(2); f = client.files.get(name=str(f.name))
    return f
