"""fx-argon-read: Argon reads one web clip, video only (1 fps overview, then grid zooms), and proposes a LIBERO twin.

  GEMINI_API_KEY=... .venv/bin/python -I playgrounds/footage/fx_argon_read.py data/web-footage/<slug> runs/footage/argon_read/<slug>
Two calls: (1) video at 1 fps -> draft steps + up to 2 zoom windows; (2) 3x3 frame grids of those windows,
previous_interaction_id -> final read: steps, objects -> LIBERO assets, layout sketch, success predicates, twin verdict.
"""
import base64, json, os, subprocess, sys, time
from pathlib import Path
from google import genai

clip_dir, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
video = clip_dir / "video.mp4"
src = json.loads((clip_dir / "source.json").read_text())
client = genai.Client(http_options=genai.types.HttpOptions(timeout=900_000))
M = "gemini-4-argon"
LIBERO_OBJECTS = ("akita_black_bowl white_bowl plate porcelain_mug red_coffee_mug white_yellow_mug basket wooden_tray "
                  "moka_pot chefmate_8_frypan glazed_rim_porcelain_ramekin ketchup milk butter cream_cheese "
                  "alphabet_soup tomato_sauce orange_juice bbq_sauce salad_dressing wine_bottle black_book yellow_book "
                  "desk_caddy cookies chocolate_pudding popcorn")
LIBERO_FIXTURES = ("kitchen_table living_room_table study_table wooden_cabinet (3 drawers) white_cabinet short_cabinet "
                   "slide_cabinet microwave (door) flat_stove (knob) short_fridge wine_rack wooden_shelf "
                   "wooden_two_layer_shelf dish_rack bowl_drainer faucet basin_faucet")
PRED = "On(a,b), In(a, region_of_container), Open(fixture_region), Close(fixture_region), TurnOn(stove), TurnOff(stove)"

def cost(u):
    i, o = getattr(u, "total_input_tokens", 0) or 0, (getattr(u, "total_output_tokens", 0) or 0) + (getattr(u, "total_thought_tokens", 0) or 0)
    c = getattr(u, "total_cached_tokens", 0) or 0
    return {"input": i, "cached": c, "output_incl_thought": o, "usd": round(((i - c) * 2 + c * 0.1 + o * 10) / 1e6, 4)}

log = {"source": src, "calls": []}
t = time.time()
f = client.files.upload(file=str(video), config={"mime_type": "video/mp4"})
while f.state is not None and f.state.name == "PROCESSING":
    time.sleep(2); f = client.files.get(name=str(f.name))
log["upload_s"] = round(time.time() - t, 1)

P1 = f"""You are reading a short web video of a person doing a household task, to build a robot simulation twin of it.
Only the video is available (audio may contain narration). Duration {src.get('duration')} s, shown at 1 frame per second.
Reply with JSON only: {{"draft_steps":[{{"t0":s,"t1":s,"step":"..."}}], "objects_seen":["..."],
"zoom_requests":[{{"t0":s,"t1":s,"why":"..."}}]}}. Ask for at most 2 zoom windows (each up to 12 s) where a closer look
would settle object identity, the start/end layout, or whether a step succeeded."""
t = time.time()
ix1 = client.interactions.create(model=M, input=[
    {"type": "video", "uri": f.uri, "mime_type": f.mime_type, "processing": {"type": "static", "fps": 1.0}},
    {"type": "text", "text": P1}], generation_config={"thinking_level": "medium"})
lat1 = round(time.time() - t, 1)
txt1 = ix1.output_text or ""
log["calls"].append({"call": 1, "latency_s": lat1, "usage": cost(ix1.usage), "text": txt1})
try:
    d1 = json.loads(txt1[txt1.find("{"): txt1.rfind("}") + 1])
except Exception:
    d1 = {"zoom_requests": []}
zr = (d1.get("zoom_requests") or [])[:2] or [{"t0": 0, "t1": min(12, float(src.get("duration") or 12)), "why": "default"}]

blocks, desc = [], []
for k, z in enumerate(zr):
    t0, t1 = float(z["t0"]), float(z["t1"])
    t1 = max(t1, t0 + 1.0)
    fps = 9.0 / (t1 - t0)
    img = out / f"zoom_{k}.jpg"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0}", "-to", f"{t1}", "-i", str(video),
                    "-vf", f"fps={fps:.4f},scale=480:-2,tile=3x3", "-frames:v", "1", str(img)], check=True)
    times = [round(t0 + (i + 0.5) / fps, 2) for i in range(9)]
    desc.append(f"Grid {k}: window {t0:.1f}-{t1:.1f} s, 3x3 tiles in reading order at about t = {times} s.")
    blocks.append({"type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(img.read_bytes()).decode()})

P2 = f"""Here are the zooms you asked for. {' '.join(desc)}
Now give the final read as JSON only, with these keys:
"task": one-line instruction in LIBERO style (e.g. "open the microwave and put the white bowl in it"),
"steps": [{{"t0","t1","step","succeeded":true/false/null}}],
"objects": [{{"seen":"...","libero_asset":"one of the lists below or null","role":"manipulated|target|distractor|fixture"}}],
"layout_sketch": top-down sketch of the start layout on a tabletop frame with x toward the person/camera and y to the
camera's right, metres, origin at table centre: [{{"name","x","y","note"}}],
"success_conditions": goal predicates using only {PRED},
"twin_supported": "yes|partial|no", "unsupported_reasons": ["..."], "confidence": 0-1.
LIBERO objects: {LIBERO_OBJECTS}. LIBERO fixtures: {LIBERO_FIXTURES}.
If an object is deformable (paper, cloth, food paste) or has no LIBERO asset, say so; do not force a match."""
t = time.time()
ix2 = client.interactions.create(model=M, previous_interaction_id=ix1.id, input=blocks + [{"type": "text", "text": P2}],
                                 generation_config={"thinking_level": "medium"})
lat2 = round(time.time() - t, 1)
txt2 = ix2.output_text or ""
log["calls"].append({"call": 2, "latency_s": lat2, "usage": cost(ix2.usage), "zooms": zr, "text": txt2})
try:
    log["final"] = json.loads(txt2[txt2.find("{"): txt2.rfind("}") + 1])
except Exception as e:
    log["final_parse_error"] = str(e)
log["total_usd"] = round(sum(c["usage"]["usd"] for c in log["calls"]), 4)
log["total_latency_s"] = round(lat1 + lat2 + log["upload_s"], 1)
(out / "read.json").write_text(json.dumps(log, indent=1))
print(json.dumps({"total_usd": log["total_usd"], "latency": [log["upload_s"], lat1, lat2], "final": log.get("final")}, indent=1))
