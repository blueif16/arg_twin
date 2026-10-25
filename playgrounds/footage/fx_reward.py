"""fx-reward: Argon writes success(s, history) for a footage task from the video, its read and the scene's names.
  GEMINI_API_KEY=... .venv/bin/python -I playgrounds/footage/fx_reward.py <slug>
Writes runs/footage/scenes/<slug>/{reward.py, reward.json}. No sim truth is shown to Argon."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/ran/arg_twin/playgrounds/footage")
from fx_context import REPO, M, api_doc, extract, cost, upload
from google import genai

slug = sys.argv[1]
out = REPO / "runs/footage/scenes" / slug
read = json.loads((REPO / "runs/footage/argon_read" / slug / "read.json").read_text())
bddl = (out / "scene.bddl").read_text()
client = genai.Client(http_options=genai.types.HttpOptions(timeout=900_000))
t = time.time(); f = upload(client, REPO / "data/web-footage" / slug / "video.mp4"); up = round(time.time() - t, 1)
P = f"""Above is a real web video of a household task. Earlier you read it as:
{json.dumps(read.get('final'), indent=1)}

A LIBERO simulation scene of the task was written from that read (BDDL below; names in it are the names the state uses):
{bddl}

Write the success check for the task as the video shows it: a Python function `success(s, history) -> bool` that is
True exactly when the task is done in the simulated scene (and the scene is at rest), robustly to small placement
noise, and False for near-misses (object next to the target, hovering above it, a door left open when the video
closes it). Use the scene's names. Work only from the video and the API below; you get no simulator access.

{api_doc()}

Reply with one ```python block holding `success` (and helpers), then two lines on what you check and why."""
t = time.time()
ix = client.interactions.create(model=M, input=[
    {"type": "video", "uri": f.uri, "mime_type": f.mime_type, "processing": {"type": "static", "fps": 1.0}},
    {"type": "text", "text": P}], generation_config={"thinking_level": "medium"})
txt = ix.output_text or ""
code = extract(txt, "python") or ""
(out / "reward.py").write_text(code)
rec = {"slug": slug, "upload_s": up, "latency_s": round(time.time() - t, 1), "usage": cost(ix.usage), "text": txt}
(out / "reward.json").write_text(json.dumps(rec, indent=1))
print(json.dumps({k: rec[k] for k in ("slug", "upload_s", "latency_s", "usage")}))
