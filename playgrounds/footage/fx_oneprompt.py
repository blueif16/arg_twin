"""fx-oneprompt (baseline arm): the whole video + one prompt -> BDDL scene + success check. No read, zooms or repair.
  GEMINI_API_KEY=... .venv/bin/python -I playgrounds/footage/fx_oneprompt.py <slug>
Writes runs/footage/oneprompt/<slug>/{scene.bddl, reward.py, call.json, parse.json}."""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, "/home/ran/arg_twin/playgrounds/footage")
from fx_context import REPO, HERE, M, RULES, EXAMPLES, api_doc, extract, cost, upload
from google import genai
slug = sys.argv[1]
out = REPO / "runs/footage/oneprompt" / slug; out.mkdir(parents=True, exist_ok=True)
client = genai.Client(http_options=genai.types.HttpOptions(timeout=900_000))
t = time.time(); f = upload(client, REPO / "data/web-footage" / slug / "video.mp4"); up = round(time.time() - t, 1)
P = f"""Write the LIBERO BDDL scene and a success check for this task.

{RULES}

Two valid BDDL examples (other tasks, for syntax only):
{EXAMPLES}

The success check is a Python function `success(s, history) -> bool` over this state API (object names = the BDDL's):
{api_doc()}

Reply with one ```bddl block and one ```python block."""
t = time.time()
ix = client.interactions.create(model=M, input=[{"type": "video", "uri": f.uri, "mime_type": f.mime_type},
                                                {"type": "text", "text": P}], generation_config={"thinking_level": "medium"})
txt = ix.output_text or ""
(out / "scene.bddl").write_text(extract(txt, "bddl") or extract(txt, "lisp") or "")
(out / "reward.py").write_text(extract(txt, "python") or "")
r = subprocess.run([str(REPO / ".venv-libero/bin/python"), "-I", str(HERE / "fx_parse.py"), str(out / "scene.bddl")],
                   capture_output=True, text=True, cwd="/tmp", timeout=120)
try:
    pr = json.loads(r.stdout.strip().splitlines()[-1])
except Exception:
    pr = {"ok": False, "error": (r.stderr or r.stdout)[-500:]}
(out / "parse.json").write_text(json.dumps(pr, indent=1, default=str))
rec = {"slug": slug, "upload_s": up, "latency_s": round(time.time() - t, 1), "usage": cost(ix.usage), "text": txt}
(out / "call.json").write_text(json.dumps(rec, indent=1))
print(json.dumps({"slug": slug, "parse_ok": pr.get("ok"), "goal": pr.get("goal"), "objects": pr.get("objects"),
                  "error": pr.get("error"), "usd": rec["usage"]["usd"], "latency_s": rec["latency_s"]}, default=str))
