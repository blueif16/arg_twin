"""fx-bddl-write: Argon turns its own video read (read.json) into a LIBERO BDDL; offline parse; one repair retry.
  GEMINI_API_KEY=... .venv/bin/python -I playgrounds/footage/fx_bddl_write.py <slug>
Writes runs/footage/scenes/<slug>/{scene.bddl, write.json}."""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, "/home/ran/arg_twin/playgrounds/footage")
from fx_context import REPO, HERE, M, RULES, EXAMPLES, extract, cost
from google import genai

slug = sys.argv[1]
read = json.loads((REPO / "runs/footage/argon_read" / slug / "read.json").read_text())
out = REPO / "runs/footage/scenes" / slug; out.mkdir(parents=True, exist_ok=True)
client = genai.Client(http_options=genai.types.HttpOptions(timeout=900_000))
final = read.get("final") or read["calls"][-1]["text"]
P = f"""You read a web video of a household task (your read is below). Write a LIBERO BDDL task file that recreates the
task's start layout and goal on the LIBERO kitchen table, so a simulated robot can practise it.

Your read of the footage:
{json.dumps(final, indent=1)}

{RULES}

Two valid examples (other tasks, for syntax only):
{EXAMPLES}

Reply with the BDDL in one ```bddl block, then one line per object of the footage you left out and why."""
def parse(path):
    r = subprocess.run([str(REPO / ".venv-libero/bin/python"), "-I", str(HERE / "fx_parse.py"), str(path)],
                       capture_output=True, text=True, cwd="/tmp", timeout=120)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "error": (r.stderr or r.stdout)[-800:]}
log = {"slug": slug, "calls": []}
t = time.time()
ix = client.interactions.create(model=M, input=P, generation_config={"thinking_level": "medium"})
txt = ix.output_text or ""
log["calls"].append({"call": "write", "latency_s": round(time.time() - t, 1), "usage": cost(ix.usage), "text": txt})
bddl = extract(txt, "bddl") or extract(txt, "lisp") or extract(txt, "")
(out / "scene_v0.bddl").write_text(bddl or "")
pr = parse(out / "scene_v0.bddl"); log["parse_v0"] = pr
final_path = out / "scene_v0.bddl"
if not pr.get("ok"):
    t = time.time()
    ix2 = client.interactions.create(model=M, previous_interaction_id=ix.id, generation_config={"thinking_level": "medium"},
        input=f"The LIBERO parser rejected it: {pr.get('error')}\nFix it and reply with the full corrected BDDL in one ```bddl block.")
    txt2 = ix2.output_text or ""
    log["calls"].append({"call": "repair", "latency_s": round(time.time() - t, 1), "usage": cost(ix2.usage), "text": txt2})
    (out / "scene_v1.bddl").write_text(extract(txt2, "bddl") or extract(txt2, "") or "")
    pr = parse(out / "scene_v1.bddl"); log["parse_v1"] = pr; final_path = out / "scene_v1.bddl"
log["parse_ok"] = bool(pr.get("ok")); log["bddl"] = str(final_path.relative_to(REPO))
(out / "scene.bddl").write_text(final_path.read_text())
log["usd"] = round(sum(c["usage"]["usd"] for c in log["calls"]), 4)
(out / "write.json").write_text(json.dumps(log, indent=1))
print(json.dumps({"slug": slug, "parse_ok": log["parse_ok"], "repairs": len(log["calls"]) - 1, "usd": log["usd"],
                  "goal": pr.get("goal"), "objects": pr.get("objects"), "error": pr.get("error")}))
