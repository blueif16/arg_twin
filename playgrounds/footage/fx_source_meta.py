"""Write source.json for each data/web-footage/<slug>/ from yt-dlp's info json. Run with python -I."""
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    info_p = d / "video.info.json"
    if not info_p.exists():
        continue
    i = json.loads(info_p.read_text())
    src = {k: i.get(k) for k in ("webpage_url", "title", "uploader", "channel", "license", "duration", "upload_date", "width", "height")}
    src["license"] = src["license"] or "not stated (standard YouTube licence)"
    src["clip"] = "video.mp4"
    src["trimmed_to_s"] = 60 if (i.get("duration") or 0) > 60 else None
    (d / "source.json").write_text(json.dumps(src, indent=1))
    print(d.name, json.dumps(src))
