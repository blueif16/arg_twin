# read-footage: video only

For footage with no robot logs: human demos, phone video, a dataset's exocentric views (track `footage`; Assembly101 probe).

**Best practice (evidence: `runs/argon/probes/longvideo/findings.md`)**
- **Overview:** the whole recording as one 1 fps video through the Files API, default resolution, time burned in, medium thinking, `max_output_tokens` 65,536. Single pass S1: F1 0.80, boundary 1.6 s, $0.014/min.
- **Zoom:** frame grids (`zoom_grid`, about 1.1k tokens per image) were the best value: S2h F1 0.84, boundary 1.1 s, $0.026/min, about 3.3 F1 points per extra cent. Video zooms at 4–5 fps in a user step were most accurate (S2v F1 0.85, 0.92 s) at $0.068/min. Never use grids as the overview.
- **Budget:** at most 2 zoom levels and 30 % of the recording; refuse identical windows; a reason per zoom; record what each zoom changed. Zooming found zero extra failures (12 of 29 either way).
- **Outcome rule:** judge from a frame 0.5–1 s after release and the next sighting of the object (proposal 6).
- **Conventions and glossary first** (step granularity, failure definitions, one crop per part); keep a per-recording identity glossary to stop part-name drift (proposals 4–5). Transfer to Assembly101 was weak (F1 0.45–0.49) for convention reasons.
- **No labels to score against?** Report the labels as provisional and have a human confirm 3 sampled outcome frames (the project's human confirmation step).
