# reward-from-video: video-only input

For footage with no robot logs and no simulator truth (track `footage`). Not run yet.

**What changes from the core**
- **No truth to score precision against.** "Done" comes from the demo's own end frames. Before freezing, a human confirms or rejects 2–3 frames near the threshold the writer chose; this is the project's human confirmation step. Report the gate as provisional.
- **The state API** is whatever the twin built from the video exposes (video-to-twin; docs/knowledge/video-to-twin.md). Thresholds in metres or degrees inherit the reconstruction's scale error, so state each threshold as a margin from the end state.
- **Near-misses** can still be made by perturbing the reconstructed end state. Their labels come from the human, not from a predicate.
- **Real failures:** the video's own failed attempts, if any (Assembly101-style mistake labels).
- **Expect weak transfer of conventions.** On Assembly101, Argon's step labels scored F1 0.45–0.49 because of dataset conventions, not perception (`runs/argon/probes/longvideo/findings.md`). Write the task's step definitions into the prompt.
