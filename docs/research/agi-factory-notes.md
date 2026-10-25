# AGI Factory (factory-autopilot): design notes

Source: `vendors/factory-autopilot` (shallow clone of https://github.com/ToukoUrsin/factory-autopilot, commit `fc57b4d`). It was built at the Recursive Agents Hackathon on 17 Oct 2026. This is reference code only: we read it and do not run it.

## What it actually is

AGI Factory is a self-improving maintenance agent team. Its "twin" is **not a physics simulator**:

- `factory/engine.py` is a deterministic **event simulator**. Incidents, sensor drift, work orders and stockouts are modeled as rates over 180 simulated days, and stations sit at 2D floor-plan fractions. It has no geometry, no contact and no robot dynamics.
- The 14.2M-splat scan is a **public SuperSplat scene** (`8be1404d`), streamed at runtime and not bundled. The team didn't capture it. It provides the visual stage only.
- Collision and navigation use a separate **2D walk grid** (0.3 m cells) and surveyed station anchors and bounding boxes (`static/scene-map.js`). A teammate built these by hand, so the scan was never automatically aligned to anything.

This setup is the same split as our rule D1, in its crudest form: the splat renders, and a separate occupancy grid handles collisions.

## Designs worth copying

1. **The model proposes; deterministic code scores.** The LLM may only reorder a whitelist of candidate edits. The gate, the ablation and the exam make no model calls (`engine.py` docstring, `gate()`). This boundary is the same as our D2.
2. **Disjoint seed splits.** Training seeds 11–18 feed the audit, gate seeds 101–112 decide acceptance, and exam seeds 201–212 are scored once and never used for a decision. This matches our §4 held-out rule and makes it executable.
3. **Paired comparison with common random numbers.** The candidate and its parent face identical incidents per seed. Acceptance requires a mean gain of at least 1, more wins than losses, and **no increase in unsafe actions**, which is a hard constraint.
4. **Leave-one-out ablation.** Each kept edit is removed in turn and stays only if removing it makes the result worse or less safe.
5. **A planted trap the gate visibly rejects.** `fast_track` closes work faster but skips checks, and the gate rejects it because unsafe actions go from 0 to 15. This was their strongest demo beat, and it is the same idea as our T4 planted intent violations.
6. **Three run modes: live, recorded and offline.** The demo never depends on a live API call. Live calls are written to `factory/evidence/*.json` (with no secrets) and replayed exactly, and a rule-based fallback runs with no key at all.
7. **Splat as a camera.** `viewer.captureFrame()` renders a virtual CCTV frame from the scan, and that frame is sent to the VLM (`/api/detect`, `/api/verify`). This is a cheap way to get "what the camera sees" from a splat.
8. **Constrained LLM actuation.** The pilot accepts 5 commands at most, from a fixed 5-word vocabulary. The browser rejects moves that would hit an obstacle and falls back to A* when the model is unavailable.
9. **Negative evidence shown on purpose.** The CAM-02 verification returns `arm_on_site: false`, and the UI reports an evidence gap instead of a success.
10. **Docs written for judges.** The README opens with a one-sentence product description, a before/after table and honest limits. `docs/JUDGING.md` maps each judging criterion to evidence, and `docs/EVIDENCE.md` ties every number to a file.

## Not to copy

- The twin's numbers come from modeled mechanisms (for example, a verifier that raises evidence completeness from 70% to 91%). The gains show only that the selection loop works, not anything physical. Our T3 and T4 need real simulator state.
- Station positions were surveyed by hand. Our plan needs the scan registered to the robot's coordinate frame.

## Reusable pieces

- `factory/static/vendor/splat-viewer.js` is PlayCanvas `supersplat-viewer` (MIT). It provides `createViewer` and `captureFrame`.
- `engine.gate()` is about 20 lines and can serve as a template for a paired, safety-constrained acceptance test.
