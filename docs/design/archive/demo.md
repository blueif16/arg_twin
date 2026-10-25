# Demo

> **Status: proposal under discussion. Nothing in this file is agreed yet.**

Draft v1, 2026-10-24. The presentation is 3 minutes of results. The loop itself runs before it, not live on stage. Everything is visual, with things moving and visibly getting better. There are no comparison tables; at most a small counter is on screen. The numbers that decide what is accepted stay backstage ([policy-learning.md](policy-learning.md#backstage-checks)).

## Beats

1. **The twin snaps into place.** Real and sim are overlaid on all three cameras, and the difference heatmap fades round by round as the reconstruction units are fixed ([reconstruction.md](reconstruction.md)).
2. **The twin behaves like reality.** A real attempt and its twin replay play side by side, including a real jam that the twin reproduces.
3. **One skill, three drivers.** From the same start:
   - Argon controlling through tools
   - the learned small model
   - the real operator footage from the training data

   This shows how closely the learned behavior matches Argon's and the human's.
4. **Learning.** Argon's first clumsy attempts at a skill, next to the mastered program.
5. **The full board.** The small-model team runs the full board, and the monitor catches a slip and retries.
6. **Optional: orchestration.** Many builders at work in parallel.

## What it is built from

- The real-vs-sim viewer: a synced timeline, the three camera views side by side, F/T traces and contact arrows. Argon generates it.
- Recorded runs from the loop, replayed exactly, so the presentation never depends on a live API call (AGI Factory's recorded mode).
