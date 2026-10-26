# Bot18: recording script, in order

**Page:** http://100.69.173.48:8765/index.html (Tailscale) or http://localhost:8765/runs/status/index.html.
The long version with sources is `docs/presentation.md`.

**How the page works:**
- The **track switcher** sits at the start of the flowchart: REASSEMBLE, SO-101, Footage only, LIBERO, Web video.
- **Pill cards** sit under the subtitle.
- Click a **step** on the flowchart to open its chapter. Chapters scroll sideways, and tiles open into cards.

There are 9 screens, about 3 minutes in all. Each screen says what to **look at**, what to **say**, and where to **go next**.

---

### 1. Cover (REASSEMBLE track, the default)
**Look at:** the title "From footage to a policy that checks itself", the flowchart, and the frame line "Same 10 skills for every source…".

**Say:**
> "We give Argon a video of a task. It builds a simulation of that task, writes its own success check from the video, practises, and turns what works into a routine. A watcher and an auditor check every step. The same 10 skills run on every kind of footage; only a small adapter changes. It's all in simulation for now."

**Next:** hover the **Task loop** region for a second, then switch the track to **Web video**.

---

### 2. Web video → step "Build twin" (chapter "Three real clips, three working sim tasks")
**Look at:** the side-by-side image: the YouTube frame, the sim at the start, the sim at the goal. Start with the microwave.

**Say:**
> "This is a YouTube clip of someone putting a bowl in a microwave. Argon read it, wrote the scene, and the simulator built it. Three real clips became three working sim tasks. Argon's own success check agrees with the simulator's on 11 of 12 test states."

**Next:** open the pill **"Web video, honestly"** for one line, then switch to **LIBERO**.

> "To be fair: on clips this simple, a single prompt does about as well."

---

### 3. LIBERO → step 2 (chapter "Reward from video, gated before it is trusted")
**Look at:** the pill card **"Reward v1 → v2 → v3"**, which you can also open from the cover.

**Say:**
> "On LIBERO we can check everything against the simulator's truth. Argon watches one demo and writes the code that says 'did it work?'. We don't trust it blindly: a gate tests it on real endings and on near-misses. Version 1 was too loose and accepted a mug left off the plate. Version 2 was too strict and rejected real successes. The gate caught both, and version 3 passes. Its progress score tracks true progress at 0.997."

**Next:** go to the **Practise** step.

---

### 4. LIBERO → Practise (chapter "Feedback works only when the reward is right")
**Look at:** the trial 1 vs trial 2 clip. Then open the pill **"One-shot vs loop"**.

**Say:**
> "When a try fails, Argon watches the clip and says why, and the next try starts from that fix. It named the real cause 5 out of 5 times, and a second try took us from 3 to 5 successes out of 9. Honest result: when we practised on the too-strict reward, Argon trying once did better than our loop, 2 of 5 against 1 of 5. A bad reward hurts practice, and that's exactly why the gate exists."

**Next:** go to the **Compile** step.

---

### 5. LIBERO → Compile (chapter "Practice turns into a routine")
**Look at:** the drawer routine clip. Then open the pill **"After practice: 3/3 vs 1/3"**.

**Say:**
> "After practice, Argon compiled what worked into a routine: one look, then it runs with no more thinking. On the same three starts, Argon on its own opened the drawer once; the routine opened it all three times. On 15 new starts it got 11. Every miss was the look putting the handle in the wrong place, never the motion. And it's about 13 minutes per success while practising against about 1 minute with the routine."

**Next:** switch to **REASSEMBLE** and click **Read footage**.

---

### 6. REASSEMBLE → Read footage (chapter "Argon reads the footage")
**Look at:** the signals plot and wrist grid. Then open **"vs Google's prompt"** from this step.

**Say:**
> "This is real robot footage. On episodes it had never seen, Argon found each action almost perfectly, 0.99, and judged success right 95% of the time. We also ran Google's own prompt from their blog on the same videos. Our skill wins, but mostly because it knows each dataset's labelling rules and reads at the right frame rate. The method alone adds a little."

**Next:** click **Practise**.

---

### 7. REASSEMBLE → Practise (chapter "Argon practises, and retries with feedback")
**Look at:** the Claude vs Argon peg clip.

**Say:**
> "Same skills, real robot data: Argon seated 11 of 14 pegs on its first try, as well as Claude did. On the hard ones, its diagnosis was right 5 of 6 times."

**Next:** click **Integrity**.

---

### 8. REASSEMBLE → Integrity (chapter "Nothing was faked, and cheats get caught")
**Look at:** the cheat clip and the check tiles.

**Say:**
> "We planted cheats on purpose, and the auditor caught all 4. The red marks you've seen on the page are real failures that we show on purpose."

**Next:** go back to the cover and open the pill **"Not proven yet"**.

---

### 9. Cover → "Not proven yet"
**Look at:** the card.

**Say:**
> "Three honest gaps. Practice beats a single try only once it becomes a routine, and only on a small sample. Our twin of the real robot isn't certified yet. And everything is in simulation until a real arm confirms it. Thanks."

---

## If they ask
- **"Is this Google's benchmark?"** "No. Google published a prompt, not scores. We ran their prompt ourselves on our videos."
- **"Does your loop beat Argon on its own?"** "After practice, the routine did: 3 of 3 against 1 of 3 on the same starts, and 11 of 15 on new ones. During practice on a bad reward, it didn't."
- **"Why not a real robot?"** "Sim first. Every result is labelled sim-only until a real arm confirms it."
