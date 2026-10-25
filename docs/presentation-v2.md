# Bot18: the short version (read-out script)

Page: http://100.69.173.48:8765/index.html (Tailscale) or http://localhost:8765/runs/status/index.html.
The long version with sources is `docs/presentation.md`.

---

## Open (cover)

> "We give Argon a video of a task. It builds a simulation of that task, writes its own success check from the video, practises, and turns what works into a routine. A watcher and an auditor check every step, so it can't fool itself. It's all in simulation for now."

> "The same 10 skills run on every kind of footage: robot logs, simulator demos, and plain YouTube clips. Only a small adapter changes."

---

## 1. Practice turns into a routine (LIBERO, cover pill "After practice: 3/3 vs 1/3")

> "Argon practised opening a drawer, then compiled what worked into a routine. Now it takes one look and runs the routine, with no more thinking."

- **On the same 3 starts:** Argon trying once got 1 of 3. The routine got **3 of 3**.
- **On 15 new starts:** the routine got **11 of 15**. Every miss was the one look putting the handle in the wrong place; the motion itself never failed.
- **Speed:** about 13 minutes per success while practising, about 1 minute with the routine.

---

## 2. Argon writes its own reward, and a gate checks it (LIBERO, card "Reward v1 → v2 → v3")

> "Argon watches one demo and writes the code that says 'did it work?'. We don't just trust it. A gate tests it on real endings and on near-misses."

- **Version 1** was too loose: it said yes to a mug left a few centimetres off the plate. The gate caught it.
- **Version 2** was too strict: it said no to real successes. The gate caught that too.
- **Version 3** passes both tests.
- Its progress score tracks the true progress almost perfectly (0.997).

---

## 3. Feedback between tries (LIBERO practice step)

> "When a try fails, Argon looks at the video and says why. The next try starts from that fix."

- **Diagnoses:** Argon named the real cause 5 of 5 times on LIBERO and 5 of 6 on the robot pegs.
- **Second tries:** with a second try, successes went from **3 of 9 to 5 of 9**.
- **Honest loss:** when we practised on the too-strict reward (version 2), Argon trying once did better than our loop, 2 of 5 vs 1 of 5. A bad reward hurts practice, and that's why the gate exists.

---

## 4. Real videos become sim tasks (Web video track)

> "Here's a YouTube clip of someone putting a bowl in a microwave. Argon read it, wrote the scene, and the simulator built it."

- **3 real clips** became 3 working sim tasks: microwave, an egocentric clip, table setting.
- **Argon's own check agrees with the simulator's** on 11 of 12 test states.
- **Honest:** on clips this simple, a single prompt does about as well as our full pipeline.

---

## 5. Reading footage (REASSEMBLE, step 1)

> "On robot footage it has never seen, Argon finds each action and whether it worked."

- **Blind test, 9 episodes:** it labels actions almost perfectly (0.99) and judges success right 95% of the time.
- **Vs Google's own prompt from their blog:** we win on every recording, but most of that is because our skill knows each dataset's labelling rules and reads at the right frame rate. The method alone helps a little, on human assembly video (0.38 → 0.45).

---

## 6. It checks itself (REASSEMBLE, integrity step)

> "We planted cheats on purpose. The auditor caught all 4."

- The red ✕ marks on the page are real failures we show on purpose.

---

## Close: what we haven't proven yet

> "Three honest gaps."

1. "Practice beats a single try only after it becomes a routine, and only on a small sample."
2. "Our twin of the real robot isn't certified yet."
3. "Everything is in simulation until a real arm confirms it."

---

## If they ask

- **"Is this Google's benchmark?"** "No. Google published the prompt, not scores. We ran their prompt ourselves on our videos."
- **"Does your loop beat Argon on its own?"** "After practice, the routine did, 3 of 3 vs 1 of 3 on the same starts, and 11 of 15 on new ones. During practice with a bad reward, it didn't."
- **"Why not a real robot?"** "Sim first. Every result is labelled sim-only until a real arm confirms it."
