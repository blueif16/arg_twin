# Bot18: presenter's guide

Present from the Bot18 page: http://localhost:8765/runs/status/index.html.
Every number below comes from `runs/orchestrator/story_numbers.md` (each line there names its findings file), or from the late-result files named in place.
Status as of 16:49 PDT, 2026-10-25. Late results checked at 16:44 and 16:48: round 2, LIBERO compile, the compile vs one-shot paired test and the fairness controls have all landed. A larger fresh-init compile run is going until 16:57 (not included).

---

## 1. The story in one line

**Argon understands and judges physical tasks, and turns practice into cheap routines.**

**Argon reads footage, builds a sim of the task, writes its own reward from the video, practises with feedback and compiles what works into cheap routines. A watcher and an auditor check every run. Everything is sim-only.**

**20-second opening (say this on the cover):**
> "We give Argon real footage of a task. It builds a sim of that task. It writes its own success check from the video, and a gate measures that check before we trust it. Then it practises with retries and feedback, and compiles what works into a routine that needs no brain calls. A watcher scores every run and an auditor reads every trace, so the system catches its own mistakes. All of this is in sim only, until a real arm confirms it."

---

## 2. The unified approach (one breath)

- **10 shared job skills**, one per job: read-footage, build-twin, replay-validate, reward-from-video, author-tools, practice, compile-skill, sim-integrity, auditor, brains. Down from 13 that were mostly tied to REASSEMBLE.
- **Each skill has a "Choose your path"** by input type: robot logs + cameras, video only, simulator demos, prior runs. Only the adapter and the `references/<source>.md` file change per source. All 44 dated lessons are kept.
- **Two self-optimization loops:** the **Twin loop** (steps A, B0, B1, B2: the twin explains observations it was not fitted on) and the **Task loop** (steps C, D, E: success in a frozen budget, with cost and brain share falling).
- **Watcher** scores every run and never acts. **Auditor** reads the traces, starting mid-run, and decides advance / iterate / roll back.
- **Observers never pass sim truth** to the actor.
- **Frozen evaluator:** LIBERO's predicate code and task files are hashed and frozen (314 files).

Page cue: the line in the cover frame reads "Same 10 skills for every source; only the adapter and reference change".

---

## 3. Flow and setup per data source

| source | input | steps run | judged by |
|---|---|---|---|
| **REASSEMBLE** (default track) | Real Franka robot logs, plus 3 cameras and force | A read, B0 cameras, B1 scene, B2 replay (not certified), C tools, D practise, E compile, G integrity | Dataset labels; real seat rate (78%); watcher checks; auditor |
| **LIBERO** | Simulator demo videos with exact sim state | A read demos, B reward from video + gate, D trials + feedback, E compile, G integrity | LIBERO's own goal check (frozen) |
| **Web video** | YouTube and HOT3D egocentric clips, no robot logs | A read clip, B scene + reward (gate). C, D, E not run (needs one server change) | LIBERO's goal check on the scene Argon wrote; HOT3D ground truth for placement |
| **Assembly101** | Human assembly video | Reading only (part of the vs-Google comparison) | Dataset action labels |

---

## 4. Core demo path (6 stops by strength, an optional vs-Google stop, then "What's not proven"; about 3 minutes)

**Frame for the whole path:** "Argon understands and judges physical tasks, and turns practice into cheap routines. Higher success than one-shot is not proven yet; reward quality is the bottleneck, and the gate now measures it."

### Stop 1. Practice → routine (compile): the strongest improvement
- **Click:** cover → track **LIBERO** → step E, "Practice turns into a routine". Then track **REASSEMBLE** → step E, "Practice compiles into routines".
- **Claim:** Practice compiles into a routine: one look, then 0 brain calls to act.
- **Headline, paired vs one-shot (final 16:48, `runs/libero/compile-e/paired.jsonl`):** on the same 3 drawer starts (inits 6, 18, 38), by LIBERO's check: **compiled routine 3/3, one-shot Argon 1/3.** Cost per success **$0.0057 vs $1.09** (about 1/190). Calls per episode 1 vs 22; wall time 34–36 s vs 301–805 s. Both discordant starts favour the routine; exact McNemar p = 0.5 (n = 3 can't reach significance).
- **Numbers:** LIBERO drawer (`runs/libero/compile-e/report.md`): one compile call ($0.043) from 3 successful practice trials; **3/3 new starts** opened by LIBERO's check, still true 1 s later. Per success, practice vs routine: **758 s vs 40 s, $0.61 vs $0.006**, 18.3 brain calls vs 1 look + 0. REASSEMBLE: 8/24 unseen starts vs Claude's routine 5/24, 42 s vs 186 s, 0 brain calls.
- **Caveat:** "Direction, not proof." n = 3 in one scene, and the handle moves at most 2 cm. A larger fresh-init run is going until 16:57.

### Stop 2. Reward from video, plus the gate
- **Click:** track **LIBERO** → step B, "Reward from video, gated before it is trusted" → card **"Reward v1 → v2 → v3"**.
- **Claim:** "The gate, not the model, decides." Argon writes the reward from one demo video; a no-model gate measures it before anyone trusts it.
- **Numbers:** on LIBERO the reward's progress tracks the truth at ρ 0.997. REASSEMBLE peg reward: precision 1.0, recall 0.88, 100% of near-misses rejected. On LIBERO the gate caught v1 too loose (mug precision 0.928, accepted all 17 near-misses at 4 cm) and v2 too strict (recall mug 0.04, butter 0.31, drawer 0.625). v3 balances: true at demo end mug 1.0, butter 0.94, drawer 1.0, 0 near-misses accepted; round-1 re-score 8/9.
- **Caveat:** v3 is offline gate only, never used in practice; one mug placed 4.0 cm off still passes.

### Stop 3. Feedback diagnosis
- **Click:** LIBERO → step D, "Practice with feedback, and a reward hole caught". REASSEMBLE → step D, "Argon practises, and retries with feedback".
- **Claim:** Between trials Argon diagnoses what went wrong, without seeing sim truth.
- **Numbers:** reflections named the real cause 5/5 on LIBERO and 5/6 on REASSEMBLE, with 0 truth leaks. LIBERO round 1 (good reward): 3/9 at the first try → 5/9 by the second, by LIBERO's check. The audit caught the mug hole: placements 4–6 cm off passed (3 false successes).
- **Caveat:** these are practice starts. Nudges had no measurable effect on REASSEMBLE.

### Stop 4. Integrity
- **Click:** REASSEMBLE → step G, "Nothing was faked, and cheats get caught". Point at the watcher's ✕ marks: shown on purpose.
- **Claim:** The system catches cheats and its own mistakes.
- **Numbers:** the Argon auditor caught 4/4 planted cheats, $0.05 per audit (13/16 outcomes judged right). The red team was caught 15/15, 0 sandbox escapes. Claude pilot: auditor said ITERATE (raw 21/24 vs tools + routine 16/24).
- **Disclosed deviations:** five REASSEMBLE episodes marked for unlogged pings; LIBERO round 2's leak check found no sim truth (the one hit was a z value the brain itself commanded).

### Stop 5. Reading footage: the blind REASSEMBLE gate
- **Click:** default track **REASSEMBLE** → step A, "Argon reads the footage".
- **Claim:** Argon reads long robot footage, labels every step and tells failed attempts from good ones.
- **Numbers:** on 9 blind gate episodes, segment F1 0.992 and balanced accuracy 0.947 (Claude and Argon both pass 3/3). Argon $0.22 vs Claude's $0.32 per recording minute. The skill self-learned from balanced accuracy 0.78 to 0.95.
- **Caveat:** 9 episodes; the self-learning gain is across runs of the same skill.

### Stop 6. Footage → sim scene (web video)
- **Click:** track **Web video** → step B, "Three real clips, three working sim tasks" → card **"Web video, honestly"**.
- **Claim:** Three real clips (YouTube microwave 11 s, HOT3D 5 s, YouTube table setting 51 s) became three working sim tasks.
- **Numbers:** 3/3 scene files parsed first try and built by LIBERO (20–63 s); goal check False at reset, True at goal, 3/3. Argon's check agrees with LIBERO on 11/12 saved states (4/4, 4/4, 3/4). About $0.05 per clip.
- **Caveat, say it:** a one-prompt baseline ties ours on these 3 simple clips at about 40% of the cost; ours is better only on HOT3D placement (4.3 vs 6.3 cm). Practice needs one server change.

### Optional. Our skill vs Google's published prompt, honestly
- **Click:** cover → card **"vs Google's prompt"**.
- **Raw result:** wins or ties 12/12 recordings (F1 A101 0.41 → 0.54, REASSEMBLE 0.60 → 0.94, LIBERO 0.72 → 0.85). Not Google's scores: Google published none; we ran their prompt verbatim on our labelled recordings, same model.
- **Controls (final 16:44) show most of the gap is conventions and frame rate, not method:** REASSEMBLE, none survives (0.59 vs 0.60 without our conventions, which were measured on these same train episodes, so the +0.34 is leakage). Assembly101, about 40% survives (0.38 → 0.45 method alone → 0.55 with conventions, same 3 recordings). LIBERO progress curve: frame rate explains it (Google at 4 fps ρ 0.97 vs our 0.98). LIBERO segmentation: the gap survives at equal frame rate, 0.63 vs 0.85.
- **Say:** "The gain is mostly writing the source's conventions down and reading at the right frame rate. The method alone adds a modest gain. A held-out claim needs new recordings."

### Last stop. What's not proven (short)
- **Click:** cover → card **"Not proven yet"**; LIBERO → step D → card **"One-shot vs loop"**.
- **No significant success boost over one-shot on LIBERO yet:** during practice, paired on 5 starts, one-shot 2/5 vs our loop within 2 tries 1/5 (p = 1.0). After practice, the compiled drawer routine beat one-shot 3/3 vs 1/3 on the same starts (p = 0.5): the direction favours practice, but significance is not shown. Our arm practised on the over-strict reward v2, which never fired on a correct mug placement. "Reward quality dominates; the gate exists for exactly this; v3 is the fix and runs next."
- **The twin's replay isn't certified (B2):** our audit found a leak worth about 0.10.

---

## 5. Best comparisons and improvements

| comparison | baseline | ours | fair? |
|---|---|---|---|
| Reading, action-segment F1 (12 recordings; A101 / REASSEMBLE / LIBERO) | Google's prompt, run by us: 0.41 / 0.60 / 0.72 | 0.54 / 0.94 / 0.85; wins or ties 12/12 | **Partly fair; controls show leakage.** Without our conventions (measured on these recordings): REASSEMBLE 0.59 vs 0.60, A101 0.45 vs 0.38. At equal fps, LIBERO segmentation still 0.85 vs 0.63. |
| Progress vs sim truth (LIBERO) | Google's prompt ρ 0.55 | ρ 0.97 | **Partly fair; controls show it is frame rate.** Google's prompt at 4 fps reaches ρ 0.97 vs our 0.98. |
| Reading cost, REASSEMBLE gate | Claude $0.32 / rec. min | Argon $0.22 / rec. min | **Yes.** Same 9 blind gate episodes, both pass 3/3. |
| Read-footage skill self-learning | bacc 0.78 | bacc 0.95 | **Partly.** Across runs of the same skill. |
| Cameras (B0) | skill alone 4.46 px | skill's recipe 1.43 / 1.72 px | **Yes.** Same Argon, same cameras. |
| **Compile, LIBERO drawer: practice vs routine, per success** | practice: 18.3 brain calls, 758 s, $0.61 | routine: 3/3 new starts, 1 look + 0 calls, 40 s, $0.006 | **Yes: same task, same check, new starts; small n** (n = 3, one scene, handle moves at most 2 cm). |
| **Compiled routine vs one-shot Argon, LIBERO drawer (paired)** | one-shot 1/3, $1.09 / success, 22 calls / episode | routine 3/3, $0.0057 / success, 1 call / episode | **Yes: same starts, same check; small n.** n = 3, p = 0.5; one scene. Direction, not significance. |
| Compile, REASSEMBLE unseen starts | Claude's routine 5/24, 186 s | Argon's routine 8/24, 42 s, 0 brain calls | **Yes.** Same unseen starts. Small n. |
| LIBERO practice, round 1 | first try 3/9 | 5/9 by second try | **Partly.** Practice starts, not held out. |
| **LIBERO one-shot vs our loop, round 2 (an honest loss)** | one-shot 2/5 (4/9 on all 9 starts, $1.00 / success) | first trial 0/5; loop within 2 tries 1/5 | **Yes, and it goes against us.** Paired n = 5, LIBERO's check, p = 1.0 (0.5 for first trial). Our arm practised on the over-strict reward v2. |
| LIBERO reward gate, near-miss rejection | v1: mug 0.865 | v3: 1.0 on all 3 tasks; true at demo end mug 1.0, butter 0.94, drawer 1.0 | **Yes.** No-model gate. Offline only. |
| LIBERO reward vs LIBERO, round 1 re-scored | v1: 11/14 after +1 s (3 false positives) | v3: 8/9 (one mug 4.0 cm off still passes) | **Partly.** Post hoc, offline, not used in practice. |
| Web video, end to end | one-prompt baseline: ties, 40% of the cost | ties; better only on HOT3D placement 4.3 vs 6.3 cm | **Yes.** We report the tie. |
| LIBERO anchor (published protocol) | paper 31/60 on the same tasks, $5.90 / ep. | 2/6, $5.70 / ep. | **Partly.** Only 6 episodes; the published views are mirrored. We are below the paper. |
| REASSEMBLE practice, attempt 1 | Claude raw 12/14, 10.2 calls / success | Argon 11/14, 20.9 calls / success | **Yes.** Same starts; no significant difference (p = 1.0). Argon is not better here. |

---

## 6. "Not proven yet" (say it out loud)

- **The vs-Google reading result is mostly conventions and frame rate.** The controls show the REASSEMBLE gap is leakage (0.59 vs 0.60 without our conventions) and the progress gap is frame rate. The method alone adds a modest gain (A101 0.38 → 0.45; LIBERO segmentation 0.63 → 0.85 at equal fps). A held-out claim needs new recordings.
- **No scored held-out boost.** Every attempt-1 vs attempt-N number is on practice starts. On LIBERO, one-shot Argon (2/5) beat our loop practising on the over-strict reward v2 (1/5 within 2 tries) on 5 paired starts. After practice, the compiled drawer routine beat one-shot 3/3 vs 1/3 on the same starts (p = 0.5). The direction favours practice, but significance is not shown.
- **Loop 1, the twin, isn't certified.** The REASSEMBLE replay audit found a leak worth about 0.10.
- **On the LIBERO anchor we're below the paper** (2/6 vs the paper's 31/60 on the same tasks; published 43% on LIBERO-10).
- **Two rungs are designed but not run:** retries without feedback, and one-shot after practice.
- **Everything is sim-only** until a real arm confirms it.

---

## 7. Likely judge questions

- **"Is this Google's benchmark?"** No. Google published a prompt, not scores, and we don't have their videos. We ran their prompt verbatim on Argon, on our own labelled recordings, next to our skill. The claim is "our skill vs their published prompt, same model, same recordings". Our own controls show most of the gap is conventions and frame rate.
- **"Is your reading better than Google's prompt?"** Mostly not by method. Most of the gap is conventions learned on the same recordings plus frame rate: on REASSEMBLE none survives (0.59 vs 0.60 without our conventions). The method alone gives a modest gain: Assembly101 0.38 → 0.45, LIBERO segmentation 0.63 → 0.85 at equal frame rate. A held-out claim needs new recordings.
- **"Is there a held-out boost?"** No, not yet. Every attempt-1 vs attempt-N number is on practice starts. The closest positive is compile: on the same 3 drawer starts the routine went 3/3 vs one-shot's 1/3 (p = 0.5), and REASSEMBLE's routine seated 8/24 unseen starts vs Claude's 5/24. Direction and cost, not a significant held-out boost.
- **"Your loop lost to one-shot?"** Yes, on round 2: 1/5 vs 2/5 on 5 paired starts, p = 1.0. Practice ran on an over-strict reward (v2) that never fired on a correct mug placement. Round 1, with the right reward, went 3/9 → 5/9, but it isn't paired with the one-shot run, so it's not a proven boost. Reward quality dominates; v3 is the fix and runs next on held-out starts.
- **"So does your loop beat one-shot on actions?"** After practice, the compiled routine did: 3/3 vs 1/3 on the same starts, about 190× cheaper per success. That's direction, not significance (p = 0.5). During practice on a bad reward, it didn't.
- **"Why is your LIBERO number below the paper's 43%?"** We reproduced the paper's protocol for 6 episodes: 2/6, where the paper scored 31/60 on the same tasks, at a similar cost ($5.70 vs $5.90 per episode). It's a small sample, and the published adapter's camera views are mirrored left to right. We show the gap rather than hide it.
- **"Does it work on a real robot?"** Not yet. Everything is sim-only. REASSEMBLE is real robot footage, but the twin's replay isn't certified and nothing has run on a real arm.
- **"What does Argon do that a smaller model can't?"** We did not run a smaller model head to head, so we don't claim that. What Argon does here: reads whole long recordings in one pass ($0.22 vs Claude's $0.32 per minute, both pass the same blind gate); writes the reward and the scene file from one video (3/3 scene files parsed first try); revises its own reward from gate evidence; audits traces (4/4 planted cheats). It is not better at everything: Claude raw matched it on REASSEMBLE practice (12/14 vs 11/14, p = 1.0).
- **"Isn't the comparison rigged?"** We list the biases on the card ourselves, and we report the ties: the web-video one-prompt baseline ties us at 40% of the cost.

---

## 8. Extras and backup

- **REASSEMBLE step A, "Argon reads the footage":** 9 blind gate episodes, segment F1 0.992, balanced accuracy 0.947; $0.22 vs $0.32 per minute.
- **REASSEMBLE step B, "A twin that lines up; replay not certified":** cameras 1.43 / 1.72 px; scene passes 4/4, MJCF sound on 10/12 checks, parts 7 mm off; seat rate vs real 78%: 54% → 63% → 69% (train only), not certified.
- **REASSEMBLE step D, "Argon practises, and retries with feedback":** 11/14 seated (79%, 95% CI 49–95%), 0 false claims, $1.48 per success. Retries: 0/4 → 1/4 on hard starts, 5/6 reflections right, 0 truth leaks, nudges had no measurable effect.
- **LIBERO step A, "Argon reads one demo into a task spec".** Step C, "The same tool skill, a published protocol first": 10 `libero_*` tools, all pass the schema check; on REASSEMBLE the tool chain seated 22/24 vs 9/24 for one raw descent.
- **LIBERO step G, "The actor never sees the answer":** reward and LIBERO agreed on 10/14 round-1 trials, 0 misses, 4 false positives. 314 frozen files.
- **External anchor chip (LIBERO):** published 43% on LIBERO-10, $7.84 per episode; 41 of 100 episodes hit the call, spending or token limit.
- **REASSEMBLE peg reward:** precision 1.0, recall 0.88, rejected 100% of near-misses.
- **Twin loop / Task loop cards:** the two loops and their headline checks.
- **Other tracks:** SO-101 (a twin from two cameras and joint logs; later steps wait for REASSEMBLE) and Footage only (not started).
