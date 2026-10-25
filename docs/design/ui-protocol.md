# Bot18 page protocol

The one page that shows the Bot18 pipeline, its results and its evidence: `runs/status/index.html`. Every update follows this protocol, whether it is new information (a number, a clip, a decision) or a change to the page itself.

The page shows one data track at a time. Every track (`tracks` in `playbook/validation.yaml`: REASSEMBLE, SO-101, footage only) runs through the same loop, and a switcher at the flowchart's start picks which one the whole page shows: footage, node states, step pages, cards and ledger.

- The page is generated. `harness/status.py` collects the data and `harness/status_template.html` renders it. Never edit `index.html`, which is overwritten on every rebuild.
- Rebuild: any `record_check`, `record_event` or `publish_media` call, or `.venv/bin/python -m harness.status` by hand (`--bundle` copies the linked media so the folder can be zipped or published).
- Visual reference: Physical Intelligence's blog post "Frontier Models for Every Robot" (pi.website/blog/api-beta, 13 Oct 2026). The colours, fonts and diagram geometry below were read from that page's CSS and from its `api-workflow-desktop.svg`, not guessed.

## 1. Structure

```
 Bot18  [ Claude stand-in | Argon | Both ]
 ─────────────────────────────── progress along the top edge ───────────────────────────────

 page 0  OVERVIEW  headline | subtitle
                   the loop as a pi-style flowchart, as large as the screen allows, for the track picked below
                                     .TWIN LOOP ○.......   .TASK LOOP ○......... (soft regions; hover or focus: a panel
                   REAL FOOTAGE (clips) -> [ 1 -> 2 ] -> [ 3 -> 4 -> 5 ] -> REAL ARM     in the quietest space beside it)
                   |◐ REASSEMBLE  |          |    |    ^       |     (the data-source switcher: the pressed row,
                   |◐ SO-101      |        tags  tags  |       tags       filled ink, is the track every page shows)
                   |○ FOOTAGE ONLY|       result result +-- feedback --+   (skill tags under each node; integrity
                   note · watch · about                                     sits on the frame)
 pages   STEPS     one full-screen page per story section, scrolled sideways, snapping to each page
 1..n              +-----------+---------------+--------+
                   | intro     | hero media    | number |
                   |           |               | number |
                   +-----------+---------------+--------+
                   | checks    | more visuals  | log    |
                   +-----------+---------------+--------+
 then    LEDGER    normal vertical reading, for the shown track: timeline, cost, decision logs, tool calls (every
                   track), hypotheses (the first track's unless they name one), validation history
 last    BUILDERS  a dark panel: the builders' names at top right, one note line, then the wordmark
                   "Bot18 × Argon" across the full width, cut off by the panel's lower edge

   ( ‹ )  Overview  1 READ → 2 TWIN → … → 5 COMPILE | ↺ FEEDBACK  INTEGRITY | Ledger  ( › )      validation v9
                                                                                         [● Live] [Builders]
```

- **Chrome.** No top bar. Top left: the text logo and the brain switch (Claude stand-in first, then Argon, then Both). The top right stays empty. The progress line runs along the top edge. At the bottom: the navigator, a miniature of the loop with previous and next buttons, centred; at the bottom right, the validation version and build time above the Live switch and the Builders link (an icon and the word). The navigator moves left when the right side needs the room. On laptops (960–1499 px) the build time moves into the tooltip and the navigator tightens; below 1200 px the validation line and the Live and Builders labels hide; below 960 px Live and Builders float above the navigator.
- **Named cards.** Under the cover's subtitle sits a row of pill links, one per note in `cards` in `story.yaml` that applies to the shown track ("Not proven yet", "vs Google's single prompt"). A card that names a step is also a button on that step's page. Each opens a normal card. Pending numbers in a card are written as running, with the time the result is due, never guessed.
- **Frame line.** The line inside the frame under its title comes from `frame_note` in `story.yaml` (default "Argon plans, the twin tests, the harness judges every number"); the skills that serve every step sit beside it.
- **Builders page.** The last page, after the ledger, reached with › from the ledger or the Builders link (gold while it is open). Names and the note come from `builders` in `story.yaml`. The wordmark is sized to the panel's width, at most half its height; on phones it breaks into two lines.
- **Data tracks and the switcher.** The kinds of recordings the playbook runs on, from `tracks` in `validation.yaml`, read at every build. Every track runs through the same loop steps (the stages without `track:`, A to G). The first track's checks are those steps' own; another track keeps its checks in a container stage (`track: so101`, stage T2) and each check names the step it sits on (`step: B0`). A container stage is never a node, a step or a navigator item.
  - Under the flowchart's real-footage clips sits the data-source switcher: one row per track, in `validation.yaml` order, each with the state glyph of its `status` (active ◐, done ✓, not started ○) and one line: passing gated checks over all its steps ("13/31 checks pass", per brain, never merged), "nothing measured yet", or, for a track not started, "not started" on a dashed row like a planned skill. The pressed row is filled `--ink`, like the brain switch. Under it: the track's `note`, "watch N clips" and "about <track>", which opens the track's card.
  - Pressing a row switches the whole page to that track at once, from data built in at build time (no fetch, so `--bundle` keeps working): the clips at the start, every node's state, the arrows, the lead brain, the integrity chip and feedback box, every step page's text, checks, numbers, media and log, the navigator's dots, every card, the brain switch's counts and the ledger. A track's state on a step is the state of its checks placed there. A step with no measured check but at least one event or media item from a shown brain is ◐ "in progress · not scored yet": it never shows a count of passing checks and never reads as a verdict. A step with neither is "not started". Once any check on a step (or on any stage of a section) is measured, the checks alone decide its state. Events and media count only for their own brain: a record with no brain, and a track's footage, mark no step. The gold latest-work node is the step where a shown brain last measured, logged or published. Skill tags stay the same: every track uses the same playbook.
  - A track with no work on any step yet shows a white card inside the frame where the lead brain would sit, above the loop regions: its status, `why` and scope from `validation.yaml`, one line each (the full text is in its tooltip and the track's card), nothing else. A track not started is the same flowchart greyed: every node idle, every arrow tan, its skill tags faded.
  - The lead brain card names only brains with something on record at that step for the track; with none, it is not drawn.
  - **A track's own chain.** A track can run the same loop steps as a different chain (LIBERO: read demos, reward from video, tools, trials with feedback, compile later). Its view then names the steps its own way: node labels, navigator words, arrow labels, the footage end's label and a several-stage step shown as one (no parts line), all from its text in `story.yaml` (section 3). A step it runs later is drawn like a planned skill: dashed, ○, "planned" and when. The first track's labels never change, and a track without overrides keeps them.
  - **External anchor.** A track measured against a published number (LIBERO: Argon's 43% on LIBERO-10 [llm-libero-2026]) shows it as a white chip on the frame's top edge, left of integrity: `EXTERNAL ANCHOR published 43% · ours ◐ 2 of 6 so far`. "Ours" is counted live from our own run's episode log, says "so far" while the run is short of its planned count, and shows only for the brain that ran it (○ "Argon only" otherwise). A click opens its card: published and ours side by side, never merged, with the sources; the same pair is a tile on the anchor's step page. It never reads as our result: nothing in it is a check until a check is recorded.
- **The two loops.** The twin loop (read, then build the twin: A, B0, B1, B2) and the task loop (task tools, practise, compile, then practise again cheaper: C, D, E), from `loops` in `story.yaml` (docs/design/vision-draft.md). The feedback box stays the bridge from the task loop back to the twin loop, and integrity stays the frame around both.
  - **At rest** each loop is a soft rounded tint behind its nodes, skill tags and clips, with no outline, and a small capital label at its top left (`TWIN LOOP`) followed by one trend glyph per shown brain (section 5). Nothing else is added: no numbers, no text. The label moves right of the lead brain's arrow when the arrow would cross it. A loop the track's source supplies (`given`, such as a simulator's exact scene) is drawn fainter with a dashed outline and the glyph `=`.
  - **Hover** (a mouse anywhere over the region, its nodes included) or **keyboard focus** on the label darkens the tint, outlines the region and opens a panel in the quietest space beside the region. Every spot along its four sides is tried; any spot that touches a node (steps, feedback box, integrity chip), the region itself or the page chrome is ruled out (the anchor chip, like the lead brain card, only counts against a spot), then the headline and subtitle are ruled out too unless nothing else fits, and the spot covering the least text wins. Where one column finds no room (a short window, a wide integrity chip), the same blocks are laid out in two short columns. With no free spot at all, or on a window under 960 × 560, there is no panel. Leaving the region, Escape or scrolling closes it.
  - **The panel**, for the shown brains and the shown track only, never merged: its label and trend in words with the brain and track; the objective in one line; the headline checks, each with its state glyph, latest value, target and, for one brain, its learning curve (a sparkline of its records); with no headline checks on the shown track, the first two numbers of the latest result each shown brain published on each of the loop's steps (four in the card, which opens the result), marked as published results and not gated checks; then "Optimized so far", one line per step (`1 READ`, `2 CAMERAS` …; a step shown as one gets one line, and beside published results a loop with nothing optimized yet says so in one line): versions first (twin parameters `params v2 → v6`, the task tool release `task/v1 r3`, compiled routines with their status, frozen practice runs `attempt 1 vs N frozen`), then the count of the step's dated lessons and the latest one's headline; last, what the loop runs on. Several brains get a value column each and a brain tag on each step line.
  - **A click, a tap or Enter** opens the loop's card: the same blocks in full (dates, skills, every frozen run, the check behind each row), what the trend glyph follows, and buttons to the loop's steps. On a touch screen or a window under 960 × 560 the card rises from the bottom as a sheet with the page still in view above it.
- **Names, not codes.** Main steps are numbered 1, 2, 3 … in loop order, and the feedback loop and integrity check are named. Stage codes (A, B0, B1 …) stay internal: they appear only in tooltips and in check ids inside cards. A section with several stages names each one with a single word from `parts` (cameras, contact, replay).
- **Moving between pages.** Scrolling moves sideways through the steps and stops on every page; the ledger scrolls freely until the builders page is half in view, then it snaps to it. The ‹ › buttons, arrow keys and `j`/`k` go one page at a time with a short slide. Jumping further (a navigator item or a flowchart node) fades straight to the page, which settles in from below. It never sweeps through the pages in between.
- **Deep links.** `#B` opens a step, `#builders` the builders page, `#track=so101` switches the page to that data track (the first track is the default and is left out), and `#brain=argon` sets the brain switch; they combine (`#B&track=so101`). The page keeps the last track and brain in the browser and writes both into the hash as the viewer moves.
- **Skill tags.** One row of tags under each node names the playbook skills that step uses, each stage's own skill first. Tags that do not fit fold into a count chip ("+4 skills") that opens the step's list of skills. Integrity's tags hang under its chip on the frame and the feedback loop's under its box. Skills that serve every step (`steps: [loop]`, such as `brains`) sit beside the frame's subtitle. The step's method card lists the same skills as chips.
- **Cards.** Every tile, flowchart clip, result, skill tag and track link opens a card holding the long read: the track's footage and what its recordings are, method and lessons, all checks with their history, media at full size, the full log, a skill (its description, Goal, checks with live state, latest lesson headline, sources count and a link to its `SKILL.md`), or a data track (its status, why, recordings and scope, its state on every step with a row that switches to it and opens that step, its decision logs, and a button to show it). A card's kicker names the step and the track it shows (`Step 2 of 5 · SO-101 · method`). Cards open at the same place near the top whatever their length, with a frosted paper backdrop so the page recedes instead of going grey. Close a card with ×, Esc, or a click outside it.
- **Small screens.** Below 960 × 560, or with `prefers-reduced-motion`, the same pages stack vertically with no pinning and no animation.

## 2. Where information goes

Every record sits on one data track and one loop step, and shows only while the page shows that track:

- **Track.** A check's comes from the stage that holds it. An event's or media item's comes from its `stage` when that is a track's container (T2 is SO-101), else from `track`, else from its `run_id` (the run ids listed under each track's `runs:` in `validation.yaml`), else it is the first track's. Records written before tracks existed have no track and stay on the first track.
- **Step.** A check's is its `step:` (for the first track, its stage). An event's or media item's is `step`, else its stage, else, for a container stage, the step most of its checks sit on (B0 for T2). Media can also take `step="input"`: the track's real footage at the loop's start. A media item without a step whose main file is under `runs/explain/` (cut_clip's folder) counts as footage too.
- The tools store `track` and `step` on every new record and refuse a track or step that does not exist, or a stage that belongs to another track. Records without them are placed by the same rules at build time (`harness/status.py` `place`).

| You have | Send it with | It appears, on its track's view |
|---|---|---|
| A measured number for a check | `record_check(check_id, value, evidence, brain)`; track and step come from the check | the check row on its step, the big-number tile if `story.yaml` lists it, the node state on the overview, the switcher's count, the history table and sparkline in the check card |
| A result: video, image, real vs sim | `publish_media(stage, title, paths, brain, ...)`; another track's result with its container stage (`stage="T2", step="B2"`) or `track="so101", step="B2"` | the step's hero tile (latest `featured` item, else latest), its "More visuals" reel, the thumbnail under its node on the overview, the step's card; a step with no measured check shows ◐ "in progress · not scored yet" |
| A track's real footage (clean clips of its recordings) | `publish_media(..., track="so101", step="input")`, or `cut_clip(episode, t0, t1, name=...)` and the clip under that track's `input: explain` in `runs/status/story.yaml` (top-level `input` for the first track) | the clip stack at "Real footage" and the synced player in its card, only while that track is shown; a track with none shows a film icon and an empty state |
| Footage that explains a step (a task, a failure) | `cut_clip(...)`, then the clip under that section's `explain` in `story.yaml` (under `tracks: <id>: sections:` for another track) | a clip under the step's node, the synced multi-camera player in that step's card |
| A decision, blocker or note | `record_event(kind, text, ...)`, with `stage`, or `track` and `step`, and `brain` | the step's log tile, the ledger timeline; with a brain, a step with no measured check shows ◐ "in progress · not scored yet" in that brain's view |
| A reconstruction decision | `log_decision(...)` (harness/tools/decisions.py) into `runs/<brain>/<run_id>/decisions.jsonl`; the run's track comes from `runs:` under the track in `validation.yaml` | the "Decision logs" section of that track's ledger and its track card |
| What the method is and what runs taught | the skill's `SKILL.md` (description, `## Goal`, `## Approach`, `## Checks`, `## Lessons`, `## Sources`) | the step's "Method and lessons" card, the skill's card |
| Which skills a step uses | `steps:` in each skill's `SKILL.md` frontmatter: the `validation.yaml` stage ids it serves (`[B2, F]`), or `[loop]` for every step. This is the one place the mapping lives. Without the field, the coarse `stage:` word places the skill (`build` puts it on every B stage, `infra` on the loop). A skill that `validation.yaml` names as a stage's `skill` always sits at that stage; before its `SKILL.md` exists it shows there as planned | a tag under the step's node on the overview, a chip in the step's method card, the skill's card |
| The narrative: headline, claim, which numbers back it | `runs/status/story.yaml`; another track's text under `tracks: <id>:` (section 3) | the overview, each step's intro, its number tiles |
| Who built it, and the line above the wordmark | `builders: {names, note}` in `runs/status/story.yaml` | the builders page |
| A new stage or check | `playbook/validation.yaml` (with a version bump and a `history` line) | a new row, or a new node once a `story.yaml` section covers the stage. Another track's check goes in its container stage with `step:` naming its loop step |
| A new data track (another kind of recordings) | an entry under `tracks` in `playbook/validation.yaml` (`id`, `name`, `scope`, `status`, `why`, `runs`), and a container stage with `track: <id>` whose checks each name a `step` | a row in the switcher, its own view of the whole loop and its card, on the next rebuild with no template change |
| A track's later stages (practice, compile, self-learning: tokens per success, attempt-N curves, brain share) | checks with `step: D` or `step: E` in that track's container stage; results and events with the same step | that track's practise and compile nodes, step pages, numbers and cost tile, with no template change |
| A loop's text: its steps, objective, headline checks, curve check, what it runs on | `loops` in `runs/status/story.yaml`; another track's under `tracks: <id>: loops: <loop id>:` (section 3), with its own checks only; `given:` for a loop the track's source supplies | the loop's region, label and trend glyph on the overview, its panel and its card |
| A published number a track is measured against, and our run of the same protocol | `anchor:` under `tracks: <id>:` in `runs/status/story.yaml` (section 3): the published value, text and reference, and our run's episode log (`episodes:`, one JSON line per finished episode with `outcome`) | the anchor chip on the frame, a tile on its step page and its card, only on that track; ours counts itself as episodes finish |
| What a loop's step optimized | nothing new to send: it is read from what is already on record (`loop_progress` in `harness/status.py`). Lessons: the dated lessons of the skills at the step, each placed by the brain its header names (author or caller) else the brain folder of the runs it cites, on the tracks of those runs, at a stage code its caller names (`claude:b2-replay` is B2) else the stage `validation.yaml` gives the skill else its first step; a lesson naming no brain counts for none. Versions: `<stage> vN` or `<stage> round N` right after the stage code, `params_vN`, and tool releases `<interface>/vN rM`, in that step's records; validation versions never count. Task tool releases also from `runs/<brain>/<run>/task_tools/manifest.json`; routines from `runs/<brain>/<run>/library/index.json` (a copy counts once; a run its manifest calls a test is left out); frozen practice runs from `runs/<brain>/<run>/manifest.json` (attempt 1 or N, not dry runs) | the "Optimized so far" lines in the loop's panel and card, per brain and track |

Milestones and regressions are logged automatically when a check flips; do not record them by hand. `status_summary` keeps each stage's `state` as its checks' verdict and adds `activity` (a brain's events and media per step) and `unscored_work` (work with no measured check yet), the same rule the page uses. Claude stand-in and Argon results always stay apart: the brain switch shows one at a time, or both side by side, never merged into one number. Tracks stay apart the same way: a number, clip or event never shows under another track. Cost per brain counts the runs a track lists under `runs:` (runs no track lists count for the first track); tool calls by caller cover every track, because callers do not name one.

### Explainer clips

`explain` blocks can sit on `input` (real footage), `output` (real arm) or any section, and, for another track, on `tracks: <id>: input` or `tracks: <id>: sections: <id>`:

```yaml
explain:
  title: The REASSEMBLE recordings        # the card's headline
  text: |                                 # Markdown; every number cites its source file
    ...
  clips:
    - title: Insert round peg 1           # what happens, in a few words
      caption: Episode 1, 42.6–60.8 s. ...  # where it is from and what to notice
      tag: success                        # success / failure, or leave it out
      paths: [runs/explain/reassemble/<name>_hama1.mp4, ..._hama2.mp4, ..._hand.mp4]
      labels: [fixed camera 1, fixed camera 2, wrist camera]
```

- One clip is one moment. Its `paths` are the camera views of that moment, played side by side and in sync.
- Cut clips with `cut_clip`: overlay-free H.264, 30 fps, a poster JPG beside each mp4, under 60 s (10–20 s reads best). Pick moments that teach something: a clean success, a typical failure, the attempt the twin replays.
- Order clips by importance; the first plays on the overview. Name files `ep<NNN>_<what happens>`.

## 3. Writing rules

Plain words, sentence case, active voice; the reader is a judge or a teammate who has never seen the repo.

`story.yaml`, per section:

| Field | Limit | Notes |
|---|---|---|
| `node` | 18 characters | the flowchart label, shown in capitals; name the step, not the tool |
| `short` | one word | the navigator label |
| `parts` | one word per stage | only for a section with several stages |
| `in`, `out` | 2 lines of 9 characters | what the step takes and hands on (`labelled steps`, `twin`); `in` defaults to the previous section's `out` |
| `title` | 8 words | a statement of what the step achieves |
| `claim` | one sentence, about 25 words | every number shown under it must back it up; if the numbers say otherwise, change the claim |
| `numbers` | 2 check ids | gated checks the claim depends on, not diagnostics |
| `read` | optional Markdown | extra method text for the method card; the skill's own text is shown already |

Top level, `tracks:` (optional) holds per-track text. Everything else in `story.yaml` is the first track's (REASSEMBLE) text.

```yaml
tracks:
  so101:
    label: SO-101                    # the switcher's label, at most 14 characters; default: the name up to its bracket
    input: {note: ..., explain: ...} # the line under the switcher (2 lines of 24 characters) and its footage card
    output: {note: ...}
    sections:
      B: {title: ..., claim: ..., numbers: [T2.kp_px, T2.replay_success], read: ..., explain: ..., feature: ...}
  libero:                            # a track that runs the loop as its own chain
    input: {label: Demo videos, out: demo video, note: ...}   # label and out rename the footage end (output: label too)
    sections:
      B: {node: Reward from video, short: Reward, out: reward code, parts: none, title: ..., claim: ...}
      E: {node: Compile, planned: later, title: ..., claim: ...}
    anchor:
      label: External anchor         # the chip's label, shown in capitals
      step: D                        # the step page that carries the anchor tile
      published: {value: 43%, text: ..., ref: llm-libero-2026, detail: ...}   # the paper's number, from docs/knowledge/
      ours: {text: ..., run_id: ..., brain: argon, episodes: runs/libero/<run>/episodes.jsonl, planned: 10, detail: ...}
      sources: [docs/knowledge/..., runs/libero/<run>/report.md]
```

- A track's `node`, `short`, `in`, `out` and `parts` rename the step on that track's view only, with the same limits as a section's fields; `parts: none` shows a several-stage step as one step; `planned: <when>` draws a step the track runs later like a planned skill until work lands on it; `pending: <words>` (a result due soon, "running, result 16:45") replaces the node's state words the same way and is never a state. The first track's labels come only from its sections.
- `anchor.published.value` is at most 6 characters; `text` names the model, benchmark and setting in one line. `ours` is never typed in: it is counted from `episodes`.

- Same limits as a section's fields. `numbers` must be that track's checks on the section's steps; others are dropped with a build warning.
- Another track never falls back to the first track's claims, titles, numbers, method text or clips. Without an override a step shows its node name as the title and says how many of the track's checks sit there ("No checks on this step yet for SO-101."), with the track's first gated checks as numbers.
- Text for a track with no results states what the step will test or why it has not started, from `validation.yaml` (`why`, `scope`) and the docs; it never reads as a result.
- Status, scope, why and runs come from `validation.yaml`. In the track card, stage codes in `scope` ("stages A-G") are left out, the name's bracket becomes the "Recordings" line, and `why` is the lead.

Top level, `cards:` (optional), one per named note:

| Field | Limit | Notes |
|---|---|---|
| `id`, `label` | label 28 characters | the pill link on the cover and the button on its step |
| `title`, `text` | Markdown | the card; every number cites its source file; a pending result says when it is due |
| `tracks` | track ids | the tracks it shows on (default every track) |
| `step` | a section id | optional: also a button on that step's page |
| `figure` | image paths | optional: shown large at the top of the card |
| `cover` | `false` | optional: only on its step's page, not on the cover |

`story.yaml`, per loop (`loops:`):

| Field | Limit | Notes |
|---|---|---|
| `id`, `label` | label 14 characters | the region's label, shown in capitals (`Twin loop`) |
| `steps` | loop stages | the `validation.yaml` stages it encloses (`[A, B0, B1, B2]`); keep them adjacent in the flowchart |
| `objective` | 56 characters | one line, what makes the loop better; the same for every track unless a track overrides it |
| `headline` | up to 5 checks, labels 18 characters | `{check id: short label}`, the checks that measure the objective; gated ones are judged against target |
| `curve` | 1 check | the check whose records give the trend glyph (default: the first headline check) |
| `better` | `lower` or `higher` | only for a tracked (no target) curve check, so the trend can say improving |
| `playground` | 60 characters | what the loop runs on for this track (default: the track's footage note) |
| `given` | 60 characters | only for a track whose source supplies the loop: shown instead of the objective, with no checks |

Another track's loop text never falls back to the first track's `headline` or `playground`: without its own, the headline is the track's checks on the loop's steps, gated first, and the panel says when there are none.

- **Media:** `title` up to 60 characters, saying what is shown (`Real jam replayed: square peg 2 (ep 2 seg 38)`). `caption` is one or two sentences saying what to look at and what it proves. Up to 4 `metrics`, using check ids as keys where possible so they are judged against their targets.
- **Clip text:** state only what the labels and the footage show. Facts about a dataset come from `docs/research/` notes, with the source named in the text.

## 4. Media rules

- Order `paths` main file first. How a media item renders:
  - `kind: compare` with two images: a wipe slider (labels from `labels`, default real / sim).
  - `kind: compare` with two videos: side by side.
  - one video plus images: the video large, up to two stills (for example the force plot) beside it.
  - images only: the first large, the rest as a strip to switch between.
- **Videos:** H.264 MP4 (`publish_media` re-encodes when needed), muted and loop-friendly. 16:9 is preferred for results and 4:3 for REASSEMBLE footage, under about 30 s. A video plays muted while its page is on screen and pauses behind an open card; cards have controls.
- **Plots:** white background, readable at 600 px wide, axis labels with units.
- **Featured item:** mark one item per stage `featured` as its hero: the most convincing evidence for the claim, preferably real next to sim.

## 5. Visual tokens

Colours (from pi.website; the CSS custom properties in the template carry the same names):

| Token | Hex | Use |
|---|---|---|
| `--paper` | `#f5f4ef` | page background, card backdrop (84% with blur) |
| `--tile` | `#eeece4` | tiles, flowchart nodes, navigator items |
| `--line` | `#d7cfb6` | node outlines, the dashed frame, the dot grid |
| `--divider` | `#e5e4dc` | rules between rows |
| `--ink` / `--ink-2` / `--ink-3` | `#29271f` / `#625c4b` / `#746d58` | text, secondary text, labels; `--ink` also fills the selected brain, the shown data track and the ‹ › buttons |
| `--gold` / `--gold-line` | `#fbd45b` / `#c9a533` | the one current thing: the latest-work node, the current page, data arrows |
| `--arrow` | `#aaa184` | idle arrows, nothing has passed yet |
| `--green` / `--green-deep` | `#9bbc6c` / `#577b37` | the feedback loop; a brain that has results |
| `--tan` | `#e6e0cb` | the two loop regions' tint (30% at rest, 50% on hover or focus, 12% for a given loop); it was already in the template's palette |

A loop's trend (its `curve` check for one brain, first record to latest) has its own glyphs, never mixed with step states: ↗ improving (`--pass`), ↘ getting worse (`--fail`), → flat, ~ tracked with no direction, • one measurement (these three `--ink-3`), ○ not measured yet (`--none`), = given by the source (`--ink-3`). At rest the region's label carries the glyph alone; its words are in the label's accessible name, the panel's heading and the card. A trend is not a verdict: a check can be improving and still miss its target, and its row says so with its state glyph.

States always travel as glyph, colour and words together: ✓ passing, ◐ in progress, ◐ in progress · not scored yet (work on record, no measured check; the node says "in progress / not scored yet", never "0/N pass"), ✕ failing, ▼ regressed, ○ not started, ~ tracked (no target). Colour never carries meaning alone. A planned skill (named in `validation.yaml`, not written yet) is a dashed tag with ○ in `--none` and `--ink-3` text, like an idle node; the key, its tooltip and its card say "planned". A not-started data track is drawn the same way: a dashed switcher row with ○, `--ink-3` text and the words "not started". An active track's ◐ is its status, not a verdict on its checks; its row's tooltip and card say "active", and its line gives its passing checks in words. The pressed switcher row turns glyph and words `--paper` on `--ink`, so the glyph's shape carries the state there.

Type:
- The system monospace stack for page chrome, the flowchart, the navigator and data.
- Source Sans 3 for prose: claims, captions, cards.
- Newsreader as the serif for the logo, headlines and big numbers. Pi uses Signifier, a licensed face, so Newsreader stands in.
- Gloock (`--display`) only for the builders wordmark, loaded for just its letters. The reference, creativeclaw.co's footer, sets its wordmark in Larken, a licensed face (read from its CSS, 24 Oct 2026); Gloock was the closest free match in a side-by-side render against Newsreader, Fraunces, DM Serif Display, Playfair Display, Instrument Serif, Young Serif and Bodoni Moda.
- Capitals only in flowchart and navigator labels, as in pi's diagram.

Geometry: 12 px radius on tiles and nodes, 14 px on cards, 8 px on media, clips and thumbnails, fully round pills on chips and skill tags, 32 px on the builders panel (22 px on phones), which is filled with `--ink` and carries paper-coloured text. Skill tags are `--paper` with a `--line` outline and 11 px monospace `--ink-2` text; the count chip is `--tile`. The data-source switcher is built like the brain switch: a `--paper` box with a `--line` outline and 12 px radius, rows of 9 px radius, a 14 px monospace capital label over a 12 px line, all in the flowchart's units so it scales with it; with more than three tracks the rows tighten from 46 to 38 px, and with five or more to 32 px (label 13 px, line 11 px), so the column (switcher, note, links) still ends above the key. The cover's card links are fully round pills, `--white` with a `--line` outline, 12.5 px semi-bold monospace `--ink`. A track's card inside the frame is the lead brain's white card (`--white`, `--line` outline, 12 px radius) with its why in Source Sans. Loop regions are 20 px rounded rectangles with no outline at rest (a 1 px `--ink-3` outline on hover, 2 px `--ink` on keyboard focus, a dashed `--line` outline when given); their label is 11.5 px bold monospace capitals in `--ink-3` with 0.1 em letter spacing, `--ink` on hover. The loop panel is the brain switch's material: `--paper` at 97% with blur, a `--line` outline, 12 px radius and the light lift; inside, an 11.5 px monospace heading and table, the objective in 14.5 px Source Sans, step lines in 13 px Source Sans with the latest lesson in italic `--ink-2`, and 52 × 14 px sparklines; it is 420–470 px wide (680 px in two columns). The loop's card is a normal card; from the bottom it has 14 px top corners only and spans up to 760 px. The anchor chip is the integrity chip's shape (30 px tall, 8 px radius) filled `--white` with a 1.5 px `--line` outline, its label 15 px bold monospace, the rest 13 px with the published value in bold `--ink`. A planned step is an idle node with a 4 3 dashed outline and `--ink-3` text. The 16 px dot grid appears on the overview only, and borders are 1 px. Two shadows: a light lift under the brain switch, the data-source switcher and the navigator, and a deep one under cards.

## 6. Motion

Motion shows the flow of information or answers the viewer; it is never decoration.

- **Arrows:** moving dashes on an arrow mean data has passed along it, because the downstream step has measurements or visuals. A still tan arrow means nothing yet. The arrow to the real arm stays dashed and still until a real arm confirms a result. The navigator's arrows follow the same rule.
- **Overview:** draws itself once on first load (skipped on auto-reload and deep links); skill tags, the switcher and a track's card fade in with the nodes and have no motion of their own. The front dataset clip plays while the overview is on screen.
- **Switching data track:** instant, with no motion of its own: the page re-renders in place on the same page, the new track's front clip starts and its flow dashes run where its data has passed. The overview does not draw itself again.
- **Steps:** sideways scrolling with snapping, a slight lag on the intro heading, and sparklines drawn once when their step first appears.
- **Page changes:** one page is a 0.7 s slide; further is a 0.16 s fade out, then the page settles in from below.
- **Cards:** open with a short fade and rise. A loop's card opened from the bottom rises 48 px in 0.28 s.
- **Loop regions:** no motion of their own; they fade in with the nodes when the overview draws itself. On hover or focus the tint and outline change in 0.15 s. The panel appears 0.11 s after the pointer settles in a region (at once when moving from one region's panel to the other), fades in and rises 4 px in 0.15 s, and fades out 0.16 s after the pointer leaves. It never follows the pointer: it stays where it opened until it closes.
- **Reduced motion** (`prefers-reduced-motion`): no animation, no pinning, flow dashes hidden; the loop panel appears and disappears without fading.

## 7. Changing the page

- New information never needs a template change: use the tools in section 2.
- **A new section:** add it to `story.yaml` with `stages`, `node`, `short`, `out`, `title`, `claim` and `numbers`. The stage's role in `validation.yaml` places its node: `main` in the numbered row, `feedback` on the loop below, `integrity` on the frame.
- **A new data track:** add it under `tracks` in `validation.yaml` (with `runs:` once it has runs), put its checks in a container stage with `track: <id>`, and give each check the `step:` it belongs to; never list the container in a `story.yaml` section (the build drops it and reports it under `track_warnings`). Add its text under `tracks: <id>:` in `story.yaml`, with step names of its own if it runs the loop as a different chain, and an `anchor` if a published number measures it. The switcher row, its view of the loop and its card appear on the next rebuild. When validation.yaml cannot change yet (a live frozen run hashes it), a track can be defined provisionally in `story.yaml` under `tracks: <id>: define:` (`name`, `scope`, `status`, `why`, `runs`): it gets its switcher row, loop view, events and media, but no checks, and the build lists it under `track_warnings`. Once validation.yaml lists the track, that entry wins and the `define:` block is removed.
- **A new skill:** add `steps: [...]` to its `SKILL.md` frontmatter. Its tag appears on the next rebuild. The build output lists the skills per step and warns about a skill that sits on no step or names a stage `validation.yaml` lacks.
- **A new kind of tile:** add one renderer function in the template's chapter block and give it a grid area or put it in the right-hand `.side` stack. Add a card in `SHEETS` if it has more to say, and add a row to the table in section 2.
- Keep the single page, the token names, the state encoding and the motion rules. Anything that adds a colour, a font or a new kind of motion is a protocol change: edit this file in the same change.
- **Before calling a change done:** rebuild, then look at these, checking for console errors and for horizontal overflow on narrow screens:
  - at 1600 × 1000: the overview on every track (switch with the switcher and with the keyboard), two steps, one card, a track card, a far jump, the ledger and the builders page;
  - at 1280 × 720: the overview fits above the navigator on every track, and the navigator shows every item without scrolling, beside validation, Live and Builders;
  - at 390 × 844: the overview and one step;
  - for a change to records or tracks: simulated records (a check, a media item, an event) written by the tools into a temporary copy of `runs/status/` appear under their own track only, never in the real records.
  - for the loop regions and their panels: `.venv/bin/python -m tests.ui_loops_check` (headless Chrome over the page server, or the built file). It checks rest, hover and keyboard focus at 1600 × 1000 and 1280 × 720 on every track for the Claude stand-in and for both brains, the bottom sheet on a 390 × 844 touch screen, no overflow, no console errors, that a panel never covers a node or its own region, and a simulated fourth track whose twin loop is given; screenshots go to `runs/status/ui-check/`.
