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
                   REAL FOOTAGE (clips) -> [ 1 -> 2 -> 3 -> 4 -> 5 ] -> REAL ARM
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
- **Builders page.** The last page, after the ledger, reached with › from the ledger or the Builders link (gold while it is open). Names and the note come from `builders` in `story.yaml`. The wordmark is sized to the panel's width, at most half its height; on phones it breaks into two lines.
- **Data tracks and the switcher.** The kinds of recordings the playbook runs on, from `tracks` in `validation.yaml`, read at every build. Every track runs through the same loop steps (the stages without `track:`, A to G). The first track's checks are those steps' own; another track keeps its checks in a container stage (`track: so101`, stage T2) and each check names the step it sits on (`step: B0`). A container stage is never a node, a step or a navigator item.
  - Under the flowchart's real-footage clips sits the data-source switcher: one row per track, in `validation.yaml` order, each with the state glyph of its `status` (active ◐, done ✓, not started ○) and one line: passing gated checks over all its steps ("13/31 checks pass", per brain, never merged), "nothing measured yet", or, for a track not started, "not started" on a dashed row like a planned skill. The pressed row is filled `--ink`, like the brain switch. Under it: the track's `note`, "watch N clips" and "about <track>", which opens the track's card.
  - Pressing a row switches the whole page to that track at once, from data built in at build time (no fetch, so `--bundle` keeps working): the clips at the start, every node's state, the arrows, the lead brain, the integrity chip and feedback box, every step page's text, checks, numbers, media and log, the navigator's dots, every card, the brain switch's counts and the ledger. A track's state on a step is the state of its checks placed there; a step with none of its checks is "not started". Skill tags stay the same: every track uses the same playbook.
  - A track with no work on any step yet shows a white card inside the frame where the lead brain would sit: its status, `why` and scope from `validation.yaml`, nothing else. A track not started is the same flowchart greyed: every node idle, every arrow tan, its skill tags faded.
  - The lead brain card names only brains with something on record at that step for the track; with none, it is not drawn.
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
| A result: video, image, real vs sim | `publish_media(stage, title, paths, brain, ...)`; another track's result with its container stage (`stage="T2", step="B2"`) or `track="so101", step="B2"` | the step's hero tile (latest `featured` item, else latest), its "More visuals" reel, the thumbnail under its node on the overview, the step's card |
| A track's real footage (clean clips of its recordings) | `publish_media(..., track="so101", step="input")`, or `cut_clip(episode, t0, t1, name=...)` and the clip under that track's `input: explain` in `runs/status/story.yaml` (top-level `input` for the first track) | the clip stack at "Real footage" and the synced player in its card, only while that track is shown; a track with none shows a film icon and an empty state |
| Footage that explains a step (a task, a failure) | `cut_clip(...)`, then the clip under that section's `explain` in `story.yaml` (under `tracks: <id>: sections:` for another track) | a clip under the step's node, the synced multi-camera player in that step's card |
| A decision, blocker or note | `record_event(kind, text, ...)`, with `stage`, or `track` and `step` | the step's log tile, the ledger timeline |
| A reconstruction decision | `log_decision(...)` (harness/tools/decisions.py) into `runs/<brain>/<run_id>/decisions.jsonl`; the run's track comes from `runs:` under the track in `validation.yaml` | the "Decision logs" section of that track's ledger and its track card |
| What the method is and what runs taught | the skill's `SKILL.md` (description, `## Goal`, `## Approach`, `## Checks`, `## Lessons`, `## Sources`) | the step's "Method and lessons" card, the skill's card |
| Which skills a step uses | `steps:` in each skill's `SKILL.md` frontmatter: the `validation.yaml` stage ids it serves (`[B2, F]`), or `[loop]` for every step. This is the one place the mapping lives. Without the field, the coarse `stage:` word places the skill (`build` puts it on every B stage, `infra` on the loop). A skill that `validation.yaml` names as a stage's `skill` always sits at that stage; before its `SKILL.md` exists it shows there as planned | a tag under the step's node on the overview, a chip in the step's method card, the skill's card |
| The narrative: headline, claim, which numbers back it | `runs/status/story.yaml`; another track's text under `tracks: <id>:` (section 3) | the overview, each step's intro, its number tiles |
| Who built it, and the line above the wordmark | `builders: {names, note}` in `runs/status/story.yaml` | the builders page |
| A new stage or check | `playbook/validation.yaml` (with a version bump and a `history` line) | a new row, or a new node once a `story.yaml` section covers the stage. Another track's check goes in its container stage with `step:` naming its loop step |
| A new data track (another kind of recordings) | an entry under `tracks` in `playbook/validation.yaml` (`id`, `name`, `scope`, `status`, `why`, `runs`), and a container stage with `track: <id>` whose checks each name a `step` | a row in the switcher, its own view of the whole loop and its card, on the next rebuild with no template change |
| A track's later stages (practice, compile, self-learning: tokens per success, attempt-N curves, brain share) | checks with `step: D` or `step: E` in that track's container stage; results and events with the same step | that track's practise and compile nodes, step pages, numbers and cost tile, with no template change |

Milestones and regressions are logged automatically when a check flips; do not record them by hand. Claude stand-in and Argon results always stay apart: the brain switch shows one at a time, or both side by side, never merged into one number. Tracks stay apart the same way: a number, clip or event never shows under another track. Cost per brain counts the runs a track lists under `runs:` (runs no track lists count for the first track); tool calls by caller cover every track, because callers do not name one.

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
```

- Same limits as a section's fields. `numbers` must be that track's checks on the section's steps; others are dropped with a build warning.
- Another track never falls back to the first track's claims, titles, numbers, method text or clips. Without an override a step shows its node name as the title and says how many of the track's checks sit there ("No checks on this step yet for SO-101."), with the track's first gated checks as numbers.
- Text for a track with no results states what the step will test or why it has not started, from `validation.yaml` (`why`, `scope`) and the docs; it never reads as a result.
- Status, scope, why and runs come from `validation.yaml`. In the track card, stage codes in `scope` ("stages A-G") are left out, the name's bracket becomes the "Recordings" line, and `why` is the lead.

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

States always travel as glyph, colour and words together: ✓ passing, ◐ in progress, ✕ failing, ▼ regressed, ○ not started, ~ tracked (no target). Colour never carries meaning alone. A planned skill (named in `validation.yaml`, not written yet) is a dashed tag with ○ in `--none` and `--ink-3` text, like an idle node; the key, its tooltip and its card say "planned". A not-started data track is drawn the same way: a dashed switcher row with ○, `--ink-3` text and the words "not started". An active track's ◐ is its status, not a verdict on its checks; its row's tooltip and card say "active", and its line gives its passing checks in words. The pressed switcher row turns glyph and words `--paper` on `--ink`, so the glyph's shape carries the state there.

Type:
- The system monospace stack for page chrome, the flowchart, the navigator and data.
- Source Sans 3 for prose: claims, captions, cards.
- Newsreader as the serif for the logo, headlines and big numbers. Pi uses Signifier, a licensed face, so Newsreader stands in.
- Gloock (`--display`) only for the builders wordmark, loaded for just its letters. The reference, creativeclaw.co's footer, sets its wordmark in Larken, a licensed face (read from its CSS, 24 Oct 2026); Gloock was the closest free match in a side-by-side render against Newsreader, Fraunces, DM Serif Display, Playfair Display, Instrument Serif, Young Serif and Bodoni Moda.
- Capitals only in flowchart and navigator labels, as in pi's diagram.

Geometry: 12 px radius on tiles and nodes, 14 px on cards, 8 px on media, clips and thumbnails, fully round pills on chips and skill tags, 32 px on the builders panel (22 px on phones), which is filled with `--ink` and carries paper-coloured text. Skill tags are `--paper` with a `--line` outline and 11 px monospace `--ink-2` text; the count chip is `--tile`. The data-source switcher is built like the brain switch: a `--paper` box with a `--line` outline and 12 px radius, rows of 9 px radius, a 14 px monospace capital label over a 12 px line, all in the flowchart's units so it scales with it. A track's card inside the frame is the lead brain's white card (`--white`, `--line` outline, 12 px radius) with its why in Source Sans. The 16 px dot grid appears on the overview only, and borders are 1 px. Two shadows: a light lift under the brain switch, the data-source switcher and the navigator, and a deep one under cards.

## 6. Motion

Motion shows the flow of information or answers the viewer; it is never decoration.

- **Arrows:** moving dashes on an arrow mean data has passed along it, because the downstream step has measurements or visuals. A still tan arrow means nothing yet. The arrow to the real arm stays dashed and still until a real arm confirms a result. The navigator's arrows follow the same rule.
- **Overview:** draws itself once on first load (skipped on auto-reload and deep links); skill tags, the switcher and a track's card fade in with the nodes and have no motion of their own. The front dataset clip plays while the overview is on screen.
- **Switching data track:** instant, with no motion of its own: the page re-renders in place on the same page, the new track's front clip starts and its flow dashes run where its data has passed. The overview does not draw itself again.
- **Steps:** sideways scrolling with snapping, a slight lag on the intro heading, and sparklines drawn once when their step first appears.
- **Page changes:** one page is a 0.7 s slide; further is a 0.16 s fade out, then the page settles in from below.
- **Cards:** open with a short fade and rise.
- **Reduced motion** (`prefers-reduced-motion`): no animation, no pinning, flow dashes hidden.

## 7. Changing the page

- New information never needs a template change: use the tools in section 2.
- **A new section:** add it to `story.yaml` with `stages`, `node`, `short`, `out`, `title`, `claim` and `numbers`. The stage's role in `validation.yaml` places its node: `main` in the numbered row, `feedback` on the loop below, `integrity` on the frame.
- **A new data track:** add it under `tracks` in `validation.yaml` (with `runs:` once it has runs), put its checks in a container stage with `track: <id>`, and give each check the `step:` it belongs to; never list the container in a `story.yaml` section (the build drops it and reports it under `track_warnings`). Add its text under `tracks: <id>:` in `story.yaml`. The switcher row, its view of the loop and its card appear on the next rebuild.
- **A new skill:** add `steps: [...]` to its `SKILL.md` frontmatter. Its tag appears on the next rebuild. The build output lists the skills per step and warns about a skill that sits on no step or names a stage `validation.yaml` lacks.
- **A new kind of tile:** add one renderer function in the template's chapter block and give it a grid area or put it in the right-hand `.side` stack. Add a card in `SHEETS` if it has more to say, and add a row to the table in section 2.
- Keep the single page, the token names, the state encoding and the motion rules. Anything that adds a colour, a font or a new kind of motion is a protocol change: edit this file in the same change.
- **Before calling a change done:** rebuild, then look at these, checking for console errors and for horizontal overflow on narrow screens:
  - at 1600 × 1000: the overview on every track (switch with the switcher and with the keyboard), two steps, one card, a track card, a far jump, the ledger and the builders page;
  - at 1280 × 720: the overview fits above the navigator on every track, and the navigator shows every item without scrolling, beside validation, Live and Builders;
  - at 390 × 844: the overview and one step;
  - for a change to records or tracks: simulated records (a check, a media item, an event) written by the tools into a temporary copy of `runs/status/` appear under their own track only, never in the real records.
