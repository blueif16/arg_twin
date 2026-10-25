"""Headless-Chrome check of the overview's loop regions and their panels (docs/design/ui-protocol.md).

    .venv/bin/python -m tests.ui_loops_check              # rebuild the page, check it, save screenshots
    .venv/bin/python -m tests.ui_loops_check --no-build   # check the page as it is
    .venv/bin/python -m tests.ui_loops_check --url file:///.../index.html

What it checks, for the Claude stand-in and every brain together, on every data track (switching with #track=...):
- 1600x1000 and 1280x720: rest state; hovering each loop opens its panel; the panel lies inside the viewport and never
  covers a node (steps, feedback box, integrity chip) or its own region; moving away closes it; the panel is still
  there while the pointer is over a node inside the region; Tab reaches each loop's label and focus opens its panel,
  Escape closes it, Enter opens the loop's card.
- 390x844 (touch): rest state; a tap on each loop's label opens its card from the bottom (a bottom sheet).
- No horizontal page overflow and no console errors anywhere (failed requests to the external font host are listed
  apart as network noise, since the fonts fall back to local faces).
- A simulated fourth track, built into a temporary copy of runs/status/ with a copy of validation.yaml: LIBERO, whose
  twin loop is given by the simulator. Records written by the real tools (record_check, record_event) land on
  LIBERO only; its twin region is drawn as given; its task panel shows its own records and trend; REASSEMBLE's
  panels never show them. The real records and validation.yaml are not touched.

Screenshots go to runs/status/ui-check/. Exit code 0 when every check passes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.cdp import Chrome  # noqa: E402

OUT = REPO / "runs" / "status" / "ui-check"
SERVER = "http://localhost:8765/runs/status/index.html"
FONT_HOSTS = ("fonts.googleapis.com", "fonts.gstatic.com")
NODES = "#cover .nd, #cover .fb, #cover .gchip"

results: list[tuple[bool, str]] = []
noise: list[str] = []


def check(ok: bool, what: str) -> bool:
    results.append((bool(ok), what))
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    return bool(ok)


def errors_of(page, where: str) -> None:
    real = [e for e in page.errors if not any(h in e for h in FONT_HOSTS)]
    noise.extend(f"{where}: {e}" for e in page.errors if any(h in e for h in FONT_HOSTS))
    check(not real, f"{where}: no console errors" + (f" (got {real[:3]})" if real else ""))
    page.errors.clear()


JS_RECTS = """(() => {
  const r = (e) => { const b = e.getBoundingClientRect(); return {l: b.left, t: b.top, r: b.right, b: b.bottom}; };
  const p = document.querySelector('#lpp');
  return {on: p.classList.contains('on'), panel: r(p), text: p.textContent,
          nodes: [...document.querySelectorAll('%s')].map(r),
          regions: Object.fromEntries([...document.querySelectorAll('#cover .lp-r')].map((e) => [e.dataset.loop, r(e)])),
          vw: document.documentElement.clientWidth, vh: innerHeight};
})()""" % NODES


def overlap(a: dict, b: dict) -> float:
    return max(0.0, min(a["r"], b["r"]) - max(a["l"], b["l"])) * max(0.0, min(a["b"], b["b"]) - max(a["t"], b["t"]))


def no_overflow(page, where: str) -> None:
    sw = page.eval("[document.documentElement.scrollWidth, document.documentElement.clientWidth]")
    check(sw[0] <= sw[1] + 1, f"{where}: no horizontal overflow ({sw[0]} <= {sw[1]})")


def label_center(page, loop: str):
    return page.eval(f"""(() => {{ const e = document.querySelector('#cover .lpl[data-loop="{loop}"]'); if (!e) return null;
      const b = e.getBoundingClientRect(); return [b.left + b.width / 2, b.top + b.height / 2]; }})()""")


def panel_ok(page, loop: str, where: str) -> dict:
    s = page.eval(JS_RECTS)
    p, own = s["panel"], s["regions"].get(loop)
    check(s["on"], f"{where}: the {loop} panel opens")
    if not s["on"]:
        return s
    check(p["l"] >= 0 and p["t"] >= 0 and p["r"] <= s["vw"] and p["b"] <= s["vh"], f"{where}: the {loop} panel is inside the viewport")
    hits = [n for n in s["nodes"] if overlap(p, n) > 0]
    check(not hits, f"{where}: the {loop} panel covers no node" + (f" (covers {len(hits)})" if hits else ""))
    check(own is not None and overlap(p, own) == 0, f"{where}: the {loop} panel does not cover its own region")
    return s


def tab_to(page, selector: str, limit: int = 90) -> bool:
    page.eval("document.activeElement && document.activeElement.blur(); window.scrollTo(0, 0); true")
    for _ in range(limit):
        page.key("Tab")
        page.drain(0.03)
        if page.eval(f"!!(document.activeElement && document.activeElement.matches('{selector}'))"):
            return True
    return False


def desktop(c: Chrome, url: str, w: int, h: int, tracks: list[str]) -> None:
    for brain in ("claude", "all"):
        for track in tracks:
            frag = f"#brain={brain}" + ("" if track == tracks[0] else f"&track={track}")
            page = c.page(w, h)
            page.goto(url + frag, settle=3.0)
            where = f"{w}x{h} {brain} {track}"
            n = page.eval("document.querySelectorAll('#cover .lp-r').length")
            check(n == 2, f"{where}: two loop regions drawn (got {n})")
            no_overflow(page, where)
            tag = f"{w}-{brain}-{track}"
            if brain == "claude":
                page.shot(OUT / f"rest-{tag}.png")
            for loop in ("twin", "task"):
                at = label_center(page, loop)
                if not check(at is not None, f"{where}: {loop} label drawn"):
                    continue
                page.mouse(at[0] + 30, at[1] + 70)  # inside the region, then onto its label
                page.drain(0.15)
                page.mouse(*at)
                page.drain(0.5)
                s = panel_ok(page, loop, f"{where} hover")
                if s.get("on") and brain == "claude" or (brain == "all" and track == tracks[0]):
                    page.shot(OUT / f"hover-{loop}-{tag}.png")
                node = page.eval(f"""(() => {{ const r = document.querySelector('#cover .lp-r[data-loop="{loop}"]').getBoundingClientRect();
                  const n = [...document.querySelectorAll('#cover .nd')].map((e) => e.getBoundingClientRect())
                    .find((b) => b.left > r.left && b.right < r.right); return n ? [n.left + n.width / 2, n.top + n.height / 2] : null; }})()""")
                if node:  # over a node inside the region, the panel stays open and still off every node
                    page.mouse(*node)
                    page.drain(0.35)
                    panel_ok(page, loop, f"{where} over a node")
                page.mouse(w / 2, h - 110)  # the key line, outside both regions
                page.drain(0.45)
                check(not page.eval("document.querySelector('#lpp').classList.contains('on')"), f"{where}: the {loop} panel closes on leave")
            if brain == "claude" and track == tracks[0]:  # keyboard: Tab to each label, focus opens, Escape closes, Enter opens the card
                for loop in ("twin", "task"):
                    if check(tab_to(page, f'.lpl[data-loop="{loop}"]'), f"{where}: Tab reaches the {loop} label"):
                        page.drain(0.25)
                        panel_ok(page, loop, f"{where} focus")
                        check(page.eval(f"""!!document.querySelector('#cover .lp-r.kb[data-loop="{loop}"]')"""), f"{where}: focus outlines the {loop} region")
                        page.shot(OUT / f"focus-{loop}-{w}.png")
                        page.key("Escape")
                        page.drain(0.3)
                        check(not page.eval("document.querySelector('#lpp').classList.contains('on')"), f"{where}: Escape closes the {loop} panel")
                        page.key("Enter", "Enter")
                        page.drain(0.6)
                        title = page.eval("document.querySelector('#sheet').open ? document.querySelector('#sheet-title').textContent : ''")
                        check(title.lower().startswith(loop), f"{where}: Enter opens the {loop} card (title {title!r})")
                        if w == 1600:
                            page.shot(OUT / f"card-{loop}-{w}.png")
                        page.key("Escape")
                        page.drain(0.4)
            errors_of(page, where)
            page.close()


def touch(c: Chrome, url: str, tracks: list[str]) -> None:
    w, h = 390, 844
    for track in tracks:
        page = c.page(w, h, mobile=True)
        frag = "#brain=claude" + ("" if track == tracks[0] else f"&track={track}")
        page.goto(url + frag, settle=3.0)
        where = f"{w}x{h} claude {track}"
        no_overflow(page, where)
        if track == tracks[0]:
            page.shot(OUT / f"rest-{w}.png")
        for loop in ("twin", "task"):
            page.eval(f"""(() => {{ const e = document.querySelector('#cover .lpl[data-loop="{loop}"]'), f = document.querySelector('#cover .fsc');
              f.scrollLeft = Math.max(0, e.getBoundingClientRect().left - f.getBoundingClientRect().left + f.scrollLeft - 40);
              e.scrollIntoView({{block: 'center', inline: 'nearest'}}); return true; }})()""")
            page.drain(0.4)
            at = label_center(page, loop)
            if not check(at is not None and 0 < at[0] < w and 0 < at[1] < h, f"{where}: {loop} label on screen"):
                continue
            page.tap(*at)
            page.drain(0.8)
            st = page.eval("""(() => { const d = document.querySelector('#sheet'), b = d.getBoundingClientRect();
              return {open: d.open, bottom: d.classList.contains('bottom'), title: document.querySelector('#sheet-title').textContent,
                      top: b.top, bot: b.bottom, h: innerHeight, panel: document.querySelector('#lpp').classList.contains('on')}; })()""")
            check(st["open"] and st["title"].lower().startswith(loop), f"{where}: a tap opens the {loop} card")
            check(st["bottom"] and abs(st["bot"] - st["h"]) <= 2 and st["top"] > 0, f"{where}: the {loop} card is a bottom sheet")
            check(not st["panel"], f"{where}: no hover panel on touch")
            no_overflow(page, f"{where} {loop} sheet")
            if track == tracks[0]:
                page.shot(OUT / f"sheet-{loop}-{w}.png")
            page.eval("document.querySelector('#sheet').close(); true")
            page.drain(0.3)
        errors_of(page, where)
        page.close()


def simulate_libero(c: Chrome) -> None:
    """An extra simulated track (id libsim, LIBERO-like), in temporary copies: its twin loop is given by the
    simulator. Its own id, so it never clashes with the real LIBERO track."""
    import harness.status as st
    from harness.tools import checks as ck

    tmp = Path(tempfile.mkdtemp(prefix="bot18-ui-"))
    saved = (st.VALIDATION, st.STATUS_DIR)
    try:
        raw = yaml.safe_load(st.VALIDATION.read_text())
        raw["tracks"].append({"id": "libsim", "name": "LIBERO (sim playground, exact ground truth)", "status": "active",
                              "scope": "task loop only; the twin is the simulator's own scene",
                              "why": "Simulated test only: a fourth track whose twin loop is given by the simulator.",
                              "runs": ["run-libero-sim-1"]})
        raw["stages"].append({"id": "T9", "name": "LIBERO practice (simulated)", "track": "libsim", "skill": "practice", "checks": [
            {"id": "T9.success_rate", "step": "D", "name": "LIBERO-10 success", "metric": "success rate", "direction": ">=", "target": 0.5},
            {"id": "T9.tokens_per_success", "step": "D", "name": "Tokens per success", "metric": "tokens", "direction": "track"}]})
        (tmp / "validation.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
        sd = tmp / "status"
        sd.mkdir()
        for f in ("checks.jsonl", "events.jsonl", "media.jsonl", "story.yaml"):
            if (st.STATUS_DIR / f).exists():
                shutil.copy2(st.STATUS_DIR / f, sd / f)
        story = yaml.safe_load((sd / "story.yaml").read_text())
        story.setdefault("tracks", {})["libsim"] = {"label": "LIBERO", "input": {"note": "simulated scenes, exact state"}, "loops": {
            "twin": {"given": "Given by the simulator: exact scene files", "playground": "LIBERO scene files"},
            "task": {"headline": {"T9.success_rate": "success", "T9.tokens_per_success": "tokens per success"},
                     "better": {"T9.tokens_per_success": "lower"}, "playground": "LIBERO-90 practice, LIBERO-10 held out"}}}
        (sd / "story.yaml").write_text(yaml.safe_dump(story, sort_keys=False, allow_unicode=True))
        st.VALIDATION, st.STATUS_DIR = tmp / "validation.yaml", sd
        ev = ["playbook/validation.yaml"]
        ck.record_check("T9.success_rate", 0.3, ev, "claude", by="ui-check", run_id="run-libero-sim-1", note="simulated record")
        ck.record_check("T9.success_rate", 0.46, ev, "claude", by="ui-check", run_id="run-libero-sim-1", note="simulated record")
        ck.record_event("note", "Simulated LIBERO practice note (ui check).", brain="claude", by="ui-check",
                        track="libsim", step="D", run_id="run-libero-sim-1")
        data = st.collect()
        lv = data["views"]["libsim"]["loops"]
        check(lv["twin"]["given"] and not lv["twin"]["headline"], "libero (simulated): the twin loop is given, with no headline checks")
        check([c for c, _ in lv["task"]["headline"]] == ["T9.success_rate", "T9.tokens_per_success"], "libero (simulated): task headline is its own checks")
        tr = data["results"]["claude"]["tracks"]
        check(tr["libsim"]["n"] == 2 and tr["reassemble"]["n"] == data["results"]["claude"]["n"] - 2 - tr["so101"]["n"],
              "libero (simulated): its records count on LIBERO only")
        check(all(e["track"] == "libsim" for e in data["events"] if "LIBERO practice note" in e["text"]), "libero (simulated): its event sits on LIBERO")
        check(not any(c == "T9.success_rate" for v in (data["views"]["reassemble"], data["views"]["so101"])
                      for c, _ in v["loops"]["task"]["headline"]), "libero (simulated): other tracks never list its checks")
        st.build()
        url = (sd / "index.html").as_uri()
        page = c.page(1600, 1000)
        page.goto(url + "#brain=claude&track=libsim", settle=3.0)
        where = "1600x1000 claude libero (simulated)"
        check(page.eval("!!document.querySelector('#cover .lp-r.given[data-loop=\"twin\"]')"), f"{where}: the twin region is drawn as given")
        check(page.eval("document.querySelector('#cover .lpl[data-loop=\"twin\"]').textContent.includes('=')"), f"{where}: the twin label shows the given glyph")
        for loop in ("twin", "task"):
            at = label_center(page, loop)
            page.mouse(*at)
            page.drain(0.5)
            s = panel_ok(page, loop, where)
            if loop == "twin":
                check("Given by the simulator" in s["text"], f"{where}: the twin panel says it is given")
            else:
                check("0.46" in s["text"] and "improving" in s["text"], f"{where}: the task panel shows LIBERO's own record and its trend")
            page.shot(OUT / f"hover-{loop}-1600-libero-simulated.png")
            page.mouse(800, 890)
            page.drain(0.4)
        page.goto(url + "#brain=claude", settle=2.5)
        at = label_center(page, "task")
        page.mouse(*at)
        page.drain(0.5)
        txt = page.eval("document.querySelector('#lpp').textContent")
        check("0.46" not in txt, "1600x1000 claude reassemble (with libero simulated): its task panel never shows LIBERO's record")
        no_overflow(page, where)
        errors_of(page, where)
        page.close()
    finally:
        st.VALIDATION, st.STATUS_DIR = saved
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=SERVER)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--skip-sim", action="store_true", help="skip the simulated LIBERO track")
    a = ap.parse_args()
    import harness.status as st
    if not a.no_build:
        st.build()
    url = a.url
    if url.startswith("http"):
        import urllib.request
        try:
            urllib.request.urlopen(url, timeout=3).read(64)
        except OSError:
            url = st.page_path().as_uri()
            print(f"note: {a.url} is not served; checking {url}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tracks = [t["id"] for t in st.load_validation()["tracks"]]
    with Chrome() as c:
        desktop(c, url, 1600, 1000, tracks)
        desktop(c, url, 1280, 720, tracks)
        touch(c, url, tracks)
        if not a.skip_sim:
            simulate_libero(c)
    bad = [w for ok, w in results if not ok]
    print(json.dumps({"checks": len(results), "failed": len(bad), "failures": bad, "external_font_noise": len(noise),
                      "screenshots": str(OUT.relative_to(REPO))}, indent=1), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
