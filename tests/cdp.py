"""A tiny Chrome DevTools Protocol driver for page checks: headless Chrome, no extra packages.

Uses the `websockets` package already in .venv and the system google-chrome. Used by tests/ui_loops_check.py.

    with Chrome() as c:
        page = c.page(1600, 1000)
        page.goto("http://localhost:8765/runs/status/index.html")
        page.eval("document.title")
        page.shot("runs/status/ui-check/x.png")
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

CHROME = shutil.which("google-chrome") or shutil.which("google-chrome-stable") or shutil.which("chromium")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Page:
    def __init__(self, chrome: "Chrome", ws_url: str):
        self.c, self.ws_url, self.n = chrome, ws_url, 0
        self.loop = chrome.loop
        self.ws = self.loop.run_until_complete(websockets.connect(ws_url, max_size=2 ** 28))
        self.errors: list[str] = []
        self.events: list[dict] = []
        for m in ("Runtime.enable", "Log.enable", "Page.enable"):
            self.send(m)

    async def _send(self, method: str, params: dict | None = None, timeout: float = 60):
        self.n += 1
        mid = self.n
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        end = time.time() + timeout
        while True:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=max(0.1, end - time.time()))
            msg = json.loads(raw)
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            self._event(msg)

    def _event(self, msg: dict) -> None:
        m, p = msg.get("method"), msg.get("params", {})
        if m == "Runtime.exceptionThrown":
            d = p.get("exceptionDetails", {})
            self.errors.append("exception: " + str((d.get("exception") or {}).get("description") or d.get("text")))
        elif m == "Runtime.consoleAPICalled" and p.get("type") in ("error", "assert"):
            self.errors.append("console: " + " ".join(str(a.get("value", a.get("description", ""))) for a in p.get("args", [])))
        elif m == "Log.entryAdded" and p.get("entry", {}).get("level") == "error":
            e = p["entry"]
            self.errors.append(f"log: {e.get('text')} {e.get('url', '')}".strip())
        elif m:
            self.events.append(msg)

    def send(self, method: str, params: dict | None = None, timeout: float = 60):
        return self.loop.run_until_complete(self._send(method, params, timeout))

    def drain(self, seconds: float) -> None:
        """Let the page run for a while, collecting console errors."""
        async def run():
            end = time.time() + seconds
            while time.time() < end:
                try:
                    self._event(json.loads(await asyncio.wait_for(self.ws.recv(), timeout=max(0.05, end - time.time()))))
                except asyncio.TimeoutError:
                    break
        self.loop.run_until_complete(run())

    def size(self, w: int, h: int, mobile: bool = False) -> None:
        self.send("Emulation.setDeviceMetricsOverride", {"width": w, "height": h, "deviceScaleFactor": 1, "mobile": mobile})
        self.send("Emulation.setTouchEmulationEnabled", {"enabled": mobile, **({"maxTouchPoints": 5} if mobile else {})})

    def goto(self, url: str, settle: float = 2.5, reduce_motion: bool = False) -> None:
        self.send("Emulation.setEmulatedMedia", {"features": [{"name": "prefers-reduced-motion",
                                                                "value": "reduce" if reduce_motion else "no-preference"}]})
        self.events.clear()
        self.send("Page.navigate", {"url": url})
        end = time.time() + 30
        while time.time() < end and not any(e.get("method") == "Page.loadEventFired" for e in self.events):
            self.drain(0.2)
        self.drain(settle)

    def eval(self, expr: str, await_promise: bool = False):
        r = self.send("Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": await_promise})
        if "exceptionDetails" in r:
            raise RuntimeError(f"eval failed: {r['exceptionDetails'].get('exception', {}).get('description') or r['exceptionDetails']}")
        return r.get("result", {}).get("value")

    def mouse(self, x: float, y: float, kind: str = "mouseMoved", button: str = "none") -> None:
        self.send("Input.dispatchMouseEvent", {"type": kind, "x": x, "y": y, "button": button, "clickCount": 1 if button != "none" else 0})

    def click(self, x: float, y: float) -> None:
        self.mouse(x, y)
        self.mouse(x, y, "mousePressed", "left")
        self.mouse(x, y, "mouseReleased", "left")

    def tap(self, x: float, y: float) -> None:
        pt = [{"x": x, "y": y}]
        self.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": pt})
        self.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})

    def key(self, key: str, code: str | None = None, shift: bool = False) -> None:
        vk = {"Tab": 9, "Enter": 13, "Escape": 27, " ": 32}.get(key, 0)
        base = {"key": key, "code": code or key, "windowsVirtualKeyCode": vk, "modifiers": 8 if shift else 0}
        self.send("Input.dispatchKeyEvent", {"type": "keyDown", **base})
        self.send("Input.dispatchKeyEvent", {"type": "keyUp", **base})

    def shot(self, path: str | Path, full: bool = False) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        params = {"format": "png"}
        if full:
            params["captureBeyondViewport"] = True
        data = self.send("Page.captureScreenshot", params)["data"]
        p.write_bytes(base64.b64decode(data))
        return p

    def close(self) -> None:
        try:
            self.loop.run_until_complete(self.ws.close())
        except Exception:
            pass


class Chrome:
    def __init__(self):
        if not CHROME:
            raise RuntimeError("google-chrome not found")
        self.port = _free_port()
        self.dir = tempfile.mkdtemp(prefix="bot18-cdp-")
        self.loop = asyncio.new_event_loop()
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            [CHROME, "--headless=new", f"--remote-debugging-port={self.port}", f"--user-data-dir={self.dir}",
             "--no-first-run", "--no-default-browser-check", "--disable-gpu", "--hide-scrollbars",
             "--mute-audio", "--autoplay-policy=no-user-gesture-required", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        end = time.time() + 20
        while time.time() < end:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/version", timeout=1).read()
                break
            except OSError:
                time.sleep(0.2)
        return self

    def page(self, w: int = 1600, h: int = 1000, mobile: bool = False) -> Page:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/json/new?about:blank", method="PUT")
        info = json.loads(urllib.request.urlopen(req, timeout=10).read())
        p = Page(self, info["webSocketDebuggerUrl"])
        p.size(w, h, mobile)
        return p

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        shutil.rmtree(self.dir, ignore_errors=True)
        self.loop.close()
        return False


if __name__ == "__main__":  # smoke: python -m tests.cdp URL OUT.png [W H]
    import sys
    url, out = sys.argv[1], sys.argv[2]
    w, h = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1600, 1000)
    with Chrome() as c:
        pg = c.page(w, h, mobile=w < 600)
        pg.goto(url)
        pg.shot(out)
        print(json.dumps({"errors": pg.errors, "out": out}), flush=True)
    os._exit(0)
