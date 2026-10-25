"""Client side of the LIBERO playground boundary (runs under the repo's .venv; never imports LIBERO).

SimClient starts one server process under .venv-libero and speaks JSON lines with it. The child gets no API keys
(every *KEY* / *TOKEN* / *SECRET* variable is removed) and single-threaded BLAS/OMP, since the CPU is shared.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LIBERO_PY = REPO / ".venv-libero" / "bin" / "python"
SERVER = HERE / "server.py"


def child_env() -> dict:
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")
    return env


class SimClient:
    def __init__(self, cfg: dict, log_path: Path | None = None):
        self.log = open(log_path, "a") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen([str(LIBERO_PY), "-u", str(SERVER), "stdio", "--config", json.dumps(cfg)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
                                     env=child_env(), cwd=str(REPO))
        self.lock = threading.Lock()
        ready = self._read()
        if not ready.get("ready"):
            raise RuntimeError(f"LIBERO server did not start: {ready}")
        self.evaluator = ready.get("evaluator")

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"LIBERO server exited (code {self.proc.poll()}); see its log")
        return json.loads(line)

    def call(self, op: str, **kw) -> dict:
        with self.lock:
            self.proc.stdin.write(json.dumps(dict(op=op, **kw)) + "\n")
            self.proc.stdin.flush()
            return self._read()

    def close(self):
        try:
            if self.proc.poll() is None:
                self.call("close")
                self.proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            self.proc.kill()
        if self.log is not subprocess.DEVNULL:
            self.log.close()


def call_socket(sock_path: str, req: dict, timeout: float = 900.0) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(str(sock_path))
    with s:
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode())
