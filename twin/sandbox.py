"""Skill-program boundary: a static checker for skill code and a sandboxed process to run it in.

Compiled skill programs (and red-team actor scripts) never share a process with the simulator. They run as a separate
Python process that talks to the env (twin/env.py) over JSON lines on stdin/stdout through `envapi.py`, a stdlib-only
client copied into the program's workspace. In-process Python restriction is not a security boundary; the process is.

Layers, outermost first:
1. Process: the program holds no reference to MjModel/MjData; it can only send the env's commands (move with speed and
   force ceilings enforced in env code, gripper, wait, observe, status, task, finish).
2. Sandbox (when Linux user namespaces work, which they do in this WSL box): `unshare -r -m -n -p -f --mount-proc`,
   no network, a fresh /proc, tmpfs over $HOME, /tmp, /mnt, /dev/shm and /run/user, then read-only binds of the Python
   install and .venv with a MINIMAL site-packages (numpy only: mujoco, mujoco_warp, warp and everything else are not
   there), a private copy of the workspace, RLIMIT_NPROC = 0 (no fork, exec-spawn or threads), and every capability
   dropped with setpriv (noroot securebits), so root-in-the-namespace cannot unmount the hiding tmpfs.
   The program starts under twin/skillboot.py, whose audit hook (not removable from Python) blocks ctypes/_ctypes,
   subprocess, sockets, multiprocessing, mujoco/twin/harness imports, process creation and writes outside the
   workspace. This Python links _ctypes, _posixsubprocess and _socket statically, which is why hiding files is not
   enough on its own. The repo (runs/, twin/, harness/, data/, playbook/,
   starts and evaluator files) is not visible. rlimits: CPU seconds, address space, file size, no core dumps.
   If namespaces fail, the program still runs as a separate process (`sandbox: process-only` in the result).
3. Static checker (`check_source`, AST): a policy filter, not a security boundary. Rejects
   - imports outside ALLOWED_IMPORTS; explicitly banned with reasons: mujoco, mujoco_warp, warp, harness, twin,
     ctypes, cffi, subprocess, socket, os, sys, pathlib, importlib, multiprocessing, threading, pickle, marshal, ...
   - eval, exec, compile, __import__, globals, locals, vars, breakpoint, input
   - getattr/setattr/delattr unless the attribute is a plain string literal (no dunders)
   - dunder attribute access (__class__, __subclasses__, __globals__, __builtins__, __dict__, ...) and string literals
     containing '__' (str.format can walk attributes: "{0.__class__}")
   - numpy file and FFI entry points (np.load, np.fromfile, np.save*, np.memmap, np.ctypeslib, np.lib, ...)
   - open() unless the path is a string literal relative to the workspace (no absolute path, no '..'); this covers
     writes outside the workspace and reads of evaluator files
"""

from __future__ import annotations

import ast
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from twin import REPO_DIR

ALLOWED_IMPORTS = {
    "math", "cmath", "json", "time", "random", "statistics", "itertools", "functools", "collections", "dataclasses",
    "typing", "enum", "copy", "heapq", "bisect", "fractions", "decimal", "numbers", "abc", "re", "numpy", "envapi",
    "__future__",
}  # not operator (attrgetter) or string (Formatter): both reach attributes by computed name
BANNED_IMPORTS = {
    "mujoco": "the simulator: skills act only through the env API",
    "mujoco_warp": "the simulator (GPU)", "warp": "GPU kernels / simulator backend",
    "harness": "harness internals (tools, registry, checks)", "twin": "twin internals (scene, evaluator, monitors)",
    "ctypes": "raw memory / FFI", "cffi": "FFI", "subprocess": "spawns processes", "socket": "network / IPC",
    "os": "files, environment, processes", "sys": "interpreter internals", "pathlib": "file system access",
    "shutil": "file system access", "glob": "file system access", "io": "file access", "importlib": "dynamic import",
    "builtins": "interpreter builtins", "inspect": "frame / object introspection", "gc": "object graph access",
    "pickle": "arbitrary code on load", "marshal": "code objects", "multiprocessing": "spawns processes",
    "threading": "concurrency outside the env protocol", "asyncio": "event loop / IPC", "signal": "process control",
    "resource": "process limits", "pty": "terminals", "fcntl": "file descriptors", "mmap": "raw memory",
    "code": "interpreter", "codeop": "interpreter", "runpy": "runs other code", "urllib": "network", "http": "network",
    "requests": "network", "scipy": "not on the allowlist (add with a reason if a skill needs it)",
    "operator": "attrgetter/methodcaller reach attributes by computed name",
    "string": "Formatter reaches attributes through format fields",
}
BANNED_CALLS = {"eval", "exec", "compile", "__import__", "globals", "locals", "vars", "breakpoint", "input",
                "memoryview", "exit", "quit", "help"}
NUMPY_BANNED_ATTRS = {"load", "save", "savez", "savez_compressed", "fromfile", "tofile", "loadtxt", "savetxt",
                      "genfromtxt", "memmap", "ctypeslib", "lib", "DataSource", "f2py", "distutils", "testing",
                      "fromregex", "frombuffer"}
DUNDER_OK = {"__init__", "__name__", "__doc__", "__len__", "__iter__", "__next__", "__enter__", "__exit__", "__eq__",
             "__hash__", "__repr__", "__str__", "__lt__", "__le__", "__gt__", "__ge__", "__post_init__",
             "__future__", "__call__", "__contains__", "__getitem__", "__setitem__", "__add__", "__sub__", "__mul__"}
MAX_SOURCE_BYTES = 200_000


def _root(mod: str | None) -> str:
    return (mod or "").split(".")[0]


def check_source(src: str, filename: str = "<skill>") -> dict:
    """Static policy check of one skill source file. Returns {ok, violations: [{line, rule, detail}], imports}."""
    v: list[dict] = []
    imports: list[str] = []
    if len(src.encode()) > MAX_SOURCE_BYTES:
        return dict(ok=False, violations=[dict(line=0, rule="size", detail=f"source > {MAX_SOURCE_BYTES} bytes")],
                    imports=[], sha256=hashlib.sha256(src.encode()).hexdigest())
    try:
        tree = ast.parse(src, filename)
    except SyntaxError as e:
        return dict(ok=False, violations=[dict(line=e.lineno or 0, rule="syntax", detail=str(e))], imports=[],
                    sha256=hashlib.sha256(src.encode()).hexdigest())
    numpy_alias = set()

    def bad(node, rule, detail):
        v.append(dict(line=getattr(node, "lineno", 0), rule=rule, detail=detail))

    def check_mod(node, mod: str, names=()):
        r = _root(mod)
        imports.append(mod)
        if r in BANNED_IMPORTS:
            bad(node, "banned_import", f"{mod}: {BANNED_IMPORTS[r]}")
        elif r not in ALLOWED_IMPORTS:
            bad(node, "import_not_allowed", f"{mod} is not on the allowlist {sorted(ALLOWED_IMPORTS)}")
        if r == "numpy":
            parts = mod.split(".")[1:] + list(names)
            for p in parts:
                if p in NUMPY_BANNED_ATTRS:
                    bad(node, "numpy_io_ffi", f"numpy.{p}: file I/O or FFI")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                check_mod(node, a.name)
                if _root(a.name) == "numpy":
                    numpy_alias.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                bad(node, "relative_import", "relative imports are not allowed")
                continue
            check_mod(node, node.module or "", [a.name for a in node.names])
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else None
            if name in BANNED_CALLS:
                bad(node, "banned_call", f"{name}()")
            if name in ("getattr", "setattr", "delattr", "hasattr"):
                arg = node.args[1] if len(node.args) > 1 else None
                if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                        and not arg.value.startswith("__")):
                    bad(node, "dynamic_attr", f"{name}() needs a plain string literal attribute name")
            if name == "open":
                p = node.args[0] if node.args else None
                ok = isinstance(p, ast.Constant) and isinstance(p.value, str) and not os.path.isabs(p.value) \
                    and ".." not in Path(p.value).parts and p.value.strip() != ""
                if not ok:
                    bad(node, "file_path", "open() needs a string-literal path inside the workspace (relative, no '..')")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in DUNDER_OK:
                bad(node, "dunder_attr", f".{node.attr}")
            if isinstance(node.value, ast.Name) and node.value.id in numpy_alias and node.attr in NUMPY_BANNED_ATTRS:
                bad(node, "numpy_io_ffi", f"{node.value.id}.{node.attr}: file I/O or FFI")
        elif isinstance(node, ast.Name):
            if node.id in ("__builtins__", "__loader__", "__spec__", "__file__"):
                bad(node, "dunder_name", node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "__" in node.value and node.value != "__main__":
                bad(node, "dunder_string", "string literal with '__' (format-string attribute access)")
    return dict(ok=not v, violations=v, imports=sorted(set(imports)), sha256=hashlib.sha256(src.encode()).hexdigest())


def check_dir(path) -> dict:
    """Check every .py file of a skill workspace (or one file). The entry point is skill.py."""
    p = Path(path)
    files = [p] if p.is_file() else sorted(q for q in p.rglob("*.py") if q.name != "envapi.py")
    out = {}
    for f in files:
        out[str(f.relative_to(p.parent if p.is_file() else p))] = check_source(f.read_text(), str(f))
    h = hashlib.sha256()
    for k in sorted(out):
        h.update(k.encode() + out[k]["sha256"].encode())
    return dict(ok=all(r["ok"] for r in out.values()), files=out, sha256=h.hexdigest()[:16])


# --------------------------------------------------------------------------------------------- the client
ENVAPI_SOURCE = r'''"""envapi: the only door from a skill program to the practice env (JSON lines over stdin/stdout).

Units SI, world = FR3 base frame, quaternions w-first. Poses are of the TCP site (the logged O_T_EE, 22.9 mm beyond
the finger-pad centre along the hand z axis). Forces: F on the tool in the base frame (REASSEMBLE force.base sign).
Use print() freely: it goes to stderr; stdout is the protocol channel.
"""
import json as _json
import sys as _sys

_proto_out = _sys.stdout
_sys.stdout = _sys.stderr
_proto_in = _sys.stdin


class EnvError(RuntimeError):
    pass


def _call(op, **kw):
    req = dict(op=op, **{k: v for k, v in kw.items() if v is not None})
    _proto_out.write(_json.dumps(req) + "\n")
    _proto_out.flush()
    line = _proto_in.readline()
    if not line:
        raise SystemExit("env closed the channel")
    rep = _json.loads(line)
    if "error" in rep:
        raise EnvError(rep["error"])
    return rep


def task():
    """Part, nominal hole entry pose, nominal grasp, limits, budget."""
    return _call("task")


def observe():
    return _call("observe")


def status():
    return _call("status")


def move(pos, quat=None, speed=None, force_max=None, stop_force=None, timeout=None):
    """Absolute TCP target. speed m/s (cap enforced by the env), force_max N (spring force ceiling), stop_force N
    (stop the move when |F| reaches it: a guarded move)."""
    return _call("move", pos=[float(x) for x in pos], quat=None if quat is None else [float(x) for x in quat],
                 speed=speed, force_max=force_max, stop_force=stop_force, timeout=timeout)


def gripper(action, width=None):
    """action: open | close | width (width = finger opening in m)."""
    return _call("gripper", action=action, width=width)


def wait(seconds):
    return _call("wait", seconds=float(seconds))


def finish(claim=None):
    """End the episode. claim: 'success' | 'failure' | None (recorded next to the evaluator's verdict)."""
    return _call("finish", claim=claim)
'''


# --------------------------------------------------------------------------------------------- the sandbox
def _python_home() -> Path:
    return Path(os.path.realpath(sys.executable)).parent.parent  # .../cpython-3.12.x/bin/python3.12 -> install root


def _venv() -> Path:
    return Path(sys.prefix)


def _site() -> Path:
    import sysconfig
    return Path(sysconfig.get_paths()["purelib"])  # .venv/lib/python3.12/site-packages


SANDBOX_SH = r'''set -e
S="$1"; WS="$2"; PYHOME="$3"; VENV="$4"; HOMED="$5"; ENTRY="$6"; BOOT="$7"; SITE="$8"
mount -t tmpfs -o size=64m tmpfs "$S"
mkdir -p "$S/py" "$S/venv" "$S/ws" "$S/boot" "$S/np" "$S/nplibs"
mount --bind "$PYHOME" "$S/py"; mount -o remount,bind,ro "$S/py"
mount --bind "$VENV" "$S/venv"; mount -o remount,bind,ro "$S/venv"
mount --bind "$SITE/numpy" "$S/np"; mount -o remount,bind,ro "$S/np"
HAVE_NPLIBS=0
if [ -d "$SITE/numpy.libs" ]; then mount --bind "$SITE/numpy.libs" "$S/nplibs"; mount -o remount,bind,ro "$S/nplibs"; HAVE_NPLIBS=1; fi
mount --bind "$BOOT" "$S/boot"; mount -o remount,bind,ro "$S/boot"
mount --bind "$WS" "$S/ws"
mount -t tmpfs -o size=64m tmpfs "$HOMED"
mkdir -p "$PYHOME" "$VENV" "$HOMED/ws" "$HOMED/boot"
mount --move "$S/py" "$PYHOME"
mount --move "$S/venv" "$VENV"
# minimal site-packages: numpy only; mujoco, mujoco_warp, warp and every other installed package are absent
mount -t tmpfs -o size=1m tmpfs "$SITE"
mkdir -p "$SITE/numpy" "$SITE/numpy.libs"
mount --move "$S/np" "$SITE/numpy"
if [ "$HAVE_NPLIBS" = 1 ]; then mount --move "$S/nplibs" "$SITE/numpy.libs"; fi
mount -o remount,bind,ro "$SITE"
mount --move "$S/boot" "$HOMED/boot"
mount --move "$S/ws" "$HOMED/ws"
for d in /mnt /dev/shm /run/user /var/tmp; do [ -d "$d" ] && mount -t tmpfs -o size=8m tmpfs "$d" 2>/dev/null || true; done
mount -t tmpfs -o size=64m tmpfs /tmp
cd "$HOMED/ws"
# drop every capability (root in this user namespace could otherwise unmount the hiding tmpfs and see the repo),
# forbid regaining them, and allow no processes or threads
exec env -i PATH=/usr/bin:/bin HOME="$HOMED/ws" LANG=C.UTF-8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 prlimit --nproc=0:0 -- setpriv --no-new-privs \
  --securebits +noroot,+noroot_locked,+no_setuid_fixup,+no_setuid_fixup_locked --inh-caps -all --bounding-set -all \
  -- "$VENV/bin/python" -E -s -B -u "$HOMED/boot/skillboot.py" "$ENTRY"
'''


def _limits(cpu_s: int, mem_bytes: int, fsize: int):
    def fn():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return fn


_NS_OK: bool | None = None


def namespaces_available() -> bool:
    global _NS_OK
    if _NS_OK is None:
        try:
            r = subprocess.run(["unshare", "-r", "-m", "-n", "-p", "-f", "--mount-proc", "--", "/bin/true"],
                               capture_output=True, timeout=10)
            _NS_OK = r.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            _NS_OK = False
    return _NS_OK


def prepare(skill_path, entry: str = "skill.py") -> tuple[Path, str]:
    """Private copy of the skill workspace (outside the repo) with envapi.py dropped in. Returns (dir, entry)."""
    src = Path(skill_path)
    tmp = Path(tempfile.mkdtemp(prefix="skillbox-"))
    ws = tmp / "ws"
    if src.is_file():
        ws.mkdir()
        shutil.copy2(src, ws / src.name)
        entry = src.name
    else:
        shutil.copytree(src, ws, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "envapi.py"))
    (ws / "envapi.py").write_text(ENVAPI_SOURCE)
    (tmp / "stage").mkdir()
    (tmp / "boot").mkdir()
    shutil.copy2(Path(__file__).with_name("skillboot.py"), tmp / "boot" / "skillboot.py")
    return tmp, entry


def popen(tmp: Path, entry: str, cpu_s: int = 600, mem_gb: float = 4.0, fsize_mb: int = 64,
          stderr_path: Path | None = None) -> tuple[subprocess.Popen, str]:
    """Start the program in the sandbox (or as a plain separate process if namespaces fail)."""
    ws = tmp / "ws"
    if stderr_path:
        Path(stderr_path).parent.mkdir(parents=True, exist_ok=True)
    err = open(stderr_path, "wb") if stderr_path else subprocess.DEVNULL
    pre = _limits(cpu_s, int(mem_gb * 2 ** 30), fsize_mb * 2 ** 20)
    if namespaces_available():
        argv = ["unshare", "-r", "-m", "-n", "-p", "-f", "--mount-proc", "--kill-child", "--", "/bin/sh", "-c",
                SANDBOX_SH, "sandbox", str(tmp / "stage"), str(ws), str(_python_home()), str(_venv()),
                str(Path.home()), entry, str(tmp / "boot"), str(_site())]
        mode = "namespaces"
    else:
        argv = [str(_venv() / "bin" / "python"), "-E", "-s", "-B", "-u", str(tmp / "boot" / "skillboot.py"), entry]
        mode = "process-only"
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, cwd=str(ws),
                         env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(ws)} if mode == "process-only"
                         else {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                         preexec_fn=pre, start_new_session=True, bufsize=0)
    if stderr_path:
        err.close()
    return p, mode


def cleanup(tmp: Path):
    shutil.rmtree(tmp, ignore_errors=True)


def repo_visible_paths() -> list[str]:
    """Paths a sandboxed program must not see (used by the boundary self-test)."""
    return [str(REPO_DIR / d) for d in ("runs", "twin", "harness", "playbook", "data")]
