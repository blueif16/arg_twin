"""Skill-sandbox bootstrap (copied next to the program and run first, inside the sandbox; never imported by twin).

It installs an audit hook (PEP 578), then runs the skill. Python code cannot remove an audit hook; only native code
could, and the native doors (ctypes, _ctypes) are among the blocked imports. The hook makes the boundary deterministic
even when the filesystem layer is weaker (process-only fallback):
  - every ctypes.* audit event (dlopen, dlsym, call_function, ...), which _ctypes raises in C, so raw _ctypes
    reached through importlib (which skips the import audit event for builtins) is still unusable
  - imports of ctypes/_ctypes, subprocess/_posixsubprocess, socket/_socket/ssl, multiprocessing, pty, mujoco,
    mujoco_warp, warp, twin, harness and the CPython test hooks raise ImportError (this Python links _ctypes,
    _posixsubprocess and _socket statically, so hiding files alone cannot remove them)
  - process creation and signalling (fork, exec, spawn, system, Popen, kill) raise PermissionError; the kernel
    backstop is RLIMIT_NPROC = 0 (prlimit), which also blocks threads
  - sockets raise PermissionError (the network namespace has no route anyway)
  - writes, renames and deletes outside the workspace raise PermissionError
"""
import os
import runpy
import sys

WS = os.path.realpath(os.getcwd())
BLOCKED = frozenset({"ctypes", "_ctypes", "subprocess", "_posixsubprocess", "socket", "_socket", "ssl", "_ssl",
                     "multiprocessing", "_multiprocessing", "_posixshmem", "pty", "mujoco", "mujoco_warp", "warp",
                     "twin", "harness", "_testcapi", "_testinternalcapi", "_xxsubinterpreters", "_interpreters"})
SPAWN = frozenset({"os.fork", "os.forkpty", "os.exec", "os.posix_spawn", "os.spawn", "os.system", "os.startfile",
                   "subprocess.Popen", "pty.spawn", "os.kill", "os.killpg", "os.setuid", "os.setgid"})
NET = frozenset({"socket.__new__", "socket.connect", "socket.bind", "socket.getaddrinfo", "socket.sendto",
                 "socket.sendmsg"})
PATH_EVENTS = frozenset({"os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.symlink",
                         "os.link", "os.truncate", "os.utime", "shutil.rmtree", "shutil.move", "shutil.copyfile"})
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def _inside(path) -> bool:
    try:
        p = os.path.realpath(os.fsdecode(path))
    except (TypeError, ValueError):
        return False
    return p == WS or p.startswith(WS + os.sep)


def _hook(event, args):
    if event == "import":
        name = args[0] or ""
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name}: not available in the skill sandbox")
    elif event.startswith("ctypes."):        # raised by the _ctypes builtin itself: dlopen, call_function, ...
        raise PermissionError(f"{event}: no FFI in the skill sandbox")
    elif event in SPAWN:
        raise PermissionError(f"{event}: skill programs cannot start or signal processes")
    elif event in NET:
        raise PermissionError(f"{event}: no network in the skill sandbox")
    elif event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                  (isinstance(flags, int) and bool(flags & WRITE_FLAGS))
        if writing and isinstance(path, (str, bytes, os.PathLike)) and not _inside(path):
            raise PermissionError(f"write outside the skill workspace: {path!r}")
    elif event in PATH_EVENTS:
        for a in args:
            if isinstance(a, (str, bytes, os.PathLike)) and not _inside(a):
                raise PermissionError(f"{event} outside the skill workspace: {a!r}")


if __name__ == "__main__":
    for name in list(sys.modules):          # nothing blocked may already be loaded
        if name.split(".")[0] in BLOCKED:
            del sys.modules[name]
    sys.addaudithook(_hook)
    entry = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path[0] = WS
    runpy.run_path(entry, run_name="__main__")
