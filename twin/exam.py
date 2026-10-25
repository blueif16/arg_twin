"""The exam guard shared by every REASSEMBLE data loader and every scan across episodes.

playbook/splits.json marks the exam episodes; they are never read before the final score (rules.md). Loaders call
`check(ep)` and scans call `visible(eps)`. Only a call by the `exam-scorer` caller may read them:
harness.registry.call sets EXAM_OK for the duration of such a call. Everything else, direct Python use included, is
refused. SO-101 has no exam split (playbook/splits_so101.json), so its loaders don't use this module.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from functools import lru_cache

from twin import REPO_DIR

SPLITS = REPO_DIR / "playbook" / "splits.json"
EXAM_OK: ContextVar[bool] = ContextVar("twin_exam_ok", default=False)


@lru_cache(maxsize=1)
def _exam() -> tuple[frozenset, frozenset]:
    if not SPLITS.exists():
        return frozenset(), frozenset()
    eps = json.loads(SPLITS.read_text())["episodes"]
    return (frozenset(int(k) for k, v in eps.items() if v["split"] == "exam"),
            frozenset(v["recording"] for v in eps.values() if v["split"] == "exam"))


def exam_episodes() -> frozenset:
    return _exam()[0]


def exam_recordings() -> frozenset:
    return _exam()[1]


def is_exam(ep) -> bool:
    try:
        return int(ep) in _exam()[0]
    except (TypeError, ValueError):
        return str(ep) in _exam()[1]


def allowed(ep) -> bool:
    return EXAM_OK.get() or not is_exam(ep)


def check(ep, what: str = "") -> None:
    """Raise PermissionError if `ep` (episode index or recording id) is an exam episode and the caller is not
    exam-scorer."""
    if not allowed(ep):
        raise PermissionError(f"episode {ep} is in the exam split (playbook/splits.json): held out until the final "
                              f"score{'; ' + what if what else ''}")


def visible(episodes) -> list:
    """The episodes a scan may include: all of them for exam-scorer, no exam episode for anyone else."""
    return [e for e in episodes if allowed(e)]
