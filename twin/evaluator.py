"""Frozen evaluator for peg insertion in the practice env: stage rubric and outcome, read from simulator state only.

rules.md: success is read from simulator state by evaluator code, never from a model's opinion; the evaluator is frozen
before practice. Every episode record carries EVALUATOR_VERSION and `code_hash()` (sha256 of this file and
twin/monitors.py), so results scored by different evaluator code are never pooled.

Stage rubric (robocurve-astra-2026, docs/knowledge/llm-as-robot-policy.md), the highest level reached at any step:
  0 no_approach   nothing below holds
  1 contact       a finger or the hand touches the part
  2 lifted        both fingers hold the part, its tip is above the surface and it touches nothing else
  3 above_target  the tip is within 2.0 mm of the TRUE hole axis and at most 20 mm above the surface (or deeper)
  4 seated        twin/predicates.py `seated`: depth >= seat_depth - 1 mm, lateral <= clearance + 0.2 mm, tilt <= 3 deg
In-hand starts begin at stage 2 (the part is already held); levels 0-1 matter for pick starts, which come later.
Also recorded: `engaged` (tip > 2 mm into the hole and on its axis) and the deepest tip depth.

Outcome, decided after the actor finishes and the env holds the last command for SETTLE_S:
  success              seated, released (no finger touches the part) and at rest (part speed < 5 mm/s)
  seated_not_released  seated but still held
  not_seated           part still in the hand or resting off the seat
  dropped              part out of the hand, not engaged in the hole
  limit_reached        the frozen budget ran out first (reported apart from failures, validation.yaml hypotheses)
  stalled              the stuck monitor ended the episode
  abandoned            the actor reset or closed the session mid-episode
  error                NaN or a solver blow-up
A success counts (`counted`) only if no monitor raised an invalidating flag and the dt/2 rerun agrees
(validation.yaml G.exploit_flags); otherwise the outcome becomes success_flagged.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

EVALUATOR_VERSION = 1
STAGES = ("no_approach", "contact", "lifted", "above_target", "seated")
ABOVE_LATERAL_M = 0.002   # above_target: tip within 2 mm of the true hole axis ...
ABOVE_HEIGHT_M = 0.020    # ... and at most 20 mm above the surface
LIFT_CLEAR_M = 0.001      # lifted: tip at least 1 mm above the surface
REST_SPEED_M_S = 0.005    # at rest: part speed below 5 mm/s at the final check
SETTLE_S = 0.5            # the env holds the last command this long before scoring
FINGERS = ("gripper/left_finger", "gripper/right_finger")
HAND = "gripper/hand"


def code_hash() -> str:
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for f in ("evaluator.py", "monitors.py"):
        h.update((here / f).read_bytes())
    return h.hexdigest()[:16]


class Evaluator:
    """Reads stage and outcome from (m, d). Built once per episode, after the model is final."""

    def __init__(self, m, slug: str, board_body: str = "board"):
        import mujoco
        self.m = m
        self.slug = slug
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.part = bid(slug)
        self.board = bid(board_body)
        self.fingers = {bid(f) for f in FINGERS}
        self.hand = bid(HAND)
        self.max_stage = 0
        self.max_depth = -np.inf
        self.engaged_ever = False
        self.timeline: list[tuple[float, int]] = []

    def touches(self, d) -> dict:
        """Which bodies the part touches right now."""
        out = dict(fingers=set(), hand=False, board=False, other=False)
        if d.ncon == 0:
            return out
        g = d.contact.geom[: d.ncon]
        b = self.m.geom_bodyid[g]
        for b1, b2 in b:
            if self.part not in (b1, b2):
                continue
            o = b2 if b1 == self.part else b1
            if o in self.fingers:
                out["fingers"].add(int(o))
            elif o == self.hand:
                out["hand"] = True
            elif o == self.board:
                out["board"] = True
            else:
                out["other"] = True
        return out

    def state(self, d) -> dict:
        from twin.predicates import insertion_state
        st = insertion_state(self.m, d, self.slug)
        tc = self.touches(d)
        held = len(tc["fingers"]) == 2
        stage = 0
        if tc["fingers"] or tc["hand"]:
            stage = 1
        if held and st["depth"] < -LIFT_CLEAR_M and not (tc["board"] or tc["other"]):
            stage = 2
        if st["lateral"] <= ABOVE_LATERAL_M and st["depth"] >= -ABOVE_HEIGHT_M:
            stage = max(stage, 3)
        if st["seated"]:
            stage = 4
        st.update(stage=stage, held=held, touching_fingers=len(tc["fingers"]), touching_board=tc["board"])
        return st

    def update(self, d) -> dict:
        st = self.state(d)
        if st["stage"] > self.max_stage:
            self.max_stage = st["stage"]
            self.timeline.append((round(float(d.time), 4), st["stage"]))
        self.max_depth = max(self.max_depth, st["depth"])
        self.engaged_ever |= bool(st["engaged"])
        return st

    def part_speed(self, d) -> float:
        import mujoco
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, self.part, v, 0)
        return float(np.linalg.norm(v[3:]))

    def outcome(self, d, ended_by: str | None = None, nan: bool = False) -> dict:
        """Final outcome (call after the settle hold)."""
        st = self.update(d)
        speed = self.part_speed(d)
        released = st["touching_fingers"] == 0
        if nan:
            oc = "error"
        elif ended_by in ("limit_reached", "stalled", "abandoned"):
            oc = ended_by
        elif st["seated"] and released and speed < REST_SPEED_M_S:
            oc = "success"
        elif st["seated"] and not released:
            oc = "seated_not_released"
        elif not st["held"] and not st["engaged"] and not st["seated"]:
            oc = "dropped"
        else:
            oc = "not_seated"
        return dict(outcome=oc, max_stage=STAGES[self.max_stage], max_stage_index=self.max_stage,
                    stage_timeline=[(t, STAGES[s]) for t, s in self.timeline], engaged_ever=self.engaged_ever,
                    max_depth_mm=round(1000 * float(self.max_depth), 3),
                    final=dict(depth_mm=round(1000 * st["depth"], 3), lateral_mm=round(1000 * st["lateral"], 3),
                               tilt_deg=round(st["tilt_deg"], 3), seated=bool(st["seated"]), released=released,
                               part_speed_mm_s=round(1000 * speed, 3), seat_depth_mm=round(1000 * st["seat_depth"], 3)))


def counted(outcome: str, flags: list[dict], dt_half: dict | None) -> tuple[str, bool, list[str]]:
    """(final outcome, counted, why) after integrity: a success with an invalidating flag or a dt/2 flip is not counted."""
    why = [f"{f['monitor']}: {f['reason']}" for f in flags if f.get("invalidates")]
    if outcome != "success":
        return outcome, False, why
    if dt_half is not None and dt_half.get("outcome") != "success":
        why.append(f"numerics.dt_half: outcome at dt/2 is {dt_half.get('outcome')!r}")
    if why:
        return "success_flagged", False, why
    if dt_half is None:
        return "success_unconfirmed", False, ["numerics.dt_half: rerun not done"]
    return "success", True, []
