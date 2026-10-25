"""LIBERO playground server: the simulator behind a narrow JSON command API (runs under .venv-libero).

Same boundary as twin/env.py, for LIBERO. One process owns one simulator (sim.LiberoSim), the frozen success
predicate (LIBERO's own goal predicates, hash-checked against frozen.json at start), the per-episode budget and the
runtime monitors. Actors never touch it; they send commands and get observations back:

  ACTOR ops (what a brain or a compiled program may send)
    task                       instruction, frame and action conventions, bounds, per-step limits, budget
    observe                    the current observation: eef_state (10 labelled values), gripper_qpos, and the
                               agentview + wrist PNGs (written to the episode's frames/ dir, returned as paths)
    move_to {targets, note}    absolute EEF target for any subset of x y z r00 r10 r20 r01 r11 r21 gripper;
                               interpolated in a straight line at the declared per-step limit (0.01 m, 0.05 rot6d,
                               0.1 gripper per 20 Hz step), at most 10 s, then each step passes the clamp and
                               delta-limit approvers and runs; the success predicate is checked after every step
    done {summary, hindsight}  the actor claims the task is finished; the episode ends, no action runs
    give_up {reason, hindsight}
    status                     budget used / left
  HARNESS ops (the brain runner or the experiment driver only)
    reset {task_id, init_state, split}   start an episode; the split must list it (splits.json); a scored episode
                               is attempted once per (run, actor)
    begin_call {reserve_usd}   ask before each model call: refuses (and ends the episode as limit_reached) when the
                               call cap is used up or the reservation would pass the spend cap
    charge {usage, usd}        account one model call's tokens and cost
    end_limit {what, used, limit, note}   end the episode as limit_reached (e.g. no tool call 3 turns in a row)
    ping / close

What an actor never gets: object poses, sim state, the predicate or its value. On scored and feasibility splits the
outcome is not reported back to the actor at all (reveal_outcome is a practice-split setting).

Transport: JSON lines on stdin/stdout (`stdio`), or one JSON request per connection on a Unix socket (`serve`).

    .venv-libero/bin/python playgrounds/libero/server.py stdio --config '<json>'
    .venv-libero/bin/python playgrounds/libero/server.py serve --socket <path> --config '<json>'
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import sim as S  # noqa: E402

REPO = S.REPO
SPLITS = HERE / "splits.json"
FROZEN = HERE / "frozen.json"
SERVER_VERSION = "libero-playground-1"
MAX_PLAYOUT_S = 10.0                       # upstream _tools.py _MAX_DURATION_S
MAX_PLAYOUT_STEPS = math.ceil(MAX_PLAYOUT_S * S.CONTROL_HZ)
RELATIVE_HEADROOM = 1e-6
DEFAULT_BUDGET = dict(max_llm_calls=80, max_steps=900, max_usd=30.0, max_output_tokens_per_call=16000)
ACTOR_OPS = ("task", "observe", "move_to", "done", "give_up", "status")
HARNESS_OPS = ACTOR_OPS + ("reset", "begin_call", "charge", "end_limit", "ping", "close")
LIMIT_WHAT = ("llm_calls", "usd", "output_tokens", "no_tool_call", "tool_errors", "wall_s", "api_error", "other")
# plausibility monitor thresholds (flags; only non-finite physics invalidates)
PEN_FLAG_M = 0.02
SPEED_FLAG_M_S = 5.0


class CmdError(ValueError):
    pass


def _r(a, k=4):
    return [round(float(v), k) for v in np.asarray(a).ravel()]


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def load_splits() -> dict:
    return json.loads(SPLITS.read_text())


def split_members(sp: dict) -> set[tuple[int, int]]:
    out = set()
    for e in sp["episodes"]:
        if "init_states" in e:
            out |= {(int(e["task_id"]), int(i)) for i in e["init_states"]}
        else:
            out.add((int(e["task_id"]), int(e["init_state"])))
    return out


def state_text_labels(eef: np.ndarray) -> str:
    """upstream policy._state_lines: values rounded to 4 decimals, labelled with the action dims."""
    vals = np.round(np.asarray(eef, np.float64), 4).tolist()
    return " ".join(f"{k}={v}" for k, v in zip(S.DIM_LABELS, vals))


class Playground:
    def __init__(self, run_id: str, actor: str = "argon", brain: str = "argon", split: str = "feasibility",
                 budget: dict | None = None, out_root: str | None = None, reveal_outcome: bool | None = None,
                 video: bool = True, replay_check: str = "success", frozen_check: bool = True,
                 views: str = "true"):
        self.run_id, self.actor, self.brain = run_id, actor, brain
        self.views = views
        self.splits = load_splits()
        if split not in self.splits["splits"]:
            raise CmdError(f"unknown split {split!r}; splits.json has {sorted(self.splits['splits'])}")
        self.split = split
        self.split_def = self.splits["splits"][split]
        self.members = split_members(self.split_def)
        self.budget = {**DEFAULT_BUDGET, **(budget or {})}
        self.out = Path(out_root) if out_root else REPO / "runs" / "libero" / run_id
        self.reveal = bool(self.split_def.get("reveal_outcome", False)) if reveal_outcome is None else reveal_outcome
        if self.split_def.get("kind") in ("scored", "feasibility") and self.reveal:
            raise PermissionError(f"outcomes are never revealed to actors on the {split} split")
        self.video = video
        self.replay_check = replay_check
        self.evaluator = S.evaluator_hash()
        if frozen_check:
            fz = json.loads(FROZEN.read_text())
            if fz["evaluator"]["sha256"] != self.evaluator["sha256"]:
                raise RuntimeError(f"success predicate changed since it was frozen ({FROZEN.name}): "
                                   f"{fz['evaluator']['sha256'][:12]} != {self.evaluator['sha256'][:12]}")
        self.sim = S.LiberoSim(self.split_def.get("suite", S.SUITE), views=views)
        self.ep = None
        self.n_episodes = 0

    # ------------------------------------------------------------------ dispatch
    def command(self, req: dict, allowed=ACTOR_OPS) -> dict:
        try:
            if not isinstance(req, dict):
                raise CmdError("request must be a JSON object")
            op = req.get("op")
            if op not in allowed:
                raise CmdError(f"op {op!r} not allowed here; allowed: {list(allowed)}")
            args = {k: v for k, v in req.items() if k != "op"}
            fn = {"task": self.task, "observe": self.observe_op, "move_to": self.move_to, "done": self.done,
                  "give_up": self.give_up, "status": self.status, "reset": self.reset,
                  "begin_call": self.begin_call, "charge": self.charge, "end_limit": self.end_limit,
                  "ping": lambda: dict(pong=True, version=SERVER_VERSION), "close": self.close}[op]
            return fn(**args)
        except TypeError as ex:
            return dict(error=f"bad arguments for {req.get('op')!r}: {ex}")
        except (CmdError, PermissionError, KeyError) as ex:
            return dict(error=f"{type(ex).__name__}: {ex}")
        except Exception as ex:  # noqa: BLE001  a bug must not kill the session
            return dict(error=f"{type(ex).__name__}: {ex}", trace=traceback.format_exc(limit=3))

    def _need_ep(self):
        if self.ep is None:
            raise CmdError("no episode: reset first")
        if self.ep["done"]:
            raise CmdError("the episode is over")

    # ------------------------------------------------------------------ actor ops
    def task(self) -> dict:
        e = self.ep
        out = dict(conventions=dict(
            frame="world frame, metres; z up", action_dims=list(S.DIM_LABELS),
            rot6d="r00 r10 r20 = first column, r01 r11 r21 = second column of the EEF rotation matrix; the "
                  "server orthonormalises them", gripper="0 open, 1 close; only the side of 0.5 matters",
            control_hz=S.CONTROL_HZ, bounds=dict(low=S.LOW.tolist(), high=S.HIGH.tolist()),
            max_step_per_control_step=S.MAX_STEP.tolist(), max_playout_s=MAX_PLAYOUT_S,
            cameras=dict(agentview="fixed third-person, 256x256, upright", wrist="eye-in-hand, 256x256")),
            budget=self.budget, split=self.split, suite=self.sim.suite_name)
        if e is not None:
            out.update(instruction=e["instruction"], embodiment=f"libero_10_task{e['task_id']:02d}")
        return out

    def observe_op(self) -> dict:
        if self.ep is None:
            raise CmdError("no episode: reset first")
        return self._obs(self.ep["last_pack"])

    def _obs(self, pack: dict) -> dict:
        """The observation an actor may see. Images are written once per decision point."""
        e = self.ep
        step = int(pack["steps"])
        paths = {}
        for cam in ("agentview", "wrist"):
            p = e["frames"] / f"s{step:04d}_{cam}.png"
            if not p.exists():
                from PIL import Image
                Image.fromarray(pack["images"][cam]).save(p)
            paths[cam] = _rel(p)
        appr = e["approvals_pending"]
        line = None
        if appr:
            details = [a for a in appr if a]
            if details:
                from collections import Counter
                c = Counter(details)
                line = (f"approver: {len(appr)} step(s) modified ("
                        + ", ".join(f"{k} ×{n}" if n > 1 else k for k, n in c.items()) + ").")
            else:
                line = f"approver: {len(appr)} step(s) modified."
        return dict(step=step, t=round(step / S.CONTROL_HZ, 3), eef_state=_r(pack["eef_state"], 6),
                    eef_state_text=state_text_labels(pack["eef_state"]),
                    gripper_qpos=_r(pack["gripper_qpos"], 6), images=paths, approvals=line,
                    instruction=e["instruction"], episode_over=bool(e["done"]))

    def move_to(self, targets=None, note=None) -> dict:
        """upstream inspect_robots_agent/_tools.py Toolset._move_absolute, then rollout + approvers per step."""
        self._need_ep()
        e = self.ep
        cur = np.asarray(e["last_pack"]["eef_state"], np.float64)
        if not np.all(np.isfinite(cur)):
            raise CmdError("proprioceptive reference contains a non-finite value")
        if not isinstance(note, str) or not note.strip():
            return self._tool_error("note is required: describe what you observe and why you chose this motion")
        if not isinstance(targets, dict) or not targets:
            return self._tool_error("targets must be a non-empty object of name: value")
        tgt = cur.copy()
        named = []
        for label, raw in targets.items():
            if label not in S.DIM_LABELS:
                return self._tool_error(f"unknown dimension {label!r}; valid names: {', '.join(S.DIM_LABELS)}")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                return self._tool_error(f"value for {label!r} must be a finite number, got {raw!r}")
            v = float(raw)
            if not math.isfinite(v):
                return self._tool_error(f"value for {label!r} must be a finite number, got {raw!r}")
            i = S.DIM_LABELS.index(label)
            if v < S.LOW[i] or v > S.HIGH[i]:
                return self._tool_error(f"target for {label} is outside [{float(S.LOW[i])!r}, {float(S.HIGH[i])!r}]")
            tgt[i] = v
            named.append(i)
        ratio = max([abs(tgt[i] - cur[i]) / S.MAX_STEP[i] for i in named if abs(tgt[i] - cur[i]) > 0], default=0.0)
        headed = ratio / (1.0 - RELATIVE_HEADROOM)
        if headed > MAX_PLAYOUT_STEPS:
            return self._tool_error(f"requested motion exceeds the {MAX_PLAYOUT_S:g}s playout cap; split the move "
                                    "into smaller motions")
        n = max(1, math.ceil(headed))
        fr = np.linspace(1.0 / n, 1.0, n)
        acts = [np.clip(cur + (tgt - cur) * f, S.LOW, S.HIGH) for f in fr]
        acts[-1] = np.clip(tgt.copy(), S.LOW, S.HIGH)
        e["tool_errors_in_row"] = 0
        note_txt = f"executing move_to over {n} steps ({n / S.CONTROL_HZ:.1f}s)"
        cmd = dict(i=len(e["commands"]), step0=self.sim.steps, op="move_to", targets=targets, note=note,
                   n_planned=n, call=e["calls"])
        played = 0
        e["approvals_pending"] = []
        for a in acts:
            a2, flags = self._approve(a)
            if flags:
                e["approvals_pending"].append(", ".join(flags))
            try:
                pack = self.sim.step(a2)
            except Exception as ex:  # noqa: BLE001  an embodiment fault ends the trial (upstream EmbodimentFault)
                cmd.update(n_played=played, fault=f"{type(ex).__name__}: {ex}")
                e["commands"].append(cmd)
                self._end("error", detail=f"embodiment fault: {type(ex).__name__}: {ex}")
                return dict(result=note_txt, error=f"EmbodimentFault: {type(ex).__name__}: {ex}",
                            episode_over=True)
            played += 1
            e["actions"].append(a2.tolist())
            self._after_step(pack)
            if e["done"]:
                break
        cmd.update(n_played=played, end_step=self.sim.steps,
                   n_modified=len(e["approvals_pending"]), reached_err_m=round(float(np.linalg.norm(
                       np.asarray(e["last_pack"]["eef_state"][:3]) - tgt[:3])), 4))
        e["commands"].append(cmd)
        out = dict(result=note_txt, steps_played=played, episode_over=bool(e["done"]))
        if not e["done"]:
            out["obs"] = self._obs(e["last_pack"])
        return out

    def _tool_error(self, msg: str) -> dict:
        e = self.ep
        e["tool_errors_in_row"] += 1
        e["tool_errors"] += 1
        return dict(tool_error=msg, episode_over=False)

    def _approve(self, a: np.ndarray):
        """ClampApprover (bounds) then DeltaLimitApprover (|a - last approved| <= max_step; first action passes)."""
        e = self.ep
        flags = []
        c = np.clip(a, S.LOW, S.HIGH)
        if not np.array_equal(c, a):
            flags.append("clamped")
        last = e["last_approved"]
        if last is not None:
            d = np.clip(c, last - S.MAX_STEP, last + S.MAX_STEP)
            if not np.array_equal(d, c):
                flags.append("delta_clamped")
            c = d
        e["last_approved"] = c.copy()
        return c, flags

    def _after_step(self, pack: dict):
        e = self.ep
        e["last_pack"] = pack
        succ = self.sim.success()
        e["success_trace"].append(bool(succ))
        e["states"].append(self.sim.record_state())
        h = self.sim.physics_health()
        mon = e["monitor"]
        mon["max_penetration_m"] = max(mon["max_penetration_m"], h["max_penetration_m"])
        mon["max_free_body_speed_m_s"] = max(mon["max_free_body_speed_m_s"], h["max_free_body_speed_m_s"])
        mon["bad_qacc"] = max(mon["bad_qacc"], h["bad_qacc"])
        if not h["finite"]:
            mon["nonfinite_step"] = mon.get("nonfinite_step", self.sim.steps)
        if e["video"] is not None:
            e["video"].append_data(np.concatenate([pack["images"]["agentview"], pack["images"]["wrist"]], axis=1))
        if succ:
            self._end("success")
        elif self.sim.steps >= int(self.budget["max_steps"]):
            self._end("step_limit")

    def done(self, summary=None, hindsight=None) -> dict:
        return self._stop("done", summary, hindsight)

    def give_up(self, reason=None, hindsight=None) -> dict:
        return self._stop("give_up", reason, hindsight)

    def _stop(self, kind, detail, hindsight) -> dict:
        self._need_ep()
        e = self.ep
        e["claim"] = dict(kind=kind, detail=str(detail or "")[:4000], hindsight=str(hindsight or "")[:8000])
        e["commands"].append(dict(i=len(e["commands"]), step0=self.sim.steps, op=kind, call=e["calls"]))
        rec = self._end(kind)
        return dict(result=f"{kind}: {e['claim']['detail']}", episode_over=True, report=self._actor_report(rec))

    def status(self) -> dict:
        out = dict(session=self.run_id, split=self.split, episodes=self.n_episodes, budget=self.budget)
        e = self.ep
        if e is not None:
            out.update(task_id=e["task_id"], init_state=e["init_state"], steps_used=self.sim.steps,
                       llm_calls_used=e["calls"], usd_used=round(e["usd"], 6), tokens_used=e["tokens"],
                       episode_over=e["done"])
            if e["done"] and self.reveal:
                out["ended_by"] = e["ended_by"]
        return out

    # ------------------------------------------------------------------ harness ops
    def reset(self, task_id: int, init_state: int, split: str | None = None) -> dict:
        if split is not None and split != self.split:
            raise PermissionError(f"this session serves the {self.split} split, not {split}")
        key = (int(task_id), int(init_state))
        if key not in self.members:
            raise PermissionError(f"(task {key[0]}, init {key[1]}) is not in the {self.split} split")
        if self.ep is not None and not self.ep["done"]:
            self._end("abandoned")
        self._claim_attempt(key)
        t0 = time.perf_counter()
        pack = self.sim.reset(*key)
        self.n_episodes += 1
        name = f"{self.n_episodes:03d}_t{key[0]:02d}_i{key[1]:02d}"
        d = self.out / "episodes" / name
        (d / "frames").mkdir(parents=True, exist_ok=True)
        video = None
        if self.video:
            import imageio.v2 as imageio
            video = imageio.get_writer(str(d / "rollout.mp4"), fps=S.CONTROL_HZ, macro_block_size=1)
            video.append_data(np.concatenate([pack["images"]["agentview"], pack["images"]["wrist"]], axis=1))
        self.ep = dict(task_id=key[0], init_state=key[1], name=name, dir=d, frames=d / "frames",
                       instruction=self.sim.instruction, last_pack=pack, last_approved=None, approvals_pending=[],
                       commands=[], actions=[], states=[self.sim.record_state()], success_trace=[],
                       calls=0, tokens=dict(input=0, cached=0, output=0, thought=0, total=0), usd=0.0,
                       tool_errors=0, tool_errors_in_row=0, claim=None, done=False, ended_by=None, limit=None,
                       wall0=time.time(), reset_s=round(time.perf_counter() - t0, 2), video=video,
                       monitor=dict(max_penetration_m=0.0, max_free_body_speed_m_s=0.0, bad_qacc=0),
                       success_at_reset=self.sim.success())
        return dict(task=self.task(), obs=self._obs(pack))

    def _claim_attempt(self, key):
        """A scored episode is attempted once per (run, actor); every attempt is logged."""
        f = self.out / "attempts.jsonl"
        f.parent.mkdir(parents=True, exist_ok=True)
        if self.split_def.get("kind") == "scored" and f.exists():
            for ln in f.read_text().splitlines():
                r = json.loads(ln)
                if (r["split"], r["task_id"], r["init_state"], r["actor"]) == (self.split, *key, self.actor):
                    raise PermissionError(f"scored episode (task {key[0]}, init {key[1]}) was already attempted by "
                                          f"{self.actor} in run {self.run_id}")
        with open(f, "a") as fh:
            fh.write(json.dumps(dict(split=self.split, task_id=key[0], init_state=key[1], actor=self.actor,
                                     t=time.time())) + "\n")

    def begin_call(self, reserve_usd: float = 0.0) -> dict:
        self._need_ep()
        e = self.ep
        b = self.budget
        if e["calls"] >= int(b["max_llm_calls"]):
            return self.end_limit("llm_calls", e["calls"], b["max_llm_calls"], "LLM call budget exhausted")
        if e["usd"] + float(reserve_usd) > float(b["max_usd"]):
            return self.end_limit("usd", round(e["usd"], 4), b["max_usd"],
                                  f"next call (reserve ${float(reserve_usd):.3f}) would pass the spend cap")
        return dict(ok=True, calls_used=e["calls"], usd_used=round(e["usd"], 6))

    def charge(self, usage: dict | None = None, usd: float = 0.0) -> dict:
        self._need_ep()
        e = self.ep
        e["calls"] += 1
        for k in e["tokens"]:
            e["tokens"][k] += int((usage or {}).get(k, 0) or 0)
        e["usd"] += float(usd or 0.0)
        return dict(calls_used=e["calls"], usd_used=round(e["usd"], 6), tokens_used=e["tokens"])

    def end_limit(self, what: str, used=None, limit=None, note: str = "") -> dict:
        self._need_ep()
        if what not in LIMIT_WHAT:
            raise CmdError(f"what must be one of {list(LIMIT_WHAT)}")
        self.ep["limit"] = dict(what=what, used=used, limit=limit, note=str(note)[:300])
        self._end("limit_reached", detail=f"{what}: {used} of {limit}")
        return dict(episode_over=True, ended_by="limit_reached", limit=self.ep["limit"])

    def close(self) -> dict:
        if self.ep is not None and not self.ep["done"]:
            self._end("abandoned")
        self.sim.close()
        return dict(closed=True, episodes=self.n_episodes)

    # ------------------------------------------------------------------ episode end: the frozen evaluator
    def _end(self, ended_by: str, detail: str | None = None) -> dict:
        e = self.ep
        if e["done"]:
            return e.get("record", {})
        e["done"], e["ended_by"] = True, ended_by
        if e["video"] is not None:
            e["video"].close()
            e["video"] = None
        success = ended_by == "success"
        # the evaluator re-reads the predicate on the final state; it must agree with the per-step trace
        final_pred = self.sim.success() if self.sim.env is not None else None
        flags = []
        mon = e["monitor"]
        if "nonfinite_step" in mon:
            flags.append(dict(monitor="plausibility.nonfinite", invalidates=True,
                              reason=f"non-finite physics at step {mon['nonfinite_step']}"))
        if mon["bad_qacc"]:
            flags.append(dict(monitor="plausibility.bad_qacc", invalidates=True, reason="MuJoCo BADQACC warning"))
        if mon["max_penetration_m"] > PEN_FLAG_M:
            flags.append(dict(monitor="plausibility.penetration", invalidates=False,
                              reason=f"max contact penetration {1000 * mon['max_penetration_m']:.1f} mm"))
        if mon["max_free_body_speed_m_s"] > SPEED_FLAG_M_S:
            flags.append(dict(monitor="plausibility.object_speed", invalidates=False,
                              reason=f"a free body reached {mon['max_free_body_speed_m_s']:.2f} m/s"))
        if e["success_at_reset"]:
            flags.append(dict(monitor="integrity.success_at_reset", invalidates=True,
                              reason="the predicate already held at reset"))
        replay = None
        if success and self.replay_check in ("success", "all") or self.replay_check == "all":
            replay = self._replay(e)
            if replay.get("ok") is False:
                flags.append(dict(monitor="integrity.replay", invalidates=True,
                                  reason=f"open-loop replay of the executed actions disagrees: {replay}"))
        counted = success and not any(f["invalidates"] for f in flags)
        claim = (e["claim"] or {}).get("kind")
        st = np.asarray(e["states"], np.float64)
        np.savez_compressed(e["dir"] / "states.npz", states=st, actions=np.asarray(e["actions"], np.float64),
                            success=np.asarray(e["success_trace"], bool))
        rec = dict(
            server=SERVER_VERSION, run_id=self.run_id, actor=self.actor, brain=self.brain, split=self.split,
            suite=self.sim.suite_name, views=self.views, task_id=e["task_id"], init_state=e["init_state"],
            env_seed=self.sim.env_seed, instruction=e["instruction"], task_name=self.sim.task_name,
            outcome="success" if success else "failure", counted_success=bool(counted), ended_by=ended_by,
            detail=detail, limit=e["limit"], claim=e["claim"],
            false_claim=bool(claim == "done" and not success), final_predicate=final_pred,
            steps=self.sim.steps, sim_s=round(self.sim.steps / S.CONTROL_HZ, 3), llm_calls=e["calls"],
            tokens=e["tokens"], usd=round(e["usd"], 6), tool_errors=e["tool_errors"],
            wall_s=round(time.time() - e["wall0"], 2), reset_s=e["reset_s"], n_commands=len(e["commands"]),
            monitor=mon, flags=flags, replay=replay, budget=self.budget,
            evaluator=self.evaluator, bddl_sha=self.sim.bddl_sha, init_state_sha=self.sim.init_state_sha,
            commands=e["commands"], files=dict(dir=_rel(e["dir"]), states="states.npz", video="rollout.mp4"))
        (e["dir"] / "episode.json").write_text(json.dumps(rec, indent=1, default=float))
        with open(self.out / "episodes.jsonl", "a") as fh:
            fh.write(json.dumps({k: rec[k] for k in ("run_id", "actor", "split", "task_id", "init_state", "outcome",
                                                     "counted_success", "ended_by", "steps", "llm_calls", "usd",
                                                     "wall_s")} | dict(file=_rel(e["dir"] / "episode.json"))) + "\n")
        e["record"] = rec
        return rec

    def _replay(self, e: dict) -> dict:
        """Integrity: replay the executed (approved) actions open-loop in a fresh env from the same start; the
        per-step predicate trace and the final state must reproduce."""
        try:
            sim2 = S.LiberoSim(self.sim.suite_name, views=self.views)
            sim2.reset(e["task_id"], e["init_state"])
            trace = []
            for a in e["actions"]:
                sim2.step(a)
                trace.append(sim2.success())
            fin = sim2.record_state()
            sim2.close()
            ref = e["states"][-1]
            err = float(np.max(np.abs(fin - ref))) if len(fin) == len(ref) else float("inf")
            ok = trace == e["success_trace"] and err < 1e-6
            return dict(ok=bool(ok), trace_match=trace == e["success_trace"], max_state_err=err,
                        n_actions=len(e["actions"]))
        except Exception as ex:  # noqa: BLE001
            return dict(ok=None, error=f"{type(ex).__name__}: {ex}")

    def _actor_report(self, rec: dict) -> dict:
        if not self.reveal:
            return dict(recorded=True, note=f"the {self.split} outcome goes to the evaluator log only")
        return dict(outcome=rec["outcome"], ended_by=rec["ended_by"], steps=rec["steps"])


# ====================================================================== transports
def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def stdio(cfg: dict):
    """JSON lines on stdin/stdout. Everything LIBERO prints goes to stderr; stdout is protocol only."""
    wire = sys.stdout
    sys.stdout = sys.stderr
    pg = Playground(**cfg)
    wire.write(json.dumps(dict(ready=True, version=SERVER_VERSION, evaluator=pg.evaluator)) + "\n")
    wire.flush()
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except json.JSONDecodeError as ex:
            rep = dict(error=f"bad JSON: {ex}")
            req = None
        else:
            rep = pg.command(req, allowed=HARNESS_OPS)
        wire.write(json.dumps(_clean(rep), default=float) + "\n")
        wire.flush()
        if req is not None and req.get("op") == "close":
            break


def serve(sock_path: str, cfg: dict, idle_s: float = 1800.0):
    """Session daemon for harness tools: one JSON request per connection on a Unix socket (as twin/env.py)."""
    pg = Playground(**cfg)
    p = Path(sock_path)
    p.unlink(missing_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(p))
    os.chmod(p, 0o600)
    srv.listen(4)
    srv.settimeout(idle_s)
    try:
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                pg.close()
                break
            with conn:
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                try:
                    req = json.loads(buf.decode() or "{}")
                    rep = pg.command(req, allowed=HARNESS_OPS)
                except json.JSONDecodeError as ex:
                    req, rep = None, dict(error=f"bad JSON: {ex}")
                conn.sendall((json.dumps(_clean(rep), default=float) + "\n").encode())
                if req is not None and req.get("op") == "close":
                    break
    finally:
        srv.close()
        p.unlink(missing_ok=True)


def freeze():
    """Write frozen.json: the hash of everything that defines success. Done once, by hand, before any episode."""
    ev = S.evaluator_hash()
    FROZEN.write_text(json.dumps(dict(frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), evaluator=ev,
                                      note="LIBERO goal predicates, object-state code, BDDL goals and init states "
                                           "of libero_10 and libero_90 (sim.evaluator_files). The server refuses "
                                           "to start if this changes."), indent=1) + "\n")
    print(json.dumps(ev))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("stdio")
    a.add_argument("--config", required=True)
    b = sub.add_parser("serve")
    b.add_argument("--socket", required=True)
    b.add_argument("--config", required=True)
    b.add_argument("--idle-s", type=float, default=1800.0)
    sub.add_parser("freeze")
    args = ap.parse_args()
    if args.cmd == "stdio":
        stdio(json.loads(args.config))
    elif args.cmd == "serve":
        serve(args.socket, json.loads(args.config), args.idle_s)
    else:
        freeze()


if __name__ == "__main__":
    main()
