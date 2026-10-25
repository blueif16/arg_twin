"""Stage B2 tools: replay real insert attempts in the twin (skill: playbook/skills/replay-validate).

Core code: twin/replay.py (attempts, setpoint reconstruction, scene, replay loop, scoring), twin/control.py
(the REASSEMBLE Cartesian impedance controller), twin/replay_viz.py (plots, videos). Outputs go under
runs/<brain>/<run_id>/replay/.

Tools:
  replay_attempts        in-hand Insert attempts of an episode: contact onset, release, real outcome
  replay_attempt         replay one attempt (compliant | stiff | free) -> outcome, force and onset vs real,
                         trace plot, optional sim video from the calibrated cameras
  replay_batch           replay every in-hand attempt of a split (train | gate) -> balanced accuracy, force peak
                         and onset errors (the B2 numbers), per-attempt rows
  replay_controller_check  free-space check of the controller + setpoint (approach windows), and the
                         stiffness bracket (release jumps, setpoint roughness per K)
  replay_geometry_probe  stiff kinematic replay with the board shifted 0..3 mm: force vs hole offset

Frozen parameters: runs/<brain>/<run_id>/replay/params.json ("params"); tools start from them when
frozen=true (default; params_version v4/v3/v2/v1 = params_v4.json / params_v3.json / params_v2.json / params.json),
then `params_file` (a repo-relative JSON: {"params": {...}} or a flat dict of twin.replay.ReplayParams fields), then
inline `params`. With frozen=false the params_file (or params) is used alone on top of ReplayParams defaults. Exam episodes are refused here and by the registry.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from harness.registry import REPO, tool

_RUN = {"brain": {"type": "string", "default": "claude", "description": "claude (stand-in) or argon: whose run directory"},
        "run_id": {"type": "string", "default": "run-standin-1", "description": "outputs and frozen params: runs/<brain>/<run_id>/replay/"}}
_PARAMS = {"type": "object", "description":
           "Replay parameters on top of the frozen ones (twin/replay.py ReplayParams), e.g. {k_trans: 500, k_rot: 50, "
           "grasp_from_top: 0.005, pad_friction: 1.0, pad_torsion: 0.002, grip_force: 60, friction: 0.3, "
           "seat_depth: 0.02, board_source: 'auto', registration: 'none', hole_widen: 0, timestep: 0.001}. "
           "registration 'release_oracle' / 'full_oracle' copy where the real part went at release: diagnostics "
           "only, never a B2 score."}
SPLITS = REPO / "playbook" / "splits.json"


def _out(brain: str, run_id: str) -> Path:
    for v, f in ((brain, "brain"), (run_id, "run_id")):
        if not v or "/" in v or v.startswith("."):
            raise ValueError(f"bad {f}: {v!r}")
    p = REPO / "runs" / brain / run_id / "replay"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(REPO))


PARAM_FILES = {"v4": "params_v4.json", "v3": "params_v3.json", "v2": "params_v2.json", "v1": "params.json"}


def _frozen(brain: str, run_id: str, version: str = "v3") -> dict:
    """Frozen replay params: runs/<brain>/<run_id>/replay/params_v2.json (default) or params.json (v1)."""
    d = REPO / "runs" / brain / run_id / "replay"
    f = d / PARAM_FILES.get(version, "params_v2.json")
    if not f.exists() and version in ("v3", "v2"):
        f = d / "params.json"
    return json.loads(f.read_text()).get("params", {}) if f.exists() else {}


def _params_file(path: str | None) -> dict:
    """A repo-relative JSON file with replay params: either {"params": {...}} (the frozen-file layout) or a flat
    dict of ReplayParams fields (twin/replay.py). Unknown keys are rejected by ReplayParams.from_dict."""
    if not path:
        return {}
    f = Path(path) if Path(path).is_absolute() else REPO / path
    if not f.exists():
        raise FileNotFoundError(f"params_file {path} not found")
    d = json.loads(f.read_text())
    return dict(d.get("params", d))


def _merge(frozen: bool, params: dict | None, brain: str, run_id: str, version: str = "v3",
           params_file: str | None = None) -> dict:
    """Order: frozen file (if frozen) < params_file < params (inline overrides)."""
    base = _frozen(brain, run_id, version) if frozen else {}
    return {**base, **_params_file(params_file), **(params or {})}


def _split_ok(ep: int):
    s = json.loads(SPLITS.read_text())["episodes"].get(str(int(ep)), {}).get("split")
    if s not in ("train", "gate"):
        raise PermissionError(f"episode {ep} is in split {s!r}; replay tools take train or gate episodes only")


@tool("replay_attempts",
      "List the in-hand Insert attempts of one episode for a part family (fingers closed at the start; pushes with "
      "open fingers are left out): segment index, part, real success, contact onset (|F - F_ref| > 2 N), gripper "
      "release, and the replay window start (onset - 1.5 s). Train or gate episodes only.",
      {"type": "object", "required": ["episode"], "properties": {
          "episode": {"type": "integer"},
          "family": {"type": "string", "default": "peg", "description": "substring of the part name: 'peg', 'gear', 'BNC', ..."}}})
def replay_attempts(episode: int, family: str = "peg") -> dict:
    from twin.replay import list_attempts
    _split_ok(episode)
    att = list_attempts(int(episode), family)
    return dict(episode=int(episode), n=len(att), attempts=[
        dict(segment_index=a["segment_index"], part=a["part"], success=a["success"], released=a["released"],
             t=a["t"], real_peak_dF=a["real_peak_dF"], grip_width_mm=a["grip_width_mm"]) for a in att])


@tool("replay_attempt",
      "Replay one real in-hand Insert attempt in the twin. mode 'compliant' (the validation replay): the twin's copy "
      "of the REASSEMBLE impedance controller drives the arm toward the setpoint rebuilt from the log (x - F/K + "
      "damping/inertia terms, F/T tared in the tool frame), the part is held by the finger pads (contact grasp) and "
      "released when the real gripper opened; MuJoCo decides where it ends up. 'stiff': track the logged pose with "
      "stiff gains (geometry probe). 'free': approach phase only, scored on tracking. Returns sim vs real outcome, "
      "force peak and onset, tracking error, a trace plot, and with video=true a sim render from the calibrated "
      "cameras plus the real clip of the same window.",
      {"type": "object", "required": ["episode", "segment_index"], "properties": {
          "episode": {"type": "integer"}, "segment_index": {"type": "integer"},
          "mode": {"type": "string", "enum": ["compliant", "stiff", "free"], "default": "compliant"},
          "params": _PARAMS,
          "frozen": {"type": "boolean", "default": True, "description": "start from runs/<brain>/<run_id>/replay/params_v3.json (params_version)"},
          "params_version": {"type": "string", "enum": ["v4", "v3", "v2", "v1"], "default": "v3"},
          "params_file": {"type": "string", "description": "repo-relative JSON with replay params ({params: {...}} or flat); applied on top of the frozen file (set frozen=false to use it alone), under inline params"},
          "video": {"type": "boolean", "default": False},
          "camera": {"type": "string", "default": "hama1", "description": "camera for the video (hama1, hama2, hand)"},
          "halved_timestep": {"type": "boolean", "default": False, "description": "rerun at dt/2 (rules.md check)"},
          **_RUN}})
def replay_attempt(episode: int, segment_index: int, mode: str = "compliant", params: dict | None = None,
                   frozen: bool = True, video: bool = False, camera: str = "hama1", halved_timestep: bool = False,
                   brain: str = "claude", run_id: str = "run-standin-1", params_version: str = "v3",
                   params_file: str | None = None) -> dict:
    from twin.replay import run_replay
    from twin.replay_viz import trace_plot
    _split_ok(episode)
    out = _out(brain, run_id)
    p = _merge(frozen, params, brain, run_id, params_version, params_file)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"ep{int(episode):03d}_seg{int(segment_index):02d}_{mode}_{stamp}"
    vid = dict(dir=str(out / "frames" / tag), cameras=[camera]) if video else None
    r = run_replay(int(episode), int(segment_index), p, mode=mode, record_video=vid, halved_timestep=halved_timestep)
    plot = trace_plot(r, out / "plots" / f"{tag}.png")
    full = out / "attempts" / f"{tag}.json"
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps(r, default=float))
    res = {k: r[k] for k in ("episode", "segment_index", "part", "mode", "outcome", "sim_success", "real_success",
                             "match", "final", "at_release", "max_depth_mm", "force", "tracking", "board_pose",
                             "timestep", "wall_s", "window")}
    res["params"] = r["params"]
    res["plot"] = _rel(plot)
    res["json"] = _rel(full)
    if video:
        res["video"] = _make_videos(r, out, tag, camera)
    return res


def _make_videos(r: dict, out: Path, tag: str, camera: str) -> dict:
    """Sim frames (+ force strip) and the real clip of the same window (+ force strip), then one side-by-side."""
    from harness.tools.compare import make_comparison
    from twin.replay_viz import real_frames, strip_frames
    rows = np.array(r["traces"]["rows"], float)
    t = rows[:, 0]
    sim_frames = r.get("video_frames", {}).get(camera, [])
    fdir = out / "frames" / tag
    real = real_frames(r["episode"], camera, r["window"]["t0"], r["window"]["t1"], fdir / f"real_{camera}")
    ser_real = [(rows[:, 2], (60, 60, 60), "real")]
    ser_both = [(rows[:, 2], (60, 60, 60), "real"), (rows[:, 1], (209, 73, 91), "sim")]
    sl = strip_frames(real, t, ser_real, fdir / "real_strip", title=f"real {camera}")
    ss = strip_frames(sim_frames, t, ser_both, fdir / "sim_strip", title=f"twin {camera}")
    cam_src = r.get("camera_source", "")
    cmp_ = make_comparison(_rel(fdir / "real_strip") + "/f_*.png", _rel(fdir / "sim_strip") + "/f_*.png",
                           labels=[f"real {'success' if r['real_success'] else 'failure'}", f"twin {r['outcome']}"],
                           out=_rel(out / "videos" / f"{tag}_{camera}.mp4"), fps=30, max_seconds=40)
    return dict(comparison=cmp_["out"], n_frames=[len(sl), len(ss)], camera=camera, camera_calib=cam_src)


@tool("replay_batch",
      "Replay every in-hand Insert attempt of a family in the local episodes of one split (train or gate; splits.json "
      "filter) with the same parameters, in parallel. Returns the B2 numbers: balanced accuracy of seated vs not "
      "(sim vs dataset success label), median relative error of peak |dF| (attempts with real contact), median "
      "contact-onset error (s), plus per-attempt rows. Writes the full result JSON.",
      {"type": "object", "properties": {
          "split": {"type": "string", "enum": ["train", "gate"], "default": "train"},
          "family": {"type": "string", "default": "peg"},
          "episodes": {"type": "array", "items": {"type": "integer"}, "description": "restrict to these episodes (must be in the split)"},
          "params": _PARAMS, "frozen": {"type": "boolean", "default": True},
          "params_version": {"type": "string", "enum": ["v4", "v3", "v2", "v1"], "default": "v3"},
          "params_file": {"type": "string", "description": "repo-relative JSON with replay params ({params: {...}} or flat); applied on top of the frozen file (set frozen=false to use it alone), under inline params"},
          "mode": {"type": "string", "enum": ["compliant", "free"], "default": "compliant"},
          "workers": {"type": "integer", "default": 15},
          "tag": {"type": "string", "description": "name for the result file"},
          **_RUN}})
def replay_batch(split: str = "train", family: str = "peg", episodes: list[int] | None = None, params: dict | None = None,
                 frozen: bool = True, mode: str = "compliant", workers: int = 15, tag: str | None = None,
                 brain: str = "claude", run_id: str = "run-standin-1", params_version: str = "v3",
                 params_file: str | None = None) -> dict:
    import pandas as pd
    from twin.replay import DATA, _data_file, evaluate, list_attempts, split_of
    if split not in ("train", "gate"):
        raise PermissionError("replay_batch scores train or gate only")
    e = pd.read_parquet(DATA / "meta/episodes/chunk-000/file-000.parquet")
    att = []
    for ep in sorted(int(x) for x in e.episode_index):
        if split_of(ep) != split or (episodes and ep not in episodes):
            continue
        try:
            _data_file(ep)
        except FileNotFoundError:
            continue
        att += list_attempts(ep, family)
    if not att:
        return dict(split=split, n=0, note="no local episodes of this split with in-hand attempts")
    p = _merge(frozen, params, brain, run_id, params_version, params_file)
    ev = evaluate(att, p, mode=mode, workers=workers)
    out = _out(brain, run_id)
    name = tag or f"batch_{split}_{family}_{mode}_{time.strftime('%Y%m%d-%H%M%S')}"
    for r in ev["results"]:
        r.pop("traces", None)
    f = out / f"{name}.json"
    f.write_text(json.dumps(dict(split=split, family=family, params=p, mode=mode, **ev), default=float))
    rows = [dict(ep=r["episode"], seg=r["segment_index"], part=r.get("part"), real=r.get("real_success"),
                 sim=r.get("sim_success"), peak_sim=r.get("force", {}).get("peak_sim_N"),
                 peak_real=r.get("force", {}).get("peak_real_N"), onset_err=r.get("force", {}).get("onset_err_s"),
                 error=r.get("error")) for r in ev["results"]]
    from twin.replay import summarize
    rel = [r for r in ev["results"] if "error" not in r and (r.get("attempt") or {}).get("released", True)]
    return dict(split=split, family=family, params=p, summary=ev["summary"], summary_released_only=summarize(rel),
                rows=rows, json=_rel(f))


@tool("replay_controller_check",
      "Check the twin's controller and setpoint reconstruction against the real arm. what='free_space': replay the "
      "approach phase (window start to 0.1 s before contact) of every in-hand attempt of the train split and report "
      "the EE tracking error (sim vs logged). what='k_bracket': for one episode, contact-release jumps (static K = "
      "dF / jump, dynamic K = omega^2 m_eff from a critically damped fit) and the roughness of the rebuilt setpoint "
      "for each candidate K on one attempt (needs segment_index).",
      {"type": "object", "properties": {
          "what": {"type": "string", "enum": ["free_space", "k_bracket"], "default": "free_space"},
          "episode": {"type": "integer"}, "segment_index": {"type": "integer"},
          "k_list": {"type": "array", "items": {"type": "number"}, "default": [200, 500, 800]},
          "params": _PARAMS, "frozen": {"type": "boolean", "default": True},
          "params_version": {"type": "string", "enum": ["v4", "v3", "v2", "v1"], "default": "v3"},
          "params_file": {"type": "string", "description": "repo-relative JSON with replay params (see replay_batch)"},
          "episodes": {"type": "array", "items": {"type": "integer"}},
          **_RUN}})
def replay_controller_check(what: str = "free_space", episode: int | None = None, segment_index: int | None = None,
                            k_list=(200, 500, 800), params: dict | None = None, frozen: bool = True,
                            episodes: list[int] | None = None, brain: str = "claude", run_id: str = "run-standin-1",
                            params_version: str = "v3", params_file: str | None = None) -> dict:
    from twin.replay import evaluate, release_events, setpoint_roughness, train_attempts
    p = _merge(frozen, params, brain, run_id, params_version, params_file)
    if what == "free_space":
        att = [a for a in train_attempts(episodes=episodes) if a["frames"]["onset"] is not None]
        ev = evaluate(att, p, mode="free")
        tr = [r["tracking"] for r in ev["results"] if "tracking" in r]
        pr = np.array([x["pos_rmse_mm"] for x in tr])
        ar = np.array([x["ang_rmse_deg"] for x in tr])
        return dict(what=what, n=len(tr), pos_rmse_mm_median=round(float(np.median(pr)), 3),
                    pos_rmse_mm_p90=round(float(np.percentile(pr, 90)), 3), ang_rmse_deg_median=round(float(np.median(ar)), 3),
                    params=p)
    if episode is None:
        raise ValueError("k_bracket needs episode")
    _split_ok(episode)
    ev = release_events(int(episode))
    clean = [e for e in ev if e["K_static"] and 1.0 < abs(e["jump_mm"]) < 25 and 3 < e["omega"] < 60 and e["fit_rms_mm"] < 0.3]
    out = dict(what=what, episode=int(episode), release_events=clean,
               K_static_median=round(float(np.median([e["K_static"] for e in clean])), 1) if clean else None,
               K_dynamic_median=round(float(np.median([e["K_dynamic"] for e in clean])), 1) if clean else None)
    if segment_index is not None:
        from twin.replay import ReplayParams
        out["roughness"] = setpoint_roughness(int(episode), int(segment_index), k_list=tuple(map(float, k_list)),
                                              rp=ReplayParams.from_dict(p))
    return out


@tool("replay_geometry_probe",
      "Geometry probe, not validation: stiff kinematic replay of one attempt (the twin's arm tracks the logged pose "
      "with stiff_k N/m) with the board shifted by each offset in +-x and +-y of its own frame. The real part "
      "followed this path, so the twin's force says how far its hole is from where the real part went. Returns "
      "peak and pre-release force per offset. Use registration='full_oracle' in params to measure sensitivity "
      "around a registered hole.",
      {"type": "object", "required": ["episode", "segment_index"], "properties": {
          "episode": {"type": "integer"}, "segment_index": {"type": "integer"},
          "offsets_mm": {"type": "array", "items": {"type": "number"}, "default": [0, 1, 2, 3]},
          "params": _PARAMS, "frozen": {"type": "boolean", "default": True},
          "params_version": {"type": "string", "enum": ["v4", "v3", "v2", "v1"], "default": "v3"},
          "params_file": {"type": "string", "description": "repo-relative JSON with replay params (see replay_batch)"},
          **_RUN}})
def replay_geometry_probe(episode: int, segment_index: int, offsets_mm=(0, 1, 2, 3), params: dict | None = None,
                          frozen: bool = True, brain: str = "claude", run_id: str = "run-standin-1",
                          params_version: str = "v3", params_file: str | None = None) -> dict:
    from twin.replay import geometry_probe
    _split_ok(episode)
    p = _merge(frozen, params, brain, run_id, params_version, params_file)
    r = geometry_probe(int(episode), int(segment_index), tuple(map(float, offsets_mm)), params=p)
    f = _out(brain, run_id) / f"geometry_probe_ep{int(episode):03d}_seg{int(segment_index):02d}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    f.write_text(json.dumps(dict(params=p, **r), default=float))
    return dict(summary_by_offset_mm=r["summary_by_offset_mm"], json=_rel(f), stiff_k=r["stiff_k"])
