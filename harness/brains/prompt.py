"""System instruction shared by every brain: a short preamble, orchestrator.md, rules.md and the
skill index. Stable parts come first and run-specific parts last, so the prefix stays identical
across turns and runs (implicit caching matches on the prompt prefix; interactions-api-notes.md §3).
"""

from __future__ import annotations

import json

from harness.tools import playbook

PREAMBLE = """\
You are the brain of the Bot18 loop: real robot footage -> MuJoCo twin -> practice, getting more
reliable and cheaper with each attempt. The orchestrator and rules below are fixed; follow them.
When a rule blocks you, stop and say so instead of working around it.

How you work:
- You act only through the harness tools (see "Calling tools" at the end). Every call and result
  is logged and can be replayed.
- Skills: the index at the end lists every playbook skill (name, stage, when to use it). Load a skill
  with read_skill before doing its job, and read its helper files with read_skill(name, file).
  Skills give goals and checks, not recipes: work out the specifics, test them, and judge done by
  the skill's Checks, never by your own opinion of a frame.
- Media: tools write images and clips under runs/ and return their paths. Look at them instead of
  guessing.
- Working notes and decision points go in log_note calls, not in text before a tool call.
- When an attempt teaches something, record it with append_lesson on the skill, citing evidence
  (a file under runs/, a tool output path, or a source URL), with numbers.
- Cost is part of the result: tokens, calls and wall time count. Batch independent tool calls in
  one turn. Stop before the budget runs out.
- Finish with a plain text reply and no tool call: what was done, check results with evidence
  paths, what is blocked, and which skill should be revised.
"""


HOW_TO_CALL = {
    "claude": """\
You are the Claude stand-in. Call tools through the CLI, never by importing harness code:
    .venv/bin/python -m harness.cli list                                    # declarations
    .venv/bin/python -m harness.cli call <tool> '<json args>' --caller claude:{run_id}
Keep --caller exactly as shown so runs/calls/ entries can be imported into this run's record.
When a result names an image, open it with your file-reading tool. You read frames as images, not
video clips, so ask tools for PNG grids rather than mp4 clips.""",
    "default": """\
Tools are function calls. Image and video files named in a tool result are attached right after the
JSON result, with a line naming each path (or saying why one was not attached). You can take video
clips directly. Calls you make in the same turn run in order and return together.""",
}


def skill_index() -> str:
    rows = playbook.list_skills()["skills"]
    if not rows:
        return "(no skills in playbook/skills/ yet)"
    lines = ["| name | stage | when to use |", "|---|---|---|"]
    lines += [f"| {s['name']} | {s['stage'] or ''} | {(s['description'] or '').replace('|', '/')} |"
              for s in rows]
    return "\n".join(lines)


def build_system_instruction(brain: str, run_id: str, budget: dict | None = None,
                             extra: str | None = None) -> str:
    top = {k: playbook.read_playbook_file(k)["text"].strip() for k in ("orchestrator", "rules")}
    parts = [
        PREAMBLE,
        "# playbook/orchestrator.md\n\n" + top["orchestrator"],
        "# playbook/rules.md\n\n" + top["rules"],
        "# Skill index (playbook/skills/*/SKILL.md)\n\n" + skill_index(),
        "# This run\n\n" + f"- run id: {run_id}\n- brain: {brain}\n"
        + (f"- budget: {json.dumps(budget)}\n" if budget else "")
        + f"- records: runs/loop/{run_id}/",
        "# Calling tools\n\n" + HOW_TO_CALL.get(brain, HOW_TO_CALL["default"]).replace("{run_id}", run_id),
    ]
    if extra:
        parts.append(extra.strip())
    return "\n\n".join(parts) + "\n"
