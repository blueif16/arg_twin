"""CLI over the tool registry, so a Claude agent can call exactly the tools Argon will get.

    .venv/bin/python -m harness.cli list
    .venv/bin/python -m harness.cli call <tool> '<json args>' [--caller NAME]
"""

import argparse
import json

from harness.registry import call, load_all


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    c = sub.add_parser("call")
    c.add_argument("tool")
    c.add_argument("args", nargs="?", default="{}")
    c.add_argument("--caller", default="claude")
    a = p.parse_args()

    if a.cmd == "list":
        for t in load_all().values():
            print(json.dumps(t.declaration()))
    else:
        print(json.dumps(call(a.tool, json.loads(a.args), caller=a.caller), indent=1, default=str))


if __name__ == "__main__":
    main()
