"""Audit one or more Embodied Codex runs without executing a robot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_codex.legacy.conformance import audit_run


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("runs",nargs="+")
    parser.add_argument("--output")
    args=parser.parse_args()
    reports=[audit_run(path) for path in args.runs]
    summary={"protocol":"embodied-codex-conformance-matrix-v1",
             "conformant":all(report["conformant"] for report in reports),
             "runs":reports}
    text=json.dumps(summary,indent=2)+"\n"
    if args.output:Path(args.output).write_text(text)
    print(text,end="")
    return 0 if summary["conformant"] else 2


if __name__=="__main__":raise SystemExit(main())
