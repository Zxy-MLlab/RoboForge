"""Public OpenHands hook that prevents experiment-history contamination."""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any


_SENSITIVE_ENV_NAME = re.compile(
    r"(?i)\b[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?)[A-Z0-9_]*\b"
)
_BROAD_ENV_READ = re.compile(
    r"(?ix)"
    r"(?:^|[;&|]\s*)(?:env|printenv)(?:\s*(?:[;&|]|$))|"
    r"/proc/(?:self|[0-9]+)/environ|"
    r"(?:print|pprint)\s*\(\s*(?:dict\s*\(\s*)?os\.environ\b(?!\s*\.get\s*\()|"
    r"os\.environ\.(?:items|keys|values)\s*\(|"
    r"(?:^|[;&|]\s*)(?:export\s+-p|declare\s+-x?p)(?:\s|$)"
)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _candidate_paths(event: dict[str, Any], working_dir: Path) -> list[Path]:
    tool_input = event.get("tool_input") or {}
    candidates: list[Path] = []
    direct = tool_input.get("path")
    if isinstance(direct, str) and direct:
        candidates.append((working_dir / direct).resolve() if not Path(direct).is_absolute() else Path(direct).resolve())
    command = tool_input.get("command")
    if isinstance(command, str):
        try:
            tokens = shlex.split(command, comments=False, posix=True)
        except ValueError:
            tokens = command.split()
        for raw in tokens:
            token = raw.strip(";|&<>(){}[]")
            if not token or token.startswith(("http://", "https://")):
                continue
            if token.startswith(("/", "./", "../")):
                path = Path(token)
                candidates.append(path.resolve() if path.is_absolute() else (working_dir / path).resolve())
    return candidates


def _sensitive_environment_read(command: str) -> bool:
    """Reject commands that could copy model or service credentials to logs."""
    return bool(_SENSITIVE_ENV_NAME.search(command) or _BROAD_ENV_READ.search(command))


def evaluate(event: dict[str, Any], *, workspace: Path,
             forbidden: list[Path]) -> dict[str, str]:
    working_dir = Path(event.get("working_dir") or workspace).resolve()
    tool_input = event.get("tool_input") or {}
    command = tool_input.get("command")
    if isinstance(command, str) and _sensitive_environment_read(command):
        return {
            "decision": "deny",
            "reason": (
                "Credential-bearing environment values may not be read or "
                "copied into OpenHands conversation logs."
            ),
        }
    for candidate in _candidate_paths(event, working_dir):
        for root in forbidden:
            # Block both direct reads below a forbidden root and broad listings
            # of an ancestor that would reveal its contents. Explicit allowed
            # Workspace/source paths remain outside the configured roots.
            if _within(candidate, root) or _within(root, candidate):
                return {
                    "decision": "deny",
                    "reason": (
                        "This clean autonomous-development run may not read "
                        f"historical experiment data: {candidate}"
                    ),
                }
    return {"decision": "allow", "reason": "path is outside forbidden history roots"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--forbid", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    event = json.load(sys.stdin)
    result = evaluate(
        event,
        workspace=args.workspace.resolve(),
        forbidden=[path.resolve() for path in args.forbid],
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
