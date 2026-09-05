"""External paired LIBERO evaluation for two frozen Controllers."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

from embodied_codex.adapters.factory import load_adapter
from roboforge.candidate_runtime import ControllerRuntime
from evaluation.sealed_evaluation import SealedEvaluationPolicy
from roboforge.store import atomic_write, canonical_json


def states(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1)); result.extend(range(start, end + 1))
        else: result.append(int(part))
    if not result or len(set(result)) != len(result): raise argparse.ArgumentTypeError("invalid states")
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(controller: Path, *, task: str, state: int, root: Path) -> dict:
    started = time.monotonic()
    adapter = load_adapter("libero", task=task, run_dir=root, case=state,
                           configuration={"disable_agent_verifier": True})
    try:
        report = SealedEvaluationPolicy().evaluate_frozen(adapter=adapter,
            runtime=ControllerRuntime(timeout_seconds=600), controller=controller)
        trace_files=list(root.rglob("trace.json")); trace_digest=sha256(trace_files[-1]) if trace_files else None
        return {**report, "elapsed_seconds": time.monotonic() - started,
                "controller_digest": sha256(controller), "trace_digest": trace_digest,
                "episode_digest": hashlib.sha256(canonical_json(report)).hexdigest(),
                "environment": {"task":task,"state":state},
                "receipt_digest": hashlib.sha256(canonical_json(report.get("sealed_evaluation_cases",[]))).hexdigest()}
    finally:
        adapter.close(); del adapter; gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True); parser.add_argument("--states", type=states, required=True)
    parser.add_argument("--baseline", type=Path, required=True); parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    baseline, candidate = args.baseline.resolve(), args.candidate.resolve()
    manifest = {"schema_version": 1, "protocol": "roboforge-paired-libero-v1",
        "task": args.task, "development_states": [0], "held_out_states": args.states,
        "baseline": {"path": str(baseline), "sha256": sha256(baseline)},
        "candidate": {"path": str(candidate), "sha256": sha256(candidate)},
        "evaluator": "LIBERO env.check_success after seal_controller_execution",
        "agent_visible_evaluator": False, "results": [], "started_unix": time.time()}
    atomic_write(output / "preregistered-manifest.json", canonical_json(manifest))
    for state in args.states:
        pair = {"state": state}
        for arm, controller in (("baseline", baseline), ("candidate", candidate)):
            arm_root = output / f"state-{state:02d}" / arm
            try: pair[arm] = evaluate(controller, task=args.task, state=state, root=arm_root)
            except Exception as exc:
                pair[arm] = {"error": f"{type(exc).__name__}: {exc}", "evaluator_successes": 0,
                             "episodes": 1, "success_rate": 0.0}
            manifest["results"].append({"state": state, "arm": arm, **pair[arm]})
            atomic_write(output / "results.partial.json", canonical_json(manifest))
    totals = {}
    for arm in ("baseline", "candidate"):
        rows = [row for row in manifest["results"] if row["arm"] == arm]
        successes = sum(int(row.get("evaluator_successes", 0)) for row in rows)
        totals[arm] = {"successes": successes, "episodes": len(rows),
                       "success_rate": successes / len(rows)}
    manifest["totals"] = totals; manifest["finished_unix"] = time.time()
    atomic_write(output / "paired-evaluation.json", canonical_json(manifest))
    print(json.dumps(totals, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
