"""Run one frozen Controller through direct and RoboForge LIBERO paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from embodied_codex.adapters.factory import load_adapter
from embodied_codex.kernel.runtime import ControllerRuntime
from roboforge.bridge import LegacyAdapterBridge
from roboforge.service import ExperimentService


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instruction(adapter) -> str:
    value = adapter.instruction
    return str(value() if callable(value) else value)


def direct_arm(controller: Path, root: Path, task: str, state: int) -> dict:
    adapter = load_adapter("libero", task=task, run_dir=root, case=state,
                           configuration={"disable_agent_verifier": True})
    try:
        adapter.begin_execution("physical_trial")
        adapter.reset_case()
        identity = adapter.execution_identity()
        started = time.monotonic()
        execution = ControllerRuntime(timeout_seconds=600).execute(controller, adapter)
        report = adapter.sensor_report(execution)
        public = adapter.agent_evidence(execution, report)
        receipt = adapter.verification_receipt(execution)
        return {"elapsed_seconds": time.monotonic() - started,
                "instruction": instruction(adapter), "identity": identity,
                "public": public, "receipt": receipt}
    finally:
        adapter.close()


def provider_arm(controller: Path, root: Path, task: str, state: int) -> dict:
    adapter = load_adapter("libero", task=task, run_dir=root / "legacy", case=state,
                           configuration={"disable_agent_verifier": True})
    try:
        service = ExperimentService(root / "service",
            LegacyAdapterBridge(adapter, ControllerRuntime(timeout_seconds=600)), max_trials=1)
        started = time.monotonic()
        evidence = service.run_controller(request_id="runtime-consistency",
            controller_path=controller, intent="runtime consistency validation")
        return {"elapsed_seconds": time.monotonic() - started,
                "instruction": instruction(adapter), "identity": adapter.execution_identity(),
                "evidence": evidence.public_dict()}
    finally:
        adapter.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="0"); parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--controller", type=Path,
        default=Path("validation/controllers/runtime_consistency.py"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    controller = args.controller.resolve()
    manifest = {"schema_version": 1, "task": args.task, "state": args.state,
                "controller": str(controller), "controller_sha256": digest(controller),
                "direct": direct_arm(controller, output / "direct", args.task, args.state),
                "roboforge": provider_arm(controller, output / "roboforge", args.task, args.state)}
    direct = manifest["direct"]; candidate = manifest["roboforge"]
    public = candidate["evidence"]["public"]
    manifest["comparison"] = {
        "instruction_equal": direct["instruction"] == candidate["instruction"],
        "task_identity_equal": direct["identity"].get("task_identity") ==
                               candidate["identity"].get("task_identity"),
        "direct_final_step": direct["public"].get("final_step"),
        "roboforge_final_step": public.get("final_step"),
        "trace_semantics_equal": bool(direct["public"].get("final_step") ==
                                      public.get("final_step")),
    }
    (output / "runtime-consistency.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest["comparison"], indent=2, sort_keys=True))
    return 0 if all(value for key, value in manifest["comparison"].items()
                    if key.endswith("_equal")) else 1


if __name__ == "__main__": raise SystemExit(main())
