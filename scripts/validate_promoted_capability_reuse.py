"""Materialize a promoted capability and reuse it in a fresh LIBERO trial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_codex.adapters.factory import load_adapter
from embodied_codex.kernel.runtime import ControllerRuntime
from roboforge.assets import AssetLibrary
from roboforge.bridge import LegacyAdapterBridge
from roboforge.capability import CapabilityAcquirer
from roboforge.service import ExperimentService


CONTROLLER = '''from point_refs import find_point_ref

def run(robot):
    frame = robot.observe(channel="rgbd", request={})
    detections = robot.use("libero.rgbd_perception:v001", {
        "frame": frame, "queries": ["black bowl"], "max_detections_per_query": 4})
    point_ref = find_point_ref(detections, ("bowl",))
    robot.record({"kind": "promoted_capability_reuse", "point_ref": point_ref})
    return {"point_ref": point_ref}
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--capability", required=True)
    parser.add_argument("--task", default="0"); parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"; workspace.mkdir(); controller = workspace / "controller.py"
    controller.write_text(CONTROLLER, encoding="utf-8")
    library = AssetLibrary(args.asset_root); asset = library.read(args.capability, session_id="external-reuse")
    if asset.get("verification_status") != "promoted":
        raise ValueError("cross-task reuse requires an externally promoted capability")
    materialized = CapabilityAcquirer(workspace, library).materialize(
        args.capability, "point_refs.py", session_id="external-reuse")
    adapter = load_adapter("libero", task=args.task, run_dir=output / "provider" / "legacy",
        case=args.state, configuration={"disable_agent_verifier": True})
    try:
        service = ExperimentService(output / "provider" / "service",
            LegacyAdapterBridge(adapter, ControllerRuntime(timeout_seconds=600)), max_trials=1)
        evidence = service.run_controller(request_id="promoted-capability-reuse",
            controller_path=controller, intent="cross-task promoted capability reuse",
            assets_used=[args.capability])
        result = {"schema_version": 1, "task": args.task, "state": args.state,
            "capability": args.capability, "verification_status": asset["verification_status"],
            "materialized": materialized, "trial": evidence.public_dict()}
        (output / "reuse-validation.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"ref": evidence.ref, "assets_used": list(evidence.assets_used),
                          "execution_error": evidence.execution_error}, indent=2))
        return 0 if evidence.execution_error is None else 1
    finally:
        adapter.close()


if __name__ == "__main__": raise SystemExit(main())
