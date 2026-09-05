"""Paired LIBERO comparison with and without a reusable capability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_codex.adapters.factory import load_adapter
from roboforge.candidate_runtime import ControllerRuntime
from roboforge.assets import AssetLibrary
from roboforge.providers.libero import LiberoProvider
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
    parser.add_argument("--task", default="0"); parser.add_argument("--states", default="0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"; workspace.mkdir(); controller = workspace / "controller.py"
    controller.write_text(CONTROLLER, encoding="utf-8")
    library = AssetLibrary(args.asset_root); asset = library.read(args.capability, session_id="external-reuse")
    if asset.get("verification_status") != "promoted":
        raise ValueError("cross-task reuse requires an externally promoted capability")
    materialized = CapabilityAcquirer(workspace, library).materialize(
        args.capability, "point_refs.py", session_id="external-reuse")
    rows=[]
    for state in [int(x) for x in args.states.split(",")]:
      for arm, used in (("without",[]),("with",[args.capability])):
        adapter = load_adapter("libero", task=args.task, run_dir=output / f"state-{state}" / arm / "legacy",
            case=state, configuration={"disable_agent_verifier": True})
        try:
          service = ExperimentService(output / f"state-{state}" / arm / "service", LiberoProvider(adapter, ControllerRuntime(timeout_seconds=600)), max_trials=1)
          evidence = service.run_controller(request_id=f"reuse-{state}-{arm}",controller_path=controller,intent=arm,assets_used=used)
          rows.append({"state":state,"arm":arm,"success":bool((evidence.physical_verification or {}).get("verified")),"trial":evidence.public_dict()})
        finally: adapter.close()
    totals={arm:sum(r["success"] for r in rows if r["arm"]==arm)/max(1,sum(r["arm"]==arm for r in rows)) for arm in ("without","with")}
    result = {"schema_version": 2, "task": args.task,
            "capability": args.capability, "verification_status": asset["verification_status"],
            "materialized": materialized, "results":rows,"success_rates":totals,"improvement":totals["with"]-totals["without"]}
    (output / "reuse-validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(totals, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
