from __future__ import annotations

import json
from pathlib import Path

import pytest

from capability_workspace import CapabilityWorkspace
from controller_harness import (
    ControllerValidationError,
    ControllerWorkspace,
    audit_controller_source,
    normalize_controller_spec,
    render_controller_source,
)


BASE_SPEC = {
    "stages": [
        "observe_rgbd",
        "detect_open_vocabulary",
        "segment_source",
        "generate_ranked_grasps",
        "execute_guarded_grasp",
        "verify_attachment",
        "transport",
        "place",
        "verify_placement",
        "finish",
    ],
    "detector_queries": ["black bowl", "plate"],
    "max_grasp_attempts": 3,
    "max_place_corrections": 1,
}


def test_spec_is_bounded_and_canonical():
    normalized = normalize_controller_spec(BASE_SPEC)
    assert normalized["max_grasp_attempts"] == 3
    assert normalized["detector_queries"] == ["black bowl", "plate"]
    with pytest.raises(ControllerValidationError):
        normalize_controller_spec({**BASE_SPEC, "max_grasp_attempts": 50})
    with pytest.raises(ControllerValidationError, match="must not repeat"):
        normalize_controller_spec({
            **BASE_SPEC,
            "stages": ["observe_rgbd", "observe_rgbd", "execute_guarded_grasp", "finish"],
        })
    normalized = normalize_controller_spec({
        **BASE_SPEC,
        "capability_hooks": {"support_relation_profile": "strict_support:v001"},
    })
    assert normalized["capability_hooks"]["support_relation_profile"] == "strict_support:v001"
    normalized = normalize_controller_spec({
        **BASE_SPEC,
        "capability_hooks": {"grasp_execution_profile": "careful_contact:v001"},
    })
    assert normalized["capability_hooks"]["grasp_execution_profile"] == "careful_contact:v001"


def test_generated_source_is_valid_python_and_passes_audit():
    source = render_controller_source(BASE_SPEC)
    compile(source, "controller.py", "exec")
    assert audit_controller_source(source) == {"eligible": True, "violations": []}


def test_audit_rejects_privileged_or_arbitrary_code():
    report = audit_controller_source("import os\nvalue = sim.data.body_xpos\n")
    assert not report["eligible"]
    assert "forbidden_import:os" in report["violations"]
    assert "body_xpos" in report["violations"]


def test_workspace_versions_immutable_candidates(tmp_path: Path):
    workspace = ControllerWorkspace(tmp_path)
    first = workspace.create("pick_place", BASE_SPEC, "initial")
    second = workspace.create("pick_place", BASE_SPEC, "retry")
    assert first["controller_id"] == "pick_place:v001"
    assert second["controller_id"] == "pick_place:v002"
    manifest = json.loads((workspace.resolve(first["controller_id"]) / "manifest.json").read_text())
    assert manifest["audit"]["eligible"]
    assert manifest["status"] == "candidate"
    assert manifest["runtime_dependencies"]


def test_workspace_rejects_path_traversal(tmp_path: Path):
    workspace = ControllerWorkspace(tmp_path)
    with pytest.raises(ControllerValidationError):
        workspace.create("../escape", BASE_SPEC)
    with pytest.raises(ControllerValidationError):
        workspace.resolve("../escape:v001")


def test_workspace_validates_execution_budget(tmp_path: Path):
    assert ControllerWorkspace(tmp_path, max_executions=1).max_executions == 1
    with pytest.raises(ValueError):
        ControllerWorkspace(tmp_path / "bad", max_executions=0)


def test_run_reanalysis_exposes_control_error_but_not_evaluator(tmp_path: Path):
    workspace = ControllerWorkspace(tmp_path)
    created = workspace.create("pick_place", BASE_SPEC)
    controller = workspace.resolve(created["controller_id"])
    run = controller / "runs" / "task3_state0_seed7"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({
        "protocol": "groundingdino-rgbd-closed-loop-v1",
        "language": "put the bowl on the plate",
        "success": True,
        "attachment_verified": False,
    }))
    (run / "trace.json").write_text(json.dumps([
        {"phase": "place", "error": 0.05},
        {"phase": "place", "error": 0.04},
    ]))
    evaluator = run / "_evaluator_only"
    evaluator.mkdir()
    (evaluator / "result.json").write_text(json.dumps({"success": True, "reward": 1}))
    report = workspace.inspect_run(created["controller_id"], task=3, state=0, seed=7)
    assert report["sensor_evidence"]["phase_control_error"]["place"]["final_m"] == 0.04
    assert "success" not in json.dumps(report["sensor_evidence"]).lower()
    assert "reward" not in json.dumps(report["sensor_evidence"]).lower()


def _tested_transport_tool(tmp_path: Path) -> tuple[CapabilityWorkspace, str]:
    capabilities = CapabilityWorkspace(
        tmp_path / "capabilities", python="/data/zxy/envs/vla-report/bin/python"
    )
    created = capabilities.create(
        "gentle_transport",
        "def run(payload):\n"
        "    return {'lift_margin_m': .12, 'horizontal_segments': 3, "
        "'position_gain': .25, 'max_translation_action': .3}\n",
        "Use a slower segmented transport profile",
    )
    tested = capabilities.test(
        created["tool_id"],
        [{
            "input": {},
            "expected": {
                "lift_margin_m": .12,
                "horizontal_segments": 3,
                "position_gain": .25,
                "max_translation_action": .3,
            },
        }],
    )
    assert tested["success"]
    assert capabilities.test_hook(created["tool_id"], "transport_profile")["success"]
    return capabilities, created["tool_id"]


def test_workspace_freezes_previously_tested_capability_into_controller(tmp_path: Path):
    capabilities, tool_id = _tested_transport_tool(tmp_path)
    workspace = ControllerWorkspace(
        tmp_path / "controllers", capability_workspace=capabilities
    )
    created = workspace.create(
        "capability_controller",
        {**BASE_SPEC, "capability_hooks": {"transport_profile": tool_id}},
    )
    controller = workspace.resolve(created["controller_id"])
    manifest = json.loads((controller / "manifest.json").read_text())
    binding = manifest["spec"]["runtime_capability_hooks"]["transport_profile"]
    frozen = controller / binding["module"]
    assert frozen.is_file()
    assert binding["sha256"] == manifest["runtime_dependencies"][str(frozen)]
    assert "runtime_capability_hooks" in (controller / "controller.py").read_text()


def test_current_round_capability_is_not_bindable_until_next_round(tmp_path: Path):
    capabilities = CapabilityWorkspace(
        tmp_path / "capabilities", python="/data/zxy/envs/vla-report/bin/python"
    )
    current_round = ControllerWorkspace(
        tmp_path / "controllers", capability_workspace=capabilities
    )
    _, tool_id = _tested_transport_tool(tmp_path)
    with pytest.raises(ControllerValidationError, match="before this authoring round"):
        current_round.create(
            "too_early",
            {**BASE_SPEC, "capability_hooks": {"transport_profile": tool_id}},
        )
