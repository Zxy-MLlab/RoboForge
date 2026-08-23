import json

import pytest

from capability_workspace import (
    CapabilityValidationError,
    CapabilityWorkspace,
    audit_capability_source,
    register_capability_workspace_tools,
)


class _Registry:
    def __init__(self):
        self.tools = {}

    def tool(self, *, name, **metadata):
        def decorate(function):
            self.tools[name] = (function, metadata)
            return function
        return decorate


def test_capability_is_audited_tested_registered_and_invoked(tmp_path):
    library = tmp_path / "library.json"
    workspace = CapabilityWorkspace(
        tmp_path / "tools", python="/data/zxy/envs/vla-report/bin/python", library_path=library
    )
    created = workspace.create(
        "rank_candidates",
        "def run(payload):\n    return sorted(payload['values'])\n",
        "Sort generic candidate scores",
    )
    result = workspace.test(created["tool_id"], [{"input": {"values": [3, 1, 2]}, "expected": [1, 2, 3]}])
    assert result["success"]
    assert workspace.invoke(created["tool_id"], {"values": [2, 1]})["result"] == [1, 2]
    assets = json.loads(library.read_text())["assets"]
    assert assets[0]["asset_id"] == "tool.agent-authored-rank_candidates.v1"
    assert workspace.tested_tools()[0]["tool_id"] == created["tool_id"]


def test_capability_audit_rejects_privileged_and_unsafe_code(tmp_path):
    report = audit_capability_source("import os\ndef run(payload):\n return check_success()\n")
    assert not report["eligible"]
    assert "forbidden_import:os" in report["violations"]
    assert "check_success" in report["violations"]
    workspace = CapabilityWorkspace(tmp_path, python="/data/zxy/envs/vla-report/bin/python")
    with pytest.raises(CapabilityValidationError):
        workspace.create("unsafe_tool", "def run(payload):\n return open('/tmp/x')\n", "unsafe")


def test_runtime_hook_contract_controls_controller_eligibility(tmp_path):
    workspace = CapabilityWorkspace(
        tmp_path / "tools", python="/data/zxy/envs/vla-report/bin/python"
    )
    valid = workspace.create(
        "bounded_transport",
        "def run(payload):\n"
        "    return {'lift_margin_m': .12, 'horizontal_segments': 3, "
        "'position_gain': .3, 'max_translation_action': .3}\n",
        "Bounded transport profile",
    )
    report = workspace.test_hook(valid["tool_id"], "transport_profile")
    assert report["success"]
    assert workspace.tested_tools()[0]["compatible_hooks"] == ["transport_profile"]

    invalid = workspace.create(
        "unbounded_transport",
        "def run(payload):\n"
        "    return {'lift_margin_m': .12, 'horizontal_segments': 3, "
        "'position_gain': .72, 'max_translation_action': .03}\n",
        "Invalid transport profile",
    )
    rejected = workspace.test_hook(invalid["tool_id"], "transport_profile")
    assert not rejected["success"]
    assert "position_gain" in rejected["outcomes"][0]["error"]


def test_tested_tool_listing_includes_exact_deduplicated_hook_contract(tmp_path):
    workspace = CapabilityWorkspace(
        tmp_path / "tools", python="/data/zxy/envs/vla-report/bin/python"
    )
    created = workspace.create(
        "bounded_transport",
        "def run(payload):\n"
        "    return {'lift_margin_m': .12, 'horizontal_segments': 3, "
        "'position_gain': .3, 'max_translation_action': .3}\n",
        "Bounded transport profile",
    )
    assert workspace.test_hook(created["tool_id"], "transport_profile")["success"]
    registry = _Registry()
    register_capability_workspace_tools(registry, workspace)
    listing = registry.tools["list_tested_capability_tools"][0]()
    assert set(listing["contracts"]) == {"transport_profile"}
    contract = listing["contracts"]["transport_profile"]
    assert "current_eef_xyz" in contract["input_fields"]
    assert "horizontal_segments" in contract["output_fields"]


def test_runtime_hook_rejects_behavioral_duplicate(tmp_path):
    workspace = CapabilityWorkspace(
        tmp_path / "tools", python="/data/zxy/envs/vla-report/bin/python"
    )
    source = (
        "def run(payload):\n"
        "    return {'lift_margin_m': .12, 'horizontal_segments': 3, "
        "'position_gain': .3, 'max_translation_action': .3}\n"
    )
    first = workspace.create("first_transport", source, "first implementation")
    assert workspace.test_hook(first["tool_id"], "transport_profile")["success"]
    second = workspace.create("renamed_transport", source, "same behavior, new name")
    result = workspace.test_hook(second["tool_id"], "transport_profile")
    assert result["success"] is False
    assert result["behavioral_duplicate_of"] == first["tool_id"]
    manifest = json.loads((workspace.resolve(second["tool_id"]) / "manifest.json").read_text())
    assert manifest["status"] == "behavior_duplicate"
    assert manifest["compatible_hooks"] == []


def test_generic_schema_tool_is_tested_and_listed_without_fixed_hook(tmp_path):
    workspace = CapabilityWorkspace(
        tmp_path / "tools", python="/data/zxy/envs/vla-report/bin/python"
    )
    created = workspace.create(
        "articulation_pull_recovery",
        "def run(payload):\n"
        "    p = payload['progress_m']\n"
        "    return {'retry': p < .002, 'lateral_gain': .5 if p < .002 else 0.0}\n",
        "Recover a stalled articulation pull from legal motion progress",
        input_schema={
            "type": "object",
            "properties": {"progress_m": {"type": "number"}},
            "required": ["progress_m"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "retry": {"type": "boolean"},
                "lateral_gain": {"type": "number"},
            },
            "required": ["retry", "lateral_gain"],
            "additionalProperties": False,
        },
        stage="articulation",
    )
    tested = workspace.test(created["tool_id"], [{
        "input": {"progress_m": 0.0003},
        "expected": {"retry": True, "lateral_gain": 0.5},
    }])
    assert tested["success"] is True
    manifest = workspace.tested_tools()[0]
    assert manifest["compatible_hooks"] == []
    assert manifest["generic_contract"]["stage"] == "articulation"
    registry = _Registry()
    register_capability_workspace_tools(registry, workspace)
    listing = registry.tools["list_tested_capability_tools"][0]()
    assert listing["tools"][0]["generic_contract"]["input_schema"]["required"] == [
        "progress_m"
    ]


def test_generic_schema_tool_rejects_unrepresentative_test_payload(tmp_path):
    workspace = CapabilityWorkspace(
        tmp_path / "tools", python="/data/zxy/envs/vla-report/bin/python"
    )
    created = workspace.create(
        "typed_recovery",
        "def run(payload):\n    return {'retry': True}\n",
        "typed recovery",
        input_schema={
            "type": "object", "properties": {"error_m": {"type": "number"}},
            "required": ["error_m"], "additionalProperties": False,
        },
        output_schema={
            "type": "object", "properties": {"retry": {"type": "boolean"}},
            "required": ["retry"], "additionalProperties": False,
        },
    )
    result = workspace.test(created["tool_id"], [{
        "input": {}, "expected": {"retry": True},
    }])
    assert result["success"] is False
    assert "error_m" in result["outcomes"][0]["error"]
