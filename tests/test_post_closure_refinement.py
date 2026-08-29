import json

from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT
from embodied_codex.fake_adapter import FakeAdapter
from embodied_codex.kernel.agent_loop import LoopBudget
from embodied_codex.kernel.capability_manager import CapabilityManager


def test_diagnostic_budget_is_independent_and_bounded():
    budget = LoopBudget(max_trials=1, max_diagnostics=2)
    assert not budget.diagnostics_exhausted()
    budget.diagnostics = 2
    assert budget.diagnostics_exhausted()
    assert not budget.exhausted()


def test_sdk_exposes_numeric_and_reference_control_equally():
    point = LIBERO_ROBOT_SDK_CONTRACT["actions"]["move_to_point"]
    pose = LIBERO_ROBOT_SDK_CONTRACT["actions"]["move_to_pose"]
    assert {"target_ref"} in [set(item["required"]) for item in point["any_of"]]
    assert {"frame", "position_m"} in [set(item["required"]) for item in point["any_of"]]
    assert {"pose_ref"} in [set(item["required"]) for item in pose["any_of"]]
    assert {"frame", "position_m", "quaternion_xyzw"} in [set(item["required"]) for item in pose["any_of"]]
    assert "both first-class" in point["rule"]


def test_native_capability_search_is_strategy_neutral(tmp_path):
    adapter = FakeAdapter("task", tmp_path / "adapter")
    adapter.native_capability_index = lambda: [{"capability_id": "native:vision",
                                                "purpose": "read-only sensor inspection"}]
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=object(), adapter=adapter)
    result = manager.search("vision")
    assert result["native"][0]["source"] == "native"
    assert "grasp" not in json.dumps(result).lower()
