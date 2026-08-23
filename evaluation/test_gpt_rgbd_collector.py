import pytest

from run_gpt_rgbd_collector import validate_plan


def test_plan_validation_allows_only_symbolic_sensor_targets():
    phases = [{"name": str(i), "target": target, "gripper": -1 if i < 2 else 1, "max_steps": 50, "gain": 8, "settle_steps": 0} for i, target in enumerate(["source_above", "source_grasp", "source_lift", "target_above", "target_place"])]
    assert len(validate_plan({"phases": phases})["phases"]) == 5


def test_plan_validation_rejects_privileged_target():
    phases = [{"name": str(i), "target": "object_ground_truth" if i == 0 else "home", "gripper": -1, "max_steps": 10, "gain": 5, "settle_steps": 0} for i in range(5)]
    with pytest.raises(ValueError):
        validate_plan({"phases": phases})
