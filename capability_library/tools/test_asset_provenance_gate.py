import hashlib
import json

from asset_provenance_gate import evaluate_asset_manifest, register_asset_provenance_tool


def manifest(tmp_path, **overrides):
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"general model")
    data = {
        "id": "model.general",
        "artifact": {
            "local_path": str(artifact),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
        "training_provenance": {
            "status": "documented",
            "benchmark_families": [],
        },
        "preprocessing_provenance": {
            "status": "documented",
            "benchmark_families": [],
        },
        "inference_inputs": ["rgb", "language_instruction"],
    }
    data.update(overrides)
    return data


def test_general_model_is_allowed(tmp_path):
    result = evaluate_asset_manifest(manifest(tmp_path))
    assert result == {
        "asset_id": "model.general",
        "eligible": True,
        "track": "harness_acquired_task_zero_shot",
        "reporting_stratum": "benchmark_family_disjoint",
        "reasons": [],
    }


def test_current_task_training_is_rejected(tmp_path):
    data = manifest(tmp_path)
    data["training_provenance"]["benchmark_families"] = ["LIBERO"]
    data["training_provenance"]["task_ids"] = ["libero_object:task_3"]
    result = evaluate_asset_manifest(data, current_tasks=("libero_object:task_3",))
    assert not result["eligible"]
    assert result["reasons"] == ["current task exposure: libero_object:task_3"]
    assert result["reporting_stratum"] == "task_disjoint_transfer"


def test_other_task_training_is_disclosed_but_allowed(tmp_path):
    data = manifest(tmp_path)
    data["training_provenance"]["benchmark_families"] = ["LIBERO"]
    data["training_provenance"]["task_ids"] = ["libero_spatial:task_0"]
    result = evaluate_asset_manifest(data, current_tasks=("libero_object:task_3",))
    assert result["eligible"]
    assert result["reporting_stratum"] == "task_disjoint_transfer"


def test_unknown_normalization_and_privileged_input_are_rejected(tmp_path):
    data = manifest(tmp_path)
    data["preprocessing_provenance"] = {"status": "unknown"}
    data["inference_inputs"] = ["rgb", "reward"]
    result = evaluate_asset_manifest(data)
    assert not result["eligible"]
    assert "preprocessing/action-normalization provenance is not documented" in result["reasons"]
    assert "forbidden inference inputs: reward" in result["reasons"]


def test_thea_registration_exposes_gate(tmp_path):
    class Registry:
        def tool(self, description):
            assert "no-current-task-training" in description

            def decorate(function):
                self.function = function
                return function

            return decorate

    data = manifest(tmp_path)
    path = tmp_path / "asset.json"
    path.write_text(json.dumps(data))
    registry = Registry()
    register_asset_provenance_tool(registry)
    assert registry.function(str(path), ["libero_object:task_0"])["eligible"]
