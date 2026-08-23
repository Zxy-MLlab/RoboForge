import hashlib
import json

from self_evolve import SelfEvolutionController, register_self_evolution_tool


def test_evolution_round_searches_gates_and_exports(tmp_path):
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"general")
    manifest = {
        "id": "perception.general",
        "artifact": {"local_path": str(artifact), "sha256": hashlib.sha256(b"general").hexdigest()},
        "training_provenance": {"status": "documented", "benchmark_families": []},
        "preprocessing_provenance": {"status": "documented", "benchmark_families": []},
        "inference_inputs": ["agentview_rgb"],
    }
    calls = []

    def search(query, **kwargs):
        calls.append((query, kwargs))
        return {"success": True, "results": [{"name": "public/tool"}]}

    controller = SelfEvolutionController(
        current_tasks=("libero_object:task_3",),
        ledger_path=str(tmp_path / "events.jsonl"),
    )
    record = controller.evolve_round(
        "grasp slips after contact",
        ["general force closure grasp policy"],
        asset_manifests=[manifest],
        search_fn=search,
    )
    assert record.round_id == 1
    assert record.accepted_assets == ("perception.general",)
    assert len(calls) == 1
    state_path = tmp_path / "state.json"
    controller.export_state(state_path)
    assert json.loads(state_path.read_text())["sealed_results_consumed"] is False


def test_tool_registration_is_bounded():
    class Registry:
        def tool(self, **kwargs):
            def decorate(fn):
                self.fn = fn
                assert kwargs["name"] == "self_evolve_from_failure"
                return fn

            return decorate

    controller = SelfEvolutionController(current_tasks=())
    registry = Registry()
    register_self_evolution_tool(registry, controller)
    result = registry.fn("failure", [], [])
    assert result["success"]
    assert result["sealed_results_consumed"] is False


def test_evolution_round_integrates_and_retries_only_after_accepted_asset(tmp_path):
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"general")
    manifest = {
        "id": "policy.general",
        "artifact": {"local_path": str(artifact), "sha256": hashlib.sha256(b"general").hexdigest()},
        "training_provenance": {"status": "documented", "benchmark_families": []},
        "preprocessing_provenance": {"status": "documented", "benchmark_families": []},
        "inference_inputs": ["agentview_rgb", "language_instruction"],
    }
    integrated = []
    retried = []
    controller = SelfEvolutionController(
        current_tasks=("libero_object:task_3",),
        ledger_path=str(tmp_path / "events.jsonl"),
    )
    record = controller.evolve_round(
        "grasp failure",
        [],
        asset_manifests=[manifest],
        integrate_fn=lambda item: integrated.append(item["id"]) or {"success": True},
        retry_fn=lambda: retried.append(True) or {"success": True, "score": 0.2},
    )
    assert integrated == ["policy.general"]
    assert retried == [True]
    assert record.integration_results[0]["success"] is True
    assert record.retry_result["score"] == 0.2
    events = tmp_path.joinpath("events.jsonl").read_text()
    assert "asset_integration_attempt" in events
    assert "development_retry" in events
