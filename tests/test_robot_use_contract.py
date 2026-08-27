import json

from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT
from embodied_codex.deployments.libero import LiberoDeployment
from embodied_codex.kernel.runtime import ControllerRuntime


def _deployment(tmp_path, capability, output_schema):
    deployment = LiberoDeployment.__new__(LiberoDeployment)
    deployment.capabilities = {"public_tool:v001": capability}
    deployment.capability_contracts = {"public_tool:v001": {
        "input_schema": {"type": "object", "additionalProperties": False},
        "output_schema": output_schema,
    }}
    deployment._native_capability_ids = frozenset()
    deployment.step = 7
    deployment.trace = []
    deployment.references = {}
    deployment.artifact_dir = tmp_path
    deployment._controller_artifact_paths = {}
    return deployment


def test_sdk_declares_robot_use_receipt_wrapper():
    contract = LIBERO_ROBOT_SDK_CONTRACT["methods"]["use"]
    assert contract["output_fields"] == ["tool_id", "step", "result"]
    assert contract["shape"] == {
        "tool_id": "<tool id>",
        "step": "<current step>",
        "result": "<Tool-native result>",
    }
    prose = " ".join(str(contract[key]) for key in ("returns", "rule", "example"))
    assert "receipt['result']" in prose
    assert "no receipt/result wrapper" not in prose


def test_libero_use_success_round_trips_nested_tool_result(tmp_path):
    native = {"detections": [{"label": "public-object", "confidence": 0.9}]}
    deployment = _deployment(tmp_path, lambda _payload: native, {
        "type": "object",
        "properties": {"detections": {"type": "array"}},
        "required": ["detections"],
        "additionalProperties": False,
    })

    raw = deployment._use("public_tool:v001", {})
    receipt = deployment.project_rpc_output("use", {}, raw)

    assert set(receipt) == {"tool_id", "step", "result"}
    assert receipt["tool_id"] == "public_tool:v001"
    assert receipt["step"] == 7
    assert receipt["result"] == native
    assert "detections" not in receipt


def test_libero_use_failure_round_trips_as_public_nested_tool_error(tmp_path):
    deployment = _deployment(tmp_path, lambda _payload: {"detections": "invalid"}, {
        "type": "object",
        "properties": {"detections": {"type": "array"}},
        "required": ["detections"],
        "additionalProperties": False,
    })

    raw = deployment._use("public_tool:v001", {})
    receipt = deployment.project_rpc_output("use", {}, raw)

    assert set(receipt) == {"tool_id", "step", "result"}
    assert receipt["result"]["ok"] is False
    assert receipt["result"]["tool_error"]["type"] == "ToolContractError"
    serialized = json.dumps(receipt)
    assert str(tmp_path) not in serialized
    assert not any(private in serialized for private in
                   ("robot0_", "reward", "done", "hidden_evaluator"))


def test_controller_robot_use_returns_projected_receipt(tmp_path):
    class Deployment:
        instruction = "public contract test"

        def dispatch(self, method, arguments):
            assert method == "use"
            return {"tool_id": arguments["tool_id"], "step": 7,
                    "result": {"detections": ["public-object"]}}

        def project_rpc_output(self, method, arguments, result):
            assert method == "use"
            return dict(result)

    controller = tmp_path / "controller.py"
    controller.write_text(
        "def run(robot):\n"
        "    receipt = robot.use('public_tool:v001', {})\n"
        "    return receipt\n")

    execution = ControllerRuntime(timeout_seconds=10).execute(controller, Deployment())

    assert execution["completed"] is True
    assert execution["result"] == {"tool_id": "public_tool:v001", "step": 7,
                                    "result": {"detections": ["public-object"]}}
