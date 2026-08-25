"""Real-model breadth test for the benchmark-neutral Embodied Codex kernel.

These tiny adapters deliberately expose different action vocabularies and state
shapes.  They catch SDK composition and lifecycle defects before an expensive
simulator rollout; they are not a substitute for embodied benchmark results.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from embodied_codex.conformance import audit_run
from embodied_codex.evolution import EvolutionEngine
from embodied_codex.model import OpenAIModel


CASES: dict[str, dict[str, Any]] = {
    "cursor": {
        "instruction": "Move the cursor to the beacon observed by the position sensor.",
        "initial": {"cursor": 0.0, "beacon": 0.73},
        "actions": {
            "move_cursor": {"required": ["type", "position"],
                            "optional": {}, "returns": "measured cursor position"}},
        "verifier": "cursor_at_beacon",
    },
    "valves": {
        "instruction": "Set every valve to the requested configuration shown by the sensor.",
        "initial": {"valves": [False, False, True], "requested": [True, False, False]},
        "actions": {
            "set_valve": {"required": ["type", "index", "open"],
                          "optional": {}, "returns": "measured valve configuration"}},
        "verifier": "valves_match_request",
    },
    "thermal": {
        "instruction": "Bring the chamber temperature into the safe band reported by the sensor.",
        "initial": {"temperature_c": 17.0, "safe_band_c": [20.5, 21.5]},
        "actions": {
            "change_temperature": {"required": ["type", "delta_c"],
                                   "optional": {}, "returns": "measured temperature"}},
        "verifier": "temperature_in_safe_band",
    },
}


def sdk_contract(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": "embodied-codex-conformance-sdk-v1",
        "methods": {
            "observe": {"signature": "robot.observe(channel='state', request={})",
                        "channels": ["state"], "returns": "direct live sensor object"},
            "use": {"signature": "robot.use(tool_id, payload)",
                    "returns": "direct registered Tool result"},
            "act": {"signature": "robot.act(action)", "returns": "action receipt"},
            "verify": {"signature": "robot.verify(verifier, payload)",
                       "returns": "{verified: boolean, ...measured state}"},
            "record": {"signature": "robot.record(event)",
                       "returns": "{recorded: true}"},
        },
        "actions": case["actions"],
        "verifiers": {case["verifier"]: {"required": [], "optional": {}}},
        "opaque_reference_fields": [],
    }


class ConformanceDeployment:
    def __init__(self, case_name: str, artifact_dir: Path):
        self.case_name = case_name
        self.case = CASES[case_name]
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=False)
        self.state = json.loads(json.dumps(self.case["initial"]))
        self.trace: list[dict[str, Any]] = []
        self.capabilities = {}
        self.closed = False

    @property
    def instruction(self):
        return self.case["instruction"]

    def register_capability(self, tool_id, function, contract):
        self.capabilities[str(tool_id)] = function

    def dispatch(self, method, arguments):
        if method == "observe":
            if arguments.get("channel") != "state":
                raise RuntimeError("this Adapter exposes only the state channel")
            result = json.loads(json.dumps(self.state))
        elif method == "use":
            tool_id = str(arguments.get("tool_id") or "")
            if tool_id not in self.capabilities:
                raise RuntimeError(f"unregistered Tool: {tool_id}")
            result = {"tool_id": tool_id,
                      "result": self.capabilities[tool_id](arguments.get("payload") or {})}
        elif method == "act":
            result = self._act(dict(arguments.get("action") or {}))
        elif method == "verify":
            if arguments.get("verifier") != self.case["verifier"]:
                raise RuntimeError("unknown verifier")
            result = {"verified": self._success(), **json.loads(json.dumps(self.state))}
        elif method == "record":
            result = {"recorded": True}
        else:
            raise RuntimeError(f"unsupported method: {method}")
        self.trace.append({"method": method, "arguments": arguments, "result": result})
        return result

    def project_rpc_output(self,method,arguments,result):
        # This tiny Adapter's public state schema is exactly the case's declared
        # initial sensor keys plus the fixed Robot SDK envelope fields.
        allowed=set(self.case["initial"])|{"tool_id","result","recorded","verified",
                                           "reached","type"}
        unknown=set(result)-allowed
        if unknown:raise RuntimeError(f"undeclared output: {sorted(unknown)}")
        return dict(result)

    def _act(self, action):
        kind = action.get("type")
        if self.case_name == "cursor" and kind == "move_cursor":
            self.state["cursor"] = float(action["position"])
        elif self.case_name == "valves" and kind == "set_valve":
            index = int(action["index"])
            if index not in range(len(self.state["valves"])):
                raise RuntimeError("valve index out of range")
            self.state["valves"][index] = bool(action["open"])
        elif self.case_name == "thermal" and kind == "change_temperature":
            delta = max(-2.0, min(2.0, float(action["delta_c"])))
            self.state["temperature_c"] += delta
        else:
            raise RuntimeError(f"unsupported action: {kind}")
        return {"reached": True, **json.loads(json.dumps(self.state))}

    def _success(self):
        if self.case_name == "cursor":
            return abs(self.state["cursor"] - self.state["beacon"]) <= 1e-6
        if self.case_name == "valves":
            return self.state["valves"] == self.state["requested"]
        low, high = self.state["safe_band_c"]
        return low <= self.state["temperature_c"] <= high

    def sensor_report(self, execution):
        return {
            "sensor_verification_passed": self._success(),
            "benchmark_signal_exposed": False,
            "trace_path": str(self.artifact_dir / "adapter_trace.json"),
            "rollout_path": str(self.artifact_dir / "rollout.mp4"),
        }

    def close(self):
        if self.closed:
            return
        self.closed = True
        (self.artifact_dir / "adapter_trace.json").write_text(
            json.dumps(self.trace, indent=2) + "\n")
        # The conformance adapter has no camera. Keep an explicit empty media
        # artifact so lifecycle/audit code exercises the same transaction path.
        (self.artifact_dir / "rollout.mp4").write_bytes(b"")


class Factory:
    def __init__(self, case_name: str, root: Path):
        self.case_name = case_name
        self.root = root
        self.count = 0

    def __call__(self):
        self.count += 1
        return ConformanceDeployment(
            self.case_name, self.root / "episodes" / f"episode_{self.count:03d}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url", default="https://api.apexin.ai/v1")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-iterations", type=int, default=2)
    args = parser.parse_args()
    key = os.environ.get("APEX_API_KEY")
    if not key:
        raise SystemExit("APEX_API_KEY missing")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = OpenAIModel(api_key=key, base_url=args.base_url, model=args.model,
                        reasoning_effort=args.reasoning_effort, max_tokens=6000)
    summary = []
    for name, case in CASES.items():
        root = output / name
        engine = EvolutionEngine(
            root=root, model=model, deployment_factory=Factory(name, root),
            python=sys.executable,
            deployment_guidance={
                "adapter": {"name": f"conformance-{name}",
                            "observation_channels": ["state"]},
                "robot_sdk_contract": sdk_contract(case),
                "seed_tool_ids": [],
            })
        state = engine.run(task=case["instruction"],
                           skill_name=f"conformance_{name}_skill",
                           max_iterations=args.max_iterations)
        summary.append({"case": name, "status": state["status"],
                        "iterations": len(state["iterations"]),
                        "audit": audit_run(root)})
    report = {"protocol": "embodied-codex-kernel-conformance-v1",
              "model": args.model, "cases": summary,
              "all_conformant": all(row["audit"]["conformant"] for row in summary),
              "all_sensor_success": all(row["status"] == "sensor_success" for row in summary)}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["all_conformant"] and report["all_sensor_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
