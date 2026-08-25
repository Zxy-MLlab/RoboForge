"""Real-model conformance test for failure-driven capability acquisition.

This is deliberately a tiny, benchmark-neutral Adapter.  Its first episode
exposes an unsupported cyclic coordinate convention; a valid solution must
research the public convention, publish and test a reusable Tool, and make the
next Controller consume that Tool.  The Adapter never exposes an evaluator or
hidden benchmark state to the engineering agent.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from embodied_codex.legacy.conformance import audit_run
from embodied_codex.legacy.evolution import EvolutionEngine
from embodied_codex.model import OpenAIModel


INSTRUCTION = """Calibrate the rotary stage to the canonical signed heading reported by
the live sensor.  The hardware reports an unbounded heading in degrees, while the
stage accepts the public canonical interval [-180, 180).  Start by observing and
attempting the task with the current Controller.  If the sensor verifier rejects
the result, diagnose the missing reusable conversion capability from evidence,
search a public source for the standard cyclic angle-normalization formula,
implement it as a small Tool, register and test its contract, then make the
Controller call the tested Tool on each live reading before moving the stage.
Do not embed the observed reading or expected answer in the Controller."""


def sdk_contract() -> dict[str, Any]:
    return {
        "protocol": "embodied-codex-acquisition-conformance-v1",
        "methods": {
            "observe": {"signature": "robot.observe(channel='state', request={})",
                        "channels": ["state"], "returns": "live stage sensor"},
            "use": {"signature": "robot.use(tool_id, payload)",
                    "returns": "direct registered Tool output"},
            "act": {"signature": "robot.act(action)", "returns": "measured stage heading"},
            "verify": {"signature": "robot.verify(verifier, payload)",
                       "returns": "sensor-only verification"},
            "record": {"signature": "robot.record(event)", "returns": "receipt"},
        },
        "actions": {"move_stage": {"required": ["type", "heading_deg"],
                                     "optional": {},
                                     "returns": "measured heading_deg"}},
        "verifiers": {"stage_at_canonical_heading": {"required": [], "optional": {}}},
        "opaque_reference_fields": [],
    }


class AcquisitionDeployment:
    instruction = INSTRUCTION

    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=False)
        self.raw_heading_deg = 725.0
        self.stage_heading_deg = 0.0
        self.tools: dict[str, Any] = {}
        self.used_tools: list[str] = []
        self.trace: list[dict[str, Any]] = []

    def register_capability(self, tool_id, function, contract):
        self.tools[str(tool_id)] = function

    def dispatch(self, method, arguments):
        if method == "observe":
            result = {"raw_heading_deg": self.raw_heading_deg,
                      "stage_heading_deg": self.stage_heading_deg,
                      "accepted_interval_deg": [-180.0, 180.0]}
        elif method == "use":
            tool_id = str(arguments.get("tool_id") or "")
            if tool_id not in self.tools:
                raise RuntimeError(f"unregistered Tool: {tool_id}")
            self.used_tools.append(tool_id)
            result = {"tool_id": tool_id,
                      "result": self.tools[tool_id](arguments.get("payload") or {})}
        elif method == "act":
            action = dict(arguments.get("action") or {})
            if action.get("type") != "move_stage":
                raise RuntimeError("unsupported action")
            self.stage_heading_deg = float(action["heading_deg"])
            result = {"reached": True, "stage_heading_deg": self.stage_heading_deg}
        elif method == "verify":
            if arguments.get("verifier") != "stage_at_canonical_heading":
                raise RuntimeError("unknown verifier")
            expected = (self.raw_heading_deg + 180.0) % 360.0 - 180.0
            # This is a sensor relation check, not benchmark reward.  Tool use
            # is included because this conformance case specifically validates
            # deployment of a newly acquired capability rather than inlining.
            result = {"verified": abs(self.stage_heading_deg - expected) <= 1e-6,
                      "stage_heading_deg": self.stage_heading_deg,
                      "capability_used": bool(self.used_tools)}
        elif method == "record":
            result = {"recorded": True}
        else:
            raise RuntimeError(method)
        self.trace.append({"method": method, "arguments": arguments, "result": result})
        return result

    def project_rpc_output(self, method, arguments, result):
        allowed = {"raw_heading_deg", "stage_heading_deg", "accepted_interval_deg",
                   "tool_id", "result", "reached", "verified", "capability_used",
                   "recorded"}
        unknown = set(result) - allowed
        if unknown:
            raise RuntimeError(f"undeclared output: {sorted(unknown)}")
        return dict(result)

    def sensor_report(self, execution):
        expected = (self.raw_heading_deg + 180.0) % 360.0 - 180.0
        return {"sensor_verification_passed": (
                    abs(self.stage_heading_deg - expected) <= 1e-6 and bool(self.used_tools)),
                "benchmark_signal_exposed": False,
                "trace_path": str(self.artifact_dir / "adapter_trace.json"),
                "rollout_path": str(self.artifact_dir / "rollout.mp4")}

    def close(self):
        (self.artifact_dir / "adapter_trace.json").write_text(
            json.dumps(self.trace, indent=2) + "\n")
        (self.artifact_dir / "rollout.mp4").write_bytes(b"")


class Factory:
    def __init__(self, root: Path):
        self.root = root
        self.count = 0

    def __call__(self):
        self.count += 1
        return AcquisitionDeployment(self.root / "episodes" / f"episode_{self.count:03d}")


def successful_tool_calls(root: Path, name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "iterations").glob("iteration_*/agent_trace.jsonl")):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if (event.get("type") == "tool_result" and event.get("name") == name
                    and event.get("ok")):
                rows.append(event)
    return rows


def controller_used_generated_tool(root: Path) -> bool:
    for path in sorted((root / "episodes").glob("episode_*/adapter_trace.json")):
        trace = json.loads(path.read_text())
        if any(row.get("method") == "use" for row in trace):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url", default="https://api.apexin.ai/v1")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-iterations", type=int, default=3)
    args = parser.parse_args()
    key = os.environ.get("APEX_API_KEY")
    if not key:
        raise SystemExit("APEX_API_KEY missing")
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    model = OpenAIModel(api_key=key, base_url=args.base_url, model=args.model,
                        reasoning_effort=args.reasoning_effort, max_tokens=7000)
    engine = EvolutionEngine(
        root=root, model=model, deployment_factory=Factory(root),
        python=sys.executable,
        deployment_guidance={"adapter": {"name": "acquisition-conformance",
                                           "observation_channels": ["state"]},
                             "robot_sdk_contract": sdk_contract(), "seed_tool_ids": []})
    state = engine.run(task=INSTRUCTION, skill_name="cyclic_heading_normalization",
                       max_iterations=args.max_iterations)
    audit = audit_run(root)
    counts = {name: len(successful_tool_calls(root, name)) for name in (
        "search_web", "register_tool", "test_tool")}
    report = {"protocol": "embodied-codex-acquisition-conformance-v1",
              "model": args.model, "status": state["status"],
              "iterations": len(state["iterations"]), "tool_calls": counts,
              "controller_used_generated_tool": controller_used_generated_tool(root),
              "audit": audit}
    report["passed"] = (state["status"] == "sensor_success" and audit["conformant"]
                        and all(counts[name] > 0 for name in counts)
                        and report["controller_used_generated_tool"])
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
