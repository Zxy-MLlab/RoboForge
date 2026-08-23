"""Persistent autonomous learn-debug-revise loop for embodied programs."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from .agent import Agent
from .asset_store import AssetStore
from .authoring_api import AuthoringAPI
from .graph_store import GraphStore
from .model import Model
from .node_store import NodeStore
from .tool_store import ToolStore


SYSTEM_PROMPT = '''You are the engineering model inside a standalone Embodied Harness.
Build executable Python Stage Nodes and Controller Graphs, then execute exactly one
graph per round. Revise only from sensor evidence. Never use reward, done, evaluator
state, BDDL, MuJoCo state/IDs, task/state branches, fixed object coordinates, or a
policy trained on this evaluated task.

STAGE NODE CONTRACT (follow literally):
- source is complete executable Python, not prose or pseudocode.
- It defines exactly: def run_stage(adapter, context):
- The only adapter methods are:
  adapter.instruction()
  adapter.sense(channel="rgbd", request={})
  adapter.act(action_dict)
  adapter.use(tool_id, payload_dict)
  adapter.verify(verifier_name, payload_dict)
  adapter.record(event_dict)
- Every return is exactly an object literal with keys outcome and updates:
  return {"outcome": "observed", "updates": {"frame": observation}}
- The literal updates keys must exactly equal provides_by_outcome[outcome].
- Implement a literal return for every declared outcome.
- checkpoint_outcomes is [] for sensing/action/recovery nodes. A checkpoint node
  must call adapter.verify, must not call adapter.act, and only its genuinely
  verified outcome belongs in checkpoint_outcomes.

Minimal valid sensing source:
def run_stage(adapter, context):
    obs = adapter.sense("rgbd", {})
    if obs.get("frame_id"):
        return {"outcome": "observed", "updates": {"observation": obs}}
    return {"outcome": "failed", "updates": {"failure_reason": "no frame"}}

Minimal valid action source:
def run_stage(adapter, context):
    result = adapter.act({"target_x": context["target_x"]})
    if result.get("reached") is True:
        return {"outcome": "moved", "updates": {"action_result": result}}
    return {"outcome": "failed", "updates": {"failure_reason": result}}

Minimal valid final verification source:
def run_stage(adapter, context):
    proof = adapter.verify("task_relation", {"observation": context["observation"]})
    if proof.get("verified") is True:
        return {"outcome": "verified", "updates": {"proof": proof}}
    return {"outcome": "rejected", "updates": {"failure_reason": proof}}

Every $success edge must come directly from a declared checkpoint outcome. Reuse
required frozen node IDs exactly. Search public resources and create/test a reusable
Tool when evidence shows a missing capability. Do not finish before executing a graph.

LANGUAGE GROUNDING:
Compile every spatial phrase in the task into an explicit calculation over live
detected instances. For example, "between A and B" requires selecting the candidate
nearest the A/B midpoint; "next to A" requires candidate-to-A distance. Never replace
a stated relation with list position, highest detector score, or an arbitrary single-
axis minimum/maximum. Store the selected source and target sensor references so later
verification checks the same independently grounded entities.
'''


class EvolutionEngine:
    def __init__(
        self, *, root: str | Path, model: Model,
        adapter_factory: Callable[[], Any], available_initial_fields: set[str],
        python: str | Path | None = None, max_agent_turns: int = 40,
        deployment_guidance: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.model = model; self.adapter_factory = adapter_factory
        self.available_initial_fields = set(available_initial_fields)
        self.max_agent_turns = int(max_agent_turns)
        self.deployment_guidance = dict(deployment_guidance or {})
        self.nodes = NodeStore(self.root / "nodes", python=python)
        self.tools = ToolStore(self.root / "tools")
        self.assets = AssetStore(self.root / "assets")
        self.state_path = self.root / "evolution_state.json"

    def _load_state(self, task: str, skill_name: str) -> dict[str, Any]:
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text())
            if state["task"] != task or state["skill_name"] != skill_name:
                raise RuntimeError("run directory belongs to a different task")
            return state
        return {
            "task": task, "skill_name": skill_name, "rounds": [],
            "required_revision": None, "status": "evolving",
        }

    def _save_state(self, state: dict[str, Any]) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2, default=str) + "\n")
        temporary.replace(self.state_path)

    @classmethod
    def _compact(cls, value: Any, depth: int = 0) -> Any:
        if depth >= 7: return "<nested evidence omitted>"
        if isinstance(value, dict):
            return {str(key): cls._compact(item, depth + 1)
                    for key, item in list(value.items())[:40]}
        if isinstance(value, list):
            return [cls._compact(item, depth + 1) for item in value[:12]]
        if isinstance(value, str) and len(value) > 600:
            return value[:600] + "..."
        return value

    @staticmethod
    def _selected(mapping: Any, keys: tuple[str, ...]) -> dict[str, Any]:
        if not isinstance(mapping, dict):
            return {}
        return {key: mapping[key] for key in keys if key in mapping}

    @classmethod
    def _diagnostic_rpc_event(cls, event: Any) -> dict[str, Any]:
        """Keep actionable sensor evidence without replaying raw RGB-D metadata."""
        if not isinstance(event, dict):
            return {"malformed_event": str(event)[:240]}
        method = str(event.get("method", "unknown"))
        arguments = event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        brief: dict[str, Any] = {"method": method}
        if method == "sense":
            brief["result"] = cls._selected(result, ("frame_id", "step"))
            proprio = result.get("proprioception", {})
            brief["result"]["proprioception"] = cls._selected(
                proprio, ("robot0_eef_pos", "robot0_eef_quat",
                          "robot0_gripper_qpos", "robot0_gripper_qvel"),
            )
        elif method == "act":
            brief["action"] = cls._compact(arguments.get("action", {}), depth=4)
            brief["result"] = cls._selected(
                result, ("action_type", "step", "reached", "eef_before",
                         "eef_after", "gripper_qpos", "target_xyz", "error"),
            )
        elif method == "use":
            brief["tool_id"] = arguments.get("tool_id")
            payload = arguments.get("payload", {})
            if isinstance(payload, dict):
                brief["queries"] = payload.get("queries")
            receipt = result.get("result", result)
            detections = receipt.get("detections", {}) if isinstance(receipt, dict) else {}
            concise_detections: dict[str, Any] = {}
            if isinstance(detections, dict):
                for query, items in list(detections.items())[:12]:
                    concise_detections[str(query)] = [
                        cls._selected(item, ("label", "score", "sam_score",
                                             "world_xyz", "world_bounds_10_90",
                                             "point_ref"))
                        for item in (items if isinstance(items, list) else [])[:8]
                        if isinstance(item, dict)
                    ]
            brief["detections"] = concise_detections
        elif method == "verify":
            brief["verifier"] = arguments.get("verifier")
            payload = arguments.get("payload", {})
            brief["payload"] = cls._selected(
                payload, ("object_query", "source_ref", "target_ref"),
            )
            brief["result"] = cls._selected(
                result, ("verified", "target_xy_error_m", "vertical_offset_m",
                         "source_vacated", "nearest_source_detection_m",
                         "source_anchor_world_xyz", "target_anchor_world_xyz",
                         "criterion", "capability_tool_id", "error"),
            )
            if isinstance(result.get("object"), dict):
                brief["result"]["object"] = cls._selected(
                    result["object"], ("query", "label", "score", "sam_score",
                                       "world_xyz", "world_bounds_10_90"),
                )
        else:
            brief["arguments"] = cls._compact(arguments, depth=3)
            brief["result"] = cls._compact(result, depth=3)
        return brief

    @classmethod
    def _previous_evidence_summary(cls, previous: Any) -> Any:
        if not isinstance(previous, dict):
            return previous
        summary = cls._selected(
            previous, ("round", "agent_completed", "agent_error", "graph_id",
                       "graph_outcome", "completed", "error",
                       "verified_prefix_aliases", "failure_signature"),
        )
        report = previous.get("sensor_report", {})
        summary["sensor_report"] = cls._selected(
            report, ("sensor_verification_passed", "final_step", "error",
                     "trace_path", "rollout_path", "benchmark_signal_exposed"),
        )
        summary["node_trace"] = [
            cls._selected(node, ("alias", "node_id", "completed", "outcome", "error"))
            for node in previous.get("node_trace", [])
            if isinstance(node, dict)
        ]
        summary["rpc_evidence"] = [
            cls._diagnostic_rpc_event(event)
            for event in previous.get("rpc_evidence", [])[-12:]
        ]
        return summary

    def run(self, *, task: str, skill_name: str, max_rounds: int) -> dict[str, Any]:
        """Run one exclusive evolution writer for this experiment directory."""
        import fcntl
        lock_path = self.root / ".evolution.lock"
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    f"another evolution process owns run directory: {self.root}"
                ) from exc
            lock.seek(0); lock.truncate()
            lock.write(json.dumps({"pid": os.getpid(), "task": task}) + "\n")
            lock.flush()
            return self._run_locked(
                task=task, skill_name=skill_name, max_rounds=max_rounds,
            )

    def _run_locked(
        self, *, task: str, skill_name: str, max_rounds: int,
    ) -> dict[str, Any]:
        state = self._load_state(task, skill_name)
        if state.get("status") == "sensor_success": return state
        start = len(state["rounds"]) + 1
        for round_index in range(start, int(max_rounds) + 1):
            required = state.get("required_revision")
            graphs = GraphStore(
                self.root / "graphs", nodes=self.nodes,
                available_initial_fields=self.available_initial_fields,
                required_revision=required,
            )
            round_dir = self.root / "rounds" / f"round_{round_index:03d}"
            previous = state["rounds"][-1] if state["rounds"] else None
            repeated = 0
            if previous and previous.get("failure_signature"):
                repeated = sum(
                    item.get("failure_signature") == previous["failure_signature"]
                    for item in state["rounds"]
                )
            capability_required = repeated >= 2
            api = AuthoringAPI(
                nodes=self.nodes, graphs=graphs, tools=self.tools,
                assets=self.assets, adapter_factory=self.adapter_factory,
                artifact_dir=round_dir,
                require_capability_activity=capability_required,
            )
            instruction = json.dumps({
                "task": task, "round": round_index,
                "required_revision": required,
                "previous_sensor_evidence": self._previous_evidence_summary(previous),
                "tested_tools": [item["tool_id"] for item in self.tools.tested()],
                "deployment_guidance": self.deployment_guidance,
                "capability_acquisition_required": capability_required,
                "requirement": "author, compile, and execute one controller graph",
            }, default=str)
            agent = Agent(
                model=self.model, tools=api.registry(), system_prompt=SYSTEM_PROMPT,
                trace_path=round_dir / "agent_trace.jsonl",
                max_turns=self.max_agent_turns,
            )
            agent_result = agent.run(instruction)
            if (not agent_result["completed"]
                    and str(agent_result.get("error", "")).startswith(
                        "model_transport_error:")):
                # Transport outages are infrastructure events, not embodied
                # attempts. Preserve their trace and leave this round resumable.
                raise RuntimeError(agent_result["error"])
            record: dict[str, Any] = {
                "round": round_index, "agent_completed": agent_result["completed"],
                "agent_error": agent_result.get("error"),
            }
            if api.last_execution is None:
                record["error"] = "model did not execute a controller graph"
                state["rounds"].append(record); self._save_state(state)
                continue
            run = api.last_execution
            execution = run["execution"]
            record.update({
                "graph_id": run["graph_id"],
                "graph_outcome": execution.get("graph_outcome"),
                "completed": execution.get("completed"),
                "error": execution.get("error"),
                "verified_prefix_aliases": execution.get("verified_prefix_aliases", []),
                "sensor_report": run["sensor_report"],
                "node_trace": self._compact(execution.get("node_trace", [])),
                "rpc_evidence": self._compact(execution.get("rpc_events", [])[-12:]),
            })
            last_node = (execution.get("node_trace") or [{}])[-1]
            record["failure_signature"] = json.dumps({
                "error": execution.get("error"), "alias": last_node.get("alias"),
                "outcome": last_node.get("outcome"),
            }, sort_keys=True)
            state["rounds"].append(record)
            sensor_pass = (
                execution.get("completed") is True
                and execution.get("graph_outcome") == "success"
                and run["sensor_report"].get("sensor_verification_passed") is True
            )
            if sensor_pass:
                used_tools = sorted({
                    str(event["arguments"]["tool_id"])
                    for event in execution.get("rpc_events", [])
                    if event.get("method") == "use"
                    and isinstance(event.get("arguments"), dict)
                    and event["arguments"].get("tool_id")
                } | {
                    str(event["result"]["capability_tool_id"])
                    for event in execution.get("rpc_events", [])
                    if event.get("method") == "verify"
                    and isinstance(event.get("result"), dict)
                    and event["result"].get("capability_tool_id")
                })
                frozen = self.assets.freeze_skill(
                    name=skill_name, task=task, graph_id=run["graph_id"],
                    graphs=graphs, tools=self.tools,
                    development_evidence={
                        "sensor_only_conclusion": "sensor_verification_passed",
                        "round": round_index, "sensor_report": run["sensor_report"],
                    }, tool_ids=used_tools,
                )
                state.update({"status": "sensor_success", "skill": frozen,
                              "required_revision": None})
                self._save_state(state); return state
            state["required_revision"] = {
                "base_graph_id": run["graph_id"],
                "frozen_aliases": execution.get("verified_prefix_aliases", []),
            }
            self._save_state(state)
        return state

    def reject_sensor_candidate(
        self, *, reason: str, evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Resume evolution after an independent audit proves sensor false positive."""
        if not self.state_path.is_file(): raise RuntimeError("no evolution state")
        state = json.loads(self.state_path.read_text())
        if state.get("status") != "sensor_success" or not state.get("skill"):
            raise RuntimeError("run has no sensor-success candidate to reject")
        rejected = self.assets.invalidate_skill(
            state["skill"]["path"], reason=reason, evidence=evidence,
        )
        last = state["rounds"][-1]
        last["candidate_rejected"] = True
        last["external_sensor_audit"] = {"reason": reason, "evidence": evidence}
        last["sensor_report"]["sensor_verification_passed"] = False
        state["rejected_skills"] = state.get("rejected_skills", []) + [rejected]
        state.pop("skill", None)
        state["status"] = "evolving"
        state["required_revision"] = {
            "base_graph_id": last["graph_id"], "frozen_aliases": [],
        }
        self._save_state(state)
        return state


__all__ = ["EvolutionEngine", "SYSTEM_PROMPT"]
