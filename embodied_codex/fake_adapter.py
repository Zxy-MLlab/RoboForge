"""Deterministic, persistent Adapter and model for release smoke testing.

The environment is intentionally domain-neutral and uses the public Adapter
contract.  It exercises the real controller and Tool sandboxes.
"""
from __future__ import annotations

import json
from pathlib import Path
import uuid

import cv2
import numpy as np


class FakeAdapter:
    observation_protocol = "canonical_embodied"
    sdk_index = {"protocol": "roboforge-fake-adapter-v1",
                 "sensors": ["rgb", "proprioception"],
                 "actions": {"set_value": {"value": "integer"}},
                 "verifiers": ["target"]}

    def __init__(self, task: str, root: str | Path, case=None):
        self._instruction = str(task)
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = self.root / "adapter"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.artifact_dir / "environment_state.json"
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text())
        else:
            state = {"value": 0, "generation": uuid.uuid4().hex,
                     "resume_token": uuid.uuid4().hex, "case": case}
        self.value = int(state.get("value", 0))
        self.generation = str(state["generation"])
        self.resume_token = str(state["resume_token"])
        self.case = state.get("case", case)
        self.capabilities = {}
        self.contracts = {}
        self.actions = []
        self.frame = 0
        self.closed = False
        self._persist()

    @property
    def instruction(self):
        return self._instruction

    def _persist(self):
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"value": self.value, "generation": self.generation,
            "resume_token": self.resume_token, "case": self.case}, indent=2) + "\n")
        temporary.replace(self.state_path)

    def execution_identity(self):
        return {"adapter": "fake", "episode_id": f"fake:{self.case}",
                "environment_generation": self.generation}

    def initial_observation(self):
        raw = self.dispatch("observe", {"channel": "rgb", "request": {}})
        return self.project_rpc_output("observe", {"channel": "rgb", "request": {}}, raw)

    def canonical_embodied_state(self):
        return {"frames": {"base": {"name": "base", "parent": None}},
                "eef_frame": "base",
                "robot": {"eef_pose": None,
                          "gripper": {"width_m": None, "state": None},
                          "joint_state": {},
                          "proprioception": {"value": self.value}},
                "observations": {"step": len(self.actions)}}

    def canonical_observation(self, observation):
        if not isinstance(observation, dict):
            return observation
        result = {key: observation[key] for key in ("frame_id", "step", "cameras")
                  if key in observation}
        result["proprioception"] = self.canonical_embodied_state()["robot"]
        return result

    def resume_protocol(self):
        return {"supports_resume": True, "resume_token": self.resume_token,
                "environment_generation": self.generation,
                "actions_idempotent": False, "replay_allowed": True}

    def register_capability(self, tool_id, function, contract):
        if tool_id in self.capabilities:
            return
        self.capabilities[str(tool_id)] = function
        self.contracts[str(tool_id)] = dict(contract)

    def capability_consequence(self, tool_id):
        return str(self.contracts.get(str(tool_id), {}).get("consequence", "UNKNOWN")).upper()

    def begin_execution(self, kind="physical_trial"):
        self._diagnostic_state = {"kind": kind, "frame": self.frame, "verified": False}

    def native_capability_index(self):
        return []

    def dispatch(self, method, arguments):
        if self.closed:
            raise RuntimeError("Adapter is closed")
        if method == "observe":
            channel = str(arguments.get("channel") or "rgb")
            if channel == "proprioception":
                return {"step": len(self.actions), "proprioception": {"value": self.value}}
            if channel != "rgb":
                raise ValueError("unsupported sensor channel")
            self.frame += 1
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[:, :, 1 if self.value == 1 else 2] = 220
            path = self.artifact_dir / f"frame-{self.frame:04d}.png"
            cv2.imwrite(str(path), image)
            return {"step": len(self.actions), "rgb_path": f"artifact://adapter/{path.name}",
                    "proprioception": {"value": self.value}}
        if method == "act":
            action = dict(arguments.get("action") or {})
            if action.get("type") != "set_value" or not isinstance(action.get("value"), int):
                raise ValueError("action must be set_value with an integer value")
            self.value = int(action["value"])
            self.actions.append(action)
            self._persist()
            return {"accepted": True, "step": len(self.actions), "value": self.value}
        if method == "verify":
            if arguments.get("verifier") != "target":
                raise ValueError("unknown verifier")
            return {"verified": self.value == 1, "observed_value": self.value}
        if method == "use":
            tool_id = str(arguments.get("tool_id") or "")
            if tool_id not in self.capabilities:
                raise ValueError(f"unregistered Tool: {tool_id}")
            return {"tool_id": tool_id, "result": self.capabilities[tool_id](dict(arguments.get("payload") or {}))}
        if method == "record":
            return {"recorded": True}
        raise ValueError(method)

    def project_rpc_output(self, method, arguments, result):
        return dict(result)

    def sensor_report(self, execution):
        return {"sensor_success": bool(self.value == 1), "value": self.value,
                "action_log": list(self.actions), "case": self.case}

    def agent_evidence(self, execution, sensor_report):
        return {"observed_value": self.value,
                "actions": len(self.actions)}

    def verification_receipt(self, execution):
        return {"verified": bool(self.value == 1 and execution.get("completed") is True
                                 and not execution.get("error")
                                 and execution.get("sensor_verification_observed") is True),
                "controller_sha256": execution.get("program_sha256"),
                "environment_identity": self.execution_identity(),
                "episode_id": self.execution_identity()["episode_id"],
                "environment_generation": self.generation}

    def validate_execution_receipt(self, receipt):
        return bool(receipt.get("verified") is True
                    and receipt.get("environment_identity") == self.execution_identity()
                    and self.value == 1)

    def reset_generation(self):
        self.generation = uuid.uuid4().hex
        self.resume_token = uuid.uuid4().hex
        self.value = 0
        self._persist()

    def reset_case(self):
        self.generation = uuid.uuid4().hex
        self.resume_token = uuid.uuid4().hex
        self.value = 0
        self.actions = []
        self.frame = 0
        self._persist()
        return self.initial_observation()

    def close(self):
        self._persist()
        self.closed = True


def _call(turn, name, arguments):
    return {"content": "", "tool_calls": [{"id": f"fake-{turn}", "name": name,
        "arguments": json.dumps(arguments)}]}


class FakeModel:
    """State-reactive model used only to exercise the real Harness mechanics."""

    def __init__(self):
        self.turn = 0
        # The fixture models the public intervention protocol explicitly.
        self.decision_open = False
        self.searched = False
        self.reuse = False
        self.inspected = False
        self.shared_bound = False
        self.tool_id = "fake_target:v001"
        self.bad_controller_written = False
        self.execution_inspected = False
        self.image_viewed = False
        self.tool_source_written = False
        self.tool_registered = False
        self.tool_verified = False
        self.fixed_controller_written = False
        self.tool_promoted = False
        self.experience_id = None
        self.experience_promoted = False
        self.skill_id = None
        self.skill_promoted = False
        self.cases = None
        self.passed_cases = set()
        self.evidence_refs = []

    @staticmethod
    def _documents(messages):
        values = []
        for message in messages:
            if not isinstance(message.get("content"), str):
                continue
            try:
                values.append(json.loads(message["content"]))
            except Exception:
                continue
        return values

    @staticmethod
    def _find(value, key):
        if isinstance(value, dict):
            if key in value:
                return value[key]
            for item in value.values():
                found = FakeModel._find(item, key)
                if found is not None:
                    return found
        if isinstance(value, list):
            for item in value:
                found = FakeModel._find(item, key)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _last_call(messages):
        for message in reversed(messages):
            calls = message.get("tool_calls") if isinstance(message, dict) else None
            if calls:
                return calls[-1].get("function", {}).get("name")
        return None

    @staticmethod
    def _current_context(documents):
        return next((value for value in reversed(documents)
                     if isinstance(value, dict) and "workspace" in value
                     and "state" in value), {})

    @staticmethod
    def _latest_tool_payload(messages):
        for message in reversed(messages):
            if message.get("role") != "tool" or not isinstance(message.get("content"), str):
                continue
            try:
                return json.loads(message["content"])
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _evidence_paths(context):
        latest = context.get("latest_evidence") or {}
        return [latest["evidence_ref"]] if latest.get("evidence_ref") else []

    def _consume_result(self, messages):
        name = self._last_call(messages)
        payload = self._latest_tool_payload(messages)
        result = payload.get("result") if payload.get("ok") else None
        if name == "record_decision" and payload.get("ok"):
            self.decision_open = True
        elif name == "run_controller" and payload.get("ok"):
            self.decision_open = False
        if name == "search_assets" and isinstance(result, dict):
            self.searched = True
            promoted = next((item for item in result.get("tools", [])
                             if item.get("status") == "promoted"), None)
            if promoted:
                self.reuse = True
                self.tool_id = promoted["tool_id"]
        elif name == "inspect_asset" and payload.get("ok"):
            self.inspected = True
        elif name == "activate_shared_tool" and payload.get("ok"):
            self.shared_bound = True
        elif name == "list_cases" and isinstance(result, dict):
            self.cases = [str(item) for item in result.get("cases") or []]
        elif name == "write_file" and payload.get("ok"):
            changed = set((result or {}).get("changed") or [])
            if "target_tool.py" in changed:
                self.tool_source_written = True
            elif "controller.py" in changed:
                if self.tool_verified or self.reuse:
                    self.fixed_controller_written = True
                else:
                    self.bad_controller_written = True
        elif name == "inspect_execution" and payload.get("ok"):
            self.execution_inspected = True
        elif name == "view_sensor_artifact" and payload.get("ok"):
            self.image_viewed = True
        elif name == "register_tool" and isinstance(result, dict):
            self.tool_registered = True
            self.tool_id = result["tool_id"]
        elif name == "test_tool" and isinstance(result, dict):
            self.tool_verified = result.get("status") == "verified"
        elif name == "register_experience" and isinstance(result, dict):
            self.experience_id = result.get("experience_id")
        elif name == "register_skill" and isinstance(result, dict):
            self.skill_id = result.get("skill_id")
        elif name == "promote_asset" and isinstance(result, dict):
            asset_id = result.get("tool_id") or result.get("experience_id") or result.get("skill_id")
            if asset_id == self.tool_id:
                self.tool_promoted = True
            elif asset_id == self.experience_id:
                self.experience_promoted = True
            elif asset_id == self.skill_id:
                self.skill_promoted = True
        elif name == "run_controller" and isinstance(result, dict):
            diagnostics = result.get("diagnostics") or {}
            if diagnostics.get("observed_value") == 1:
                evidence_ref = result.get("evidence_ref")
                if evidence_ref and evidence_ref not in self.evidence_refs:
                    self.evidence_refs.append(evidence_ref)

    def decide(self, *, messages, tools):
        self.turn += 1
        documents = self._documents(messages)
        self._consume_result(messages)
        if self.turn > 1 and not self.decision_open:
            return _call(self.turn, "record_decision", {
                "goal": "repair", "evidence_refs": [],
                "hypothesis": "the current public result is not verified",
                "decision": "perform the next intervention",
                "expected_effect": "the public verification result improves",
                "uncertainty": None})
        context = self._current_context(documents)
        latest = context.get("latest_evidence") or {}
        diagnostics = latest.get("diagnostics") or {}
        observed_value = diagnostics.get("observed_value")
        state = context.get("state") or {}
        selected_case = str(state.get("selected_case")) if state.get("selected_case") is not None else "default"
        schema_names = {item.get("function", {}).get("name") for item in tools}
        if observed_value == 1 and self.fixed_controller_written:
            self.passed_cases.add(selected_case)
        workspace_files = {item.get("path") for item in context.get("workspace", [])}
        last_call = self._last_call(messages)
        last_payload = self._latest_tool_payload(messages)
        last_changed = set(((last_payload.get("result") or {}).get("changed") or [])
                           if last_payload.get("ok") else [])
        if not self.searched:
            return _call(self.turn, "search_assets", {"query": "set marker target value", "limit": 5})
        if self.reuse:
            if not self.inspected:
                return _call(self.turn, "inspect_asset", {"asset_id": self.tool_id})
            if not self.shared_bound:
                return _call(self.turn, "activate_shared_tool", {"tool_id": self.tool_id})
            if "list_cases" in schema_names and self.cases is None:
                return _call(self.turn, "list_cases", {})
            if not self.fixed_controller_written:
                source = ("def run(robot):\n    receipt=robot.use(%r,{})\n    target=receipt['result']\n    robot.act({'type':'set_value','value':target['value']})\n"
                          "    return robot.verify('target', {})\n") % self.tool_id
                return _call(self.turn, "write_file", {"path": "controller.py", "content": source})
            required_cases = set(self.cases or ["default"])
            remaining = sorted(required_cases - self.passed_cases)
            if remaining and remaining[0] != selected_case:
                return _call(self.turn, "select_case", {"case_id": remaining[0]})
            if remaining:
                return _call(self.turn, "run_controller", {})
            return _call(self.turn, "finish", {"summary": "reused promoted shared Tool"})

        if "controller.py" not in workspace_files and not self.bad_controller_written:
            bad = "def run(robot):\n    robot.observe('rgb', {})\n    robot.act({'type':'set_value','value':0})\n    return robot.verify('target', {})\n"
            return _call(self.turn, "write_file", {"path": "controller.py", "content": bad})
        if last_call == "write_file" and "controller.py" in last_changed:
            return _call(self.turn, "run_controller", {})
        if observed_value == 0:
            if not self.execution_inspected:
                return _call(self.turn, "inspect_execution", {})
            if not self.image_viewed:
                image_uri = self._find(latest, "rgb_path") or "artifact://adapter/frame-0001.png"
                return _call(self.turn, "view_sensor_artifact", {"path": image_uri})
        if not self.tool_source_written:
            return _call(self.turn, "write_file", {"path": "target_tool.py",
                "content": "def run(payload):\n    return {'value': 1}\n"})
        if "register_tool" not in schema_names:
            return _call(self.turn, "activate_tool_group", {"group": "asset_authoring"})
        if not self.tool_registered:
            return _call(self.turn, "register_tool", {"name": "fake_target", "source_path": "target_tool.py",
                "description": "Return the generic target value", "input_schema": {"type": "object", "properties": {},
                "additionalProperties": False}, "output_schema": {"type": "object", "properties": {"value": {"type": "integer"}},
                "required": ["value"], "additionalProperties": False}})
        if not self.tool_verified:
            return _call(self.turn, "test_tool", {"tool_id": self.tool_id,
                "cases": [{"input": {}, "expected": {"value": 1}}]})
        if not self.fixed_controller_written:
            source = ("def run(robot):\n    receipt=robot.use(%r,{})\n    target=receipt['result']\n    robot.act({'type':'set_value','value':target['value']})\n"
                      "    return robot.verify('target', {})\n") % self.tool_id
            return _call(self.turn, "write_file", {"path": "controller.py", "content": source})
        if "list_cases" in schema_names and self.cases is None:
            return _call(self.turn, "list_cases", {})
        required_cases = set(self.cases or ["default"])
        remaining = sorted(required_cases - self.passed_cases)
        if remaining and remaining[0] != selected_case:
            return _call(self.turn, "select_case", {"case_id": remaining[0]})
        if remaining:
            return _call(self.turn, "run_controller", {})
        evidence_paths = list(self.evidence_refs) or self._evidence_paths(context)
        if not self.tool_promoted:
            return _call(self.turn, "promote_asset", {"asset_id": self.tool_id,
                "evidence_paths": evidence_paths, "applicability": {"adapter_contract": "target-value"}})
        if self.experience_id is None:
            return _call(self.turn, "register_experience", {"name": "verified_target_repair",
                "summary": "A tested target Tool corrected the failed value selection.",
                "applicability": "Adapters exposing a target-value action contract.",
                "keywords": ["target", "repair"], "evidence_paths": evidence_paths})
        if not self.experience_promoted:
            return _call(self.turn, "promote_asset", {"asset_id": self.experience_id,
                "evidence_paths": evidence_paths, "applicability": {"adapter_contract": "target-value"}})
        if len(evidence_paths) >= 2 and self.skill_id is None:
            return _call(self.turn, "register_skill", {"name": "verified_target_skill",
                "task": "set target value", "controller": "controller.py", "tool_ids": [self.tool_id],
                "evidence": {"verified": True}, "evidence_paths": evidence_paths})
        if self.skill_id and not self.skill_promoted:
            return _call(self.turn, "promote_asset", {"asset_id": self.skill_id,
                "evidence_paths": evidence_paths, "applicability": {"adapter_contract": "target-value"}})
        return _call(self.turn, "finish", {"summary": "verified after evidence-driven repair"})


__all__ = ["FakeAdapter", "FakeModel"]
