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
        self.state_path = self.root / "fake_environment.json"
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

    def resume_protocol(self):
        return {"supports_resume": True, "resume_token": self.resume_token,
                "environment_generation": self.generation,
                "actions_idempotent": False, "replay_allowed": True}

    def register_capability(self, tool_id, function, contract):
        if tool_id in self.capabilities:
            return
        self.capabilities[str(tool_id)] = function
        self.contracts[str(tool_id)] = dict(contract)

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

    def close(self):
        self._persist()
        self.closed = True


def _call(turn, name, arguments):
    return {"content": "", "tool_calls": [{"id": f"fake-{turn}", "name": name,
        "arguments": json.dumps(arguments)}]}


class FakeModel:
    """Evidence-driven scripted model used only to prove the real Harness path."""

    def __init__(self):
        self.turn = 0
        self.reuse = False
        self.tool_id = "fake_target:v001"
        self.evidence_uri = None
        self.image_uri = None

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

    def decide(self, *, messages, tools):
        self.turn += 1
        documents = self._documents(messages)
        if self.turn == 1:
            return _call(self.turn, "search_assets", {"query": "set marker target value", "limit": 5})
        if self.turn == 2:
            tool_rows = []
            for document in documents:
                result = self._find(document, "tools")
                if isinstance(result, list):
                    tool_rows.extend(result)
            tested = next((row for row in tool_rows if row.get("status") == "tested"), None)
            self.reuse = tested is not None
            if tested:
                self.tool_id = tested["tool_id"]
                return _call(self.turn, "inspect_asset", {"asset_id": self.tool_id})
            bad = "def run(robot):\n    robot.observe('rgb', {})\n    robot.act({'type':'set_value','value':0})\n    return robot.verify('target', {})\n"
            return _call(self.turn, "write_file", {"path": "controller.py", "content": bad})
        if self.reuse:
            if self.turn == 3:
                source = ("def run(robot):\n    target=robot.use(%r,{})\n    robot.act({'type':'set_value','value':target['value']})\n"
                          "    return robot.verify('target', {})\n") % self.tool_id
                return _call(self.turn, "write_file", {"path": "controller.py", "content": source})
            if self.turn == 4:
                return _call(self.turn, "run_controller", {})
            return _call(self.turn, "finish", {"summary": "reused tested shared Tool"})
        if self.turn == 3:
            return _call(self.turn, "run_controller", {})
        if self.turn == 4:
            return _call(self.turn, "inspect_execution", {})
        if self.turn == 5:
            for document in reversed(documents):
                candidate = self._find(document, "rgb_path")
                if isinstance(candidate, str):
                    self.image_uri = candidate
                    break
            return _call(self.turn, "view_sensor_artifact", {"path": self.image_uri or "artifact://adapter/frame-0001.png"})
        if self.turn == 6:
            return _call(self.turn, "write_file", {"path": "target_tool.py",
                "content": "def run(payload):\n    return {'value': 1}\n"})
        if self.turn == 7:
            return _call(self.turn, "register_tool", {"name": "fake_target", "source_path": "target_tool.py",
                "description": "Return the generic target value", "input_schema": {"type": "object", "properties": {},
                "additionalProperties": False}, "output_schema": {"type": "object", "properties": {"value": {"type": "integer"}},
                "required": ["value"], "additionalProperties": False}})
        if self.turn == 8:
            return _call(self.turn, "test_tool", {"tool_id": self.tool_id,
                "cases": [{"input": {}, "expected": {"value": 1}}]})
        if self.turn == 9:
            source = ("def run(robot):\n    target=robot.use(%r,{})\n    robot.act({'type':'set_value','value':target['value']})\n"
                      "    return robot.verify('target', {})\n") % self.tool_id
            return _call(self.turn, "write_file", {"path": "controller.py", "content": source})
        if self.turn == 10:
            return _call(self.turn, "run_controller", {})
        for document in reversed(documents):
            candidate = self._find(document, "artifact_uri")
            if isinstance(candidate, str):
                self.evidence_uri = candidate
                break
        if self.turn == 11:
            return _call(self.turn, "register_experience", {"name": "verified_target_repair",
                "summary": "A tested target Tool corrected the failed value selection.",
                "applicability": "Adapters exposing a target-value action contract.",
                "keywords": ["target", "repair"], "evidence_paths": [self.evidence_uri]})
        if self.turn == 12:
            return _call(self.turn, "register_skill", {"name": "verified_target_skill",
                "task": "set target value", "controller": "controller.py", "tool_ids": [self.tool_id],
                "evidence": {"verified": True}, "evidence_paths": [self.evidence_uri]})
        return _call(self.turn, "finish", {"summary": "verified after evidence-driven repair"})


__all__ = ["FakeAdapter", "FakeModel"]
