"""Isolated arbitrary controller-program runtime with a stable direct Robot SDK."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import selectors
import subprocess
import sys
import time
from typing import Any, Mapping

from .interfaces import ALLOWED_RPC, RobotDeployment


class ControllerRuntimeError(RuntimeError): pass


_CHILD = r'''
import importlib.util,json,sys

def emit(value):
    sys.stdout.write(json.dumps(value,separators=(",",":"))+"\n");sys.stdout.flush()

class Robot:
    def __init__(self,instruction): self.instruction=instruction;self.request_id=0
    def _rpc(self,method,arguments):
        self.request_id+=1
        emit({"kind":"rpc","id":self.request_id,"method":method,"arguments":arguments})
        line=sys.stdin.readline()
        if not line: raise RuntimeError("deployment parent closed")
        response=json.loads(line)
        if response.get("id")!=self.request_id: raise RuntimeError("RPC response mismatch")
        if not response.get("ok"): raise RuntimeError(str(response.get("error") or "RPC failed"))
        return response.get("result")
    def observe(self,channel="rgbd",request=None):
        return self._rpc("observe",{"channel":channel,"request":request or {}})
    def act(self,action): return self._rpc("act",{"action":action})
    def use(self,tool_id,payload):
        receipt=self._rpc("use",{"tool_id":tool_id,"payload":payload})
        if not isinstance(receipt,dict) or "result" not in receipt:
            raise RuntimeError("deployment use() violated ToolResult contract")
        return receipt["result"]
    def verify(self,verifier,payload):
        return self._rpc("verify",{"verifier":verifier,"payload":payload})
    def record(self,event): return self._rpc("record",{"event":event})

try:
    spec=importlib.util.spec_from_file_location("task_controller",sys.argv[1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if not hasattr(module,"run"): raise RuntimeError("controller must define run(robot)")
    result=module.run(Robot(json.loads(sys.argv[2])))
    emit({"kind":"finished","result":result})
except BaseException as exc:
    emit({"kind":"controller_error","error":type(exc).__name__+": "+str(exc)})
'''


def _public(value: Any) -> Any:
    hidden = {"reward", "done", "success", "check_success", "task_success",
              "evaluator", "evaluator_result", "terminated", "truncated",
              "bddl", "object_pose", "object_id"}
    if isinstance(value, Mapping):
        return {str(k): _public(v) for k, v in value.items()
                if str(k).casefold() not in hidden}
    if isinstance(value, list): return [_public(v) for v in value]
    return value


class ControllerRuntime:
    def __init__(self, *, python: str | Path | None = None,
                 timeout_seconds: float = 600, max_rpc_calls: int = 10000) -> None:
        self.python = str(python or sys.executable)
        self.timeout_seconds = float(timeout_seconds); self.max_rpc_calls = int(max_rpc_calls)

    def execute(self, program_path: str | Path, deployment: RobotDeployment) -> dict[str, Any]:
        path = Path(program_path).resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        process = subprocess.Popen(
            [self.python, "-u", "-I", "-c", _CHILD, str(path),
             json.dumps(str(deployment.instruction))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        assert process.stdin is not None and process.stdout is not None
        selector = selectors.DefaultSelector(); selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + self.timeout_seconds
        events: list[dict[str, Any]] = []; result = None; error = None; completed = False
        try:
            while time.monotonic() < deadline:
                ready = selector.select(max(0.0, deadline - time.monotonic()))
                if not ready: break
                line = process.stdout.readline()
                if not line: break
                message = json.loads(line)
                if message.get("kind") == "rpc":
                    method = str(message.get("method") or "")
                    if method not in ALLOWED_RPC: raise ControllerRuntimeError(f"unsupported RPC: {method}")
                    if len(events) >= self.max_rpc_calls: raise ControllerRuntimeError("RPC budget exceeded")
                    arguments = _public(message.get("arguments") or {})
                    event = {"method": method, "arguments": arguments}
                    try:
                        rpc_result = _public(deployment.dispatch(method, arguments))
                        response = {"id": message.get("id"), "ok": True, "result": rpc_result}
                        event["result"] = rpc_result
                    except Exception as exc:
                        response = {"id": message.get("id"), "ok": False,
                                    "error": f"{type(exc).__name__}: {exc}"}
                        event["error"] = response["error"]
                    events.append(event)
                    process.stdin.write(json.dumps(response, separators=(",", ":")) + "\n")
                    process.stdin.flush()
                elif message.get("kind") == "finished":
                    result = _public(message.get("result")); completed = True; break
                elif message.get("kind") == "controller_error":
                    error = str(message.get("error") or "controller failed"); break
                else: raise ControllerRuntimeError("invalid controller protocol")
        finally:
            if process.poll() is None: process.terminate()
            try: _, stderr = process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill(); _, stderr = process.communicate()
        if not completed and error is None:
            error = "controller timed out" if time.monotonic() >= deadline else "controller exited"
        verified = any(
            event["method"] == "verify"
            and isinstance(event.get("result"), Mapping)
            and event["result"].get("verified") is True
            for event in events
        )
        return {"completed": completed, "program_sha256": digest,
                "result": result, "error": error, "rpc_events": events,
                "sensor_verification_observed": verified, "stderr": stderr[-2000:]}

__all__ = ["ControllerRuntime", "ControllerRuntimeError"]
