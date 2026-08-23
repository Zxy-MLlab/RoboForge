"""Isolated Stage Node runtime using a narrow JSON-RPC robot boundary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import selectors
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

from .adapter import RPC_METHODS
from .errors import NodeRuntimeError


_CHILD = r'''
import importlib.util,json,sys

def emit(value):
    sys.stdout.write(json.dumps(value,separators=(",",":"))+"\n")
    sys.stdout.flush()

class Adapter:
    def __init__(self): self.request_id=0
    def _rpc(self,method,arguments):
        self.request_id+=1
        emit({"kind":"rpc","id":self.request_id,"method":method,"arguments":arguments})
        line=sys.stdin.readline()
        if not line: raise RuntimeError("adapter parent closed")
        response=json.loads(line)
        if response.get("id")!=self.request_id: raise RuntimeError("adapter response mismatch")
        if not response.get("ok"): raise RuntimeError(str(response.get("error") or "adapter call failed"))
        return response.get("result")
    def instruction(self): return self._rpc("instruction",{})
    def sense(self,channel="rgbd",request=None):
        return self._rpc("sense",{"channel":channel,"request":request or {}})
    def act(self,action): return self._rpc("act",{"action":action})
    def use(self,tool_id,payload): return self._rpc("use",{"tool_id":tool_id,"payload":payload})
    def verify(self,verifier,payload): return self._rpc("verify",{"verifier":verifier,"payload":payload})
    def record(self,event): return self._rpc("record",{"event":event})

try:
    spec=importlib.util.spec_from_file_location("embodied_stage",sys.argv[1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.run_stage(Adapter(),json.loads(sys.argv[2]))
    emit({"kind":"finished","result":result})
except BaseException as exc:
    emit({"kind":"node_error","error":type(exc).__name__+": "+str(exc)})
'''


def _sensor_only(value: Any) -> Any:
    hidden = {
        "reward", "done", "success", "check_success", "task_success",
        "evaluator", "evaluator_result", "terminated", "truncated",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _sensor_only(item)
            for key, item in value.items()
            if str(key).casefold() not in hidden
        }
    if isinstance(value, list):
        return [_sensor_only(item) for item in value]
    return value


class NodeRuntime:
    def __init__(
        self, *, python: str | Path | None = None,
        timeout_seconds: float = 300, max_rpc_calls: int = 5000,
    ) -> None:
        self.python = str(python or sys.executable)
        self.timeout_seconds = float(timeout_seconds)
        self.max_rpc_calls = int(max_rpc_calls)

    def execute(
        self, source_path: str | Path, *, expected_sha256: str,
        context: Mapping[str, Any],
        dispatch: Callable[[str, Mapping[str, Any]], Any],
    ) -> dict[str, Any]:
        path = Path(source_path).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise NodeRuntimeError("immutable node source hash mismatch")
        process = subprocess.Popen(
            [self.python, "-u", "-I", "-c", _CHILD, str(path),
             json.dumps(_sensor_only(dict(context)), separators=(",", ":"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        assert process.stdin is not None and process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + self.timeout_seconds
        events: list[dict[str, Any]] = []
        result: Any = None
        error: str | None = None
        completed = False
        try:
            while time.monotonic() < deadline:
                ready = selector.select(max(0.0, deadline - time.monotonic()))
                if not ready:
                    break
                line = process.stdout.readline()
                if not line:
                    break
                message = json.loads(line)
                if message.get("kind") == "rpc":
                    method = str(message.get("method") or "")
                    if method not in RPC_METHODS:
                        raise NodeRuntimeError(f"unsupported adapter RPC: {method}")
                    if len(events) >= self.max_rpc_calls:
                        raise NodeRuntimeError("node exceeded RPC budget")
                    arguments = _sensor_only(message.get("arguments") or {})
                    row = {"method": method, "arguments": arguments}
                    try:
                        rpc_result = _sensor_only(dispatch(method, arguments))
                        response = {"id": message.get("id"), "ok": True,
                                    "result": rpc_result}
                        row["result"] = rpc_result
                    except Exception as exc:
                        response = {"id": message.get("id"), "ok": False,
                                    "error": f"{type(exc).__name__}: {exc}"}
                        row["error"] = response["error"]
                    events.append(row)
                    process.stdin.write(json.dumps(response, separators=(",", ":")) + "\n")
                    process.stdin.flush()
                elif message.get("kind") == "finished":
                    result = _sensor_only(message.get("result"))
                    completed = True
                    break
                elif message.get("kind") == "node_error":
                    error = str(message.get("error") or "node failed")
                    break
                else:
                    raise NodeRuntimeError("node emitted invalid protocol message")
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                _, stderr = process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill(); _, stderr = process.communicate()
        if not completed and error is None:
            error = "node timed out" if time.monotonic() >= deadline else "node exited"
        return {
            "completed": completed, "result": result, "error": error,
            "rpc_events": events, "stderr": stderr[-2000:],
        }


__all__ = ["NodeRuntime"]
