"""Isolated runtime for agent-authored embodied controller programs.

Programs own loops and branching, but the only outside-world object they
receive is a JSON-RPC Robot SDK. The deployment owns that SDK and therefore
decides exactly which sensor, Tool, and action operations are available.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import selectors
import subprocess
import sys
import time
from typing import Any, Callable, Mapping


_HIDDEN_KEYS = {
    "reward", "done", "terminated", "truncated", "check_success",
    "task_success", "evaluator_result", "evaluator_success",
}
_METHODS = {"instruction", "observe", "call_tool", "act", "record"}


class ControllerProgramRuntimeError(RuntimeError):
    pass


def sensor_only(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): sensor_only(item)
            for key, item in value.items()
            if str(key).casefold() not in _HIDDEN_KEYS
        }
    if isinstance(value, list):
        return [sensor_only(item) for item in value]
    return value


_CHILD = r'''
import importlib.util,json,sys

def emit(value):
    sys.stdout.write(json.dumps(value,separators=(",",":"))+"\n")
    sys.stdout.flush()

class Robot:
    def __init__(self): self.request_id=0
    def _rpc(self,method,arguments):
        self.request_id+=1
        emit({"kind":"rpc","id":self.request_id,"method":method,"arguments":arguments})
        line=sys.stdin.readline()
        if not line: raise RuntimeError("Robot SDK parent closed")
        response=json.loads(line)
        if response.get("id")!=self.request_id: raise RuntimeError("Robot SDK response mismatch")
        if not response.get("ok"): raise RuntimeError(str(response.get("error") or "Robot SDK call failed"))
        return response.get("result")
    def instruction(self): return self._rpc("instruction",{})
    def observe(self): return self._rpc("observe",{})
    def call_tool(self,name,arguments): return self._rpc("call_tool",{"name":name,"arguments":arguments})
    def act(self,action): return self._rpc("act",{"action":action})
    def record(self,event): return self._rpc("record",{"event":event})

try:
    spec=importlib.util.spec_from_file_location("agent_controller",sys.argv[1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    entrypoint=sys.argv[2]
    arguments=json.loads(sys.argv[3])
    function=getattr(module,entrypoint)
    result=function(Robot()) if entrypoint=="run" else function(Robot(),arguments)
    emit({"kind":"finished","result":result})
except BaseException as exc:
    emit({"kind":"program_error","error":type(exc).__name__+": "+str(exc)})
    raise
'''


class ControllerProgramRuntime:
    """Execute one hash-frozen program against a deployment-owned dispatcher."""

    def __init__(
        self,
        *,
        python: str | Path | None = None,
        timeout_sec: float = 30.0,
        max_rpc_calls: int = 500,
    ) -> None:
        self.python = str(python or sys.executable)
        self.timeout_sec = float(timeout_sec)
        self.max_rpc_calls = int(max_rpc_calls)
        if not 0.5 <= self.timeout_sec <= 3600:
            raise ValueError("timeout_sec must be within [0.5, 3600]")
        if not 1 <= self.max_rpc_calls <= 100000:
            raise ValueError("max_rpc_calls must be within [1, 100000]")

    def run(
        self,
        module: str | Path,
        *,
        expected_sha256: str,
        dispatch: Callable[[str, Mapping[str, Any]], Any],
        entrypoint: str = "run",
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = Path(module).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != str(expected_sha256):
            raise ControllerProgramRuntimeError("controller program source hash changed")
        if entrypoint not in {"run", "run_stage"}:
            raise ControllerProgramRuntimeError("unsupported controller entrypoint")
        safe_arguments = sensor_only(dict(arguments or {}))
        process = subprocess.Popen(
            [
                self.python, "-u", "-I", "-c", _CHILD, str(path),
                entrypoint, json.dumps(safe_arguments, separators=(",", ":")),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdin is not None and process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + self.timeout_sec
        rpc_events = []
        finished: Any = None
        finished_received = False
        error: str | None = None
        try:
            while time.monotonic() < deadline:
                ready = selector.select(max(0.0, deadline - time.monotonic()))
                if not ready:
                    break
                line = process.stdout.readline()
                if not line:
                    break
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ControllerProgramRuntimeError("program emitted non-protocol output") from exc
                kind = message.get("kind")
                if kind == "rpc":
                    method = str(message.get("method") or "")
                    if method not in _METHODS:
                        raise ControllerProgramRuntimeError(f"unsupported Robot SDK method: {method}")
                    if len(rpc_events) >= self.max_rpc_calls:
                        raise ControllerProgramRuntimeError("controller program exceeded RPC budget")
                    arguments = sensor_only(message.get("arguments") or {})
                    event = {"id": message.get("id"), "method": method, "arguments": arguments}
                    try:
                        result = sensor_only(dispatch(method, arguments))
                        response = {"id": message.get("id"), "ok": True, "result": result}
                        event["result"] = result
                    except Exception as exc:
                        response = {
                            "id": message.get("id"), "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        event["error"] = response["error"]
                    rpc_events.append(event)
                    process.stdin.write(json.dumps(response, separators=(",", ":")) + "\n")
                    process.stdin.flush()
                elif kind == "finished":
                    finished = sensor_only(message.get("result"))
                    finished_received = True
                    break
                elif kind == "program_error":
                    error = str(message.get("error") or "controller program failed")
                    break
                else:
                    raise ControllerProgramRuntimeError(f"unknown child protocol message: {kind}")
        except Exception:
            process.kill()
            process.wait(timeout=5)
            raise
        if not finished_received and error is None and time.monotonic() >= deadline:
            process.kill()
            process.wait(timeout=5)
            return {
                "execution_completed": False,
                "error": "controller program timed out",
                "rpc_events": rpc_events,
            }
        if process.poll() is None:
            process.terminate()
        try:
            _, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate()
        return {
            "execution_completed": error is None and finished_received,
            "result": finished,
            "error": error,
            "rpc_events": rpc_events,
            "stderr": stderr[-2000:],
        }


__all__ = [
    "ControllerProgramRuntime",
    "ControllerProgramRuntimeError",
    "sensor_only",
]
