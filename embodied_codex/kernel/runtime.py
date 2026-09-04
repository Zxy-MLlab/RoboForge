"""Network-isolated Controller runtime and generic Robot RPC boundary."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

from ..interfaces import ALLOWED_RPC, RobotDeployment
from .sandbox import SandboxBackend, default_sandbox


class ControllerRuntimeError(RuntimeError):
    pass


_CHILD = r'''
import base64,json,os,sys,types
try:
 import numpy as np
except ImportError:
 np = None
sys.dont_write_bytecode=True
def _pack(value):
    if np is not None and isinstance(value,np.ndarray):
        data=np.ascontiguousarray(value)
        return {"__roboforge_ndarray__":True,"dtype":str(data.dtype),"shape":list(data.shape),
                "data_base64":base64.b64encode(data.tobytes()).decode("ascii")}
    if np is not None and isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {str(k):_pack(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_pack(v) for v in value]
    return value
def _unpack(value):
    if isinstance(value,dict) and value.get("__roboforge_ndarray__"):
        if "data_base64" in value:
            if np is None: raise RuntimeError("numpy is required to decode Robot SDK arrays")
            raw=base64.b64decode(value["data_base64"])
            return np.frombuffer(raw,dtype=value["dtype"]).reshape(value["shape"]).copy()
        return np.asarray(value["data"],dtype=value.get("dtype"))
    if isinstance(value,dict): return {k:_unpack(v) for k,v in value.items()}
    if isinstance(value,list): return [_unpack(v) for v in value]
    return value
def emit(value):
    sys.stdout.write(json.dumps(_pack(value),separators=(",",":"))+"\n");sys.stdout.flush()
class Robot:
    def __init__(self,instruction): self.instruction=instruction;self.request_id=0
    def _rpc(self,method,arguments):
        self.request_id+=1;emit({"kind":"rpc","id":self.request_id,"method":method,"arguments":arguments})
        line=sys.stdin.readline()
        if not line: raise RuntimeError("Adapter parent closed")
        response=json.loads(line)
        if response.get("id")!=self.request_id: raise RuntimeError("RPC response mismatch")
        if not response.get("ok"): raise RuntimeError(str(response.get("error") or "RPC failed"))
        return _unpack(response.get("result"))
    def observe(self,channel="rgbd",request=None): return self._rpc("observe",{"channel":channel,"request":request or {}})
    def act(self,action): return self._rpc("act",{"action":action})
    def use(self,tool_id,payload):
        receipt=self._rpc("use",{"tool_id":tool_id,"payload":payload})
        if not isinstance(receipt,dict) or "result" not in receipt: raise RuntimeError("Adapter use() violated ToolResult contract")
        return receipt
    def check_observable_condition(self,verifier,payload):
        return self._rpc("check_observable_condition",{"verifier":verifier,"payload":payload})
    def verify(self,verifier,payload):
        return self._rpc("verify",{"verifier":verifier,"payload":payload})
    def record(self,event): return self._rpc("record",{"event":event})
    def sdk(self,method,*args,**kwargs):
        receipt=self._rpc("sdk",{"method":str(method),"args":list(args),"kwargs":dict(kwargs)})
        if not isinstance(receipt,dict) or receipt.get("method")!=str(method):
            raise RuntimeError("Adapter sdk() violated Robot SDK contract")
        return receipt.get("result")
    def __getattr__(self,name):
        if name.startswith("_"): raise AttributeError(name)
        return lambda *args,**kwargs:self.sdk(name,*args,**kwargs)
try:
    path=os.path.abspath(sys.argv[1]);source_root=os.path.abspath(sys.argv[3])
    if os.path.commonpath((path,source_root))!=source_root: raise RuntimeError("Controller escaped source root")
    sys.path.insert(0,source_root);sys.path.insert(0,os.path.dirname(path))
    module=types.ModuleType("task_controller");module.__file__=path
    with open(path,"rb") as stream:source=stream.read()
    exec(compile(source,path,"exec"),module.__dict__)
    if not hasattr(module,"run"): raise RuntimeError("controller must define run(robot)")
    emit({"kind":"finished","result":module.run(Robot(json.loads(sys.argv[2])))})
except BaseException as exc:
    emit({"kind":"controller_error","error":type(exc).__name__+": "+str(exc)})
'''

_ARGUMENT_KEYS = {"observe": {"channel", "request"}, "act": {"action"},
                  "use": {"tool_id", "payload"},
                  "check_observable_condition": {"verifier", "payload"},
                  "verify": {"verifier", "payload"},
                  "record": {"event"}, "sdk": {"method", "args", "kwargs"}}


def _assert_json(value: Any, path: str = "result") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_json(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_json(item, f"{path}[{index}]")
    try:
        json.dumps(value, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ControllerRuntimeError(f"RPC output is not strict JSON at {path}") from exc


def _decode_controller_value(value: Any) -> Any:
    if isinstance(value, Mapping) and value.get("__roboforge_ndarray__"):
        import numpy as np
        raw = value.get("data_base64")
        if raw is not None:
            import base64
            return np.frombuffer(base64.b64decode(raw), dtype=value["dtype"]).reshape(value["shape"]).copy()
        return np.asarray(value.get("data"), dtype=value.get("dtype"))
    if isinstance(value, Mapping): return {str(k): _decode_controller_value(v) for k, v in value.items()}
    if isinstance(value, list): return [_decode_controller_value(v) for v in value]
    return value


def _trace_value(value: Any) -> Any:
    """Keep RPC trace semantics without duplicating bulk array payloads.

    The full projected value is still returned to the isolated Controller.
    Only the retained event log replaces ndarray bytes with a stable content
    fingerprint, shape, and dtype.  This prevents legitimate perception and
    feedback-control loops from exhausting the bounded event log merely by
    reading public observations repeatedly.
    """
    if isinstance(value, Mapping):
        if value.get("__roboforge_ndarray__") is True:
            encoded = value.get("data_base64")
            if isinstance(encoded, str):
                raw = base64.b64decode(encoded, validate=True)
                return {
                    "__roboforge_ndarray__": True,
                    "dtype": str(value.get("dtype") or ""),
                    "shape": list(value.get("shape") or []),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "byte_length": len(raw),
                }
        return {str(key): _trace_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trace_value(item) for item in value]
    # RPC arguments have already been decoded and may therefore contain numpy
    # arrays.  Avoid importing numpy in the generic runtime just for tracing.
    if (type(value).__module__.startswith("numpy") and hasattr(value, "tobytes")
            and hasattr(value, "shape") and hasattr(value, "dtype")):
        raw = value.tobytes(order="C")
        return {
            "__roboforge_ndarray__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "byte_length": len(raw),
        }
    return value


def _rpc_arguments(method: str, value: Any):
    if not isinstance(value, Mapping):
        raise ControllerRuntimeError("RPC arguments must be an object")
    unknown = set(str(key) for key in value) - _ARGUMENT_KEYS[method]
    if unknown:
        raise ControllerRuntimeError(f"unsupported {method} argument fields: {sorted(unknown)}")
    return {str(key): item for key, item in value.items()}


class ControllerRuntime:
    def __init__(self, *, python: str | Path | None = None,
                 timeout_seconds: float = 600, max_rpc_calls: int = 10000,
                 max_process_output_bytes: int = 1024 * 1024,
                 max_rpc_event_bytes: int = 8 * 1024 * 1024,
                 sandbox: SandboxBackend | None = None,
                 protected_paths: list[str | Path] | None = None):
        self.python = str(python or sys.executable)
        self.timeout_seconds = float(timeout_seconds)
        self.max_rpc_calls = int(max_rpc_calls)
        self.max_process_output_bytes = int(max_process_output_bytes)
        self.max_rpc_event_bytes = int(max_rpc_event_bytes)
        if min(self.max_process_output_bytes, self.max_rpc_event_bytes) < 1:
            raise ValueError("Controller Runtime byte limits must be positive")
        self.sandbox = sandbox or default_sandbox()
        self.protected_paths = [Path(value).resolve()
                                for value in (protected_paths or [])]
        self.sandbox.require()

    @staticmethod
    def _safe_environment():
        return {"LANG": os.environ.get("LANG", "C.UTF-8"), "PYTHONNOUSERSITE": "1"}

    def execute(self, program_path: str | Path, deployment: RobotDeployment, *,
                execution_kind: str = "physical_trial",
                source_root: str | Path | None = None) -> dict[str, Any]:
        if execution_kind not in {"physical_trial", "diagnostic"}:
            raise ValueError("unsupported execution kind")
        path = Path(program_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        import_root = Path(source_root).resolve() if source_root is not None else path.parent
        try:
            path.relative_to(import_root)
        except ValueError as exc:
            raise ControllerRuntimeError("Controller entrypoint is outside source root") from exc
        with tempfile.TemporaryDirectory(prefix="roboforge-controller-") as temporary:
            process = self.sandbox.popen([self.python, "-u", "-I", "-c", _CHILD,
                str(path), json.dumps(str(deployment.instruction)), str(import_root)], cwd=path.parent,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=0, env=self._safe_environment(),
                read_only_paths=[import_root, Path(self.python).resolve().parents[1]],
                read_write_paths=[temporary], temporary_dir=temporary,
                timeout_seconds=self.timeout_seconds)
            assert process.stdin is not None and process.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            deadline = time.monotonic() + self.timeout_seconds
            events = []; result = None; error = None; completed = False
            stdout_buffer = bytearray(); stderr_buffer = bytearray(); event_bytes = 0
            try:
                while time.monotonic() < deadline:
                    ready = selector.select(max(0.0, deadline - time.monotonic()))
                    if not ready:
                        break
                    for key, _mask in ready:
                        data = os.read(key.fd, 64 * 1024)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        if key.data == "stderr":
                            remaining = self.max_process_output_bytes - len(stderr_buffer)
                            stderr_buffer.extend(data[:max(0, remaining)])
                            if len(data) > max(0, remaining):
                                error = "controller process output exceeded the byte limit"
                                break
                            continue
                        stdout_buffer.extend(data)
                        if len(stdout_buffer) > self.max_process_output_bytes:
                            error = "controller process output exceeded the byte limit"
                            break
                        while b"\n" in stdout_buffer:
                            line, _, tail = stdout_buffer.partition(b"\n")
                            stdout_buffer = bytearray(tail)
                            try:
                                message = json.loads(line.decode("utf-8"))
                            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                                raise ControllerRuntimeError("invalid Controller protocol") from exc
                            if message.get("kind") == "rpc":
                                method = str(message.get("method") or "")
                                if method not in ALLOWED_RPC:
                                    raise ControllerRuntimeError(f"unsupported RPC: {method}")
                                if len(events) >= self.max_rpc_calls:
                                    raise ControllerRuntimeError("RPC budget exceeded")
                                arguments = _rpc_arguments(method, _decode_controller_value(message.get("arguments") or {}))
                                if execution_kind == "diagnostic":
                                    if method == "act":
                                        raise ControllerRuntimeError("diagnostic execution forbids physical act")
                                    if method == "use":
                                        checker = getattr(deployment, "capability_consequence", None)
                                        consequence = checker(str(arguments.get("tool_id") or "")) if callable(checker) else None
                                        if not isinstance(consequence, str) or consequence.upper() != "READ_ONLY":
                                            raise ControllerRuntimeError("diagnostic execution forbids mutating capability")
                                    if method == "sdk":
                                        checker = getattr(deployment, "sdk_consequence", None)
                                        consequence = checker(str(arguments.get("method") or "")) if callable(checker) else None
                                        if not isinstance(consequence, str) or consequence.upper() != "READ_ONLY":
                                            raise ControllerRuntimeError("diagnostic execution forbids mutating Robot SDK method")
                                if method == "sdk":
                                    sdk_method = str(arguments.get("method") or "")
                                    if not sdk_method or sdk_method.startswith("_"):
                                        raise ControllerRuntimeError("invalid Robot SDK method")
                                event = {"method": method,
                                         "arguments": _trace_value(arguments)}
                                capture_state = getattr(deployment, "canonical_embodied_state", None)
                                state_before = capture_state() if method == "act" and callable(capture_state) else None
                                if state_before is not None:
                                    event["state_before"] = state_before
                                try:
                                    raw_result = deployment.dispatch(method, arguments)
                                    projector = getattr(deployment, "project_rpc_output", None)
                                    if not callable(projector):
                                        raise ControllerRuntimeError("Adapter must implement project_rpc_output")
                                    rpc_result = projector(method, arguments, raw_result)
                                    _assert_json(rpc_result)
                                    if method == "use":
                                        entity_projector = getattr(deployment, "project_public_entities", None)
                                        if callable(entity_projector):
                                            entities = entity_projector(str(arguments.get("tool_id") or ""),
                                                                         (rpc_result.get("result")
                                                                          if isinstance(rpc_result, Mapping)
                                                                          else rpc_result))
                                            if entities:
                                                _assert_json(entities)
                                                event["entities"] = entities
                                    if method == "act" and callable(capture_state):
                                        state_after = capture_state()
                                        if state_after is not None:
                                            event["state_after"] = state_after
                                    response = {"id": message.get("id"), "ok": True, "result": rpc_result}
                                    event["result"] = _trace_value(rpc_result)
                                except Exception as exc:
                                    response = {"id": message.get("id"), "ok": False,
                                                "error": f"{type(exc).__name__}: {exc}"}
                                    event["error"] = response["error"]
                                encoded_event = json.dumps(event, default=str,
                                                           separators=(",", ":")).encode()
                                event_bytes += len(encoded_event)
                                if event_bytes > self.max_rpc_event_bytes:
                                    raise ControllerRuntimeError("RPC event log exceeded the byte limit")
                                events.append(event)
                                process.stdin.write((json.dumps(response, separators=(",", ":")) + "\n").encode())
                                process.stdin.flush()
                            elif message.get("kind") == "finished":
                                result = message.get("result"); completed = True; break
                            elif message.get("kind") == "controller_error":
                                error = str(message.get("error") or "controller failed"); break
                            else:
                                raise ControllerRuntimeError("invalid Controller protocol")
                        if error is not None or completed:
                            break
                    if error is not None or completed:
                        break
            finally:
                selector.close()
                self.sandbox.terminate(process)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.sandbox.terminate(process, grace_seconds=0)
                    process.wait(timeout=3)
                stderr = bytes(stderr_buffer).decode("utf-8", errors="replace")
        if not completed and error is None:
            error = "controller timed out" if time.monotonic() >= deadline else "controller exited"
        verified = any(event["method"] in {"verify", "check_observable_condition"}
                       and isinstance(event.get("result"), Mapping)
                       and event["result"].get("verified") is True for event in events)
        return {"completed": completed, "program_sha256": digest, "result": result,
                "execution_kind": execution_kind,
                "error": error, "rpc_events": events, "sensor_verification_observed": verified,
                "stderr": stderr[-2000:],
                "runtime_isolation": f"{self.sandbox.name}-controller-v1",
                "rpc_output_projection": "adapter-owned-v1",
                "rpc_output_validation": "strict-json-v1"}


__all__ = ["ControllerRuntime", "ControllerRuntimeError"]
