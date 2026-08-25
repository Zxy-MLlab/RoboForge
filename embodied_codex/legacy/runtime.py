"""Isolated arbitrary controller-program runtime with a stable direct Robot SDK."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

from ..interfaces import ALLOWED_RPC, RobotDeployment


class ControllerRuntimeError(RuntimeError): pass


_CHILD = r'''
import importlib.util,json,os,sys

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
    sys.path.insert(0,os.path.dirname(os.path.abspath(sys.argv[1])))
    spec=importlib.util.spec_from_file_location("task_controller",sys.argv[1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if not hasattr(module,"run"): raise RuntimeError("controller must define run(robot)")
    result=module.run(Robot(json.loads(sys.argv[2])))
    emit({"kind":"finished","result":result})
except BaseException as exc:
    emit({"kind":"controller_error","error":type(exc).__name__+": "+str(exc)})
'''


_ARGUMENT_KEYS={"observe":{"channel","request"},"act":{"action"},
                "use":{"tool_id","payload"},"verify":{"verifier","payload"},
                "record":{"event"}}

_EVALUATOR_ONLY_KEYS={"reward","done","success","is_success","task_success",
    "check_success","goal_achieved","predicate_satisfied","bddl","bddl_goal_predicates",
    "object_pose","object_poses","privileged_object_poses","simulator_state",
    "simulator_internal_state","initial_state_index","episode_id","state_id",
    "evaluation_episode_identifier","termination_reason"}


def _assert_public_rpc_output(value: Any,path: str="result"):
    """Reject evaluator leakage after the Adapter's positive projection."""
    if isinstance(value,Mapping):
        for key,item in value.items():
            normalized=str(key).casefold().replace("-","_")
            if normalized in _EVALUATOR_ONLY_KEYS:
                raise ControllerRuntimeError(f"forbidden evaluator field in RPC output: {path}.{key}")
            _assert_public_rpc_output(item,f"{path}.{key}")
    elif isinstance(value,(list,tuple)):
        for index,item in enumerate(value):_assert_public_rpc_output(item,f"{path}[{index}]")
    try:json.dumps(value,separators=(",",":"),allow_nan=False)
    except (TypeError,ValueError) as exc:
        raise ControllerRuntimeError(f"RPC output is not strict JSON at {path}") from exc


def _rpc_arguments(method: str,value: Any):
    """Validate the controller-to-Adapter RPC envelope by positive allowlist."""
    if not isinstance(value,Mapping):raise ControllerRuntimeError("RPC arguments must be an object")
    unknown=set(str(key) for key in value)-_ARGUMENT_KEYS[method]
    if unknown:raise ControllerRuntimeError(f"unsupported {method} argument fields: {sorted(unknown)}")
    return {str(key):item for key,item in value.items()}


class ControllerRuntime:
    def __init__(self, *, python: str | Path | None = None,
                 timeout_seconds: float = 600, max_rpc_calls: int = 10000) -> None:
        self.python = str(python or sys.executable)
        self.timeout_seconds = float(timeout_seconds); self.max_rpc_calls = int(max_rpc_calls)

    @staticmethod
    def _system_binds():
        args=[]
        for value in ("/usr","/bin","/lib","/lib64","/etc"):
            if Path(value).exists():args.extend(["--ro-bind",value,value])
        return args

    def _isolated_command(self,path,deployment):
        bwrap=shutil.which("bwrap")
        if not bwrap:raise ControllerRuntimeError("bubblewrap is required for controller isolation")
        executable=Path(self.python).resolve();prefix=executable.parents[1]
        runtime_executable=Path("/runtime")/executable.relative_to(prefix)
        workspace=path.parent.resolve();sandbox_path=Path("/workspace")/path.name
        command=[bwrap,"--die-with-parent","--new-session","--unshare-pid",
                 "--unshare-ipc","--unshare-uts","--unshare-net",*self._system_binds(),
                 "--ro-bind",str(prefix),"/runtime","--dev","/dev","--proc","/proc",
                 "--tmpfs","/tmp","--ro-bind",str(workspace),"/workspace",
                 "--chdir","/workspace","--",str(runtime_executable),"-u","-I","-c",
                 _CHILD,str(sandbox_path),json.dumps(str(deployment.instruction))]
        return command

    @staticmethod
    def _safe_environment():
        return {"LANG":os.environ.get("LANG","C.UTF-8"),"PYTHONNOUSERSITE":"1"}

    def execute(self, program_path: str | Path, deployment: RobotDeployment) -> dict[str, Any]:
        path = Path(program_path).resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        process = subprocess.Popen(
            self._isolated_command(path,deployment),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,env=self._safe_environment(),
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
                    arguments = _rpc_arguments(method,message.get("arguments") or {})
                    event = {"method": method, "arguments": arguments}
                    try:
                        raw_result=deployment.dispatch(method,arguments)
                        projector=getattr(deployment,"project_rpc_output",None)
                        if not callable(projector):
                            raise ControllerRuntimeError(
                                "Robot Adapter must implement explicit project_rpc_output(method, arguments, result)")
                        rpc_result=projector(method,arguments,raw_result)
                        _assert_public_rpc_output(rpc_result)
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
                    result = message.get("result"); completed = True; break
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
                "sensor_verification_observed": verified, "stderr": stderr[-2000:],
                "runtime_isolation":"bubblewrap-controller-v1",
                "rpc_output_projection":"adapter-explicit-allowlist-v1",
                "rpc_output_defense":"kernel-evaluator-field-deny-v1"}

__all__ = ["ControllerRuntime", "ControllerRuntimeError"]
