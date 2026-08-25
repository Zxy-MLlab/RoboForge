"""Isolated runtime for model-authored analytic Tools.

Tools are untrusted acquired code.  They never execute in the Harness process:
the runtime exposes only the immutable Tool bundle, the Python environment, and
explicit input files named in the JSON payload.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from .kernel.sandbox import SandboxBackend, default_sandbox


class ToolRuntimeError(RuntimeError): pass


_CHILD = r'''
import contextlib,importlib.util,importlib.metadata,io,json,os,sys,traceback
sys.dont_write_bytecode=True
try:
    payload=json.load(sys.stdin)
    with open(sys.argv[2]) as stream:manifest=json.load(stream)
    requirements=(manifest.get("runtime_spec") or {}).get("runtime_requirements")
    if requirements is None:
        requirements=(manifest.get("dependencies") or {}).get("runtime_requirements",[])
    for requirement in requirements:
        name,expected=requirement.split("==",1)
        actual=importlib.metadata.version(name)
        if actual!=expected:raise RuntimeError(
            "runtime dependency mismatch for %s: expected %s, got %s"%(name,expected,actual))
    if os.path.isdir("/tool/vendor"):sys.path.insert(0,"/tool/vendor")
    entrypoint=sys.argv[1]
    bundle=os.path.dirname(entrypoint)
    if bundle not in sys.path:sys.path.insert(0,bundle)
    bundled_vendor=os.path.join(bundle,"vendor")
    if os.path.isdir(bundled_vendor):sys.path.insert(0,bundled_vendor)
    spec=importlib.util.spec_from_file_location("acquired_tool",entrypoint)
    if spec is None or spec.loader is None: raise RuntimeError("Tool load failed")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if not hasattr(module,"run"): raise RuntimeError("Tool must define run(payload)")
    logs=io.StringIO()
    with contextlib.redirect_stdout(logs),contextlib.redirect_stderr(logs):
        result=module.run(payload)
    print(json.dumps({"ok":True,"result":result,"logs":logs.getvalue()[-4000:]},
                     separators=(",",":"),default=str))
except BaseException as exc:
    print(json.dumps({"ok":False,"error":type(exc).__name__+": "+str(exc),
                      "traceback":traceback.format_exc()[-4000:]},separators=(",",":")))
'''


class ToolRuntime:
    def __init__(self, *, python: str|Path|None=None, timeout_seconds: float=120,
                 allowed_input_roots: list[str|Path]|None=None,
                 sandbox: SandboxBackend|None=None):
        self.python=str(python or sys.executable)
        self.timeout_seconds=float(timeout_seconds)
        self.allowed_input_roots=[Path(value).resolve() for value in (allowed_input_roots or [])]
        self.sandbox=sandbox or default_sandbox()
        self.sandbox.require()

    def _authorized_file(self,value: str):
        candidate=Path(value)
        if not candidate.is_absolute() or not candidate.exists():return None
        resolved=candidate.resolve()
        if not resolved.is_file():
            raise ToolRuntimeError("Tool inputs may expose files, not host directories")
        if not any(resolved==root or root in resolved.parents for root in self.allowed_input_roots):
            raise ToolRuntimeError("Tool input file is outside the sensor-evidence roots")
        return resolved

    def _rewrite_payload(self,value: Any,bindings: set[Path]):
        if isinstance(value,Mapping):
            return {str(key):self._rewrite_payload(item,bindings) for key,item in value.items()}
        if isinstance(value,list):return [self._rewrite_payload(item,bindings) for item in value]
        if isinstance(value,tuple):return [self._rewrite_payload(item,bindings) for item in value]
        if isinstance(value,str):
            source=self._authorized_file(value)
            if source is not None:
                bindings.add(source)
                return str(source)
        return value

    @staticmethod
    def _gpu_device_binds():
        bindings=[]
        for source in sorted(Path("/dev").glob("nvidia*")):
            bindings.extend(["--dev-bind",str(source),str(source)])
        return bindings

    @staticmethod
    def _safe_environment(accelerator: str):
        allowed=("LANG","LC_ALL","LC_CTYPE","CUDA_VISIBLE_DEVICES",
                 "NVIDIA_VISIBLE_DEVICES","CUDA_HOME","LD_LIBRARY_PATH")
        result={key:os.environ[key] for key in allowed if key in os.environ}
        result["PYTHONNOUSERSITE"]="1"
        if accelerator!="cuda":
            result.pop("CUDA_VISIBLE_DEVICES",None)
            result.pop("NVIDIA_VISIBLE_DEVICES",None)
        return result

    def execute(self, tool_dir: str|Path, payload: Mapping[str,Any], *,
                python: str | Path | None = None):
        directory=Path(tool_dir).resolve()
        manifest_path=directory/"manifest.json"
        manifest=json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        runtime_spec=dict(manifest.get("runtime_spec") or {})
        if runtime_spec:
            relative=Path(str(runtime_spec.get("entrypoint") or ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise ToolRuntimeError("invalid capability package entrypoint")
            entrypoint=directory/"bundle"/relative
        else:entrypoint=directory/"tool.py"
        if not entrypoint.is_file():raise ToolRuntimeError("missing Tool entrypoint")
        bindings:set[Path]=set();rewritten=self._rewrite_payload(dict(payload),bindings)
        timeout=min(max(float(runtime_spec.get("timeout_seconds",
            self.timeout_seconds)),0.1),600)
        runtime_python = str(python or self.python)
        completed=self.sandbox.run([runtime_python,"-u","-I","-c",_CHILD,
            str(entrypoint),str(manifest_path)],cwd=directory,
            input_text=json.dumps(rewritten),
            env=self._safe_environment(str(runtime_spec.get("accelerator") or "cpu")),
            read_only_paths=[directory,*bindings,Path(runtime_python).resolve().parents[1]],
            timeout_seconds=timeout)
        if completed.timed_out:
            raise ToolRuntimeError("Tool execution timed out")
        if completed.returncode!=0:
            raise ToolRuntimeError(f"Tool sandbox failed: {completed.stderr[-2000:]}")
        lines=[line for line in completed.stdout.splitlines() if line.strip()]
        if len(lines)!=1:
            raise ToolRuntimeError("Tool returned an invalid runtime envelope")
        try:response=json.loads(lines[0])
        except json.JSONDecodeError as exc:raise ToolRuntimeError("Tool returned invalid JSON") from exc
        if response.get("ok") is not True:
            raise ToolRuntimeError(str(response.get("error") or "Tool execution failed"))
        return response.get("result")


__all__=["ToolRuntime","ToolRuntimeError"]
