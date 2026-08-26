"""Isolated runtime for model-authored analytic Tools.

Tools are untrusted acquired code.  They never execute in the Harness process:
the runtime exposes only the immutable Tool bundle, the Python environment, and
explicit input files named in the JSON payload.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, Mapping

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
                 sandbox: SandboxBackend|None=None):
        self.python=str(python or sys.executable)
        self.timeout_seconds=float(timeout_seconds)
        self.sandbox=sandbox or default_sandbox()
        self.sandbox.require()

    @staticmethod
    def _authorized_file(value: str,
                         artifact_resolver: Callable[[str], str | Path] | None):
        if not value.startswith("artifact://"):
            if Path(value).is_absolute():
                raise ToolRuntimeError(
                    "Tool file inputs must use an opaque registered artifact handle")
            return None
        if not callable(artifact_resolver):
            raise ToolRuntimeError("Tool artifact resolver is unavailable")
        try:
            resolved=Path(artifact_resolver(value)).resolve()
        except Exception as exc:
            raise ToolRuntimeError("Tool artifact handle is not authorized") from exc
        if not resolved.is_file():
            raise ToolRuntimeError("Tool artifact handle does not resolve to a file")
        return resolved

    def _rewrite_payload(self,value: Any,bindings: set[Path],
                         artifact_resolver: Callable[[str], str | Path] | None,
                         staging: Path, staged: dict[Path, Path]):
        if isinstance(value,Mapping):
            return {str(key):self._rewrite_payload(
                        item,bindings,artifact_resolver,staging,staged)
                    for key,item in value.items()}
        if isinstance(value,list):
            return [self._rewrite_payload(item,bindings,artifact_resolver,staging,staged)
                    for item in value]
        if isinstance(value,tuple):
            return [self._rewrite_payload(item,bindings,artifact_resolver,staging,staged)
                    for item in value]
        if isinstance(value,str):
            source=self._authorized_file(value,artifact_resolver)
            if source is not None:
                destination=staged.get(source)
                if destination is None:
                    destination=staging/f"input-{len(staged)+1:04d}{source.suffix}"
                    shutil.copyfile(source,destination)
                    destination.chmod(0o400)
                    staged[source]=destination
                bindings.add(destination)
                return str(destination)
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
                python: str | Path | None = None,
                artifact_resolver: Callable[[str], str | Path] | None = None):
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
        timeout=min(max(float(runtime_spec.get("timeout_seconds",
            self.timeout_seconds)),0.1),600)
        runtime_python = str(python or self.python)
        with tempfile.TemporaryDirectory(prefix="roboforge-tool-input-") as temporary:
            staging=Path(temporary).resolve()
            bindings:set[Path]=set();rewritten=self._rewrite_payload(
                dict(payload),bindings,artifact_resolver,staging,{})
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
