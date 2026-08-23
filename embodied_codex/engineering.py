"""Full coding surface plus one evaluator-blind robot experiment per iteration."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any, Callable

from .assets import CapabilityLibrary
from .registry import FunctionRegistry
from .runtime import ControllerRuntime
from .web import fetch_web_page,search_web
from .workspace import TaskWorkspace


def _obj(properties, required):
    return {"type":"object","properties":properties,"required":required,
            "additionalProperties":False}


def _result_shape(value,depth=0):
    """Describe large Tool output without replaying sensor arrays into context."""
    if depth>=3:return type(value).__name__
    if isinstance(value,dict):
        return {str(key):_result_shape(item,depth+1) for key,item in value.items()
                if str(key) not in {"frame","cameras","provenance"}}
    if isinstance(value,list):
        return {"type":"list","length":len(value),
                "item":_result_shape(value[0],depth+1) if value else None}
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    return type(value).__name__


def _agent_robot_summary(result,detail_path):
    """Compact evidence returned through the model tool channel.

    The complete immutable execution remains in robot_execution.json.  This
    summary keeps actionable motion receipts and sensor decisions while
    replacing multi-megabyte perception / grasp payloads with output shapes.
    """
    execution=result.get("execution") or {};events=[]
    for event in execution.get("rpc_events") or []:
        method=event.get("method");item={"method":method}
        arguments=event.get("arguments") or {};rpc_result=event.get("result")
        if method=="act":
            item["action"]=arguments.get("action");item["result"]=rpc_result
        elif method=="verify":
            item["verifier"]=arguments.get("verifier")
            item["result"]={key:value for key,value in (rpc_result or {}).items()
                            if key not in {"frame","cameras"}} \
                if isinstance(rpc_result,dict) else rpc_result
        elif method=="record":item["event"]=arguments.get("event")
        elif method=="observe":
            cameras=(rpc_result or {}).get("cameras") or {} if isinstance(rpc_result,dict) else {}
            item["result"]={"frame_id":(rpc_result or {}).get("frame_id"),
                            "step":(rpc_result or {}).get("step"),
                            "rgb_paths":{name:camera.get("rgb_path") for name,camera in cameras.items()
                                         if isinstance(camera,dict) and camera.get("rgb_path")}}
        elif method=="use":
            item["tool_id"]=arguments.get("tool_id")
            direct=(rpc_result or {}).get("result") if isinstance(rpc_result,dict) else rpc_result
            item["output_shape"]=_result_shape(direct)
        if event.get("error"):item["error"]=event["error"]
        events.append(item)
    report={key:value for key,value in (result.get("sensor_report") or {}).items()
            if not str(key).startswith("_harness_")}
    return {"controller_path":result.get("controller_path"),
            "controller_snapshot":result.get("controller_snapshot"),
            "completed":execution.get("completed"),"error":execution.get("error"),
            "controller_result":execution.get("result"),"rpc_evidence":events,
            "sensor_report":report,
            "sensor_success_candidate":result.get("sensor_success_candidate"),
            "robot_contract_preflight":result.get("robot_contract_preflight"),
            "task_model_preflight":result.get("task_model_preflight"),
            "full_execution_artifact":str(Path(detail_path).resolve()),
            "execution_artifact_ref":"latest_robot_execution"}


class EngineeringSurface:
    def __init__(self, *, workspace: TaskWorkspace, capabilities: CapabilityLibrary,
                 runtime: ControllerRuntime, deployment_factory: Callable[[], Any],
                 artifact_dir: str | Path, task_model: dict|None=None,
                 semantic_reviewer: Callable[...,dict]|None=None,
                 allowed_action_types: list[str]|None=None,
                 sdk_contract: dict|None=None,
                 active_deployment_tool_ids: list[str]|None=None,
                 execution_observer: Callable[[dict],None]|None=None):
        self.workspace=workspace; self.capabilities=capabilities; self.runtime=runtime
        self.deployment_factory=deployment_factory; self.artifact_dir=Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True,exist_ok=True)
        # Sensor artifacts and rollouts for this task live below the run root.
        # The model may inspect those files, but it cannot use this surface to
        # read arbitrary host images.
        self.run_root=self.artifact_dir.resolve().parents[1]
        self.robot_runs=0; self.last_execution=None
        self.task_model=dict(task_model) if task_model else None
        self.semantic_reviewer=semantic_reviewer
        self.sdk_contract=dict(sdk_contract or {})
        self.active_deployment_tool_ids=set(str(x) for x in
                                            (active_deployment_tool_ids or []))
        self.execution_observer=execution_observer
        contracted=(self.sdk_contract.get("actions") or {}).keys()
        self.allowed_action_types=set(str(x) for x in (allowed_action_types or contracted))

    def list_available_tools(self):
        rows=self.capabilities.list_summaries()
        if not self.active_deployment_tool_ids:return rows
        return [row for row in rows if not row.get("execution_owned_by_deployment")
                or row.get("tool_id") in self.active_deployment_tool_ids]

    def _lint_robot_contract(self,controller: Path):
        """Compile-time checks for literal Robot SDK contract violations.

        Dynamic action selection remains legal.  Literal misspellings and null
        opaque references are unambiguously invalid and must not consume a
        physical episode merely to discover a typed-interface error.
        """
        tree=ast.parse(controller.read_text());issues=[]
        for node in ast.walk(tree):
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute):continue
            if not isinstance(node.func.value,ast.Name) or node.func.value.id!="robot":continue
            if node.func.attr not in {"act","verify"} or len(node.args)<1:continue
            payload=node.args[0] if node.func.attr=="act" else (node.args[1] if len(node.args)>1 else None)
            if not isinstance(payload,ast.Dict):continue
            values={key.value:value for key,value in zip(payload.keys,payload.values)
                    if isinstance(key,ast.Constant) and isinstance(key.value,str)}
            if node.func.attr=="act":
                action_type=values.get("type")
                if (self.allowed_action_types and isinstance(action_type,ast.Constant)
                        and isinstance(action_type.value,str)
                        and action_type.value not in self.allowed_action_types):
                    issues.append(f"line {node.lineno}: unsupported literal action type "
                                  f"{action_type.value!r}; allowed={sorted(self.allowed_action_types)}")
                if (isinstance(action_type,ast.Constant) and isinstance(action_type.value,str)
                        and action_type.value in (self.sdk_contract.get("actions") or {})):
                    required=self.sdk_contract["actions"][action_type.value].get("required") or []
                    missing=[key for key in required if key not in values]
                    if missing:
                        issues.append(f"line {node.lineno}: action {action_type.value!r} "
                                      f"missing literal fields {missing}")
                # Opaque motion references are minted only by live Adapter
                # Tool results.  Any source-code string is necessarily a
                # fabricated reference, even if it happens to look like one.
                for key in ("target_ref","pose_ref"):
                    value=values.get(key)
                    if isinstance(value,ast.Constant) and isinstance(value.value,str):
                        issues.append(f"line {node.lineno}: action {key} cannot be a "
                                      "literal; use an Adapter-issued live reference")
            else:
                verifier=node.args[0] if node.args else None
                verifier_name=(verifier.value if isinstance(verifier,ast.Constant)
                               and isinstance(verifier.value,str) else None)
                verifier_contracts=self.sdk_contract.get("verifiers") or {}
                if verifier_name and verifier_contracts and verifier_name not in verifier_contracts:
                    issues.append(f"line {node.lineno}: unknown literal verifier "
                                  f"{verifier_name!r}; allowed={sorted(verifier_contracts)}")
                if verifier_name in verifier_contracts:
                    required=verifier_contracts[verifier_name].get("required") or []
                    missing=[key for key in required if key not in values]
                    if missing:
                        issues.append(f"line {node.lineno}: verifier {verifier_name!r} "
                                      f"missing literal fields {missing}")
                for key in ("source_ref","target_ref"):
                    value=values.get(key)
                    if isinstance(value,ast.Constant) and value.value is None:
                        issues.append(f"line {node.lineno}: verifier {key} cannot be literal None")
                    if isinstance(value,ast.Constant) and isinstance(value.value,str):
                        issues.append(f"line {node.lineno}: verifier {key} cannot be a "
                                      "literal; use an Adapter-issued live reference")
        if issues:raise RuntimeError("controller Robot SDK preflight failed: "+"; ".join(issues))
        return {"passed":True,"contract_protocol":self.sdk_contract.get("protocol"),
                "allowed_action_types":sorted(self.allowed_action_types)}

    @staticmethod
    def _controller_call_graph(source: str):
        """Return top-level functions, local call edges, and Robot SDK calls."""
        tree=ast.parse(source);functions={}
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                functions[node.name]=node
        edges={name:set() for name in functions};operations={name:set() for name in functions}
        for name,node in functions.items():
            for child in ast.walk(node):
                if not isinstance(child,ast.Call):continue
                function=child.func
                if isinstance(function,ast.Name) and function.id in functions:
                    edges[name].add(function.id)
                if (isinstance(function,ast.Attribute) and
                        isinstance(function.value,ast.Name) and function.value.id=="robot" and
                        function.attr in {"observe","use","act","verify","record"}):
                    operations[name].add(function.attr)
        return functions,edges,operations

    def preflight_controller(self,path: str,phase_bindings: list[dict]):
        """Bind every causal task phase to reachable executable controller code."""
        if self.task_model is None:
            return {"eligible":True,"task_model_required":False}
        controller=self.workspace._path(path);source=controller.read_text()
        functions,edges,operations=self._controller_call_graph(source)
        if "run" not in functions:raise RuntimeError("controller must define run(robot)")
        if not isinstance(phase_bindings,list):raise RuntimeError("phase_bindings must be a list")
        expected={str(p["id"]):p for p in self.task_model.get("phases") or []}
        supplied={}
        for row in phase_bindings:
            if not isinstance(row,dict) or not isinstance(row.get("phase_id"),str):
                raise RuntimeError("every binding needs phase_id")
            if row["phase_id"] in supplied:raise RuntimeError("duplicate phase binding")
            names=row.get("functions");declared=row.get("robot_operations")
            if not isinstance(names,list) or not names or not all(isinstance(x,str) for x in names):
                raise RuntimeError(f"phase {row['phase_id']} needs bound functions")
            if not isinstance(declared,list) or not all(isinstance(x,str) for x in declared):
                raise RuntimeError(f"phase {row['phase_id']} needs robot_operations")
            supplied[row["phase_id"]]=dict(row)
        if set(supplied)!=set(expected):
            raise RuntimeError(f"phase binding mismatch; missing={sorted(set(expected)-set(supplied))}, "
                               f"extra={sorted(set(supplied)-set(expected))}")
        reachable=set();pending=["run"]
        while pending:
            name=pending.pop()
            if name in reachable:continue
            reachable.add(name);pending.extend(edges.get(name,()))
        receipts=[]
        for phase_id,phase in expected.items():
            row=supplied[phase_id];names=row["functions"]
            missing=[name for name in names if name not in functions]
            if missing:raise RuntimeError(f"phase {phase_id} binds unknown functions: {missing}")
            dead=[name for name in names if name not in reachable]
            if dead:raise RuntimeError(f"phase {phase_id} binds unreachable functions: {dead}")
            closure=set(names);front=list(names)
            while front:
                for child in edges.get(front.pop(),()):
                    if child not in closure:closure.add(child);front.append(child)
            actual=set().union(*(operations.get(name,set()) for name in closure))
            required=set(phase.get("required_robot_operations") or [])
            declared=set(row["robot_operations"])
            if not required.issubset(declared):
                raise RuntimeError(f"phase {phase_id} binding omits required operations: "
                                   f"{sorted(required-declared)}")
            if not declared.issubset(actual):
                raise RuntimeError(f"phase {phase_id} claims absent operations: "
                                   f"{sorted(declared-actual)}")
            receipts.append({"phase_id":phase_id,"functions":names,
                             "required_robot_operations":sorted(required),
                             "observed_robot_operations":sorted(actual)})
        controller_sha=hashlib.sha256(controller.read_bytes()).hexdigest()
        task_sha=str(self.task_model.get("task_model_sha256") or "")
        binding={"protocol":"embodied-codex-controller-task-binding-v1",
                 "controller_path":path,"controller_sha256":controller_sha,
                 "task_model_sha256":task_sha,"phase_bindings":phase_bindings,
                 "static_receipts":receipts}
        if self.semantic_reviewer is not None:
            review=dict(self.semantic_reviewer(task_model=self.task_model,source=source,
                                               binding=binding))
            binding["semantic_review"]=review
            if review.get("approved") is not True:
                issues=review.get("issues") or ["semantic coverage rejected"]
                raise RuntimeError("controller semantic preflight rejected: "+"; ".join(map(str,issues)))
        else:
            binding["semantic_review"]={"approved":True,"source":"previously frozen review"}
        self.workspace.write_file("task_plan_binding.json",json.dumps(binding,indent=2)+"\n")
        return {"eligible":True,"controller_sha256":controller_sha,
                "task_model_sha256":task_sha,"phases":receipts,
                "semantic_review":binding["semantic_review"]}

    def _require_current_preflight(self,controller: Path):
        if self.task_model is None:return None
        path=self.workspace._path("task_plan_binding.json")
        if not path.is_file():raise RuntimeError("task-model controller preflight is required")
        binding=json.loads(path.read_text())
        digest=hashlib.sha256(controller.read_bytes()).hexdigest()
        if binding.get("controller_sha256")!=digest:
            raise RuntimeError("controller changed after task-model preflight")
        if binding.get("task_model_sha256")!=self.task_model.get("task_model_sha256"):
            raise RuntimeError("task model changed after controller preflight")
        if (binding.get("semantic_review") or {}).get("approved") is not True:
            raise RuntimeError("controller semantic review is not approved")
        return binding

    def view_sensor_image(self, path: str):
        target=Path(path)
        if not target.is_absolute():
            workspace_candidate=(self.workspace.root/target).resolve()
            target=workspace_candidate if workspace_candidate.is_file() else self.run_root/target
        target=target.resolve()
        if target != self.run_root and self.run_root not in target.parents:
            raise RuntimeError("image must be inside the current run")
        mime=mimetypes.guess_type(target.name)[0]
        if mime not in ("image/png","image/jpeg","image/webp"):
            raise RuntimeError("view_sensor_image supports PNG, JPEG, or WEBP")
        if not target.is_file(): raise FileNotFoundError(path)
        if target.stat().st_size > 8*1024*1024: raise RuntimeError("image exceeds 8 MiB")
        return {"_embodied_codex_image":{
            "path":str(target),"mime_type":mime,
            "data_base64":base64.b64encode(target.read_bytes()).decode("ascii")}}

    def read_file(self,path: str,start_line: int=1,end_line: int=400):
        """Read workspace source or an immutable text artifact in this run.

        Evidence gives the model absolute controller-snapshot paths.  Requiring
        it to mentally translate those into a different Tool/path namespace is
        needless interface friction.  Writes remain strictly workspace-relative;
        absolute reads are accepted only below the current run root.
        """
        requested=Path(str(path))
        if not requested.is_absolute():
            return self.workspace.read_file(str(path),start_line,end_line)
        target=requested.resolve()
        if target!=self.run_root and self.run_root not in target.parents:
            raise RuntimeError("absolute read path must be inside the current run")
        if not target.exists():
            return {"path":str(target),"exists":False,"start_line":1,"end_line":0,
                    "total_lines":0,"content":""}
        if not target.is_file() or target.stat().st_size>4*1024*1024:
            raise RuntimeError("run artifact is not a readable text file")
        try:lines=target.read_text().splitlines()
        except UnicodeDecodeError as exc:raise RuntimeError("run artifact is not text") from exc
        start=max(1,int(start_line));end=max(start,min(int(end_line),start+999))
        return {"path":str(target),"exists":True,"start_line":start,
                "end_line":min(end,len(lines)),"total_lines":len(lines),
                "content":"\n".join(lines[start-1:end])}

    def read_run_artifact(self,path: str,start_line: int=1,end_line: int=400):
        requested=str(path)
        if requested=="latest_robot_execution":
            current=self.artifact_dir/"robot_execution.json"
            prior=sorted(self.run_root.glob("iterations/iteration_*/robot_execution.json"))
            target=current if current.is_file() else (prior[-1] if prior else current)
        elif requested=="robot_execution.json":
            target=self.artifact_dir/"robot_execution.json"
        else:target=Path(requested)
        if not target.is_absolute(): target=self.run_root/target
        target=target.resolve()
        if target != self.run_root and self.run_root not in target.parents:
            raise RuntimeError("artifact must be inside the current run")
        if not target.is_file():raise FileNotFoundError(path)
        start=max(1,int(start_line));end=max(start,min(int(end_line),start+999))
        selected=[];total=0;returned_end=start-1;characters=0;truncated=False
        with target.open(errors="replace") as stream:
            for total,line in enumerate(stream,start=1):
                if total<start or total>end:continue
                text=line.rstrip("\r\n")
                remaining=256*1024-characters
                if remaining<=0:truncated=True;continue
                if len(text)>remaining:
                    selected.append(text[:remaining]);characters+=remaining;truncated=True
                else:
                    selected.append(text);characters+=len(text)+1
                returned_end=total
        return {"path":str(target),"start_line":start,"end_line":returned_end,
                "total_lines":total,"content":"\n".join(selected),
                "content_truncated":truncated,
                "next_start_line":returned_end+1 if (truncated or returned_end<total) else None}

    def list_sensor_artifacts(self, pattern: str="episodes/**/*"):
        pattern=str(pattern or "episodes/**/*")
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise RuntimeError("artifact pattern must stay inside the current run")
        items=[]
        for path in self.run_root.glob(pattern):
            if path.is_file(): items.append(str(path.resolve()))
        return sorted(items)[-2000:]

    def run_robot_controller(self, path: str):
        if self.robot_runs: raise RuntimeError("one robot episode per iteration")
        controller=self.workspace._path(path)
        contract_lint=self._lint_robot_contract(controller)
        preflight=self._require_current_preflight(controller)
        # Preserve the exact program that produced this physical episode before
        # any later iteration can rewrite the persistent workspace.  The SHA in
        # ControllerRuntime proves the snapshot corresponds to the executed
        # source, while the source itself lets the agent recover and compose
        # previously successful strategies instead of reconstructing them from
        # action traces.
        controller_snapshot=self.artifact_dir/"controller.py"
        controller_snapshot.write_bytes(controller.read_bytes())
        self.robot_runs += 1; deployment=self.deployment_factory()
        try:
            register=getattr(deployment,"register_capability",None)
            functions=self.capabilities.runtime_functions()
            if functions and not callable(register):
                raise RuntimeError("deployment lacks register_capability")
            for tool_id,function in functions.items(): register(tool_id,function)
            execution=self.runtime.execute(controller,deployment)
            sensor_report=dict(deployment.sensor_report(execution))
        finally: deployment.close()
        final_verify=(execution["rpc_events"][-1] if execution["rpc_events"] else {})
        claimed=(isinstance(execution.get("result"),dict)
                 and execution["result"].get("status")=="sensor_success")
        passed=(execution.get("completed") is True and claimed
                and final_verify.get("method")=="verify"
                and isinstance(final_verify.get("result"),dict)
                and final_verify["result"].get("verified") is True
                and sensor_report.get("sensor_verification_passed") is True)
        result={"controller_path":path,"controller_snapshot":str(controller_snapshot.resolve()),
                "execution":execution,"sensor_report":sensor_report,
                "sensor_success_candidate":passed,"task_model_preflight":preflight,
                "robot_contract_preflight":contract_lint}
        self.last_execution=result
        detail_path=self.artifact_dir/"robot_execution.json"
        temporary=detail_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result,indent=2,default=str)+"\n")
        temporary.replace(detail_path)
        # Persist the physical episode before returning control to the coding
        # model.  A later API timeout or process interruption must never erase
        # an already completed robot experiment or cause it to be repeated.
        if self.execution_observer is not None:self.execution_observer(result)
        return _agent_robot_summary(result,detail_path)

    def registry(self):
        r=FunctionRegistry(); string={"type":"string"}; integer={"type":"integer"}
        free={"type":"object","additionalProperties":True}
        r.add("list_files","List files in the persistent task workspace.",
              _obj({"pattern":string},["pattern"]),self.workspace.list_files)
        r.add("read_file","Read a relative workspace file or an absolute immutable text artifact inside this run.",
              _obj({"path":string,"start_line":integer,"end_line":integer},["path","start_line","end_line"]),
              self.read_file)
        r.add("write_file","Create or fully rewrite any workspace file.",
              _obj({"path":string,"content":string},["path","content"]),self.workspace.write_file)
        r.add("replace_in_file","Replace one exact region in a workspace file.",
              _obj({"path":string,"old":string,"new":string},["path","old","new"]),
              self.workspace.replace_in_file)
        r.add("run_command","Run an argv command in the workspace; use it for tests and engineering.",
              _obj({"argv":{"type":"array","items":string},"timeout_seconds":{"type":"number"}},
                   ["argv","timeout_seconds"]),self.workspace.run_command)
        r.add("search_web","Search public internet resources and repositories.",
              _obj({"query":string,"limit":integer},["query","limit"]),search_web)
        r.add("fetch_web_page","Read a public HTTP(S) page returned by web search.",
              _obj({"url":string,"max_chars":integer},["url","max_chars"]),fetch_web_page)
        r.add("view_sensor_image","Visually inspect a PNG/JPEG/WEBP sensor frame or mask from this run.",
              _obj({"path":string},["path"]),self.view_sensor_image)
        r.add("list_sensor_artifacts","List sensor frames, masks, traces, and rollouts in this run.",
              _obj({"pattern":string},["pattern"]),self.list_sensor_artifacts)
        r.add("read_run_artifact","Read a text/JSON/log artifact from this run with line numbers; "
              "use path=latest_robot_execution for the current complete robot evidence.",
              _obj({"path":string,"start_line":integer,"end_line":integer},
                   ["path","start_line","end_line"]),self.read_run_artifact)
        r.add("register_tool","Freeze a workspace implementation as a versioned Tool.",
              _obj({"name":string,"source_path":string,"description":string,
                    "input_schema":free,"output_schema":free,
                    "source_urls":{"type":"array","items":string},
                    "trained_on_current_task":{"type":"boolean"}},
                   ["name","source_path","description","input_schema","output_schema",
                    "source_urls","trained_on_current_task"]),self.capabilities.register_tool)
        test_case={"type":"object","properties":{"input":free,"expected":{}},
                   "required":["input","expected"],"additionalProperties":False}
        r.add("test_tool","Run deterministic exact-output cases formatted as "
              "{input: <payload passed to run(payload)>, expected: <exact return value>}; "
              "only Tools whose every case passes become deployable.",
              _obj({"tool_id":string,"cases":{"type":"array","items":test_case}},["tool_id","cases"]),
              self.capabilities.test_tool)
        r.add("list_tools","List concise Tool contracts and status; only status=tested is deployable. "
              "Use inspect_tool only when source, provenance, or test details are needed.",
              _obj({},[]),self.list_available_tools)
        r.add("inspect_tool","Inspect immutable Tool source and manifest.",
              _obj({"tool_id":string},["tool_id"]),self.capabilities.inspect)
        if self.task_model is not None:
            binding_item={"type":"object","properties":{
                "phase_id":string,"functions":{"type":"array","items":string},
                "robot_operations":{"type":"array","items":{
                    "type":"string","enum":["observe","use","act","verify","record"]}}},
                "required":["phase_id","functions","robot_operations"],
                "additionalProperties":False}
            r.add("preflight_controller",
                  "Statically and semantically prove that every immutable task-model phase "
                  "is implemented by reachable controller code before starting the robot.",
                  _obj({"path":string,"phase_bindings":{"type":"array","items":binding_item}},
                       ["path","phase_bindings"]),self.preflight_controller)
        r.add("run_robot_controller","Run one arbitrary controller.py in a fresh sensor-only episode.",
              _obj({"path":string},["path"]),self.run_robot_controller,
              available=lambda:self.robot_runs==0)
        return r

__all__ = ["EngineeringSurface"]
