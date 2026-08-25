"""Full coding surface plus one evaluator-blind robot experiment per iteration."""
from __future__ import annotations

import ast
import base64
import builtins
import dis
import hashlib
import io
import json
import mimetypes
from pathlib import Path
import time
import symtable
import tokenize
import types
from typing import Any, Callable

import cv2

from ..assets import (CapabilityLibrary, ExperienceLibrary, SkillLibrary,
                     CapabilityGapLibrary, AssetError,
                     execution_evidence_assessment, bind_authoritative_validation)
from .registry import FunctionRegistry
from ..runtime import ControllerRuntime
from ..web import download_public_file,fetch_web_page,search_web
from ..workspace import TaskWorkspace


def _obj(properties, required):
    return {"type":"object","properties":properties,"required":required,
            "additionalProperties":False}


def _is_robot_record(node):
    return (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and node.func.attr=="record" and isinstance(node.func.value,ast.Name)
            and node.func.value.id=="robot")


def _side_effect_free(node):
    if isinstance(node,(ast.Constant,ast.Name)):return True
    if isinstance(node,(ast.List,ast.Tuple,ast.Set)):
        return all(_side_effect_free(item) for item in node.elts)
    if isinstance(node,ast.Dict):
        return all((key is None or _side_effect_free(key)) and _side_effect_free(value)
                   for key,value in zip(node.keys,node.values))
    if isinstance(node,ast.UnaryOp):return _side_effect_free(node.operand)
    if isinstance(node,ast.BinOp):return _side_effect_free(node.left) and _side_effect_free(node.right)
    return False


class _RemoveDiagnosticRecords(ast.NodeTransformer):
    def visit_Expr(self,node):
        return None if _is_robot_record(node.value) else self.generic_visit(node)


class _RemoveUnusedSimpleAssignments(ast.NodeTransformer):
    def __init__(self,loaded):self.loaded=loaded
    def visit_Assign(self,node):
        node=self.generic_visit(node)
        names=[target.id for target in node.targets if isinstance(target,ast.Name)]
        if (len(names)==len(node.targets) and all(name not in self.loaded for name in names)
                and _side_effect_free(node.value)):
            return None
        return node
    def visit_AnnAssign(self,node):
        node=self.generic_visit(node)
        if (isinstance(node.target,ast.Name) and node.target.id not in self.loaded
                and node.value is not None and _side_effect_free(node.value)):
            return None
        return node


def _controller_semantic_sha256(path: str|Path):
    tree=_RemoveDiagnosticRecords().visit(ast.parse(Path(path).read_text()))
    # Iterate because removing one unused assignment can make an upstream pure
    # assignment unused as well. Calls and other potentially effectful RHS
    # expressions are always retained.
    for _ in range(8):
        before=ast.dump(tree,annotate_fields=True,include_attributes=False)
        loaded={node.id for node in ast.walk(tree)
                if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load)}
        tree=_RemoveUnusedSimpleAssignments(loaded).visit(tree)
        after=ast.dump(tree,annotate_fields=True,include_attributes=False)
        if after==before:break
    canonical=ast.dump(tree,annotate_fields=True,include_attributes=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _static_string_bindings(tree):
    """Resolve only unambiguous, single-assignment string names."""
    stores={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store):
            stores[node.id]=stores.get(node.id,0)+1
    candidates={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) \
                and isinstance(node.value.value,str):
            for target in node.targets:
                if isinstance(target,ast.Name):candidates[target.id]=node.value.value
        elif (isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name)
                and isinstance(node.value,ast.Constant)
                and isinstance(node.value.value,str)):
            candidates[node.target.id]=node.value.value
    return {name:value for name,value in candidates.items() if stores.get(name)==1}


class _RobotStrategyVisitor(ast.NodeVisitor):
    """Extract robot-facing control structure while ignoring parameter tuning."""
    def __init__(self,bindings=None):
        self.events=[];self.timeline=[];self.bindings=dict(bindings or {})

    def _constant(self,node):
        if isinstance(node,ast.Constant):return node.value
        if isinstance(node,ast.Name):return self.bindings.get(node.id)
        return None

    @staticmethod
    def _dict_node(node,key):
        if not isinstance(node,ast.Dict):return None
        for item_key,item_value in zip(node.keys,node.values):
            if isinstance(item_key,ast.Constant) and item_key.value==key:
                return item_value
        return None

    def _dict_value(self,node,key):
        return self._constant(self._dict_node(node,key))

    @staticmethod
    def _expression_mode(node):
        """Describe a control input's structure without retaining tuned values."""
        if node is None:return None
        if isinstance(node,ast.Name):return "name"
        if isinstance(node,ast.Attribute):return "attribute"
        if isinstance(node,ast.Subscript):return "subscript"
        if isinstance(node,ast.Call):return "call"
        if isinstance(node,(ast.List,ast.Tuple)):
            return ("sequence",len(node.elts))
        if isinstance(node,ast.Constant):return "constant"
        return type(node).__name__.removeprefix("ast.").casefold()

    def visit_Call(self,node):
        if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id=="robot"):
            method=node.func.attr
            if method=="record":return
            if method=="use":
                tool_id=self._constant(node.args[0]) if node.args else None
                self.timeline.append(("tool",tool_id))
                return
            detail=None
            if method=="verify" and node.args:
                detail=self._constant(node.args[0])
            elif method=="act" and node.args:
                action_type=self._dict_value(node.args[0],"type")
                command=self._dict_value(node.args[0],"command")
                # Numeric tolerances, gains, offsets, and timeouts remain tuning
                # and are intentionally ignored. Supplying an explicit
                # orientation, however, changes which controller target is
                # executed and must receive one causal trial.
                orientation_mode=tuple((key,self._expression_mode(value))
                    for key in ("quaternion_xyzw","rotation_matrix")
                    if (value:=self._dict_node(node.args[0],key)) is not None)
                detail=(action_type,command,orientation_mode)
            elif method=="observe":
                detail=next((self._constant(item.value) for item in node.keywords
                             if item.arg=="channel"),None)
            event=("robot",method,detail)
            self.events.append(event);self.timeline.append(event)
        self.generic_visit(node)

    def _control(self,node,name):
        start=("control",name,"start")
        self.events.append(start);self.timeline.append(start)
        self.generic_visit(node)
        end=("control",name,"end")
        self.events.append(end);self.timeline.append(end)

    # Ordinary guards are ubiquitous and small receipt-check changes are not a
    # new physical strategy. Loops and exception/fallback blocks are retained
    # because they can express genuinely different recovery behavior.
    def visit_If(self,node):self.generic_visit(node)
    def visit_For(self,node):self._control(node,"for")
    def visit_While(self,node):self._control(node,"while")
    def visit_Try(self,node):self._control(node,"try")
    def visit_With(self,node):self._control(node,"with")


def _controller_strategy_sha256(path: str|Path):
    """Hash Tool/action/control-flow strategy, not numeric or formatting tweaks."""
    visitor=_RobotStrategyVisitor();visitor.visit(ast.parse(Path(path).read_text()))
    encoded=json.dumps(visitor.events,separators=(",",":"),default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _controller_strategy_prefix_sha256(path: str|Path, robot_event_count: int):
    """Hash only the static strategy prefix that could cause a failure."""
    requested=max(1,int(robot_event_count))
    visitor=_RobotStrategyVisitor();visitor.visit(ast.parse(Path(path).read_text()))
    prefix=[];observed=0
    for event in visitor.events:
        prefix.append(event)
        if event[0]=="robot":
            observed+=1
            if observed>=requested:break
    # A wrapper such as ``try: ... finally:`` can be introduced around the
    # whole Controller without changing any robot action.  Remove control
    # markers for empty blocks so formatting/refactoring cannot evade the gate.
    filtered=[]
    for index,event in enumerate(prefix):
        if event[0]=="control" and event[2] in {"start","end"}:
            name=event[1]
            if event[2]=="start":
                depth=1;has_robot=False
                for child in prefix[index+1:]:
                    if child==("control",name,"start"):depth+=1
                    elif child==("control",name,"end"):
                        depth-=1
                        if depth==0:break
                    elif child[0]=="robot":has_robot=True
                if not has_robot:continue
            else:
                # Matching empty-block end markers were omitted with their start.
                start=next((j for j in range(len(filtered)-1,-1,-1)
                            if filtered[j]==("control",name,"start")),None)
                if start is None:continue
        filtered.append(event)
    prefix=filtered
    # Normalize a whole-program exception wrapper.  It is a transaction/error
    # handling refactor, not a different grasp or transport strategy; inner
    # try/fallback blocks remain visible.
    if (len(prefix)>=2 and prefix[0]==("control","try","start")
            and prefix[-1]==("control","try","end")):
        prefix=prefix[1:-1]
    encoded=json.dumps({"requested_robot_events":requested,
        "observed_robot_events":observed,"events":prefix},
        separators=(",",":"),default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _controller_tool_ids_before_robot_event(path: str|Path, robot_event_count: int):
    """Return Tool IDs that can affect execution before a failed prefix."""
    requested=max(1,int(robot_event_count))
    tree=ast.parse(Path(path).read_text())
    visitor=_RobotStrategyVisitor(_static_string_bindings(tree));visitor.visit(tree)
    result=set();observed=0
    for event in visitor.timeline:
        if event[0]=="tool" and isinstance(event[1],str):result.add(event[1])
        elif event[0]=="robot":
            observed+=1
            if observed>=requested:break
    return result


def _controller_tool_ids(path: str|Path):
    result=set();tree=ast.parse(Path(path).read_text())
    bindings=_static_string_bindings(tree)
    for node in ast.walk(tree):
        if (not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute)
                or not isinstance(node.func.value,ast.Name)
                or node.func.value.id!="robot" or node.func.attr!="use" or not node.args):
            continue
        value=node.args[0]
        if isinstance(value,ast.Constant) and isinstance(value.value,str):result.add(value.value)
        elif isinstance(value,ast.Name) and value.id in bindings:result.add(bindings[value.id])
    return result


def remap_controller_tool_ids(source: str, replacements: dict[str,str]):
    """Rebind exact Python string constants while preserving all program logic."""
    try:tree=ast.parse(source)
    except SyntaxError:
        # A persistent workspace may contain the very syntax error that the
        # coding agent must repair after a restart. Tokenization can still
        # safely rebind exact string literals without touching comments,
        # identifiers, or arbitrary substrings.
        tokens=[];generator=tokenize.generate_tokens(io.StringIO(source).readline)
        while True:
            try:tokens.append(next(generator))
            except StopIteration:break
            except (tokenize.TokenError,IndentationError,SyntaxError):break
        line_starts=[];offset=0
        for line in source.splitlines(keepends=True):
            line_starts.append(offset);offset+=len(line)
        edits=[]
        for token in tokens:
            if token.type!=tokenize.STRING:continue
            try:value=ast.literal_eval(token.string)
            except (SyntaxError,ValueError):continue
            if not isinstance(value,str) or value not in replacements:continue
            start=line_starts[token.start[0]-1]+token.start[1]
            end=line_starts[token.end[0]-1]+token.end[1]
            edits.append((start,end,repr(str(replacements[value]))))
        for start,end,literal in sorted(edits,reverse=True):
            source=source[:start]+literal+source[end:]
        return source,len(edits)
    encoded=source.encode("utf-8")
    line_starts=[];offset=0
    for line in source.splitlines(keepends=True):
        line_starts.append(offset);offset+=len(line.encode("utf-8"))
    edits=[]
    for node in ast.walk(tree):
        if (not isinstance(node,ast.Constant) or not isinstance(node.value,str)
                or node.value not in replacements):
            continue
        if not all(hasattr(node,name) for name in
                   ("lineno","col_offset","end_lineno","end_col_offset")):
            raise RuntimeError("Tool-id constant lacks source coordinates")
        start=line_starts[node.lineno-1]+node.col_offset
        end=line_starts[node.end_lineno-1]+node.end_col_offset
        segment=encoded[start:end].decode("utf-8")
        replacement=str(replacements[node.value])
        if (len(segment)>=2 and segment[0] in {"'",'"'}
                and segment[-1]==segment[0]
                and segment[:3] not in {"'''",'"""'}):
            quote=segment[0]
            literal=quote+replacement.replace("\\","\\\\").replace(
                quote,"\\"+quote)+quote
        else:literal=repr(replacement)
        edits.append((start,end,literal.encode("utf-8")))
    for start,end,literal in sorted(edits,reverse=True):
        encoded=encoded[:start]+literal+encoded[end:]
    return encoded.decode("utf-8"),len(edits)


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


def _selected_mapping(value, keys):
    """Project a diagnostic receipt onto a small, stable evidence surface."""
    if not isinstance(value,dict):return value
    return {key:value[key] for key in keys if key in value}


def _motion_receipt_summary(value):
    return _selected_mapping(value,(
        "type","reached","step","eef_before","eef_after","target_xyz",
        "target_quaternion_xyzw","final_position_error_m",
        "final_orientation_error_rad","gripper_qpos","error",
    ))


def _verification_summary(value):
    if not isinstance(value,dict):return value
    summary=_selected_mapping(value,(
        "verified","source_vacated","nearest_source_detection_m",
        "object_to_eef_distance_m","gripper_width_m","retained_width",
        "target_xy_error_m","vertical_offset_m","support_overlap_fraction",
        "object_coverage_fraction","target_coverage_fraction","support_gap_m",
        "center_inside_target_bounds","target_surface_height_error_m",
        "criterion","reason","contradiction","confidence",
    ))
    # Centroids are often enough to diagnose grounding, attachment, and
    # placement.  Full masks, boxes, and bounds remain queryable on disk.
    for name in ("object","target"):
        record=value.get(name)
        if isinstance(record,dict):
            summary[name]=_selected_mapping(record,(
                "query","label","score","world_xyz","point_ref","mask_path"))
    return summary


def _transient_error_record(value):
    if not isinstance(value,dict):return None
    kind=str(value.get("type") or "")
    message=str(value.get("message") or value.get("error") or "")
    text=(kind+" "+message).lower()
    if any(token in text for token in (
            "connectionerror","connection error","timeout","timed out",
            "exceeded 90 seconds","did not reach decision quorum",
            "without a decision quorum","temporarily unavailable",
            "service unavailable")):
        return {"error_type":kind or "infrastructure_error","message":message[:1000]}
    return None


def transient_infrastructure_failure(execution, sensor_report=None):
    """Separate retryable service outages from task-level sensor failures."""
    events=(execution or {}).get("rpc_events") or []
    robot_actions=sum(1 for event in events if event.get("method")=="act")
    # The independent task verifier runs after controller I/O is sealed.  Its
    # transport outage cannot be diagnosed by changing robot behavior, even
    # though physical actions already occurred.  Preserve the rollout and
    # replay the exact controller/case; never promote a local positive verifier
    # to success while the independent verifier is unavailable.
    outcome=(sensor_report or {}).get("independent_task_outcome") or {}
    outage=_transient_error_record(outcome)
    if outage is not None:
        return {"kind":"transient_post_action_sensor_verifier_outage",
                "tool_id":"independent_task_outcome_verifier",
                **outage,"robot_actions":robot_actions,
                "retry":"same_controller_same_case"}
    consensus=outcome.get("consensus") or {}
    try:
        required=int(consensus.get("required"))
        true_votes=int(consensus.get("true_votes"))
        false_votes=int(consensus.get("false_votes"))
    except (TypeError,ValueError):
        required=true_votes=false_votes=0
    if required>0 and max(true_votes,false_votes)<required:
        return {"kind":"transient_post_action_sensor_verifier_outage",
                "tool_id":"independent_task_outcome_verifier",
                "error_type":"VLMConsensusInconclusive",
                "message":(
                    f"no decision quorum: true_votes={true_votes}, "
                    f"false_votes={false_votes}, required={required}"),
                "robot_actions":robot_actions,
                "retry":"same_controller_same_case"}
    if robot_actions:return None
    candidates=[]
    for event in events:
        if event.get("method")=="use":
            candidates.append(((event.get("arguments") or {}).get("tool_id"),
                               event.get("result")))
    controller_result=(execution or {}).get("result")
    if isinstance(controller_result,dict):candidates.append((None,controller_result))
    def errors(value):
        if isinstance(value,dict):
            item=value.get("tool_error")
            if isinstance(item,dict):yield item
            for child in value.values():yield from errors(child)
        elif isinstance(value,list):
            for child in value:yield from errors(child)
    for tool_id,value in candidates:
        for item in errors(value):
            outage=_transient_error_record(item)
            if outage is not None:
                return {"kind":"transient_tool_outage_before_action",
                        "tool_id":tool_id,**outage,"robot_actions":0,
                        "retry":"same_controller_same_case"}
    return None


def _potential_unbound_local_loads(source: str, filename: str):
    """Find reachable LOAD_FAST operations lacking a dominating assignment.

    Python compilation catches syntax and unresolved globals, but deliberately
    permits a local that is assigned on only one control-flow branch. Such a
    program raises UnboundLocalError only when that branch is exercised, which
    must not consume a simulator or physical-robot episode. This small forward
    data-flow analysis intersects definitely initialized locals at every merge.
    """
    syntax_tree=ast.parse(source,filename);root=compile(syntax_tree,filename,"exec");issues=[]
    unconditional_jumps={
        "JUMP_ABSOLUTE","JUMP_BACKWARD","JUMP_BACKWARD_NO_INTERRUPT",
        "JUMP_FORWARD","CONTINUE_LOOP",
    }
    terminals={"RETURN_VALUE","RAISE_VARARGS","RERAISE"}

    def statically_nonempty(iterable):
        if isinstance(iterable,(ast.List,ast.Tuple,ast.Set)):
            return bool(iterable.elts)
        if (isinstance(iterable,ast.Call) and isinstance(iterable.func,ast.Name)
                and iterable.func.id=="range" and not iterable.keywords
                and 1<=len(iterable.args)<=3
                and all(isinstance(item,ast.Constant) and isinstance(item.value,int)
                        for item in iterable.args)):
            try:return bool(range(*(item.value for item in iterable.args)))
            except (TypeError,ValueError):return False
        return False

    def target_names(target):
        if isinstance(target,(ast.Tuple,ast.List)):
            return {name for item in target.elts for name in target_names(item)}
        return {target.id} if isinstance(target,ast.Name) else set()

    def outer_loop_control(node):
        """Find break/continue for this loop, excluding nested loop scopes."""
        found=False
        def visit(value):
            nonlocal found
            if found:return
            if isinstance(value,(ast.FunctionDef,ast.AsyncFunctionDef,ast.Lambda,
                                 ast.For,ast.AsyncFor,ast.While)):
                return
            if isinstance(value,(ast.Break,ast.Continue)):
                found=True;return
            for child in ast.iter_child_nodes(value):visit(child)
        visit(node);return found

    # Bytecode represents a for-loop exit as reachable before its first body
    # iteration, even when a literal tuple or range proves otherwise. Record
    # direct body assignments that every continuing first iteration must cross,
    # so the CFG does not create false positives for those loops. A break or
    # continue before the assignment invalidates the proof; returns are safe
    # because they never reach a later load.
    loop_guarantees={}
    def collect_loops(function):
        for node in ast.walk(function):
            if not isinstance(node,ast.For) or not statically_nonempty(node.iter):continue
            blocked=False;guaranteed=set()
            for statement in node.body:
                if outer_loop_control(statement):blocked=True
                if blocked:continue
                if isinstance(statement,(ast.Assign,ast.AnnAssign)):
                    targets=(statement.targets if isinstance(statement,ast.Assign)
                             else [statement.target])
                    guaranteed.update(name for target in targets for name in target_names(target))
            if guaranteed:
                deleted={child.id for child in ast.walk(node)
                         if isinstance(child,ast.Name) and isinstance(child.ctx,ast.Del)}
                guaranteed-=deleted
            if guaranteed:loop_guarantees[(function.name,node.lineno)]=guaranteed
    for node in ast.walk(syntax_tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):collect_loops(node)

    def analyze(code):
        instructions=list(dis.get_instructions(code))
        if not instructions:return
        by_offset={item.offset:index for index,item in enumerate(instructions)}
        lines=[];line=code.co_firstlineno
        for item in instructions:
            if item.starts_line is not None:line=item.starts_line
            lines.append(line)
        argument_count=code.co_argcount+code.co_kwonlyargcount
        if code.co_flags & 0x04:argument_count+=1
        if code.co_flags & 0x08:argument_count+=1
        entry=set(code.co_varnames[:argument_count])
        incoming=[None for _ in instructions];incoming[0]=entry
        queue=[0];reported=set()
        while queue:
            index=queue.pop(0);state=set(incoming[index] or ())
            item=instructions[index]
            if item.opname=="LOAD_FAST" and item.argval not in state:
                key=(code.co_name,lines[index],str(item.argval))
                if key not in reported:
                    reported.add(key);issues.append(key)
            if item.opname=="STORE_FAST":state.add(str(item.argval))
            elif item.opname=="DELETE_FAST":state.discard(str(item.argval))
            successors=[]
            is_jump=item.opcode in dis.hasjabs or item.opcode in dis.hasjrel
            if is_jump and isinstance(item.argval,int) and item.argval in by_offset:
                successors.append(by_offset[item.argval])
            conditional=(item.opname not in unconditional_jumps and is_jump)
            if (item.opname not in terminals and (not is_jump or conditional)
                    and index+1<len(instructions)):
                successors.append(index+1)
            for successor in successors:
                edge_state=set(state)
                if (item.opname=="FOR_ITER" and isinstance(item.argval,int)
                        and successor==by_offset.get(item.argval)):
                    edge_state.update(loop_guarantees.get(
                        (code.co_name,lines[index]),set()))
                prior=incoming[successor]
                merged=edge_state if prior is None else set(prior)&edge_state
                if prior is None or merged!=prior:
                    incoming[successor]=merged;queue.append(successor)
        for constant in code.co_consts:
            if isinstance(constant,types.CodeType):analyze(constant)

    analyze(root)
    return issues


def _record_summary(value):
    if not isinstance(value,dict):return value
    summary=_selected_mapping(value,(
        "phase","attempt","fraction","xy_offset_m","old_source_shift_m",
        "new_bowl_world_xyz","credible_narrow_capture","reason",
    ))
    if "receipt" in value:summary["receipt"]=_motion_receipt_summary(value["receipt"])
    if "attachment" in value:summary["attachment"]=_verification_summary(value["attachment"])
    return summary or {"keys":sorted(str(key) for key in value)[:40]}


def _agent_robot_summary(result,detail_path):
    """Compact evidence returned through the model tool channel.

    The complete immutable execution remains in robot_execution.json.  This
    summary keeps actionable motion receipts and sensor decisions while
    replacing multi-megabyte perception / grasp payloads with output shapes.
    """
    execution=result.get("execution") or {};events=[]
    for event_index,event in enumerate(execution.get("rpc_events") or []):
        method=event.get("method");item={"event_index":event_index,"method":method}
        arguments=event.get("arguments") or {};rpc_result=event.get("result")
        if method=="act":
            item["action"]=_selected_mapping(arguments.get("action"),(
                "type","command","repeat","steps","pose_ref","target_ref",
                "offset","gripper"))
            item["result"]=_motion_receipt_summary(rpc_result)
        elif method=="verify":
            item["verifier"]=arguments.get("verifier")
            item["result"]=_verification_summary(rpc_result)
        elif method=="record":
            item["event"]=_record_summary(arguments.get("event"))
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
    event_count=len(events);omitted_event_range=None
    # A long recovery may contain hundreds of nearly identical motion RPCs.
    # Keep the grounding prefix and failure-local suffix in working context;
    # every omitted event remains addressable through inspect_execution_event.
    if len(json.dumps(events,default=str))>20_000 and len(events)>24:
        prefix=events[:6];suffix=events[-18:]
        omitted_event_range={"start":prefix[-1]["event_index"]+1,
                             "end":suffix[0]["event_index"]-1,
                             "count":event_count-len(prefix)-len(suffix)}
        events=prefix+suffix
    report=_compact_evidence_value({
        key:value for key,value in (result.get("sensor_report") or {}).items()
        if not str(key).startswith("_harness_")},max_list_items=8)
    return {"controller_path":result.get("controller_path"),
            "controller_snapshot":result.get("controller_snapshot"),
            "completed":execution.get("completed"),"error":execution.get("error"),
            "controller_result":_compact_evidence_value(
                execution.get("result"),max_list_items=8),"rpc_evidence":events,
            "rpc_event_count":event_count,
            "rpc_evidence_omitted_range":omitted_event_range,
            "rpc_inspection_hint":("Call inspect_execution_event with the event_index and "
                                   "execution_artifact_ref for complete selected evidence."),
            "sensor_report":report,
            "sensor_success_candidate":result.get("sensor_success_candidate"),
            "transient_infrastructure_failure":result.get(
                "transient_infrastructure_failure"),
            "robot_contract_preflight":result.get("robot_contract_preflight"),
            "task_model_preflight":result.get("task_model_preflight"),
            "full_execution_artifact":str(Path(detail_path).resolve()),
            "execution_artifact_ref":"latest_robot_execution"}


def _compact_evidence_value(value, *, max_list_items=8, depth=0, max_depth=7):
    """Keep diagnostic values while bounding large perception payloads."""
    if isinstance(value,(str,int,float,bool)) or value is None:
        if isinstance(value,str) and len(value)>2048:
            return value[:2048]+"...<truncated>"
        return value
    if depth>=max_depth:
        if (isinstance(value,list) and len(value)<=16
                and all(isinstance(item,(str,int,float,bool)) or item is None
                        for item in value)):
            return value
        return {"type":type(value).__name__,"truncated":True}
    if isinstance(value,dict):
        compact={}
        for key,item in value.items():
            key=str(key)
            if key=="frame" and isinstance(item,dict):
                cameras=item.get("cameras") or {}
                compact[key]={"frame_id":item.get("frame_id"),"step":item.get("step"),
                    "camera_rgb_paths":{name:camera.get("rgb_path")
                        for name,camera in cameras.items()
                        if isinstance(camera,dict) and camera.get("rgb_path")}}
            elif key in {"data_base64","rgb","depth"}:
                compact[key]={"type":"omitted_sensor_array"}
            else:
                compact[key]=_compact_evidence_value(item,max_list_items=max_list_items,
                    depth=depth+1,max_depth=max_depth)
        return compact
    if isinstance(value,list):
        if len(value)<=max_list_items:
            return [_compact_evidence_value(item,max_list_items=max_list_items,
                depth=depth+1,max_depth=max_depth) for item in value]
        return {"type":"list","count":len(value),
                "head":[_compact_evidence_value(item,max_list_items=max_list_items,
                    depth=depth+1,max_depth=max_depth)
                        for item in value[:max_list_items]],
                "remaining":len(value)-max_list_items}
    return str(value)


class EngineeringSurface:
    def __init__(self, *, workspace: TaskWorkspace, capabilities: CapabilityLibrary,
                 runtime: ControllerRuntime, deployment_factory: Callable[[], Any],
                 artifact_dir: str | Path, task_model: dict|None=None,
                 semantic_reviewer: Callable[...,dict]|None=None,
                 task_instruction: str|None=None,
                 task_fidelity_reviewer: Callable[...,dict]|None=None,
                 acquisition_reviewer: Callable[...,dict]|None=None,
                 allowed_action_types: list[str]|None=None,
                 sdk_contract: dict|None=None,
                 active_deployment_tool_ids: list[str]|None=None,
                 execution_observer: Callable[[dict],None]|None=None,
                 experiences: ExperienceLibrary|None=None,
                 skills: SkillLibrary|None=None,
                 gaps: CapabilityGapLibrary|None=None,
                 rejected_controller_semantic_sha256: str|None=None,
                 rejected_controller_strategy_failures: dict[str,dict]|None=None,
                 controller_tool_replacements: dict[str,str]|None=None,
                 required_acquisition_gap_id: str|None=None,
                 acquisition_baseline_tool_ids: list[str]|None=None):
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
        self.task_instruction=str(task_instruction) if task_instruction else None
        self.task_fidelity_reviewer=task_fidelity_reviewer
        self.acquisition_reviewer=acquisition_reviewer
        self.sdk_contract=dict(sdk_contract or {})
        self.active_deployment_tool_ids=set(str(x) for x in
                                            (active_deployment_tool_ids or []))
        self.execution_observer=execution_observer
        self.experiences=experiences
        self.skills=skills
        self.gaps=gaps
        self.rejected_controller_semantic_sha256=(
            str(rejected_controller_semantic_sha256)
            if rejected_controller_semantic_sha256 else None)
        self.rejected_controller_strategy_failures={str(key):dict(value) for key,value in
            dict(rejected_controller_strategy_failures or {}).items()}
        self.controller_tool_replacements={str(old):str(new) for old,new in
            dict(controller_tool_replacements or {}).items() if old and new and old!=new}
        self.required_acquisition_gap_id=(str(required_acquisition_gap_id)
            if required_acquisition_gap_id else None)
        self.acquisition_baseline_tool_ids=set(str(item) for item in
            (acquisition_baseline_tool_ids or []))
        self.capability_integration_preflight=None
        self.skill_interface_path=self.artifact_dir/"skill_interface.json"
        self.research_ledger_path=self.artifact_dir/"research_ledger.jsonl"
        # Model-facing evidence handles avoid making the coding agent copy and
        # edit long host paths.  The mapping is scoped to this EngineeringSurface
        # and every target is still checked by the normal artifact allowlist.
        self._image_artifact_refs: dict[str,Path]={}
        contracted=(self.sdk_contract.get("actions") or {}).keys()
        self.allowed_action_types=set(str(x) for x in (allowed_action_types or contracted))

    def _authorize_run_artifact(self,target: Path):
        """Expose sensor/program evidence, never deployment/evaluator metadata."""
        target=target.resolve()
        try:relative=target.relative_to(self.run_root)
        except ValueError as exc:raise RuntimeError("artifact must be inside the current run") from exc
        parts=relative.parts;allowed=False
        if len(parts)>=3 and parts[0]=="iterations" and parts[1].startswith("iteration_"):
            allowed=(parts[2] in {"robot_execution.json","controller.py","research_ledger.jsonl",
                                  "rollout_keyframes"})
        elif len(parts)>=3 and parts[0]=="episodes" and parts[1].startswith("episode_"):
            allowed=(parts[2] in {"sensors","outcome"} or
                     (len(parts)==3 and parts[2] in {"adapter_trace.json","rollout.mp4"}))
        elif len(parts)==1 and parts[0]=="bootstrap_experience.json":allowed=True
        if not allowed:
            raise RuntimeError("artifact is outside the controller-visible evidence surface")
        return target

    def list_available_tools(self):
        rows=self.capabilities.search("",limit=50)
        if not self.active_deployment_tool_ids:return rows
        return [row for row in rows if not row.get("execution_owned_by_deployment")
                or row.get("tool_id") in self.active_deployment_tool_ids]

    def search_assets(self,query: str,asset_types: list[str],limit: int=8):
        requested_limit=int(limit);limit=max(1,min(requested_limit,20))
        kinds=set(asset_types);result={}
        def text(value,maximum=600):
            value=str(value or "")
            return value if len(value)<=maximum else value[:maximum-3]+"..."
        def score(item):
            return item.get("retrieval_score")
        if "tool" in kinds:
            result["tools"]=[]
            for item in self.capabilities.search(query,limit=limit):
                input_schema=item.get("input_schema") or {}
                output_schema=item.get("output_schema") or {}
                result["tools"].append({
                    "tool_id":item.get("tool_id"),"name":item.get("name"),
                    "asset_kind":item.get("asset_kind"),"status":item.get("status"),
                    "description":text(item.get("description")),
                    "input_fields":sorted((input_schema.get("properties") or {}).keys()),
                    "required_inputs":list(input_schema.get("required") or []),
                    "output_fields":sorted((output_schema.get("properties") or {}).keys()),
                    "execution_owned_by_deployment":item.get("execution_owned_by_deployment",False),
                    "retrieval_score":score(item)})
        if "experience" in kinds and self.experiences is not None:
            result["experiences"]=[{
                "experience_id":item.get("experience_id"),"name":item.get("name"),
                "summary":text(item.get("summary")),
                "applicability":text(item.get("applicability"),400),
                "keywords":list(item.get("keywords") or [])[:12],
                "status":item.get("status"),"retrieval_score":score(item)}
                for item in self.experiences.search(query,limit=limit)]
        if "skill" in kinds and self.skills is not None:
            result["skills"]=[]
            for item in self.skills.search(query,limit=limit):
                interface=item.get("interface") or {}
                result["skills"].append({
                    "skill_id":item.get("skill_id"),"task":text(item.get("task")),
                    "status":item.get("status"),
                    "effects":[text(value,300) for value in (interface.get("effects") or [])[:4]],
                    "required_sensors":list(interface.get("required_sensors") or [])[:12],
                    "tool_ids":list(item.get("tool_ids") or [])[:20],
                    "retrieval_score":score(item)})
        if "gap" in kinds and self.gaps is not None:
            result["gaps"]=[]
            for item in self.gaps.search(query,limit=limit):
                capability=item.get("required_capability") or {}
                result["gaps"].append({
                    "gap_id":item.get("gap_id"),"name":item.get("name"),
                    "task":text(item.get("task")),"status":item.get("status"),
                    "failure_summary":text(item.get("failure_summary")),
                    "selected_diagnosis":text(item.get("selected_diagnosis")),
                    "required_capability_kind":text(capability.get("kind"),300),
                    "previous_gap_id":item.get("previous_gap_id"),
                    "retrieval_score":score(item)})
        return {"query":query,"limit":limit,"requested_limit":requested_limit,**result}

    def inspect_skill(self,skill_id: str):
        if self.skills is None:raise RuntimeError("Skill Library unavailable")
        manifest=self.skills.inspect(skill_id);result=dict(manifest)
        development=result.pop("development_evidence",{}) or {}
        report=development.get("report") or {}
        sensor_report=report.get("sensor_report") or {}
        outcome=sensor_report.get("independent_task_outcome") or {}
        result["development_evidence_summary"]={
            "iteration":development.get("iteration"),
            "sensor_only":development.get("sensor_only"),
            "sensor_success_candidate":report.get("sensor_success_candidate"),
            "program_sha256":((report.get("execution") or {}).get("program_sha256")
                              or report.get("program_sha256")),
            "sensor_verification_passed":sensor_report.get("sensor_verification_passed"),
            "independent_task_outcome":{key:outcome.get(key) for key in
                ("verified","reason","confidence","method","sensor_only")
                if key in outcome},
            "controller_result_keys":sorted((report.get("controller_result") or {}).keys()),
        }
        result["evidence_files"]=[]
        for item in manifest.get("evidence_files") or []:
            public={key:value for key,value in item.items() if key!="original_path"}
            result["evidence_files"].append({**public,
                "asset_ref":f"{skill_id}#{item['path']}"})
        return result

    def inspect_experience(self,experience_id: str):
        if self.experiences is None:raise RuntimeError("Experience Library unavailable")
        manifest=self.experiences.inspect(experience_id);result=dict(manifest)
        result["evidence"]=[]
        for item in manifest.get("evidence") or []:
            public={key:value for key,value in item.items() if key!="original_path"}
            result["evidence"].append({**public,
                "asset_ref":f"{experience_id}#{item['path']}"})
        return result

    def read_skill_source(self,skill_id: str,start_line: int=1,end_line: int=200):
        if self.skills is None:raise RuntimeError("Skill Library unavailable")
        try:path=self.skills._path(skill_id)/"controller.py"
        except (AssetError,FileNotFoundError):
            return {"skill_id":skill_id,"exists":False,
                    "instruction":"Call search_assets with asset_types=['skill'] and pass an exact returned skill_id."}
        lines=path.read_text().splitlines()
        start=max(1,int(start_line));end=max(start,min(int(end_line),start+399));returned=min(end,len(lines))
        return {"skill_id":skill_id,"exists":True,
                "source":{"start_line":start,"end_line":returned,
            "total_lines":len(lines),"content":"\n".join(lines[start-1:returned]),
            "next_start_line":returned+1 if returned<len(lines) else None}}

    def checkout_skill_controller(self,skill_id: str,destination: str="controller.py"):
        """Load a verified frozen Skill as editable workspace source."""
        if self.skills is None:raise RuntimeError("Skill Library unavailable")
        manifest=self.skills.inspect(skill_id)
        source=self.skills._path(skill_id)/"controller.py"
        content=source.read_text()
        content,rebound_constants=remap_controller_tool_ids(
            content,self.controller_tool_replacements)
        target=self.workspace._path(destination)
        previous_sha256=(hashlib.sha256(target.read_bytes()).hexdigest()
                         if target.is_file() else None)
        written=self.workspace.write_file(destination,content)
        return {**written,"source_skill_id":skill_id,
            "source_controller_sha256":manifest["controller_sha256"],
            "checked_out_controller_sha256":hashlib.sha256(content.encode()).hexdigest(),
            "deployment_tool_constants_rebound":rebound_constants,
            "deployment_tool_replacements":dict(self.controller_tool_replacements),
            "previous_controller_sha256":previous_sha256,
            "editable":True,
            "next_step":"Inspect and edit only the task-relevant differences, then test and preflight the workspace controller."}

    def restore_previous_executed_controller(self,destination: str="controller.py"):
        """Restore the newest immutable Controller that produced a robot ledger.

        Workspace commands intentionally cannot read arbitrary run artifacts.
        Recovery therefore needs a narrow audited operation instead of asking
        the model to reconstruct hundreds of source lines or bypass isolation
        with ``cp``.
        """
        current=(self.artifact_dir/"controller.py").resolve()
        candidates=[]
        for source in self.run_root.glob("iterations/iteration_*/controller.py"):
            if source.resolve()==current:continue
            if not (source.parent/"robot_execution.json").is_file():continue
            candidates.append(source)
        if not candidates:raise RuntimeError("no previous executed Controller snapshot")
        source=max(candidates,key=lambda item:item.parent.name)
        target=self.workspace._path(destination)
        previous_sha256=(hashlib.sha256(target.read_bytes()).hexdigest()
                         if target.is_file() else None)
        before=None
        if destination=="controller.py" and target.is_file():
            try:before=_controller_semantic_sha256(target)
            except (OSError,SyntaxError):before="invalid-controller-source"
        content=source.read_text();written=self.workspace.write_file(destination,content)
        after=None
        if destination=="controller.py":
            try:after=_controller_semantic_sha256(target)
            except (OSError,SyntaxError):after="invalid-controller-source"
        changed=before!=after
        return {**written,
            "source_artifact_ref":str(source.relative_to(self.run_root)),
            "source_controller_sha256":hashlib.sha256(content.encode()).hexdigest(),
            "previous_controller_sha256":previous_sha256,
            "controller_semantic_progress":changed,
            "_embodied_codex_semantic_progress":changed,
            "restored_from_executed_robot_ledger":True,
            "next_step":"Compile the restored Controller, make only the required correction, then run it."}

    def inspect_tool(self, tool_id: str):
        """Read the Tool manual and provenance without implementation source."""
        inspected=self.capabilities.inspect(tool_id)
        manifest=dict(inspected["manifest"]);tests=manifest.pop("tests",[])
        cases=[case for batch in tests if isinstance(batch,list) for case in batch
               if isinstance(case,dict)]
        if manifest.get("execution_owned_by_deployment") and not cases:
            # Adapter-owned model/services are validated by the deployment
            # binding and provenance gate, not by CapabilityLibrary's isolated
            # Python test runner. Reporting all_passed=false would contradict
            # status=tested and invite needless source inspection.
            manifest["test_summary"]={"batches":0,"cases":0,"all_passed":None,
                "status_authority":"deployment_adapter_binding"}
        else:
            manifest["test_summary"]={"batches":len(tests),"cases":len(cases),
                "all_passed":bool(cases) and all(case.get("passed") is True for case in cases),
                "status_authority":"capability_library_tests"}
        return {"manifest":manifest,
                "manual":self.capabilities.manual(tool_id)}

    def read_tool_source(self, tool_id: str, start_line: int=1, end_line: int=200):
        """Explicit exceptional-path, paginated Tool implementation access."""
        inspected=self.capabilities.inspect(tool_id)
        lines=str(inspected.get("source") or "").splitlines()
        start=max(1,int(start_line));end=max(start,min(int(end_line),start+399))
        returned_end=min(end,len(lines))
        return {"tool_id":tool_id,"source":{"start_line":start,"end_line":returned_end,
                "total_lines":len(lines),"content":"\n".join(lines[start-1:returned_end]),
                "next_start_line":returned_end+1 if returned_end<len(lines) else None}}

    def revise_tool_manual(self, tool_id: str, manual: dict, evidence_refs: list[str]):
        paths=[]
        for value in evidence_refs:
            paths.append(self._authorized_evidence_path(value))
        return self.capabilities.revise_manual(tool_id,manual,evidence_paths=paths)

    def _evidence_reference(self,value: str):
        requested=str(value);current=self.artifact_dir/"robot_execution.json"
        if requested in {"controller.py","executed_controller"}:
            current_controller=self.artifact_dir/"controller.py"
            if current_controller.is_file():return current_controller
            prior=sorted(self.run_root.glob("iterations/iteration_*/controller.py"))
            prior=[item for item in prior if item.resolve()!=current_controller.resolve()]
            return prior[-1] if prior else current_controller
        if requested in {"research_ledger","current_research_ledger"}:
            return self.research_ledger_path
        if requested in {"latest_robot_execution","previous_robot_execution"}:
            prior=sorted(self.run_root.glob("iterations/iteration_*/robot_execution.json"))
            prior=[item for item in prior if item.resolve()!=current.resolve()]
            if requested=="latest_robot_execution":
                # `latest` is transaction-local: before this iteration commits
                # an episode it must not silently alias the previous episode.
                # That fallback let models write post-run diagnoses for an
                # execution that had not happened yet.
                return current
            return prior[-1] if prior else current
        return Path(requested)

    def _authorized_evidence_path(self,value: str):
        """Resolve immutable run evidence or a hash-validated Tool manifest."""
        requested=str(value)
        if "#" in requested and self.gaps is not None:
            gap_id,relative=requested.split("#",1)
            try:
                manifest=self.gaps.inspect(gap_id)
                match=next((item for item in manifest.get("evidence") or []
                            if str(item.get("path"))==relative),None)
                if match is not None:
                    target=(self.gaps._path(gap_id).parent/relative).resolve()
                    if target.is_file():return target
            except (AttributeError,FileNotFoundError,RuntimeError,ValueError):
                pass
        if "#" in requested and self.experiences is not None:
            experience_id,relative=requested.split("#",1)
            try:
                manifest=self.experiences.inspect(experience_id)
                if any(str(item.get("path"))==relative
                       for item in manifest.get("evidence") or []):
                    target=(self.experiences._path(experience_id)/relative).resolve()
                    if target.is_file():return target
            except (AttributeError,FileNotFoundError,RuntimeError,ValueError):pass
        if "#" in requested and self.skills is not None:
            skill_id,relative=requested.split("#",1)
            try:
                manifest=self.skills.inspect(skill_id)
                if any(str(item.get("path"))==relative
                       for item in manifest.get("evidence_files") or []):
                    target=(self.skills._path(skill_id)/relative).resolve()
                    if target.is_file():return target
            except (AttributeError,FileNotFoundError,RuntimeError,ValueError):pass
        if ":" in requested and not Path(requested).is_absolute():
            try:
                self.capabilities.inspect(requested)
                manifest=(self.capabilities._path(requested)/"manifest.json").resolve()
                if manifest.is_file():return manifest
            except (AttributeError,FileNotFoundError,RuntimeError,ValueError):
                pass
        target=self._evidence_reference(requested)
        if not target.is_absolute():target=self.run_root/target
        target=target.resolve();self._authorize_run_artifact(target)
        if not target.is_file():raise FileNotFoundError(requested)
        return target

    def inspect_execution_event(self,path: str,event_index: int,max_list_items: int=8):
        """Read one RPC event without replaying a whole execution log into context."""
        # Use the same resolver as read_run_artifact.  In particular, an
        # Experience/Gap/Skill evidence asset_ref is already hash-validated by
        # its immutable manifest and must remain queryable without exposing an
        # arbitrary host path.
        target=self._authorized_evidence_path(path)
        document=json.loads(target.read_text())
        events=((document.get("execution") or {}).get("rpc_events") or [])
        index=int(event_index)
        if index<0 or index>=len(events):
            raise RuntimeError(f"event_index out of range: {index}; event_count={len(events)}")
        item=events[index]
        return {"path":str(target),"event_index":index,"event_count":len(events),
                "event":_compact_evidence_value(item,
                    max_list_items=max(1,min(int(max_list_items),20)),max_depth=5)}

    @staticmethod
    def _json_pointer(value, pointer: str):
        current=value
        if pointer in ("", "/"):return current
        if not isinstance(pointer,str) or not pointer.startswith("/"):
            raise RuntimeError("JSON pointer must be empty or start with /")
        for raw in pointer.split("/")[1:]:
            token=raw.replace("~1","/").replace("~0","~")
            if isinstance(current,dict) and token in {"events","rpc_events"}:
                # `events` is the natural public name; `rpc_events` is the
                # immutable ledger's storage name. Accept either at the root
                # or under /execution so models do not have to learn a
                # serialization detail before querying structured evidence.
                if isinstance(current.get("rpc_events"),list):
                    current=current["rpc_events"]
                    continue
                execution=current.get("execution")
                if (isinstance(execution,dict)
                        and isinstance(execution.get("rpc_events"),list)):
                    current=execution["rpc_events"]
                    continue
            # Robot.use() returns the Tool result directly to a Controller,
            # while the immutable RPC ledger wraps it in one or more `result`
            # receipts. Make that Harness-owned serialization transparent to
            # evidence queries so model-visible and persisted contracts agree.
            unwraps=0
            while (isinstance(current,dict) and token not in current
                   and isinstance(current.get("result"),dict) and unwraps<4):
                current=current["result"];unwraps+=1
            if isinstance(current,dict) and token in current:current=current[token]
            elif isinstance(current,list) and token.isdigit() and int(token)<len(current):
                current=current[int(token)]
            else:
                available=sorted(str(key) for key in current)[:30] \
                    if isinstance(current,dict) else []
                raise RuntimeError(f"JSON pointer not found: {pointer}; available_keys={available}")
        return current

    def query_run_json(self,path: str,array_pointer: str,filters: list[dict]|None=None,
                       sort_by: str="",descending: bool=False,
                       fields: list[str]|None=None,offset: int=0,limit: int=8):
        """Query a run-local JSON array without paging its serialized text."""
        target=self._authorized_evidence_path(path)
        if target.stat().st_size>64*1024*1024:raise RuntimeError("JSON artifact exceeds 64 MiB")
        try:document=json.loads(target.read_text())
        except (OSError,json.JSONDecodeError) as exc:
            raise RuntimeError("run artifact is not valid JSON") from exc
        values=self._json_pointer(document,array_pointer)
        if not isinstance(values,list):raise RuntimeError("array_pointer must resolve to an array")
        is_event_array=str(array_pointer).rstrip("/").split("/")[-1] in {
            "events","rpc_events"}
        rows=list(enumerate(values));conditions=filters or []
        def field_pointer(value):
            pointer=str(value or "")
            return pointer if not pointer or pointer.startswith("/") else "/"+pointer
        if not isinstance(conditions,list) or len(conditions)>12:
            raise RuntimeError("filters must be a list with at most 12 conditions")
        def resolve(row,index,pointer):
            if is_event_array and pointer=="/event_index":return index
            return self._json_pointer(row,pointer)
        def matches(index,row):
            for condition in conditions:
                if not isinstance(condition,dict):raise RuntimeError("filter must be an object")
                pointer=field_pointer(condition.get("field"))
                operation=str(condition.get("op") or "eq");expected=condition.get("value")
                try:actual=resolve(row,index,pointer)
                except RuntimeError:return False
                if operation=="eq":accepted=actual==expected
                elif operation=="ne":accepted=actual!=expected
                elif operation in {"lt","lte","gt","gte"}:
                    try:accepted={"lt":actual<expected,"lte":actual<=expected,
                        "gt":actual>expected,"gte":actual>=expected}[operation]
                    except TypeError:accepted=False
                else:raise RuntimeError(f"unsupported filter op: {operation}")
                if not accepted:return False
            return True
        rows=[(index,row) for index,row in rows if matches(index,row)]
        if sort_by:
            sort_pointer=field_pointer(sort_by)
            def key(indexed):
                index,row=indexed
                try:value=resolve(row,index,sort_pointer)
                except RuntimeError:return (1,0)
                return (0,value) if isinstance(value,(int,float,str)) else (1,str(value))
            rows.sort(key=key,reverse=bool(descending))
        start=max(0,int(offset));count=max(1,min(int(limit),20));selected=rows[start:start+count]
        pointers=fields or []
        if not isinstance(pointers,list) or len(pointers)>24:
            raise RuntimeError("fields must contain at most 24 JSON pointers")
        if pointers:
            projected=[];missing_counts={str(pointer):0 for pointer in pointers}
            for index,row in selected:
                item={}
                for pointer in pointers:
                    requested=str(pointer);normalized=field_pointer(requested)
                    try:item[requested]=_compact_evidence_value(
                        resolve(row,index,normalized),max_list_items=8,max_depth=4)
                    except RuntimeError:
                        item[requested]=None;missing_counts[requested]+=1
                projected.append(item)
        else:
            projected=[_compact_evidence_value(row,max_list_items=8,max_depth=4)
                       for _,row in selected]
        response={"path":str(target),"array_pointer":array_pointer,
            "source_count":len(values),"matched_count":len(rows),"offset":start,
            "returned_count":len(projected),"next_offset":start+len(projected)
                if start+len(projected)<len(rows) else None,"rows":projected}
        if pointers:
            warnings=[{"field":field,"missing_rows":count,
                "returned_rows":len(selected),
                "note":"null denotes a field absent from this heterogeneous row; use filters or a nested pointer to narrow the projection"}
                for field,count in missing_counts.items() if count]
            if warnings:response["projection_warnings"]=warnings
        return response

    def _record_research(self,kind: str,payload: dict):
        unsigned={"protocol":"embodied-codex-research-evidence-v1","kind":kind,
                  "unix":time.time(),**payload}
        encoded=json.dumps(unsigned,sort_keys=True,separators=(",",":"),default=str)
        record={**unsigned,"record_sha256":hashlib.sha256(encoded.encode()).hexdigest()}
        with self.research_ledger_path.open("a") as stream:
            stream.write(json.dumps(record,default=str)+"\n")
        return record

    def search_web(self,query: str,limit: int=5):
        result=search_web(query,limit)
        rows=[{"url":str(item.get("url")),"title":str(item.get("title") or ""),
               "source":str(item.get("source") or "")}
              for item in (result.get("results") or []) if item.get("url")]
        record=self._record_research("search",{"query":str(query),"results":rows})
        return {**result,"research_record_sha256":record["record_sha256"]}

    def fetch_web_page(self,url: str,max_chars: int=30000):
        result=fetch_web_page(url,max_chars)
        content=str(result.get("content") or "")
        record=self._record_research("fetch",{"url":str(result.get("url") or url),
            "content_type":result.get("content_type"),"content_sha256":hashlib.sha256(
                content.encode()).hexdigest(),"content_characters":len(content),
            "truncated":bool(result.get("truncated"))})
        return {**result,"research_record_sha256":record["record_sha256"]}

    def download_public_asset(self,url: str,destination: str,max_bytes: int,
                              expected_sha256: str=""):
        target=self.workspace._path(destination);target.parent.mkdir(parents=True,exist_ok=True)
        result=download_public_file(url,target,max_bytes)
        expected=str(expected_sha256 or "").casefold()
        if expected and (len(expected)!=64 or result["sha256"].casefold()!=expected):
            target.unlink(missing_ok=True)
            raise RuntimeError("downloaded public asset sha256 mismatch")
        record=self._record_research("download",{"url":result["url"],
            "destination":destination,"bytes":result["bytes"],
            "content_sha256":result["sha256"],"expected_sha256":expected or None})
        return {**result,"path":destination,"research_record_sha256":record["record_sha256"]}

    def _research_records(self):
        records=[]
        for path in sorted(self.run_root.glob("iterations/iteration_*/research_ledger.jsonl")):
            for line in path.read_text(errors="replace").splitlines():
                try:record=json.loads(line)
                except json.JSONDecodeError:continue
                digest=record.pop("record_sha256",None)
                encoded=json.dumps(record,sort_keys=True,separators=(",",":"),default=str)
                if digest==hashlib.sha256(encoded.encode()).hexdigest():
                    records.append({**record,"record_sha256":digest})
        return records

    def list_research_sources(self,query: str="",offset: int=0,limit: int=20):
        """Return a bounded page of exact URLs eligible for asset provenance."""
        sources={}
        for record in self._research_records():
            candidates=[]
            if record.get("url"):
                candidates.append({"url":record["url"],"title":None})
            candidates.extend({"url":item.get("url"),"title":item.get("title")}
                              for item in record.get("results") or [] if item.get("url"))
            for item in candidates:
                url=str(item["url"])
                sources[url]={"url":url,"title":item.get("title"),
                    "kind":record.get("kind"),"query":record.get("query"),
                    "research_record_sha256":record["record_sha256"]}
        rows=list(sources.values())
        needle=str(query or "").strip().casefold()
        if needle:
            rows=[item for item in rows if needle in " ".join(str(item.get(key) or "")
                  for key in ("url","title","kind","query")).casefold()]
        # Most recently observed sources are normally the ones needed for the
        # Tool currently being authored. Bound the default response so a long
        # autonomous research campaign cannot flood the coding context merely
        # to copy one exact provenance URL.
        rows.reverse();start=max(0,int(offset));page_size=max(1,min(int(limit),50))
        page=rows[start:start+page_size]
        return {"sources":page,"count":len(sources),"matched_count":len(rows),
                "offset":start,"returned_count":len(page),
                "next_offset":start+len(page) if start+len(page)<len(rows) else None}

    def _bind_acquisition_evidence(self,payload: dict):
        urls={str(item) for item in payload.get("source_urls") or []}
        provenance=dict(payload.get("provenance") or {})
        urls.update(str(item) for item in provenance.get("model_card_urls") or [])
        records=self._research_records();matched=[]
        for record in records:
            observed=set()
            if record.get("url"):observed.add(str(record["url"]))
            observed.update(str(item.get("url")) for item in record.get("results") or []
                            if item.get("url"))
            if urls.intersection(observed):matched.append(record)
        observed_urls=set()
        for record in matched:
            if record.get("url"):observed_urls.add(str(record["url"]))
            observed_urls.update(str(item.get("url")) for item in record.get("results") or []
                                 if item.get("url"))
        missing=sorted(urls-observed_urls)
        if missing:
            raise RuntimeError("asset sources lack autonomous research evidence: "+", ".join(missing))
        provenance["authoring_context"]="autonomous_engineering_agent"
        provenance["acquisition_evidence"]=[{
            "kind":record.get("kind"),"url":record.get("url"),
            "query":record.get("query"),"record_sha256":record["record_sha256"]}
            for record in matched]
        payload=dict(payload);payload["provenance"]=provenance
        return payload

    def register_tool(self, **payload):
        # A lightweight Tool is deterministic source code by definition.
        # Learned/checkpoint-backed assets must use Capability Package, so the
        # Harness can generate this boilerplate provenance without asking the
        # model to reproduce a nested anti-cheating form on every registration.
        payload=dict(payload)
        origin=dict(payload.pop("implementation_origin",{}) or {})
        if not origin:
            origin={"kind":"original_synthesis",
                    "summary":"Agent-authored deterministic implementation; cited URLs are research background."}
        kind=str(origin.get("kind") or "")
        summary=str(origin.get("summary") or "").strip()
        if kind not in {"original_synthesis","adapted_source","adopted_source"} or not summary:
            raise RuntimeError("implementation_origin requires kind and a concrete summary")
        implementation_urls=[str(item) for item in
                             (origin.get("implementation_source_urls") or [])]
        declared_urls=set(str(item) for item in (payload.get("source_urls") or []))
        if not set(implementation_urls).issubset(declared_urls):
            raise RuntimeError("implementation source URLs must also appear in source_urls")
        if kind in {"adapted_source","adopted_source"}:
            if not implementation_urls:
                raise RuntimeError("adapted/adopted Tool requires implementation_source_urls")
            fetched={str(record.get("url")) for record in self._research_records()
                     if record.get("kind") in {"fetch","download"} and record.get("url")}
            missing=sorted(set(implementation_urls)-fetched)
            if missing:
                raise RuntimeError("implementation sources must be fetched or downloaded before attribution: "+
                                   ", ".join(missing))
        payload.setdefault("dependency_spec",{"mode":"stdlib"})
        payload.setdefault("provenance",{
            "training_data_declaration":"Deterministic source-code algorithm; no learned parameters.",
            "contamination_check":{"evaluated_benchmark":"current evaluation task",
                "method":"source and dependency inspection",
                "result":"not_applicable_source_code"}})
        payload["provenance"]={**dict(payload["provenance"]),
                               "implementation_origin":{
                                   "kind":kind,"summary":summary,
                                   "implementation_source_urls":implementation_urls,
                                   "other_source_role":"research_background"}}
        return self.capabilities.register_tool(**self._bind_acquisition_evidence(payload))

    def register_capability_package(self, **payload):
        return self.capabilities.register_package(**self._bind_acquisition_evidence(payload))

    def register_experience(self, name: str, summary: str, applicability: str,
                            keywords: list[str], evidence_refs: list[str]|None=None):
        if self.experiences is None:raise RuntimeError("shared Experience Library unavailable")
        paths=[]
        default_alias=("latest_robot_execution"
                       if (self.artifact_dir/"robot_execution.json").is_file()
                       else "previous_robot_execution")
        for value in (evidence_refs or [default_alias]):
            paths.append(str(self._authorized_evidence_path(value)))
        assessment=execution_evidence_assessment(paths)
        return self.experiences.register(name=name,summary=summary,
            applicability=applicability,keywords=keywords,evidence_paths=paths,
            evidence_assessment=assessment)

    def record_capability_gap(self, **payload):
        if self.gaps is None:raise RuntimeError("Capability Gap Library unavailable")
        payload=dict(payload)
        existing=self.gaps.latest_for_name(payload.get("name",""))
        if existing is not None:
            evidence_refs=payload.pop("evidence_refs",None)
            changes={key:value for key,value in payload.items() if key not in {"name"}}
            result=self.revise_capability_gap(existing["gap_id"],
                evidence_refs=evidence_refs,**changes)
            return {**result,"upserted_existing_gap":True,
                    "upserted_from_gap_id":existing["gap_id"]}
        payload.setdefault("hypotheses",[])
        payload.setdefault("selected_diagnosis","")
        payload.setdefault("required_capability",{})
        payload.setdefault("searched_candidates",[])
        payload.setdefault("provenance_decision",{})
        payload.setdefault("integration_result",{})
        payload.setdefault("task_validation",{})
        payload.setdefault("reuse_evidence",{})
        payload.setdefault("status","observed")
        paths=[]
        default_alias=("latest_robot_execution"
                       if (self.artifact_dir/"robot_execution.json").is_file()
                       else "previous_robot_execution")
        evidence_refs=payload.pop("evidence_refs",None) or [default_alias]
        for requested in evidence_refs:
            paths.append(self._authorized_evidence_path(requested))
        payload["task_validation"]=bind_authoritative_validation(
            payload.get("task_validation"),execution_evidence_assessment(paths))
        return self.gaps.publish(**payload,evidence_paths=paths)

    def inspect_capability_gap(self,gap_id: str):
        manifest=self.gaps.inspect(gap_id)
        result=dict(manifest);result["evidence"]=[]
        for item in manifest.get("evidence") or []:
            public={key:value for key,value in item.items() if key!="original_path"}
            result["evidence"].append({**public,
                "asset_ref":f"{gap_id}#{item['path']}"})
        return result

    def revise_capability_gap(self,previous_gap_id: str,
                              evidence_refs: list[str]|None=None,**changes):
        """Publish a partial Gap revision and retain the highest legal state."""
        if self.gaps is None:raise RuntimeError("Capability Gap Library unavailable")
        previous=self.gaps.inspect(previous_gap_id)
        allowed={"task","failure_summary","hypotheses","selected_diagnosis",
                 "required_capability","searched_candidates","provenance_decision",
                 "integration_result","task_validation","reuse_evidence","status"}
        unknown=set(changes)-allowed
        if unknown:raise RuntimeError(f"unsupported Capability Gap changes: {sorted(unknown)}")
        payload={key:previous.get(key) for key in allowed}
        payload.update(dict(changes))
        diagnosis_changed=(
            ("selected_diagnosis" in changes and
             changes.get("selected_diagnosis")!=previous.get("selected_diagnosis")) or
            ("required_capability" in changes and
             changes.get("required_capability")!=previous.get("required_capability")))
        if diagnosis_changed:
            # A changed causal explanation invalidates downstream acquisition
            # and validation records unless the model explicitly rebuilds them
            # in this same immutable revision. Never combine a new diagnosis
            # with an inherited integration claim from a different mechanism.
            for key,empty in (("searched_candidates",[]),("provenance_decision",{}),
                              ("integration_result",{}),("task_validation",{}),
                              ("reuse_evidence",{})):
                if key not in changes:payload[key]=empty
        requested=str(payload.get("status") or previous.get("status") or "observed")
        diagnosis=bool(str(payload.get("selected_diagnosis") or "").strip())
        capability=bool(payload.get("required_capability"))
        searched=bool(payload.get("searched_candidates"))
        provenance=bool(payload.get("provenance_decision"))
        integration=bool(payload.get("integration_result"))
        validation=bool(payload.get("task_validation"))
        def structurally_valid(status):
            if status=="observed":return True
            if not diagnosis or not capability:return False
            if status in {"searching","integrating","validated","rejected"} and not searched:
                return False
            if status in {"integrating","validated"} and not (provenance and integration):
                return False
            if status=="validated" and not validation:return False
            if status=="rejected" and not (provenance or integration):return False
            return True
        fallback={
            "validated":["validated","integrating","searching","diagnosed","observed"],
            "integrating":["integrating","searching","diagnosed","observed"],
            "searching":["searching","diagnosed","observed"],
            "rejected":["rejected","searching","diagnosed","observed"],
            "unresolved":["unresolved","diagnosed","observed"],
            "diagnosed":["diagnosed","observed"],
            "observed":["observed"]}.get(requested,["observed"])
        transitions=CapabilityGapLibrary.TRANSITIONS.get(previous.get("status"),set())
        if previous.get("status") not in fallback:
            fallback.append(previous.get("status"))
        payload["status"]=next(status for status in fallback
            if status in transitions and structurally_valid(status))
        paths=[]
        if evidence_refs is not None:
            for requested in evidence_refs:
                relative_match=next((item for item in previous.get("evidence") or []
                    if str(item.get("path"))==str(requested)),None)
                if relative_match is not None:
                    paths.append((self.gaps._path(previous_gap_id).parent/
                                  str(relative_match["path"])).resolve())
                else:paths.append(self._authorized_evidence_path(requested))
        else:
            latest=self._evidence_reference("latest_robot_execution")
            if not latest.is_absolute():latest=self.run_root/latest
            latest=latest.resolve()
            if latest.is_file():
                self._authorize_run_artifact(latest);paths.append(latest)
            else:
                # A new run may refine a retrieved Gap using fresh research
                # before its first robot episode. Preserve the prior revision's
                # hash-verified evidence instead of requiring nonexistent
                # current-run evidence or publishing an unsupported revision.
                previous_path=self.gaps._path(previous_gap_id).parent
                paths.extend((previous_path/str(item["path"])).resolve()
                             for item in previous.get("evidence") or [])
        payload["task_validation"]=bind_authoritative_validation(
            payload.get("task_validation"),execution_evidence_assessment(paths))
        return self.gaps.publish(name=previous["name"],previous_gap_id=previous_gap_id,
                                 evidence_paths=paths,**payload)

    def propose_skill_interface(self, **payload):
        """Persist a compositional interface candidate for a successful run."""
        payload=dict(payload)
        for key in ("preconditions","effects","required_sensors",
                    "required_robot_operations","parameters","failure_modes"):
            payload.setdefault(key,[])
        payload.setdefault("composition_notes",
            "Model-authored partial interface; the Harness augments it from the successful trace.")
        required={"preconditions":list,"effects":list,"required_sensors":list,
                  "required_robot_operations":list,"parameters":list,
                  "failure_modes":list,"composition_notes":str}
        for key,kind in required.items():
            if not isinstance(payload.get(key),kind):
                raise RuntimeError(f"Skill interface field {key} is invalid")
        allowed_operations={"observe","use","act","verify","record"}
        operations={str(item) for item in payload["required_robot_operations"]}
        if not operations.issubset(allowed_operations):
            raise RuntimeError("Skill interface names an unsupported Robot operation")
        normalized={key:payload[key] for key in required}
        temporary=self.skill_interface_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"protocol":"embodied-codex-skill-interface-v1",
            "interface":normalized},indent=2)+"\n");temporary.replace(self.skill_interface_path)
        return {"accepted":True,"path":str(self.skill_interface_path),
                "interface":normalized}

    def _lint_robot_contract(self,controller: Path):
        """Compile-time checks for literal Robot SDK contract violations.

        Dynamic action selection remains legal.  Literal misspellings and null
        opaque references are unambiguously invalid and must not consume a
        physical episode merely to discover a typed-interface error.
        """
        source=controller.read_text();tree=ast.parse(source);issues=[]
        # The runtime imports the module and calls one public entry point with
        # exactly one Robot facade.  A file can be perfectly valid Python while
        # still lacking that entry point (for example after an in-progress
        # helper rename).  Reject this before constructing a deployment so an
        # interface-only mistake never consumes a simulator or real-robot
        # episode.
        run_nodes=[node for node in tree.body
                   if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
                   and node.name=="run"]
        if not run_nodes:
            issues.append("controller must define top-level run(robot)")
        elif isinstance(run_nodes[0],ast.AsyncFunctionDef):
            issues.append("controller run(robot) must be synchronous")
        else:
            arguments=run_nodes[0].args
            positional=list(arguments.posonlyargs)+list(arguments.args)
            if (len(positional)!=1 or positional[0].arg!="robot"
                    or arguments.vararg is not None or arguments.kwarg is not None
                    or arguments.kwonlyargs or arguments.defaults):
                issues.append("controller entry point must have exact signature run(robot)")
        # ``py_compile`` proves only that Python can parse and compile the
        # file.  It does not catch a forgotten or misspelled name that would
        # otherwise waste a physical rollout.  Python's symbol table tells us
        # whether a reference resolves locally, through a closure, or at module
        # scope, so reject only globals with no module definition or builtin.
        table=symtable.symtable(source,str(controller),"exec")
        module_definitions={symbol.get_name() for symbol in table.get_symbols()
            if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
               or symbol.is_parameter()}
        runtime_globals=set(dir(builtins)) | {"__name__","__file__","__package__",
                                               "__spec__","__loader__","__builtins__"}
        unresolved=set();pending=[table]
        while pending:
            scope=pending.pop();pending.extend(scope.get_children())
            for symbol in scope.get_symbols():
                if (symbol.is_referenced() and symbol.is_global()
                        and symbol.get_name() not in module_definitions
                        and symbol.get_name() not in runtime_globals):
                    unresolved.add(symbol.get_name())
        if unresolved:
            issues.append("unresolved runtime names "+repr(sorted(unresolved)))
        for scope,line,name in _potential_unbound_local_loads(source,str(controller)):
            issues.append(
                f"line {line}: local {name!r} may be read before assignment in {scope}()")
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
                    alternatives=self.sdk_contract["actions"][action_type.value].get(
                        "any_of") or []
                    if alternatives and not any(all(key in values for key in
                                                    (option.get("required") or []))
                                                for option in alternatives):
                        required_sets=[option.get("required") or []
                                       for option in alternatives]
                        issues.append(f"line {node.lineno}: action {action_type.value!r} "
                                      f"requires one of literal field sets {required_sets}")
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

    def _require_task_fidelity(self,controller: Path):
        """Reject semantic task drift without prescribing a complete strategy."""
        if self.task_fidelity_reviewer is None or not self.task_instruction:
            return None
        source=controller.read_text()
        controller_sha=hashlib.sha256(controller.read_bytes()).hexdigest()
        instruction_sha=hashlib.sha256(self.task_instruction.encode()).hexdigest()
        path=self.workspace._path("task_fidelity_binding.json")
        binding=None
        if path.is_file():
            try:binding=json.loads(path.read_text())
            except (OSError,json.JSONDecodeError):binding=None
        if not (isinstance(binding,dict)
                and binding.get("controller_sha256")==controller_sha
                and binding.get("instruction_sha256")==instruction_sha):
            review=dict(self.task_fidelity_reviewer(
                instruction=self.task_instruction,source=source))
            binding={"protocol":"embodied-codex-task-fidelity-binding-v1",
                     "controller_sha256":controller_sha,
                     "instruction_sha256":instruction_sha,"review":review}
            self.workspace.write_file("task_fidelity_binding.json",
                json.dumps(binding,indent=2)+"\n")
        review=binding.get("review") or {}
        if review.get("approved") is not True:
            issues=review.get("issues") or ["controller pursues a different task"]
            raise RuntimeError("controller task-fidelity preflight rejected: "+
                               "; ".join(map(str,issues)))
        return binding

    def _image_artifact_ref(self,target: Path):
        target=target.resolve()
        digest=hashlib.sha256(str(target).encode()).hexdigest()[:16]
        reference=f"image-{digest}"
        self._image_artifact_refs[reference]=target
        return reference

    def view_sensor_image(self, path: str|None=None, artifact_ref: str|None=None):
        if bool(path) == bool(artifact_ref):
            raise RuntimeError("provide exactly one of path or artifact_ref")
        if artifact_ref:
            target=self._image_artifact_refs.get(str(artifact_ref))
            if target is None:raise RuntimeError("unknown image artifact_ref")
        else:target=Path(str(path))
        if not target.is_absolute():
            workspace_candidate=(self.workspace.root/target).resolve()
            target=workspace_candidate if workspace_candidate.is_file() else self.run_root/target
        target=target.resolve()
        workspace_allowed=(target==self.workspace.root or self.workspace.root in target.parents)
        if not workspace_allowed:self._authorize_run_artifact(target)
        mime=mimetypes.guess_type(target.name)[0]
        if mime not in ("image/png","image/jpeg","image/webp"):
            raise RuntimeError("view_sensor_image supports PNG, JPEG, or WEBP")
        if not target.is_file(): raise FileNotFoundError(path)
        if target.stat().st_size > 8*1024*1024: raise RuntimeError("image exceeds 8 MiB")
        source_bytes=target.read_bytes();delivery_bytes=source_bytes;delivery_mime=mime
        # Preserve immutable sensor evidence byte-for-byte on disk. Large
        # camera PNGs get a high-quality, smaller transport preview so base64
        # payloads cannot crowd the coding model's context. Compact masks and
        # images remain lossless, and the original hash is always returned.
        if mime=="image/png" and len(source_bytes)>32*1024:
            image=cv2.imread(str(target),cv2.IMREAD_COLOR)
            if image is not None:
                height,width=image.shape[:2]
                longest=max(height,width)
                if longest>192:
                    scale=192.0/longest
                    image=cv2.resize(image,(max(1,round(width*scale)),
                                            max(1,round(height*scale))),
                                     interpolation=cv2.INTER_AREA)
                ok,encoded=cv2.imencode(".jpg",image,
                    [int(cv2.IMWRITE_JPEG_QUALITY),82])
                if ok and len(encoded)<len(source_bytes)*0.8:
                    delivery_bytes=encoded.tobytes();delivery_mime="image/jpeg"
        return {"_embodied_codex_image":{
            "path":str(target),"artifact_ref":self._image_artifact_ref(target),
            "mime_type":delivery_mime,"source_mime_type":mime,
            "source_sha256":hashlib.sha256(source_bytes).hexdigest(),
            "source_bytes":len(source_bytes),"delivery_bytes":len(delivery_bytes),
            "transport_preview":delivery_bytes is not source_bytes,
            "data_base64":base64.b64encode(delivery_bytes).decode("ascii")}}

    def extract_rollout_frames(self, path: str, frame_indices: list[int],
                               max_frames: int=12):
        """Extract model-viewable keyframes from a run-local rollout video.

        A physical event can occur between explicit controller observations.
        The coding model therefore needs the same basic video-inspection
        ability as an external engineer.  Empty ``frame_indices`` requests
        uniform coverage; explicit indices support action-step-directed
        inspection.  This is read-only with respect to the rollout and exposes
        no simulator or evaluator state.  ``latest_rollout`` and
        ``previous_rollout`` resolve the video recorded in the corresponding
        immutable robot execution.  The robot-execution aliases are accepted
        as a compatibility convenience, but never authorize paths outside the
        current run.
        """
        symbolic=str(path)
        execution_alias={
            "latest_rollout":"latest_robot_execution",
            "previous_rollout":"previous_robot_execution",
            "latest_robot_execution":"latest_robot_execution",
            "previous_robot_execution":"previous_robot_execution",
        }.get(symbolic)
        if execution_alias:
            execution_path=self._evidence_reference(execution_alias)
            if not execution_path.is_absolute():execution_path=self.run_root/execution_path
            execution_path=execution_path.resolve()
            self._authorize_run_artifact(execution_path)
            if not execution_path.is_file():raise FileNotFoundError(symbolic)
            try:execution=json.loads(execution_path.read_text())
            except (OSError,json.JSONDecodeError) as exc:
                raise RuntimeError("robot execution artifact is not valid JSON") from exc
            rollout_path=(execution.get("sensor_report") or {}).get("rollout_path")
            if not isinstance(rollout_path,str) or not rollout_path.strip():
                raise RuntimeError("robot execution does not reference a rollout video")
            requested=Path(rollout_path)
        else:
            requested=Path(symbolic)
        if not requested.is_absolute():requested=self.run_root/requested
        target=requested.resolve()
        self._authorize_run_artifact(target)
        if not target.is_file() or target.suffix.lower() not in {".mp4",".avi",".mov"}:
            raise RuntimeError("rollout must be a run-local video file")
        limit=max(1,min(int(max_frames),24))
        if not isinstance(frame_indices,list) or any(
                isinstance(value,bool) or not isinstance(value,int) or value<0
                for value in frame_indices):
            raise RuntimeError("frame_indices must be a list of nonnegative integers")
        capture=cv2.VideoCapture(str(target))
        try:
            total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if total<=0:raise RuntimeError("rollout contains no readable frames")
            if frame_indices:
                indices=[]
                for value in frame_indices:
                    value=min(int(value),total-1)
                    if value not in indices:indices.append(value)
                    if len(indices)>=limit:break
            else:
                count=min(limit,total)
                indices=(sorted(set(round(i*(total-1)/max(1,count-1))
                                    for i in range(count))))
            folder=self.artifact_dir/"rollout_keyframes"/target.stem
            folder.mkdir(parents=True,exist_ok=True)
            frames=[]
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES,index)
                ok,image=capture.read()
                if not ok:continue
                destination=folder/f"frame_{index:06d}.png"
                if not cv2.imwrite(str(destination),image):
                    raise RuntimeError("failed to write rollout keyframe")
                frames.append({"index":index,"time_seconds":index/fps if fps>0 else None,
                               "artifact_ref":self._image_artifact_ref(destination),
                               "image_path":str(destination.resolve())})
        finally:capture.release()
        if not frames:raise RuntimeError("requested rollout frames were unreadable")
        return {"rollout_path":str(target),"total_frames":total,"fps":fps,
                "frames":frames,
                "next_step":"call view_sensor_image with each returned artifact_ref"}

    def read_file(self,path: str,start_line: int=1,end_line: int=400):
        """Read workspace source or an immutable text artifact in this run.

        Evidence gives the model absolute controller-snapshot paths.  Requiring
        it to mentally translate those into a different Tool/path namespace is
        needless interface friction.  Writes remain strictly workspace-relative;
        absolute reads are accepted only below the current run root.
        """
        symbolic=str(path);asset_ref="#" in symbolic
        if asset_ref:target=self._authorized_evidence_path(symbolic)
        else:
            requested=Path(symbolic)
            if not requested.is_absolute():
                return self.workspace.read_file(symbolic,start_line,end_line)
            target=requested.resolve();self._authorize_run_artifact(target)
        if not target.exists():
            return {"path":str(target),"exists":False,"start_line":1,"end_line":0,
                    "total_lines":0,"content":""}
        if not target.is_file() or target.stat().st_size>4*1024*1024:
            raise RuntimeError("run artifact is not a readable text file")
        try:lines=target.read_text().splitlines()
        except UnicodeDecodeError as exc:raise RuntimeError("run artifact is not text") from exc
        start=max(1,int(start_line));requested_end=max(start,int(end_line))
        end=min(requested_end,start+199,len(lines))
        return {"path":str(target),"exists":True,"start_line":start,
                "end_line":end,"total_lines":len(lines),
                "content":"\n".join(lines[start-1:end]),
                "content_truncated":end<min(requested_end,len(lines)),
                "next_start_line":end+1 if end<len(lines) else None}

    def read_run_artifact(self,path: str,start_line: int=1,end_line: int=400):
        requested=str(path)
        asset_ref="#" in requested
        if asset_ref:target=self._authorized_evidence_path(requested)
        elif requested in {"latest_robot_execution","previous_robot_execution"}:
            current=self.artifact_dir/"robot_execution.json"
            prior=sorted(self.run_root.glob("iterations/iteration_*/robot_execution.json"))
            if requested=="previous_robot_execution":
                prior=[item for item in prior if item.resolve()!=current.resolve()]
                target=prior[-1] if prior else current
            else:
                target=current
        elif requested=="robot_execution.json":
            target=self.artifact_dir/"robot_execution.json"
        else:target=Path(requested)
        if not target.is_absolute(): target=self.run_root/target
        target=target.resolve()
        if not asset_ref:self._authorize_run_artifact(target)
        if not target.is_file():
            if requested in {"latest_robot_execution","previous_robot_execution",
                             "robot_execution.json"}:
                return {"path":str(target),"exists":False,"start_line":1,"end_line":0,
                        "total_lines":0,"content":"","content_truncated":False,
                        "next_start_line":None}
            raise FileNotFoundError(path)
        start=max(1,int(start_line));end=max(start,min(int(end_line),start+999))
        selected=[];total=0;returned_end=start-1;characters=0;truncated=False
        with target.open(errors="replace") as stream:
            for total,line in enumerate(stream,start=1):
                if total<start or total>end:continue
                text=line.rstrip("\r\n")
                remaining=64*1024-characters
                if remaining<=0:truncated=True;continue
                if len(text)>remaining:
                    selected.append(text[:remaining]);characters+=remaining;truncated=True
                else:
                    selected.append(text);characters+=len(text)+1
                returned_end=total
        return {"path":str(target),"exists":True,"start_line":start,"end_line":returned_end,
                "total_lines":total,"content":"\n".join(selected),
                "content_truncated":truncated,
                "next_start_line":returned_end+1 if (truncated or returned_end<total) else None}

    def list_sensor_artifacts(self, pattern: str="episodes/**/*"):
        pattern=str(pattern or "episodes/**/*")
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise RuntimeError("artifact pattern must stay inside the current run")
        items=[]
        for path in self.run_root.glob(pattern):
            if path.is_file():
                try:self._authorize_run_artifact(path)
                except RuntimeError:continue
                items.append(str(path.resolve()))
        return sorted(items)[-2000:]

    def inspect_robot_sdk_contract(self, section: str=""):
        """Return the authoritative SDK contract, optionally at one dotted path.

        The task prompt carries a compact index so ordinary controller work
        does not pay for every example and field description on every model
        call.  This method is the lossless, on-demand manual for unfamiliar or
        rejected SDK operations.
        """
        value=self.sdk_contract
        normalized=str(section or "").strip(".")
        if normalized:
            for key in normalized.split("."):
                if not isinstance(value,dict) or key not in value:
                    raise RuntimeError(f"unknown Robot SDK contract section {normalized!r}")
                value=value[key]
        return {"protocol":self.sdk_contract.get("protocol"),
                "section":normalized or "root","contract":value}

    def _review_capability_integration(self, controller: Path, gap: Mapping[str,Any],
                                       tool_ids: list[str]):
        """Bind an independent causal review to exact Gap, Tool, and Controller bytes."""
        bundles=[]
        for tool_id in sorted(set(tool_ids)):
            inspected=self.capabilities.inspect(tool_id)
            manual_reader=getattr(self.capabilities,"manual",None)
            manual=(manual_reader(tool_id) if callable(manual_reader) else {})
            bundles.append({"manifest":inspected.get("manifest") or {},
                            "manual":manual,
                            "source":inspected.get("source") or ""})
        binding_payload={"gap":dict(gap),"tools":bundles,
                         "controller_sha256":hashlib.sha256(controller.read_bytes()).hexdigest()}
        binding_sha=hashlib.sha256(json.dumps(binding_payload,sort_keys=True,
            separators=(",",":"),default=str).encode()).hexdigest()
        binding_path=self.workspace._path("capability_integration_binding.json")
        binding=None
        if binding_path.is_file():
            try:binding=json.loads(binding_path.read_text())
            except (OSError,json.JSONDecodeError):binding=None
        if not isinstance(binding,dict) or binding.get("binding_sha256")!=binding_sha:
            if self.acquisition_reviewer is None:
                review={"approved":True,"approved_tool_ids":sorted(set(tool_ids)),
                        "covered_requirements":["reviewer_not_configured"],"issues":[],
                        "review_mode":"structural_fallback"}
            else:
                review=dict(self.acquisition_reviewer(gap=dict(gap),tools=bundles,
                    controller_source=controller.read_text()))
            binding={"protocol":"embodied-codex-capability-integration-binding-v1",
                     "binding_sha256":binding_sha,"controller_sha256":binding_payload[
                         "controller_sha256"],"gap_id":gap.get("gap_id"),
                     "tool_ids":sorted(set(tool_ids)),"review":review}
            self.workspace.write_file("capability_integration_binding.json",
                                      json.dumps(binding,indent=2,default=str)+"\n")
        review=binding.get("review") or {}
        self.capability_integration_preflight=binding
        if review.get("approved") is not True:
            issues=review.get("issues") or ["Tool/Controller does not satisfy the Capability Gap"]
            raise RuntimeError("capability integration preflight rejected: "+
                               "; ".join(str(item) for item in issues))
        approved=set(str(item) for item in review.get("approved_tool_ids") or [])
        if not approved.intersection(tool_ids):
            raise RuntimeError("capability integration preflight approved no bound candidate Tool")
        return binding

    def _require_capability_acquisition(self,controller: Path):
        """Escalate a persistent diagnosed Gap before another physical trial.

        Early hypothesis tests remain unrestricted. Once the same Gap has an
        evidence-backed follow-up revision, another rollout must either bind a
        newly tested Tool or turn that Gap into an audited acquisition record.
        This keeps task algorithms out of one-off Controller patches and makes
        the Internet-search/asset-growth claim mechanically inspectable.
        """
        gap_id=self.required_acquisition_gap_id
        if not gap_id:return None
        if self.gaps is None:
            raise RuntimeError("capability acquisition gate requires a Gap Library")
        original=self.gaps.inspect(gap_id);latest=self.gaps.latest_for_name(original["name"])
        review_gap=latest or original
        current_tools=_controller_tool_ids(controller)
        new_tools=current_tools-self.acquisition_baseline_tool_ids
        tested_new=[]
        for tool_id in sorted(new_tools):
            try:manifest=self.capabilities.inspect(tool_id)["manifest"]
            except (AssetError,FileNotFoundError,KeyError):continue
            if manifest.get("status")=="tested":tested_new.append(tool_id)
        if tested_new:
            return self._review_capability_integration(controller,review_gap,tested_new)
        if (latest and latest.get("status")=="validated"
                and (latest.get("task_validation") or {}).get("authoritative_outcome")=="success"):
            return {"approved":True,"source":"previous authoritative Gap validation",
                    "gap_id":latest.get("gap_id")}
        raise RuntimeError(
            "capability_acquisition_required: repeated evidence has already revised "
            f"{gap_id} as a persistent diagnosed Capability Gap. Before another robot "
            "episode, call search_web, inspect credible public candidates, and revise "
            "this same Gap lineage to status=searching with the actual query/candidates; "
            "then acquire, test, and bind a reusable Tool. Searching or changing a Gap to "
            "searching/integrating is durable progress, but does not by itself authorize another "
            "physical episode. The bound Tool and Controller must also pass the independent "
            "Capability-Gap integration review. Do not hide the missing capability in another "
            "one-off Controller algorithm.")

    def run_robot_controller(self, path: str):
        if self.robot_runs: raise RuntimeError("one robot episode per iteration")
        controller=self.workspace._path(path)
        acquisition_preflight=self._require_capability_acquisition(controller)
        controller_semantic_sha256=_controller_semantic_sha256(controller)
        if controller_semantic_sha256==self.rejected_controller_semantic_sha256:
            raise RuntimeError(
                "unchanged_controller_after_failed_episode: this Controller AST already "
                "produced sensor-only failure. Modify executable Controller behavior "
                "or its Tool binding before consuming another robot episode; unchanged programs "
                "are reserved for validation after sensor success.")
        strategy_sha256=_controller_strategy_sha256(controller)
        repeated=None;new_tools=set()
        for rejected_sha256,entry in self.rejected_controller_strategy_failures.items():
            prefix_count=entry.get("robot_event_count")
            if prefix_count is None:
                matches=strategy_sha256==rejected_sha256
                current_tools=_controller_tool_ids(controller)
            else:
                current_prefix=_controller_strategy_prefix_sha256(controller,prefix_count)
                expected=str(entry.get("strategy_prefix_sha256") or rejected_sha256)
                matches=current_prefix==expected
                current_tools=_controller_tool_ids_before_robot_event(
                    controller,prefix_count)
            if not matches:continue
            prior_tools=set(entry.get("prior_tool_ids") or [])
            candidate_new_tools=current_tools-prior_tools
            if candidate_new_tools:continue
            repeated=entry;new_tools=candidate_new_tools;break
        if repeated and not new_tools:
            raise RuntimeError(
                "repeated_strategy_after_failed_episodes: this robot-facing failure-prefix "
                "Tool/action/control-flow strategy already failed repeatedly with sensor mechanism(s): "
                f"{repeated.get('failures') or []}. Numeric offsets, timeouts, comments, and one-candidate "
                "parameter or unreachable downstream changes are not a new strategy. Acquire or bind a new capability before the failed stage, "
                "change the action/control-flow family, or implement sensor-verified candidate "
                "retry before consuming another robot episode. A genuinely new Tool ID also "
                "unlocks one causal integration trial.")
        contract_lint=self._lint_robot_contract(controller)
        fidelity_preflight=self._require_task_fidelity(controller)
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
            for tool_id,function in functions.items():
                manifest=self.capabilities.inspect(tool_id)["manifest"]
                contract={key:manifest[key] for key in ("input_schema","output_schema")}
                register(tool_id,function,contract)
            execution=self.runtime.execute(controller,deployment)
            sensor_report=dict(deployment.sensor_report(execution))
        finally: deployment.close()
        final_verify=(execution["rpc_events"][-1] if execution["rpc_events"] else {})
        passed=(execution.get("completed") is True
                and final_verify.get("method")=="verify"
                and isinstance(final_verify.get("result"),dict)
                and final_verify["result"].get("verified") is True
                and sensor_report.get("sensor_verification_passed") is True)
        transient=transient_infrastructure_failure(execution,sensor_report)
        result={"controller_path":path,"controller_snapshot":str(controller_snapshot.resolve()),
                "execution":execution,"sensor_report":sensor_report,
                "sensor_success_candidate":passed,"task_model_preflight":preflight,
                "task_fidelity_preflight":fidelity_preflight,
                "capability_integration_preflight":acquisition_preflight,
                "robot_contract_preflight":contract_lint}
        if transient is not None:result["transient_infrastructure_failure"]=transient
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
        def workspace_engineering_progress(arguments):
            suffix=Path(str(arguments.get("path") or "")).suffix.lower()
            return suffix not in {".md",".txt",".log"}
        def workspace_read_policy(arguments):
            path=str(arguments.get("path") or "")
            return ("working_memory" if path and not Path(path).is_absolute()
                    and "#" not in path else "read_once")
        def edit_with_semantic_receipt(method, arguments):
            path=str(arguments.get("path") or "")
            target=self.workspace.root/path if path else None
            before=None
            if path=="controller.py" and target is not None and target.is_file():
                try:before=_controller_semantic_sha256(target)
                except (OSError,SyntaxError):before=None
            result=method(**arguments)
            if path=="controller.py":
                try:after=_controller_semantic_sha256(target)
                except (OSError,SyntaxError):after="invalid-controller-source"
                changed=before!=after
                result={**result,"controller_semantic_progress":changed,
                        "_embodied_codex_semantic_progress":changed}
            return result
        r.add("list_files","List files in the persistent task workspace.",
              _obj({"pattern":string},[]),self.workspace.list_files,
              evidence_policy="read_once",evidence_group="workspace")
        r.add("read_file","Read a relative workspace file or an absolute immutable text artifact inside this run.",
              _obj({"path":string,"start_line":integer,"end_line":integer},["path"]),
              self.read_file,evidence_policy=workspace_read_policy,
              evidence_group="workspace")
        r.add("write_file","Create or fully rewrite any workspace file.",
              _obj({"path":string,"content":string},["path","content"]),
              lambda **arguments:edit_with_semantic_receipt(
                  self.workspace.write_file,arguments),
              evidence_policy="invalidates_reads",invalidates_evidence_groups=("workspace",),
              evidence_progress=workspace_engineering_progress,
              execution_progress=workspace_engineering_progress)
        r.add("replace_in_file","Replace one exact region in a workspace file.",
              _obj({"path":string,"old":string,"new":string},["path","old","new"]),
              lambda **arguments:edit_with_semantic_receipt(
                  self.workspace.replace_in_file,arguments),
              evidence_policy="invalidates_reads",
              invalidates_evidence_groups=("workspace",),
              evidence_progress=workspace_engineering_progress,
              execution_progress=workspace_engineering_progress)
        r.add("replace_file_lines",("Replace an inclusive line range in a workspace file. "
              "Use this after read_file when exact-string replacement would require copying "
              "a large existing block; optionally supply a SHA-256 stale-read guard."),
              _obj({"path":string,"start_line":integer,"end_line":integer,
                    "new_content":string,"expected_old_sha256":string},
                   ["path","start_line","end_line","new_content"]),
              lambda **arguments:edit_with_semantic_receipt(
                  self.workspace.replace_file_lines,arguments),
              evidence_policy="invalidates_reads",
              invalidates_evidence_groups=("workspace",),
              evidence_progress=workspace_engineering_progress,
              execution_progress=workspace_engineering_progress)
        r.add("run_command","Run an argv command in the workspace; use it for tests and engineering.",
              _obj({"argv":{"type":"array","items":string},"timeout_seconds":{"type":"number"}},
                   ["argv"]),self.workspace.run_command,
              evidence_policy="budgeted_output",evidence_group="workspace")
        r.add("restore_previous_executed_controller",("Restore the newest immutable Controller "
              "snapshot that actually produced a robot_execution ledger. Use this after a broken "
              "edit or failed composition instead of shell-copying hidden run artifacts or "
              "reconstructing a large Controller page by page."),
              _obj({"destination":string},[]),self.restore_previous_executed_controller,
              evidence_policy="invalidates_reads",
              invalidates_evidence_groups=("workspace",),evidence_progress=True,
              execution_progress=True)
        r.add("search_web","Search public internet resources and repositories. Results are "
              "recorded as immutable provenance evidence for later asset registration.",
              _obj({"query":string,"limit":integer},["query"]),self.search_web,
              evidence_policy="read_once",evidence_group="web",
              post_mutation_read_allowed=True)
        r.add("fetch_web_page","Read a public HTTP(S) page returned by web search.",
              _obj({"url":string,"max_chars":integer},["url"]),self.fetch_web_page,
              evidence_policy="read_once",evidence_group="web",
              post_mutation_read_allowed=True)
        r.add("download_public_asset","Download a public HTTP(S) repository archive, model, "
              "checkpoint, or dependency into the workspace through the audited network broker. "
              "run_command is network-isolated; use it only after this call to verify/unpack/build. "
              "Provide expected_sha256 when the publisher documents one, otherwise verify the "
              "returned hash before registration.",
              _obj({"url":string,"destination":string,
                    "max_bytes":{"type":"integer","minimum":1,"maximum":8589934592},
                    "expected_sha256":string},
                   ["url","destination","max_bytes"]),
              self.download_public_asset,evidence_policy="invalidates_reads",
              invalidates_evidence_groups=("workspace",),evidence_progress=True)
        r.add("view_sensor_image","Visually inspect a PNG/JPEG/WEBP sensor frame or mask from this run. "
              "When another Tool returns artifact_ref, pass that short opaque value unchanged; use "
              "path only for an existing sensor path that was not returned as a reference.",
              _obj({"path":string,"artifact_ref":string},[]),self.view_sensor_image,
              evidence_policy="image_twice",evidence_group="run")
        r.add("extract_rollout_frames",
              "Extract run-local rollout video keyframes for visual diagnosis. Prefer "
              "path=latest_rollout (previous_rollout for the prior episode). The compatible "
              "robot-execution aliases are also resolved to their recorded rollout. Pass an empty "
              "frame_indices list for uniform coverage, or use approximate action step/3 "
              "indices around contact, close, lift, release, and recovery; then inspect the "
              "returned artifact_ref values with view_sensor_image; do not reconstruct paths.",
              _obj({"path":string,
                    "frame_indices":{"type":"array","items":integer},
                    "max_frames":integer},["path","frame_indices"]),
              self.extract_rollout_frames,evidence_policy="read_once",evidence_group="run")
        r.add("list_sensor_artifacts","List sensor frames, masks, traces, and rollouts in this run.",
              _obj({"pattern":string},[]),self.list_sensor_artifacts,
              evidence_policy="read_once",evidence_group="run")
        r.add("read_run_artifact","Read a text/JSON/log artifact from this run with line numbers; "
              "use path=latest_robot_execution for current evidence, or "
              "previous_robot_execution before this iteration has executed. Hash-verified "
              "Gap, Experience, and Skill evidence asset_ref values are also accepted.",
              _obj({"path":string,"start_line":integer,"end_line":integer},
                   ["path"]),self.read_run_artifact,evidence_policy="read_once",
              evidence_group="run")
        r.add("inspect_execution_event","Inspect one indexed RPC event from a robot execution "
              "with large candidate arrays compacted. Prefer this over paging through the full "
              "robot_execution.json after the compact run_robot_controller summary identifies "
              "the relevant event_index.",
              _obj({"path":string,"event_index":integer,
                    "max_list_items":{"type":"integer","minimum":1,"maximum":20}},
                   ["path","event_index"]),self.inspect_execution_event,
              evidence_policy="read_once",evidence_group="run")
        filter_schema={"type":"object","properties":{"field":string,
            "op":{"type":"string","enum":["eq","ne","lt","lte","gt","gte"]},
            "value":{}},"required":["field","op","value"],"additionalProperties":False}
        r.add("query_run_json","Query, filter, sort, project, and paginate an array in a "
              "run-local JSON artifact. For latest_robot_execution and previous_robot_execution, "
              "the robot event array is exactly /execution/rpc_events; /rpc_events, "
              "/controller_records, and /execution/controller_records do not exist. Prefer "
              "inspect_execution_event when an event_index is known. For detector or planner "
              "artifacts, use their actual RFC 6901 array pointer and optional sort_by. "
              "Filter, sort, and projection fields accept either a top-level key such as "
              "method or an RFC 6901 pointer such as /arguments/tool_id or /result/reached. "
              "RPC arrays are heterogeneous: absent fields return null together with an "
              "explicit projection_warnings entry; filter by method when uniform rows are needed.",
              _obj({"path":string,"array_pointer":string,
                    "filters":{"type":"array","items":filter_schema},
                    "sort_by":string,"descending":{"type":"boolean"},
                    "fields":{"type":"array","items":string},
                    "offset":integer,"limit":integer},["path","array_pointer"]),
              self.query_run_json,evidence_policy="read_once",evidence_group="run")
        manual_schema={"type":"object","properties":{
            "purpose":string,"when_to_use":{"type":"array","items":string},
            "inputs":free,"outputs":free,"examples":{"type":"array","items":free},
            "failure_modes":{"type":"array","items":string},
            "limitations":{"type":"array","items":string}},
            "required":["purpose","when_to_use","inputs","outputs","examples",
                        "failure_modes","limitations"],"additionalProperties":False}
        dependency_schema={"type":"object","properties":{
            "mode":{"type":"string","enum":["stdlib","vendored"]},
            "requirements_lock_path":string,"vendor_path":string},
            "required":["mode"],"additionalProperties":False}
        contamination_schema={"type":"object","properties":{
            "evaluated_benchmark":{"type":"string","minLength":1},
            "method":{"type":"string","minLength":1},
            "result":{"type":"string","enum":[
                "no_declared_overlap","not_applicable_source_code"]}},
            "required":["evaluated_benchmark","method","result"],
            "additionalProperties":False}
        # Learned/checkpoint packages retain an explicit provenance contract.
        # Lightweight Tools are deterministic by construction and receive the
        # corresponding boilerplate record in register_tool().
        package_provenance_schema={"type":"object","properties":{
            "training_data_declaration":{"type":"string","minLength":1},
            "contamination_check":contamination_schema,
            "checkpoint_sha256":{"type":"object",
                "propertyNames":{"type":"string","minLength":1},
                "additionalProperties":{"type":"string",
                    "pattern":"^[0-9a-fA-F]{64}$"}},
            "models":{"type":"array","minItems":1,"items":string,
                "description":"Actual learned model identifiers; omit for deterministic packages."},
            "model":{"type":"string","minLength":1,
                "description":"Actual learned model identifier; omit for deterministic packages."},
            "model_card_urls":{"type":"array","items":{
                "type":"string","pattern":"^https://"}}},
            "required":["training_data_declaration","contamination_check"],
            "additionalProperties":True}
        source_urls_schema={"type":"array","minItems":1,"uniqueItems":True,
                            "items":{"type":"string","pattern":"^https://"}}
        implementation_origin_schema={"type":"object","properties":{
            "kind":{"type":"string","enum":["original_synthesis","adapted_source",
                                                   "adopted_source"]},
            "summary":{"type":"string","minLength":12},
            "implementation_source_urls":source_urls_schema},
            "required":["kind","summary"],"additionalProperties":False}
        r.add("list_research_sources","Search and paginate exact public HTTPS URLs observed by "
              "this run and eligible for asset provenance. Results are newest-first and default "
              "to 20 entries. Prefer query over paging through unrelated history; copy URLs from "
              "the result instead of reconstructing or guessing them.",
              _obj({"query":string,"offset":{"type":"integer","minimum":0},
                    "limit":{"type":"integer","minimum":1,"maximum":50}},[]),
              self.list_research_sources)
        r.add("register_tool","Freeze a small deterministic workspace implementation defining "
              "exactly one top-level def run(payload) as a versioned Tool. The Harness generates "
              "the stdlib dependency record, deterministic provenance, and a schema-consistent "
              "baseline manual. source_urls must be copied from list_research_sources. Declare "
              "whether the code is original synthesis, adapted, or adopted; adapted/adopted "
              "implementation sources must first be fetched or downloaded. Search-only URLs are "
              "research background, not proof of implementation origin. Supply manual only when "
              "a richer schema-exact manual is useful.",
              _obj({"name":string,"source_path":string,"description":string,
                    "input_schema":free,"output_schema":free,
                    "source_urls":source_urls_schema,
                    "implementation_origin":implementation_origin_schema,
                    "trained_on_current_task":{"type":"boolean"},"manual":manual_schema,
                    "dependency_spec":dependency_schema},
                   ["name","source_path","description","input_schema","output_schema",
                    "source_urls","implementation_origin","trained_on_current_task"]),
              self.register_tool,evidence_policy="invalidates_reads",
              invalidates_evidence_groups=("assets",),evidence_progress=True)
        package_spec={"type":"object","properties":{
            "kind":{"type":"string","enum":["algorithm","model","perception",
                "planner","policy","service"]},
            "entrypoint":string,"accelerator":{"type":"string","enum":["cpu","cuda"]},
            "network":{"type":"boolean"},"timeout_seconds":{"type":"number"},
            "runtime_requirements":{"type":"array","items":string}},
            "required":["kind","entrypoint","accelerator","network","timeout_seconds"],
            "additionalProperties":False}
        r.add("register_capability_package",
              "Freeze an acquired repository/model/planner or self-contained service wrapper as "
              "a versioned, per-invocation network-isolated JSON Tool. Learned packages require "
              "verified checkpoint hashes; the entrypoint defines run(payload) and may request "
              "CPU or CUDA. Host ROS and robot services must be Adapter-owned Tools because "
              "their IPC and safety boundary belong to the deployment.",
              _obj({"name":string,"bundle_path":string,"description":string,
                    "input_schema":free,"output_schema":free,
                    "source_urls":source_urls_schema,
                    "trained_on_current_task":{"type":"boolean"},"manual":manual_schema,
                    "provenance":package_provenance_schema,"package_spec":package_spec},
                   ["name","bundle_path","description","input_schema","output_schema",
                    "source_urls","trained_on_current_task","provenance","package_spec"]),
              self.register_capability_package,evidence_policy="invalidates_reads",
              invalidates_evidence_groups=("assets",),evidence_progress=True)
        test_case={"type":"object","properties":{"input":free,"expected":{}},
                   "required":["input","expected"],"additionalProperties":False}
        r.add("test_tool","Run deterministic exact-output cases formatted as "
              "{input: <payload passed to run(payload)>, expected: <exact return value>}; "
              "only Tools whose every case passes become deployable.",
              _obj({"tool_id":string,"cases":{"type":"array","items":test_case}},["tool_id","cases"]),
              self.capabilities.test_tool,evidence_policy="invalidates_reads",
              invalidates_evidence_groups=("assets",),evidence_progress=True)
        r.add("list_tools","List bounded Tool contract summaries and status; only status=tested "
              "is deployable. Use search_assets for task-relevant discovery, then inspect_tool "
              "for one selected Tool's manual.",
              _obj({},[]),self.list_available_tools,evidence_policy="read_once",
              evidence_group="assets")
        r.add("search_assets","Retrieve the most relevant Tool, Skill, and Experience indices. "
              "Use this instead of enumerating an entire growing library. Results are capped "
              "at 20 per asset type; issue a narrower query instead of requesting a bulk dump.",
              _obj({"query":string,"asset_types":{"type":"array","items":{
                    "type":"string","enum":["tool","skill","experience","gap"]}},
                    "limit":{"type":"integer","minimum":1,"maximum":20}},
                   ["query","asset_types"]),self.search_assets,
              evidence_policy="read_once",evidence_group="assets")
        r.add("inspect_tool","Read one Tool's dedicated manual, provenance, and test metadata; "
              "implementation source is intentionally excluded.",
              _obj({"tool_id":string},["tool_id"]),self.inspect_tool,
              evidence_policy="read_once",evidence_group="assets")
        r.add("read_tool_source","Explicitly read a bounded implementation-source page only when "
              "runtime evidence contradicts the manual or the Tool must be modified/replaced.",
              _obj({"tool_id":string,"start_line":integer,"end_line":integer},
                   ["tool_id"]),self.read_tool_source,evidence_policy="read_once",
              evidence_group="assets")
        r.add("revise_tool_manual","Publish an evidence-backed new manual revision after proving "
              "the existing manual incomplete or wrong; Tool implementation stays immutable.",
              _obj({"tool_id":string,"manual":manual_schema,
                    "evidence_refs":{"type":"array","items":string}},
                   ["tool_id","manual","evidence_refs"]),self.revise_tool_manual,
              evidence_policy="invalidates_reads",invalidates_evidence_groups=("assets",))
        if self.experiences is not None:
            r.add("inspect_experience","Inspect one immutable Experience and its evidence hashes. "
                  "Read evidence using the returned asset_ref, never original_path.",
                  _obj({"experience_id":string},["experience_id"]),self.inspect_experience,
                  evidence_policy="read_once",evidence_group="assets")
            r.add("register_experience","Publish a reusable model-authored lesson backed by files "
                  "from this run. The Harness binds latest_robot_execution when evidence_refs "
                  "is omitted. The Harness derives success_evidence/failure_evidence from the "
                  "referenced execution; model prose cannot promote failed evidence to success.",
                  _obj({"name":string,"summary":string,"applicability":string,
                        "keywords":{"type":"array","items":string},
                        "evidence_refs":{"type":"array","items":string}},
                       ["name","summary","applicability","keywords"]),
                  self.register_experience,evidence_policy="invalidates_reads",
                  invalidates_evidence_groups=("assets",),evidence_progress=True)
        if self.gaps is not None:
            gap_mapping={"type":"object","additionalProperties":True}
            r.add("inspect_capability_gap","Inspect one immutable capability-gap revision and "
                  "its hash-verified evidence. Reuse an evidence asset via its returned asset_ref, "
                  "not the relative evidence path alone.",
                  _obj({"gap_id":string},["gap_id"]),self.inspect_capability_gap,
                  evidence_policy="read_once",evidence_group="assets")
            r.add("record_capability_gap","Publish a new immutable capability-gap revision linking "
                  "an observed failure to a capability need. Only name, task, and failure_summary "
                  "are required. If that normalized name already exists, the Harness atomically "
                  "revises its latest version; the model does not manage version IDs. Omitted "
                  "lifecycle fields default to an observed empty state for a new Gap, and "
                  "the Harness automatically binds latest_robot_execution. Add a diagnosis and "
                  "capability contract when status is diagnosed; use revisions for later search, "
                  "integration, and validation evidence.",
                  _obj({"name":string,"task":string,"failure_summary":string,
                        "hypotheses":{"type":"array","items":string},
                        "selected_diagnosis":string,"required_capability":gap_mapping,
                        "searched_candidates":{"type":"array","items":gap_mapping},
                        "provenance_decision":gap_mapping,"integration_result":gap_mapping,
                        "task_validation":gap_mapping,"reuse_evidence":gap_mapping,
                        "status":{"type":"string","enum":sorted(CapabilityGapLibrary.STATUSES)},
                        "evidence_refs":{"type":"array","items":string}},
                       ["name","task","failure_summary"]),self.record_capability_gap,
                  evidence_policy="invalidates_reads",invalidates_evidence_groups=("assets",))
            gap_changes={"type":"object","properties":{
                "task":string,"failure_summary":string,
                "hypotheses":{"type":"array","items":string},
                "selected_diagnosis":string,"required_capability":gap_mapping,
                "searched_candidates":{"type":"array","items":gap_mapping},
                "provenance_decision":gap_mapping,"integration_result":gap_mapping,
                "task_validation":gap_mapping,"reuse_evidence":gap_mapping,
                "status":{"type":"string","enum":sorted(CapabilityGapLibrary.STATUSES)}},
                "additionalProperties":False}
            r.add("revise_capability_gap","Incrementally revise an existing Capability Gap by "
                  "passing only changed fields at top level. Omitted fields are copied from the "
                  "prior revision. The latest execution is bound when available; before a new "
                  "run's first episode, the prior revision's verified evidence is retained. If a requested "
                  "status lacks required search/integration/validation evidence, the Harness "
                  "keeps the highest legal earlier status instead of rejecting the revision.",
                  _obj({"previous_gap_id":string,
                        "evidence_refs":{"type":"array","items":string},
                        **gap_changes["properties"]},
                       ["previous_gap_id"]),
                  self.revise_capability_gap,evidence_policy="invalidates_reads",
                  invalidates_evidence_groups=("assets",),evidence_progress=True)
        if self.skills is not None:
            r.add("inspect_skill","Inspect a retrieved Skill's structured composition interface, "
                  "manifest, and hash-verified evidence asset_ref values.",
                  _obj({"skill_id":string},["skill_id"]),self.inspect_skill,
                  evidence_policy="read_once",evidence_group="assets")
            r.add("read_skill_source","Read a bounded page of a retrieved Skill controller for composition. "
                  "Pass only an exact skill_id returned by search_assets; never use placeholders such as none.",
                  _obj({"skill_id":string,"start_line":integer,"end_line":integer},
                       ["skill_id"]),self.read_skill_source,
                  evidence_policy="read_once",evidence_group="assets")
            r.add("checkout_skill_controller","Copy one hash-verified frozen Skill controller into the "
                  "persistent workspace as editable source. Use this when a retrieved Skill is the closest "
                  "starting point; then inspect and modify task-relevant behavior instead of repeatedly "
                  "paging or manually reproducing its full source.",
                  _obj({"skill_id":string,"destination":string},["skill_id"]),
                  self.checkout_skill_controller,evidence_policy="invalidates_reads",
                  invalidates_evidence_groups=("workspace",),evidence_progress=True,
                  execution_progress=True)
            string_list={"type":"array","items":string}
            r.add("propose_skill_interface","After obtaining execution evidence, describe the reusable "
                  "Skill contract that should be frozen if this controller succeeds. Actual Tool dependencies "
                  "and robot operations are checked against the rollout by the Harness.",
                  _obj({"preconditions":string_list,"effects":string_list,
                        "required_sensors":string_list,
                        "required_robot_operations":{"type":"array","items":{"type":"string",
                            "enum":["observe","use","act","verify","record"]}},
                        "parameters":{"type":"array","items":free},
                        "failure_modes":string_list,"composition_notes":string},
                       []),self.propose_skill_interface)
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
        r.add("inspect_robot_sdk_contract",
              "Read the authoritative Robot SDK manual at an optional dotted section such as "
              "actions.move_to_pose or verifiers.visual_attachment. Use the compact task-prompt "
              "index by default and inspect only a section when exact optional semantics are needed.",
              _obj({"section":string},[]),self.inspect_robot_sdk_contract,
              evidence_policy="read_once",evidence_group="workspace")
        r.add("run_robot_controller","Run one arbitrary controller.py in a fresh sensor-only episode.",
              _obj({"path":string},["path"]),self.run_robot_controller,
              available=lambda:self.robot_runs==0,evidence_policy="invalidates_reads",
              invalidates_evidence_groups=("run",))
        return r

__all__ = ["EngineeringSurface"]
