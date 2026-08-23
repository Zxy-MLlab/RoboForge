"""Machine-auditable conformance checks for Embodied Codex experiment runs."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str,Any]:
    return json.loads(path.read_text())


def _trace(path: Path) -> list[dict[str,Any]]:
    rows=[]
    if not path.is_file():return rows
    for number,line in enumerate(path.read_text(errors="replace").splitlines(),1):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"type":"trace_decode_error","line":number})
    return rows


def _controller_forbidden(path: Path) -> list[str]:
    if not path.is_file():return ["missing_controller"]
    try:tree=ast.parse(path.read_text())
    except SyntaxError as exc:return [f"syntax_error:{exc.lineno}"]
    forbidden={"reward","done","check_success","task_success","evaluator",
               "bddl","object_pose","object_id","initial_state_index"}
    found=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Name) and node.id.casefold() in forbidden:found.add(node.id)
        if isinstance(node,ast.Attribute) and node.attr.casefold() in forbidden:found.add(node.attr)
    return sorted(found)


def _tool_assets(root: Path) -> tuple[bool,list[str],dict[str,int]]:
    errors=[];counts={"registered":0,"tested":0,"test_failed":0}
    configuration=root/"harness_configuration.json"
    capability_root=root/"capabilities"
    if configuration.is_file():
        try:capability_root=Path(_json(configuration)["capability_root"]).resolve()
        except Exception as exc:
            return False,[f"invalid_harness_configuration:{type(exc).__name__}"],counts
    for manifest_path in capability_root.glob("*/v*/manifest.json"):
        try:manifest=_json(manifest_path)
        except Exception as exc:
            errors.append(f"invalid_manifest:{manifest_path}:{type(exc).__name__}");continue
        status=str(manifest.get("status") or "registered")
        counts[status]=counts.get(status,0)+1
        source=manifest_path.parent/"tool.py"
        expected=manifest.get("source_sha256")
        if not source.is_file():errors.append(f"missing_tool_source:{manifest.get('tool_id')}")
        elif expected and hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            errors.append(f"tool_hash_mismatch:{manifest.get('tool_id')}")
    return not errors,errors,counts


def audit_run(run_root: str|Path) -> dict[str,Any]:
    root=Path(run_root).resolve();state_path=root/"state.json"
    state=_json(state_path) if state_path.is_file() else {}
    iteration_dirs=sorted((root/"iterations").glob("iteration_*"))
    traces=[event for directory in iteration_dirs
            for event in _trace(directory/"agent_trace.jsonl")]
    executions=[]
    for directory in iteration_dirs:
        path=directory/"robot_execution.json"
        if path.is_file():executions.append((directory,path,_json(path)))

    interface_errors=[];recovered_model_errors=[];recovered_tool_errors=[]
    for index,event in enumerate(traces):
        if event.get("type")=="model_error":
            # CodingAgent has bounded retries.  A transient transport failure
            # followed by a model response in the same task/pass demonstrates
            # that the recovery path worked; retain it diagnostically without
            # misclassifying the whole robot Harness as broken.
            recovered=False
            for later in traces[index+1:]:
                if later.get("type")=="task":break
                if (later.get("type")=="model"
                        and later.get("turn")==event.get("turn")):
                    recovered=True;break
            record={key:event.get(key) for key in
                    ("type","turn","attempt","error","line")}
            (recovered_model_errors if recovered else interface_errors).append(record)
        elif event.get("type")=="trace_decode_error":
            interface_errors.append({key:event.get(key) for key in
                                     ("type","turn","attempt","error","line")})
        if event.get("type")=="tool_result" and event.get("ok") is False:
            segment_start=max((position for position in range(index+1)
                               if traces[position].get("type")=="task"),default=0)
            segment_end=next((position for position in range(index+1,len(traces))
                              if traces[position].get("type")=="task"),len(traces))
            # A model can issue multiple independent calls in one response.
            # If one malformed path fails while another same-Tool call in that
            # exact task/pass succeeds and the loop continues, retain the typo
            # as recovered diagnostic evidence instead of declaring the whole
            # Harness broken.  Never match across iteration/task boundaries.
            recovered=any(
                candidate.get("type")=="tool_result"
                and candidate.get("name")==event.get("name")
                and candidate.get("ok") is True
                and (position>index or candidate.get("turn")==event.get("turn"))
                for position,candidate in enumerate(
                    traces[segment_start:segment_end],start=segment_start))
            record={"type":"tool_error","tool":event.get("name"),
                    "error":event.get("error")}
            (recovered_tool_errors if recovered else interface_errors).append(record)

    controller_errors=[];recovered_controller_errors=[];forbidden=[];contract_pass=True;sensor_reports=[]
    artifacts_ok=True
    for execution_index,(directory,path,record) in enumerate(executions):
        execution=record.get("execution") or {}
        if execution.get("completed") is not True:
            error={"iteration":directory.name,"error":execution.get("error")}
            # Runtime/controller failures are valid autonomous debugging
            # evidence when a later physical iteration executes to completion.
            # Do not erase the failure; classify it as recovered only after a
            # real subsequent rollout, never merely after a source edit.
            later_completed=any(
                (later.get("execution") or {}).get("completed") is True
                for _,_,later in executions[execution_index+1:])
            (recovered_controller_errors if later_completed else controller_errors).append(error)
        if (record.get("robot_contract_preflight") or {}).get("passed") is not True:
            contract_pass=False
        controller=directory/"controller.py"
        violations=_controller_forbidden(controller)
        if violations:forbidden.append({"iteration":directory.name,"violations":violations})
        report=record.get("sensor_report") or {};sensor_reports.append(report)
        trace_path=Path(str(report.get("trace_path") or ""))
        rollout_path=Path(str(report.get("rollout_path") or ""))
        if not (trace_path.is_file() and rollout_path.is_file()):artifacts_ok=False
        snapshot=Path(str(record.get("controller_snapshot") or ""))
        if not snapshot.is_file():artifacts_ok=False

    task_events=[event for event in traces if event.get("type")=="task"]
    model_events=[event for event in traces if event.get("type")=="model"]
    tool_names=[event.get("name") for event in traces if event.get("type")=="tool_result"
                and event.get("ok") is True]
    feedback_seen=False
    for event in task_events[1:]:
        try:instruction=json.loads(event.get("instruction") or "{}")
        except Exception:continue
        if instruction.get("previous_sensor_evidence") is not None:feedback_seen=True

    asset_ok,asset_errors,asset_counts=_tool_assets(root)
    reports_present=bool(sensor_reports) and all(
        "benchmark_signal_exposed" in report for report in sensor_reports)
    evaluator_blind=reports_present and all(
        report.get("benchmark_signal_exposed") is False for report in sensor_reports)
    state_iterations=state.get("iterations") or []
    persisted_evidence=sum(1 for row in state_iterations if row.get("evidence") is not None)
    gates={
        "model_driven_workspace":bool(task_events and model_events and "write_file" in tool_names),
        "robot_contract_preflight":bool(executions) and contract_pass,
        "controller_execution_completed":bool(executions) and not controller_errors,
        "sensor_evidence_returned":reports_present,
        "evaluator_blind":evaluator_blind and not forbidden,
        "artifacts_persisted":bool(executions) and artifacts_ok,
        "failure_feedback_loop":len(executions)<2 or feedback_seen,
        "state_persistence":bool(state) and persisted_evidence==len(executions),
        "tool_asset_integrity":asset_ok,
        "clean_engineering_interfaces":not interface_errors,
    }
    sensor_successes=sum(bool(record.get("sensor_success_candidate"))
                         for _,_,record in executions)
    return {"protocol":"embodied-codex-conformance-v1","run_root":str(root),
            "conformant":all(gates.values()),"gates":gates,
            "task":state.get("task"),"status":state.get("status"),
            "metrics":{"iterations_with_robot":len(executions),
                       "sensor_successes":sensor_successes,
                       "model_turns":len(model_events),
                       "web_searches":tool_names.count("search_web"),
                       "sensor_images_inspected":tool_names.count("view_sensor_image"),
                       "tools_registered":tool_names.count("register_tool"),
                       "tools_tested":tool_names.count("test_tool"),
                       "asset_status_counts":asset_counts},
            "interface_errors":interface_errors,"controller_errors":controller_errors,
            "recovered_controller_errors":recovered_controller_errors,
            "recovered_model_errors":recovered_model_errors,
            "recovered_tool_errors":recovered_tool_errors,
            "forbidden_controller_access":forbidden,"asset_errors":asset_errors}


__all__=["audit_run"]
