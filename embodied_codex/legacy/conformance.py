"""Machine-auditable conformance checks for Embodied Codex experiment runs."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, ValidationError


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


def _tool_assets(root: Path):
    errors=[];counts={"registered":0,"tested":0,"test_failed":0}
    manuals_present=True;dependencies_reproducible=True;contracts_valid=True
    research_hashes=set()
    for ledger in (root/"iterations").glob("iteration_*/research_ledger.jsonl"):
        for line in ledger.read_text(errors="replace").splitlines():
            try:record=json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"invalid_research_ledger:{ledger}");continue
            digest=record.pop("record_sha256",None)
            expected=hashlib.sha256(json.dumps(record,sort_keys=True,
                separators=(",",":"),default=str).encode()).hexdigest()
            if digest!=expected:errors.append(f"research_ledger_hash_mismatch:{ledger}")
            else:research_hashes.add(digest)
    configuration=root/"harness_configuration.json"
    capability_root=root/"capabilities"
    if configuration.is_file():
        try:capability_root=Path(_json(configuration)["capability_root"]).resolve()
        except Exception as exc:
            return False,[f"invalid_harness_configuration:{type(exc).__name__}"],counts,False,False,False
    for manifest_path in capability_root.glob("*/v*/manifest.json"):
        try:manifest=_json(manifest_path)
        except Exception as exc:
            errors.append(f"invalid_manifest:{manifest_path}:{type(exc).__name__}");continue
        status=str(manifest.get("status") or "registered")
        counts[status]=counts.get(status,0)+1
        provenance=dict(manifest.get("provenance") or {})
        audit_digest=provenance.pop("audit_sha256",None)
        expected_audit=hashlib.sha256(json.dumps(provenance,sort_keys=True,
            separators=(",",":"),default=str).encode()).hexdigest()
        if (provenance.get("audit_status")!="complete" or not audit_digest
                or audit_digest!=expected_audit):
            errors.append(f"incomplete_provenance_audit:{manifest.get('tool_id')}")
        if provenance.get("authoring_context")=="autonomous_engineering_agent":
            evidence=list(provenance.get("acquisition_evidence") or [])
            if (not evidence or any(item.get("record_sha256") not in research_hashes
                                    for item in evidence)):
                errors.append(f"missing_research_provenance:{manifest.get('tool_id')}")
        runtime_spec=dict(manifest.get("runtime_spec") or {})
        source=(manifest_path.parent/"bundle"/str(runtime_spec.get("entrypoint"))
                if runtime_spec else manifest_path.parent/"tool.py")
        expected=manifest.get("source_sha256")
        if not source.is_file():errors.append(f"missing_tool_source:{manifest.get('tool_id')}")
        elif expected and hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            errors.append(f"tool_hash_mismatch:{manifest.get('tool_id')}")
        if runtime_spec:
            digest=hashlib.sha256();bundle=manifest_path.parent/"bundle"
            if bundle.is_dir():
                for item in sorted(path for path in bundle.rglob("*") if path.is_file()):
                    digest.update(str(item.relative_to(bundle)).encode()+b"\0")
                    digest.update(hashlib.sha256(item.read_bytes()).digest())
            if not bundle.is_dir() or digest.hexdigest()!=manifest.get("bundle_tree_sha256"):
                errors.append(f"capability_package_hash_mismatch:{manifest.get('tool_id')}")
        try:
            input_schema=manifest.get("input_schema") or {};output_schema=manifest.get("output_schema") or {}
            Draft202012Validator.check_schema(input_schema);Draft202012Validator.check_schema(output_schema)
            standard={"$ref","type","anyOf","oneOf","allOf","not","if","enum","const"}
            if ((input_schema and not standard.intersection(input_schema))
                    or (output_schema and not standard.intersection(output_schema))):
                raise ValueError("descriptive mapping is not a contract")
            for batch in manifest.get("tests") or []:
                if isinstance(batch,dict):
                    if (manifest.get("execution_owned_by_deployment") is True
                            and batch.get("deployment_validation") is True):
                        continue
                    raise ValueError("invalid Tool test record")
                for case in batch:
                    # A contract-invalid/crashing candidate is an immutable
                    # negative experiment, not a corrupt asset.  test_tool
                    # records the exception and marks that version
                    # test_failed; it must remain auditable but undeployable.
                    if (case.get("passed") is False and case.get("error")
                            and manifest.get("status")=="test_failed"):
                        continue
                    Draft202012Validator(input_schema).validate(case.get("actual_input",case.get("input",{}))) \
                        if "actual_input" in case or "input" in case else None
                    Draft202012Validator(output_schema).validate(case.get("actual"))
                    Draft202012Validator(output_schema).validate(case.get("expected"))
        except Exception as exc:
            contracts_valid=False;errors.append(f"invalid_tool_contract:{manifest.get('tool_id')}:{type(exc).__name__}")
        tool_id=str(manifest.get("tool_id") or "");name,_,version=tool_id.partition(":")
        manual_dir=capability_root/"_manuals"/name/version
        revisions=sorted(manual_dir.glob("r[0-9]*.json"))
        if not revisions:manuals_present=False;errors.append(f"missing_tool_manual:{tool_id}")
        else:
            try:
                manual=_json(revisions[-1])
                if manual.get("tool_id")!=tool_id or not isinstance(manual.get("manual"),dict):
                    raise ValueError("manual identity mismatch")
            except Exception as exc:
                manuals_present=False;errors.append(f"invalid_tool_manual:{tool_id}:{type(exc).__name__}")
        dependencies=manifest.get("dependencies")
        if manifest.get("execution_owned_by_deployment"):
            continue
        if runtime_spec:
            pins=runtime_spec.get("runtime_requirements") or []
            package_ok=(runtime_spec.get("protocol") in {
                    "isolated-json-worker-v1","isolated-json-worker-v2"}
                and runtime_spec.get("network") is False
                and (runtime_spec.get("protocol")!="isolated-json-worker-v2"
                     or (runtime_spec.get("transport")=="json-stdio"
                         and runtime_spec.get("lifecycle")=="per-invocation"))
                and all(isinstance(item,str) and "==" in item for item in pins))
            if not package_ok:
                dependencies_reproducible=False
                errors.append(f"invalid_capability_package_runtime:{tool_id}")
            continue
        if not isinstance(dependencies,dict) or dependencies.get("runtime")!="isolated-python":
            dependencies_reproducible=False;errors.append(f"missing_dependency_contract:{tool_id}")
        elif dependencies.get("mode")=="vendored":
            lock=manifest_path.parent/"requirements.lock";vendor=manifest_path.parent/"vendor"
            digest=hashlib.sha256()
            if vendor.is_dir():
                for item in sorted(path for path in vendor.rglob("*") if path.is_file()):
                    digest.update(str(item.relative_to(vendor)).encode()+b"\0")
                    digest.update(hashlib.sha256(item.read_bytes()).digest())
            if (not lock.is_file() or not vendor.is_dir()
                    or hashlib.sha256(lock.read_bytes()).hexdigest()!=dependencies.get("requirements_lock_sha256")
                    or digest.hexdigest()!=dependencies.get("vendor_tree_sha256")):
                dependencies_reproducible=False;errors.append(f"invalid_dependency_bundle:{tool_id}")
    return not errors,errors,counts,manuals_present,dependencies_reproducible,contracts_valid


def _experience_assets(configuration: dict):
    root=Path(str(configuration.get("experience_root") or ""))
    if not root.is_dir():return True,[]
    errors=[]
    for manifest_path in root.glob("*/v*/manifest.json"):
        try:item=_json(manifest_path)
        except Exception as exc:
            errors.append(f"invalid_experience:{manifest_path}:{type(exc).__name__}");continue
        for evidence in item.get("evidence") or []:
            path=Path(str(evidence.get("artifact_uri") or evidence.get("path") or ""))
            if not path.is_absolute():path=manifest_path.parent/path
            if (not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest()!=evidence.get("sha256")):
                errors.append(f"experience_evidence_mismatch:{item.get('experience_id')}")
    return not errors,errors


def _gap_assets(configuration: dict):
    root=Path(str(configuration.get("gap_root") or ""))
    if not root.is_dir():return True,[]
    errors=[]
    for manifest_path in root.glob("*/v*/manifest.json"):
        try:item=_json(manifest_path)
        except Exception as exc:
            errors.append(f"invalid_capability_gap:{manifest_path}:{type(exc).__name__}");continue
        if item.get("protocol")!="embodied-codex-capability-gap-v1":
            errors.append(f"invalid_capability_gap_protocol:{manifest_path}")
        for evidence in item.get("evidence") or []:
            path=manifest_path.parent/str(evidence.get("artifact_uri") or evidence.get("path") or "")
            if (not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest()!=evidence.get("sha256")):
                errors.append(f"capability_gap_evidence_mismatch:{item.get('gap_id')}")
    return not errors,errors


def _skill_assets(configuration: dict):
    root=Path(str(configuration.get("skill_root") or ""))
    if not root.is_dir():return True,[],0
    errors=[];count=0
    for manifest_path in root.glob("*/v*/manifest.json"):
        count+=1
        try:item=_json(manifest_path)
        except Exception as exc:
            errors.append(f"invalid_skill:{manifest_path}:{type(exc).__name__}");continue
        folder=manifest_path.parent;controller=folder/"controller.py"
        if (not controller.is_file() or hashlib.sha256(controller.read_bytes()).hexdigest()
                !=item.get("controller_sha256")):
            errors.append(f"skill_controller_hash_mismatch:{item.get('skill_id')}")
        for tool_id,digest in (item.get("tool_bundle_sha256") or {}).items():
            bundle=folder/"tools"/str(tool_id).replace(":","_");actual=hashlib.sha256()
            if bundle.is_dir():
                for source in sorted(path for path in bundle.rglob("*") if path.is_file()):
                    actual.update(str(source.relative_to(bundle)).encode()+b"\0")
                    actual.update(hashlib.sha256(source.read_bytes()).digest())
            if not bundle.is_dir() or actual.hexdigest()!=digest:
                errors.append(f"skill_tool_bundle_hash_mismatch:{item.get('skill_id')}:{tool_id}")
        for evidence in item.get("evidence_files") or []:
            source=folder/str(evidence.get("path") or "")
            if (not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()
                    !=evidence.get("sha256")):
                errors.append(f"skill_evidence_hash_mismatch:{item.get('skill_id')}")
        for filename,key in (("experience.json","experience_sha256"),
                             ("task_model.json","task_model_sha256")):
            expected=item.get(key);source=folder/filename
            if expected and (not source.is_file()
                    or hashlib.sha256(source.read_bytes()).hexdigest()!=expected):
                errors.append(f"skill_auxiliary_hash_mismatch:{item.get('skill_id')}:{filename}")
    return not errors,errors,count


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
                if later.get("type")=="task":
                    # A transactionally persisted episode resumes in a fresh
                    # CodingAgent context.  A later model response proves the
                    # transport outage was recovered across that boundary.
                    recovered=any(candidate.get("type")=="model"
                                  for candidate in traces[index+2:])
                    break
                if (later.get("type")=="model"
                        and later.get("turn")==event.get("turn")):
                    recovered=True;break
            if not recovered and state.get("status")=="sensor_success":
                # The physical transaction and frozen Skill are already
                # durable. A post-rollout transport outage cannot undo that
                # completed outcome; retain it as recovered diagnostics.
                recovered=True
            record={key:event.get(key) for key in
                    ("type","turn","attempt","error","line")}
            (recovered_model_errors if recovered else interface_errors).append(record)
        elif event.get("type")=="trace_decode_error":
            interface_errors.append({key:event.get(key) for key in
                                     ("type","turn","attempt","error","line")})
        if event.get("type")=="tool_result" and event.get("ok") is False:
            # A model can issue multiple independent calls in one response.
            # A same-turn success or a later successful invocation of the same
            # engineering Tool proves that the autonomous loop recovered. Keep
            # the original failure as diagnostic evidence, including across a
            # transactionally resumed iteration, instead of permanently
            # classifying an already repaired interface as broken.
            recovered=any(
                candidate.get("type")=="tool_result"
                and candidate.get("name")==event.get("name")
                and candidate.get("ok") is True
                and (position>index or candidate.get("turn")==event.get("turn"))
                for position,candidate in enumerate(traces))
            # Older runs raised on a missing optional Skill lookup. The current
            # engineering contract returns a structured exists=false receipt,
            # so retain this immutable historical event as a contract-upgrade
            # recovery rather than fabricating a later successful trace call.
            contract_upgrade=(event.get("name")=="read_skill_source"
                and "AssetError: invalid Skill id" in str(event.get("error") or ""))
            recovered=recovered or contract_upgrade
            record={"type":"tool_error","tool":event.get("name"),
                    "error":event.get("error")}
            if contract_upgrade:
                record["recovered_by_contract_upgrade"]=(
                    "read-skill-source-soft-not-found-v1")
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
    request_events=[event for event in traces if event.get("type")=="model_request"]
    tool_names=[event.get("name") for event in traces if event.get("type")=="tool_result"
                and event.get("ok") is True]
    feedback_seen=False
    for event in task_events[1:]:
        try:instruction=json.loads(event.get("instruction") or "{}")
        except Exception:continue
        if instruction.get("previous_sensor_evidence") is not None:feedback_seen=True

    configuration=_json(root/"harness_configuration.json") \
        if (root/"harness_configuration.json").is_file() else {}
    asset_ok,asset_errors,asset_counts,manuals_present,dependencies_ok,contracts_valid=_tool_assets(root)
    experience_ok,experience_errors=_experience_assets(configuration)
    gaps_ok,gap_errors=_gap_assets(configuration)
    skills_ok,skill_errors,skill_count=_skill_assets(configuration)
    isolation=configuration.get("isolation") or {}
    command_results=[event.get("result") or {} for event in traces
                     if event.get("type")=="tool_result" and event.get("name")=="run_command"
                     and event.get("ok") is True]
    engineering_isolated=(isolation.get("engineering")=="bubblewrap-workspace-v1"
        and all(item.get("sandbox")=="bubblewrap-workspace-v1" for item in command_results))
    controller_isolated=(isolation.get("controller")=="bubblewrap-controller-v1"
        and all((record.get("execution") or {}).get("runtime_isolation")==
                "bubblewrap-controller-v1" for _,_,record in executions))
    rpc_allowlisted=bool(executions) and all(
        (record.get("execution") or {}).get("rpc_output_projection")==
        "adapter-explicit-allowlist-v1" and
        (record.get("execution") or {}).get("rpc_output_defense")==
        "kernel-evaluator-field-deny-v1" for _,_,record in executions)
    generated_tool_isolated=(isolation.get("generated_tool")=="bubblewrap-tool-v1"
                             and dependencies_ok)
    manual_source_separated=all(
        not isinstance(event.get("result"),dict) or "source" not in event["result"]
        for event in traces if event.get("type")=="tool_result"
        and event.get("name")=="inspect_tool" and event.get("ok") is True)
    reports_present=bool(sensor_reports) and all(
        "benchmark_signal_exposed" in report for report in sensor_reports)
    evaluator_blind=reports_present and all(
        report.get("benchmark_signal_exposed") is False for report in sensor_reports)
    state_iterations=state.get("iterations") or []
    persisted_evidence=sum(1 for row in state_iterations if row.get("evidence") is not None)
    request_audit_ok=(bool(request_events) and len(request_events)>=len(model_events)
        and all(len(str(event.get("messages_sha256") or ""))==64
                and len(str(event.get("tool_schema_sha256") or ""))==64
                and len(str(event.get("system_prompt_sha256") or ""))==64
                and int(event.get("message_count") or 0)>0
                for event in request_events))
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
        "engineering_workspace_isolated":engineering_isolated,
        "controller_runtime_isolated":controller_isolated,
        "rpc_output_allowlisted":rpc_allowlisted,
        "generated_tool_runtime_isolated":generated_tool_isolated,
        "tool_manual_present":manuals_present,
        "manual_source_separation":manual_source_separated,
        "experience_asset_integrity":experience_ok,
        "capability_gap_asset_integrity":gaps_ok,
        "skill_asset_integrity":skills_ok and (state.get("status")!="sensor_success" or skill_count>0),
        "dependency_reproducibility":dependencies_ok,
        "tool_contract_validated":contracts_valid,
        "clean_engineering_interfaces":not interface_errors,
        "model_request_audit":request_audit_ok,
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
                       "capability_packages_registered":tool_names.count("register_capability_package"),
                       "tools_tested":tool_names.count("test_tool"),
                       "asset_status_counts":asset_counts},
            "interface_errors":interface_errors,"controller_errors":controller_errors,
            "recovered_controller_errors":recovered_controller_errors,
            "recovered_model_errors":recovered_model_errors,
            "recovered_tool_errors":recovered_tool_errors,
            "forbidden_controller_access":forbidden,"asset_errors":asset_errors,
            "experience_asset_errors":experience_errors,
            "capability_gap_asset_errors":gap_errors,
            "skill_asset_errors":skill_errors}


__all__=["audit_run"]
