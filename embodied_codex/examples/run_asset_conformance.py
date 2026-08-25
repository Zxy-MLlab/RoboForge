"""Real-model checks for Tool manuals and cross-task Experience transfer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from embodied_codex.assets import CapabilityLibrary, ExperienceLibrary
from embodied_codex.conformance import audit_run
from embodied_codex.evolution import EvolutionEngine
from embodied_codex.model import OpenAIModel
from embodied_codex.workspace import TaskWorkspace
from embodied_codex.examples import run_kernel_conformance as kernel


PYTHON=sys.executable


class CalibratedDeployment:
    instruction=("Use the registered calibrated offset Tool on the live raw_target reading, "
                 "move the cursor to its returned command, and verify with sensors.")
    def __init__(self,artifact_dir):
        self.artifact_dir=artifact_dir;artifact_dir.mkdir(parents=True,exist_ok=False)
        self.cursor=0.;self.raw_target=4.;self.tools={};self.trace=[]
    def register_capability(self,tool_id,function,contract):self.tools[str(tool_id)]=function
    def dispatch(self,method,arguments):
        if method=="observe":result={"cursor":self.cursor,"raw_target":self.raw_target}
        elif method=="use":
            tool_id=str(arguments["tool_id"])
            result={"tool_id":tool_id,"result":self.tools[tool_id](arguments.get("payload") or {})}
        elif method=="act":
            action=arguments["action"]
            if action.get("type")!="move_cursor":raise RuntimeError("unsupported action")
            self.cursor=float(action["position"]);result={"cursor":self.cursor}
        elif method=="verify":result={"verified":abs(self.cursor-5.)<1e-6,"cursor":self.cursor}
        elif method=="record":result={"recorded":True}
        else:raise RuntimeError(method)
        self.trace.append({"method":method,"arguments":arguments,"result":result});return result
    def project_rpc_output(self,method,arguments,result):
        allowed={"cursor","raw_target","tool_id","result","verified","recorded"}
        unknown=set(result)-allowed
        if unknown:raise RuntimeError(f"undeclared output: {sorted(unknown)}")
        return dict(result)
    def sensor_report(self,execution):
        return {"sensor_verification_passed":abs(self.cursor-5.)<1e-6,
                "benchmark_signal_exposed":False,
                "trace_path":str(self.artifact_dir/"adapter_trace.json"),
                "rollout_path":str(self.artifact_dir/"rollout.mp4")}
    def close(self):
        (self.artifact_dir/"adapter_trace.json").write_text(json.dumps(self.trace,indent=2)+"\n")
        (self.artifact_dir/"rollout.mp4").write_bytes(b"")


class Factory:
    def __init__(self,root):self.root=root;self.count=0
    def __call__(self):
        self.count+=1
        return CalibratedDeployment(self.root/"episodes"/f"episode_{self.count:03d}")


def install_offset_tool(capability_root: Path,bootstrap: Path,bad_manual: bool):
    workspace=TaskWorkspace(bootstrap)
    workspace.write_file("offset.py","def run(payload):\n    return {'command': float(payload['reading']) + 1.0}\n")
    library=CapabilityLibrary(capability_root,workspace.root,python=PYTHON)
    output_key="target" if bad_manual else "command"
    result=library.register_tool(name="calibrated_offset",source_path="offset.py",
        description="Apply the public +1 calibration to a live scalar reading.",
        input_schema={"type":"object","properties":{"reading":{"type":"number"}},
                      "required":["reading"],"additionalProperties":False},
        output_schema={"type":"object","properties":{"command":{"type":"number"}},
                       "required":["command"],"additionalProperties":False},
        source_urls=["https://example.org/calibration"],
        trained_on_current_task=False,dependency_spec={"mode":"stdlib"},manual={
            "purpose":"Convert a live raw target into a calibrated cursor command.",
            "when_to_use":["A task explicitly requires the calibrated offset Tool."],
            "inputs":{"reading":"Live numeric raw_target."},
            "outputs":{output_key:"Numeric calibrated command."},
            "examples":[{"input":{"reading":4},"output":{output_key:5}}],
            "failure_modes":["Rejects a missing or nonnumeric reading."],
            "limitations":["Only applies the documented +1 calibration."]})
    tool_id=result["tool_id"]
    library.test_tool(tool_id,[{"input":{"reading":4},"expected":{"command":5.0}}])
    return tool_id


def sdk():
    return {"protocol":"asset-conformance-sdk-v1","methods":{},
        "actions":{"move_cursor":{"required":["type","position"],"optional":{}}},
        "verifiers":{"cursor_at_calibrated_target":{"required":[],"optional":{}}},
        "opaque_reference_fields":[]}


def tool_calls(root: Path,name: str,iteration: int|None=None):
    rows=[]
    pattern=(f"iteration_{iteration:03d}/agent_trace.jsonl"
             if iteration is not None else "iteration_*/agent_trace.jsonl")
    for path in sorted((root/"iterations").glob(pattern)):
        for line in path.read_text().splitlines():
            event=json.loads(line)
            if event.get("type")=="tool_result" and event.get("name")==name and event.get("ok"):
                rows.append(event)
    return rows


def first_iteration_transfer_evidence(root: Path):
    trace=root/"iterations"/"iteration_001"/"agent_trace.jsonl"
    retrieved=0
    if trace.is_file():
        for line in trace.read_text().splitlines():
            event=json.loads(line)
            if event.get("type")=="task":
                instruction=json.loads(event["instruction"])
                retrieved=len(instruction.get("retrieved_experiences") or [])
                break
    execution=root/"iterations"/"iteration_001"/"robot_execution.json"
    methods=[]
    if execution.is_file():
        document=json.loads(execution.read_text())
        methods=[event.get("method") for event in
                 ((document.get("execution") or {}).get("rpc_events") or [])]
    first_act=methods.index("act") if "act" in methods else -1
    closed_loop=(first_act>=0 and "observe" in methods[first_act+1:])
    return {"retrieved_experience_count":retrieved,
            "closed_loop_action_reobservation":closed_loop}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output-dir",required=True)
    parser.add_argument("--model",default="gpt-5.6-sol");args=parser.parse_args()
    key=os.environ.get("APEX_API_KEY")
    if not key:raise SystemExit("APEX_API_KEY missing")
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    model=OpenAIModel(api_key=key,base_url="https://api.apexin.ai/v1",model=args.model,
                      reasoning_effort="low",max_tokens=6000)
    rows=[]
    for label,bad in (("manual_correct",False),):
        root=output/label;capabilities=root/"capabilities"
        tool_id=install_offset_tool(capabilities,root/"bootstrap",bad)
        engine=EvolutionEngine(root=root,model=model,deployment_factory=Factory(root),python=PYTHON,
            capability_root=capabilities,deployment_guidance={"robot_sdk_contract":sdk(),
                "seed_tool_ids":[tool_id]})
        state=engine.run(task=CalibratedDeployment.instruction,
            skill_name=f"{label}_skill",max_iterations=3)
        manual=engine.capabilities.manual(tool_id)
        rows.append({"case":label,"status":state["status"],"audit":audit_run(root),
            "read_source_calls":len(tool_calls(root,"read_tool_source")),
            "revise_manual_calls":len(tool_calls(root,"revise_tool_manual")),
            "manual_revision":manual["manual_revision"]})
    rejected=False
    try:install_offset_tool(output/"manual_inconsistent"/"capabilities",
                            output/"manual_inconsistent"/"bootstrap",True)
    except Exception as exc:
        rejected="manual output fields" in str(exc)
    rows.append({"case":"manual_inconsistent_registration","rejected":rejected})

    # Seed two evidence-backed prior-task experiences locally so this
    # conformance remains reproducible in a fresh clone.  The test is whether
    # the Agent retrieves and transfers them, not whether a machine happens to
    # retain an old ignored run directory.
    transfer=output/"experience_transfer";shared=output/"shared_experiences"
    seed_evidence=output/"prior_task_sensor_evidence.json"
    seed_evidence.write_text(json.dumps({
        "protocol":"embodied-codex-conformance-prior-evidence-v1",
        "observation":{"temperature_c":17.0,"safe_band_c":[20.5,21.5]},
        "actuation_note":"The actuator may realize only part of a requested delta."
    },indent=2)+"\n")
    prior=ExperienceLibrary(shared)
    prior.register(name="parse semantic interval containers",
        summary=("A live thermal state can encode the acceptable interval as the "
                 "two-element safe_band_c list rather than separate scalar bounds. "
                 "Validate both numeric entries and their ordering before control."),
        applicability=("State-driven thermal control with temperature_c and a live "
                       "safe_band_c pair; do not assume the values are fixed."),
        keywords=["thermal","safe_band_c","interval","sensor parsing"],
        evidence_paths=[seed_evidence])
    prior.register(name="iterative midpoint thermal control",
        summary=("When measured actuation may be partial, reobserve after every "
                 "temperature command and recompute the delta toward the live band "
                 "midpoint until sensor verification passes."),
        applicability=("Closed-loop scalar regulation with fresh measurements and a "
                       "bounded action budget."),
        keywords=["thermal","closed loop","midpoint","reobserve"],
        evidence_paths=[seed_evidence])
    initial_shared=len(list(shared.glob("*/v*/manifest.json")))
    kernel.CASES["thermal"]["initial"]={"temperature_c":10.0,"safe_band_c":[20.5,21.5]}
    engine=EvolutionEngine(root=transfer,model=model,
        deployment_factory=kernel.Factory("thermal",transfer),python=PYTHON,
        experience_root=shared,deployment_guidance={"adapter":{"name":"thermal-transfer"},
            "robot_sdk_contract":kernel.sdk_contract(kernel.CASES["thermal"]),"seed_tool_ids":[]})
    state=engine.run(task=kernel.CASES["thermal"]["instruction"],
                     skill_name="thermal_transfer_skill",max_iterations=2)
    first_iteration_experience_inspections=len(
        tool_calls(transfer,"inspect_experience",iteration=1))
    transfer_evidence=first_iteration_transfer_evidence(transfer)
    rows.append({"case":"experience_transfer","status":state["status"],
        "iterations":len(state["iterations"]),"initial_shared_experiences":initial_shared,
        "first_iteration_experience_inspections":first_iteration_experience_inspections,
        **transfer_evidence,
        "shared_experiences_after":len(engine.experiences.list_summaries()),
        "audit":audit_run(transfer)})
    report={"protocol":"embodied-codex-asset-conformance-v1","model":args.model,"cases":rows}
    report["passed"]=(rows[0]["status"]=="sensor_success" and rows[0]["read_source_calls"]==0
        and rows[1]["rejected"] is True
        and rows[2]["status"]=="sensor_success" and rows[2]["iterations"]<=2
        and rows[2]["retrieved_experience_count"]>=1
        and rows[2]["closed_loop_action_reobservation"] is True
        and rows[2]["initial_shared_experiences"]>=2
        and rows[2]["shared_experiences_after"]>=rows[2]["initial_shared_experiences"]
        and all(row["audit"]["conformant"] for row in (rows[0],rows[2])))
    (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2));return 0 if report["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
