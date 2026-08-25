"""Canonical end-to-end LIBERO campaign for the free-coding Embodied Codex.

It freezes a sensor-developed Skill first, then runs all predeclared unseen
episodes behind one sealed evaluator barrier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from embodied_codex.providers import resolve_provider


ROOT=Path(__file__).resolve().parents[1]
PYTHON=sys.executable


def _task_list(value: str)->list[int]:
    tasks=[]
    for token in value.split(","):
        token=token.strip()
        if not token:continue
        if "-" in token:
            start,end=(int(item) for item in token.split("-",1));tasks.extend(range(start,end+1))
        else:tasks.append(int(token))
    tasks=list(dict.fromkeys(tasks))
    if not tasks or any(task<0 or task>9 for task in tasks):
        raise argparse.ArgumentTypeError("tasks must be LIBERO task selectors 0..9")
    return tasks


def _write_json(path: Path,value: Any):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n");temporary.replace(path)


def _file_sha256(path: str|Path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def _prepare_campaign_root(output: Path)->bool:
    """Create the root while remembering whether user content pre-existed.

    The launcher itself creates the shared asset directory before writing the
    campaign manifest.  New-run validation must therefore use the directory
    state from before those Harness-owned paths are created.
    """
    preexisting_nonempty=output.exists() and any(output.iterdir())
    output.mkdir(parents=True,exist_ok=True)
    return preexisting_nonempty


def _resolve_campaign_capability_library(*, output: Path,
                                         requested: Path|None)->Path:
    """Keep resumed tasks bound to the capability library they were born with.

    Early campaign versions did not persist this root in ``campaign.json``.
    Their per-task immutable Harness configuration did, however.  Falling back
    to ``output/assets/tools`` on resume silently supplied a different library
    and made every otherwise-valid task fail before coding could resume.
    """
    if requested is not None:return requested.resolve()
    def legacy_asset_root(value):
        path=Path(str(value)).resolve()
        return path.parent if path.name=="tools" else path
    campaign_path=output/"campaign.json"
    if campaign_path.is_file():
        campaign=json.loads(campaign_path.read_text())
        declared=campaign.get("capability_library")
        if declared:return legacy_asset_root(declared)
        roots=set()
        for path in output.glob("task_*/development/harness_configuration.json"):
            try:configuration=json.loads(path.read_text())
            except (OSError,json.JSONDecodeError):continue
            root=configuration.get("capability_root")
            if root:roots.add(str(legacy_asset_root(root)))
        if len(roots)>1:
            raise RuntimeError(
                "resumed campaign tasks reference multiple capability libraries")
        if roots:return Path(next(iter(roots)))
    return (output/"assets").resolve()


def _ranked_states(*,task: int,anchor: int,state_count: int,seed: int):
    if anchor<0 or anchor>=state_count:raise ValueError("development state outside state range")
    candidates=[state for state in range(state_count) if state!=anchor]
    return sorted(candidates,key=lambda state:hashlib.sha256(
        f"embodied-codex-case-partition-v1:{seed}:{task}:{state}".encode()).hexdigest())


def _predeclared_partition(*,task: int,development_state: int,
                           development_count: int,sealed_count: int,
                           state_count: int,seed: int):
    ranked=_ranked_states(task=task,anchor=development_state,state_count=state_count,seed=seed)
    if development_count<1 or sealed_count<1 or development_count+sealed_count>state_count:
        raise ValueError("invalid development/sealed state counts")
    development=[development_state,*ranked[:development_count-1]]
    sealed=ranked[development_count-1:development_count-1+sealed_count]
    return {"development":development,"sealed":sealed}


def _predeclared_states(*, task: int, development_state: int, count: int,
                        state_count: int, seed: int):
    """Backward-compatible sealed-only helper used by external callers."""
    candidates=[state for state in range(state_count) if state!=development_state]
    ranked=sorted(candidates,key=lambda state:hashlib.sha256(
        f"embodied-codex-sealed-v1:{seed}:{task}:{state}".encode()).hexdigest())
    if count<1 or count>len(ranked):raise ValueError("invalid unseen state count")
    return ranked[:count]


def development_command(*, task: int,states: list[int],max_iterations: int,output: Path,
                        capabilities: Path,model: str,reasoning_effort: str,device: str,
                        python: str,groundingdino_checkpoint: str,base_url: str,
                        provider: str|None=None,
                        retry_locked_validation: bool=False,
                        verifier_reasoning_effort: str="low"):
    # Development uses the autonomous Harness over every predeclared case.  No
    # sealed result is exposed while the model can still change the Controller.
    command=([python,"-m","embodied_codex","run","--adapter","libero",
        "--task",str(task),"--profile","autonomous","--run-dir",str(output),
        "--asset-root",str(capabilities),"--model-name",model,
        "--reasoning-effort",reasoning_effort,
        *(["--provider",provider] if provider else []),"--base-url",base_url,
        "--max-steps",str(max_iterations),"--controller-timeout","600",
        "--states",*[str(state) for state in states]])
    return command


def validation_command(*, skill_dir: str|Path,task: int,states: list[int],output: Path,
                       model: str,reasoning_effort: str,device: str,python: str,
                       groundingdino_checkpoint: str,base_url: str,
                       provider: str|None=None,
                       capabilities: str|Path|None=None):
    skill_dir=Path(skill_dir).resolve()
    controller=skill_dir/"controller.py"
    if capabilities is None:
        if len(skill_dir.parents)<3 or skill_dir.parents[1].name!="skills":
            raise ValueError("capabilities is required for a non-Skill Controller")
        capabilities=skill_dir.parents[2]
    return ([python,"-m","embodied_codex","run","--adapter","libero",
        "--task",str(task),"--profile","benchmark","--run-dir",str(output),
        "--asset-root",str(Path(capabilities).resolve()),
        "--controller-source",str(controller),"--frozen-controller",
        "--states",*[str(state) for state in states],
        "--model-name",model,"--reasoning-effort",reasoning_effort,
        *(["--provider",provider] if provider else []),
        "--base-url",base_url,"--max-steps","8"])


def _matching_skill(asset_root: Path, controller_sha256: str|None):
    if not controller_sha256:
        return None
    candidates=[]
    for path in asset_root.glob("skills/*/v*/manifest.json"):
        try:manifest=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):continue
        if manifest.get("controller_sha256")==controller_sha256:
            candidates.append((float(manifest.get("created_unix",0)),path.parent,manifest))
    if not candidates:return None
    _created,path,manifest=max(candidates,key=lambda row:(row[0],str(row[1])))
    return {"skill_id":manifest.get("skill_id"),"path":str(path)}


def _development_status(root: Path,asset_root: Path|None=None):
    path=root/"result.json"
    if not path.is_file():
        return {"status":"process_failed","iterations":0,"skill":None,
                "controller_path":None}
    try:result=json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):
        return {"status":"process_failed","iterations":0,"skill":None,
                "controller_path":None}
    cases=list(result.get("cases") or [result])
    hashes={case.get("latest_evidence",{}).get("controller_sha256") for case in cases}
    hashes.discard(None)
    controller_sha256=next(iter(hashes)) if len(hashes)==1 else None
    controller_paths=[]
    for case in cases:
        state=case.get("case")
        workspace=root/"workspace" if state is None else root/f"state_{state}"/"workspace"
        controller=workspace/"controller.py"
        if (controller.is_file() and controller_sha256
                and _file_sha256(controller)==controller_sha256):
            controller_paths.append(controller)
    finished=bool(result.get("finished") is True and len(hashes)==1
                  and len(controller_paths)==len(cases))
    return {"status":"sensor_success" if finished else "evolving",
            "iterations":sum(int(case.get("steps",0)) for case in cases),
            "skill":_matching_skill(asset_root,controller_sha256) if asset_root else None,
            "controller_path":str(controller_paths[0]) if controller_paths else None,
            "controller_sha256":controller_sha256}


def _sealed_status(root: Path,*,skill_id: str|None,states: list[int]):
    path=root/"result.json"
    if not path.is_file():return None
    result=json.loads(path.read_text());cases=list(result.get("cases") or [result])
    successes=sum(bool(case.get("finished") is True
        and case.get("evaluation_passed") is True) for case in cases)
    return {"protocol":"roboforge-libero-sealed-v1","skill_id":skill_id,
            "states":list(states),"episodes":len(cases),
            "evaluator_successes":successes,
            "controller_sha256":result.get("cross_case_controller_sha256")
                or cases[0].get("latest_evidence",{}).get("controller_sha256")}


def _resolve_packaging_skill(skill_dir: str|Path):
    """Resolve the newest audited repackage of one immutable Controller."""
    source=Path(skill_dir).resolve();manifest_path=source/"manifest.json"
    if not manifest_path.is_file():raise FileNotFoundError(f"Skill manifest: {manifest_path}")
    original=json.loads(manifest_path.read_text())
    candidates=[]
    for path in source.parent.glob("v*/manifest.json"):
        item=json.loads(path.read_text());migration=dict(item.get("packaging_migration") or {})
        if (migration.get("source_skill_id")==original.get("skill_id")
                and migration.get("controller_sha256_unchanged") is True
                and item.get("controller_sha256")==original.get("controller_sha256")):
            candidates.append((int(item.get("version",0)),path.parent,item))
    if not candidates:return source,original
    _version,resolved,manifest=max(candidates,key=lambda row:row[0])
    return resolved,manifest


def _campaign_exit_code(campaign: dict):
    infrastructure=[]
    incomplete=[]
    for row in campaign.get("task_results") or []:
        if row.get("development_returncode") not in {0,2,None}:
            infrastructure.append({"task":row.get("task"),"phase":"development",
                                   "returncode":row.get("development_returncode")})
        if row.get("sealed_returncode") not in {0,2,None}:
            infrastructure.append({"task":row.get("task"),"phase":"sealed",
                                   "returncode":row.get("sealed_returncode")})
        sealed=row.get("sealed_evaluation") or {}
        if (row.get("status")!="sensor_success" or not sealed
                or sealed.get("evaluator_successes")!=sealed.get("episodes")):
            incomplete.append(row.get("task"))
    campaign["infrastructure_failures"]=infrastructure
    campaign["capability_incomplete_tasks"]=incomplete
    if infrastructure:return 1
    return 2 if incomplete else 0


def _development_must_halt(returncode: int)->bool:
    """Only learned (0) and exhausted-frontier (2) runs may advance a campaign."""
    return int(returncode) not in {0,2}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tasks",type=_task_list,default=[0])
    parser.add_argument("--development-state",type=int,default=0)
    parser.add_argument("--development-state-count",type=int,default=3)
    parser.add_argument("--max-iterations",type=int,default=16)
    parser.add_argument("--unseen-state-count",type=int,default=3)
    parser.add_argument("--state-count",type=int,default=50)
    parser.add_argument("--seed",type=int,default=2909)
    parser.add_argument("--model",default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort",default="high")
    parser.add_argument("--verifier-reasoning-effort",default=os.environ.get(
        "EMBODIED_CODEX_VERIFIER_REASONING_EFFORT","low"))
    parser.add_argument("--provider",choices=("openai","apex"))
    parser.add_argument("--base-url",default=os.environ.get("EMBODIED_CODEX_BASE_URL"))
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--python",default=sys.executable)
    parser.add_argument("--groundingdino-checkpoint",default=os.environ.get(
        "EMBODIED_CODEX_GROUNDINGDINO_CHECKPOINT","checkpoints/groundingdino_swint_ogc.pth"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--capability-library",type=Path)
    parser.add_argument("--retry-locked-validation",action="store_true",
        help="Replay the current immutable Controller once after a verifier/Adapter correction")
    args=parser.parse_args()
    provider=resolve_provider(provider=args.provider,base_url=args.base_url)
    args.provider=provider.provider
    args.base_url=provider.endpoint
    if not Path(args.python).is_file():raise FileNotFoundError(f"Python executable: {args.python}")
    if not Path(args.groundingdino_checkpoint).is_file():
        raise FileNotFoundError("GroundingDINO checkpoint; pass --groundingdino-checkpoint "
            "or EMBODIED_CODEX_GROUNDINGDINO_CHECKPOINT")
    output=args.output.resolve()
    preexisting_nonempty=_prepare_campaign_root(output)
    capabilities=_resolve_campaign_capability_library(
        output=output,requested=args.capability_library)
    capabilities.mkdir(parents=True,exist_ok=True)
    partitions={task:_predeclared_partition(task=task,
        development_state=args.development_state,
        development_count=args.development_state_count,sealed_count=args.unseen_state_count,
        state_count=args.state_count,seed=args.seed)
        for task in args.tasks}
    development_plans={task:value["development"] for task,value in partitions.items()}
    plans={task:value["sealed"] for task,value in partitions.items()}
    expected={"protocol":"embodied-codex-libero-campaign-v3",
        "kernel":"embodied_codex.kernel.AgentLoop","historical_graph_harness_used":False,
        "model":args.model,"provider":args.provider,"base_url":args.base_url,
        "provider_key_env":provider.key_env,"suite":"libero_spatial","tasks":args.tasks,
        "coding_agent_reasoning_effort":args.reasoning_effort,
        "outcome_verifier_reasoning_effort":args.verifier_reasoning_effort,
        "groundingdino_checkpoint_sha256":_file_sha256(args.groundingdino_checkpoint),
        "development_state":args.development_state,"sealed_states_predeclared":plans,
        "development_states_predeclared":development_plans,
        "sealed_results_consumed_for_iteration":False}
    campaign_path=output/"campaign.json"
    if campaign_path.is_file():
        campaign=json.loads(campaign_path.read_text())
        for key,value in expected.items():
            if key not in campaign and key in {
                    "coding_agent_reasoning_effort",
                    "outcome_verifier_reasoning_effort"}:
                campaign[key]=value
                campaign.setdefault("configuration_migrations",[]).append({
                    "kind":"bind_model_role_configuration_v1",
                    "field":key,"value":value})
                continue
            # JSON object keys turn integer task IDs into strings on disk.
            actual=campaign.get(key)
            if key in {"sealed_states_predeclared","development_states_predeclared"}:
                actual={int(k):v for k,v in (actual or {}).items()}
            if actual!=value:raise RuntimeError(f"resumed campaign mismatch: {key}")
        declared=campaign.get("capability_library")
        declared_path=Path(str(declared)).resolve() if declared is not None else None
        declared_root=(declared_path.parent if declared_path is not None
                       and declared_path.name=="tools" else declared_path)
        if declared_root is not None and declared_root!=capabilities.resolve():
            raise RuntimeError("resumed campaign capability library mismatch")
        if declared_path is None or declared_path!=capabilities.resolve():
            campaign["capability_library"]=str(capabilities)
            campaign.setdefault("configuration_migrations",[]).append({
                "kind":"normalize_shared_asset_root",
                "value":str(capabilities)})
        campaign["resumed"]=True
    elif preexisting_nonempty:
        raise FileExistsError(f"nonempty output is not a resumable campaign: {output}")
    else:campaign={**expected,"capability_library":str(capabilities),
                   "task_results":[],"resumed":False}
    _write_json(output/"campaign.json",campaign)
    runtime_env=os.environ.copy()
    runtime_env.update({"ROBOFORGE_DEVICE":args.device,
        "ROBOFORGE_GROUNDINGDINO_CHECKPOINT":str(Path(args.groundingdino_checkpoint).resolve())})
    for task in args.tasks:
        task_root=output/f"task_{task:02d}";development=task_root/"development"
        status=_development_status(development,capabilities)
        frontier_path=task_root/"frontier_failure.json"
        if status["status"]!="sensor_success":
            # A prior process failure or a deliberately resumed frontier must
            # not leave a stale terminal marker while this task is active.
            frontier_path.unlink(missing_ok=True)
            command=development_command(task=task,states=development_plans[task],
                max_iterations=args.max_iterations,output=development,capabilities=capabilities,
                model=args.model,reasoning_effort=args.reasoning_effort,device=args.device,
                python=args.python,groundingdino_checkpoint=args.groundingdino_checkpoint,
                base_url=args.base_url,provider=args.provider,
                retry_locked_validation=args.retry_locked_validation,
                verifier_reasoning_effort=args.verifier_reasoning_effort)
            completed=subprocess.run(command,cwd=ROOT,env=runtime_env)
            status=_development_status(development,capabilities);development_returncode=completed.returncode
        else:
            development_returncode=0
            frontier_path.unlink(missing_ok=True)
        row={"task":task,"development_returncode":development_returncode,**status}
        if _development_must_halt(development_returncode):
            row["development_error"]=(
                "development process failed before a valid learned/frontier terminal state; "
                "campaign halted and this task remains resumable")
            campaign["task_results"]=[item for item in campaign["task_results"]
                                      if item.get("task")!=task]
            campaign["task_results"].append(row)
            campaign["task_results"].sort(key=lambda item:item["task"])
            campaign["halted_on_task"]=task
            campaign["halt_reason"]="development_infrastructure_failure"
            campaign["development_sensor_successes"]=sum(
                item["status"]=="sensor_success" for item in campaign["task_results"])
            campaign["sealed_evaluator_successes"]=sum(
                (item.get("sealed_evaluation") or {}).get("evaluator_successes",0)
                for item in campaign["task_results"])
            _campaign_exit_code(campaign)
            _write_json(output/"campaign.json",campaign)
            _write_json(output/"summary.json",campaign)
            print(json.dumps(campaign,indent=2))
            return 1
        skill=status.get("skill") or {};skill_dir=skill.get("path")
        controller_path=status.get("controller_path")
        if status["status"]=="sensor_success" and controller_path:
            if skill_dir:
                skill_dir,evaluated_skill=_resolve_packaging_skill(skill_dir)
                controller_path=skill_dir/"controller.py"
            else:
                evaluated_skill={"skill_id":None,
                    "controller_sha256":status.get("controller_sha256")}
                skill_dir=Path(controller_path).parent
            row["evaluated_skill"]={"skill_id":evaluated_skill.get("skill_id"),
                                    "path":str(skill_dir),
                                    "controller_sha256":evaluated_skill.get("controller_sha256")}
            sealed=task_root/"sealed_evaluation"
            if (sealed/"result.json").is_file():
                summary=_sealed_status(sealed,skill_id=evaluated_skill.get("skill_id"),
                                       states=plans[task])
                if summary.get("skill_id")!=evaluated_skill.get("skill_id"):
                    row["sealed_returncode"]=1
                    row["sealed_error"]="sealed summary Skill id does not match resolved frozen Skill"
                else:
                    row["sealed_returncode"]=0
                    row["sealed_evaluation"]=summary
            elif sealed.exists() and any(sealed.iterdir()):
                row["sealed_returncode"]=1
                row["sealed_error"]="partial sealed batch cannot be resumed or exposed to evolution"
            else:
                evaluated=subprocess.run(validation_command(skill_dir=skill_dir,task=task,
                    states=plans[task],output=sealed,model=args.model,
                    reasoning_effort=args.reasoning_effort,device=args.device,
                    python=args.python,groundingdino_checkpoint=args.groundingdino_checkpoint,
                    base_url=args.base_url,provider=args.provider,capabilities=capabilities),
                    cwd=ROOT,env=runtime_env)
                row["sealed_returncode"]=evaluated.returncode
                summary=_sealed_status(sealed,skill_id=evaluated_skill.get("skill_id"),
                                       states=plans[task])
                if summary:
                    row["sealed_evaluation"]=summary
                    _write_json(sealed/"summary.json",summary)
        else:
            _write_json(frontier_path,{
                "task":task,"development_status":status["status"],
                "iterations":status["iterations"],"evaluator_used":False})
        campaign["task_results"]=[item for item in campaign["task_results"]
                                  if item.get("task")!=task]
        campaign["task_results"].append(row)
        campaign["task_results"].sort(key=lambda item:item["task"])
        _write_json(output/"campaign.json",campaign)
    campaign["development_sensor_successes"]=sum(
        row["status"]=="sensor_success" for row in campaign["task_results"])
    campaign["sealed_evaluator_successes"]=sum(
        (row.get("sealed_evaluation") or {}).get("evaluator_successes",0)
        for row in campaign["task_results"])
    exit_code=_campaign_exit_code(campaign)
    _write_json(output/"summary.json",campaign);print(json.dumps(campaign,indent=2))
    return exit_code


if __name__=="__main__":raise SystemExit(main())
