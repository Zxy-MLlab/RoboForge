"""Run the free-coding Embodied Codex loop on one LIBERO task/state."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import time

from embodied_codex.capabilities import (GraspNetRGBD,OpenVocabularyRGBD,
                                         VLMVisualRelationGrounder,
                                         VLMVisualTaskOutcomeVerifier)
from embodied_codex.deployments import LiberoDeployment,LiberoEpisode
from embodied_codex.evolution import EvolutionEngine
from embodied_codex.model import OpenAIModel
from embodied_codex.sdk_contract import LIBERO_ROBOT_SDK_CONTRACT

ROOT=Path(__file__).resolve().parents[2]
LIBERO_REQUIRE_TASK_MODEL=False
LIBERO_REQUIRE_TASK_FIDELITY_REVIEW=False

class Factory:
    def __init__(self,*,episodes,run_root,capabilities,capability_contracts,
                 verifiers,outcome_verifier=None):
        self.episodes=list(episodes);self.root=Path(run_root)
        self.outcome_verifier=outcome_verifier
        if not self.episodes:raise ValueError("at least one LIBERO episode case is required")
        self.capabilities=dict(capabilities);self.verifiers=dict(verifiers)
        self.capability_contracts={str(key):dict(value) for key,value in capability_contracts.items()}
        episode_root=self.root/"episodes";episode_root.mkdir(parents=True,exist_ok=True)
        existing=[int(p.name.split("_")[-1]) for p in episode_root.glob("episode_[0-9]*")
                  if p.name.split("_")[-1].isdigit()]
        self.count=max(existing,default=0)
        state_path=self.root/"state.json"
        if state_path.is_file():
            state=json.loads(state_path.read_text())
            self._mark_orphaned_rollouts(episode_root,state)
            self.case_cursor=sum(
                row.get("evidence") is not None
                and not row.get("infrastructure_replay_without_model")
                and not row.get("transient_infrastructure_failure")
                for row in (state.get("iterations") or []))
        else:
            # Legacy runs may predate state transactions. Count only complete
            # execution artifacts; an interrupted sensor probe must not consume
            # a development case on restart.
            self.case_cursor=sum(1 for _ in self.root.glob(
                "iterations/iteration_*/robot_execution.json"))
        self._next_case_handle=None

    @staticmethod
    def _mark_orphaned_rollouts(episode_root: Path,state: dict):
        """Make pre-transaction physical episodes explicit after a crash.

        The evolution state is committed only after controller execution and
        sensor reporting finish.  A process interruption can therefore leave
        video/action evidence without a state record.  Preserve it as an
        auditable infrastructure-aborted rollout; never silently delete it or
        treat it as task evidence on resume.
        """
        records=[row for row in (state.get("iterations") or [])
                 if row.get("evidence") is not None]
        committed=set()
        for row in records:
            report=(row.get("evidence") or {}).get("sensor_report") or {}
            for key in ("rollout_path","trace_path"):
                value=report.get(key)
                if value:committed.add(str(Path(value).resolve().parent))
        directories=sorted(path for path in episode_root.glob("episode_[0-9]*")
                           if path.is_dir())
        # Legacy/minimal states may not include artifact paths.  Their first N
        # episode directories predate the crash and correspond to N committed
        # evidence records by the old allocator contract.
        if not committed:
            committed={str(path.resolve()) for path in directories[:len(records)]}
        for directory in directories:
            if str(directory.resolve()) in committed:continue
            trace=directory/"adapter_trace.json";rollout=directory/"rollout.mp4"
            marker=directory/"aborted_infrastructure.json"
            if marker.is_file() or (not trace.is_file() and not rollout.is_file()):continue
            events=[]
            if trace.is_file():
                try:
                    decoded=json.loads(trace.read_text())
                    if isinstance(decoded,list):events=decoded
                except (OSError,json.JSONDecodeError):pass
            action_count=sum(1 for event in events
                             if isinstance(event,dict) and event.get("event")=="act")
            payload={
                "protocol":"embodied-codex-aborted-rollout-v1",
                "status":"aborted_infrastructure",
                "reason":"process_terminated_before_execution_transaction_commit",
                "created_unix":time.time(),
                "event_count":len(events),"robot_action_count":action_count,
                "benchmark_signal_exposed":False,
                "artifacts":{
                    "adapter_trace":str(trace.resolve()) if trace.is_file() else None,
                    "adapter_trace_sha256":(hashlib.sha256(trace.read_bytes()).hexdigest()
                                             if trace.is_file() else None),
                    "rollout":str(rollout.resolve()) if rollout.is_file() else None,
                    "rollout_sha256":(hashlib.sha256(rollout.read_bytes()).hexdigest()
                                      if rollout.is_file() else None)}}
            temporary=marker.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload,indent=2)+"\n");temporary.replace(marker)
    def select_case(self,case_handle: str):
        """Select one opaque case for an infrastructure-attribution replay."""
        matches=[episode for episode in self.episodes if episode.case_handle==case_handle]
        if len(matches)!=1:
            raise ValueError("infrastructure replay case handle is not uniquely registered")
        self._next_case_handle=str(case_handle)
    def __call__(self):
        self.count+=1
        if self._next_case_handle is None:
            episode=self.episodes[self.case_cursor%len(self.episodes)]
            self.case_cursor+=1
        else:
            episode=next(item for item in self.episodes
                         if item.case_handle==self._next_case_handle)
            self._next_case_handle=None
        return LiberoDeployment(episode=episode,
            artifact_dir=self.root/"episodes"/f"episode_{self.count:03d}",
            capabilities=self.capabilities,capability_contracts=self.capability_contracts,
            verifiers=self.verifiers,
            outcome_verifier=self.outcome_verifier)


def language(config,suite_name,task_index):
    os.environ["LIBERO_CONFIG_PATH"]=str(Path(config).resolve())
    from libero.libero import benchmark
    return str(benchmark.get_benchmark_dict()[suite_name]().get_task(task_index).language)


def _bind_runtime_model_configuration(root: Path, value: dict) -> None:
    """Persist model-role settings so a resumed run cannot silently change them."""
    path=root/"harness_configuration.json"
    configuration=json.loads(path.read_text())
    previous=configuration.get("runtime_model_configuration")
    if previous is not None and previous!=value:
        raise RuntimeError("resumed run runtime model configuration mismatch")
    if previous is None:
        configuration["runtime_model_configuration"]=value
        configuration.setdefault("configuration_migrations",[]).append({
            "kind":"bind_runtime_model_roles_v1","value":value})
        path.write_text(json.dumps(configuration,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-dir",required=True)
    p.add_argument("--suite",default="libero_spatial");p.add_argument("--task",type=int,default=0)
    p.add_argument("--state",type=int,default=0);p.add_argument("--max-iterations",type=int,default=12)
    p.add_argument("--states",type=int,nargs="+")
    p.add_argument("--device",default="cuda");p.add_argument("--model",default="gpt-5.6-sol")
    p.add_argument("--bootstrap-skill")
    p.add_argument("--capability-library",
                   help="Shared versioned Tool library reused across task runs")
    p.add_argument("--skill-library",help="Shared structured Skill library")
    p.add_argument("--experience-library",help="Shared evidence-backed Experience library")
    p.add_argument("--gap-library",help="Shared evidence-backed Capability Gap library")
    p.add_argument("--import-tools-from-skill",action="append",default=[])
    p.add_argument("--retry-locked-validation",action="store_true")
    p.add_argument("--reasoning-effort",default="high");p.add_argument("--base-url",default=
        os.environ.get("EMBODIED_CODEX_BASE_URL","https://api.apexin.ai/v1"))
    p.add_argument("--verifier-reasoning-effort",default=os.environ.get(
        "EMBODIED_CODEX_VERIFIER_REASONING_EFFORT","low"),
        help="Reasoning effort for the independent before/after visual verifier")
    p.add_argument("--python",default=sys.executable)
    p.add_argument("--groundingdino-root",default=str(ROOT/"third_party"/"GroundingDINO"))
    p.add_argument("--groundingdino-config",default=str(ROOT/"third_party"/"GroundingDINO"/
        "groundingdino"/"config"/"GroundingDINO_SwinT_OGC.py"))
    p.add_argument("--groundingdino-checkpoint",default=os.environ.get(
        "EMBODIED_CODEX_GROUNDINGDINO_CHECKPOINT",str(ROOT/"checkpoints"/"groundingdino_swint_ogc.pth")))
    p.add_argument("--sam-root",default=str(ROOT/"third_party"/"segment-anything"))
    p.add_argument("--sam-checkpoint",default=str(ROOT/"checkpoints"/"sam_vit_b_01ec64.pth"))
    p.add_argument("--graspnet-backend",default=str(ROOT/"embodied_codex"/"capabilities"/"graspnet_backend.py"))
    p.add_argument("--graspnet-checkpoint",default=str(ROOT/"checkpoints"/"graspnet-checkpoint-rs.tar"))
    p.add_argument("--config",default="config/standalone_libero")
    args=p.parse_args();key=(os.environ.get("EMBODIED_CODEX_API_KEY") or
        os.environ.get("OPENAI_API_KEY") or os.environ.get("APEX_API_KEY"))
    if not key:raise SystemExit("set EMBODIED_CODEX_API_KEY, OPENAI_API_KEY, or APEX_API_KEY")
    root=Path(args.run_dir).resolve();task=language(args.config,args.suite,args.task)
    perception=OpenVocabularyRGBD(
        groundingdino_root=args.groundingdino_root,
        groundingdino_config=args.groundingdino_config,
        groundingdino_checkpoint=args.groundingdino_checkpoint,
        sam_root=args.sam_root,sam_checkpoint=args.sam_checkpoint,
        device=args.device)
    graspnet=GraspNetRGBD(
        backend_script=args.graspnet_backend,
        checkpoint=args.graspnet_checkpoint,python=args.python)
    relation_grounder=VLMVisualRelationGrounder(
        api_key=key,base_url=args.base_url,model=args.model,
        reasoning_effort=args.reasoning_effort)
    outcome_verifier=VLMVisualTaskOutcomeVerifier(
        api_key=key,base_url=args.base_url,model=args.model,
        reasoning_effort=args.verifier_reasoning_effort,consensus_rounds=3)
    model=OpenAIModel(api_key=key,base_url=args.base_url,model=args.model,
                      reasoning_effort=args.reasoning_effort)
    state_indices=list(dict.fromkeys(args.states or [args.state]))
    case_ledger=root/"episode_case_ledger.json"
    if case_ledger.is_file():
        private_cases=json.loads(case_ledger.read_text())
        if private_cases.get("states")!=state_indices:raise RuntimeError("resumed episode case ledger mismatch")
        case_handles=list(private_cases["opaque_handles"])
    else:
        case_handles=["case-"+secrets.token_hex(12) for _ in state_indices]
        case_ledger.parent.mkdir(parents=True,exist_ok=True)
        case_ledger.write_text(json.dumps({"protocol":"embodied-codex-private-case-map-v1",
            "states":state_indices,"opaque_handles":case_handles},indent=2)+"\n")
    episodes=[LiberoEpisode(args.suite,args.task,state,config_path=args.config,
                            case_handle=handle)
              for state,handle in zip(state_indices,case_handles)]
    engine=EvolutionEngine(root=root,model=model,deployment_factory=lambda:None,
                           python=args.python,
                           capability_root=args.capability_library,
                           required_success_cases=(case_handles if len(state_indices)>1 else None),
                           retry_locked_validation_once=args.retry_locked_validation,
                           # The canonical path is a free Coding Agent. Optional
                           # semantic critics remain available for controlled
                           # ablations, but do not sit in front of every LIBERO
                           # rollout as an LLM-authored execution gate.
                           require_task_model=LIBERO_REQUIRE_TASK_MODEL,
                           require_task_fidelity_review=LIBERO_REQUIRE_TASK_FIDELITY_REVIEW,
                           success_evidence_protocol=(
                               "libero-sensor-v2-independent-vlm-before-after"),
                           experience_root=(args.experience_library or
                               ((Path(args.capability_library).resolve().parent/"shared_experiences")
                                if args.capability_library else None)),
                           skill_root=(args.skill_library or
                               ((Path(args.capability_library).resolve().parent/"shared_skills")
                                if args.capability_library else None)),
                           gap_root=(args.gap_library or
                               ((Path(args.capability_library).resolve().parent/"shared_gaps")
                                if args.capability_library else None)))
    _bind_runtime_model_configuration(root,{
        "coding_agent":{"model":args.model,
                        "reasoning_effort":args.reasoning_effort},
        "visual_relation_grounder":{"model":args.model,
                                    "reasoning_effort":args.reasoning_effort},
        "independent_task_outcome_verifier":{
            "model":args.model,
            "reasoning_effort":args.verifier_reasoning_effort,
            "consensus_rounds":3,
            "total_timeout_seconds":outcome_verifier.total_timeout},
    })
    bootstrap=engine.bootstrap_skill(args.bootstrap_skill) if args.bootstrap_skill else None
    imported_tool_sources=[]
    for skill_path in args.import_tools_from_skill:
        skill_root=Path(skill_path).resolve()
        imported=engine.capabilities.import_skill_tools(skill_root)
        skill_manifest=json.loads((skill_root/"manifest.json").read_text())
        # The coding agent composes imported Tool contracts; it does not need
        # (and cannot use) a host path to another run's controller.  Exposing
        # that path invites an irrelevant cross-workspace read attempt instead
        # of engineering the current task from its complete instruction.
        imported_tool_sources.append({
            "source_skill_id":skill_manifest.get("skill_id"),**imported})
    perception_asset=engine.capabilities.register_deployment_tool(
        name="open_vocab_rgbd_grounded_sam",
        implementation_path=str(Path(__file__).parents[2]/"embodied_codex/capabilities/open_vocab_rgbd.py"),
        dependency_paths={"perception_reliability":str(Path(__file__).parents[2]/
            "embodied_codex/capabilities/perception_reliability.py")},
        description="Public task-disjoint GroundingDINO + SAM metric RGB-D perception",
        input_schema={"type":"object","properties":{"frame":{"type":"object"},
            "queries":{"type":"array","items":{"type":"string"},"minItems":1},
            "distinct_query_pairs":{"type":"array","items":{"type":"array",
                "items":{"type":"string"},"minItems":2,"maxItems":2}}},
            "required":["frame","queries"],"additionalProperties":False},
        output_schema={"type":"object","properties":{"detections":{"type":"object",
            "additionalProperties":{"type":"array","items":{"type":"object",
                "properties":{"box_xyxy":{"type":"array"},"mask_path":{"type":"string"},
                    "world_xyz":{"type":"array"},"world_bounds_10_90":{"type":"array"},
                    "point_ref":{"type":"string"},"projection_error":{"type":"string"}},
                "required":["box_xyxy"],
                "anyOf":[{"required":["mask_path","world_xyz"]},
                         {"required":["projection_error"]}]}}},
            "reliability":{"type":"object"}},
            "required":["detections","reliability"]},
        provenance=perception.provenance,
        manual={
          "purpose":"Detect open-vocabulary instances in an Adapter RGB-D frame and attach metric, opaque motion references.",
          "when_to_use":["When a controller needs live object candidates from RGB-D."],
          "inputs":{"frame":"Unmodified robot.observe(channel='rgbd') result",
                    "queries":"List of noun phrases; output keys exactly match these strings",
                    "distinct_query_pairs":"Optional pairs which task semantics require to denote different physical entities"},
          "outputs":{"detections":"Mapping from each exact query to original records containing box_xyxy, mask_path, world_xyz, world_bounds_10_90, and point_ref",
                     "reliability":"Evidence-sufficiency report; it does not correct labels or select a target"},
          "examples":[{"call":"robot.use(tool_id, {'frame': frame, 'queries': ['object', 'destination']})"}],
          "failure_modes":["A query can return zero candidates.",
                           "Open-vocabulary labels can overlap or refer to the same pixels.",
                           "A detection label or confidence does not prove a spatial relation."],
          "limitations":["Only original returned records own valid point_ref provenance.",
                         "Uses visible RGB-D surfaces; occluded entities may be absent."]})
    grasp_asset=engine.capabilities.register_deployment_tool(
        name="graspnet_rgbd_6dof",
        implementation_path=str(Path(__file__).parents[2]/"embodied_codex/capabilities/graspnet_rgbd.py"),
        dependency_paths={"open_vocab_rgbd":str(Path(__file__).parents[2]/
            "embodied_codex/capabilities/open_vocab_rgbd.py")},
        description="Public task-disjoint GraspNet RGB-D ranked 6-DoF grasp proposals",
        input_schema={"type":"object","properties":{"frame":{"type":"object"},
            "detection":{"type":"object"}},"required":["frame","detection"],
            "additionalProperties":False},
        output_schema={"type":"object","properties":{
            "full_6dof_grasps":{"type":"array","items":{"type":"object",
                "properties":{"world_xyz":{"type":"array"},
                    "eef_rotation_world":{"type":"array"},"pose_ref":{"type":"string"},
                    "score":{"type":"number"}},
                "required":["world_xyz","eef_rotation_world"]}},
            "calibrated_topdown_grasps":{"type":"array","items":{"type":"object",
                "properties":{"world_xyz":{"type":"array"},
                    "eef_rotation_world":{"type":"array"},"pose_ref":{"type":"string"},
                    "score":{"type":"number"}},
                "required":["world_xyz","eef_rotation_world"]}}},
            "required":["full_6dof_grasps","calibrated_topdown_grasps"]},
        provenance=graspnet.provenance,
        manual={
          "purpose":"Generate task-disjoint RGB-D grasp proposals for one original detector record.",
          "when_to_use":["After selecting a live detection and before choosing grasp motion."],
          "inputs":{"frame":"The same RGB-D observation used for detection",
                    "detection":"One original detector record, not a copied or reconstructed record"},
          "outputs":{"full_6dof_grasps":"Ranked poses retaining learned 6-DoF orientation",
                     "calibrated_topdown_grasps":"Ranked poses with Panda-compatible downward orientation; records contain world_xyz, eef_rotation_world, pose_ref, scores, and geometry diagnostics"},
          "examples":[{"call":"robot.use(tool_id, {'frame': frame, 'detection': selected_original_record})"}],
          "failure_modes":["A proposed pose can be unreachable under the Adapter controller.",
                           "A reachable pose can still close empty or collide.",
                           "Candidate lists can be empty after geometric filtering."],
          "limitations":["Ranking is proposal evidence, not physical success.",
                         "Use action receipts and fresh sensor verification after execution."]})
    relation_asset=engine.capabilities.register_deployment_tool(
        name="vlm_visual_relation_grounder",
        implementation_path=str(Path(__file__).parents[2]/
                                "embodied_codex/capabilities/vlm_relation_grounder.py"),
        dependency_paths={"_vlm_support":str(Path(__file__).parents[2]/
            "embodied_codex/capabilities/_vlm_support.py")},
        description=("Foundation VLM consensus selection of a live object-reference "
                     "pair from an annotated Adapter RGB image; sensor-only"),
        input_schema={"type":"object","properties":{"frame":{"type":"object"},
            "instruction":{"type":"string"},"relation":{"type":"string"},
            "candidates":{"type":"array","items":{"type":"object"}},
            "reference_candidates":{"type":"array","items":{"type":"object"}},
            "consensus_rounds":{"type":"integer","minimum":1,"maximum":5}},
            "required":["frame","instruction","relation","candidates","reference_candidates"]},
        output_schema={"type":"object","properties":{
            "selected_index":{"type":["integer","null"]},
            "selected_reference_index":{"type":["integer","null"]},
            "identity_rule":{"type":"string"},"confidence":{"type":"number"},
            "consensus":{}},"required":["selected_index","selected_reference_index"]},
        provenance=relation_grounder.provenance,
        manual={
          "purpose":"Use a foundation VLM to select a language-grounded object/reference pair from annotated live candidates.",
          "when_to_use":["When task language distinguishes same-class instances by a visual relation."],
          "inputs":{"frame":"Adapter RGB observation",
                    "instruction":"Complete live task instruction",
                    "relation":"The relation to evaluate",
                    "candidates":"Exact object candidate list",
                    "reference_candidates":"Exact reference candidate list",
                    "consensus_rounds":"Optional integer 1..5"},
          "outputs":{"selected_index":"Index into candidates or null",
                     "selected_reference_index":"Index into reference_candidates or null",
                     "identity_rule":"How returned indices bind to original candidate records",
                     "confidence":"VLM confidence, not metric proof",
                     "consensus":"Vote evidence"},
          "examples":[{"call":"selection = robot.use(tool_id, payload); source = candidates[selection['selected_index']]"}],
          "failure_modes":["Can return null indices or a structured tool_error.",
                           "Visual ambiguity can produce a wrong but confident pair."],
          "limitations":["Returned indices select the original arrays; copied descriptions are not motion references.",
                         "Must be cross-checked against task evidence when ambiguity matters."]})
    current_deployment_tools={
        "open_vocab_rgbd_grounded_sam":perception_asset["tool_id"],
        "graspnet_rgbd_6dof":grasp_asset["tool_id"],
        "vlm_visual_relation_grounder":relation_asset["tool_id"]}
    engine.guidance["active_deployment_tool_ids"]=sorted(current_deployment_tools.values())
    engine.guidance["deployment_dependency_binding"]=engine.bind_current_deployment_tools(
        current_deployment_tools)
    if bootstrap:
        replacements={
            item["tool_id"]:current_deployment_tools[item["name"]]
            for item in bootstrap["deployment_dependencies"]
            if item.get("name") in current_deployment_tools}
        engine.guidance["bootstrap_dependency_replacements"]=replacements
        engine.guidance["bootstrap_dependency_binding"]=engine.bind_bootstrap_deployment_tools(
            replacements)
    engine.deployment_factory=Factory(episodes=episodes,run_root=root,
        capabilities={perception_asset["tool_id"]:perception.detect,
                      grasp_asset["tool_id"]:graspnet.infer,
                      relation_asset["tool_id"]:relation_grounder.select},
        capability_contracts={tool_id:{key:engine.capabilities.inspect(tool_id)["manifest"][key]
                              for key in ("input_schema","output_schema")}
                              for tool_id in current_deployment_tools.values()},
        verifiers={"visual_attachment":perception.verify_attachment,
                   "visual_support_relation":perception.verify_support_relation},
        outcome_verifier=outcome_verifier.verify)
    # The Adapter exposes mechanics and immutable Tool contracts only. It does
    # not tell the coding agent which task strategy or perception flow to use.
    # The same seed Tools are optional building blocks that the agent may test,
    # combine, replace, or ignore after inspecting the live task and evidence.
    engine.guidance.update({
        "adapter":{
            "name":"LIBERO",
            "robot":"Panda arm with OSC pose control",
            "observation_channels":["rgb","rgbd","proprioception"],
        },
        "robot_sdk_contract":LIBERO_ROBOT_SDK_CONTRACT,
        "imported_capability_assets":imported_tool_sources,
        "seed_tool_ids":sorted(current_deployment_tools.values()),
    })
    state=engine.run(task=task,skill_name=f"libero_spatial_task_{args.task}_skill",
                     max_iterations=args.max_iterations)
    print(json.dumps({"status":state["status"],"iterations":len(state["iterations"]),
                      "skill":state.get("skill")},indent=2));return 0 if state["status"]=="sensor_success" else 2


if __name__=="__main__":raise SystemExit(main())
