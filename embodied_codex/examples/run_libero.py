"""Run the free-coding Embodied Codex loop on one LIBERO task/state."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from embodied_codex.capabilities import (GraspNetRGBD,OpenVocabularyRGBD,
                                         VLMVisualRelationGrounder,
                                         VLMVisualTaskOutcomeVerifier)
from embodied_codex.deployments import LiberoDeployment,LiberoEpisode
from embodied_codex.evolution import EvolutionEngine
from embodied_codex.model import OpenAIModel
from embodied_codex.sdk_contract import LIBERO_ROBOT_SDK_CONTRACT

class Factory:
    def __init__(self,*,episodes,run_root,capabilities,verifiers,outcome_verifier=None):
        self.episodes=list(episodes);self.root=Path(run_root)
        self.outcome_verifier=outcome_verifier
        if not self.episodes:raise ValueError("at least one LIBERO episode case is required")
        self.capabilities=dict(capabilities);self.verifiers=dict(verifiers)
        episode_root=self.root/"episodes";episode_root.mkdir(parents=True,exist_ok=True)
        existing=[int(p.name.split("_")[-1]) for p in episode_root.glob("episode_[0-9]*")
                  if p.name.split("_")[-1].isdigit()]
        self.count=max(existing,default=0)
    def __call__(self):
        self.count+=1
        episode=self.episodes[(self.count-1)%len(self.episodes)]
        return LiberoDeployment(episode=episode,
            artifact_dir=self.root/"episodes"/f"episode_{self.count:03d}",
            capabilities=self.capabilities,verifiers=self.verifiers,
            outcome_verifier=self.outcome_verifier)


def language(config,suite_name,task_index):
    os.environ["LIBERO_CONFIG_PATH"]=str(Path(config).resolve())
    from libero.libero import benchmark
    return str(benchmark.get_benchmark_dict()[suite_name]().get_task(task_index).language)


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-dir",required=True)
    p.add_argument("--suite",default="libero_spatial");p.add_argument("--task",type=int,default=0)
    p.add_argument("--state",type=int,default=0);p.add_argument("--max-iterations",type=int,default=12)
    p.add_argument("--states",type=int,nargs="+")
    p.add_argument("--device",default="cuda");p.add_argument("--model",default="gpt-5.6-sol")
    p.add_argument("--bootstrap-skill")
    p.add_argument("--capability-library",
                   help="Shared versioned Tool library reused across task runs")
    p.add_argument("--import-tools-from-skill",action="append",default=[])
    p.add_argument("--retry-locked-validation",action="store_true")
    p.add_argument("--reasoning-effort",default="high");p.add_argument("--base-url",default="https://api.apexin.ai/v1")
    p.add_argument("--config",default="config/standalone_libero")
    args=p.parse_args();key=os.environ.get("APEX_API_KEY")
    if not key:raise SystemExit("APEX_API_KEY missing")
    root=Path(args.run_dir).resolve();task=language(args.config,args.suite,args.task)
    perception=OpenVocabularyRGBD(
        groundingdino_root="third_party/GroundingDINO",
        groundingdino_config="third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        groundingdino_checkpoint="/data/zxy/GroundingDINO/models/groundingdino_swint_ogc.pth",
        sam_root="third_party/segment-anything",sam_checkpoint="checkpoints/sam_vit_b_01ec64.pth",
        device=args.device)
    graspnet=GraspNetRGBD(
        backend_script="capability_library/tools/graspnet_rgbd_grasp.py",
        checkpoint="checkpoints/graspnet-checkpoint-rs.tar",
        python="/data/zxy/envs/vla-report/bin/python")
    relation_grounder=VLMVisualRelationGrounder(
        api_key=key,base_url=args.base_url,model=args.model,
        reasoning_effort=args.reasoning_effort)
    outcome_verifier=VLMVisualTaskOutcomeVerifier(
        api_key=key,base_url=args.base_url,model=args.model,
        reasoning_effort=args.reasoning_effort,consensus_rounds=3)
    model=OpenAIModel(api_key=key,base_url=args.base_url,model=args.model,
                      reasoning_effort=args.reasoning_effort)
    state_indices=list(dict.fromkeys(args.states or [args.state]))
    episodes=[LiberoEpisode(args.suite,args.task,state,config_path=args.config)
              for state in state_indices]
    engine=EvolutionEngine(root=root,model=model,deployment_factory=lambda:None,
                           python="/data/zxy/envs/vla-report/bin/python",
                           capability_root=args.capability_library,
                           required_success_cases=([f"state_{state:03d}" for state in state_indices]
                                                   if len(state_indices)>1 else None),
                           retry_locked_validation_once=args.retry_locked_validation,
                           success_evidence_protocol=(
                               "libero-sensor-v2-independent-vlm-before-after"))
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
        description="Public task-disjoint GroundingDINO + SAM metric RGB-D perception",
        input_schema={"frame":"RGB-D observation","queries":["text"]},
        output_schema={"detections":"query -> live instances with world_xyz and point_ref"},
        provenance=perception.provenance)
    grasp_asset=engine.capabilities.register_deployment_tool(
        name="graspnet_rgbd_6dof",
        implementation_path=str(Path(__file__).parents[2]/"embodied_codex/capabilities/graspnet_rgbd.py"),
        description="Public task-disjoint GraspNet RGB-D ranked 6-DoF grasp proposals",
        input_schema={"frame":"RGB-D observation","detection":"GroundingDINO+SAM instance"},
        output_schema={"full_6dof_grasps":"ranked poses with opaque pose_ref",
                       "calibrated_topdown_grasps":"ranked fallback poses with opaque pose_ref"},
        provenance=graspnet.provenance)
    relation_asset=engine.capabilities.register_deployment_tool(
        name="vlm_visual_relation_grounder",
        implementation_path=str(Path(__file__).parents[2]/
                                "embodied_codex/capabilities/vlm_relation_grounder.py"),
        description=("Foundation VLM consensus selection of a live object-reference "
                     "pair from an annotated Adapter RGB image; sensor-only"),
        input_schema={"frame":"RGB observation","instruction":"task language",
                      "relation":"relation phrase","candidates":"live object detections",
                      "reference_candidates":"live reference detections",
                      "consensus_rounds":"1..5, default 3"},
        output_schema={"selected_index":"index into the exact input candidates, or null",
                       "selected_reference_index":"index into the exact input reference_candidates, or null",
                       "identity_rule":"indices are authoritative; retrieve original detector records and point_ref values",
                       "confidence":"0..1",
                       "consensus":"agreement evidence"},
        provenance=relation_grounder.provenance)
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
        verifiers={"visual_attachment":perception.verify_attachment,
                   "visual_support_relation":perception.verify_support_relation},
        outcome_verifier=outcome_verifier.verify)
    engine.guidance.update({"robot":"LIBERO Panda with OSC pose control",
      "robot_sdk_contract":LIBERO_ROBOT_SDK_CONTRACT,
      "imported_capability_assets":imported_tool_sources,
      "perception":{"tool_id":perception_asset["tool_id"],
        "example":"robot.use(tool_id, {'frame': frame, 'queries': ['black bowl','plate','ramekin']})",
        "output":"direct dict with detections; each instance has live world_xyz and point_ref"},
      "grasp_generation":{"tool_id":grasp_asset["tool_id"],
        "example":"robot.use(tool_id, {'frame': frame, 'detection': selected_detection})",
        "output":"ranked full_6dof_grasps and calibrated_topdown_grasps; each pose has world_xyz, approach_world, point_ref, pose_ref",
        "provenance":"public GraspNet checkpoint; RGB-D + GroundingDINO/SAM only"},
      "relation_grounding":{"tool_id":relation_asset["tool_id"],
        "example":("robot.use(tool_id, {'frame': frame, 'instruction': "
                   "robot.instruction, 'relation': 'the bowl on the cookie box', "
                   "'candidates': all_live_bowl_detections, "
                   "'reference_candidates': all_live_cookie_box_detections})"),
        "output":"selected_index and selected_reference_index into the exact input lists, plus rationale, confidence and consensus; no copied motion records",
        "rule":("GroundingDINO phrase scores do not prove spatial relations. For an "
                "instruction that identifies the source by a relation, pass both the "
                "object and reference candidates to this VLM pair selector, then "
                "use selected_index and selected_reference_index to retrieve the exact "
                "original detector records. Never match or act on a copied VLM record; "
                "only original RGB-D records own point_ref motion provenance. Require "
                "the selected reference to match the independent RGB-D pair. "
                "A VLM description is a proposal and must never redefine a generic "
                "support as the named reference merely to validate its own object choice. Then "
                "retain metric RGB-D and opaque point_ref provenance for motion.")},
      "independent_outcome_gate":(
        "After controller termination the Harness independently compares initial and final "
        "RGB against the full task language. A same-class object at the destination is not "
        "success when the relation-identified source object remains in place."),
      "language_grounding_invariant":(
        "An object identified by a source relation in the instruction must be "
        "selected from a live object-reference geometry pair. If the reference "
        "is missing, fail closed; never fall back to an unrelated same-class instance. "
        "A relation phrase submitted to GroundingDINO is not itself relation evidence; "
        "visually disambiguate competing instances with the supplied VLM relation "
        "grounder and verify the chosen live metric candidate."),
      })
    state=engine.run(task=task,skill_name=f"libero_spatial_task_{args.task}_skill",
                     max_iterations=args.max_iterations)
    print(json.dumps({"status":state["status"],"iterations":len(state["iterations"]),
                      "skill":state.get("skill")},indent=2));return 0 if state["status"]=="sensor_success" else 2


if __name__=="__main__":raise SystemExit(main())
