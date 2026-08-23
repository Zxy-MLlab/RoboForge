"""Single machine-readable Robot SDK contract for the LIBERO Adapter.

The coding prompt, compile-time linter, and deployment runtime all consume this
object.  Human prose is descriptive only and cannot introduce aliases.
"""
from __future__ import annotations

from typing import Any, Mapping


LIBERO_ROBOT_SDK_CONTRACT = {
    "protocol":"embodied-codex-libero-robot-sdk-v1",
    "methods":{
        "observe":{
            "signature":"robot.observe(channel='rgbd', request={})",
            "channels":["rgb","rgbd","proprioception"],
            "returns":"direct observation object"},
        "use":{
            "signature":"robot.use(tool_id, payload)",
            "returns":"direct Tool result; no receipt/result wrapper"},
        "act":{
            "signature":"robot.act(action)",
            "returns":"action receipt with type, reached, step, eef_before, eef_after, gripper_qpos"},
        "verify":{
            "signature":"robot.verify(verifier, payload)",
            "returns":"direct sensor-only verifier result containing boolean verified"},
        "record":{"signature":"robot.record(event)","returns":"{recorded: true}"},
    },
    "actions":{
        "move_to_point":{
            "required":["type","target_ref"],
            "optional":{"offset":"world [dx,dy,dz], default [0,0,0]",
                        "tolerance_m":"0.002..0.06","gain":"1..30",
                        "max_steps":"1..100","gripper":"-1 open, +1 close"},
            "example":{"type":"move_to_point","target_ref":"<point_ref>",
                       "offset":[0,0,0.10],"gripper":-1}},
        "move_to_pose":{
            "required":["type","pose_ref"],
            "optional":{"offset":"world [dx,dy,dz], default [0,0,0]",
                        "quaternion_xyzw":"explicit orientation override",
                        "rotation_matrix":"explicit 3x3 orientation override",
                        "position_tolerance_m":"0.002..0.06",
                        "orientation_tolerance_rad":"0.02..0.5",
                        "position_gain":"1..30","orientation_gain":"0.05..1",
                        "max_steps":"1..180","gripper":"-1 open, +1 close"},
            "rule":"pose_ref must come from a Tool result with eef_rotation_world unless an explicit orientation override is supplied",
            "example":{"type":"move_to_pose","pose_ref":"<pose_ref>",
                       "offset":[0,0,0],"gripper":-1}},
        "osc_delta":{
            "required":["type","translation","rotation"],
            "optional":{"gripper":"-1 open, +1 close","repeat":"1..20"},
            "example":{"type":"osc_delta","translation":[0,0,0],
                       "rotation":[0,0,0],"gripper":-1,"repeat":1}},
        "gripper":{
            "required":["type","command"],"enum":{"command":["open","close"]},
            "optional":{"repeat":"1..40"},
            "example":{"type":"gripper","command":"open","repeat":12}},
        "settle":{
            "required":["type"],
            "optional":{"steps":"1..60","gripper":"-1 open, +1 close"},
            "example":{"type":"settle","steps":10,"gripper":-1}},
    },
    "verifiers":{
        "visual_attachment":{
            "required":["frame","object_query","source_ref"],
            "rule":"source_ref is the original pre-action detector point_ref; call only after close, lift, and a fresh observation",
            "example":{"frame":"<fresh RGB-D frame>","object_query":"bowl",
                       "source_ref":"<original source point_ref>"}},
        "visual_support_relation":{
            "required":["frame","object_query","target_query","source_ref","target_ref"],
            "rule":"source_ref is the original source point_ref and target_ref is a pre-release support point_ref; both must be non-null",
            "example":{"frame":"<fresh RGB-D frame>","object_query":"bowl",
                       "target_query":"plate","source_ref":"<source point_ref>",
                       "target_ref":"<target point_ref>"}},
    },
    "reference_rules":[
        "point_ref and pose_ref are opaque Adapter-issued strings",
        "only original metric perception records own motion point_ref provenance",
        "selector indices retrieve original candidates; selector copies are never motion identities",
        "never call act or verify with a null, fabricated, or stale reference",
    ],
}


class SDKContractError(ValueError):
    pass


def validate_action(action: Mapping[str,Any]) -> str:
    if not isinstance(action,Mapping):raise SDKContractError("action must be an object")
    kind=action.get("type")
    contracts=LIBERO_ROBOT_SDK_CONTRACT["actions"]
    if kind not in contracts:
        raise SDKContractError(f"unsupported action type {kind!r}; allowed={sorted(contracts)}")
    missing=[key for key in contracts[kind]["required"] if key not in action]
    if missing:raise SDKContractError(f"action {kind} missing required fields {missing}")
    if kind=="gripper" and action.get("command") not in ("open","close"):
        raise SDKContractError("gripper command must be 'open' or 'close'")
    for key in ("target_ref","pose_ref"):
        if key in action and (not isinstance(action[key],str) or not action[key]):
            raise SDKContractError(f"{kind} {key} must be a nonempty opaque string")
    return str(kind)


def validate_verifier_request(name: str,payload: Mapping[str,Any]) -> None:
    contracts=LIBERO_ROBOT_SDK_CONTRACT["verifiers"]
    if name not in contracts:raise SDKContractError(f"unknown verifier {name!r}")
    if not isinstance(payload,Mapping):raise SDKContractError("verifier payload must be an object")
    missing=[key for key in contracts[name]["required"] if key not in payload]
    if missing:raise SDKContractError(f"verifier {name} missing required fields {missing}")
    for key in ("source_ref","target_ref"):
        if key in contracts[name]["required"]:
            value=payload.get(key)
            if not isinstance(value,str) or not value:
                raise SDKContractError(f"verifier {name} {key} must be a nonempty opaque string")


__all__=["LIBERO_ROBOT_SDK_CONTRACT","SDKContractError","validate_action",
         "validate_verifier_request"]
