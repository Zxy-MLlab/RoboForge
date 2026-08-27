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
            "returns":"direct observation object; no receipt/result wrapper",
            "output_fields":["frame_id","step","cameras","proprioception"],
            "returns_by_channel":{
                "rgbd":{
                    "shape":"{frame_id, step, cameras, proprioception}",
                    "rule":"camera RGB/depth/calibration are under cameras; robot state is under proprioception"},
                "rgb":{
                    "shape":"{frame_id, step, cameras, proprioception}",
                    "rule":"camera RGB/calibration are under cameras; robot state is under proprioception"},
                "proprioception":{
                    "shape":"{step, proprioception}",
                    "fields":"read eef_pose, gripper, joint_state, and proprioception from result['proprioception']",
                    "example":"obs = robot.observe(channel='proprioception', request={}); pose = obs['proprioception']['eef_pose']"}}},
        "use":{
            "signature":"robot.use(tool_id, payload)",
            "returns":"direct Tool result; no receipt/result wrapper"},
        "act":{
            "signature":"robot.act(action)",
        "returns":"action receipt with type, reached, step, eef_before, eef_after, and gripper_width_m",
        "output_fields":["type","step","reached","eef_before","eef_after","gripper_width_m",
                          "target_xyz","target_quaternion_xyzw","final_position_error_m",
                          "final_orientation_error_rad","target_frame","action_frame_axis",
                          "action_frame_axis_frame"]},
        "verify":{
            "signature":"robot.verify(verifier, payload)",
            "returns":"direct sensor-only verifier result containing boolean verified",
            "output_fields":["verified","sensor_only","verifier_error","reason"]},
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
            "required":["type"],
            "any_of":[
                {"required":["pose_ref"]},
                {"required":["target_ref","quaternion_xyzw"]},
                {"required":["target_ref","rotation_matrix"]},
            ],
            "optional":{"offset":"world [dx,dy,dz], default [0,0,0]",
                        "quaternion_xyzw":"explicit orientation override",
                        "rotation_matrix":"explicit 3x3 orientation override",
                        "position_tolerance_m":"0.002..0.06",
                        "orientation_tolerance_rad":"0.02..0.5",
                        "position_gain":"1..30","orientation_gain":"0.05..1",
                        "max_steps":"1..180","gripper":"-1 open, +1 close"},
            "rule":("Use pose_ref for a Tool-issued metric pose. To keep metric position and "
                    "orientation provenance separate, use target_ref from metric perception "
                    "together with an explicit quaternion_xyzw or rotation_matrix."),
            "example":{"type":"move_to_pose","pose_ref":"<pose_ref>",
                       "offset":[0,0,0],"gripper":-1}},
        "osc_delta":{
            "required":["type","translation","rotation"],
            "field_semantics":{
                "translation":"normalized OSC command [x,y,z] in [-1,1] per simulator control step; NOT metres",
                "rotation":"normalized axis command [rx,ry,rz] in [-1,1] per simulator control step; NOT radians",
            },
            "optional":{"gripper":"-1 open, +1 close",
                        "repeat":"number of control steps, 1..20"},
            "rule":"Use move_to_point/move_to_pose for metric targets; use osc_delta only for short feedback motions and verify displacement from the returned eef_after.",
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
            "example":{"frame":"<fresh RGB-D frame>","object_query":"<object>",
                       "source_ref":"<original source point_ref>"}},
        "visual_support_relation":{
            "required":["frame","object_query","target_query","source_ref","target_ref"],
            "optional":{"transport_ref":(
                "point_ref from the successful visual_attachment receipt when a retry used "
                "a fresh detection; omit when it is identical to source_ref")},
            "rule":(
                "source_ref is the original task-source point_ref, target_ref is a pre-release "
                "support point_ref, and optional transport_ref identifies the fresh detection "
                "whose attachment was verified; all supplied refs must be non-null"),
            "example":{"frame":"<fresh RGB-D frame>","object_query":"<object>",
                       "target_query":"<support>","source_ref":"<source point_ref>",
                       "transport_ref":"<successfully attached point_ref>",
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
    alternatives=contracts[kind].get("any_of") or []
    if alternatives and not any(all(key in action for key in option.get("required") or [])
                                for option in alternatives):
        required_sets=[option.get("required") or [] for option in alternatives]
        raise SDKContractError(f"action {kind} requires one of field sets {required_sets}")
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
    for key in ("source_ref","target_ref","transport_ref"):
        if key in contracts[name]["required"] or key in payload:
            value=payload.get(key)
            if not isinstance(value,str) or not value:
                raise SDKContractError(f"verifier {name} {key} must be a nonempty opaque string")


__all__=["LIBERO_ROBOT_SDK_CONTRACT","SDKContractError","validate_action",
         "validate_verifier_request"]
