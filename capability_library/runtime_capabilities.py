"""Sandboxed runtime bindings for frozen agent-authored capability Tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


SUPPORTED_HOOKS = (
    "grasp_retry_ranking",
    "grasp_execution_profile",
    "transport_profile",
    "support_relation_profile",
)

HOOK_CONTRACTS: dict[str, dict[str, Any]] = {
    "grasp_retry_ranking": {
        "purpose": "Rank observed public GraspNet candidates for bounded retries.",
        "input_fields": {
            "candidate_count": "integer",
            "max_attempts": "integer",
            "source_xyz": "three finite RGB-D world coordinates",
            "candidates": (
                "list of objects with index, score, translation_world, and "
                "approach_world"
            ),
            "default_candidate_indices": "list of bounded fallback indices",
        },
        "output_fields": {
            "candidate_indices": (
                "non-empty unique integer list, length <= max_attempts, each "
                "within [0, candidate_count)"
            )
        },
        "test_inputs": [
            {
                "candidate_count": 3,
                "max_attempts": 2,
                "source_xyz": [0.0, 0.0, 0.95],
                "candidates": [
                    {
                        "index": 0,
                        "score": 0.80,
                        "translation_world": [0.00, 0.00, 0.96],
                        "approach_world": [0.0, 0.0, -1.0],
                    },
                    {
                        "index": 1,
                        "score": 0.75,
                        "translation_world": [0.03, 0.00, 0.96],
                        "approach_world": [0.1, 0.0, -0.99],
                    },
                    {
                        "index": 2,
                        "score": 0.65,
                        "translation_world": [-0.02, 0.02, 0.96],
                        "approach_world": [0.0, 0.2, -0.98],
                    },
                ],
                "default_candidate_indices": [0, 2],
            }
        ],
        "example_output": {"candidate_indices": [0, 2]},
    },
    "grasp_execution_profile": {
        "purpose": (
            "Select bounded approach, contact, close, lift, and sensor "
            "reobservation behavior for one observed grasp candidate."
        ),
        "input_fields": {
            "attempt": "one-based bounded retry number",
            "candidate": "observed GraspNet score, translation, and approach",
            "source_xyz": "initial RGB-D source coordinates",
            "tracked_source_xyz": "latest sensor-tracked source coordinates",
            "current_eef_xyz": "current proprioceptive EEF coordinates",
            "previous_failure": "previous legal visual attachment evidence or null",
            "default_profile": "object containing all nine valid output fields",
        },
        "output_fields": {
            "approach_clearance_m": "number within [0.06, 0.18]",
            "grasp_z_offset_m": "number within [-0.025, 0.025]",
            "source_recenter_gain": "number within [0.0, 1.0]",
            "position_gain": "number within [0.25, 0.65]",
            "max_translation_action": "number within [0.15, 1.0]",
            "close_steps": "integer within [25, 70]",
            "post_close_settle_steps": "integer within [0, 30]",
            "lift_height_m": "number within [0.10, 0.25]",
            "reobserve_before_attempt": "boolean",
        },
        "test_inputs": [
            {
                "attempt": 2,
                "candidate": {
                    "score": 0.72,
                    "translation_world": [0.02, -0.01, 0.96],
                    "approach_world": [0.0, 0.0, -1.0],
                },
                "source_xyz": [0.02, -0.01, 0.95],
                "tracked_source_xyz": [0.03, -0.01, 0.95],
                "current_eef_xyz": [0.10, 0.05, 1.12],
                "previous_failure": {
                    "source_vacated": False,
                    "gripper_width": 0.001,
                },
                "default_profile": {
                    "approach_clearance_m": 0.12,
                    "grasp_z_offset_m": 0.0,
                    "source_recenter_gain": 1.0,
                    "position_gain": 0.65,
                    "max_translation_action": 1.0,
                    "close_steps": 45,
                    "post_close_settle_steps": 0,
                    "lift_height_m": 0.18,
                    "reobserve_before_attempt": True,
                },
            }
        ],
        "example_output": {
            "approach_clearance_m": 0.10,
            "grasp_z_offset_m": -0.005,
            "source_recenter_gain": 1.0,
            "position_gain": 0.45,
            "max_translation_action": 0.5,
            "close_steps": 50,
            "post_close_settle_steps": 10,
            "lift_height_m": 0.15,
            "reobserve_before_attempt": True,
        },
    },
    "transport_profile": {
        "purpose": "Select a bounded motion profile from live carried-object features.",
        "input_fields": {
            "current_eef_xyz": "three finite proprioceptive coordinates",
            "target_eef_xyz": "three finite sensor-derived goal coordinates",
            "carried_object_offset_xyz": "three RGB-D/EFF offset coordinates or null",
            "gripper_width": "finite proprioceptive width",
            "default_profile": "object containing the four valid output fields",
            "phase_prefix": "transport or correction_transport",
        },
        "output_fields": {
            "lift_margin_m": "number within [0.08, 0.35]",
            "horizontal_segments": "integer within [1, 6]",
            "position_gain": "number within [0.15, 0.65]",
            "max_translation_action": "number within [0.10, 1.0]",
        },
        "test_inputs": [
            {
                "current_eef_xyz": [0.0, 0.0, 1.12],
                "target_eef_xyz": [0.25, -0.12, 1.10],
                "carried_object_offset_xyz": [0.02, -0.01, -0.08],
                "gripper_width": 0.01,
                "default_profile": {
                    "lift_margin_m": 0.35,
                    "horizontal_segments": 1,
                    "position_gain": 0.65,
                    "max_translation_action": 1.0,
                },
                "phase_prefix": "transport",
            }
        ],
        "example_output": {
            "lift_margin_m": 0.12,
            "horizontal_segments": 3,
            "position_gain": 0.30,
            "max_translation_action": 0.30,
        },
    },
    "support_relation_profile": {
        "purpose": (
            "Choose conservative, bounded thresholds for deciding whether an "
            "object footprint is stably supported by a sensor-observed target."
        ),
        "input_fields": {
            "first_mask_metrics": "first settled RGB mask support metrics",
            "second_mask_metrics": "second settled RGB mask support metrics",
            "first_xy_center_error_m": "RGB-D center error in metres or null",
            "second_xy_center_error_m": "RGB-D center error in metres or null",
            "world_motion_m": "inter-observation RGB-D motion in metres or null",
            "default_profile": "object containing the five valid output fields",
        },
        "output_fields": {
            "min_containment": "number within [0.70, 0.98]",
            "min_clearance_ratio": "number within [0.60, 1.50]",
            "max_centroid_motion_px": "number within [2.0, 8.0]",
            "max_world_motion_m": "number within [0.005, 0.015]",
            "max_xy_center_error_m": "number within [0.010, 0.025]",
        },
        "test_inputs": [
            {
                "first_mask_metrics": {"containment": 0.91, "clearance_ratio": 1.08},
                "second_mask_metrics": {"containment": 0.90, "clearance_ratio": 1.04},
                "first_xy_center_error_m": 0.008,
                "second_xy_center_error_m": 0.009,
                "world_motion_m": 0.003,
                "default_profile": {
                    "min_containment": 0.75,
                    "min_clearance_ratio": 0.75,
                    "max_centroid_motion_px": 6.0,
                    "max_world_motion_m": 0.010,
                    "max_xy_center_error_m": 0.020,
                },
            }
        ],
        "example_output": {
            "min_containment": 0.80,
            "min_clearance_ratio": 0.90,
            "max_centroid_motion_px": 5.0,
            "max_world_motion_m": 0.008,
            "max_xy_center_error_m": 0.018,
        },
    },
}


class RuntimeCapabilityError(ValueError):
    """Raised when a frozen capability cannot be safely loaded or applied."""


def _validate_grasp_retry_ranking(
    value: Any, payload: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeCapabilityError("grasp_retry_ranking must return an object")
    indices = value.get("candidate_indices")
    if not isinstance(indices, list) or not indices:
        raise RuntimeCapabilityError("candidate_indices must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in indices):
        raise RuntimeCapabilityError("candidate_indices must contain integers")
    candidate_count = int(payload.get("candidate_count", 0))
    max_attempts = int(payload.get("max_attempts", 0))
    if len(indices) > max_attempts or len(set(indices)) != len(indices):
        raise RuntimeCapabilityError("candidate_indices exceed the attempt budget or repeat")
    if any(item < 0 or item >= candidate_count for item in indices):
        raise RuntimeCapabilityError("candidate index is outside the observed candidate set")
    return {"candidate_indices": indices}


def _bounded_number(
    value: Any, name: str, minimum: float, maximum: float
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeCapabilityError(f"{name} must be numeric")
    number = float(value)
    if not minimum <= number <= maximum:
        raise RuntimeCapabilityError(
            f"{name} must be within [{minimum}, {maximum}]"
        )
    return number


def _validate_transport_profile(
    value: Any, payload: Mapping[str, Any]
) -> dict[str, Any]:
    del payload
    if not isinstance(value, Mapping):
        raise RuntimeCapabilityError("transport_profile must return an object")
    allowed = {
        "lift_margin_m",
        "horizontal_segments",
        "position_gain",
        "max_translation_action",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise RuntimeCapabilityError(f"unknown transport profile fields: {unknown}")
    required = allowed
    missing = sorted(required - set(value))
    if missing:
        raise RuntimeCapabilityError(f"missing transport profile fields: {missing}")
    segments = value["horizontal_segments"]
    if isinstance(segments, bool) or not isinstance(segments, int) or not 1 <= segments <= 6:
        raise RuntimeCapabilityError("horizontal_segments must be an integer within [1, 6]")
    return {
        "lift_margin_m": _bounded_number(value["lift_margin_m"], "lift_margin_m", 0.08, 0.35),
        "horizontal_segments": segments,
        "position_gain": _bounded_number(value["position_gain"], "position_gain", 0.15, 0.65),
        "max_translation_action": _bounded_number(
            value["max_translation_action"], "max_translation_action", 0.10, 1.0
        ),
    }


def _validate_grasp_execution_profile(
    value: Any, payload: Mapping[str, Any]
) -> dict[str, Any]:
    del payload
    if not isinstance(value, Mapping):
        raise RuntimeCapabilityError("grasp_execution_profile must return an object")
    numeric_bounds = {
        "approach_clearance_m": (0.06, 0.18),
        "grasp_z_offset_m": (-0.025, 0.025),
        "source_recenter_gain": (0.0, 1.0),
        "position_gain": (0.25, 0.65),
        "max_translation_action": (0.15, 1.0),
        "lift_height_m": (0.10, 0.25),
    }
    integer_bounds = {
        "close_steps": (25, 70),
        "post_close_settle_steps": (0, 30),
    }
    required = set(numeric_bounds) | set(integer_bounds) | {"reobserve_before_attempt"}
    unknown = sorted(set(value) - required)
    missing = sorted(required - set(value))
    if unknown:
        raise RuntimeCapabilityError(f"unknown grasp execution fields: {unknown}")
    if missing:
        raise RuntimeCapabilityError(f"missing grasp execution fields: {missing}")
    result = {
        name: _bounded_number(value[name], name, minimum, maximum)
        for name, (minimum, maximum) in numeric_bounds.items()
    }
    for name, (minimum, maximum) in integer_bounds.items():
        number = value[name]
        if isinstance(number, bool) or not isinstance(number, int) or not minimum <= number <= maximum:
            raise RuntimeCapabilityError(
                f"{name} must be an integer within [{minimum}, {maximum}]"
            )
        result[name] = number
    reobserve = value["reobserve_before_attempt"]
    if not isinstance(reobserve, bool):
        raise RuntimeCapabilityError("reobserve_before_attempt must be boolean")
    result["reobserve_before_attempt"] = reobserve
    return result


def _validate_support_relation_profile(
    value: Any, payload: Mapping[str, Any]
) -> dict[str, Any]:
    del payload
    if not isinstance(value, Mapping):
        raise RuntimeCapabilityError("support_relation_profile must return an object")
    bounds = {
        "min_containment": (0.70, 0.98),
        "min_clearance_ratio": (0.60, 1.50),
        "max_centroid_motion_px": (2.0, 8.0),
        "max_world_motion_m": (0.005, 0.015),
        "max_xy_center_error_m": (0.010, 0.025),
    }
    unknown = sorted(set(value) - set(bounds))
    missing = sorted(set(bounds) - set(value))
    if unknown:
        raise RuntimeCapabilityError(f"unknown support relation fields: {unknown}")
    if missing:
        raise RuntimeCapabilityError(f"missing support relation fields: {missing}")
    return {
        name: _bounded_number(value[name], name, minimum, maximum)
        for name, (minimum, maximum) in bounds.items()
    }


_VALIDATORS = {
    "grasp_retry_ranking": _validate_grasp_retry_ranking,
    "grasp_execution_profile": _validate_grasp_execution_profile,
    "transport_profile": _validate_transport_profile,
    "support_relation_profile": _validate_support_relation_profile,
}


def validate_hook_output(
    hook: str, value: Any, payload: Mapping[str, Any]
) -> dict[str, Any]:
    if hook not in _VALIDATORS:
        raise RuntimeCapabilityError(f"unsupported capability hook: {hook}")
    return _VALIDATORS[hook](value, payload)


def transport_waypoints(
    current_eef_xyz: list[float],
    target_eef_xyz: list[float],
    profile: Mapping[str, Any],
    *,
    maximum_height_m: float = 1.55,
) -> dict[str, Any]:
    """Convert a validated transport profile into explicit Cartesian waypoints."""
    if len(current_eef_xyz) != 3 or len(target_eef_xyz) != 3:
        raise RuntimeCapabilityError("transport endpoints must be 3-D")
    current = [float(item) for item in current_eef_xyz]
    target = [float(item) for item in target_eef_xyz]
    validated = _validate_transport_profile(profile, {})
    safe_z = min(
        max(current[2], target[2]) + validated["lift_margin_m"],
        float(maximum_height_m),
    )
    lift = [current[0], current[1], safe_z]
    horizontal = []
    for segment in range(1, validated["horizontal_segments"] + 1):
        fraction = segment / validated["horizontal_segments"]
        horizontal.append(
            [
                current[0] + fraction * (target[0] - current[0]),
                current[1] + fraction * (target[1] - current[1]),
                safe_z,
            ]
        )
    return {
        "lift": lift,
        "horizontal": horizontal,
        "target": target,
        "position_gain": validated["position_gain"],
        "max_translation_action": validated["max_translation_action"],
    }


def grasp_execution_waypoints(
    base_grasp_xyz: list[float],
    source_xyz: list[float],
    tracked_source_xyz: list[float],
    approach_world: list[float],
    profile: Mapping[str, Any],
    *,
    model_orientation: bool,
) -> dict[str, Any]:
    """Turn a validated Tool profile into explicit sensor-relative grasp goals."""
    vectors = (base_grasp_xyz, source_xyz, tracked_source_xyz, approach_world)
    if any(len(vector) != 3 for vector in vectors):
        raise RuntimeCapabilityError("grasp waypoint inputs must be 3-D")
    base, source, tracked, approach = (
        [float(item) for item in vector] for vector in vectors
    )
    validated = _validate_grasp_execution_profile(profile, {})
    gain = validated["source_recenter_gain"]
    grasp = [base[index] + (tracked[index] - source[index]) * gain for index in range(3)]
    grasp[2] += validated["grasp_z_offset_m"]
    clearance = validated["approach_clearance_m"]
    if model_orientation:
        pregrasp = [grasp[index] - approach[index] * clearance for index in range(3)]
    else:
        pregrasp = [grasp[0], grasp[1], grasp[2] + clearance]
    lift = [grasp[0], grasp[1], grasp[2] + validated["lift_height_m"]]
    return {"pregrasp": pregrasp, "grasp": grasp, "lift": lift, "profile": validated}


class FrozenCapabilityRuntime:
    """Invoke hash-frozen JSON Tools without importing them into the robot process."""

    def __init__(
        self,
        controller_root: str | Path,
        bindings: Mapping[str, Mapping[str, Any]] | None,
        *,
        python: str | Path | None = None,
        timeout_sec: float = 2.0,
    ) -> None:
        self.controller_root = Path(controller_root).resolve()
        self.bindings = dict(bindings or {})
        self.python = str(python or sys.executable)
        self.timeout_sec = float(timeout_sec)
        if not 0.1 <= self.timeout_sec <= 10.0:
            raise ValueError("timeout_sec must be within [0.1, 10.0]")
        unknown = sorted(set(self.bindings) - set(SUPPORTED_HOOKS))
        if unknown:
            raise RuntimeCapabilityError(f"unsupported capability hooks: {unknown}")

    def _module(self, hook: str) -> tuple[Path, Mapping[str, Any]]:
        binding = self.bindings.get(hook)
        if not isinstance(binding, Mapping):
            raise RuntimeCapabilityError(f"no frozen capability bound for {hook}")
        relative = Path(str(binding.get("module", "")))
        module = (self.controller_root / relative).resolve()
        if self.controller_root not in module.parents or not module.is_file():
            raise RuntimeCapabilityError("frozen capability module is missing or escapes controller root")
        expected = str(binding.get("sha256", ""))
        actual = hashlib.sha256(module.read_bytes()).hexdigest()
        if not expected or actual != expected:
            raise RuntimeCapabilityError("frozen capability source hash changed")
        return module, binding

    def invoke(self, hook: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Return a structured audit record; invalid Tools never affect control."""
        binding = self.bindings.get(hook)
        tool_id = binding.get("tool_id") if isinstance(binding, Mapping) else None
        try:
            if hook not in _VALIDATORS:
                raise RuntimeCapabilityError(f"unsupported capability hook: {hook}")
            module, binding = self._module(hook)
            runner = (
                "import importlib.util,json,sys;"
                "s=importlib.util.spec_from_file_location('cap',sys.argv[1]);"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "v=m.run(json.loads(sys.stdin.read()));print(json.dumps(v))"
            )
            completed = subprocess.run(
                [self.python, "-I", "-c", runner, str(module)],
                input=json.dumps(dict(payload)),
                text=True,
                capture_output=True,
                timeout=self.timeout_sec,
            )
            if completed.returncode != 0:
                raise RuntimeCapabilityError(
                    completed.stderr[-500:] or "capability subprocess failed"
                )
            try:
                raw = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeCapabilityError("capability returned invalid JSON") from exc
            output = validate_hook_output(hook, raw, payload)
            return {
                "hook": hook,
                "tool_id": binding.get("tool_id"),
                "applied": True,
                "output": output,
            }
        except (RuntimeCapabilityError, subprocess.TimeoutExpired) as exc:
            return {
                "hook": hook,
                "tool_id": tool_id,
                "applied": False,
                "error": f"{type(exc).__name__}: {exc}",
            }


__all__ = [
    "FrozenCapabilityRuntime",
    "HOOK_CONTRACTS",
    "RuntimeCapabilityError",
    "SUPPORTED_HOOKS",
    "grasp_execution_waypoints",
    "transport_waypoints",
    "validate_hook_output",
]
