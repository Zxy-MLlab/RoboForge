"""Generic public embodied-state and geometry primitives.

This module deliberately contains no Adapter or benchmark imports.  It turns
already-public poses, entities, and action receipts into explicit-frame facts;
it never assigns semantic meaning to a direction or infers a diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence


Number = int | float
Vector = tuple[float, ...]
Matrix4 = tuple[tuple[float, ...], ...]


def _vector(value: Sequence[Number], size: int = 3) -> Vector:
    result = tuple(float(item) for item in value)
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise ValueError(f"expected a finite vector of length {size}")
    return result


def _matrix4(value: Sequence[Sequence[Number]]) -> Matrix4:
    result = tuple(tuple(float(item) for item in row) for row in value)
    if len(result) != 4 or any(len(row) != 4 for row in result):
        raise ValueError("expected a 4x4 transform")
    if not all(math.isfinite(item) for row in result for item in row):
        raise ValueError("transform must be finite")
    return result


def _identity() -> Matrix4:
    return ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _matmul(a: Matrix4, b: Matrix4) -> Matrix4:
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(4))
                       for j in range(4)) for i in range(4))


def _mat_inverse(transform: Matrix4) -> Matrix4:
    # Rigid transforms are the public contract: inverse(R,t) = (R^T,-R^T t).
    rotation = tuple(tuple(transform[i][j] for i in range(3)) for j in range(3))
    translation = tuple(transform[i][3] for i in range(3))
    offset = tuple(-sum(rotation[i][j] * translation[j] for j in range(3))
                   for i in range(3))
    return (rotation[0] + (offset[0],), rotation[1] + (offset[1],),
            rotation[2] + (offset[2],), (0.0, 0.0, 0.0, 1.0))


def transform_point(point: Sequence[Number], transform: Sequence[Sequence[Number]]) -> list[float]:
    """Transform a point with an explicit homogeneous 4x4 transform."""
    vector = _vector(point)
    matrix = _matrix4(transform)
    return [sum(matrix[i][j] * vector[j] for j in range(3)) + matrix[i][3]
            for i in range(3)]


def _quaternion(value: Sequence[Number]) -> tuple[float, float, float, float]:
    return tuple(_vector(value, 4))  # type: ignore[return-value]


def _quat_mul(a: Sequence[Number], b: Sequence[Number]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = _quaternion(a); bx, by, bz, bw = _quaternion(b)
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _quat_conjugate(value: Sequence[Number]) -> tuple[float, float, float, float]:
    x, y, z, w = _quaternion(value)
    return (-x, -y, -z, w)


def _quat_rotate(quaternion: Sequence[Number], vector: Sequence[Number]) -> list[float]:
    q = _quaternion(quaternion)
    v = _quaternion((_vector(vector) + (0.0,)))
    rotated = _quat_mul(_quat_mul(q, v), _quat_conjugate(q))
    return list(rotated[:3])


def _matrix_quaternion(matrix: Matrix4) -> tuple[float, float, float, float]:
    # Stable enough for public diagnostics; no robotics dependency is needed.
    m = matrix
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        return ((m[2][1] - m[1][2]) / scale, (m[0][2] - m[2][0]) / scale,
                (m[1][0] - m[0][1]) / scale, 0.25 * scale)
    index = max(range(3), key=lambda i: m[i][i])
    if index == 0:
        scale = math.sqrt(max(0.0, 1 + m[0][0] - m[1][1] - m[2][2])) * 2
        if scale == 0:
            return (0.0, 0.0, 0.0, 1.0)
        return (0.25 * scale, (m[0][1] + m[1][0]) / scale,
                (m[0][2] + m[2][0]) / scale, (m[2][1] - m[1][2]) / scale)
    if index == 1:
        scale = math.sqrt(max(0.0, 1 - m[0][0] + m[1][1] - m[2][2])) * 2
        if scale == 0:
            return (0.0, 0.0, 0.0, 1.0)
        return ((m[0][1] + m[1][0]) / scale, 0.25 * scale,
                (m[1][2] + m[2][1]) / scale, (m[0][2] - m[2][0]) / scale)
    scale = math.sqrt(max(0.0, 1 - m[0][0] - m[1][1] + m[2][2])) * 2
    if scale == 0:
        return (0.0, 0.0, 0.0, 1.0)
    return ((m[0][2] + m[2][0]) / scale, (m[1][2] + m[2][1]) / scale,
            0.25 * scale, (m[1][0] - m[0][1]) / scale)


@dataclass(frozen=True)
class Frame:
    name: str
    parent: str | None = None
    transform_to_parent: Matrix4 = field(default_factory=_identity)

    def __post_init__(self):
        if not self.name:
            raise ValueError("frame name is required")
        object.__setattr__(self, "transform_to_parent", _matrix4(self.transform_to_parent))

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "parent": self.parent,
                "transform_to_parent": [list(row) for row in self.transform_to_parent]}


@dataclass(frozen=True)
class Pose:
    frame: str
    position: Vector
    orientation: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        if not self.frame:
            raise ValueError("pose frame is required")
        object.__setattr__(self, "position", _vector(self.position))
        if self.orientation is not None:
            object.__setattr__(self, "orientation", _quaternion(self.orientation))

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"frame": self.frame, "position": list(self.position)}
        if self.orientation is not None:
            result["orientation_xyzw"] = list(self.orientation)
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, default_frame: str = "unknown") -> "Pose":
        position = value.get("position", value.get("xyz", value.get("world_xyz")))
        if position is None:
            raise ValueError("pose position is required")
        orientation = value.get("orientation_xyzw", value.get("quaternion_xyzw"))
        return cls(str(value.get("frame") or default_frame), position, orientation)


def transform_pose(pose: Pose | Mapping[str, Any], transform: Sequence[Sequence[Number]],
                  *, target_frame: str | None = None) -> Pose:
    """Transform a pose and retain an explicit target frame identity."""
    source = pose if isinstance(pose, Pose) else Pose.from_mapping(pose)
    matrix = _matrix4(transform)
    orientation = source.orientation
    if orientation is not None:
        orientation = _quat_mul(_matrix_quaternion(matrix), orientation)
    return Pose(target_frame or source.frame, transform_point(source.position, matrix), orientation)


def relative_pose(parent: Pose | Mapping[str, Any], child: Pose | Mapping[str, Any]) -> Pose:
    """Return child pose expressed in the parent's frame."""
    first = parent if isinstance(parent, Pose) else Pose.from_mapping(parent)
    second = child if isinstance(child, Pose) else Pose.from_mapping(child)
    if first.frame != second.frame:
        raise ValueError("relative poses require a common source frame")
    translation = [second.position[i] - first.position[i] for i in range(3)]
    orientation = None
    if first.orientation is not None and second.orientation is not None:
        translation = _quat_rotate(_quat_conjugate(first.orientation), translation)
        orientation = _quat_mul(_quat_conjugate(first.orientation), second.orientation)
    return Pose(first.frame, translation, orientation)


def pose_delta(requested: Pose | Mapping[str, Any], achieved: Pose | Mapping[str, Any]) -> dict[str, Any]:
    """Calculate signed tracking error without assigning a semantic direction."""
    target = requested if isinstance(requested, Pose) else Pose.from_mapping(requested)
    actual = achieved if isinstance(achieved, Pose) else Pose.from_mapping(achieved, default_frame=target.frame)
    if target.frame != actual.frame:
        raise ValueError("pose delta requires matching frames")
    delta = [actual.position[i] - target.position[i] for i in range(3)]
    result: dict[str, Any] = {"frame": target.frame, "signed_error": {
        "dx": delta[0], "dy": delta[1], "dz": delta[2]},
        "norm_m": math.sqrt(sum(item * item for item in delta))}
    if target.orientation is not None and actual.orientation is not None:
        dot = min(1.0, abs(sum(a * b for a, b in zip(target.orientation, actual.orientation))))
        result["orientation_error_rad"] = 2.0 * math.acos(dot)
    return result


def action_frame_error(requested: Sequence[Number], achieved: Sequence[Number],
                       approach_axis: Sequence[Number]) -> dict[str, Any]:
    """Decompose a position error in an explicitly supplied action frame."""
    error = [float(a) - float(r) for r, a in zip(_vector(requested), _vector(achieved))]
    axis = _vector(approach_axis)
    norm = math.sqrt(sum(item * item for item in axis))
    if norm == 0:
        raise ValueError("approach axis must be non-zero")
    unit = [item / norm for item in axis]
    along = sum(error[i] * unit[i] for i in range(3))
    lateral_vector = [error[i] - along * unit[i] for i in range(3)]
    return {"along_approach_axis_error_m": along,
            "lateral_error_m": math.sqrt(sum(item * item for item in lateral_vector)),
            "lateral_error_vector_m": lateral_vector,
            "approach_axis": list(unit)}


@dataclass(frozen=True)
class Entity:
    entity_id: str
    label: str | None = None
    confidence: float | None = None
    geometry: Mapping[str, Any] = field(default_factory=dict)
    perception: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"entity_id": self.entity_id, "label": self.label,
                "confidence": self.confidence, "geometry": dict(self.geometry),
                "perception": dict(self.perception), "uncertainty": dict(self.uncertainty),
                "provenance": dict(self.provenance)}


def normalize_entity(value: Mapping[str, Any], *, entity_id: str | None = None,
                     provenance: Mapping[str, Any] | None = None) -> Entity:
    """Normalize a public detection into a task-agnostic Entity record."""
    geometry = {key: value[key] for key in ("frame", "center", "orientation", "size", "bounds")
                if key in value}
    if "frame" not in geometry:
        geometry["frame"] = str(value.get("coordinate_frame") or "unknown")
    if "center" not in geometry:
        center = value.get("world_xyz") or value.get("xyz")
        if center is not None:
            geometry["center"] = list(center)
    perception = {key: value[key] for key in ("rgb_ref", "rgb_path", "depth_ref", "depth_path",
                                               "mask_ref", "mask_path", "bbox", "box_xyxy") if key in value}
    return Entity(str(entity_id or value.get("entity_id") or value.get("point_ref") or "entity"),
                  value.get("label") or value.get("query"), value.get("confidence", value.get("score")),
                  geometry, perception,
                  dict(value.get("uncertainty") or {}),
                  dict(provenance or value.get("provenance") or {}))


@dataclass(frozen=True)
class RobotState:
    eef_pose: Pose | None = None
    gripper_state: Any = None
    gripper_width: float | None = None
    proprioception: Mapping[str, Any] = field(default_factory=dict)
    observations: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result = {"eef_pose": self.eef_pose.as_dict() if self.eef_pose else None,
                  "gripper_state": self.gripper_state, "gripper_width": self.gripper_width,
                  "proprioception": dict(self.proprioception), "observations": dict(self.observations)}
        return result


@dataclass(frozen=True)
class InteractionState:
    entity_id: str | None = None
    relative_pose: Pose | None = None
    entity_displacement: Vector | None = None
    eef_displacement: Vector | None = None
    gripper_before: Any = None
    gripper_after: Any = None
    contact: Any = None
    attachment: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {"entity_id": self.entity_id,
                "relative_pose": self.relative_pose.as_dict() if self.relative_pose else None,
                "entity_displacement": list(self.entity_displacement) if self.entity_displacement else None,
                "eef_displacement": list(self.eef_displacement) if self.eef_displacement else None,
                "gripper_before": self.gripper_before, "gripper_after": self.gripper_after,
                "contact": self.contact, "attachment": self.attachment}


@dataclass(frozen=True)
class EmbodiedState:
    frames: Mapping[str, Frame] = field(default_factory=dict)
    robot: RobotState = field(default_factory=RobotState)
    entities: tuple[Entity, ...] = ()
    interactions: tuple[InteractionState, ...] = ()
    observations: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"frames": {key: frame.as_dict() for key, frame in self.frames.items()},
                "robot": self.robot.as_dict(), "entities": [item.as_dict() for item in self.entities],
                "interactions": [item.as_dict() for item in self.interactions],
                "observations": dict(self.observations)}


@dataclass(frozen=True)
class EmbodiedTransition:
    before: EmbodiedState | Mapping[str, Any] | None
    requested_action: Mapping[str, Any]
    achieved_action: Mapping[str, Any] | None
    after: EmbodiedState | Mapping[str, Any] | None
    delta: Mapping[str, Any] = field(default_factory=dict)
    verification: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        def encode(value):
            return value.as_dict() if hasattr(value, "as_dict") else value
        return {"before": encode(self.before), "action": {"requested": dict(self.requested_action)},
                "achieved": dict(self.achieved_action or {}), "after": encode(self.after),
                "delta": dict(self.delta), "verification": self.verification}


def build_transition(*, before: EmbodiedState | Mapping[str, Any] | None,
                     requested_action: Mapping[str, Any],
                     achieved_action: Mapping[str, Any] | None,
                     after: EmbodiedState | Mapping[str, Any] | None,
                     verification: Mapping[str, Any] | None = None) -> EmbodiedTransition:
    """Construct a transition from public state/action receipts."""
    delta: dict[str, Any] = {}
    if isinstance(achieved_action, Mapping):
        target = achieved_action.get("target_xyz")
        actual = achieved_action.get("eef_after")
        if target is not None and actual is not None:
            frame = str(achieved_action.get("target_frame")
                        or requested_action.get("frame") or requested_action.get("target_frame")
                        or "unknown")
            try:
                delta["robot_motion"] = pose_delta(
                    Pose(frame, target, achieved_action.get("target_quaternion_xyzw")),
                    Pose(frame, actual, achieved_action.get("eef_quaternion_xyzw")))
            except (TypeError, ValueError):
                pass
            axis = achieved_action.get("approach_axis") or requested_action.get("approach_axis")
            if axis is not None:
                try:
                    delta["action_frame"] = action_frame_error(target, actual, axis)
                except (TypeError, ValueError):
                    pass
        if achieved_action.get("eef_before") is not None and actual is not None:
            try:
                delta["eef_displacement"] = [float(actual[i]) - float(achieved_action["eef_before"][i])
                                               for i in range(3)]
            except (TypeError, ValueError, IndexError):
                pass
    return EmbodiedTransition(before, dict(requested_action),
                              dict(achieved_action or {}), after, delta, verification)


__all__ = ["Frame", "Pose", "Entity", "RobotState", "InteractionState", "EmbodiedState",
           "EmbodiedTransition", "transform_point", "transform_pose", "relative_pose",
           "pose_delta", "action_frame_error", "normalize_entity", "build_transition"]
