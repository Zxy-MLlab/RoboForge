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
    if result[3] != (0.0, 0.0, 0.0, 1.0):
        raise ValueError("transform must be homogeneous")
    rotation = tuple(tuple(result[i][j] for j in range(3)) for i in range(3))
    transpose_product = tuple(tuple(sum(rotation[k][i] * rotation[k][j]
                                         for k in range(3)) for j in range(3))
                               for i in range(3))
    if any(abs(transpose_product[i][j] - (1.0 if i == j else 0.0)) > 2e-5
           for i in range(3) for j in range(3)):
        raise ValueError("transform rotation must be orthonormal")
    determinant = (rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
                   - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
                   + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]))
    if abs(determinant - 1.0) > 2e-5:
        raise ValueError("transform rotation must be right-handed")
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
    raw = _vector(value, 4)
    norm = math.sqrt(sum(item * item for item in raw))
    if norm < 1e-12:
        raise ValueError("quaternion must be non-zero")
    return tuple(item / norm for item in raw)  # type: ignore[return-value]


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
        position = value.get("position")
        if position is None:
            raise ValueError("pose position is required")
        orientation = value.get("orientation_xyzw")
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


def frame_transform(frames: Mapping[str, Frame], source_frame: str,
                    target_frame: str) -> Matrix4:
    """Compose the rigid transform that expresses source coordinates in target.

    ``Frame.transform_to_parent`` maps coordinates from a frame into its parent.
    A missing frame, cycle, or disconnected graph fails closed instead of
    silently assuming that a frame is world-aligned.
    """
    source_frame, target_frame = str(source_frame), str(target_frame)
    if source_frame not in frames or target_frame not in frames:
        raise ValueError("source and target frames must be declared")

    def to_root(name: str) -> tuple[str, Matrix4]:
        current, transform = name, _identity()
        visited: set[str] = set()
        while True:
            if current in visited:
                raise ValueError("frame graph contains a cycle")
            visited.add(current)
            frame = frames.get(current)
            if frame is None:
                raise ValueError(f"unknown frame: {current}")
            if frame.parent is None:
                return current, transform
            transform = _matmul(frame.transform_to_parent, transform)
            current = frame.parent

    source_root, source_to_root = to_root(source_frame)
    target_root, target_to_root = to_root(target_frame)
    if source_root != target_root:
        raise ValueError("source and target frames are disconnected")
    return _matmul(_mat_inverse(target_to_root), source_to_root)


def relative_pose(parent: Pose | Mapping[str, Any], child: Pose | Mapping[str, Any], *,
                  result_frame: str | None = None) -> Pose:
    """Return child pose expressed in the parent's local coordinate frame.

    The two input poses describe frames in a common source frame.  Since the
    returned numbers are parent-local, callers may provide the declared name
    of that local frame; otherwise a deterministic derived frame label is used
    instead of incorrectly reusing the source-frame label.
    """
    first = parent if isinstance(parent, Pose) else Pose.from_mapping(parent)
    second = child if isinstance(child, Pose) else Pose.from_mapping(child)
    if first.frame != second.frame:
        raise ValueError("relative poses require a common source frame")
    translation = [second.position[i] - first.position[i] for i in range(3)]
    orientation = None
    if first.orientation is not None:
        translation = _quat_rotate(_quat_conjugate(first.orientation), translation)
        if second.orientation is not None:
            orientation = _quat_mul(_quat_conjugate(first.orientation), second.orientation)
        return Pose(result_frame or f"relative_to:{first.frame}", translation, orientation)
    # Without a parent orientation no parent-local transform was performed;
    # retain the common source-frame label instead of claiming local numbers.
    return Pose(result_frame or first.frame, translation, orientation)


def relative_pose_in_frames(parent: Pose | Mapping[str, Any], child: Pose | Mapping[str, Any],
                            frames: Mapping[str, Frame], *, result_frame: str | None = None) -> Pose:
    """Compute a relative pose after transforming both poses to one frame."""
    first = parent if isinstance(parent, Pose) else Pose.from_mapping(parent)
    second = child if isinstance(child, Pose) else Pose.from_mapping(child)
    common = result_frame or first.frame
    first_common = transform_pose(first, frame_transform(frames, first.frame, common),
                                  target_frame=common)
    second_common = transform_pose(second, frame_transform(frames, second.frame, common),
                                   target_frame=common)
    return relative_pose(first_common, second_common, result_frame=result_frame)


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
                       approach_axis: Sequence[Number], *, axis_frame: str = "unknown") -> dict[str, Any]:
    """Decompose a position error in an explicitly supplied action frame."""
    error = [float(a) - float(r) for r, a in zip(_vector(requested), _vector(achieved))]
    axis = _vector(approach_axis)
    norm = math.sqrt(sum(item * item for item in axis))
    if norm == 0:
        raise ValueError("approach axis must be non-zero")
    unit = [item / norm for item in axis]
    along = sum(error[i] * unit[i] for i in range(3))
    lateral_vector = [error[i] - along * unit[i] for i in range(3)]
    return {"along_action_axis_error_m": along,
            "along_approach_axis_error_m": along,
            "lateral_error_m": math.sqrt(sum(item * item for item in lateral_vector)),
            "lateral_error_vector_m": lateral_vector,
            "approach_axis": list(unit), "approach_axis_frame": str(axis_frame)}


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
    """Validate an Adapter-normalized public Entity record.

    Native detector schemas are deliberately rejected here.  Adapters and
    capabilities own translation into this explicit contract.
    """
    if not isinstance(value, Mapping) or not value.get("entity_id"):
        raise ValueError("canonical entity_id is required")
    geometry = value.get("geometry")
    if not isinstance(geometry, Mapping) or not geometry.get("frame"):
        raise ValueError("canonical entity geometry.frame is required")
    return Entity(str(entity_id or value["entity_id"]),
                  value.get("label"), value.get("confidence"), dict(geometry),
                  dict(value.get("perception") or {}),
                  dict(value.get("uncertainty") or {}),
                  dict(provenance or value.get("provenance") or {}))


def normalize_robot_state(value: Mapping[str, Any], *, eef_frame: str = "unknown") -> RobotState:
    """Validate an Adapter-normalized, physically observable RobotState."""
    canonical = value.get("robot") if isinstance(value.get("robot"), Mapping) else value
    pose_value = canonical.get("eef_pose") if isinstance(canonical, Mapping) else None
    pose = None
    if isinstance(pose_value, Pose):
        pose = pose_value
    elif isinstance(pose_value, Mapping):
        position = pose_value.get("position_m")
        orientation = pose_value.get("orientation_xyzw")
        if position is not None:
            pose = Pose(str(pose_value.get("frame") or eef_frame), position, orientation)
    gripper = canonical.get("gripper") if isinstance(canonical, Mapping) else None
    if not isinstance(gripper, Mapping):
        gripper = {}
    width = gripper.get("width_m")
    if width is not None:
        width = float(width)
        if not math.isfinite(width) or width < 0:
            raise ValueError("canonical gripper width_m must be finite and non-negative")
    joint_state = canonical.get("joint_state", {})
    if not isinstance(joint_state, Mapping):
        joint_state = {}
    proprioception = canonical.get("proprioception", {})
    if not isinstance(proprioception, Mapping):
        proprioception = {}
    observations = {key: value[key] for key in ("frame_id", "step") if key in value}
    return RobotState(eef_pose=pose, gripper_state=gripper.get("state"),
                      gripper_width=width, joint_state=dict(joint_state),
                      proprioception=dict(proprioception), observations=observations)


def normalize_embodied_state(observation: Mapping[str, Any], *,
                             entities: Sequence[Entity | Mapping[str, Any]] = (),
                             eef_frame: str = "unknown") -> EmbodiedState:
    """Build a generic state from an Adapter's canonical public observation."""
    declared_frames = observation.get("frames") if isinstance(observation, Mapping) else None
    if isinstance(declared_frames, Mapping):
        frames = {str(name): (frame if isinstance(frame, Frame)
                              else Frame(str(name), frame.get("parent") if isinstance(frame, Mapping) else None,
                                         frame.get("transform_to_parent", _identity())
                                         if isinstance(frame, Mapping) else _identity()))
                  for name, frame in declared_frames.items()}
    else:
        frames = {eef_frame: Frame(eef_frame)}
    normalized_entities = tuple(item if isinstance(item, Entity)
                                else normalize_entity(item) for item in entities)
    return EmbodiedState(frames=frames, robot=normalize_robot_state(observation, eef_frame=eef_frame),
                         entities=normalized_entities,
                         observations={key: observation[key] for key in ("frame_id", "step")
                                       if key in observation})


def embodied_state_from_mapping(value: Any) -> EmbodiedState | None:
    """Decode only the canonical state representation emitted by an Adapter."""
    if not isinstance(value, Mapping):
        return None
    try:
        raw_entities = value.get("entities") or ()
        if not isinstance(raw_entities, (list, tuple)):
            return None
        return normalize_embodied_state(value, entities=raw_entities,
                                        eef_frame=str(value.get("eef_frame") or "unknown"))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class RobotState:
    eef_pose: Pose | None = None
    gripper_state: Any = None
    gripper_width: float | None = None
    joint_state: Mapping[str, Any] = field(default_factory=dict)
    proprioception: Mapping[str, Any] = field(default_factory=dict)
    observations: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result = {"eef_pose": self.eef_pose.as_dict() if self.eef_pose else None,
                  "gripper_state": self.gripper_state, "gripper_width": self.gripper_width,
                  "joint_state": dict(self.joint_state),
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
    before_state = before if isinstance(before, EmbodiedState) else embodied_state_from_mapping(before)
    after_state = after if isinstance(after, EmbodiedState) else embodied_state_from_mapping(after)
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
            axis = (achieved_action.get("action_frame_axis")
                    or achieved_action.get("approach_axis")
                    or requested_action.get("action_frame_axis")
                    or requested_action.get("approach_axis"))
            if isinstance(axis, Mapping):
                axis = axis.get("axis")
            if axis is not None:
                try:
                    delta["action_frame"] = action_frame_error(
                        target, actual, axis,
                        axis_frame=str(achieved_action.get("action_frame_axis_frame")
                                       or requested_action.get("action_frame_axis_frame")
                                       or achieved_action.get("target_frame")
                                       or requested_action.get("frame") or "unknown"))
                except (TypeError, ValueError):
                    pass
        if achieved_action.get("eef_before") is not None and actual is not None:
            try:
                delta["eef_displacement"] = [float(actual[i]) - float(achieved_action["eef_before"][i])
                                               for i in range(3)]
            except (TypeError, ValueError, IndexError):
                pass
    if before_state is not None and after_state is not None:
        before_pose = before_state.robot.eef_pose
        after_pose = after_state.robot.eef_pose
        if before_pose is not None and after_pose is not None and before_pose.frame == after_pose.frame:
            delta.setdefault("eef_displacement", [after_pose.position[i] - before_pose.position[i]
                                                    for i in range(3)])
        before_entities = {item.entity_id: item for item in before_state.entities}
        after_entities = {item.entity_id: item for item in after_state.entities}
        displacements = {}
        for entity_id in sorted(before_entities.keys() & after_entities.keys()):
            first = before_entities[entity_id].geometry.get("center")
            second = after_entities[entity_id].geometry.get("center")
            if isinstance(first, Sequence) and isinstance(second, Sequence):
                try:
                    displacements[entity_id] = [float(second[i]) - float(first[i]) for i in range(3)]
                except (TypeError, ValueError, IndexError):
                    pass
        if displacements:
            delta["entity_displacement"] = displacements
        if before_state.robot.gripper_width is not None or after_state.robot.gripper_width is not None:
            delta["gripper_width"] = {"before": before_state.robot.gripper_width,
                                       "after": after_state.robot.gripper_width}
    return EmbodiedTransition(before, dict(requested_action),
                              dict(achieved_action or {}), after, delta, verification)


__all__ = ["Frame", "Pose", "Entity", "RobotState", "InteractionState", "EmbodiedState",
           "EmbodiedTransition", "transform_point", "transform_pose", "relative_pose",
           "relative_pose_in_frames", "frame_transform", "pose_delta", "action_frame_error",
           "normalize_entity", "normalize_robot_state", "embodied_state_from_mapping",
           "normalize_embodied_state", "build_transition"]
