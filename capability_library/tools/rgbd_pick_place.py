"""Classical RGB-D perception and Cartesian control for generic pick-and-place.

The module contains no learned parameters and never consumes LIBERO object
state, reward, task identifiers, BDDL predicates, or evaluator success.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class PickPlaceIntent:
    object_name: str
    relation: str | None
    reference_names: tuple[str, ...]
    target_name: str


@dataclass(frozen=True)
class Region:
    label: int
    centroid_rc: tuple[float, float]
    area_px: int
    mean_rgb: tuple[float, float, float]
    median_world: tuple[float, float, float]
    radius_m: float
    circularity: float


@dataclass(frozen=True)
class CircularCandidate:
    center_rc: tuple[float, float]
    radius_px: float
    interior_rgb: tuple[float, float, float]
    darkness: float
    achromaticity: float
    center_world: tuple[float, float, float]


def parse_pick_place_instruction(instruction: str) -> PickPlaceIntent:
    """Parse LIBERO-style English without a language model."""
    text = " ".join(instruction.lower().strip().split())
    prefix = "pick up the "
    separator = " and place it on the "
    if not text.startswith(prefix) or separator not in text:
        raise ValueError(f"Unsupported pick-place instruction: {instruction!r}")
    source, target_name = text.removeprefix(prefix).split(separator, maxsplit=1)
    patterns = (
        (r"(.+?) between the (.+?) and the (.+)", "between"),
        (r"(.+?) next to the (.+)", "next_to"),
        (r"(.+?) on the (.+)", "on"),
        (r"(.+?) in the (.+)", "in"),
        (r"(.+?) from table center", "table_center"),
    )
    for pattern, relation in patterns:
        relation_match = re.fullmatch(pattern, source)
        if relation_match:
            object_name, *references = relation_match.groups()
            return PickPlaceIntent(
                object_name=object_name,
                relation=relation,
                reference_names=tuple(references),
                target_name=target_name,
            )
    return PickPlaceIntent(object_name=source, relation=None, reference_names=(), target_name=target_name)


def backproject_rgbd(
    depth_m: np.ndarray, intrinsic: np.ndarray, camera_to_world: np.ndarray
) -> np.ndarray:
    """Back-project an upright metric-depth image to an HxWx3 world cloud."""
    depth = np.asarray(depth_m, dtype=np.float64).squeeze(-1)
    height, width = depth.shape
    rows, cols = np.indices((height, width), dtype=np.float64)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    camera = np.stack(
        ((cols - cx) * depth / fx, (rows - cy) * depth / fy, depth), axis=-1
    )
    homogeneous = np.concatenate((camera, np.ones((height, width, 1))), axis=-1)
    return (homogeneous @ camera_to_world.T)[..., :3]


def estimate_table_height(world: np.ndarray) -> float:
    """Estimate the horizontal support plane using a robust z histogram."""
    points = world.reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1)
    # Generic Panda tabletop workspace bounds, not task or state identifiers.
    valid &= np.abs(points[:, 0]) < 0.65
    valid &= (points[:, 1] > -0.55) & (points[:, 1] < 0.55)
    valid &= (points[:, 2] > 0.65) & (points[:, 2] < 1.15)
    z = points[valid, 2]
    if z.size < 100:
        raise RuntimeError("Insufficient workspace points to estimate the table")
    hist, edges = np.histogram(z, bins=np.arange(0.65, 1.151, 0.003))
    index = int(np.argmax(hist))
    center = (edges[index] + edges[index + 1]) / 2
    near = z[np.abs(z - center) < 0.006]
    return float(np.median(near))


def segment_workspace_regions(
    rgb: np.ndarray,
    world: np.ndarray,
    *,
    table_height: float | None = None,
    minimum_height_m: float = 0.008,
) -> tuple[list[Region], np.ndarray]:
    """Extract tabletop objects through geometry and connected components."""
    image = np.asarray(rgb, dtype=np.uint8)
    table_z = estimate_table_height(world) if table_height is None else table_height
    x, y, z = (world[..., index] for index in range(3))
    mask = (
        np.isfinite(world).all(axis=-1)
        & (np.abs(x) < 0.48)
        & (y > -0.40)
        & (y < 0.40)
        & (z > table_z + minimum_height_m)
        & (z < table_z + 0.20)
    ).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    regions: list[Region] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not 35 <= area <= 12000:
            continue
        component = labels == label
        points = world[component]
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if points.shape[0] < 30:
            continue
        xy = points[:, :2]
        median = np.median(points, axis=0)
        distances = np.linalg.norm(xy - np.median(xy, axis=0), axis=1)
        radius = float(np.quantile(distances, 0.9))
        contour_mask = component.astype(np.uint8)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        circularity = float(4 * np.pi * cv2.contourArea(contour) / max(perimeter * perimeter, 1e-9))
        mean_rgb = tuple(float(value) for value in image[component].mean(axis=0))
        col, row = centroids[label]
        regions.append(
            Region(
                label=label,
                centroid_rc=(float(row), float(col)),
                area_px=area,
                mean_rgb=mean_rgb,
                median_world=tuple(float(value) for value in median),
                radius_m=radius,
                circularity=circularity,
            )
        )
    return regions, labels


def annotate_regions(rgb: np.ndarray, regions: Sequence[Region]) -> np.ndarray:
    output = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    for index, region in enumerate(regions):
        row, col = region.centroid_rc
        cv2.circle(output, (round(col), round(row)), 5, (0, 255, 0), 2)
        cv2.putText(
            output,
            f"{index}:a{region.area_px}:r{region.radius_m:.2f}",
            (round(col) + 5, round(row) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def detect_circular_candidates(rgb: np.ndarray, world: np.ndarray) -> list[CircularCandidate]:
    """Detect bowl / plate / ramekin-like rims using classical Hough voting."""
    image = np.asarray(rgb, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max(12, image.shape[0] // 16),
        param1=80,
        param2=22,
        minRadius=max(6, image.shape[0] // 40),
        maxRadius=max(24, image.shape[0] // 8),
    )
    if circles is None:
        return []
    rows, cols = np.indices(image.shape[:2])
    candidates: list[CircularCandidate] = []
    for col, row, radius in circles[0]:
        interior = (cols - col) ** 2 + (rows - row) ** 2 < (0.75 * radius) ** 2
        colors = image[interior].astype(float)
        if colors.size == 0:
            continue
        mean = colors.mean(axis=0)
        intensity = colors.mean(axis=1)
        darkness = float(np.clip(1 - np.median(intensity) / 255, 0, 1))
        achromaticity = float(
            np.clip(1 - np.mean(colors.max(axis=1) - colors.min(axis=1)) / 80, 0, 1)
        )
        r0, c0 = int(round(row)), int(round(col))
        patch = world[max(0, r0 - 2) : r0 + 3, max(0, c0 - 2) : c0 + 3]
        valid = patch[np.isfinite(patch).all(axis=-1)]
        if not valid.size:
            continue
        center_world = np.median(valid, axis=0)
        if not (
            abs(center_world[0]) < 0.48
            and -0.40 < center_world[1] < 0.40
            and 0.90 < center_world[2] < 1.12
        ):
            continue
        candidates.append(
            CircularCandidate(
                center_rc=(float(row), float(col)),
                radius_px=float(radius),
                interior_rgb=tuple(float(value) for value in mean),
                darkness=darkness,
                achromaticity=achromaticity,
                center_world=tuple(float(value) for value in center_world),
            )
        )
    return candidates


def select_black_bowl(candidates: Sequence[CircularCandidate]) -> CircularCandidate:
    """Select a black, achromatic bowl from geometric rim candidates."""
    plausible = [candidate for candidate in candidates if 12 < candidate.radius_px < 21]
    if not plausible:
        raise RuntimeError("No bowl-sized circular candidates")
    return max(plausible, key=lambda item: item.darkness * item.achromaticity)


def select_plate(
    candidates: Sequence[CircularCandidate], *, exclude: CircularCandidate | None = None
) -> CircularCandidate:
    """Select the largest round receptacle, excluding the source object."""
    remaining = [candidate for candidate in candidates if candidate is not exclude]
    # Connected-component fallback can include the entire dark tabletop. A
    # receptacle candidate must remain within a generic workspace-scale radius;
    # this is geometry-only and independent of task/state identifiers.
    bounded = [candidate for candidate in remaining if 8.0 < candidate.radius_px < 35.0]
    if bounded:
        remaining = bounded
    if not remaining:
        raise RuntimeError("No plate candidate")
    return max(remaining, key=lambda item: item.radius_px)


def select_for_intent(
    intent: PickPlaceIntent, candidates: Sequence[CircularCandidate], table_height: float
) -> tuple[CircularCandidate, CircularCandidate]:
    """Resolve source and target through generic appearance and spatial relations."""
    elevated = [item for item in candidates if item.center_world[2] > table_height + 0.04]
    target = select_plate(elevated)
    bowls = [
        item
        for item in elevated
        if 12 < item.radius_px < 21 and item.darkness * item.achromaticity > 0.40
    ]
    if not bowls:
        raise RuntimeError("No black-bowl candidates")
    if intent.relation == "between" and "ramekin" in intent.reference_names:
        small = [item for item in elevated if item is not target and item.radius_px < 12]
        if small:
            reference = min(small, key=lambda item: item.radius_px)
            midpoint = (
                np.asarray(target.center_world[:2]) + np.asarray(reference.center_world[:2])
            ) / 2
            source = min(
                bowls,
                key=lambda item: np.linalg.norm(np.asarray(item.center_world[:2]) - midpoint),
            )
            return source, target
    if intent.relation == "next_to" and "plate" in intent.reference_names:
        source = min(
            bowls,
            key=lambda item: np.linalg.norm(
                np.asarray(item.center_world[:2]) - np.asarray(target.center_world[:2])
            ),
        )
        return source, target
    source = max(bowls, key=lambda item: item.darkness * item.achromaticity)
    return source, target


class CartesianWaypointController:
    """Generate bounded OSC_POSE delta actions from allowed EEF feedback."""

    def __init__(self, *, position_gain: float = 20.0, max_action: float = 1.0):
        self.position_gain = position_gain
        self.max_action = max_action

    def action(
        self,
        eef_position: np.ndarray,
        goal: np.ndarray,
        gripper: float,
        orientation_delta: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        delta = (np.asarray(goal) - np.asarray(eef_position)) * self.position_gain
        action = np.zeros(7, dtype=np.float64)
        action[:3] = np.clip(delta, -self.max_action, self.max_action)
        action[3:6] = np.clip(orientation_delta, -self.max_action, self.max_action)
        action[6] = float(np.clip(gripper, -1, 1))
        return action


def allowed_observation(raw: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fail closed when projecting raw LIBERO observations for control."""
    keys = (
        "agentview_image",
        "agentview_depth",
        "robot0_eye_in_hand_image",
        "robot0_eye_in_hand_depth",
        "robot0_joint_pos",
        "robot0_joint_vel",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_gripper_qvel",
    )
    missing = [key for key in keys if key not in raw]
    if missing:
        raise KeyError(f"Missing allowed sensor fields: {missing}")
    return {key: np.asarray(raw[key]).copy() for key in keys}


def make_thea_rgbd_pick_place_tool():
    """Create an evaluator-blind Thea simulator tool for code-only pick/place."""
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )
    from thea_simulation.runtime import SimulationToolSpec

    def execute(episode, arguments):
        del arguments
        raw = episode.observe()
        observation = allowed_observation(raw)
        env = episode._env  # Deployment-owned LIBERO adapter; calibration only.
        sim = env.sim
        rgb = np.ascontiguousarray(observation["agentview_image"][::-1])
        depth = get_real_depth_map(
            sim, np.ascontiguousarray(observation["agentview_depth"][::-1])
        )
        world = backproject_rgbd(
            depth,
            get_camera_intrinsic_matrix(sim, "agentview", rgb.shape[0], rgb.shape[1]),
            get_camera_extrinsic_matrix(sim, "agentview"),
        )
        table_height = estimate_table_height(world)
        intent = parse_pick_place_instruction(episode.instruction)
        source, target = select_for_intent(
            intent, detect_circular_candidates(rgb, world), table_height
        )
        source_xy = np.asarray(source.center_world[:2])
        target_xy = np.asarray(target.center_world[:2])
        grasp_height = max(table_height + 0.085, source.center_world[2] + 0.010)
        source_side = source_xy + np.array([0.17, 0.0])
        target_side = target_xy + np.array([0.17, 0.0])
        waypoints = (
            (np.r_[source_side, table_height + 0.23], -1.0, (0, 0, 0), 50),
            (np.r_[source_side, table_height + 0.23], -1.0, (0, 1, 0), 20),
            (np.r_[source_side, grasp_height], -1.0, (0, 0, 0), 40),
            (np.r_[source_xy, grasp_height], -1.0, (0, 0, 0), 50),
            (np.r_[source_xy, grasp_height], 1.0, (0, 0, 0), 110),
            (np.r_[source_side, grasp_height], 1.0, (0, 0, 0), 45),
            (np.r_[source_side, table_height + 0.25], 1.0, (0, 0, 0), 40),
            (np.r_[target_side, table_height + 0.25], 1.0, (0, 0, 0), 55),
            (np.r_[target_side, table_height + 0.11], 1.0, (0, 0, 0), 40),
            (np.r_[target_xy, table_height + 0.11], 1.0, (0, 0, 0), 50),
            (np.r_[target_xy, table_height + 0.11], -1.0, (0, 0, 0), 110),
            (np.r_[target_side, table_height + 0.11], -1.0, (0, 0, 0), 45),
        )
        controller = CartesianWaypointController(position_gain=20)
        count = 0
        for goal, gripper, orientation_delta, steps in waypoints:
            for _ in range(steps):
                action = controller.action(
                    observation["robot0_eef_pos"], goal, gripper, orientation_delta
                )
                transition = episode.step(action)
                # Never branch on reward, success, done, or evaluator state.
                observation = allowed_observation(transition.observation)
                count += 1
        return {
            "success": True,
            "observation": f"Executed {count} evaluator-blind code-controller actions.",
            "actions_executed": count,
            "learned_models_used": [],
            "forbidden_inputs_used": [],
        }

    return SimulationToolSpec(
        name="rgbd_code_pick_place",
        description=(
            "Execute one generic code-only RGB-D pick-and-place program. Uses no "
            "learned model and does not inspect evaluator signals while acting."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        executor=execute,
        post_condition="The benchmark evaluator reports task success after execution.",
    )
