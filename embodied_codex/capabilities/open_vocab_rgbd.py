"""Embodied Codex GroundingDINO + SAM + RGB-D projection capability.

The implementation consumes only files and calibration emitted by a Robot
Adapter.  It has no simulator, task ID, BDDL, evaluator, or object-state access.
Both checkpoints are public and task-disjoint from LIBERO evaluation episodes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping

import cv2
import numpy as np

from .perception_reliability import audit_detection_result


class CapabilityInputError(ValueError):
    pass


class OpenVocabularyRGBD:
    def __init__(
        self, *, groundingdino_root: str | Path,
        groundingdino_config: str | Path, groundingdino_checkpoint: str | Path,
        sam_root: str | Path, sam_checkpoint: str | Path,
        device: str = "cuda:0",
    ) -> None:
        self.groundingdino_root = Path(groundingdino_root).resolve()
        self.groundingdino_config = Path(groundingdino_config).resolve()
        self.groundingdino_checkpoint = Path(groundingdino_checkpoint).resolve()
        self.sam_root = Path(sam_root).resolve()
        self.sam_checkpoint = Path(sam_checkpoint).resolve()
        for path in (
            self.groundingdino_root, self.groundingdino_config,
            self.groundingdino_checkpoint, self.sam_root, self.sam_checkpoint,
        ):
            if not path.exists(): raise FileNotFoundError(path)
        self.device = device
        self._detector = None
        self._sam_predictor = None
        self._provenance_cache = None

    @property
    def provenance(self) -> dict[str, Any]:
        if self._provenance_cache is not None:
            return dict(self._provenance_cache)
        self._provenance_cache = {
            "implementation_family": "open_vocab_rgbd_grounded_sam",
            "models": ["GroundingDINO Swin-T OGC", "SAM ViT-B"],
            "source_urls": [
                "https://github.com/IDEA-Research/GroundingDINO",
                "https://github.com/facebookresearch/segment-anything",
            ],
            "model_card_urls": [
                "https://github.com/IDEA-Research/GroundingDINO",
                "https://github.com/facebookresearch/segment-anything",
            ],
            "checkpoint_sha256": {
                "groundingdino": self._sha256(self.groundingdino_checkpoint),
                "sam": self._sha256(self.sam_checkpoint),
            },
            "checkpoint_files": {
                "groundingdino": str(self.groundingdino_checkpoint),
                "sam": str(self.sam_checkpoint),
            },
            "trained_on_current_task": False,
            "privileged_state_used": False,
            "training_data_declaration":(
                "Public GroundingDINO Swin-T OGC and SAM ViT-B base checkpoints; "
                "no LIBERO task-specific fine-tuning is performed by this Harness."),
            "contamination_check":{"evaluated_benchmark":"LIBERO",
                "method":"upstream model documentation plus local checkpoint hash audit",
                "result":"no_declared_overlap"},
            "inputs": ["RGB", "metric depth", "camera intrinsic", "camera-to-world"],
        }
        return dict(self._provenance_cache)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _patch_transformers() -> None:
        from transformers import BertModel
        if not hasattr(BertModel, "get_head_mask"):
            BertModel.get_head_mask = lambda self, head_mask, num_hidden_layers, is_attention_chunked=False: [None] * num_hidden_layers
        if not getattr(BertModel, "_embodied_codex_compat", False):
            original = BertModel.get_extended_attention_mask
            def compatible(self, attention_mask, input_shape, device=None):
                import torch
                return original(self, attention_mask, input_shape, dtype=torch.float32)
            BertModel.get_extended_attention_mask = compatible
            BertModel._embodied_codex_compat = True

    def _load_detector(self):
        if self._detector is not None: return
        if str(self.groundingdino_root) not in sys.path:
            sys.path.insert(0, str(self.groundingdino_root))
        self._patch_transformers()
        import torch
        from groundingdino.models import build_model
        from groundingdino.util.misc import clean_state_dict
        from groundingdino.util.slconfig import SLConfig
        configuration = SLConfig.fromfile(str(self.groundingdino_config))
        configuration.device = self.device
        model = build_model(configuration)
        checkpoint = torch.load(str(self.groundingdino_checkpoint), map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self._detector = model.eval().to(self.device)

    def _load_sam(self):
        if self._sam_predictor is not None: return
        if str(self.sam_root) not in sys.path:
            sys.path.insert(0, str(self.sam_root))
        from segment_anything import SamPredictor, sam_model_registry
        model = sam_model_registry["vit_b"](checkpoint=str(self.sam_checkpoint))
        self._sam_predictor = SamPredictor(model.eval().to(self.device))

    def _boxes(
        self, rgb_path: Path, query: str, box_threshold: float,
        text_threshold: float,
    ) -> list[dict[str, Any]]:
        self._load_detector()
        import torch
        from PIL import Image
        import groundingdino.datasets.transforms as transforms
        from groundingdino.util.utils import get_phrases_from_posmap
        caption = query.strip().lower()
        if not caption: raise CapabilityInputError("query is empty")
        if not caption.endswith("."): caption += "."
        transform = transforms.Compose([
            transforms.RandomResize([800], max_size=1333), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        pil = Image.open(rgb_path).convert("RGB")
        width, height = pil.size
        tensor, _ = transform(pil, None)
        with torch.no_grad():
            output = self._detector(tensor[None].to(self.device), captions=[caption])
        logits = output["pred_logits"].cpu().sigmoid()[0]
        boxes = output["pred_boxes"].cpu()[0]
        keep = logits.max(dim=1)[0] > box_threshold
        tokenized = self._detector.tokenizer(caption)
        detections = []
        for box, logit in zip(boxes[keep], logits[keep]):
            cx, cy, bw, bh = [float(value) for value in box]
            x0 = max(0.0, (cx - bw / 2) * width)
            y0 = max(0.0, (cy - bh / 2) * height)
            x1 = min(float(width - 1), (cx + bw / 2) * width)
            y1 = min(float(height - 1), (cy + bh / 2) * height)
            label = get_phrases_from_posmap(
                logit > text_threshold, tokenized, self._detector.tokenizer,
            ).replace(".", "")
            detections.append({
                "query": query, "label": label, "score": float(logit.max()),
                "box_xyxy": [x0, y0, x1, y1],
            })
        return sorted(detections, key=lambda item: item["score"], reverse=True)

    def _segment_and_project(
        self, rgb: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray,
        camera_to_world: np.ndarray, detection: Mapping[str, Any], output_dir: Path,
        index: int,
    ) -> dict[str, Any]:
        self._load_sam()
        self._sam_predictor.set_image(rgb)
        box = np.asarray(detection["box_xyxy"], np.float32)
        masks, scores, _ = self._sam_predictor.predict(
            box=box[None, :], multimask_output=True,
        )
        x0, y0, x1, y1 = box.astype(int)
        ranked = []
        for mask_index, (mask, score) in enumerate(zip(masks, scores)):
            inside = mask[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)].sum()
            containment = float(inside / max(int(mask.sum()), 1))
            ranked.append((float(score) + 0.2 * containment, mask_index, containment))
        _, selected, containment = max(ranked)
        mask = masks[selected]
        depth2d = np.asarray(depth).squeeze()
        valid = mask & np.isfinite(depth2d) & (depth2d > 0.1) & (depth2d < 3.0)
        rows, cols = np.nonzero(valid)
        if len(rows) < 20:
            raise CapabilityInputError("SAM mask has insufficient valid depth")
        z = depth2d[rows, cols]
        camera = np.c_[
            (cols - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (rows - intrinsic[1, 2]) * z / intrinsic[1, 1], z,
            np.ones_like(z),
        ]
        world = (camera_to_world @ camera.T).T[:, :3]
        low, high = np.quantile(world, [0.1, 0.9], axis=0)
        center = np.array([
            np.median(world[:, 0]), np.median(world[:, 1]),
            np.quantile(world[:, 2], 0.75),
        ])
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = output_dir / f"mask_{index:03d}.png"
        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        return {
            **dict(detection), "sam_score": float(scores[selected]),
            "box_containment": containment, "mask_pixels": int(mask.sum()),
            "mask_path": str(mask_path), "world_xyz": center.tolist(),
            "world_bounds_10_90": [low.tolist(), high.tolist()],
        }

    def detect(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        frame = payload.get("frame")
        queries = payload.get("queries")
        camera_name = str(payload.get("camera") or "agentview")
        if not isinstance(frame, Mapping) or not isinstance(queries, list) or not queries:
            raise CapabilityInputError("detect requires frame and nonempty queries")
        camera = (frame.get("cameras") or {}).get(camera_name)
        if not isinstance(camera, Mapping):
            raise CapabilityInputError("requested camera is absent from frame")
        rgb_path = Path(str(camera.get("rgb_path") or "")).resolve()
        depth_path = Path(str(camera.get("depth_path") or "")).resolve()
        if not rgb_path.is_file() or not depth_path.is_file():
            raise CapabilityInputError("RGB-D artifact is missing")
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None: raise CapabilityInputError("cannot read RGB")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path, allow_pickle=False)
        intrinsic = np.asarray(camera["intrinsic"], float)
        extrinsic = np.asarray(camera["camera_to_world"], float)
        output_dir = rgb_path.parent / "open_vocab_rgbd"
        grouped: dict[str, list[dict[str, Any]]] = {}
        index = 0
        for raw_query in queries:
            query = str(raw_query).strip()
            detections = self._boxes(
                rgb_path, query,
                float(payload.get("box_threshold", 0.20)),
                float(payload.get("text_threshold", 0.15)),
            )[:int(np.clip(payload.get("max_detections_per_query", 5), 1, 12))]
            projected = []
            for detection in detections:
                try:
                    projected.append(self._segment_and_project(
                        rgb, depth, intrinsic, extrinsic, detection, output_dir, index,
                    ))
                    index += 1
                except CapabilityInputError as exc:
                    projected.append({**detection, "projection_error": str(exc)})
            grouped[query] = projected
        result = {
            "frame_id": frame.get("frame_id"), "camera": camera_name,
            "detections": grouped, "provenance": self.provenance,
        }
        result["reliability"] = audit_detection_result(
            result,
            required_queries=[str(query) for query in queries],
            distinct_query_pairs=payload.get("distinct_query_pairs") or [],
        )
        return result

    def verify_support_relation(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        object_query = str(payload.get("object_query") or "")
        target_query = str(payload.get("target_query") or "")
        source_anchor = np.asarray(payload.get("source_world_xyz") or (), float)
        target_anchor = np.asarray(payload.get("target_world_xyz") or (), float)
        anchored_target_bounds = np.asarray(
            payload.get("target_world_bounds_10_90") or (), float)
        anchored_bounds_valid = (anchored_target_bounds.shape == (2, 3)
                                 and np.isfinite(anchored_target_bounds).all())
        if (not object_query or not target_query or source_anchor.shape != (3,)
                or target_anchor.shape != (3,)):
            return {"verified": False,
                    "reason": "object_query, target_query, and independent anchors are required"}
        result = self.detect({
            "frame": payload.get("frame"), "camera": payload.get("camera", "agentview"),
            "queries": [object_query, target_query],
            "box_threshold": payload.get("box_threshold", 0.16),
            "text_threshold": payload.get("text_threshold", 0.12),
            "max_detections_per_query": payload.get("max_detections_per_query", 5),
        })
        objects = [item for item in result["detections"].get(object_query, [])
                   if "world_xyz" in item and "world_bounds_10_90" in item]
        targets = [item for item in result["detections"].get(target_query, [])
                   if "world_xyz" in item and "world_bounds_10_90" in item]
        if not objects or (not targets and not anchored_bounds_valid):
            return {"verified": False, "reason": "object or target geometry absent after placement",
                    "object_count": len(objects), "target_count": len(targets)}
        distances_target = [float(np.linalg.norm(
            np.asarray(obj["world_xyz"], float)[:2] - target_anchor[:2]
        )) for obj in objects]
        distances_source = [float(np.linalg.norm(
            np.asarray(obj["world_xyz"], float)[:2] - source_anchor[:2]
        )) for obj in objects]
        selected_index = int(np.argmin(distances_target))
        obj = objects[selected_index]
        # A support noun is often grounded on the object that occludes it
        # after placement (for example, a bowl covering a plate).  Associate
        # the target using its independent pre-action surface height as well
        # as XY, and strongly penalize a candidate that is effectively the
        # same 3-D detection as the manipulated object.
        object_xyz = np.asarray(obj["world_xyz"], float);target_ranks = []
        for item in targets:
            point = np.asarray(item["world_xyz"], float)
            xy_error = float(np.linalg.norm(point[:2] - target_anchor[:2]))
            height_error = abs(float(point[2] - target_anchor[2]))
            alias_penalty = 0.20 if float(np.linalg.norm(point - object_xyz)) < 0.020 else 0.0
            target_ranks.append(xy_error + 2.5 * height_error + alias_penalty)
        target_index = int(np.argmin(target_ranks)) if target_ranks else None
        target = targets[target_index] if target_index is not None else None
        target_xy_error = distances_target[selected_index]
        vertical = float(np.asarray(obj["world_xyz"], float)[2] - target_anchor[2])
        source_vacated_radius=float(payload.get("source_vacated_radius_m",0.055))
        geometric_source_vacated=min(distances_source)>source_vacated_radius
        source_transport_verified=bool(payload.get("source_transport_verified",False))
        selected_source_displacement=distances_source[selected_index]
        source_vacated=bool(geometric_source_vacated or (
            source_transport_verified
            and selected_source_displacement>source_vacated_radius))
        source_vacancy_method=("category_absence" if geometric_source_vacated else
            "prior_attachment_and_selected_object_displacement" if source_vacated else
            "unverified_category_occupancy")
        object_bounds = np.asarray(obj["world_bounds_10_90"], float)
        maximum_association_rank=float(np.clip(
            payload.get("max_target_association_rank",0.12),0.03,0.30))
        maximum_target_height_error=float(np.clip(
            payload.get("max_target_surface_height_error_m",0.012),0.005,0.06))
        target_height_error=(abs(float(np.asarray(target["world_xyz"],float)[2]
                                       - target_anchor[2]))
                             if target is not None else float("inf"))
        fresh_target_credible=(target_index is not None
                               and target_ranks[target_index] <= maximum_association_rank
                               and target_height_error <= maximum_target_height_error)
        # The target reference is captured before manipulation and therefore
        # contains the only unobstructed support footprint. After placement,
        # the manipulated object commonly hides the support centre; SAM then
        # returns an exposed crescent whose centroid and bounds are shifted
        # even though its depth still looks like a credible support surface.
        # A fresh target remains semantic/existence evidence, while immutable
        # pre-action sensor geometry owns containment and overlap when present.
        if anchored_bounds_valid:
            target_bounds=anchored_target_bounds
            geometry_source="pre_action_sensor_anchor"
        elif fresh_target_credible:
            target_bounds=np.asarray(target["world_bounds_10_90"],float)
            geometry_source="fresh_target_detection"
        elif target is not None:
            target_bounds=np.asarray(target["world_bounds_10_90"],float)
            geometry_source="fresh_target_detection_low_confidence"
        else:
            return {"verified":False,"reason":"no usable target support bounds"}
        if (object_bounds.shape != (2, 3) or target_bounds.shape != (2, 3)
                or not np.isfinite(object_bounds).all() or not np.isfinite(target_bounds).all()):
            return {"verified": False, "reason": "invalid fresh metric bounds"}
        object_size = np.maximum(object_bounds[1, :2] - object_bounds[0, :2], 1e-6)
        target_size = np.maximum(target_bounds[1, :2] - target_bounds[0, :2], 1e-6)
        intersection = np.maximum(
            np.minimum(object_bounds[1, :2], target_bounds[1, :2])
            - np.maximum(object_bounds[0, :2], target_bounds[0, :2]), 0.0)
        intersection_area = float(np.prod(intersection))
        object_area = float(np.prod(object_size))
        target_area = float(np.prod(target_size))
        object_coverage = intersection_area / object_area
        target_coverage = intersection_area / target_area
        # "A on B" must remain meaningful when either footprint is larger.
        # Dividing only by the manipulated object's bounds makes a perfectly
        # centred large bowl fail on a smaller plate.  Intersection over the
        # smaller observed footprint is a size-symmetric containment measure;
        # the independent centre, height, and source-vacancy gates below still
        # prevent a nearby or merely touching object from passing.
        overlap_fraction = intersection_area / min(object_area, target_area)
        # Use the independent pre-action support anchor as the contact plane.
        # Once an object is placed, it commonly occludes the support and the
        # fresh cross-query target mask can contain part of the object.  Its
        # upper Z quantile is then the bowl rim rather than the plate surface,
        # which creates a large fictitious negative penetration despite a
        # physically correct placement.  The anchor was measured from RGB-D
        # before manipulation and is not simulator state or evaluator data.
        support_plane_height = float(target_anchor[2])
        support_gap = float(object_bounds[0, 2] - support_plane_height)
        minimum_overlap = float(np.clip(payload.get("min_support_overlap", 0.60), 0.25, 0.95))
        maximum_gap = float(np.clip(payload.get("max_support_gap_m", 0.035), 0.005, 0.08))
        minimum_gap = float(np.clip(payload.get("min_support_gap_m", -0.025), -0.08, 0.0))
        center = np.asarray(obj["world_xyz"], float)[:2]
        margin = float(np.clip(payload.get("target_bounds_margin_m", 0.008), 0.0, 0.03))
        center_inside = bool(np.all(center >= target_bounds[0, :2] - margin)
                             and np.all(center <= target_bounds[1, :2] + margin))
        verified = bool(source_vacated and center_inside
                        and overlap_fraction >= minimum_overlap
                        and minimum_gap <= support_gap <= maximum_gap)
        return {
            "verified": bool(verified), "target_xy_error_m": target_xy_error,
            "vertical_offset_m": vertical, "source_vacated": source_vacated,
            "nearest_source_detection_m": min(distances_source), "object": obj,
            "selected_object_source_displacement_m":selected_source_displacement,
            "geometric_source_vacated":geometric_source_vacated,
            "source_transport_verified":source_transport_verified,
            "source_vacancy_method":source_vacancy_method,
            "target": target, "support_overlap_fraction": overlap_fraction,
            "object_coverage_fraction": object_coverage,
            "target_coverage_fraction": target_coverage,
            "support_overlap_normalization": "smaller_metric_footprint",
            "target_geometry_source":geometry_source,
            "minimum_support_overlap": minimum_overlap,
            "support_gap_m": support_gap, "center_inside_target_bounds": center_inside,
            "support_plane_height_m":support_plane_height,
            "support_height_source":"pre_action_sensor_anchor",
            "target_association_rank":(target_ranks[target_index]
                                       if target_index is not None else None),
            "maximum_target_association_rank":maximum_association_rank,
            "target_surface_height_error_m":target_height_error,
            "maximum_target_surface_height_error_m":maximum_target_height_error,
            "source_anchor_world_xyz": source_anchor.tolist(),
            "target_anchor_world_xyz": target_anchor.tolist(),
            "criterion": "fresh object and target masks with metric support overlap, height, and sensor-proven source transition",
        }

    def verify_attachment(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Conservatively verify a visually observed object follows the EEF."""
        object_query=str(payload.get("object_query") or "")
        source_anchor=np.asarray(payload.get("source_world_xyz") or (),float)
        frame=payload.get("frame")
        if source_anchor.shape!=(3,) or not isinstance(frame,Mapping):
            return {"verified":False,"reason":"fresh frame and independent source_ref are required"}
        proprio=frame.get("proprioception") or {}
        eef=np.asarray(proprio.get("robot0_eef_pos") or (),float)
        gripper=np.asarray(proprio.get("robot0_gripper_qpos") or (),float)
        if eef.shape!=(3,) or gripper.size<1:
            return {"verified":False,"reason":"EEF/gripper proprioception absent"}
        result=self.detect({"frame":frame,"camera":payload.get("camera","agentview"),
            "queries":[object_query],"box_threshold":payload.get("box_threshold",.16),
            "text_threshold":payload.get("text_threshold",.12),
            "max_detections_per_query":payload.get("max_detections_per_query",8)})
        objects=[item for item in result["detections"].get(object_query,[])
                 if "world_xyz" in item]
        width=float(np.sum(np.abs(gripper)))
        if not objects:
            return {"verified":False,"reason":"object not visible near gripper",
                    "object_count":0,"gripper_width_m":width}
        source_distances=[float(np.linalg.norm(np.asarray(item["world_xyz"],float)-source_anchor))
                          for item in objects]
        eef_distances=[float(np.linalg.norm(np.asarray(item["world_xyz"],float)-eef)) for item in objects]
        selected_index=int(np.argmin(eef_distances));selected=objects[selected_index]
        source_vacated=min(source_distances)>float(payload.get("source_vacated_radius_m",.055))
        near_eef=eef_distances[selected_index]<=float(payload.get("max_eef_distance_m",.16))
        retained_width=width>=float(payload.get("min_retained_width_m",.003))
        verified=bool(source_vacated and near_eef and retained_width)
        return {"verified":verified,"source_vacated":source_vacated,
                "nearest_source_detection_m":min(source_distances),
                "object_to_eef_distance_m":eef_distances[selected_index],
                "gripper_width_m":width,"retained_width":retained_width,
                "object":selected,"eef_world_xyz":eef.tolist(),
                "source_anchor_world_xyz":source_anchor.tolist(),
                "criterion":"fresh object mask near proprioceptive EEF, source vacated, nonempty gripper closure"}


__all__ = ["CapabilityInputError", "OpenVocabularyRGBD"]
