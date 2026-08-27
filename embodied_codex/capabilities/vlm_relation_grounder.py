"""Sensor-only VLM grounding of a language relation to one live candidate.

The capability receives an Adapter-issued RGB frame plus detector candidate
boxes.  It exposes neither simulator state nor benchmark success.  The VLM is
only asked to choose an index; metric pose and motion provenance continue to
come from the RGB-D detector and Robot Adapter.
"""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import io
import json
import mimetypes
from pathlib import Path
import time
import threading
from typing import Any, Mapping

from ._vlm_support import bounded_consensus, compact_image


class VLMRelationGroundingError(ValueError):
    pass


def _transient_api_error(exc: Exception) -> bool:
    name=type(exc).__name__
    if name in {"APIError","APIConnectionError","APITimeoutError","RateLimitError",
                "InternalServerError"}:return True
    status=getattr(exc,"status_code",None)
    return isinstance(status,int) and (status in {408,409,429} or status>=500)


class VLMVisualRelationGrounder:
    def __init__(self, *, api_key: str, base_url: str, model: str = "gpt-5.6-sol",
                 reasoning_effort: str = "high", timeout: float = 180,
                 total_timeout: float = 90,
                 client=None, retry_delays: tuple[float,...]=(2.0,5.0)) -> None:
        if not api_key and client is None:
            raise VLMRelationGroundingError("VLM API key is required")
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url,
                            timeout=float(timeout), max_retries=0)
        self.client = client
        self.model = str(model)
        self.reasoning_effort = str(reasoning_effort)
        self.request_timeout=max(0.1,float(timeout))
        self.total_timeout=max(0.1,float(total_timeout))
        self._deadline=threading.local()
        self.retry_delays=tuple(max(0.0,float(value)) for value in retry_delays)
        self.provenance = {
            "method": "foundation_vlm_candidate_relation_grounding",
            "model": self.model,
            "source_urls": ["https://platform.openai.com/docs/guides/images-vision"],
            "model_card_urls": ["https://platform.openai.com/docs/models"],
            "trained_on_current_task": False,
            "privileged_state_used": False,
            "training_data_declaration":(
                "General-purpose hosted foundation model used zero-shot; the Harness performs "
                "no LIBERO task-specific training or fine-tuning."),
            "contamination_check":{"evaluated_benchmark":"LIBERO",
                "method":"zero-shot API-use policy and no task-specific fine-tuning audit",
                "result":"no_declared_overlap"},
            "inputs": ["Adapter RGB image", "pixel boxes", "task language"],
        }

    @staticmethod
    def _json_object(text: str) -> Mapping[str, Any]:
        start = str(text).find("{")
        if start < 0:
            raise VLMRelationGroundingError("VLM response contains no JSON object")
        try:
            value, _end = json.JSONDecoder().raw_decode(str(text)[start:])
        except json.JSONDecodeError as exc:
            raise VLMRelationGroundingError("invalid VLM JSON response") from exc
        if not isinstance(value, Mapping):
            raise VLMRelationGroundingError("VLM response must be an object")
        return value

    @staticmethod
    def _image(frame: Mapping[str, Any], camera: str) -> tuple[str, str]:
        cameras = frame.get("cameras") if isinstance(frame, Mapping) else None
        item = cameras.get(camera) if isinstance(cameras, Mapping) else None
        if not isinstance(item, Mapping):
            raise VLMRelationGroundingError("Adapter camera frame is required")
        path = Path(str(item.get("rgb_path") or "")).resolve()
        if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
            raise VLMRelationGroundingError("invalid RGB artifact")
        expected = str(item.get("rgb_sha256") or "")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if not expected or actual != expected:
            raise VLMRelationGroundingError("RGB artifact hash mismatch")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return mime, base64.b64encode(path.read_bytes()).decode("ascii")

    def _complete(self, prompt: str, image_url: str) -> str:
        for attempt in range(len(self.retry_delays)+1):
            deadline=getattr(self._deadline,"value",None)
            remaining=(deadline-time.monotonic()) if deadline is not None else self.request_timeout
            if remaining<=0:
                raise VLMRelationGroundingError(
                    f"VLM relation consensus exceeded {self.total_timeout:g} seconds")
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]}], temperature=0, max_tokens=1000,
                    extra_body={"reasoning_effort": self.reasoning_effort},
                    timeout=min(self.request_timeout,max(0.1,remaining)))
                break
            except Exception as exc:
                if attempt>=len(self.retry_delays) or not _transient_api_error(exc):raise
                delay=min(self.retry_delays[attempt],max(0.0,
                    (deadline-time.monotonic()) if deadline is not None else self.retry_delays[attempt]))
                if delay>0:time.sleep(delay)
        return str(response.choices[0].message.content or "")

    @staticmethod
    def _annotated_image(image_bytes: bytes, objects: list[Mapping[str, Any]],
                         references: list[Mapping[str, Any]]) -> tuple[str, bytes]:
        """Render explicit object/reference IDs without changing sensor content.

        Coordinate-only prompts are unnecessarily error prone on crowded robot
        scenes.  The overlay is derived solely from live detector pixel boxes and
        lets the VLM inspect a *pair* rather than inventing which support object a
        selected object was associated with.  Invalid/non-image test fixtures fall
        back to the original bytes.
        """
        try:
            from PIL import Image, ImageDraw
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            draw = ImageDraw.Draw(image)
            groups = ((objects, "O", (255, 48, 48)),
                      (references, "R", (40, 120, 255)))
            for candidates, prefix, color in groups:
                for index, candidate in enumerate(candidates):
                    box = [float(value) for value in candidate["box_xyxy"]]
                    draw.rectangle(box, outline=color, width=3)
                    x, y = box[0], max(0.0, box[1] - 13.0)
                    draw.rectangle([x, y, x + 25, y + 13], fill=color)
                    draw.text((x + 2, y), f"{prefix}{index}", fill=(255, 255, 255))
            output = io.BytesIO()
            image.save(output, format="PNG")
            return "image/png", output.getvalue()
        except Exception:
            return "", image_bytes

    @staticmethod
    def _public_candidates(candidates: list[Mapping[str, Any]], prefix: str):
        public = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise VLMRelationGroundingError("candidate must be an object")
            box = candidate.get("box_xyxy")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise VLMRelationGroundingError("candidate pixel box is required")
            try:
                box = [round(float(value), 2) for value in box]
            except (TypeError, ValueError) as exc:
                raise VLMRelationGroundingError("invalid candidate pixel box") from exc
            public.append({"id": f"{prefix}{index}", "box_xyxy": box,
                           "detector_label": str(candidate.get("label") or "object")})
        return public

    def select(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        instruction = str(payload.get("instruction") or "").strip()
        relation = str(payload.get("relation") or instruction).strip()
        candidates = payload.get("candidates")
        references = payload.get("reference_candidates") or []
        camera = str(payload.get("camera") or "agentview")
        if not instruction or not relation:
            raise VLMRelationGroundingError("instruction and relation are required")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 32:
            raise VLMRelationGroundingError("between 1 and 32 candidates are required")
        if not isinstance(references, list) or len(references) > 32:
            raise VLMRelationGroundingError("reference_candidates must be a list of at most 32")
        public = self._public_candidates(candidates, "O")
        public_references = self._public_candidates(references, "R")
        mime, encoded = self._image(payload.get("frame") or {}, camera)
        image_bytes = base64.b64decode(encoded)
        annotated_mime, annotated = self._annotated_image(
            image_bytes, candidates, references)
        if annotated_mime:
            mime, encoded = annotated_mime, base64.b64encode(annotated).decode("ascii")
        compact_mime, compact_bytes=compact_image(base64.b64decode(encoded),mime)
        mime,encoded=compact_mime,base64.b64encode(compact_bytes).decode("ascii")
        prompt = (
            "You are a sensor-only visual grounding module for a robot. "
            "Jointly choose the object and, when provided, the reference/support "
            "instance that satisfy the stated language relation in the current image. "
            "Red O labels mark object candidates and blue R labels mark reference "
            "candidates. Detector labels and scores are proposals, not proof. Do not "
            "rename a generic platform, fixture, support surface, or unrelated box to make the "
            "relation fit: inspect visual object type, packaging/appearance, contact, "
            "and relative position. Image coordinates are x-right and y-down. If no "
            "visually supported pair exists, return null.\n"
            f"Task instruction: {instruction}\nRelation to ground: {relation}\n"
            f"Object candidates: {json.dumps(public, separators=(',', ':'))}\n"
            f"Reference candidates: {json.dumps(public_references, separators=(',', ':'))}\n"
            "Return ONLY JSON with keys selected_id (integer or null), "
            "selected_reference_id (integer or null), reference_description (string), "
            "reason (string), confidence (0..1). IDs in JSON are numeric indices."
        )
        try:
            rounds = int(payload.get("consensus_rounds", 3))
        except (TypeError, ValueError):
            rounds = 3
        rounds = max(1, min(5, rounds))
        image_url=f"data:{mime};base64,{encoded}"
        # Consensus calls are independent samples of the same immutable sensor
        # evidence.  Run them concurrently without changing the vote threshold
        # or accepting partial results.
        def complete(_index: int, deadline: float) -> str:
            self._deadline.value=deadline
            return self._complete(prompt,image_url)
        required=rounds//2+1
        def normalized_id(value, size):
            if value is None or isinstance(value,bool):return None
            try:value=int(value)
            except (TypeError,ValueError):return None
            return value if 0<=value<size else None
        def decision_pair(text):
            decision=self._json_object(text)
            object_id=normalized_id(decision.get("selected_id"),len(candidates))
            reference_id=normalized_id(
                decision.get("selected_reference_id"),len(references))
            return ((object_id,reference_id) if object_id is not None
                    and (not references or reference_id is not None) else None)
        def decision_quorum(values):
            votes=[vote for vote in (decision_pair(value) for value in values)
                   if vote is not None]
            return bool(votes and Counter(votes).most_common(1)[0][1]>=required)
        texts=bounded_consensus(rounds,self.total_timeout,complete,
            error_type=VLMRelationGroundingError,operation="VLM relation consensus",
            minimum_results=required,decision_quorum=decision_quorum)
        decisions=[self._json_object(value) for value in texts]

        votes = []
        for decision in decisions:
            object_id = normalized_id(decision.get("selected_id"), len(candidates))
            reference_id = normalized_id(
                decision.get("selected_reference_id"), len(references))
            # A joint query is not accepted without an explicit reference vote.
            if object_id is not None and (not references or reference_id is not None):
                votes.append((object_id, reference_id))
        winner = Counter(votes).most_common(1)
        agreed = bool(winner and winner[0][1] >= required)
        selected_id, selected_reference_id = winner[0][0] if agreed else (None, None)
        matching_decisions = [d for d in decisions
            if normalized_id(d.get("selected_id"), len(candidates)) == selected_id
            and normalized_id(d.get("selected_reference_id"), len(references)) == selected_reference_id]
        decision = matching_decisions[0] if matching_decisions else decisions[0]
        try:
            confidence = float(decision.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "selected_index": selected_id,
            "selected_reference_index": selected_reference_id,
            "reference_description": str(decision.get("reference_description") or ""),
            "reason": str(decision.get("reason") or ""),
            "confidence": max(0.0, min(1.0, confidence)),
            "consensus": {"rounds": rounds, "completed_rounds":len(decisions),
                          "required": required,
                          "winning_votes": winner[0][1] if winner else 0,
                          "agreed": agreed},
            "method": "foundation_vlm_candidate_relation_grounding",
            "sensor_only": True,
            "model": self.model,
        }


__all__ = ["VLMVisualRelationGrounder", "VLMRelationGroundingError"]
