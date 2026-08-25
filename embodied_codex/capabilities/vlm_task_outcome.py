"""Independent sensor-only verification of a language task from before/after RGB.

This verifier is owned by the Harness rather than the generated controller.  It
prevents a self-consistent but wrongly grounded controller from declaring
success merely because *some* same-class object moved to the destination.
"""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
import mimetypes
from pathlib import Path
import time
import threading
from typing import Any, Mapping

from ._vlm_support import bounded_consensus, compact_image


class VLMTaskOutcomeError(ValueError):
    pass


def _transient_api_error(exc: Exception) -> bool:
    name=type(exc).__name__
    if name in {"APIError","APIConnectionError","APITimeoutError","RateLimitError",
                "InternalServerError"}:return True
    status=getattr(exc,"status_code",None)
    return isinstance(status,int) and (status in {408,409,429} or status>=500)


class VLMVisualTaskOutcomeVerifier:
    def __init__(self, *, api_key: str, base_url: str, model: str = "gpt-5.6-sol",
                 reasoning_effort: str = "high", timeout: float = 180,
                 total_timeout: float = 90,
                 consensus_rounds: int = 3, client=None,
                 retry_delays: tuple[float,...]=(2.0,5.0)) -> None:
        if not api_key and client is None:
            raise VLMTaskOutcomeError("VLM API key is required")
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
        self.consensus_rounds = max(1, min(5, int(consensus_rounds)))
        self.retry_delays=tuple(max(0.0,float(value)) for value in retry_delays)
        self.provenance = {
            "method": "foundation_vlm_before_after_task_verification",
            "model": self.model,
            "source_urls": ["https://platform.openai.com/docs/guides/images-vision"],
            "model_card_urls": ["https://platform.openai.com/docs/models"],
            "trained_on_current_task": False,
            "privileged_state_used": False,
            "training_data_declaration":(
                "General-purpose hosted foundation model used zero-shot; no LIBERO "
                "task-specific training or fine-tuning is performed."),
            "contamination_check":{"evaluated_benchmark":"LIBERO",
                "method":"zero-shot API-use policy and no task-specific fine-tuning audit",
                "result":"no_declared_overlap"},
            "inputs": ["task language", "initial external+wrist RGB montage",
                       "final external+wrist RGB montage"],
        }

    @staticmethod
    def _image(item: Mapping[str, Any]) -> str:
        path = Path(str(item.get("rgb_path") or "")).resolve()
        if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
            raise VLMTaskOutcomeError("invalid RGB artifact")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != str(item.get("rgb_sha256") or ""):
            raise VLMTaskOutcomeError("RGB artifact hash mismatch")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        mime,data=compact_image(data,mime)
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    @staticmethod
    def _json_object(text: str) -> Mapping[str, Any]:
        start = str(text).find("{")
        if start < 0:
            raise VLMTaskOutcomeError("VLM response contains no JSON object")
        try:
            value, _ = json.JSONDecoder().raw_decode(str(text)[start:])
        except json.JSONDecodeError as exc:
            raise VLMTaskOutcomeError("invalid VLM JSON response") from exc
        if not isinstance(value, Mapping):
            raise VLMTaskOutcomeError("VLM response must be an object")
        return value

    @staticmethod
    def _contradiction_present(value: Any) -> bool:
        normalized=" ".join(str(value or "").strip().lower().split())
        return normalized not in {
            "","none","no contradiction","no contradictions","null","n/a",
            "not applicable",
        }

    @classmethod
    def _decision_vote(cls, decision: Mapping[str, Any]) -> bool:
        return bool(decision.get("verified") is True
            and decision.get("source_relation_satisfied") is True
            and decision.get("target_relation_satisfied") is True
            and not cls._contradiction_present(decision.get("contradiction")))

    def _complete(self, prompt: str, before_url: str, after_url: str) -> str:
        for attempt in range(len(self.retry_delays)+1):
            deadline=getattr(self._deadline,"value",None)
            remaining=(deadline-time.monotonic()) if deadline is not None else self.request_timeout
            if remaining<=0:
                raise VLMTaskOutcomeError(
                    f"VLM task-outcome consensus exceeded {self.total_timeout:g} seconds")
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt + "\nBEFORE image:"},
                        {"type": "image_url", "image_url": {"url": before_url}},
                        {"type": "text", "text": "AFTER image:"},
                        {"type": "image_url", "image_url": {"url": after_url}},
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

    def verify(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        instruction = str(payload.get("instruction") or "").strip()
        before, after = payload.get("before"), payload.get("after")
        if not instruction or not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise VLMTaskOutcomeError("instruction, before, and after are required")
        prompt = (
            "You are an independent sensor-only robot task verifier. Compare BEFORE "
            "and AFTER and decide whether the exact language task is visibly complete. "
            "Track the instance by every identifying relation in the instruction, not "
            "just category. Seeing some same-class object at the destination is "
            "insufficient if the originally specified object remains at its source. "
            "Check source change, destination relation, and contradictions. Fail closed "
            "when occlusion prevents verification. Each image may be a synchronized "
            "two-panel montage labeled EXTERNAL (left) and WRIST (right). Use the wrist "
            "panel to distinguish an object actually retained by the gripper from an "
            "object merely underneath it in the external projection. Visible overlap "
            "between gripper and object in one view is not proof of attachment. Do not "
            "infer simulator state.\n"
            f"Task: {instruction}\n"
            "Return ONLY JSON with verified (boolean), source_relation_satisfied "
            "(boolean), target_relation_satisfied (boolean), contradiction (string; "
            "use an empty string when there is no contradiction), "
            "reason (string), and confidence (0..1)."
        )
        before_url, after_url = self._image(before), self._image(after)
        def complete(_index: int, deadline: float) -> str:
            self._deadline.value=deadline
            return self._complete(prompt,before_url,after_url)
        required=self.consensus_rounds//2+1
        def decision_quorum(values):
            votes=[self._decision_vote(self._json_object(value)) for value in values]
            return bool(votes and Counter(votes).most_common(1)[0][1]>=required)
        texts=bounded_consensus(self.consensus_rounds,self.total_timeout,complete,
            error_type=VLMTaskOutcomeError,operation="VLM task-outcome consensus",
            minimum_results=required,decision_quorum=decision_quorum)
        decisions=[self._json_object(value) for value in texts]
        votes = [self._decision_vote(d) for d in decisions]
        counts = Counter(votes)
        verified = counts[True] >= required
        representative = next((d for d, vote in zip(decisions, votes)
                               if vote == verified), decisions[0])
        try:
            confidence = float(representative.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "verified": verified,
            "source_relation_satisfied": bool(
                representative.get("source_relation_satisfied") is True),
            "target_relation_satisfied": bool(
                representative.get("target_relation_satisfied") is True),
            "contradiction": str(representative.get("contradiction") or ""),
            "reason": str(representative.get("reason") or ""),
            "confidence": max(0.0, min(1.0, confidence)),
            "consensus": {"rounds": self.consensus_rounds,
                          "completed_rounds":len(decisions),"required": required,
                          "true_votes": counts[True], "false_votes": counts[False]},
            "method": "foundation_vlm_before_after_task_verification",
            "sensor_only": True,
        }


__all__ = ["VLMVisualTaskOutcomeVerifier", "VLMTaskOutcomeError"]
