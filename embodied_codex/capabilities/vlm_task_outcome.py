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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping


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
        self.consensus_rounds = max(1, min(5, int(consensus_rounds)))
        self.retry_delays=tuple(max(0.0,float(value)) for value in retry_delays)
        self.provenance = {
            "method": "foundation_vlm_before_after_task_verification",
            "model": self.model,
            "trained_on_current_task": False,
            "privileged_state_used": False,
            "inputs": ["task language", "initial Adapter RGB", "final Adapter RGB"],
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

    def _complete(self, prompt: str, before_url: str, after_url: str) -> str:
        for attempt in range(len(self.retry_delays)+1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt + "\nBEFORE image:"},
                        {"type": "image_url", "image_url": {"url": before_url}},
                        {"type": "text", "text": "AFTER image:"},
                        {"type": "image_url", "image_url": {"url": after_url}},
                    ]}], temperature=0, max_tokens=1000,
                    extra_body={"reasoning_effort": self.reasoning_effort})
                break
            except Exception as exc:
                if attempt>=len(self.retry_delays) or not _transient_api_error(exc):raise
                time.sleep(self.retry_delays[attempt])
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
            "when occlusion prevents verification. Do not infer simulator state.\n"
            f"Task: {instruction}\n"
            "Return ONLY JSON with verified (boolean), source_relation_satisfied "
            "(boolean), target_relation_satisfied (boolean), contradiction (string), "
            "reason (string), and confidence (0..1)."
        )
        before_url, after_url = self._image(before), self._image(after)
        with ThreadPoolExecutor(max_workers=self.consensus_rounds) as pool:
            texts=list(pool.map(
                lambda _index:self._complete(prompt,before_url,after_url),
                range(self.consensus_rounds)))
        decisions=[self._json_object(value) for value in texts]
        votes = [bool(d.get("verified") is True
                      and d.get("source_relation_satisfied") is True
                      and d.get("target_relation_satisfied") is True
                      and not str(d.get("contradiction") or "").strip())
                 for d in decisions]
        counts = Counter(votes)
        required = self.consensus_rounds // 2 + 1
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
            "consensus": {"rounds": self.consensus_rounds, "required": required,
                          "true_votes": counts[True], "false_votes": counts[False]},
            "method": "foundation_vlm_before_after_task_verification",
            "sensor_only": True,
        }


__all__ = ["VLMVisualTaskOutcomeVerifier", "VLMTaskOutcomeError"]
