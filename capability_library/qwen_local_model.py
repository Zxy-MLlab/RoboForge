"""Local Qwen2.5-VL adapter implementing Thea's ModelProtocol.

This adapter uses a conservative JSON tool-call envelope because the local
checkpoint is an instruction VLM rather than an OpenAI-compatible server.
It never receives evaluator-only state; Thea controls the context supplied.
"""

from __future__ import annotations

import json
import re
import base64
from io import BytesIO
from typing import Any

from harness.context import Context, ModelResponse, ToolCall


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _tool_prompt(tools: list[dict[str, Any]]) -> str:
    definitions = []
    for tool in tools:
        definitions.append(
            {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "inputSchema": tool.get("inputSchema"),
            }
        )
    return (
        "You may call at most one tool per turn. Return ONLY valid JSON with this "
        "shape: {\"text\": string, \"tool_calls\": [{\"id\": string, "
        "\"name\": string, \"arguments\": object}]}. Use an empty list when "
        "no tool is needed. Never invent evaluator state or claim success.\n"
        "AVAILABLE TOOLS:\n"
        + json.dumps(definitions, ensure_ascii=False, default=str)
    )


def parse_qwen_decision(raw: str) -> ModelResponse:
    """Parse the strict local JSON envelope, retaining malformed output as text."""
    match = _JSON_BLOCK.search(raw or "")
    if not match:
        return ModelResponse(text=raw or "", stop_reason="text")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ModelResponse(text=raw or "", stop_reason="text")
    calls: list[ToolCall] = []
    for index, item in enumerate(payload.get("tool_calls") or []):
        if not isinstance(item, dict) or not item.get("name"):
            continue
        arguments = item.get("arguments", {})
        calls.append(
            ToolCall(
                id=str(item.get("id") or f"qwen_call_{index}"),
                name=str(item["name"]),
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
    return ModelResponse(
        text=str(payload.get("text") or ""),
        tool_calls=calls,
        stop_reason="tool_use" if calls else "text",
    )


class LocalQwenVL:
    """Lazy-loading local Qwen2.5-VL model for Thea's provider boundary."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda:0",
        max_new_tokens: int = 512,
    ):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            local_files_only=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

    @staticmethod
    def _flatten_message(message: dict[str, Any]) -> dict[str, Any]:
        role = str(message.get("role") or "user")
        content = message.get("content", "")
        if isinstance(content, list):
            blocks = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        blocks.append({"type": "text", "text": str(block.get("text") or "")})
                    elif block.get("type") in {"image", "image_url"}:
                        image_url = block.get("image_url")
                        source = block.get("image") or block.get("url")
                        if isinstance(image_url, dict):
                            source = source or image_url.get("url")
                        if isinstance(source, dict):
                            source = source.get("url")
                        if source:
                            blocks.append({"type": "image", "image": source})
            content = blocks or ""
        elif isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False, default=str)
        return {"role": role, "content": content}

    def call(self, ctx: Context, tools: list[dict[str, Any]] | None = None) -> ModelResponse:
        self._load()
        selected_tools = ctx.tool_definitions if tools is None else tools
        system = "\n\n".join(
            part for part in (ctx.provider_system_content, _tool_prompt(selected_tools)) if part
        )
        messages = [{"role": "system", "content": system}]
        messages.extend(self._flatten_message(message) for message in ctx.messages_for_model())
        images = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                source = block.get("image") if isinstance(block, dict) else None
                if not isinstance(source, str) or not source.startswith("data:image/"):
                    continue
                _prefix, _separator, encoded = source.partition(",")
                try:
                    from PIL import Image

                    images.append(Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB"))
                except Exception:
                    continue
        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs = {"text": [prompt], "return_tensors": "pt"}
        if images:
            processor_kwargs["images"] = images
        inputs = self._processor(**processor_kwargs).to(self.device)
        with __import__("torch").inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output[:, inputs.input_ids.shape[1]:]
        raw = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return parse_qwen_decision(raw)


__all__ = ["LocalQwenVL", "parse_qwen_decision"]
