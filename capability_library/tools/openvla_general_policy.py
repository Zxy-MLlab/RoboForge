"""Adapter for the public, non-LIBERO OpenVLA base checkpoint.

The adapter uses the checkpoint's documented NYU Franka action statistics and
maps the physical delta action into robosuite OSC_POSE bounds. It receives only
RGB and language from the simulation episode; evaluator state is filtered by
the shared language-policy wrapper.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from libero_language_policy import make_execute_language_policy


DEFAULT_UNNORM_KEY = "nyu_franka_play_dataset_converted_externally_to_rlds"
DEFAULT_MODEL_PATH = "/data/zxy/cache/models--openvla--openvla-7b"


def map_physical_action_to_osc(
    action: Iterable[float],
    *,
    position_scale: Iterable[float] = (0.06, 0.07, 0.06),
    orientation_scale: Iterable[float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """Map source-dataset metric deltas to bounded OSC_POSE action values."""
    values = np.asarray(list(action), dtype=np.float64).reshape(-1)
    if values.size != 7:
        raise ValueError(f"OpenVLA action must have 7 values, got {values.size}")
    scales = np.concatenate(
        [np.asarray(list(position_scale)), np.asarray(list(orientation_scale))]
    )
    if scales.size != 6 or np.any(scales <= 0):
        raise ValueError("action scales must contain six positive values")
    mapped = np.zeros(7, dtype=np.float64)
    mapped[:6] = np.clip(values[:6] / scales, -1.0, 1.0)
    # OpenVLA's documented dataset convention is 0..1 gripper openness;
    # robosuite OSC_POSE uses -1 close .. +1 open.
    mapped[6] = float(np.clip(values[6] * 2.0 - 1.0, -1.0, 1.0))
    return mapped


class OpenVLAGeneralInfer:
    """Lazy OpenVLA base predictor with a documented non-LIBERO stats key."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        *,
        unnorm_key: str = DEFAULT_UNNORM_KEY,
        device: str = "cuda:0",
        action_steps: int = 10,
    ):
        self.model_path = str(model_path)
        self.unnorm_key = str(unnorm_key)
        self.device = device
        self.action_steps = int(action_steps)
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        ).to(self.device).eval()
        if self.unnorm_key not in self._model.norm_stats:
            raise ValueError(f"unnorm_key is not present in model stats: {self.unnorm_key}")

    def __call__(self, observation: Mapping[str, Any], instruction: str) -> Iterable[np.ndarray]:
        self._load()
        import torch
        from PIL import Image

        image = observation.get("agentview_image")
        if image is None:
            raise KeyError("OpenVLA adapter requires agentview_image")
        image = np.asarray(image)
        # LIBERO stores camera images upside down; use the documented RGB view.
        image = Image.fromarray(np.ascontiguousarray(image[::-1]).astype(np.uint8), mode="RGB")
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self._processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            action = self._model.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False,
            )
        if isinstance(action, tuple):
            action = action[0]
        if hasattr(action, "detach"):
            action = action.detach().float().cpu().numpy()
        mapped = map_physical_action_to_osc(action)
        for _ in range(max(1, self.action_steps)):
            yield mapped.copy()


def make_openvla_general_tool(
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    device: str = "cuda:0",
    max_actions_per_call: int = 10,
):
    """Build the Thea simulator Tool for the general OpenVLA candidate."""
    infer = OpenVLAGeneralInfer(model_path=model_path, device=device, action_steps=max_actions_per_call)
    return make_execute_language_policy(infer, max_actions_per_call=max_actions_per_call)


__all__ = [
    "DEFAULT_MODEL_PATH",
    "DEFAULT_UNNORM_KEY",
    "OpenVLAGeneralInfer",
    "make_openvla_general_tool",
    "map_physical_action_to_osc",
]
