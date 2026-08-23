"""General open-vocabulary detector backed by a frozen GroundingDINO model."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json


class GroundingDinoDetector:
    """Load once and detect text-grounded boxes without benchmark state."""

    def __init__(self, config_path: str, checkpoint_path: str, *, device: str = "cuda") -> None:
        import torch
        from groundingdino.models import build_model
        from groundingdino.util.misc import clean_state_dict
        from groundingdino.util.slconfig import SLConfig

        config = SLConfig.fromfile(config_path)
        config.device = device
        self.device = device
        self.model = build_model(config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.model.eval().to(device)

    def detect(
        self,
        image_path: str | Path,
        caption: str,
        *,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ) -> list[dict[str, Any]]:
        import torch
        from PIL import Image
        import groundingdino.datasets.transforms as transforms
        from groundingdino.util.utils import get_phrases_from_posmap

        normalized_caption = str(caption).lower().strip()
        if not normalized_caption:
            raise ValueError("caption must not be empty")
        if not normalized_caption.endswith("."):
            normalized_caption += "."
        transform = transforms.Compose(
            [
                transforms.RandomResize([800], max_size=1333),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(Image.open(image_path).convert("RGB"), None)
        with torch.no_grad():
            outputs = self.model(image[None].to(self.device), captions=[normalized_caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]
        keep = logits.max(dim=1)[0] > float(box_threshold)
        logits = logits[keep]
        boxes = boxes[keep]
        tokenized = self.model.tokenizer(normalized_caption)
        detections = []
        for box, logit in zip(boxes, logits):
            detections.append(
                {
                    "label": get_phrases_from_posmap(
                        logit > float(text_threshold), tokenized, self.model.tokenizer
                    ).replace(".", ""),
                    "score": float(logit.max()),
                    "box_cxcywh_normalized": [float(value) for value in box],
                }
            )
        return detections


__all__ = ["GroundingDinoDetector"]


def _patch_transformers_compatibility() -> None:
    """Bridge the transformers 5.x API used by this frozen checkpoint."""
    from transformers import BertModel
    if not hasattr(BertModel, "get_head_mask"):
        BertModel.get_head_mask = lambda self, head_mask, num_hidden_layers, is_attention_chunked=False: [None] * num_hidden_layers
    if not getattr(BertModel, "_gdino_compat", False):
        original = BertModel.get_extended_attention_mask
        def compat(self, attention_mask, input_shape, device=None):
            import torch
            return original(self, attention_mask, input_shape, dtype=torch.float32)
        BertModel.get_extended_attention_mask = compat
        BertModel._gdino_compat = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen GroundingDINO RGB detector")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--config", default="/data/zxy/embodied_frontier/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--checkpoint", default="/data/zxy/GroundingDINO/models/groundingdino_swint_ogc.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--box-threshold", type=float, default=.2)
    parser.add_argument("--text-threshold", type=float, default=.15)
    args = parser.parse_args()
    _patch_transformers_compatibility()
    detector = GroundingDinoDetector(args.config, args.checkpoint, device=args.device)
    result={query:detector.detect(args.image,query,box_threshold=args.box_threshold,
                                  text_threshold=args.text_threshold) for query in args.query}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
