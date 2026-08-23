"""Run the provenance-approved general detector on a CALVIN RGB frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from groundingdino_detector import GroundingDinoDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--caption", default="button . switch . led light")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    detector = GroundingDinoDetector(args.config, args.checkpoint, device=args.device)
    detections = detector.detect(args.image, args.caption)
    report = {
        "protocol": "harness-acquired-task-zero-shot-v2",
        "benchmark": "CALVIN",
        "surface": "development_only",
        "asset": "GroundingDINO Swin-T OGC",
        "asset_manifest": "capability_library/assets/groundingdino-swint-ogc.json",
        "current_task_training_used": False,
        "image": str(args.image),
        "caption": args.caption,
        "detections": detections,
        "sealed_results_consumed_for_iteration": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
