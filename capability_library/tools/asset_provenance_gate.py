"""Machine-checkable provenance gate for zero-shot embodied assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


FORBIDDEN_INFERENCE_INPUTS = {
    "reward",
    "success",
    "goal_predicates",
    "privileged_object_poses",
    "simulator_internal_state",
    "evaluation_episode_identifier",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_asset_manifest(
    manifest: Mapping[str, Any],
    *,
    current_tasks: tuple[str, ...] = (),
    verify_local_hash: bool = True,
) -> dict[str, Any]:
    """Return a deterministic current-task eligibility decision."""
    reasons: list[str] = []
    training = manifest.get("training_provenance", {})
    preprocessing = manifest.get("preprocessing_provenance", {})
    sources = (training, preprocessing)
    if training.get("status") != "documented":
        reasons.append("training provenance is not documented")
    if preprocessing.get("status") != "documented":
        reasons.append("preprocessing/action-normalization provenance is not documented")

    evaluated = {item.casefold() for item in current_tasks}
    exposed_tasks = {
        str(item).casefold()
        for source in sources
        for item in source.get("task_ids", [])
    }
    covers_all = any(
        bool(source.get("covers_all_tasks_in_benchmark_family")) for source in sources
    )
    overlap = sorted(evaluated.intersection(exposed_tasks))
    if current_tasks and covers_all:
        reasons.append("training provenance covers all tasks in the evaluated benchmark family")
    elif overlap:
        reasons.append("current task exposure: " + ", ".join(overlap))

    inference_inputs = {
        str(item).casefold() for item in manifest.get("inference_inputs", [])
    }
    bad_inputs = sorted(FORBIDDEN_INFERENCE_INPUTS.intersection(inference_inputs))
    if bad_inputs:
        reasons.append("forbidden inference inputs: " + ", ".join(bad_inputs))

    artifact = manifest.get("artifact", {})
    local_path = artifact.get("local_path")
    expected_hash = artifact.get("sha256")
    if verify_local_hash and local_path:
        path = Path(local_path)
        if not path.is_file():
            reasons.append(f"local artifact is missing: {path}")
        elif not expected_hash:
            reasons.append("local artifact has no recorded sha256")
        elif sha256_file(path) != str(expected_hash).casefold():
            reasons.append("local artifact sha256 mismatch")

    family_exposure = any(source.get("benchmark_families") for source in sources)
    return {
        "asset_id": manifest.get("id"),
        "eligible": not reasons,
        "track": "harness_acquired_task_zero_shot" if not reasons else "not_primary_eligible",
        "reporting_stratum": (
            "task_disjoint_transfer" if family_exposure else "benchmark_family_disjoint"
        ),
        "reasons": reasons,
    }


def register_asset_provenance_tool(registry: Any) -> None:
    """Register the provenance decision as a Thea in-process tool."""

    @registry.tool(
        description=(
            "Check whether a discovered learned embodied asset is eligible for "
            "the no-current-task-training Harness zero-shot track."
        )
    )
    def check_asset_provenance(
        manifest_path: str, current_tasks: list[str]
    ) -> dict[str, Any]:
        manifest = json.loads(Path(manifest_path).read_text())
        return evaluate_asset_manifest(manifest, current_tasks=tuple(current_tasks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--current-task", action="append", default=[])
    parser.add_argument("--no-verify-local-hash", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    decision = evaluate_asset_manifest(
        manifest,
        current_tasks=tuple(args.current_task),
        verify_local_hash=not args.no_verify_local_hash,
    )
    print(json.dumps(decision, indent=2))
    raise SystemExit(0 if decision["eligible"] else 2)


if __name__ == "__main__":
    main()
