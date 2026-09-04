from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import Path

import pytest

from roboforge.candidate_bundle import CandidateBundleError, CandidateBundleStore
from roboforge.fakes import FakeAdapter
from roboforge.models import AdapterResult
from roboforge.service import ExperimentService
from roboforge.trial_artifacts import materialize_trial


def _workspace(root: Path) -> tuple[Path, Path]:
    workspace = root / "workspace"
    controller = workspace / "controllers" / "controller.py"
    controller.parent.mkdir(parents=True)
    (workspace / "capabilities").mkdir()
    controller.write_text("from helper import VALUE\ndef run(robot): return VALUE\n")
    (workspace / "helper.py").write_text("VALUE = 'frozen'\n")
    (workspace / "capabilities" / "vision.py").write_text("API_VERSION = 1\n")
    (workspace / "requirements.txt").write_text("numpy==1.26.4\n")
    (workspace / "configs.yaml").write_text("model: sam3\n")
    return workspace, controller


def test_candidate_bundle_freezes_complete_dependency_tree_and_is_immutable(tmp_path):
    workspace, controller = _workspace(tmp_path)
    store = CandidateBundleStore(tmp_path / "cas")
    manifest = store.freeze(
        workspace=workspace,
        entrypoint=controller,
        runtime_metadata={
            "runtime_provider_digest": "a" * 64,
            "runtime_api_version": "libero-v1",
            "model_artifact_digests": {"sam3": "b" * 64},
        },
    )
    digest = manifest["candidate_bundle_digest"]
    assert manifest["candidate_bundle_id"] == f"candidate://{digest}"
    assert {item["path"] for item in manifest["files"]} == {
        "capabilities/vision.py", "configs.yaml", "controllers/controller.py",
        "helper.py", "requirements.txt",
    }
    assert manifest["controller_digest"] == hashlib.sha256(controller.read_bytes()).hexdigest()
    assert manifest["capability_digests"]
    assert manifest["model_artifact_digests"] == ["b" * 64]

    frozen_helper = store.source_root(digest) / "helper.py"
    (workspace / "helper.py").write_text("VALUE = 'live-edit'\n")
    assert frozen_helper.read_text() == "VALUE = 'frozen'\n"
    assert store.verify(digest) == manifest

    frozen_helper.chmod(0o644)
    frozen_helper.write_text("tampered\n")
    with pytest.raises(CandidateBundleError, match="file digest mismatch"):
        store.verify(digest)


def test_concurrent_freeze_resolves_to_one_verified_object(tmp_path):
    workspace, controller = _workspace(tmp_path)
    store = CandidateBundleStore(tmp_path / "cas")

    def freeze() -> str:
        return store.freeze(workspace=workspace, entrypoint=controller)[
            "candidate_bundle_digest"
        ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        digests = list(pool.map(lambda _: freeze(), range(8)))
    assert len(set(digests)) == 1
    store.verify(digests[0])
    assert len([path for path in store.root.iterdir() if path.is_dir()]) == 1


class BundleAwareAdapter(FakeAdapter):
    def __init__(self, live_helper: Path):
        super().__init__()
        self.live_helper = live_helper
        self.executed_source = None
        self.bundle_digest = None

    def candidate_runtime_metadata(self):
        return {"runtime_provider_digest": "c" * 64, "runtime_api_version": "test-v1"}

    def execute_controller(self, *, controller_path, controller_sha256,
                           environment_generation, candidate_bundle_digest,
                           candidate_source_root):
        self.live_helper.write_text("VALUE = 'changed-during-run'\n")
        self.executed_source = (candidate_source_root / "helper.py").read_text()
        self.bundle_digest = candidate_bundle_digest
        return AdapterResult(
            public={"controller_termination": "completed", "task_success": False},
            private_receipt={
                "kind": "physical",
                "controller_sha256": controller_sha256,
                "environment_generation": environment_generation,
                "candidate_bundle_digest": candidate_bundle_digest,
                "verified": True,
            },
        )

    def validate_receipt(self, receipt, *, controller_sha256,
                         environment_generation, candidate_bundle_digest):
        return bool(
            receipt.get("verified") is True
            and receipt.get("controller_sha256") == controller_sha256
            and receipt.get("environment_generation") == environment_generation
            and receipt.get("candidate_bundle_digest") == candidate_bundle_digest
        )


def test_service_executes_bundle_and_materializes_exact_frozen_tree(tmp_path):
    workspace, controller = _workspace(tmp_path)
    adapter = BundleAwareAdapter(workspace / "helper.py")
    service = ExperimentService(tmp_path / "run", adapter)
    evidence = service.run_controller(
        request_id="bundle-trial", controller_path=controller, intent="freeze test"
    )
    assert evidence.candidate_bundle_digest == adapter.bundle_digest
    assert adapter.executed_source == "VALUE = 'frozen'\n"
    assert (workspace / "helper.py").read_text() == "VALUE = 'changed-during-run'\n"

    result = materialize_trial(service, evidence, workspace, controller_path=controller)
    trial = workspace / ".roboforge" / "trials" / "physical-000001"
    assert (trial / "frozen_source" / "helper.py").read_text() == "VALUE = 'frozen'\n"
    manifest = json.loads((trial / "candidate_bundle.json").read_text())
    assert manifest["candidate_bundle_digest"] == evidence.candidate_bundle_digest
    assert result["candidate_bundle_digest"] == evidence.candidate_bundle_digest


def test_bundle_receipt_cannot_be_reused_for_different_bundle(tmp_path):
    workspace, controller = _workspace(tmp_path)
    adapter = BundleAwareAdapter(workspace / "helper.py")
    service = ExperimentService(tmp_path / "run", adapter, max_trials=2)
    first = service.run_controller(
        request_id="first", controller_path=controller, intent="first"
    )
    controller.write_text("def run(robot): return 'different'\n")
    second = service.run_controller(
        request_id="second", controller_path=controller, intent="second"
    )
    assert first.candidate_bundle_digest != second.candidate_bundle_digest
    assert first.controller_sha256 != second.controller_sha256
