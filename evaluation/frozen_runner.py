"""External frozen-Controller evaluation and dependency restoration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from embodied_codex.kernel.assets import CapabilityLibrary

from .sealed_evaluation import SealedEvaluationPolicy


class FrozenEvaluationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_skill(controller: Path, asset_root: Path) -> dict[str, Any] | None:
    expected = _sha256(controller)
    candidates = []
    direct = controller.parent / "manifest.json"
    if direct.is_file():
        try:
            direct_manifest = json.loads(direct.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenEvaluationError("frozen Skill manifest is invalid") from exc
        if direct_manifest.get("controller_sha256") != expected:
            raise FrozenEvaluationError("frozen Skill Controller hash mismatch")
        return {**direct_manifest, "_manifest_path": str(direct)}
    paths = list((asset_root / "skills").glob("*/v*/manifest.json"))
    for path in paths:
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("controller_sha256") == expected:
            candidates.append((path, manifest))
    if not candidates:
        return None
    path, manifest = sorted(candidates, key=lambda item: str(item[0]))[-1]
    return {**manifest, "_manifest_path": str(path)}


class FrozenDependencyResolver:
    def __init__(self, *, skill: Mapping[str, Any] | None,
                 tool_library: CapabilityLibrary | None):
        self.skill = dict(skill or {})
        self.tool_library = tool_library

    @staticmethod
    def _native(adapter) -> dict[str, Mapping[str, Any]]:
        provider = getattr(adapter, "native_capability_manifest", None)
        value = provider() if callable(provider) else {}
        return {str(key): dict(item) for key, item in dict(value or {}).items()}

    def restore(self, adapter) -> dict[str, Any]:
        restored = []
        for dependency in self.skill.get("dependency_closure") or []:
            if self.tool_library is None:
                raise FrozenEvaluationError("frozen Skill shared Tool store is unavailable")
            expected = dict(dependency)
            tool_id = str(expected.get("tool_id") or "")
            try:
                inspected = self.tool_library.inspect(tool_id)
            except Exception as exc:
                raise FrozenEvaluationError(f"missing frozen Tool dependency: {tool_id}") from exc
            manifest = inspected["manifest"]
            checks = ("version", "source_sha256", "test_receipt_sha256", "runtime_environment")
            mismatched = [key for key in checks if manifest.get(key) != expected.get(key)]
            if manifest.get("status") != "promoted" or mismatched:
                raise FrozenEvaluationError(
                    f"frozen Tool dependency mismatch: {tool_id}: {mismatched or ['status']}")
            function = self.tool_library.runtime_function(tool_id)
            adapter.register_capability(tool_id, function, manifest)
            restored.append(tool_id)
        native = self._native(adapter)
        required_native = []
        requirements = self.skill.get("adapter_requirements") or {}
        for requirement in requirements.get("capabilities") or []:
            expected = dict(requirement)
            capability_id = str(expected.get("capability_id") or "")
            actual = native.get(capability_id)
            if actual is None:
                raise FrozenEvaluationError(f"missing Adapter-native capability: {capability_id}")
            for key in ("contract_sha256", "version"):
                if expected.get(key) is not None and actual.get(key) != expected.get(key):
                    raise FrozenEvaluationError(
                        f"Adapter-native capability mismatch: {capability_id}: {key}")
            required_native.append(capability_id)
        return {"shared_tools": restored, "adapter_capabilities": required_native}


class FrozenEvaluationRunner:
    """Run frozen cases without constructing a coding model or AgentLoop."""
    def __init__(self, *, cases: Sequence[tuple[str, Any]], runtime: Any,
                 controller: Path, expected_sha256: str,
                 resolver: FrozenDependencyResolver,
                 skill_id: str | None = None):
        self.cases = tuple(cases)
        self.runtime = runtime
        self.controller = Path(controller).resolve()
        self.expected_sha256 = str(expected_sha256)
        self.resolver = resolver
        self.skill_id = skill_id

    def run(self) -> dict[str, Any]:
        actual = _sha256(self.controller)
        if actual != self.expected_sha256:
            raise FrozenEvaluationError("frozen Controller SHA256 mismatch")
        policy_results = [{"name": "frozen_controller", "passed": True}]
        dependencies = []
        for _case_id, adapter in self.cases:
            dependencies.append(self.resolver.restore(adapter))
        policy_results.append({"name": "provenance", "passed": True,
                               "dependencies": dependencies})
        forbidden = {"reward", "done", "hidden_success", "benchmark_state", "state_id"}
        for _case_id, adapter in self.cases:
            index = getattr(adapter, "sdk_index", {}) or {}
            encoded = json.dumps(index, sort_keys=True, default=str).casefold()
            if any(token in encoded for token in forbidden):
                raise FrozenEvaluationError("anti-cheating boundary exposes private benchmark truth")
        policy_results.append({"name": "anti_cheating", "passed": True})
        if not self.cases:
            raise FrozenEvaluationError("sealed evaluation requires cases")
        case_ids = [str(case_id) for case_id, _adapter in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise FrozenEvaluationError("sealed case partition contains duplicate identities")
        policy_results.append({"name": "generalization", "passed": True,
                               "sealed_cases": len(self.cases)})
        target = _FrozenCases(self.cases)
        result = SealedEvaluationPolicy(name="sealed_evaluation").evaluate_frozen(
            adapter=target, runtime=self.runtime, controller=self.controller)
        if any(row.get("controller_sha256") != actual
               for row in result["sealed_evaluation_cases"]):
            raise FrozenEvaluationError("sealed execution used a different Controller")
        policy_results.append({"name": "sealed_evaluation",
                               "passed": result.get("evaluation_passed") is True})
        return {**result, "skill_id": self.skill_id,
                "evaluation_policies": policy_results,
                "finished": result.get("evaluation_passed") is True,
                "completion_valid": result.get("evaluation_passed") is True,
                "steps": 0, "executions": result.get("episodes", 0)}


class _FrozenCases:
    def __init__(self, cases):
        self._cases = tuple(cases)

    def case_adapters(self):
        return self._cases


__all__ = ["FrozenDependencyResolver", "FrozenEvaluationError",
           "FrozenEvaluationRunner", "load_frozen_skill"]
