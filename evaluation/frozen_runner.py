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


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"),
                         default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tool_dependency(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: manifest.get(key) for key in (
        "tool_id", "version", "source_sha256", "test_receipt_sha256",
        "runtime_environment")}


def _development_record(path: Path, controller_sha256: str) -> tuple[dict[str, Any], set[str]] | None:
    try:
        evidence = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenEvaluationError(f"development evidence is invalid: {path}") from exc
    identity = evidence.get("environment_identity")
    receipt = evidence.get("verification_receipt")
    execution = evidence.get("execution")
    if (evidence.get("controller_sha256") != controller_sha256
            or not isinstance(receipt, Mapping)
            or receipt.get("verified") is not True
            or not isinstance(execution, Mapping)
            or execution.get("completed") is not True
            or execution.get("error")):
        return None
    authentic = (isinstance(identity, Mapping) and isinstance(receipt, Mapping)
        and isinstance(execution, Mapping)
        and evidence.get("controller_sha256") == controller_sha256
        and receipt.get("controller_sha256") == controller_sha256
        and receipt.get("environment_identity") == identity
        and receipt.get("episode_id") == identity.get("episode_id")
        and receipt.get("environment_generation") == identity.get("environment_generation")
        and receipt.get("verified") is True
        and execution.get("completed") is True and not execution.get("error"))
    if not authentic:
        raise FrozenEvaluationError(
            f"development evidence is not an authentic successful frozen-Controller execution: {path}")
    observed = set()
    for event in execution.get("rpc_events") or []:
        if isinstance(event, Mapping) and event.get("method") == "use":
            tool_id = (event.get("arguments") or {}).get("tool_id")
            if tool_id:
                observed.add(str(tool_id))
    receipt_summary = {key: receipt.get(key) for key in (
        "verified", "controller_sha256", "episode_id", "environment_generation")}
    return ({"sha256": _sha256(path), "controller_sha256": controller_sha256,
             "verification_receipt_sha256": _canonical_sha256(dict(receipt)),
             "receipt": receipt_summary}, observed)


def build_evaluation_bundle(*, controller: str | Path,
                            evidence_paths: Sequence[str | Path],
                            tool_library: CapabilityLibrary | Any | None,
                            native_capabilities: Mapping[str, Mapping[str, Any]],
                            development_cases: Sequence[Any], sealed_cases: Sequence[Any],
                            partition_protocol: str, partition_seed: Any,
                            destination: str | Path,
                            skill: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Package development provenance for evaluator-owned frozen execution."""
    controller_path = Path(controller).resolve()
    if not controller_path.is_file():
        raise FrozenEvaluationError("frozen Controller does not exist")
    controller_sha256 = _sha256(controller_path)
    provenance = []
    observed: set[str] = set()
    for value in evidence_paths:
        audited = _development_record(Path(value).resolve(), controller_sha256)
        if audited is None:
            continue
        record, dependencies = audited
        provenance.append(record)
        observed.update(dependencies)
    if not provenance:
        raise FrozenEvaluationError("evaluation bundle requires development evidence")

    skill_manifest = dict(skill or {})
    if skill_manifest and skill_manifest.get("controller_sha256") != controller_sha256:
        raise FrozenEvaluationError("frozen Skill Controller SHA256 mismatch")
    declared_observed = (set(skill_manifest.get("observed_tool_ids") or [])
                         if skill_manifest else set(observed))
    if declared_observed != observed:
        raise FrozenEvaluationError(
            "frozen Skill dependency audit differs from development execution evidence")

    native = {str(key): dict(value) for key, value in dict(native_capabilities).items()}
    dependency_closure = []
    native_requirements = []
    for tool_id in sorted(observed):
        if tool_id in native:
            requirement = dict(native[tool_id])
            if requirement.get("capability_id") != tool_id:
                raise FrozenEvaluationError(f"invalid Adapter-native capability manifest: {tool_id}")
            native_requirements.append(requirement)
            continue
        if tool_library is None:
            raise FrozenEvaluationError(f"observed dependency cannot be classified: {tool_id}")
        try:
            manifest = tool_library.inspect(tool_id)["manifest"]
        except Exception as exc:
            raise FrozenEvaluationError(
                f"observed dependency cannot be classified or restored: {tool_id}") from exc
        if manifest.get("status") != "promoted":
            raise FrozenEvaluationError(f"observed shared Tool is not promoted: {tool_id}")
        dependency = _tool_dependency(manifest)
        if not all(dependency.get(key) is not None for key in (
                "tool_id", "version", "source_sha256", "test_receipt_sha256")):
            raise FrozenEvaluationError(f"shared Tool provenance is incomplete: {tool_id}")
        dependency_closure.append(dependency)

    skill_shared = {_tool_dependency(row)["tool_id"]: _tool_dependency(row)
                    for row in skill_manifest.get("dependency_closure") or []}
    generated_shared = {row["tool_id"]: row for row in dependency_closure}
    skill_native = {row.get("capability_id"): dict(row) for row in
                    (skill_manifest.get("adapter_requirements") or {}).get("capabilities") or []}
    generated_native = {row["capability_id"]: row for row in native_requirements}
    if skill_manifest and (skill_shared != generated_shared or skill_native != generated_native):
        raise FrozenEvaluationError("frozen Skill dependency closure does not match audited evidence")

    partition = {"protocol": str(partition_protocol), "seed": partition_seed,
                 "development_cases": [str(value) for value in development_cases],
                 "sealed_cases": [str(value) for value in sealed_cases]}
    partition["digest"] = _canonical_sha256(partition)
    payload = {"protocol": "roboforge-evaluation-bundle-v1",
        "controller_sha256": controller_sha256,
        "observed_tool_dependencies": sorted(observed),
        "dependency_closure": dependency_closure,
        "adapter_requirements": {"capabilities": native_requirements},
        "development_provenance": provenance,
        "development_partition": partition,
        "source_skill_id": skill_manifest.get("skill_id")}
    payload["bundle_sha256"] = _canonical_sha256(payload)
    target = Path(destination).resolve()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists():
        try:
            current = load_evaluation_bundle(controller=controller_path,
                                             manifest_path=target)
        except FrozenEvaluationError as exc:
            raise FrozenEvaluationError("existing evaluation manifest is invalid") from exc
        if current != payload:
            raise FrozenEvaluationError("evaluation manifest is immutable")
        return current
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(target)
    return payload


def validate_evaluation_bundle(payload: Mapping[str, Any],
                               controller_sha256: str) -> dict[str, Any]:
    result = dict(payload)
    if result.get("protocol") != "roboforge-evaluation-bundle-v1":
        raise FrozenEvaluationError("unsupported evaluation bundle")
    recorded = result.get("bundle_sha256")
    unsigned = {key: value for key, value in result.items() if key != "bundle_sha256"}
    if recorded != _canonical_sha256(unsigned):
        raise FrozenEvaluationError("evaluation bundle integrity check failed")
    if result.get("controller_sha256") != controller_sha256:
        raise FrozenEvaluationError("evaluation bundle Controller SHA256 mismatch")
    provenance = result.get("development_provenance")
    if not isinstance(provenance, list) or not provenance:
        raise FrozenEvaluationError("evaluation bundle has no development provenance")
    for record in provenance:
        if (not isinstance(record, Mapping)
                or not str(record.get("sha256") or "")
                or record.get("controller_sha256") not in {None, controller_sha256}):
            raise FrozenEvaluationError("evaluation bundle development provenance is incomplete")
    partition = result.get("development_partition")
    if not isinstance(partition, Mapping):
        raise FrozenEvaluationError("evaluation bundle has no partition manifest")
    partition_value = {key: partition.get(key) for key in (
        "protocol", "seed", "development_cases", "sealed_cases")}
    if (not partition_value["protocol"]
            or not isinstance(partition_value["development_cases"], list)
            or not isinstance(partition_value["sealed_cases"], list)
            or not partition_value["development_cases"]
            or not partition_value["sealed_cases"]
            or partition.get("digest") != _canonical_sha256(partition_value)):
        raise FrozenEvaluationError("evaluation partition digest mismatch")
    shared = {str(row.get("tool_id")) for row in result.get("dependency_closure") or []
              if isinstance(row, Mapping) and row.get("tool_id")}
    native = {str(row.get("capability_id")) for row in
              (result.get("adapter_requirements") or {}).get("capabilities") or []
              if isinstance(row, Mapping) and row.get("capability_id")}
    if shared | native != set(result.get("observed_tool_dependencies") or []):
        raise FrozenEvaluationError("evaluation bundle dependency classification is incomplete")
    return result


def load_evaluation_bundle(*, controller: str | Path,
                           manifest_path: str | Path | None) -> dict[str, Any]:
    if manifest_path is None:
        raise FrozenEvaluationError("formal frozen evaluation requires an evaluation manifest")
    path = Path(manifest_path).resolve()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenEvaluationError("evaluation manifest is missing or invalid") from exc
    if not isinstance(payload, dict):
        raise FrozenEvaluationError("unsupported evaluation bundle")
    return validate_evaluation_bundle(payload, _sha256(Path(controller).resolve()))


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
    def __init__(self, *, bundle: Mapping[str, Any] | None = None,
                 skill: Mapping[str, Any] | None = None,
                 tool_library: CapabilityLibrary | None):
        self.bundle = dict(bundle or skill or {})
        self.tool_library = tool_library

    @staticmethod
    def _native(adapter) -> dict[str, Mapping[str, Any]]:
        provider = getattr(adapter, "native_capability_manifest", None)
        value = provider() if callable(provider) else {}
        return {str(key): dict(item) for key, item in dict(value or {}).items()}

    def restore(self, adapter) -> dict[str, Any]:
        restored = []
        for dependency in self.bundle.get("dependency_closure") or []:
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
            function = self.tool_library.runtime_function(tool_id,
                artifact_resolver=getattr(adapter, "resolve_controller_artifact", None))
            adapter.register_capability(tool_id, function, manifest)
            restored.append(tool_id)
        native = self._native(adapter)
        required_native = []
        requirements = self.bundle.get("adapter_requirements") or {}
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
                 bundle: Mapping[str, Any] | None,
                 skill_id: str | None = None):
        self.cases = tuple(cases)
        self.runtime = runtime
        self.controller = Path(controller).resolve()
        self.expected_sha256 = str(expected_sha256)
        self.resolver = resolver
        self.bundle = dict(bundle or {})
        self.skill_id = skill_id

    def run(self) -> dict[str, Any]:
        if not self.bundle:
            raise FrozenEvaluationError("formal frozen evaluation requires a verified evaluation bundle")
        actual = _sha256(self.controller)
        if actual != self.expected_sha256:
            raise FrozenEvaluationError("frozen Controller SHA256 mismatch")
        self.bundle = validate_evaluation_bundle(self.bundle, actual)
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
        partition = self.bundle.get("development_partition") or {}
        development_cases = set(str(value) for value in partition.get("development_cases") or [])
        sealed_cases = set(str(value) for value in partition.get("sealed_cases") or [])
        overlap = development_cases & sealed_cases
        if overlap:
            raise FrozenEvaluationError(
                f"development and sealed partitions overlap: {sorted(overlap)}")
        if set(case_ids) != sealed_cases:
            raise FrozenEvaluationError("runtime sealed cases do not match the evaluation bundle")
        policy_results.append({"name": "generalization", "passed": True,
                               "development_cases": len(development_cases),
                               "sealed_cases": len(self.cases),
                               "partition_digest": partition.get("digest")})
        target = _FrozenCases(self.cases)
        result = SealedEvaluationPolicy(name="sealed_evaluation").evaluate_frozen(
            adapter=target, runtime=self.runtime, controller=self.controller)
        if any(row.get("controller_sha256") != actual
               for row in result["sealed_evaluation_cases"]):
            raise FrozenEvaluationError("sealed execution used a different Controller")
        policy_results.append({"name": "sealed_evaluation",
                               "passed": result.get("evaluation_passed") is True})
        return {**result, "skill_id": self.skill_id,
                "evaluation_bundle_sha256": self.bundle.get("bundle_sha256"),
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
           "FrozenEvaluationRunner", "build_evaluation_bundle",
           "load_evaluation_bundle", "load_frozen_skill",
           "validate_evaluation_bundle"]
