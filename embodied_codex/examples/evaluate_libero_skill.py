"""Evaluator-blind generalization test for a frozen Embodied Codex Skill.

This runner deliberately has no model and no writable coding workspace.  It
checks the frozen asset hashes, binds the public deployment-owned perception
Tools, loads the Skill's immutable analytic Tools, and executes the exact
frozen controller on fresh LIBERO initial states.  Only the sensor verifier is
reported; LIBERO reward, done, and task success are never exposed or saved.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import sys
import types
from typing import Any, Callable, Mapping

ROOT=Path(__file__).resolve().parents[2]

from embodied_codex.deployments import LiberoDeployment, LiberoEpisode
from embodied_codex.legacy.runtime import ControllerRuntime
from embodied_codex.tool_runtime import ToolRuntime


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    digest=hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode()+b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _load_function(path: Path) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    spec = importlib.util.spec_from_file_location(
        f"embodied_codex_frozen_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen Tool: {path}")
    module = importlib.util.module_from_spec(spec)
    previous=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode=previous
    function = getattr(module, "run", None)
    if not callable(function):
        raise RuntimeError(f"frozen Tool has no run(payload): {path}")
    return function


def _load_class(path: Path, name: str, *,
                relative_modules: Mapping[str, Path] | None = None):
    """Load a hashed frozen class, including declared sibling dependencies.

    Deployment Tools are frozen as individually hashed files.  Some older
    versions retain package-relative imports, so the evaluator provides only
    the explicitly named, independently hash-checked frozen dependency rather
    than falling back to a mutable development workspace.
    """
    suffix = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    package_name = f"embodied_codex_frozen_package_{suffix}"
    if relative_modules:
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package
        for module_name, dependency_path in relative_modules.items():
            dependency_spec = importlib.util.spec_from_file_location(
                f"{package_name}.{module_name}", dependency_path)
            if dependency_spec is None or dependency_spec.loader is None:
                raise RuntimeError(f"cannot load frozen dependency: {dependency_path}")
            dependency = importlib.util.module_from_spec(dependency_spec)
            sys.modules[dependency_spec.name] = dependency
            previous=sys.dont_write_bytecode
            try:
                sys.dont_write_bytecode=True
                dependency_spec.loader.exec_module(dependency)
            finally:
                sys.dont_write_bytecode=previous
        qualified_name = f"{package_name}.tool"
    else:
        qualified_name = f"embodied_codex_frozen_{name}_{suffix}"
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen deployment Tool: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    previous=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode=previous
    value = getattr(module, name, None)
    if not isinstance(value, type):
        raise RuntimeError(f"frozen Tool has no {name}: {path}")
    return value


def _relative_module_paths(item: Mapping[str, Any]) -> dict[str, Path]:
    folder=Path(item["folder"])
    return {str(module):folder/str(record["path"])
            for module,record in (item["manifest"].get("relative_modules") or {}).items()}


def _controller_capability_view(skill_manifest: Mapping[str,Any], frozen_tools,
                                capabilities):
    bindings=dict(skill_manifest.get("controller_tool_bindings") or
        {tool_id:tool_id for tool_id in skill_manifest.get("tool_ids") or []})
    expanded={};contracts={}
    for logical_id,packaged_id in bindings.items():
        if packaged_id not in frozen_tools or packaged_id not in capabilities:
            raise RuntimeError(f"invalid frozen Controller Tool binding: {logical_id} -> {packaged_id}")
        expanded[str(logical_id)]=capabilities[packaged_id]
        item=frozen_tools[packaged_id]
        contracts[str(logical_id)]={key:item["manifest"][key]
            for key in ("input_schema","output_schema")}
    return expanded,contracts


def inspect_skill(skill_dir: str | Path) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    """Verify all immutable hashes before a generalization rollout."""
    root = Path(skill_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    controller = root / "controller.py"
    if manifest.get("protocol") != "embodied-codex-skill-v1":
        raise RuntimeError("unsupported Skill protocol")
    if _sha256(controller) != manifest.get("controller_sha256"):
        raise RuntimeError("frozen controller hash mismatch")
    experience = root / "experience.json"
    if "experience_sha256" in manifest:
        if not experience.is_file() or _sha256(experience) != manifest["experience_sha256"]:
            raise RuntimeError("frozen experience hash mismatch")
    task_model=root/"task_model.json"
    if "task_model_sha256" in manifest:
        if not task_model.is_file() or _sha256(task_model)!=manifest["task_model_sha256"]:
            raise RuntimeError("frozen task model hash mismatch")
    for evidence in manifest.get("evidence_files") or []:
        source=root/str(evidence.get("path") or "")
        if not source.is_file() or _sha256(source)!=evidence.get("sha256"):
            raise RuntimeError("frozen Skill evidence hash mismatch")

    bindings=dict(manifest.get("controller_tool_bindings") or
        {tool_id:tool_id for tool_id in manifest.get("tool_ids") or []})
    if set(bindings.values())-set(manifest.get("tool_ids") or []):
        raise RuntimeError("invalid frozen Controller Tool bindings")

    tools: dict[str, dict[str, Any]] = {}
    bundle_hashes=dict(manifest.get("tool_bundle_sha256") or {})
    for tool_id in manifest.get("tool_ids") or []:
        folder = root / "tools" / str(tool_id).replace(":", "_")
        tool_manifest = json.loads((folder / "manifest.json").read_text())
        runtime_spec=dict(tool_manifest.get("runtime_spec") or {})
        source=(folder/"bundle"/str(runtime_spec.get("entrypoint"))
                if runtime_spec else folder/"tool.py")
        if tool_manifest.get("tool_id") != tool_id:
            raise RuntimeError(f"Tool id mismatch: {tool_id}")
        if tool_manifest.get("status") != "tested":
            raise RuntimeError(f"frozen Tool is not tested: {tool_id}")
        if tool_manifest.get("trained_on_current_task") is not False:
            raise RuntimeError(f"evaluated-task-trained Tool is forbidden: {tool_id}")
        if tool_manifest.get("privileged_state_used") is not False:
            raise RuntimeError(f"privileged Tool is forbidden: {tool_id}")
        if _sha256(source) != tool_manifest.get("source_sha256"):
            raise RuntimeError(f"frozen Tool hash mismatch: {tool_id}")
        for module,record in (tool_manifest.get("relative_modules") or {}).items():
            relative=Path(str(record.get("path") or ""))
            dependency=(folder/relative).resolve()
            if (not str(relative) or relative.is_absolute() or ".." in relative.parts
                    or dependency.parent!=folder or not dependency.is_file()
                    or _sha256(dependency)!=record.get("sha256")):
                raise RuntimeError(f"frozen Tool dependency hash mismatch: {tool_id}:{module}")
        if tool_id not in bundle_hashes or _tree_sha256(folder)!=bundle_hashes[tool_id]:
            raise RuntimeError(f"frozen Tool bundle hash mismatch: {tool_id}")
        tools[str(tool_id)] = {"folder": folder, "manifest": tool_manifest}
    return controller, manifest, tools


def _sensor_success(execution: Mapping[str, Any], sensor_report: Mapping[str, Any]) -> bool:
    events = list(execution.get("rpc_events") or [])
    last = events[-1] if events else {}
    result = execution.get("result")
    return bool(
        execution.get("completed") is True
        and isinstance(result, Mapping)
        and last.get("method") == "verify"
        and isinstance(last.get("result"), Mapping)
        and last["result"].get("verified") is True
        and sensor_report.get("sensor_verification_passed") is True
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--states", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--base-url", default=os.environ.get(
        "EMBODIED_CODEX_BASE_URL","https://api.apexin.ai/v1"))
    parser.add_argument("--config", default="config/standalone_libero")
    parser.add_argument("--python",default=sys.executable)
    parser.add_argument("--groundingdino-root",default=str(ROOT/"third_party"/"GroundingDINO"))
    parser.add_argument("--groundingdino-config",default=str(ROOT/"third_party"/"GroundingDINO"/
        "groundingdino"/"config"/"GroundingDINO_SwinT_OGC.py"))
    parser.add_argument("--groundingdino-checkpoint",default=os.environ.get(
        "EMBODIED_CODEX_GROUNDINGDINO_CHECKPOINT",str(ROOT/"checkpoints"/"groundingdino_swint_ogc.pth")))
    parser.add_argument("--sam-root",default=str(ROOT/"third_party"/"segment-anything"))
    parser.add_argument("--sam-checkpoint",default=str(ROOT/"checkpoints"/"sam_vit_b_01ec64.pth"))
    parser.add_argument("--graspnet-backend",default=str(ROOT/"embodied_codex"/"capabilities"/"graspnet_backend.py"))
    parser.add_argument("--graspnet-checkpoint",default=str(ROOT/"checkpoints"/"graspnet-checkpoint-rs.tar"))
    args = parser.parse_args()

    controller, skill_manifest, frozen_tools = inspect_skill(args.skill_dir)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    # The sensor verifiers and open-vocabulary detector share one public,
    # task-disjoint model instance.  It is needed even when perception is not
    # explicitly listed because every accepted Skill must end in verification.
    perception_item = next((item for item in frozen_tools.values()
                            if item["manifest"].get("name")=="open_vocab_rgbd_grounded_sam"),None)
    if perception_item is None:raise RuntimeError("frozen Skill lacks perception Tool")
    perception_class=_load_class(perception_item["folder"]/"tool.py","OpenVocabularyRGBD",
        relative_modules=_relative_module_paths(perception_item))
    perception = perception_class(
        groundingdino_root=args.groundingdino_root,
        groundingdino_config=args.groundingdino_config,
        groundingdino_checkpoint=args.groundingdino_checkpoint,
        sam_root=args.sam_root,sam_checkpoint=args.sam_checkpoint,
        device=args.device,
    )
    graspnet = None
    capabilities: dict[str, Callable[..., Any]] = {}
    tool_runtime=ToolRuntime(python=args.python,
                             allowed_input_roots=[output])
    for tool_id, item in frozen_tools.items():
        tool_manifest = item["manifest"]
        if tool_manifest.get("execution_owned_by_deployment"):
            name = tool_manifest.get("name")
            if name == "open_vocab_rgbd_grounded_sam":
                capabilities[tool_id] = perception.detect
            elif name == "graspnet_rgbd_6dof":
                if graspnet is None:
                    graspnet_class=_load_class(
                        item["folder"] / "tool.py", "GraspNetRGBD",
                        relative_modules=_relative_module_paths(item))
                    graspnet = graspnet_class(
                        backend_script=args.graspnet_backend,
                        checkpoint=args.graspnet_checkpoint,python=args.python,
                    )
                capabilities[tool_id] = graspnet.infer
            elif name == "vlm_visual_relation_grounder":
                grounder_class=_load_class(
                    item["folder"] / "tool.py", "VLMVisualRelationGrounder",
                    relative_modules=_relative_module_paths(item))
                grounder=grounder_class(
                    api_key=(os.environ.get("EMBODIED_CODEX_API_KEY") or
                             os.environ.get("OPENAI_API_KEY") or os.environ.get("APEX_API_KEY", "")),
                    base_url=args.base_url,model=args.model,
                    reasoning_effort=args.reasoning_effort)
                capabilities[tool_id]=grounder.select
            else:
                raise RuntimeError(f"LIBERO Adapter cannot bind deployment Tool: {name}")
        else:
            capabilities[tool_id] = (lambda payload,_folder=item["folder"]:
                                     tool_runtime.execute(_folder,payload))
    capabilities,capability_contracts=_controller_capability_view(
        skill_manifest,frozen_tools,capabilities)

    runtime = ControllerRuntime(python=args.python)
    records = []
    for state in args.states:
        episode_dir = output / f"state_{state:03d}"
        deployment = LiberoDeployment(
            episode=LiberoEpisode(args.suite, args.task, state, config_path=args.config),
            artifact_dir=episode_dir,
            capabilities=capabilities,
            capability_contracts=capability_contracts,
            verifiers={
                "visual_attachment": perception.verify_attachment,
                "visual_support_relation": perception.verify_support_relation,
            },
        )
        try:
            execution = runtime.execute(controller, deployment)
            sensor_report = dict(deployment.sensor_report(execution))
        finally:
            deployment.close()
        passed = _sensor_success(execution, sensor_report)
        record = {
            "state": state,
            "sensor_success": passed,
            "execution": execution,
            "sensor_report": sensor_report,
        }
        (episode_dir / "skill_execution.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n")
        records.append(record)

    summary = {
        "protocol": "embodied-codex-libero-skill-evaluation-v1",
        "skill_id": skill_manifest.get("skill_id"),
        "suite": args.suite,
        "task": args.task,
        "states": list(args.states),
        "sensor_successes": sum(bool(row["sensor_success"]) for row in records),
        "total": len(records),
        "evaluator_used": False,
        "results": [{"state": row["state"], "sensor_success": row["sensor_success"],
                     "verification": (row["execution"].get("result") or {}).get("verification")}
                    for row in records],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["sensor_successes"] == summary["total"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
