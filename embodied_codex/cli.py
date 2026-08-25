"""Production RoboForge CLI backed by the canonical Kernel."""
from __future__ import annotations

import argparse
import importlib
import inspect
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from .adapters import load_adapter
from .kernel.assets import CapabilityGapLibrary, CapabilityLibrary, ExperienceLibrary, SkillLibrary
from .kernel.agent_loop import AgentLoop, LoopBudget
from .kernel.capability_manager import CapabilityManager
from .kernel.context import ContextBuilder
from .kernel.events import EventStore
from .kernel.runtime import ControllerRuntime
from .kernel.workspace import PersistentWorkspace
from .kernel.sandbox import default_sandbox
from .model import OpenAIModel


def _load(spec: str):
    module, separator, name = str(spec).partition(":")
    if not separator: raise ValueError(f"object spec must be package:object: {spec}")
    return getattr(importlib.import_module(module), name)


def _model(args):
    if args.model:
        factory = _load(args.model)
        return factory() if inspect.isclass(factory) else factory
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("APEX_API_KEY")
    if not key: raise SystemExit("set OPENAI_API_KEY/APEX_API_KEY or pass --model package:Model")
    return OpenAIModel(api_key=key, base_url=args.base_url, model=args.model_name,
                       reasoning_effort=args.reasoning_effort)


def _libraries(asset_root: Path, workspace: PersistentWorkspace, adapter=None):
    # A shared scope intentionally makes immutable tested assets reusable by independent runs.
    roots = [workspace.root, workspace.root.parent / "evidence"]
    if getattr(adapter, "artifact_dir", None):
        roots.append(Path(adapter.artifact_dir).resolve())
    tools = CapabilityLibrary(asset_root / "tools", workspace.root, python=sys.executable,
                              scope_id="shared01", allowed_input_roots=roots)
    return tools, SkillLibrary(asset_root / "skills"), ExperienceLibrary(asset_root / "experiences"), CapabilityGapLibrary(asset_root / "gaps")


def _default_asset_root() -> Path:
    configured = os.environ.get("ROBOFORGE_ASSET_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return (base / "roboforge" / "assets").resolve()


def _benchmark_policies():
    from evaluation.anti_cheating import AntiCheatingPolicy
    from evaluation.generalization import GeneralizationPolicy
    from evaluation.provenance import ProvenancePolicy
    from evaluation.sealed_evaluation import SealedEvaluationPolicy
    return [AntiCheatingPolicy(name="anti_cheating"), GeneralizationPolicy(name="generalization"),
            ProvenancePolicy(name="provenance"), SealedEvaluationPolicy(name="sealed_evaluation")]


def run_command(args) -> int:
    default_sandbox().require()
    run_dir = Path(args.run_dir or f"runs/roboforge/{args.profile}").resolve(); run_dir.mkdir(parents=True, exist_ok=True)
    asset_root = Path(args.asset_root).resolve() if args.asset_root else _default_asset_root()
    asset_root.mkdir(parents=True, exist_ok=True)
    source = Path(args.controller_source).resolve() if args.controller_source else None
    if source is not None and not source.is_file(): raise FileNotFoundError(source)
    results = []
    for state in list(args.states or [None]):
        adapter_spec = args.adapter
        case_root = run_dir if state is None else run_dir / f"state_{state}"
        case_root.mkdir(parents=True, exist_ok=True)
        workspace = PersistentWorkspace(case_root / "workspace")
        adapter = load_adapter(adapter_spec, task=str(args.task), run_dir=case_root, case=state)
        observe = getattr(adapter, "initial_observation", None)
        if not callable(observe):
            raise TypeError("Adapter must implement initial_observation()")
        initial_observation = observe()
        if source is not None: workspace.write_file("controller.py", source.read_text())
        tools, skills, experiences, gaps = _libraries(asset_root, workspace, adapter)
        manager = CapabilityManager(asset_root=asset_root, workspace=workspace, adapter=adapter,
                                   tool_library=tools, skill_library=skills,
                                   experience_library=experiences, gap_library=gaps)
        manager.bind_shared_tools()
        contract = getattr(adapter, "sdk_index", None) or getattr(adapter, "sdk_contract", None) or {
            "protocol": "adapter-provided", "operations": ["observe", "use", "act", "verify", "record"]}
        policies = _benchmark_policies() if args.profile == "benchmark" else []
        if args.frozen_controller:
            from evaluation.generalization import FrozenControllerPolicy
            if source is None:
                raise ValueError("--frozen-controller requires --controller-source")
            policies.insert(0, FrozenControllerPolicy(expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        loop = AgentLoop(model=_model(args), workspace=workspace, adapter=adapter,
            context_builder=ContextBuilder(adapter_index=contract, asset_registry=manager,
                workspace=workspace, initial_observation=initial_observation),
            capability_manager=manager, runtime=ControllerRuntime(timeout_seconds=args.controller_timeout),
            event_store=EventStore(case_root), budget=LoopBudget(max_steps=args.max_steps, max_executions=args.max_executions),
            root=case_root, web_search=manager.web_search, policies=policies, resume=True)
        try: result = loop.run(getattr(adapter, "instruction", str(args.task)))
        finally:
            close = getattr(adapter, "close", None)
            if callable(close): close()
        result["case"] = state; result["profile"] = args.profile
        result["evaluation_policies"] = [p.name for p in policies]; results.append(result)
    output = results[0] if len(results) == 1 else {
        "finished": (all(item.get("finished") and item.get("latest_evidence", {}).get(
            "verification_receipt", {}).get("verified") is True for item in results)
            and len({item.get("latest_evidence", {}).get("controller_sha256") for item in results}) == 1),
        "executions": sum(item.get("executions", 0) for item in results), "cases": results,
        "profile": args.profile,
        "cross_case_controller_sha256": results[0].get("latest_evidence", {}).get("controller_sha256")}
    result_path = run_dir / "result.json"; temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2, default=str) + "\n"); temporary.replace(result_path)
    benchmark_passed = (all(item.get("evaluation_passed") is True for item in results)
                        if args.profile == "benchmark" else True)
    print(json.dumps(output, indent=2, default=str))
    return 0 if output.get("finished") and benchmark_passed else 2


def doctor_command(args) -> int:
    sandbox_probe = default_sandbox().probe()
    checks = {"python": sys.executable, "sandbox": {
                  "backend": sandbox_probe.backend, "available": sandbox_probe.available,
                  "detail": sandbox_probe.detail}, "adapter": args.adapter,
              "api_key": bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("APEX_API_KEY")), "dependencies": {}}
    for dependency in ("jsonschema", "openai"):
        try: importlib.import_module(dependency); checks["dependencies"][dependency] = "available"
        except Exception as exc: checks["dependencies"][dependency] = f"unavailable: {exc}"
    smoke_dir = Path(args.run_dir or Path("runs/doctor") / args.adapter.replace(":", "_")); smoke_dir.mkdir(parents=True, exist_ok=True)
    adapter = None
    try:
        task = args.task if args.task is not None else ("0" if args.adapter == "libero"
            or str(args.adapter).startswith("libero@") else "doctor")
        adapter = load_adapter(args.adapter, task=str(task), run_dir=smoke_dir)
        checks["adapter_init"] = "available"
        required = ("dispatch", "project_rpc_output", "initial_observation", "sensor_report", "verification_receipt",
                    "execution_identity", "resume_protocol", "register_capability", "close")
        missing = [name for name in required if not callable(getattr(adapter, name, None))]
        if missing:
            raise RuntimeError(f"Adapter contract methods missing: {missing}")
        observation = adapter.initial_observation()
        if not isinstance(observation, dict):
            raise RuntimeError("Adapter initial_observation must return an object")
        checks["adapter_smoke"] = "available"
        workspace = PersistentWorkspace(smoke_dir / "workspace")
        workspace.write_file("controller.py", "def run(robot):\n    return robot.observe('proprioception', {})\n")
        execution = ControllerRuntime(timeout_seconds=20).execute(workspace.controller, adapter)
        checks["controller_runtime"] = "available" if execution.get("completed") is True else (
            "unavailable: " + str(execution.get("error") or "controller did not complete"))
        command = workspace.run_command([sys.executable, "-c",
            "import json; print(json.dumps({'workspace': True}))"], timeout_seconds=20)
        checks["command_smoke"] = ("available" if command.get("exit_code") == 0
                                   else "unavailable: " + str(command.get("output") or "command failed")[-1000:])
    except Exception as exc: checks["adapter_smoke"] = f"unavailable: {type(exc).__name__}: {exc}"
    finally:
        if adapter is not None:
            try: adapter.close()
            except Exception as exc: checks["adapter_close"] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        from .tool_runtime import ToolRuntime
        with tempfile.TemporaryDirectory(prefix="roboforge-doctor-") as temporary:
            bundle = Path(temporary)
            (bundle / "tool.py").write_text("def run(payload):\n    return {'echo': payload['value']}\n")
            (bundle / "manifest.json").write_text(json.dumps({
                "input_schema": {"type": "object", "properties": {"value": {"type": "integer"}},
                                  "required": ["value"], "additionalProperties": False},
                "output_schema": {"type": "object", "properties": {"echo": {"type": "integer"}},
                                   "required": ["echo"], "additionalProperties": False},
            }))
            result = ToolRuntime(python=sys.executable).execute(bundle, {"value": 7})
            if result == {"echo": 7}:
                checks["tool_runtime"] = "available"
            else:
                checks["tool_runtime"] = "unavailable: unexpected Tool result"
    except Exception as exc:
        checks["tool_runtime"] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        if args.model:
            model = _load(args.model)
            model = model() if inspect.isclass(model) else model
            response = model.decide(messages=[{"role": "user", "content": "Return finish."}],
                tools=[{"type": "function", "function": {"name": "finish", "description": "finish",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}])
            checks["model"] = "available" if isinstance(response, dict) else "unavailable: invalid response"
        elif checks["api_key"]:
            _model(argparse.Namespace(model=None, base_url=args.base_url, model_name=args.model_name,
                                      reasoning_effort="low"))
            checks["model"] = "available"
        else:
            checks["model"] = "unavailable: pass --model package:Model or configure an API key"
    except Exception as exc:
        checks["model"] = f"unavailable: {type(exc).__name__}: {exc}"
    if args.checkpoint:
        checks["checkpoint"] = "available" if Path(args.checkpoint).is_file() else "missing"
    checks["ok"] = bool(checks["sandbox"]["available"] and checks["adapter_smoke"] == "available"
                         and checks.get("controller_runtime") == "available"
                         and checks["command_smoke"] == "available"
                         and checks.get("tool_runtime") == "available"
                         and checks.get("model") == "available"
                         and all(value == "available" for value in checks["dependencies"].values())
                         and (not args.checkpoint or checks["checkpoint"] == "available"))
    print(json.dumps(checks, indent=2, default=str)); return 0 if checks["ok"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="roboforge"); sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run"); run.add_argument("--adapter", required=True); run.add_argument("--task", required=True)
    run.add_argument("--profile", choices=("dev", "autonomous", "benchmark"), default="dev")
    run.add_argument("--run-dir"); run.add_argument("--asset-root"); run.add_argument("--model"); run.add_argument("--model-name", default="gpt-5.6-sol")
    run.add_argument("--controller-source", help="load a frozen controller into the workspace before running")
    run.add_argument("--frozen-controller", action="store_true",
                     help="evaluation-owned immutable Controller mode")
    run.add_argument("--states", type=int, nargs="+", help="run the same Kernel over multiple Adapter cases")
    run.add_argument("--base-url", default=os.environ.get("APEX_BASE_URL", "https://api.apexin.ai/v1")); run.add_argument("--reasoning-effort", default="high")
    run.add_argument("--max-steps", type=int, default=60); run.add_argument("--max-executions", type=int, default=20); run.add_argument("--controller-timeout", type=float, default=600)
    run.set_defaults(handler=run_command)
    doctor = sub.add_parser("doctor"); doctor.add_argument("--adapter", required=True); doctor.add_argument("--task"); doctor.add_argument("--run-dir"); doctor.add_argument("--checkpoint")
    doctor.add_argument("--model"); doctor.add_argument("--model-name", default="gpt-5.6-sol")
    doctor.add_argument("--base-url", default=os.environ.get("APEX_BASE_URL", "https://api.apexin.ai/v1"))
    doctor.set_defaults(handler=doctor_command)
    args = parser.parse_args(argv); return args.handler(args)


if __name__ == "__main__": raise SystemExit(main())
