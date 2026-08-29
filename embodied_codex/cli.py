"""Production RoboForge CLI backed by the canonical Kernel."""
from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from .adapters import adapter_doctor_task, adapter_preflight, load_adapter
from .kernel.assets import CapabilityGapLibrary, CapabilityLibrary, ExperienceLibrary, SkillLibrary
from .kernel.agent_loop import AgentLoop, LoopBudget
from .kernel.capability_manager import CapabilityManager
from .kernel.campaign import CampaignAdapter, CampaignRunner
from .kernel.context import ContextBuilder
from .kernel.events import EventStore
from .kernel.runtime import ControllerRuntime
from .kernel.workspace import PersistentWorkspace
from .kernel.sandbox import select_sandbox
from .model import OpenAIModel
from .providers import ProviderConfigurationError, resolve_provider


def _load(spec: str):
    module, separator, name = str(spec).partition(":")
    if not separator: raise ValueError(f"object spec must be package:object: {spec}")
    return getattr(importlib.import_module(module), name)


def _model(args, configuration=None):
    if args.model:
        factory = _load(args.model)
        return factory() if inspect.isclass(factory) else factory
    configuration = configuration or resolve_provider(
        provider=getattr(args, "provider", None),
        base_url=getattr(args, "base_url", None))
    return OpenAIModel(api_key=configuration.api_key, base_url=configuration.endpoint,
                       model=args.model_name,
                       reasoning_effort=args.reasoning_effort,
                       provider=configuration.provider)


def _libraries(asset_root: Path, workspace: PersistentWorkspace, adapter=None, sandbox=None):
    # A shared scope intentionally makes immutable tested assets reusable by independent runs.
    workspace.add_protected_path(asset_root)
    tools = CapabilityLibrary(asset_root / "tools", workspace.root, python=sys.executable,
                              scope_id="shared01", sandbox=sandbox)
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


def _run_frozen_benchmark(args, *, sandbox, run_dir: Path, asset_root: Path,
                          source: Path) -> dict:
    """Evaluation-owned path: no coding model, AgentLoop, or model Tool registry."""
    from evaluation.frozen_runner import (FrozenDependencyResolver,
        FrozenEvaluationRunner, build_evaluation_bundle,
        load_evaluation_bundle, load_frozen_skill)
    cases = []
    try:
        states = list(args.states or [None])
        for index, state in enumerate(states, 1):
            case_root = run_dir / "sealed_cases" / f"case-{index:03d}"
            case_root.mkdir(parents=True, exist_ok=True)
            configuration = {"disable_agent_verifier": True}
            with contextlib.redirect_stdout(sys.stderr):
                adapter = load_adapter(args.adapter, task=str(args.task), run_dir=case_root,
                    case=state, configuration=configuration)
            cases.append((str(state) if state is not None else "default", adapter))
        skill = load_frozen_skill(source, asset_root)
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
        if skill is not None and skill.get("controller_sha256") != expected:
            raise RuntimeError("frozen Skill Controller hash mismatch")
        tools = CapabilityLibrary(asset_root / "tools", source.parent, python=sys.executable,
            scope_id="sealed", sandbox=sandbox)
        manifest_path = getattr(args, "evaluation_manifest", None)
        if manifest_path:
            bundle = load_evaluation_bundle(controller=source,
                manifest_path=manifest_path)
        else:
            development_root = getattr(args, "development_run_dir", None)
            development_cases = getattr(args, "development_cases", None)
            partition_protocol = getattr(args, "partition_protocol", None)
            if not development_root or not development_cases or not partition_protocol:
                raise RuntimeError(
                    "formal frozen evaluation requires an evaluation manifest or complete development provenance")
            evidence_paths = sorted(Path(development_root).resolve().glob("evidence/*.json"))
            native_manifests = []
            for _case_id, adapter in cases:
                provider = getattr(adapter, "native_capability_manifest", None)
                native_manifests.append(dict(provider() or {}) if callable(provider) else {})
            if any(value != native_manifests[0] for value in native_manifests[1:]):
                raise RuntimeError("sealed cases expose inconsistent Adapter-native capabilities")
            bundle = build_evaluation_bundle(controller=source,
                evidence_paths=evidence_paths, tool_library=tools,
                native_capabilities=native_manifests[0] if native_manifests else {},
                development_cases=development_cases,
                sealed_cases=[case_id for case_id, _adapter in cases],
                partition_protocol=partition_protocol,
                partition_seed=getattr(args, "partition_seed", None),
                destination=run_dir / "evaluation_manifest.json", skill=skill)
        resolver = FrozenDependencyResolver(bundle=bundle, tool_library=tools)
        runner = FrozenEvaluationRunner(cases=cases,
            runtime=ControllerRuntime(timeout_seconds=args.controller_timeout,
                sandbox=sandbox, protected_paths=[asset_root]),
            controller=source, expected_sha256=expected, resolver=resolver,
            bundle=bundle, skill_id=bundle.get("source_skill_id"))
        return runner.run()
    finally:
        for _case_id, adapter in cases:
            close = getattr(adapter, "close", None)
            if callable(close):
                close()


def run_command(args) -> int:
    sandbox = select_sandbox(args.sandbox)
    sandbox.require()
    if not getattr(sandbox, "safe", False) and args.profile != "dev":
        raise RuntimeError("unsafe sandbox is permitted only with --profile dev")
    with contextlib.redirect_stdout(sys.stderr):
        preflight = adapter_preflight(args.adapter)
    if preflight is not None and preflight.get("ok") is not True:
        raise RuntimeError("Adapter preflight failed: " + json.dumps(preflight, default=str))
    default_run_id = hashlib.sha256(json.dumps({"adapter": args.adapter,
        "task": str(args.task), "states": list(args.states or [])}, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = Path(args.run_dir or f"runs/roboforge/{args.profile}/{default_run_id}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    asset_root = Path(args.asset_root).resolve() if args.asset_root else _default_asset_root()
    asset_root.mkdir(parents=True, exist_ok=True)
    source = Path(args.controller_source).resolve() if args.controller_source else None
    if source is not None and not source.is_file(): raise FileNotFoundError(source)
    if args.profile == "benchmark" and args.frozen_controller:
        if source is None:
            raise ValueError("--frozen-controller requires --controller-source")
        output = _run_frozen_benchmark(args, sandbox=sandbox, run_dir=run_dir,
                                       asset_root=asset_root, source=source)
        output["profile"] = args.profile
        result_path = run_dir / "result.json"; temporary = result_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(output, indent=2, default=str) + "\n")
        temporary.replace(result_path)
        print(json.dumps(output, indent=2, default=str))
        return 0 if output.get("evaluation_passed") is True else 2
    provider_configuration = (None if args.model else resolve_provider(
        provider=args.provider, base_url=args.base_url))
    adapter_configuration = {
        "model_provider": (provider_configuration.provider
                           if provider_configuration is not None else None),
        "model_base_url": (provider_configuration.endpoint
                           if provider_configuration is not None else None),
        "verifier_provider": getattr(args, "verifier_provider", None),
        "verifier_base_url": getattr(args, "verifier_base_url", None),
        "verifier_model": getattr(args, "verifier_model_name", None) or args.model_name,
        "verifier_reasoning_effort": getattr(args, "verifier_reasoning_effort", "low")}
    workspace = PersistentWorkspace(run_dir / "workspace", sandbox=sandbox)
    adapter = None
    policies = _benchmark_policies() if args.profile == "benchmark" else []
    try:
        if args.states:
            cases = []
            try:
                for state in args.states:
                    case_root = run_dir / "cases" / f"state_{state}"
                    case_root.mkdir(parents=True, exist_ok=True)
                    with contextlib.redirect_stdout(sys.stderr):
                        case_adapter = load_adapter(args.adapter,
                            task=str(args.task), run_dir=case_root, case=state,
                            configuration=adapter_configuration)
                    cases.append((str(state), case_adapter))
            except Exception:
                for _case_id, case_adapter in cases:
                    close_case = getattr(case_adapter, "close", None)
                    if callable(close_case):
                        try:
                            close_case()
                        except Exception:
                            pass
                raise
            adapter = CampaignAdapter(cases)
            loop_type = CampaignRunner
        else:
            with contextlib.redirect_stdout(sys.stderr):
                adapter = load_adapter(args.adapter, task=str(args.task), run_dir=run_dir,
                                       configuration=adapter_configuration)
            loop_type = AgentLoop
        model = _model(args, provider_configuration)
        observe = getattr(adapter, "initial_observation", None)
        if not callable(observe):
            raise TypeError("Adapter must implement initial_observation()")
        with contextlib.redirect_stdout(sys.stderr):
            initial_observation = observe()
        if source is not None:
            workspace.write_file("controller.py", source.read_text())
        tools, skills, experiences, gaps = _libraries(asset_root, workspace, adapter, sandbox)
        manager = CapabilityManager(asset_root=asset_root, workspace=workspace, adapter=adapter,
                                   tool_library=tools, skill_library=skills,
                                   experience_library=experiences, gap_library=gaps)
        contract = getattr(adapter, "sdk_index", None) or getattr(adapter, "sdk_contract", None) or {
            "protocol": "adapter-provided", "operations": ["observe", "use", "act", "verify", "record"]}
        loop = loop_type(model=model, workspace=workspace, adapter=adapter,
            context_builder=ContextBuilder(adapter_index=contract, asset_registry=manager,
                workspace=workspace, initial_observation=initial_observation),
            capability_manager=manager, runtime=ControllerRuntime(
                timeout_seconds=args.controller_timeout, sandbox=sandbox,
                protected_paths=[asset_root]),
            event_store=EventStore(run_dir / "events", protect=True),
            budget=LoopBudget(max_steps=args.max_steps, max_executions=args.max_executions,
                              max_trials=args.max_trials,
                              timeout_seconds=args.session_timeout),
            root=run_dir, web_search=manager.web_search,
            resume=bool(getattr(args, "resume", False)))
        task = getattr(adapter, "instruction", str(args.task))
        if args.profile == "benchmark":
            from evaluation.runner import BenchmarkRunner
            output = BenchmarkRunner(loop, policies).run(task)
        else:
            output = loop.run(task)
    finally:
        close = getattr(adapter, "close", None) if adapter is not None else None
        if callable(close):
            close()
    output["profile"] = args.profile
    output["evaluation_policies"] = [policy.name for policy in policies]
    if (args.states and args.profile == "benchmark"
            and output.get("evaluation_passed") is True):
        output["cross_case_controller_sha256"] = ((output.get("latest_evidence") or {})
                                                   .get("controller_sha256"))
    result_path = run_dir / "result.json"; temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2, default=str) + "\n"); temporary.replace(result_path)
    benchmark_passed = (output.get("evaluation_passed") is True
                        if args.profile == "benchmark" else True)
    print(json.dumps(output, indent=2, default=str))
    return 0 if output.get("finished") and benchmark_passed else 2


def doctor_command(args) -> int:
    sandbox = select_sandbox(args.sandbox)
    sandbox_probe = sandbox.probe()
    checks = {"python": sys.executable, "sandbox": {
                  "backend": sandbox_probe.backend, "available": sandbox_probe.available,
                  "safe": bool(getattr(sandbox, "safe", False)),
                  "detail": sandbox_probe.detail, "features": dict(sandbox_probe.features)},
              "adapter": args.adapter,
              "model_provider": None, "dependencies": {}}
    if args.model:
        checks["model_provider"] = {"provider": "plugin", "endpoint": None,
                                    "key_env": None, "configured": True}
    else:
        try:
            provider = resolve_provider(provider=args.provider, base_url=args.base_url)
            checks["model_provider"] = provider.redacted()
        except ProviderConfigurationError as exc:
            checks["model_provider"] = {"provider": args.provider, "endpoint": args.base_url,
                "key_env": None, "configured": False, "error": str(exc)}
    required_dependencies = ["jsonschema"]
    if not args.model:
        required_dependencies.append("openai")
    for dependency in required_dependencies:
        try: importlib.import_module(dependency); checks["dependencies"][dependency] = "available"
        except Exception as exc: checks["dependencies"][dependency] = f"unavailable: {exc}"
    smoke_dir = Path(args.run_dir or Path("runs/doctor") / args.adapter.replace(":", "_")); smoke_dir.mkdir(parents=True, exist_ok=True)
    adapter = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            preflight = adapter_preflight(args.adapter)
        checks["adapter_preflight"] = preflight or {"ok": True, "provided": False}
        if preflight is not None and preflight.get("ok") is not True:
            raise RuntimeError("Adapter preflight failed")
        task = args.task if args.task is not None else (
            adapter_doctor_task(args.adapter) or "doctor")
        with contextlib.redirect_stdout(sys.stderr):
            adapter = load_adapter(args.adapter, task=str(task), run_dir=smoke_dir)
        checks["adapter_init"] = "available"
        required = ("dispatch", "project_rpc_output", "initial_observation", "sensor_report",
                    "agent_evidence", "verification_receipt",
                    "execution_identity", "resume_protocol", "register_capability", "close")
        missing = [name for name in required if not callable(getattr(adapter, name, None))]
        if missing:
            raise RuntimeError(f"Adapter contract methods missing: {missing}")
        with contextlib.redirect_stdout(sys.stderr):
            observation = adapter.initial_observation()
        if not isinstance(observation, dict):
            raise RuntimeError("Adapter initial_observation must return an object")
        checks["adapter_smoke"] = "available"
        workspace = PersistentWorkspace(smoke_dir / "workspace", sandbox=sandbox)
        workspace.write_file("controller.py", "def run(robot):\n    return robot.observe('proprioception', {})\n")
        execution = ControllerRuntime(timeout_seconds=20, sandbox=sandbox).execute(
            workspace.controller, adapter)
        checks["controller_runtime"] = "available" if execution.get("completed") is True else (
            "unavailable: " + str(execution.get("error") or "controller did not complete"))
        command = workspace.run_command([sys.executable, "-c",
            "import json; print(json.dumps({'workspace': True}))"], timeout_seconds=20)
        checks["command_smoke"] = ("available" if command.get("exit_code") == 0
                                   else "unavailable: " + str(command.get("output") or "command failed")[-1000:])
    except Exception as exc:
        if "adapter_preflight" not in checks:
            checks["adapter_preflight"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        checks["adapter_smoke"] = f"unavailable: {type(exc).__name__}: {exc}"
    finally:
        if adapter is not None:
            try: adapter.close()
            except Exception as exc: checks["adapter_close"] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        from .tool_runtime import ToolRuntime
        from .kernel.cas import ContentAddressedStore
        from .kernel.runtime_environment import RuntimeEnvironmentManager
        with tempfile.TemporaryDirectory(prefix="roboforge-doctor-") as temporary:
            bundle = Path(temporary)
            (bundle / "tool.py").write_text("def run(payload):\n    return {'echo': payload['value']}\n")
            (bundle / "manifest.json").write_text(json.dumps({
                "input_schema": {"type": "object", "properties": {"value": {"type": "integer"}},
                                  "required": ["value"], "additionalProperties": False},
                "output_schema": {"type": "object", "properties": {"echo": {"type": "integer"}},
                                   "required": ["echo"], "additionalProperties": False},
            }))
            cas = ContentAddressedStore(bundle / "cas")
            runtime_manager = RuntimeEnvironmentManager(bundle / "runtimes", cas=cas,
                python=sys.executable, sandbox=sandbox)
            sealed = runtime_manager.seal(runtime_manager.default_spec(),
                                          workspace_root=bundle)
            environment = runtime_manager.ensure(sealed)
            result = ToolRuntime(python=sys.executable, sandbox=sandbox).execute(
                bundle, {"value": 7}, python=environment.python)
            if result == {"echo": 7}:
                checks["tool_runtime"] = {"status": "available",
                    "runtime_id": environment.runtime_id,
                    "isolated_python": str(environment.python),
                    "reused": environment.reused}
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
        elif checks["model_provider"].get("configured"):
            model = _model(argparse.Namespace(model=None, base_url=args.base_url,
                provider=args.provider, model_name=args.model_name, reasoning_effort="low"))
            response = model.decide(messages=[{"role": "user",
                "content": "Call the finish function with an empty object."}], tools=[{
                    "type": "function", "function": {"name": "finish",
                    "description": "model API smoke test", "parameters": {
                        "type": "object", "properties": {},
                        "additionalProperties": False}}}])
            calls = response.get("tool_calls") if isinstance(response, dict) else None
            checks["model"] = "available" if isinstance(calls, list) and calls else (
                "unavailable: model API did not return a function call")
        else:
            checks["model"] = "unavailable: pass --model package:Model or configure an API key"
    except Exception as exc:
        checks["model"] = f"unavailable: {type(exc).__name__}: {exc}"
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            checks["checkpoint"] = {"available": False, "reason": "file is missing"}
        else:
            digest = hashlib.sha256()
            with checkpoint.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            digest = digest.hexdigest()
            expected = str(args.checkpoint_sha256 or "").casefold()
            valid = not expected or digest == expected
            checks["checkpoint"] = {"available": valid, "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size, "sha256": digest,
                "expected_sha256": expected or None,
                "reason": None if valid else "checksum mismatch"}
    sandbox_features = checks["sandbox"]["features"]
    checks["ok"] = bool(checks["sandbox"]["available"] and checks["sandbox"]["safe"]
                         and sandbox_features.get("filesystem_isolation") is True
                         and sandbox_features.get("unauthorized_read_denied") is True
                         and sandbox_features.get("unauthorized_write_denied") is True
                         and checks.get("adapter_smoke") == "available"
                         and checks.get("controller_runtime") == "available"
                         and checks.get("command_smoke") == "available"
                         and (checks.get("tool_runtime") == "available"
                              or (isinstance(checks.get("tool_runtime"), dict)
                                  and checks["tool_runtime"].get("status") == "available"))
                         and checks.get("model") == "available"
                         and all(value == "available" for value in checks["dependencies"].values())
                         and (not args.checkpoint or checks["checkpoint"]["available"] is True))
    print(json.dumps(checks, indent=2, default=str)); return 0 if checks["ok"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="roboforge"); sub = parser.add_subparsers(dest="command", required=True)
    def add_run_arguments(command, *, resume: bool):
        command.add_argument("--adapter", required=True); command.add_argument("--task", required=True)
        command.add_argument("--profile", choices=("dev", "autonomous", "benchmark"), default="dev")
        command.add_argument("--sandbox", choices=("auto", "posix", "bubblewrap", "unsafe"), default="auto")
        command.add_argument("--run-dir"); command.add_argument("--asset-root")
        command.add_argument("--model"); command.add_argument("--model-name", default="gpt-5.6-sol")
        command.add_argument("--controller-source", help="load a frozen controller into the workspace before running")
        command.add_argument("--frozen-controller", action="store_true",
                             help="evaluation-owned immutable Controller mode")
        command.add_argument("--evaluation-manifest",
                             help="immutable evaluator-owned frozen bundle")
        command.add_argument("--development-run-dir",
                             help="private development run used to package frozen provenance")
        command.add_argument("--development-cases", nargs="+",
                             help="private development partition identities")
        command.add_argument("--partition-protocol")
        command.add_argument("--partition-seed")
        command.add_argument("--states", type=int, nargs="+", help="run the same Kernel over multiple Adapter cases")
        command.add_argument("--provider", choices=("openai", "apex")); command.add_argument("--base-url")
        command.add_argument("--reasoning-effort", default="high")
        command.add_argument("--verifier-provider", choices=("openai", "apex"))
        command.add_argument("--verifier-base-url")
        command.add_argument("--verifier-model-name")
        command.add_argument("--verifier-reasoning-effort", default="low")
        command.add_argument("--max-steps", type=int, default=60)
        command.add_argument("--max-executions", type=int, default=20)
        command.add_argument("--max-trials", type=int,
                             help="maximum fresh physical Controller trials")
        command.add_argument("--session-timeout", type=float, default=3600)
        command.add_argument("--controller-timeout", type=float, default=600)
        command.set_defaults(handler=run_command, resume=resume)
    # A run directory is resumable by default. ``resume`` remains an explicit,
    # readable alias for automation and older shell scripts.
    add_run_arguments(sub.add_parser("run"), resume=True)
    add_run_arguments(sub.add_parser("resume"), resume=True)
    doctor = sub.add_parser("doctor"); doctor.add_argument("--adapter", required=True); doctor.add_argument("--task"); doctor.add_argument("--run-dir"); doctor.add_argument("--checkpoint")
    doctor.add_argument("--sandbox", choices=("auto", "posix", "bubblewrap", "unsafe"), default="auto")
    doctor.add_argument("--checkpoint-sha256")
    doctor.add_argument("--model"); doctor.add_argument("--model-name", default="gpt-5.6-sol")
    doctor.add_argument("--provider", choices=("openai", "apex")); doctor.add_argument("--base-url")
    doctor.set_defaults(handler=doctor_command)
    args = parser.parse_args(argv); return args.handler(args)


if __name__ == "__main__": raise SystemExit(main())
