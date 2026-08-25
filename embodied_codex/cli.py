"""Production RoboForge CLI backed by the canonical Kernel."""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .adapters import load_adapter
from .assets import CapabilityGapLibrary, CapabilityLibrary, ExperienceLibrary, SkillLibrary
from .kernel.agent_loop import AgentLoop, LoopBudget
from .kernel.capability_manager import CapabilityManager
from .kernel.context import ContextBuilder
from .kernel.events import EventStore
from .kernel.runtime import ControllerRuntime
from .kernel.workspace import PersistentWorkspace
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


def _libraries(asset_root: Path, workspace: PersistentWorkspace):
    # A shared scope intentionally makes immutable tested assets reusable by independent runs.
    tools = CapabilityLibrary(asset_root / "tools", workspace.root, python=sys.executable, scope_id="shared01")
    return tools, SkillLibrary(asset_root / "skills"), ExperienceLibrary(asset_root / "experiences"), CapabilityGapLibrary(asset_root / "gaps")


def _benchmark_policies():
    from evaluation.anti_cheating import AntiCheatingPolicy
    from evaluation.generalization import GeneralizationPolicy
    from evaluation.provenance import ProvenancePolicy
    from evaluation.sealed_evaluation import SealedEvaluationPolicy
    return [AntiCheatingPolicy(name="anti_cheating"), GeneralizationPolicy(name="generalization"),
            ProvenancePolicy(name="provenance"), SealedEvaluationPolicy(name="sealed_evaluation")]


def run_command(args) -> int:
    run_dir = Path(args.run_dir or f"runs/roboforge/{args.profile}").resolve(); run_dir.mkdir(parents=True, exist_ok=True)
    asset_root = Path(args.asset_root or (run_dir / "assets")).resolve(); asset_root.mkdir(parents=True, exist_ok=True)
    workspace = PersistentWorkspace(run_dir / "workspace")
    adapter = load_adapter(args.adapter, task=str(args.task), run_dir=run_dir)
    tools, skills, experiences, gaps = _libraries(asset_root, workspace)
    manager = CapabilityManager(asset_root=asset_root, workspace=workspace, adapter=adapter,
                               tool_library=tools, skill_library=skills,
                               experience_library=experiences, gap_library=gaps)
    manager.bind_shared_tools()
    contract = getattr(adapter, "sdk_index", None) or getattr(adapter, "sdk_contract", None) or {
        "protocol": "adapter-provided", "operations": ["observe", "use", "act", "verify", "record"]}
    policies = _benchmark_policies() if args.profile == "benchmark" else []
    loop = AgentLoop(model=_model(args), workspace=workspace, adapter=adapter,
        context_builder=ContextBuilder(adapter_index=contract, asset_registry=manager, workspace=workspace),
        capability_manager=manager, runtime=ControllerRuntime(timeout_seconds=args.controller_timeout),
        event_store=EventStore(run_dir), budget=LoopBudget(max_steps=args.max_steps, max_executions=args.max_executions),
        root=run_dir, web_search=manager.web_search, policies=policies, resume=True)
    try:
        result = loop.run(getattr(adapter, "instruction", str(args.task)))
    finally:
        close = getattr(adapter, "close", None)
        if callable(close): close()
    result["profile"] = args.profile; result["evaluation_policies"] = [p.name for p in policies]
    print(json.dumps(result, indent=2, default=str)); return 0 if result.get("finished") else 2


def doctor_command(args) -> int:
    checks = {"python": sys.executable, "sandbox": shutil.which("bwrap"), "adapter": args.adapter,
              "api_key": bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("APEX_API_KEY")), "dependencies": {}}
    for dependency in ("jsonschema", "openai"):
        try: importlib.import_module(dependency); checks["dependencies"][dependency] = "available"
        except Exception as exc: checks["dependencies"][dependency] = f"unavailable: {exc}"
    smoke_dir = Path(args.run_dir or Path("runs/doctor") / args.adapter.replace(":", "_")); smoke_dir.mkdir(parents=True, exist_ok=True)
    try:
        adapter = load_adapter(args.adapter, task=str(args.task), run_dir=smoke_dir)
        checks["adapter_init"] = "available"
        observation = adapter.dispatch("observe", {"channel": "proprioception", "request": {}})
        adapter.project_rpc_output("observe", {"channel": "proprioception", "request": {}}, observation)
        checks["adapter_smoke"] = "available"
        adapter.close()
    except Exception as exc: checks["adapter_smoke"] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        completed = subprocess.run([sys.executable, "-c", "print('roboforge-doctor')"], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        checks["command_smoke"] = "available" if completed.returncode == 0 else completed.stdout[-1000:]
    except Exception as exc: checks["command_smoke"] = f"unavailable: {exc}"
    if args.checkpoint:
        checks["checkpoint"] = "available" if Path(args.checkpoint).is_file() else "missing"
    checks["ok"] = bool(checks["sandbox"] and checks["adapter_smoke"] == "available"
                         and checks["command_smoke"] == "available"
                         and all(value == "available" for value in checks["dependencies"].values())
                         and (not args.checkpoint or checks["checkpoint"] == "available"))
    print(json.dumps(checks, indent=2, default=str)); return 0 if checks["ok"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="roboforge"); sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run"); run.add_argument("--adapter", required=True); run.add_argument("--task", required=True)
    run.add_argument("--profile", choices=("dev", "autonomous", "benchmark"), default="dev")
    run.add_argument("--run-dir"); run.add_argument("--asset-root"); run.add_argument("--model"); run.add_argument("--model-name", default="gpt-5.6-sol")
    run.add_argument("--base-url", default=os.environ.get("APEX_BASE_URL", "https://api.apexin.ai/v1")); run.add_argument("--reasoning-effort", default="high")
    run.add_argument("--max-steps", type=int, default=60); run.add_argument("--max-executions", type=int, default=20); run.add_argument("--controller-timeout", type=float, default=600)
    run.set_defaults(handler=run_command)
    doctor = sub.add_parser("doctor"); doctor.add_argument("--adapter", required=True); doctor.add_argument("--task", default="doctor"); doctor.add_argument("--run-dir"); doctor.add_argument("--checkpoint")
    doctor.set_defaults(handler=doctor_command)
    args = parser.parse_args(argv); return args.handler(args)


if __name__ == "__main__": raise SystemExit(main())
