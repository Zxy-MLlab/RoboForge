"""Unified RoboForge command line entry points."""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
from pathlib import Path
import sys
from typing import Any

from .assets import CapabilityGapLibrary, CapabilityLibrary, ExperienceLibrary, SkillLibrary
from .kernel.agent_loop import AgentLoop, LoopBudget
from .kernel.assets import AssetRegistry
from .kernel.context import ContextBuilder
from .kernel.events import EventStore
from .kernel.runtime import ControllerRuntime
from .kernel.workspace import PersistentWorkspace
from .web import search_web


def _load(spec: str):
    module, separator, name = str(spec).partition(":")
    if not separator: raise ValueError(f"object spec must be package:object: {spec}")
    return getattr(importlib.import_module(module), name)


def _make_adapter(spec: str, task: str, run_dir: Path):
    if spec == "libero":
        from .deployments.libero import LiberoDeployment, LiberoEpisode
        episode = LiberoEpisode("libero_spatial", int(task), 0,
                                config_path=os.environ.get("LIBERO_CONFIG_PATH"))
        return LiberoDeployment(episode=episode, artifact_dir=run_dir / "adapter")
    factory = _load(spec)
    if not inspect.isclass(factory) and not callable(factory): return factory
    attempts = ({"task": task, "root": run_dir}, {"task": task}, {"instruction": task}, {})
    for kwargs in attempts:
        try: return factory(**kwargs)
        except TypeError: continue
    return factory(task)


class _ModelAgent:
    def __init__(self, model): self.model = model
    def step(self, context):
        if hasattr(self.model, "step"): return self.model.step(context)
        response = self.model.decide(messages=[
            {"role": "system", "content": context["system"]},
            {"role": "user", "content": __import__("json").dumps(context, default=str)},
        ], tools=[])
        content = response.get("content") if isinstance(response, dict) else response
        try: return __import__("json").loads(content)
        except Exception: return {"message": str(content), "finishes": True}


def run_command(args) -> int:
    run_dir = Path(args.run_dir or f"runs/roboforge/{args.profile}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = PersistentWorkspace(run_dir / "workspace")
    adapter = _make_adapter(args.adapter, str(args.task), run_dir)
    assets_root = run_dir / "assets"
    tools = CapabilityLibrary(assets_root / "tools", workspace.root, python=sys.executable)
    registry = AssetRegistry(tools=tools, skills=SkillLibrary(assets_root / "skills"),
                             experiences=ExperienceLibrary(assets_root / "experiences"),
                             gaps=CapabilityGapLibrary(assets_root / "gaps"))
    if args.model:
        model_factory = _load(args.model)
        model = model_factory() if inspect.isclass(model_factory) else model_factory
    else:
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("APEX_API_KEY")
        if not key: raise SystemExit("set OPENAI_API_KEY/APEX_API_KEY or pass --model package:Model")
        from .model import OpenAIModel
        model = OpenAIModel(api_key=key, base_url=args.base_url, model=args.model_name,
                            reasoning_effort=args.reasoning_effort)
    adapter_index = getattr(adapter, "sdk_index", None) or getattr(adapter, "sdk_contract", None) or {
        "protocol": "adapter-provided", "operations": ["observe", "use", "act", "verify", "record"]}
    loop = AgentLoop(agent=_ModelAgent(model), workspace=workspace, adapter=adapter,
                     context_builder=ContextBuilder(adapter_index=adapter_index, asset_registry=registry,
                                                    workspace=workspace), asset_registry=registry,
                     runtime=ControllerRuntime(timeout_seconds=args.controller_timeout),
                     event_store=EventStore(run_dir),
                     budget=LoopBudget(max_steps=args.max_steps, max_executions=args.max_executions),
                     root=str(run_dir), web_search=search_web if args.profile != "dev" else None)
    evaluation_policies = []
    if args.profile == "benchmark":
        from evaluation.anti_cheating import AntiCheatingPolicy
        from evaluation.generalization import GeneralizationPolicy
        from evaluation.provenance import ProvenancePolicy
        from evaluation.sealed_evaluation import SealedEvaluationPolicy
        evaluation_policies = [AntiCheatingPolicy(name="anti_cheating"),
                               GeneralizationPolicy(name="generalization"),
                               ProvenancePolicy(name="provenance"),
                               SealedEvaluationPolicy(name="sealed_evaluation")]
    try:
        result = loop.run(getattr(adapter, "instruction", str(args.task)))
    finally:
        close = getattr(adapter, "close", None)
        if callable(close): close()
    result["profile"] = args.profile
    result["evaluation_policies"] = [policy.name for policy in evaluation_policies]
    print(__import__("json").dumps(result, indent=2, default=str))
    return 0


def doctor_command(args) -> int:
    checks = {"python": sys.executable, "sandbox": __import__("shutil").which("bwrap"),
              "adapter": args.adapter,
              "api_key": bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("APEX_API_KEY")),
              "checkpoint": (str(Path(args.checkpoint).resolve()) if args.checkpoint else "not configured"),
              "dependencies": {}}
    for dependency in ("jsonschema", "openai"):
        try:
            importlib.import_module(dependency); checks["dependencies"][dependency] = "available"
        except Exception as exc: checks["dependencies"][dependency] = f"unavailable: {exc}"
    try:
        from .tool_runtime import ToolRuntime
        ToolRuntime(python=sys.executable); checks["tool_runtime"] = "available"
    except Exception as exc: checks["tool_runtime"] = f"unavailable: {exc}"
    if args.checkpoint:
        checks["checkpoint"] = "available" if Path(args.checkpoint).is_file() else "missing"
    if args.adapter == "libero":
        try:
            importlib.import_module("libero"); checks["libero"] = "available"
        except Exception as exc: checks["libero"] = f"unavailable: {exc}"
    else:
        try: _load(args.adapter); checks["adapter_import"] = "available"
        except Exception as exc: checks["adapter_import"] = f"unavailable: {exc}"
    checks["ok"] = bool(checks["sandbox"] and checks["tool_runtime"] == "available"
                         and all(value == "available" for value in checks["dependencies"].values())
                         and (checks.get("adapter_import", "available") == "available")
                         and ("libero" not in checks or checks["libero"] == "available")
                         and checks["checkpoint"] != "missing")
    print(__import__("json").dumps(checks, indent=2, default=str))
    return 0 if checks["ok"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="embodied_codex")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--adapter", required=True); run.add_argument("--task", required=True)
    run.add_argument("--profile", choices=("dev", "autonomous", "benchmark"), default="dev")
    run.add_argument("--model"); run.add_argument("--model-name", default="gpt-5.6-sol")
    run.add_argument("--base-url", default=os.environ.get("APEX_BASE_URL", "https://api.apexin.ai/v1"))
    run.add_argument("--reasoning-effort", default="high"); run.add_argument("--run-dir")
    run.add_argument("--max-steps", type=int, default=60); run.add_argument("--max-executions", type=int, default=20)
    run.add_argument("--controller-timeout", type=float, default=600)
    run.set_defaults(handler=run_command)
    doctor = sub.add_parser("doctor"); doctor.add_argument("--adapter", required=True)
    doctor.add_argument("--checkpoint", help="optional model/checkpoint path to validate")
    doctor.set_defaults(handler=doctor_command)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__": raise SystemExit(main())
