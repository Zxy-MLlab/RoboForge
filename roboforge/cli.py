"""Formal OpenHands-native RoboForge entry point."""
from __future__ import annotations
import argparse, os, secrets, shlex, subprocess, sys, time
import uuid
import json
from pathlib import Path


def _lifecycle_main(argv: list[str]) -> int | None:
    if not argv or argv[0] not in {"env", "run", "trial", "replay", "compare", "submit"}:
        return None
    from .control_plane import compare, environment_info, replay, submit
    root = argparse.ArgumentParser(prog="roboforge")
    commands = root.add_subparsers(dest="command", required=True)
    env = commands.add_parser("env", help="Inspect harness and runtime availability")
    env.add_argument("action", choices=["info"], default="info", nargs="?")
    run_parser = commands.add_parser("run", help="Execute one frozen candidate through a runtime provider")
    run_parser.add_argument("entrypoint", type=Path)
    run_parser.add_argument("--runtime", default="libero")
    run_parser.add_argument("--task", required=True); run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.add_argument("--adapter-python", default=os.getenv("ROBOFORGE_ADAPTER_PYTHON") or sys.executable)
    run_parser.add_argument("--timeout", type=float, default=600)
    trial_parser = commands.add_parser("trial", help="Run through the active external Runtime and materialize public evidence")
    trial_parser.add_argument("entrypoint", type=Path)
    trial_parser.add_argument("--intent", required=True)
    trial_parser.add_argument("--workspace", type=Path, default=Path(os.getenv("ROBOFORGE_WORKSPACE", ".")))
    trial_parser.add_argument("--socket", type=Path, default=os.getenv("ROBOFORGE_RPC_SOCKET"))
    trial_parser.add_argument("--token", default=os.getenv("ROBOFORGE_RPC_TOKEN"))
    replay_parser = commands.add_parser("replay", help="Verify and project immutable trial evidence")
    replay_parser.add_argument("trial")
    compare_parser = commands.add_parser("compare", help="Compare baseline and candidate trial evidence")
    compare_parser.add_argument("baseline")
    compare_parser.add_argument("candidate")
    submit_parser = commands.add_parser("submit", help="Submit a capability candidate to the external gate")
    submit_parser.add_argument("candidate_version")
    submit_parser.add_argument("--asset-root", required=True)
    submit_parser.add_argument("--evidence", action="append", required=True)
    submit_parser.add_argument("--note", required=True)
    args = root.parse_args(argv)
    exit_code = 0
    if args.command == "env": value = environment_info()
    elif args.command == "run":
        value = _run_frozen_candidate(args)
        exit_code = int((value.get("public") or {}).get("lifecycle", {}).get("runner_exit_code", 0))
    elif args.command == "trial":
        return _run_active_trial(args)
    elif args.command == "replay": value = replay(args.trial)
    elif args.command == "compare": value = compare(args.baseline, args.candidate)
    else: value = submit(args.asset_root, args.candidate_version, args.evidence, note=args.note)
    print(json.dumps(value, indent=2, sort_keys=True))
    return exit_code


def _run_active_trial(args) -> int:
    from .rpc import ExperimentRpcClient
    from .stop_gate import write_public_status
    from .trial_artifacts import materialize_preflight_failure, materialize_trial

    if not args.socket or not args.token:
        raise SystemExit("active trial requires ROBOFORGE_RPC_SOCKET and ROBOFORGE_RPC_TOKEN")
    workspace = args.workspace.resolve(); controller = args.entrypoint.resolve()
    try:
        controller.relative_to(workspace)
    except ValueError as exc:
        raise SystemExit("Controller must be inside the active OpenHands workspace") from exc
    service = ExperimentRpcClient(args.socket, args.token, timeout=900)
    preflight = service.preflight_controller(controller)
    if preflight.get("ok") is False:
        result = materialize_preflight_failure(workspace, preflight, controller_path=controller)
        print(json.dumps(result, indent=2, sort_keys=True))
        return int(result["runner_exit_code"])
    evidence = service.run_controller(request_id=f"terminal:{uuid.uuid4()}", controller_path=controller,
                                      intent=args.intent)
    result = materialize_trial(service, evidence, workspace, controller_path=controller)
    write_public_status(service, workspace, evidence)
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result["runner_exit_code"])


def _run_frozen_candidate(args) -> dict:
    from .rpc import ExperimentRpcClient
    entrypoint = args.entrypoint.resolve(); run = args.run_dir.resolve()
    if not entrypoint.is_file() or entrypoint.suffix != ".py":
        raise ValueError("entrypoint must be a Python Controller file")
    run.mkdir(parents=True, exist_ok=False)
    socket_path = run / "adapter.sock"; token = secrets.token_urlsafe(32)
    command = [args.adapter_python, "-m", "roboforge.rpc_server", "--adapter", args.runtime,
        "--task", str(args.task), "--state", str(args.seed), "--run-root", str(run / "provider"),
        "--controller-path", str(entrypoint), "--socket", str(socket_path),
        "--max-trials", "1", "--timeout-seconds", str(args.timeout),
        "--configuration-json", json.dumps({"disable_agent_verifier": True})]
    worker_env = os.environ.copy(); worker_env["ROBOFORGE_RPC_TOKEN"] = token
    worker = subprocess.Popen(command, cwd=Path(__file__).parents[1], env=worker_env)
    client = ExperimentRpcClient(socket_path, token, timeout=args.timeout + 60)
    try:
        for _ in range(300):
            if socket_path.exists():
                try: client.status(); break
                except (ConnectionRefusedError, FileNotFoundError, OSError): pass
            if worker.poll() is not None: raise RuntimeError(f"runtime worker exited: {worker.returncode}")
            time.sleep(.1)
        else: raise TimeoutError("runtime worker did not become ready")
        evidence = client.run_controller(request_id=f"cli:{uuid.uuid4()}", controller_path=entrypoint,
            intent="frozen candidate CLI execution")
        return evidence.public_dict()
    finally:
        worker.terminate()
        try: worker.wait(timeout=10)
        except subprocess.TimeoutExpired: worker.kill(); worker.wait()


def _llm_base_url(model: str, base_url: str) -> str:
    """Avoid LiteLLM's duplicated /v1 when routing Anthropic models."""
    if model.lower().startswith(("claude", "anthropic/")) and base_url.rstrip("/").endswith("/v1"):
        return base_url.rstrip("/")[:-3]
    return base_url


def _initialize_persistent_workspace(workspace: Path) -> None:
    """Create the stable project layout exposed to the OpenHands agent.

    The directories are intentionally just ordinary workspace folders. The
    agent remains free to create, rename, or remove project files through the
    public Editor/Terminal tools; this helper does not impose a development
    workflow or register any additional tool.
    """
    for name in (
        "controllers",
        "capabilities/perception",
        "capabilities/grasping",
        "capabilities/planning",
        "capabilities/control",
        "models",
        "services",
        "robot_sdk",
        "runtime_adapters",
        "experiments",
        "diagnostics",
        "tests",
        "configs",
        "requirements",
        "task_docs",
    ):
        (workspace / name).mkdir(parents=True, exist_ok=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    lifecycle = _lifecycle_main(argv)
    if lifecycle is not None: return lifecycle
    p = argparse.ArgumentParser(prog="roboforge-openhands")
    p.add_argument("--doctor", action="store_true")
    p.add_argument("--adapter", default="libero"); p.add_argument("--task")
    p.add_argument("--state", type=int, default=0); p.add_argument("--run-dir")
    p.add_argument("--asset-root", default="assets"); p.add_argument("--model", default="gpt-5.6-sol")
    p.add_argument("--base-url", default=os.getenv("ROBOFORGE_MODEL_BASE_URL", "https://api.apexin.ai/v1"))
    p.add_argument("--api-key-env", default="APEX_API_KEY"); p.add_argument("--max-trials", type=int, default=15)
    p.add_argument("--provider", choices=["openai", "apex"],
                   default=os.getenv("ROBOFORGE_MODEL_PROVIDER"))
    p.add_argument("--max-iterations", type=int, default=80)
    p.add_argument("--wall-time-budget", type=float, default=14400)
    p.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high", "xhigh"])
    p.add_argument("--model-retries", type=int, default=2)
    p.add_argument("--model-timeout", type=int, default=180)
    p.add_argument("--max-output-tokens", type=int, default=12000)
    p.add_argument("--adapter-python", default=os.getenv("ROBOFORGE_ADAPTER_PYTHON"))
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    if a.doctor:
        import importlib.metadata, importlib.util, shutil
        checks = {
            "openhands_sdk": importlib.util.find_spec("openhands.sdk") is not None,
            "openhands_sdk_version": None,
            "adapter_python": a.adapter_python or sys.executable,
            "adapter_python_available": Path(a.adapter_python or sys.executable).is_file(),
            "provider_key_configured": bool(os.getenv(a.api_key_env)),
            "unix_socket_supported": hasattr(__import__("socket"), "AF_UNIX"),
        }
        try: checks["openhands_sdk_version"] = importlib.metadata.version("openhands-sdk")
        except importlib.metadata.PackageNotFoundError: pass
        checks["ok"] = bool(checks["openhands_sdk"] and checks["adapter_python_available"]
            and checks["provider_key_configured"] and checks["unix_socket_supported"])
        print(json.dumps(checks, indent=2, sort_keys=True)); return 0 if checks["ok"] else 1
    if not a.task or not a.run_dir: p.error("--task and --run-dir are required unless --doctor is used")
    run = Path(a.run_dir).resolve(); workspace = run / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    _initialize_persistent_workspace(workspace)
    controller = workspace / "controllers" / "controller.py"
    if not controller.exists(): controller.write_text("def run(robot):\n    return robot.observe()\n")
    from openhands.sdk import LLM
    from .rpc import ExperimentRpcClient
    from . import create_openhands_conversation
    adapter_python = a.adapter_python or sys.executable
    socket_path = run / "adapter.sock"; token = secrets.token_urlsafe(32)
    # A previous process may leave a stale Unix socket after interruption;
    # remove only the socket node before starting the authenticated worker.
    if socket_path.exists():
        try:
            socket_path.unlink()
        except OSError:
            pass
    provider = a.provider
    if provider is None:
        provider = {"OPENAI_API_KEY": "openai", "APEX_API_KEY": "apex"}.get(a.api_key_env)
    if provider is None:
        raise SystemExit("--provider is required when --api-key-env is not a standard provider key")
    adapter_configuration = {"verifier_provider": provider,
                             "verifier_base_url": a.base_url,
                             "verifier_model": a.model}
    command = [adapter_python, "-m", "roboforge.rpc_server", "--adapter", a.adapter,
        "--task", a.task, "--state", str(a.state), "--run-root", str(run / "adapter-worker"),
        "--controller-path", str(controller), "--socket", str(socket_path),
        "--max-trials", str(a.max_trials),
        "--configuration-json", json.dumps(adapter_configuration, sort_keys=True)]
    worker_env = os.environ.copy(); worker_env["ROBOFORGE_RPC_TOKEN"] = token
    worker = subprocess.Popen(command, cwd=Path(__file__).parents[1], env=worker_env)
    service = ExperimentRpcClient(socket_path, token, timeout=900)
    for _ in range(300):
        if socket_path.exists():
            try:
                service.status(); break
            except (ConnectionRefusedError, FileNotFoundError, OSError): pass
        if worker.poll() is not None: raise SystemExit(f"Adapter worker exited: {worker.returncode}")
        time.sleep(.1)
    else: worker.terminate(); raise SystemExit("Adapter worker did not become ready")
    key = os.getenv(a.api_key_env)
    if not key: raise SystemExit(f"missing {a.api_key_env}")
    # Apex exposes the OpenAI-compatible Chat Completions route; force that
    # route because LiteLLM metadata may incorrectly select Responses API.
    api_mode = "chat" if provider == "apex" else "auto"
    capability_overrides = {"supports_vision": True} if provider == "apex" else {}
    llm = LLM(model=a.model, api_key=key, base_url=_llm_base_url(a.model, a.base_url), api_mode=api_mode,
              capability_overrides=capability_overrides, usage_id=f"roboforge:{run.name}",
              reasoning_effort=a.reasoning_effort, num_retries=a.model_retries,
              timeout=a.model_timeout, max_output_tokens=a.max_output_tokens)
    conversation_id = None
    if a.resume and (run / "openhands").is_dir():
        candidates = sorted(p for p in (run / "openhands").iterdir() if p.is_dir())
        if candidates: conversation_id = uuid.UUID(hex=candidates[-1].name)
    task_info = service.task_info()
    interface_manual = workspace / "ROBOT_INTERFACE.json"
    interface_manual.write_text(json.dumps(task_info.get("robot_interface") or {}, indent=2,
        sort_keys=True), encoding="utf-8")
    from openhands.sdk.hooks import HookConfig, HookDefinition, HookMatcher
    from .stop_gate import write_public_status

    campaign_status = write_public_status(service, workspace)
    stop_hook = HookConfig(stop=[HookMatcher(hooks=[HookDefinition(
        command=(
            f"PYTHONPATH={shlex.quote(str(Path(__file__).parents[1]))} "
            f"{shlex.quote(sys.executable)} -m roboforge.stop_gate "
            f"--status {shlex.quote(str(campaign_status))}"
        ),
        timeout=10,
    )])])
    convo = create_openhands_conversation(llm=llm, workspace=workspace,
        persistence_dir=run / "openhands", service=service, controller_path=controller,
        asset_root=a.asset_root, conversation_id=conversation_id,
        max_iterations=a.max_iterations, hook_config=stop_hook,
        terminal_env={
            "ROBOFORGE_RPC_SOCKET": str(socket_path),
            "ROBOFORGE_RPC_TOKEN": token,
            "ROBOFORGE_WORKSPACE": str(workspace),
            "PYTHONPATH": str(Path(__file__).parents[1]),
        })
    prompt = f"""Unknown robot task: {task_info.get('instruction') or a.task}
The public Robot SDK manual is in {interface_manual}. Read relevant sections when needed.
Work like a coding agent. Inspect files, write and revise controllers/controller.py, and run experiments
from Terminal with `python -m roboforge trial controllers/controller.py --workspace . --intent '<hypothesis>'`.
Read `.roboforge/trials/<trial_id>/result.json`, `first_error.json`, `trace.json`, keyframes, video and logs
with ordinary Terminal/file tools, then continue until authentic verification or
the physical budget is exhausted. Do not infer hidden simulator state or use task-specific patches.
Reusable assets are under {Path(a.asset_root).resolve()} and should be searched only when useful."""
    prompt += "\nYour first response MUST invoke terminal or file_editor; do not reply with planning text alone. Continue using tools until the task is finished or the budget is exhausted."
    prompt += "\nBefore the first physical trial, consider whether existing assets are relevant. When asset search returns a relevant result, read that selected asset before deciding whether to reuse or adapt it; a search hit alone is not reuse."
    prompt += ("\nIf factual evidence reveals a missing reusable software capability, "
               "independently implement or obtain it as ordinary workspace code, inspect its "
               "source and license, validate it through Terminal, and integrate it through the "
               "Robot SDK or a normal Python/service dependency. The external Control Plane "
               "alone freezes and publishes validated capabilities; there is no Agent-side "
               "capability acquisition or materialization tool.")
    try:
        if conversation_id is None: convo.send_message(prompt)
        else: convo.send_message(
            "Resume the unknown robot task from durable physical evidence and current Controller. "
            "Do not repeat committed physical actions. A task is complete only when immutable "
            "physical evidence has physical_verification.verified=true; partial attachment, "
            "transport, geometric, or visual evidence must not override a false authentic "
            "task-level receipt. If budget remains, independently decide whether to inspect, "
            "revise, acquire a capability, or run a new Controller experiment. Before another "
            "physical trial, compare the latest failed trials and explicitly determine whether "
            "the unresolved behavior is a reusable software capability gap. If it is, implement "
            "or obtain generic workspace-local code, validate it with Terminal, and integrate "
            "it as a normal Python/service dependency before physical use."
        )
        # One official OpenHands run owns the entire edit→Terminal→inspect→edit
        # session. The public Stop hook keeps that same run alive after an
        # unverified trial; RoboForge does not implement a second AgentLoop.
        started = time.monotonic()
        convo.run()
        elapsed = time.monotonic() - started
        status = service.status()
        latest = status.get("latest_physical_evidence")
        verified = bool(
            latest
            and (service.inspect_trial(str(latest)).physical_verification or {}).get("verified") is True
        )
        termination_reason = (
            "verified" if verified
            else "physical_trial_budget_exhausted" if status["physical_trials"] >= status["max_trials"]
            else "wall_time_budget_exhausted" if elapsed >= a.wall_time_budget
            else "openhands_iteration_budget_or_agent_stop"
        )
        campaign = {
            "schema_version": 1,
            "termination_reason": termination_reason,
            "physical_trials": status["physical_trials"],
            "max_physical_trials": status["max_trials"],
            "max_openhands_iterations": a.max_iterations,
            "wall_time_budget_seconds": a.wall_time_budget,
            "elapsed_seconds": elapsed,
            "latest_verified": verified,
        }
        (workspace / ".roboforge" / "campaign-result.json").write_text(
            json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    finally:
        convo.close(); worker.terminate()
        try: worker.wait(timeout=10)
        except subprocess.TimeoutExpired: worker.kill(); worker.wait()
    return 0

if __name__ == "__main__": raise SystemExit(main())
