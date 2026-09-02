"""Formal OpenHands-native RoboForge entry point."""
from __future__ import annotations
import argparse, os, secrets, subprocess, sys, threading, time
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
    worker_env = _runtime_worker_env(token)
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


def _runtime_worker_env(token: str) -> dict[str, str]:
    """Pass runtime configuration, but never Agent/evaluator credentials."""
    allowed = {
        "PATH", "LANG", "LC_ALL", "LC_CTYPE", "PYTHONPATH", "PYTHONNOUSERSITE",
        "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES", "MUJOCO_GL", "OMP_NUM_THREADS",
        "LIBERO_CONFIG_PATH", "PYOPENGL_PLATFORM", "ROBOFORGE_DEVICE", "ROBOFORGE_ROOT",
        "ROBOFORGE_VENDOR_ROOT", "ROBOFORGE_LIBERO_VENDOR_CONFIG",
        "ROBOFORGE_GROUNDINGDINO_CHECKPOINT", "ROBOFORGE_SAM_CHECKPOINT",
        "ROBOFORGE_GRASPNET_CHECKPOINT", "ROBOFORGE_PYTHON",
        "HF_HOME", "TOKENIZERS_PARALLELISM",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["ROBOFORGE_RPC_TOKEN"] = token
    env["PYTHONNOUSERSITE"] = "1"
    return env


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


def _write_campaign_result(
    workspace: Path,
    *,
    status: dict,
    elapsed: float,
    max_iterations: int,
    wall_time_budget: float,
    latest_verified: bool,
    run_error: str | None = None,
    conversation_reason: str | None = None,
    max_agent_budget: float | None = None,
) -> dict:
    if latest_verified:
        termination_reason = "verified"
    elif status["physical_trials"] >= status["max_trials"]:
        termination_reason = "physical_trial_budget_exhausted"
    elif elapsed >= wall_time_budget:
        termination_reason = "wall_time_budget_exhausted"
    elif run_error:
        termination_reason = "openhands_run_error"
    elif conversation_reason:
        termination_reason = conversation_reason
    else:
        termination_reason = "openhands_iteration_budget_or_agent_stop"
    campaign = {
        "schema_version": 1,
        "termination_reason": termination_reason,
        "physical_trials": status["physical_trials"],
        "max_physical_trials": status["max_trials"],
        "max_openhands_iterations": max_iterations,
        "wall_time_budget_seconds": wall_time_budget,
        "max_agent_budget": max_agent_budget,
        "elapsed_seconds": elapsed,
        "latest_verified": latest_verified,
    }
    if run_error:
        campaign["run_error"] = run_error
    result_path = workspace / ".roboforge" / "campaign-result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return campaign


def _conversation_termination_reason(conversation) -> str | None:
    """Map OpenHands' public conversation state/events to a durable reason."""
    try:
        state = conversation.state
        status = getattr(getattr(state, "execution_status", None), "value", None)
        if status is None:
            status = str(getattr(state, "execution_status", ""))
        events = list(getattr(state, "events", ()) or ())
    except Exception:
        return None
    codes = []
    for event in events:
        code = getattr(event, "code", None)
        if code is None and isinstance(event, dict):
            code = event.get("code")
        if code:
            codes.append(str(code))
    if "MaxIterationsReached" in codes:
        return "openhands_iteration_budget_exhausted"
    if "MaxBudgetReached" in codes:
        return "openhands_budget_exhausted"
    mapping = {
        "finished": "agent_finished",
        "paused": "conversation_paused",
        "stuck": "conversation_stuck",
        "error": "openhands_conversation_error",
        "waiting_for_confirmation": "waiting_for_confirmation",
        "idle": "agent_stop",
    }
    return mapping.get(status)


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
    p.add_argument("--max-agent-budget", type=float, default=None,
                   help="OpenHands public max_budget_per_run limit")
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
    worker_env = _runtime_worker_env(token)
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
    convo = create_openhands_conversation(llm=llm, workspace=workspace,
        persistence_dir=run / "openhands", service=service, controller_path=controller,
        asset_root=a.asset_root, conversation_id=conversation_id,
        max_iterations=a.max_iterations, max_budget_per_run=a.max_agent_budget,
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
    prompt += ("\nUse the public coding tools and ordinary Terminal commands as appropriate. "
               "Trial preflight results, execution traces, images, logs and artifacts are available "
               "in the workspace; inspect whichever evidence is useful before deciding your next edit "
               "or experiment. Continue until you explicitly finish or an SDK/runtime budget is reached.")
    prompt += ("\nIf factual evidence reveals a missing reusable software capability, "
               "independently implement or obtain it as ordinary workspace code, inspect its "
               "source and license, validate it through Terminal, and integrate it through the "
               "Robot SDK or a normal Python/service dependency. The external Control Plane "
               "alone freezes and publishes validated capabilities; there is no Agent-side "
               "capability acquisition or materialization tool.")
    started = time.monotonic()
    wall_time_expired = threading.Event()

    def pause_at_wall_time_limit() -> None:
        wall_time_expired.set()
        # ``pause`` is an official LocalConversation lifecycle operation.  It
        # lets the current SDK run terminate without wrapping it in another
        # model/tool loop.
        try:
            convo.pause()
        except Exception:
            pass

    wall_timer = threading.Timer(a.wall_time_budget, pause_at_wall_time_limit)
    wall_timer.daemon = True
    wall_timer.start()
    run_error = None
    run_exception = None
    final_status = None
    latest_verified = False
    conversation_reason = None
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
        # session. RoboForge does not intercept Finish or implement a second
        # AgentLoop; the SDK's own iteration/budget lifecycle is authoritative.
        convo.run()
        conversation_reason = _conversation_termination_reason(convo)
    except Exception as exc:
        # Preserve a machine-readable campaign record even when the public
        # OpenHands session fails (for example, a transient provider outage).
        # The durable trial service remains the source of physical-trial truth.
        run_error = f"{type(exc).__name__}: {exc}"
        run_exception = exc
    finally:
        wall_timer.cancel()
        # Query the worker before terminating it so the final campaign record
        # reflects all durable trials completed during this run, including an
        # OpenHands exception path.
        try:
            final_status = service.status()
            latest = final_status.get("latest_physical_evidence")
            latest_verified = bool(
                latest
                and (service.inspect_trial(str(latest)).physical_verification or {}).get("verified") is True
            )
        except Exception as status_exc:
            status_error = f"{type(status_exc).__name__}: {status_exc}"
            run_error = f"{run_error}; status collection failed: {status_error}" if run_error else status_error
            if run_exception is None:
                run_exception = status_exc
        convo.close(); worker.terminate()
        try: worker.wait(timeout=10)
        except subprocess.TimeoutExpired: worker.kill(); worker.wait()
    elapsed = time.monotonic() - started
    if wall_time_expired.is_set():
        elapsed = max(elapsed, a.wall_time_budget)
    status = final_status or {"physical_trials": 0, "max_trials": a.max_trials}
    _write_campaign_result(
        workspace,
        status=status,
        elapsed=elapsed,
        max_iterations=a.max_iterations,
        wall_time_budget=a.wall_time_budget,
        latest_verified=latest_verified,
        run_error=run_error,
        conversation_reason=conversation_reason,
        max_agent_budget=a.max_agent_budget,
    )
    if run_exception is not None:
        raise run_exception
    return 0

if __name__ == "__main__": raise SystemExit(main())
