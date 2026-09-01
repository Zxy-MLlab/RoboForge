"""Formal OpenHands-native RoboForge entry point."""
from __future__ import annotations
import argparse, os, secrets, subprocess, sys, time
import uuid
import json
from pathlib import Path


def _llm_base_url(model: str, base_url: str) -> str:
    """Avoid LiteLLM's duplicated /v1 when routing Anthropic models."""
    if model.lower().startswith(("claude", "anthropic/")) and base_url.rstrip("/").endswith("/v1"):
        return base_url.rstrip("/")[:-3]
    return base_url

def main(argv=None):
    p = argparse.ArgumentParser(prog="roboforge-openhands")
    p.add_argument("--doctor", action="store_true")
    p.add_argument("--adapter", default="libero"); p.add_argument("--task")
    p.add_argument("--state", type=int, default=0); p.add_argument("--run-dir")
    p.add_argument("--asset-root", default="assets"); p.add_argument("--model", default="gpt-5.6-sol")
    p.add_argument("--base-url", default=os.getenv("ROBOFORGE_MODEL_BASE_URL", "https://api.apexin.ai/v1"))
    p.add_argument("--api-key-env", default="APEX_API_KEY"); p.add_argument("--max-trials", type=int, default=12)
    p.add_argument("--provider", choices=["openai", "apex"],
                   default=os.getenv("ROBOFORGE_MODEL_PROVIDER"))
    p.add_argument("--max-iterations", type=int, default=80)
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
    workspace.mkdir(parents=True, exist_ok=True); controller = workspace / "controller.py"
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
        "--controller-path", str(controller), "--socket", str(socket_path), f"--token={token}",
        "--max-trials", str(a.max_trials),
        "--configuration-json", json.dumps(adapter_configuration, sort_keys=True)]
    worker = subprocess.Popen(command, cwd=Path(__file__).parents[1], env=os.environ.copy())
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
    convo = create_openhands_conversation(llm=llm, workspace=workspace,
        persistence_dir=run / "openhands", service=service, controller_path=controller,
        asset_root=a.asset_root, conversation_id=conversation_id,
        max_iterations=a.max_iterations)
    task_info = service.task_info()
    interface_manual = workspace / "ROBOT_INTERFACE.json"
    interface_manual.write_text(json.dumps(task_info.get("robot_interface") or {}, indent=2,
        sort_keys=True), encoding="utf-8")
    prompt = f"""Unknown robot task: {task_info.get('instruction') or a.task}
The public Robot SDK manual is in {interface_manual}. Read relevant sections when needed.
Work like a coding agent. Observe, inspect files, write and revise controller.py, run physical
experiments, inspect and compare factual evidence, and continue until authentic verification or
the physical budget is exhausted. Do not infer hidden simulator state or use task-specific patches.
Reusable assets are under {Path(a.asset_root).resolve()} and should be searched only when useful."""
    prompt += "\nYour first response MUST invoke an available tool (observe, terminal, or file_editor); do not reply with planning text alone. Continue using tools until the task is finished or the budget is exhausted."
    prompt += "\nBefore the first physical trial, consider whether existing assets are relevant. When asset search returns a relevant result, read that selected asset before deciding whether to reuse or adapt it; a search hit alone is not reuse."
    prompt += ("\nIf factual evidence reveals a missing reusable software capability, you may "
               "independently implement or obtain a generic Python utility in the workspace, "
               "validate and register it with acquire_capability, then read and materialize it "
               "before importing it in controller.py. Decide whether this is useful yourself; "
               "the Harness does not select capabilities or trigger acquisition by rule.")
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
            "a generic workspace-local Python capability, validate/register it with "
            "acquire_capability, then read/materialize and integrate it before physical use."
        )
        convo.run()
    finally:
        convo.close(); worker.terminate()
        try: worker.wait(timeout=10)
        except subprocess.TimeoutExpired: worker.kill(); worker.wait()
    return 0

if __name__ == "__main__": raise SystemExit(main())
