"""Entry point for the Python 3.11 embodied Experiment Service process."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import threading

from .providers.libero import LiberoProvider
from .rpc import ExperimentRpcServer
from .service import ExperimentService


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a frozen Adapter to the v2 client")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--controller-path", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--token", default=os.getenv("ROBOFORGE_RPC_TOKEN"))
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--max-diagnostics", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--configuration-json", default="{}")
    args = parser.parse_args()
    if not args.token:
        parser.error("RPC token is required via ROBOFORGE_RPC_TOKEN or --token")

    # Imports are deliberately local: this process is the only side allowed to
    # load the frozen Adapter/deployment and its physical dependencies.
    from embodied_codex.adapters.factory import load_adapter
    from .candidate_runtime import ControllerRuntime

    configuration = json.loads(args.configuration_json)
    if not isinstance(configuration, dict):
        raise SystemExit("--configuration-json must be an object")
    legacy = load_adapter(
        args.adapter,
        task=str(args.task),
        run_dir=args.run_root / "legacy",
        case=args.state,
        configuration=configuration,
    )
    provider = LiberoProvider(
        legacy,
        ControllerRuntime(timeout_seconds=args.timeout_seconds),
    )
    service = ExperimentService(
        args.run_root / "service",
        provider,
        max_trials=args.max_trials,
        max_diagnostics=args.max_diagnostics,
    )
    server = ExperimentRpcServer(
        service,
        args.socket,
        args.token,
        controller_path=args.controller_path,
    )

    def stop(_signum, _frame):
        # socketserver.shutdown() must run outside the serve_forever thread.
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
    finally:
        legacy.close()


if __name__ == "__main__":
    main()
