"""Small authenticated JSON-lines RPC transport for split Python runtimes."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import socket
import socketserver
import threading
import time
from typing import Any, Callable

from .models import ArtifactHandle, ExperimentEvidence
from .service import CorruptStore, ExperimentService, ProtocolError


_MAX_REQUEST = 1 * 1024 * 1024
_MAX_RESPONSE = 128 * 1024 * 1024


def _decode_evidence(value: dict[str, Any]) -> ExperimentEvidence:
    artifacts = tuple(ArtifactHandle(**item) for item in value.get("artifacts", ()))
    optional = {
        "diagnostic_index": None,
        "physical_trial_index": None,
        "environment_generation": None,
        "controller_sha256": None,
        "intent": None,
        "assets_used": (),
        "physical_verification": None,
        "execution_error": None,
        "candidate_bundle_digest": None,
    }
    optional.update({key: value.get(key) for key in optional})
    optional["assets_used"] = tuple(value.get("assets_used") or ())
    return ExperimentEvidence(
        schema_version=int(value["schema_version"]),
        ref=str(value["ref"]),
        execution_kind=value["execution_kind"],
        request_id=str(value["request_id"]),
        public=dict(value.get("public") or {}),
        artifacts=artifacts,
        evidence_sha256=str(value.get("evidence_sha256", "")),
        **optional,
    )


class ExperimentRpcServer:
    """Serve an ExperimentService over a private Unix socket."""

    def __init__(
        self,
        service: ExperimentService,
        socket_path: str | Path,
        token: str,
        *,
        controller_path: str | Path,
    ) -> None:
        self.service = service
        self.socket_path = Path(socket_path).resolve()
        self.token = str(token)
        self.controller_path = Path(controller_path).resolve()
        self._server: socketserver.UnixStreamServer | None = None

    def _dispatch(self, request: dict[str, Any]) -> Any:
        if request.get("token") != self.token:
            raise ProtocolError("RPC authentication failed")
        operation = str(request.get("operation") or "")
        if operation == "status":
            return self.service.status()
        if operation == "task_info":
            return self.service.task_info()
        if operation == "experiment_spine":
            return self.service.experiment_spine()
        if operation == "preflight":
            return self.service.preflight_controller(self.controller_path)
        if operation == "observe":
            return self.service.observe(request_id=str(request["request_id"])).public_dict()
        if operation == "run_controller":
            evidence = self.service.run_controller(
                request_id=str(request["request_id"]),
                controller_path=self.controller_path,
                intent=str(request["intent"]),
                assets_used=[str(x) for x in request.get("assets_used", [])],
            )
            return evidence.public_dict()
        if operation == "inspect_trial":
            return self.service.inspect_trial(str(request["ref"])).public_dict()
        if operation == "list_trials":
            return [item.public_dict() for item in self.service.list_trials()]
        if operation == "compare_trials":
            return self.service.compare_trials(str(request["first_ref"]), str(request["second_ref"]))
        if operation == "read_artifact":
            data = self.service.read_artifact(
                ArtifactHandle(
                    uri=str(request["uri"]),
                    sha256=str(request["sha256"]),
                    media_type="application/octet-stream",
                    name="artifact",
                    size_bytes=0,
                )
            )
            return {"data_base64": base64.b64encode(data).decode("ascii")}
        raise ProtocolError(f"unsupported RPC operation: {operation}")

    def serve_forever(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                line = self.rfile.readline(_MAX_REQUEST + 1)
                if len(line) > _MAX_REQUEST or not line:
                    return
                try:
                    request = json.loads(line.decode("utf-8"))
                    if not isinstance(request, dict):
                        raise ProtocolError("RPC request must be an object")
                    result = owner._dispatch(request)
                    response = {"ok": True, "result": result}
                except Exception as exc:
                    response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                encoded = (json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n").encode()
                if len(encoded) <= _MAX_RESPONSE:
                    self.wfile.write(encoded)
                    self.wfile.flush()

        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True

        server = Server(str(self.socket_path), Handler)
        os.chmod(self.socket_path, 0o600)
        self._server = server
        try:
            server.serve_forever()
        finally:
            server.server_close()
            self.socket_path.unlink(missing_ok=True)

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()


class ExperimentRpcClient:
    """ExperimentService-compatible client for OpenHands' Python process."""

    def __init__(self, socket_path: str | Path, token: str, *, timeout: float = 120.0):
        self.socket_path = str(Path(socket_path).resolve())
        self.token = str(token)
        self.timeout = float(timeout)

    def _call(self, operation: str, **payload: Any) -> Any:
        request = {"token": self.token, "operation": operation, **payload}
        encoded = (json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if len(encoded) > _MAX_REQUEST:
            raise ProtocolError("RPC request exceeds size limit")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.settimeout(min(self.timeout, max(1.0, deadline - time.monotonic())))
                    connection.connect(self.socket_path)
                    connection.sendall(encoded)
                    stream = connection.makefile("rb")
                    line = stream.readline(_MAX_RESPONSE + 1)
                break
            except (ConnectionRefusedError, FileNotFoundError, socket.timeout):
                if time.monotonic() >= deadline: raise
                time.sleep(0.1)
        if len(line) > _MAX_RESPONSE or not line:
            raise ProtocolError("RPC response is unavailable or too large")
        response = json.loads(line.decode("utf-8"))
        if not response.get("ok"):
            message = str(response.get("error") or "RPC operation failed")
            if message.startswith("BudgetExhausted:"):
                from .service import BudgetExhausted
                raise BudgetExhausted(message.removeprefix("BudgetExhausted: ").strip())
            if message.startswith("CorruptStore:"):
                raise CorruptStore(message.removeprefix("CorruptStore: ").strip())
            raise ProtocolError(message)
        return response.get("result")

    def status(self) -> dict[str, Any]:
        return dict(self._call("status"))

    def task_info(self) -> dict[str, Any]:
        return dict(self._call("task_info"))

    def experiment_spine(self) -> dict[str, Any]:
        return dict(self._call("experiment_spine"))

    def preflight_controller(self, controller_path: str | Path) -> dict[str, Any]:
        del controller_path
        return dict(self._call("preflight"))

    def observe(self, *, request_id: str) -> ExperimentEvidence:
        return _decode_evidence(dict(self._call("observe", request_id=request_id)))

    def run_controller(self, *, request_id: str, controller_path: str | Path, intent: str,
                       assets_used: list[str] | None = None) -> ExperimentEvidence:
        # controller_path is accepted for ExperimentService compatibility, but
        # the server's fixed path is authoritative and cannot be overridden.
        del controller_path
        return _decode_evidence(dict(self._call("run_controller", request_id=request_id,
            intent=intent, assets_used=list(assets_used or ()))))

    def inspect_trial(self, ref: str) -> ExperimentEvidence:
        return _decode_evidence(dict(self._call("inspect_trial", ref=ref)))

    def list_trials(self) -> list[ExperimentEvidence]:
        return [_decode_evidence(dict(item)) for item in self._call("list_trials")]

    def compare_trials(self, first_ref: str, second_ref: str) -> dict[str, Any]:
        return dict(self._call("compare_trials", first_ref=first_ref, second_ref=second_ref))

    def read_artifact(self, handle: ArtifactHandle) -> bytes:
        value = self._call("read_artifact", uri=handle.uri, sha256=handle.sha256)
        return base64.b64decode(str(value["data_base64"]))


__all__ = ["ExperimentRpcClient", "ExperimentRpcServer"]
