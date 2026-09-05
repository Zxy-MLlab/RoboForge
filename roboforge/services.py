"""Lifecycle manager for the existing ASPIRE/CaP-X model services.

This module only starts the upstream servers already present on disk. It does
not install packages, download weights, or substitute a fake implementation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any


SERVICE_STATE = Path(os.getenv("ROBOFORGE_SERVICE_STATE", "/tmp/roboforge-services.json"))
UPSTREAM = Path(os.getenv("ROBOFORGE_API_UPSTREAM", "/root/autodl-tmp/roboforge-api-upstreams/cap-x"))
PYTHON = Path(os.getenv("ROBOFORGE_API_PYTHON", "/root/autodl-tmp/roboforge-api-services/bin/python"))
CHECKPOINTS = Path(os.getenv("ROBOFORGE_CHECKPOINT_DIR", "/root/autodl-tmp/roboforge-assets/checkpoints"))

SERVICES = {
    "sam3": {"port": 8114, "script": "sam3", "args": ["--device", "cuda"]},
    "graspnet": {"port": 8115, "script": "graspnet", "args": []},
    "pyroki": {"port": 8116, "module": "capx.serving.launch_pyroki_server", "args": ["--robot", "panda_description", "--target-link", "panda_hand"]},
    "curobo": {"port": 8117, "module": "capx.serving.launch_curobo_server", "args": []},
    "molmo": {"port": 8122, "command": "vllm", "args": ["serve", "allenai/Molmo2-8B", "--trust-remote-code", "--dtype", "bfloat16"]},
}


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", int(port))) == 0


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state() -> dict[str, Any]:
    try:
        value = json.loads(SERVICE_STATE.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(value: dict[str, Any]) -> None:
    SERVICE_STATE.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def doctor() -> dict[str, Any]:
    checkpoint_names = {
        "groundingdino": "groundingdino_swint_ogc.pth",
        "sam": "sam_vit_b_01ec64.pth",
        "graspnet": "graspnet-checkpoint-rs.tar",
    }
    checks: dict[str, Any] = {
        "upstream_root": str(UPSTREAM),
        "upstream_exists": UPSTREAM.is_dir(),
        "python": str(PYTHON),
        "python_exists": PYTHON.is_file(),
        "cuda_visible": os.getenv("CUDA_VISIBLE_DEVICES", "all"),
        "checkpoints": {},
        "ports": {name: {"port": spec["port"], "open": _port_open(spec["port"])}
                   for name, spec in SERVICES.items()},
    }
    for key, filename in checkpoint_names.items():
        path = CHECKPOINTS / filename
        checks["checkpoints"][key] = {
            "path": str(path), "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": _sha256(path),
        }
    checks["scripts"] = {
        name: (UPSTREAM / "capx" / "serving" / f"launch_{name}_server.py").is_file()
        for name in ("sam3", "contact_graspnet", "pyroki", "curobo")
    }
    checks["state_file"] = str(SERVICE_STATE)
    checks["ok"] = bool(checks["upstream_exists"] and checks["python_exists"] and
                         all(item["exists"] for item in checks["checkpoints"].values()))
    return checks


def status() -> dict[str, Any]:
    state = _load_state()
    rows = {}
    for name, spec in SERVICES.items():
        pid = state.get(name, {}).get("pid") if isinstance(state.get(name), dict) else None
        alive = bool(pid and _pid_alive(int(pid)))
        rows[name] = {"port": spec["port"], "pid": pid, "process_alive": alive,
                      "port_open": _port_open(spec["port"]),
                      "log": state.get(name, {}).get("log") if isinstance(state.get(name), dict) else None}
    return {"state_file": str(SERVICE_STATE), "services": rows}


def _pid_alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{int(pid)}/stat").read_text().split()[2]
        if state == "Z":
            return False
    except (OSError, ValueError, IndexError):
        pass
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def up(*, include_molmo: bool = True) -> dict[str, Any]:
    report = doctor()
    if not report["ok"]:
        raise RuntimeError("service doctor failed: " + json.dumps(report, sort_keys=True))
    state = _load_state()
    selected = ["sam3", "graspnet", "pyroki", "curobo"] + (["molmo"] if include_molmo else [])
    for name in selected:
        spec = SERVICES[name]
        if _port_open(spec["port"]):
            continue
        log = Path(os.getenv("ROBOFORGE_SERVICE_LOG_DIR", "/tmp")) / f"roboforge-{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        if spec.get("script") == "sam3":
            script = UPSTREAM.parent / "aspire" / "aspire" / "sim" / "cap" / "serving" / "launch_sam3_server.py"
            checkpoint = Path(os.getenv("ROBOFORGE_SAM3_MODEL", "/root/autodl-tmp/roboforge-assets/models/sam3/sam3.pt")).resolve()
            wrapper = (
                "import importlib.util, sam3.model_builder as mb; "
                f"p={str(script)!r}; s=importlib.util.spec_from_file_location('roboforge_sam3',p); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                f"orig=mb.build_sam3_image_model; m.build_sam3_image_model=lambda **kw: orig(checkpoint_path={str(checkpoint)!r}, load_from_HF=False, **kw); "
                f"m.main(device='cuda', port={spec['port']}, host='127.0.0.1')"
            )
            command = [str(PYTHON), "-u", "-c", wrapper, "--port", str(spec["port"]), "--host", "127.0.0.1", *spec["args"]]
        elif spec.get("script") == "graspnet":
            script = UPSTREAM.parent / "aspire" / "aspire" / "sim" / "cap" / "serving" / "launch_contact_graspnet_server.py"
            vendor = Path(os.getenv("CONTACT_GRASPNET_ROOT", "/root/autodl-tmp/roboforge-api-upstreams/contact_graspnet_pytorch")).resolve()
            checkpoint_dir = vendor / "checkpoints" / "contact_graspnet" / "checkpoints"
            command = [str(PYTHON), "-u", str(script), "--port", str(spec["port"]), "--host", "127.0.0.1",
                       "--vendor-root", str(vendor), "--checkpoint-dir", str(checkpoint_dir), *spec["args"]]
        elif name == "curobo":
            wrapper = ("import importlib, warp; import warp._src.torch as _wt; warp.torch=_wt; "
                       "m=importlib.import_module('capx.serving.launch_curobo_server'); "
                       "[getattr(m,n).model_rebuild() for n in ('IkRequest','PlanRequest','MotionPlanRequest','CuboidObstacle') if hasattr(getattr(m,n,None),'model_rebuild')]; "
                       f"m.main(port={spec['port']}, host='127.0.0.1')")
            command = [str(PYTHON), "-u", "-c", wrapper, "--port", str(spec["port"]), "--host", "127.0.0.1", *spec["args"]]
        else:
            model = os.getenv("ROBOFORGE_MOLMO_MODEL_PATH", "/root/autodl-tmp/roboforge-assets/models/molmo2-4b-fp8")
            vllm = Path(os.getenv("ROBOFORGE_VLLM", str(PYTHON.parent / "vllm")))
            command = [str(vllm), "serve", model, "--trust-remote-code", "--dtype", "auto",
                       "--port", str(spec["port"]), "--host", "127.0.0.1",
                       "--limit-mm-per-prompt", '{"image":2}', "--max-model-len", "2048"]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [
                   str(UPSTREAM), str(UPSTREAM.parent / "aspire"),
                   str(UPSTREAM.parent / "sam3"), str(UPSTREAM.parent / "contact_graspnet_pytorch"),
                   os.environ.get("PYTHONPATH", "")])),
               "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS", "2"),
               "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS", "2"),
               "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS", "2"),
               "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        with log.open("ab") as stream:
            process = subprocess.Popen(command, cwd=UPSTREAM, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        state[name] = {"pid": process.pid, "command": command, "log": str(log), "started_unix": time.time()}
        _save_state(state)
    deadline = time.time() + float(os.getenv("ROBOFORGE_SERVICE_START_TIMEOUT", "120"))
    while time.time() < deadline:
        current = status()
        if all(current["services"][name]["port_open"] for name in selected):
            current["ok"] = True
            return current
        if any(current["services"][name]["port_open"] or current["services"][name]["process_alive"]
               for name in selected):
            time.sleep(2)
            continue
        break
    current = status()
    current["ok"] = all(current["services"][name]["port_open"] for name in selected)
    return current


def down() -> dict[str, Any]:
    state = _load_state()
    for value in state.values():
        if not isinstance(value, dict) or not value.get("pid"):
            continue
        pid = int(value["pid"])
        if _pid_alive(pid):
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                try: os.kill(pid, signal.SIGTERM)
                except OSError: pass
    SERVICE_STATE.unlink(missing_ok=True)
    return status()


def warmup() -> dict[str, Any]:
    import urllib.error
    import urllib.request
    checks = {}
    for name, spec in SERVICES.items():
        url = f"http://127.0.0.1:{spec['port']}/v1/models" if name == "molmo" else f"http://127.0.0.1:{spec['port']}/health"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                checks[name] = {"reachable": True, "status": response.status}
        except urllib.error.HTTPError as exc:
            # Upstream SAM3/GraspNet/PyRoKi expose no /health route; a live
            # HTTP 404 still proves the process is listening.
            checks[name] = {"reachable": exc.code == 404, "status": exc.code}
        except Exception as exc:
            checks[name] = {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"services": checks, "ok": all(item["reachable"] for item in checks.values())}


__all__ = ["doctor", "status", "up", "down", "warmup"]
