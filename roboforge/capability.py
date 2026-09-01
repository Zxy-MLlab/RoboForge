"""Safe registration gate for agent-acquired executable capabilities."""
from __future__ import annotations
import hashlib, sys, tempfile
from pathlib import Path
from typing import Any
from .assets import AssetLibrary

class CapabilityAcquirer:
    def __init__(self, workspace: str | Path, library: AssetLibrary, sandbox=None):
        from embodied_codex.kernel.sandbox import default_sandbox
        self.workspace, self.library = Path(workspace).resolve(), library
        self.sandbox = sandbox or default_sandbox()
        self.sandbox.require()

    def acquire(self, *, source_path: str, name: str, purpose: str, description: str,
                validation_command: list[str], evidence: list[str], provenance: dict[str, Any]) -> dict[str, Any]:
        source = Path(source_path).resolve()
        try: source.relative_to(self.workspace)
        except ValueError as exc: raise ValueError(
            "capability source must be a NEW .py file in the OpenHands workspace; "
            "for an existing capability:// asset, use read_asset then materialize_capability"
        ) from exc
        if not source.is_file() or source.suffix != ".py": raise ValueError("capability source must be one Python file")
        data = source.read_bytes()
        if len(data) > 2_000_000: raise ValueError("capability source is too large")
        if not validation_command or validation_command[0] not in {"python", "python3"}:
            raise ValueError("validation command must invoke isolated Python")
        with tempfile.TemporaryDirectory(prefix="roboforge-capability-") as temporary:
            candidate = Path(temporary) / "capability.py"; candidate.write_bytes(data)
            argv = [sys.executable, "-I", str(candidate), *validation_command[1:]]
            result = self.sandbox.run(argv, cwd=temporary,
                env={"PATH": str(Path(sys.executable).parent), "PYTHONNOUSERSITE": "1"},
                read_only_paths=[candidate, Path(sys.executable).resolve().parents[1]],
                read_write_paths=[temporary], timeout_seconds=120, max_output_bytes=200_000)
        if result.returncode != 0 or result.timed_out:
            raise ValueError(f"capability validation failed: {result.stderr[-2000:]}")
        digest = hashlib.sha256(data).hexdigest()
        implementation = {"language": "python", "sha256": digest,
            "source": data.decode("utf-8"), "validation_command": validation_command,
            "validation_stdout": result.stdout[-2000:], "sandbox_backend": result.backend}
        saved = self.library.register("capabilities", name=name, purpose=purpose,
            description=description, applicability=None, evidence=evidence,
            provenance=provenance, usage=f"Materialize source sha256 {digest} in the Controller workspace and import it.",
            implementation=implementation)
        self.library.audit("acquire", asset_id=saved["asset_id"], source_sha256=digest,
            validation_command=validation_command, sandbox_backend=result.backend)
        return saved

    def materialize(self, asset_id: str, destination: str, *, session_id: str | None = None) -> dict[str, Any]:
        if not self.library.was_read(asset_id, session_id=session_id):
            raise ValueError("capability must be read before materialization")
        asset = self.library.read(asset_id, session_id=session_id)
        if not asset_id.startswith(("capability://", "capabilitie://")):
            raise ValueError("asset is not a capability")
        implementation = asset.get("implementation") or {}
        source = str(implementation.get("source") or "").encode("utf-8")
        expected = str(implementation.get("sha256") or "")
        if not source or hashlib.sha256(source).hexdigest() != expected:
            raise ValueError("capability source digest is invalid")
        target = (self.workspace / destination).resolve()
        try: target.relative_to(self.workspace)
        except ValueError as exc: raise ValueError("destination must be in workspace") from exc
        if target.suffix != ".py": raise ValueError("capability destination must be a Python file")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
        self.library.audit("materialize", asset_id=asset_id, destination=str(target.relative_to(self.workspace)),
            source_sha256=expected)
        return {"asset_id": asset_id, "destination": str(target), "source_sha256": expected}
