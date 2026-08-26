import hashlib
import errno
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from embodied_codex.kernel.assets import CapabilityLibrary
from embodied_codex.kernel.agent_loop import LoopBudget
from embodied_codex.kernel.campaign import CampaignAdapter, CampaignRunner
from embodied_codex.kernel.capability_manager import CapabilityError, CapabilityManager
from embodied_codex.kernel.cas import ContentAddressedStore
from embodied_codex.kernel.context import ContextBuilder
from embodied_codex.kernel.events import EventStore
from embodied_codex.kernel.runtime import ControllerRuntime
from embodied_codex.kernel.sandbox import (BubblewrapBackend, PosixSandboxBackend,
                                            UnsafeSandboxBackend)
from embodied_codex.kernel.workspace import PersistentWorkspace
from embodied_codex.fake_adapter import FakeAdapter


def _register_concurrently(asset_root: str, workspace_root: str, index: int, queue) -> None:
    library = CapabilityLibrary(Path(asset_root) / "tools", workspace_root,
                               python=sys.executable, require_runtime=False)
    result = library.register_tool(
        name="concurrent_tool",
        source_path=f"tool_{index}.py",
        description=f"concurrent implementation {index}",
        input_schema={"type": "object", "properties": {},
                      "additionalProperties": False},
        output_schema={"type": "object", "properties": {
            "value": {"type": "integer"}}, "required": ["value"],
            "additionalProperties": False},
        provenance={"origin": "internal", "producer": "test"},
    )
    queue.put(result["tool_id"])


def _put_blob_concurrently(cas_root: str, source: str, queue) -> None:
    store = ContentAddressedStore(cas_root)
    queue.put(store.put(source)["blob_uri"])


def test_posix_probe_never_claims_safe_without_real_path_confinement():
    backend = PosixSandboxBackend()
    probe = backend.probe()
    assert backend.safe is probe.available
    assert probe.features["filesystem_isolation"] is probe.available
    assert probe.features["unauthorized_read_denied"] is probe.available
    assert probe.features["unauthorized_write_denied"] is probe.available
    if backend.landlock_abi == 0:
        assert probe.available is False
        assert "filesystem" in probe.detail.casefold()


def test_bubblewrap_mounts_symlinked_interpreter_base_prefix(monkeypatch, tmp_path):
    base = tmp_path / "base"
    interpreter = base / "bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("runtime")
    environment = tmp_path / "environment"
    environment.mkdir()
    linked = environment / "python"
    linked.symlink_to(interpreter)
    monkeypatch.setattr(sys, "prefix", str(environment))
    monkeypatch.setattr(sys, "base_prefix", str(base))
    monkeypatch.setattr(sys, "executable", str(linked))

    command = BubblewrapBackend(executable="/usr/bin/bwrap")._wrapped(
        [str(linked), "-c", "pass"], tmp_path, (), ())
    bindings = list(zip(command, command[1:]))
    assert ("--ro-bind", str(base.resolve())) in bindings
    assert command.count(str(base.resolve())) >= 2


def test_workspace_snapshot_uses_cas_for_one_gib_sparse_file(tmp_path):
    workspace = PersistentWorkspace(tmp_path / "run/workspace",
                                    require_sandbox=False)
    large = workspace.root / "downloads/model.ckpt"
    large.parent.mkdir(parents=True)
    with large.open("wb") as stream:
        stream.write(b"header")
        stream.seek((1 << 30) - 1)
        stream.write(b"\0")

    first = workspace.snapshot()
    manifest = Path(first.path)
    payload = json.loads(manifest.read_text())
    entry = next(item for item in payload["files"]
                 if item["path"] == "downloads/model.ckpt")
    assert manifest.stat().st_size < 64 * 1024
    assert entry["bytes"] == 1 << 30
    assert entry["blob_uri"].startswith("cas://sha256/")
    assert "content" not in entry and "content_base64" not in entry

    blob = workspace.cas.resolve(entry["blob_uri"], verify=True)
    assert blob.stat().st_size == 1 << 30
    assert blob.stat().st_blocks * 512 < 64 * 1024 * 1024
    assert workspace.snapshot().snapshot_id == first.snapshot_id
    assert len(list(workspace.cas.blob_root.glob("*/*"))) == 1

    large.unlink()
    workspace.restore(first.snapshot_id)
    assert large.stat().st_size == 1 << 30
    with large.open("rb") as stream:
        assert stream.read(6) == b"header"


def test_workspace_command_stages_large_file_without_mutating_cas_blob(tmp_path):
    workspace = PersistentWorkspace(tmp_path / "run/workspace",
                                    sandbox=UnsafeSandboxBackend(),
                                    require_sandbox=False)
    large = workspace.root / "downloads/model.ckpt"
    large.parent.mkdir(parents=True)
    with large.open("wb") as stream:
        stream.seek((1 << 30) - 1)
        stream.write(b"\0")
    workspace.snapshot()
    blob = next(workspace.cas.blob_root.glob("*/*"))
    original_digest = workspace.cas.digest(blob)
    script = ("import os\n"
              "st=os.stat('downloads/model.ckpt')\n"
              "print(f'{st.st_ino}:{st.st_size}', flush=True)\n"
              "with open('downloads/model.ckpt','r+b') as f:\n"
              "    f.seek(0); f.write(b'changed')\n"
              "open('small.txt','w').write('staged')\n")
    result = workspace.run_command([sys.executable, "-c", script], timeout_seconds=20)
    assert result["exit_code"] == 0 and result["committed"] is True
    inode, size = result["output"].strip().splitlines()[-1].split(":")
    assert int(size) == 1 << 30
    # Reflink/sparse copy is allowed, but a mutable staged worktree must never
    # share an inode with or mutate the immutable CAS blob.
    assert int(inode) != blob.stat().st_ino
    assert workspace.cas.digest(blob) == original_digest
    with large.open("rb") as stream:
        assert stream.read(7) == b"changed"
    assert workspace.read("small.txt") == "staged"
    assert large.stat().st_size == 1 << 30


def test_staged_replacement_does_not_mutate_cas_blob(tmp_path):
    cas = ContentAddressedStore(tmp_path / "cas")
    workspace = PersistentWorkspace(tmp_path / "run/workspace", cas=cas,
                                    require_sandbox=False)
    workspace.write_file("weights.bin", "original")
    snapshot = workspace.snapshot()
    record = json.loads(Path(snapshot.path).read_text())["files"][0]
    blob = cas.resolve(record["blob_uri"], verify=True)
    workspace.write_file("weights.bin", "replacement")
    assert cas.digest(blob) == record["sha256"]
    assert workspace.read("weights.bin") == "replacement"


def test_concurrent_put_same_content_creates_one_verified_blob(tmp_path):
    source = tmp_path / "checkpoint.bin"
    source.write_bytes(b"same checkpoint")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_put_blob_concurrently,
        args=(str(tmp_path / "cas"), str(source), queue)) for _ in range(6)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    uris = [queue.get(timeout=2) for _ in processes]
    assert len(set(uris)) == 1
    store = ContentAddressedStore(tmp_path / "cas")
    assert store.resolve(uris[0], verify=True).is_file()
    assert len(list(store.blob_root.glob("*/*"))) == 1


def test_capability_versions_share_checkpoint_cas_blob(tmp_path):
    workspace = tmp_path / "workspace"
    bundle = workspace / "bundle"
    bundle.mkdir(parents=True)
    checkpoint = bundle / "weights.bin"
    with checkpoint.open("wb") as stream:
        stream.write(b"checkpoint")
        stream.seek((64 << 20) - 1)
        stream.write(b"\0")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    checkpoint_sha = digest.hexdigest()
    library = CapabilityLibrary(tmp_path / "assets/tools", workspace,
                               require_runtime=False)

    def register(value):
        (bundle / "tool.py").write_text(
            f"def run(payload):\n    return {{'value': {value}}}\n")
        return library.register_package(name="shared_checkpoint",
            bundle_path="bundle", description="CAS checkpoint package",
            input_schema={"type": "object", "properties": {},
                          "additionalProperties": False},
            output_schema={"type": "object", "properties": {
                "value": {"type": "integer"}}, "required": ["value"],
                "additionalProperties": False},
            package_spec={"kind": "model", "entrypoint": "tool.py",
                "accelerator": "cpu",
                "checkpoint_sha256": {"weights.bin": checkpoint_sha}})

    first = register(1)
    second = register(2)
    first_path = library.root / first["tool_id"].replace(":", "/")
    second_path = library.root / second["tool_id"].replace(":", "/")
    first_weights = first_path / "bundle/weights.bin"
    second_weights = second_path / "bundle/weights.bin"
    assert first["tool_id"] != second["tool_id"]
    first_manifest = json.loads((first_path / "manifest.json").read_text())
    weight_record = next(row for row in first_manifest["bundle_files"]
                         if row["path"] == "weights.bin")
    second_manifest = json.loads((second_path / "manifest.json").read_text())
    second_weight_record = next(row for row in second_manifest["bundle_files"]
                                if row["path"] == "weights.bin")
    assert weight_record["blob_uri"] == second_weight_record["blob_uri"]
    assert library.cas.digest(first_weights) == checkpoint_sha
    assert library.cas.digest(second_weights) == checkpoint_sha
    assert first_weights.stat().st_blocks * 512 < 8 << 20
    assert not any(str(tmp_path) in path.read_text()
                   for path in library.cas.root.glob("*.json"))


def test_shared_asset_version_allocation_is_process_safe(tmp_path):
    asset_root = tmp_path / "assets"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for index in range(8):
        (workspace / f"tool_{index}.py").write_text(
            f"def run(payload):\n    return {{'value': {index}}}\n")

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_register_concurrently,
        args=(str(asset_root), str(workspace), index, queue)) for index in range(8)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    identifiers = sorted(queue.get(timeout=2) for _ in processes)
    assert len(set(identifiers)) == 8
    assert identifiers == [f"concurrent_tool:v{index:03d}" for index in range(1, 9)]


def test_event_store_appends_and_recovers_only_torn_tail(tmp_path):
    store = EventStore(tmp_path / "events")
    store.commit("sample", {"index": 0})
    inode = store.path.stat().st_ino
    for index in range(1, 200):
        store.commit("sample", {"index": index})
    assert store.path.stat().st_ino == inode
    assert [row["payload"]["index"] for row in store.events()] == list(range(200))
    good_size = store.path.stat().st_size
    with store.path.open("ab") as stream:
        stream.write(b'{"event_id":"torn"')
        stream.flush()
        os.fsync(stream.fileno())
    assert len(store.events()) == 200
    assert store.path.stat().st_size == good_size
    store.commit("after_recovery", {"ok": True})
    assert store.events()[-1]["kind"] == "after_recovery"


def test_sigkill_recovery_reuses_committed_execution_without_replay(tmp_path):
    worker = tmp_path / "kill_recovery_worker.py"
    worker.write_text(r'''
import json
import os
from pathlib import Path
import signal
import sys

from embodied_codex.kernel.agent_loop import AgentLoop, LoopBudget
from embodied_codex.kernel.capability_manager import CapabilityManager
from embodied_codex.kernel.context import ContextBuilder
from embodied_codex.kernel.events import EventStore
from embodied_codex.kernel.runtime import ControllerRuntime
from embodied_codex.kernel.workspace import PersistentWorkspace

root = Path(sys.argv[1]).resolve()
mode = sys.argv[2]

class Adapter:
    instruction = "set persistent target"
    sdk_index = {"protocol": "kill-recovery-v1"}
    def __init__(self):
        self.root = root / "adapter"; self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.state = (json.loads(self.state_path.read_text()) if self.state_path.exists()
                      else {"value": 0, "actions": 0, "generation": "stable-1"})
    def _save(self):
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state)); temporary.replace(self.state_path)
    def initial_observation(self): return {"value": self.state["value"]}
    def dispatch(self, method, arguments):
        if method == "act":
            self.state["value"] = arguments["action"]["value"]
            self.state["actions"] += 1; self._save(); return {"accepted": True}
        if method == "verify": return {"verified": self.state["value"] == 1}
        if method == "observe": return {"value": self.state["value"]}
        if method == "record": return {"recorded": True}
        raise ValueError(method)
    def project_rpc_output(self, method, arguments, result): return dict(result)
    def sensor_report(self, execution):
        return {"sensor_success": self.state["value"] == 1,
                "value": self.state["value"], "actions": self.state["actions"]}
    def execution_identity(self):
        return {"episode_id": "persistent-episode",
                "environment_generation": self.state["generation"]}
    def resume_protocol(self):
        return {"supports_resume": True, "resume_token": "stable-resume-token",
                "environment_generation": self.state["generation"],
                "actions_idempotent": False, "replay_allowed": True}
    def verification_receipt(self, execution):
        return {"verified": self.state["value"] == 1 and execution.get("completed") is True,
            "controller_sha256": execution.get("program_sha256"),
            "environment_identity": self.execution_identity(),
            "episode_id": "persistent-episode",
            "environment_generation": self.state["generation"]}
    def validate_execution_receipt(self, receipt):
        return receipt.get("verified") is True and self.state["value"] == 1 \
            and receipt.get("environment_identity") == self.execution_identity()
    def register_capability(self, tool_id, function, contract): pass
    def close(self): self._save()

def call(name, arguments, identifier):
    return {"content": "", "tool_calls": [{"id": identifier, "name": name,
        "arguments": json.dumps(arguments)}]}

class FirstModel:
    def decide(self, *, messages, tools):
        context = next((json.loads(row["content"]) for row in reversed(messages)
            if row.get("role") == "user" and isinstance(row.get("content"), str)
            and row["content"].startswith("{")), {})
        paths = {row.get("path") for row in context.get("workspace", [])}
        if "controller.py" not in paths:
            return call("write_file", {"path": "controller.py", "content":
                "def run(robot):\n    robot.act({'value': 1})\n"
                "    return robot.verify('target', {})\n"}, "write")
        return call("run_controller", {}, "execute")

class ResumeModel:
    def decide(self, *, messages, tools):
        context = next((json.loads(row["content"]) for row in reversed(messages)
            if row.get("role") == "user" and isinstance(row.get("content"), str)
            and row["content"].startswith("{")), {})
        evidence = context.get("latest_evidence") or {}
        if not evidence:
            return call("run_controller", {}, "recover")
        return call("finish", {"summary": "resumed committed execution"}, "finish")

class KillAfterExecutionStore(EventStore):
    def commit(self, kind, payload, **kwargs):
        event = super().commit(kind, payload, **kwargs)
        if kind == "execution":
            os.kill(os.getpid(), signal.SIGKILL)
        return event

adapter = Adapter()
workspace = PersistentWorkspace(root / "workspace")
manager = CapabilityManager(asset_root=root / "assets", workspace=workspace,
                            adapter=adapter)
store = KillAfterExecutionStore(root / "events") if mode == "crash" \
    else EventStore(root / "events")
loop = AgentLoop(model=FirstModel() if mode == "crash" else ResumeModel(),
    workspace=workspace, adapter=adapter,
    context_builder=ContextBuilder(adapter_index=adapter.sdk_index,
        asset_registry=manager, workspace=workspace,
        initial_observation=adapter.initial_observation()),
    capability_manager=manager, runtime=ControllerRuntime(timeout_seconds=20),
    event_store=store, root=root, budget=LoopBudget(max_steps=8, max_executions=4),
    resume=True)
result = loop.run(adapter.instruction)
adapter.close()
print(json.dumps(result))
''')
    root = tmp_path / "run"
    environment = dict(os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    first = subprocess.run([sys.executable, str(worker), str(root), "crash"],
        env=environment, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=60)
    assert first.returncode == -signal.SIGKILL, first.stdout
    assert (root / "checkpoint/state.json").is_file()
    rows = EventStore(root / "events").events()
    assert len([row for row in rows if row["kind"] == "execution"]) == 1

    resumed = subprocess.run([sys.executable, str(worker), str(root), "resume"],
        env=environment, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=60)
    assert resumed.returncode == 0, resumed.stdout
    result = json.loads(resumed.stdout.splitlines()[-1])
    assert result["finished"] is True
    state = json.loads((root / "adapter/state.json").read_text())
    assert state == {"value": 1, "actions": 1, "generation": "stable-1"}
    rows = EventStore(root / "events").events()
    assert len([row for row in rows if row["kind"] == "execution"]) == 1


def test_provider_resolution_never_routes_openai_key_to_apex(monkeypatch):
    from embodied_codex.providers import ProviderConfigurationError, resolve_provider

    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.delenv("APEX_API_KEY", raising=False)
    monkeypatch.setenv("APEX_BASE_URL", "https://api.apexin.ai/v1")
    resolved = resolve_provider()
    assert resolved.provider == "openai"
    assert resolved.endpoint == "https://api.openai.com/v1"
    assert resolved.key_env == "OPENAI_API_KEY"
    assert "openai-secret" not in json.dumps(resolved.redacted())
    with pytest.raises(ProviderConfigurationError, match="explicit --provider"):
        resolve_provider(base_url="https://api.apexin.ai/v1")

    monkeypatch.setenv("APEX_API_KEY", "apex-secret")
    with pytest.raises(ProviderConfigurationError, match="provider"):
        resolve_provider()
    apex = resolve_provider(provider="apex")
    assert apex.endpoint == "https://api.apexin.ai/v1"
    assert apex.key_env == "APEX_API_KEY"


def test_only_promoted_assets_are_returned_by_default(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "tool.py").write_text("def run(payload):\n    return {'value': 1}\n")
    library = CapabilityLibrary(tmp_path / "assets/tools", workspace,
                               require_runtime=False)
    candidate = library.register_tool(
        name="quality_tool", source_path="tool.py", description="quality test",
        input_schema={"type": "object", "properties": {},
                      "additionalProperties": False},
        output_schema={"type": "object", "properties": {
            "value": {"type": "integer"}}, "required": ["value"],
            "additionalProperties": False},
        provenance={"origin": "internal", "producer": "test"})
    assert candidate["status"] == "candidate"
    assert library.search("quality") == []
    assert library.search("quality", statuses={"candidate"})[0]["tool_id"] == candidate["tool_id"]


def test_capability_download_uses_streaming_digest(monkeypatch, tmp_path):
    import embodied_codex.kernel.capability_manager as module

    payload = b"large artifact streamed in chunks"
    digest = hashlib.sha256(payload).hexdigest()

    def fake_download(url, destination):
        Path(destination).write_bytes(payload)
        return {"url": url, "path": str(destination), "bytes": len(payload),
                "sha256": digest}

    monkeypatch.setattr(module, "download_public_file", fake_download)
    workspace = PersistentWorkspace(tmp_path / "workspace", require_sandbox=False)
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=workspace,
                                adapter=FakeAdapter("download", tmp_path / "run"))
    result = manager.download("https://example.com/model.bin", "downloads/model.bin",
                              sha256=digest)
    assert result["sha256"] == digest
    assert workspace.read("downloads/model.bin") == payload.decode()


def test_capability_download_never_reads_download_target_into_memory(monkeypatch, tmp_path):
    import embodied_codex.kernel.capability_manager as module

    payload = b"streamed checkpoint payload"
    digest = hashlib.sha256(payload).hexdigest()
    workspace = PersistentWorkspace(tmp_path / "workspace", require_sandbox=False)
    target = (workspace.root / "downloads/model.bin").resolve()
    original_read_bytes = Path.read_bytes

    def fake_download(url, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(payload)
        return {"url": url, "path": str(destination), "bytes": len(payload),
                "sha256": digest}

    def reject_target_read(path):
        if path.resolve() == target:
            raise AssertionError("download target was loaded with read_bytes")
        return original_read_bytes(path)

    monkeypatch.setattr(module, "download_public_file", fake_download)
    monkeypatch.setattr(Path, "read_bytes", reject_target_read)
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=workspace,
                                adapter=FakeAdapter("download", tmp_path / "run"))
    result = manager.download("https://example.com/model.bin", "downloads/model.bin",
                              sha256=digest)
    assert result["bytes"] == len(payload)
    assert target.stat().st_size == len(payload)


def test_capability_download_checksum_mismatch_removes_target(monkeypatch, tmp_path):
    import embodied_codex.kernel.capability_manager as module

    workspace = PersistentWorkspace(tmp_path / "workspace", require_sandbox=False)
    target = workspace.root / "downloads/model.bin"

    def fake_download(url, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(b"unexpected")
        return {"url": url, "path": str(destination), "bytes": 10,
                "sha256": hashlib.sha256(b"unexpected").hexdigest()}

    monkeypatch.setattr(module, "download_public_file", fake_download)
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=workspace,
                                adapter=FakeAdapter("download", tmp_path / "run"))
    with pytest.raises(CapabilityError, match="checksum mismatch"):
        manager.download("https://example.com/model.bin", "downloads/model.bin",
                         sha256="0" * 64)
    assert not target.exists()


def test_builtin_provider_endpoints_cannot_cross_route_credentials():
    from embodied_codex.providers import ProviderConfigurationError, resolve_provider

    environment = {"OPENAI_API_KEY": "openai-secret", "APEX_API_KEY": "apex-secret"}
    openai = resolve_provider(provider="openai", environment=environment)
    apex = resolve_provider(provider="apex", environment=environment)
    assert openai.endpoint == "https://api.openai.com/v1"
    assert apex.endpoint == "https://api.apexin.ai/v1"
    with pytest.raises(ProviderConfigurationError):
        resolve_provider(provider="openai", base_url="https://api.apexin.ai/v1",
                         environment=environment)
    with pytest.raises(ProviderConfigurationError):
        resolve_provider(provider="apex", base_url="https://api.openai.com/v1",
                         environment=environment)


def test_workspace_and_capability_package_share_one_cas_protocol(tmp_path):
    shared_cas = ContentAddressedStore(tmp_path / "shared-cas")
    workspace = PersistentWorkspace(tmp_path / "run/workspace", cas=shared_cas,
                                    require_sandbox=False)
    bundle = workspace.root / "bundle"
    bundle.mkdir(parents=True)
    payload = b"shared checkpoint contents"
    checkpoint = bundle / "weights.bin"
    checkpoint.write_bytes(payload)
    checkpoint_sha = hashlib.sha256(payload).hexdigest()
    workspace_snapshot = workspace.snapshot()
    library = CapabilityLibrary(tmp_path / "assets/tools", workspace.root,
                                python=sys.executable, require_runtime=False,
                                cas=shared_cas)
    (bundle / "tool.py").write_text("def run(payload):\n    return {'value': 1}\n")
    registration = library.register_package(
        name="shared_package", bundle_path="bundle", description="shared CAS package",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object", "properties": {"value": {"type": "integer"}},
                       "required": ["value"], "additionalProperties": False},
        package_spec={"kind": "model", "entrypoint": "tool.py", "accelerator": "cpu",
                      "checkpoint_sha256": {"weights.bin": checkpoint_sha}})
    manifest = json.loads((library.root / registration["tool_id"].replace(":", "/") /
                           "manifest.json").read_text())
    snapshot_payload = json.loads(Path(workspace_snapshot.path).read_text())
    snapshot_uri = next(item["blob_uri"] for item in snapshot_payload["files"]
                        if item["path"] == "bundle/weights.bin")
    package_uri = next(item["blob_uri"] for item in manifest["bundle_files"]
                       if item["path"] == "weights.bin")
    assert snapshot_uri == package_uri
    assert shared_cas.resolve(snapshot_uri).is_file()
    assert len(list((shared_cas.root / "blobs").glob("*/*"))) == 2


def test_cas_materialize_falls_back_without_hardlink_and_preserves_blob(monkeypatch, tmp_path):
    cas = ContentAddressedStore(tmp_path / "cas")
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable")
    record = cas.put(source)
    original_link = os.link

    def reject_link(*args, **kwargs):
        raise OSError(errno.EXDEV, "cross-device")

    monkeypatch.setattr(os, "link", reject_link)
    destination = tmp_path / "materialized.bin"
    cas.materialize(record["blob_uri"], destination)
    assert destination.read_bytes() == b"immutable"
    assert cas.resolve(record["blob_uri"], verify=True).read_bytes() == b"immutable"
    monkeypatch.setattr(os, "link", original_link)


class _CampaignCase(FakeAdapter):
    def __init__(self, task, root, *, case, expected):
        self.expected = expected
        super().__init__(task, root, case=case)

    def dispatch(self, method, arguments):
        if method == "observe" and arguments.get("channel") == "proprioception":
            return {"step": len(self.actions), "proprioception": {
                "value": self.value, "target": self.expected}}
        if method == "verify":
            return {"verified": self.value == self.expected,
                    "observed_value": self.value, "expected": self.expected}
        return super().dispatch(method, arguments)

    def sensor_report(self, execution):
        return {"sensor_success": self.value == self.expected,
                "value": self.value, "expected": self.expected,
                "case": self.case, "action_log": list(self.actions)}

    def agent_evidence(self, execution, sensor_report):
        return {"observed_value": self.value, "target_value": self.expected}

    def verification_receipt(self, execution):
        return {"verified": bool(self.value == self.expected
            and execution.get("completed") is True and not execution.get("error")
            and execution.get("sensor_verification_observed") is True),
            "controller_sha256": execution.get("program_sha256"),
            "environment_identity": self.execution_identity(),
            "episode_id": self.execution_identity()["episode_id"],
            "environment_generation": self.generation}

    def validate_execution_receipt(self, receipt):
        return bool(receipt.get("verified") is True
                    and receipt.get("environment_identity") == self.execution_identity()
                    and self.value == self.expected)


def _decision(name, arguments):
    return {"content": "", "tool_calls": [{"id": os.urandom(4).hex(),
        "name": name, "arguments": json.dumps(arguments)}]}


class _EvidenceReactiveCampaignModel:
    """Selects cases itself and reacts to public failure evidence."""

    def __init__(self):
        self.saw_second_failure = False
        self.writes = 0
        self.first_passed = False
        self.second_passed = False
        self.regression_passed = False

    @staticmethod
    def _context(messages):
        for message in reversed(messages):
            if message.get("role") != "user" or not isinstance(message.get("content"), str):
                continue
            try:
                value = json.loads(message["content"])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "state" in value and "workspace" in value:
                return value
        return {}

    @staticmethod
    def _last_call(messages):
        for message in reversed(messages):
            calls = message.get("tool_calls") if isinstance(message, dict) else None
            if calls:
                return calls[-1]["function"]["name"]
        return None

    def decide(self, *, messages, tools):
        context = self._context(messages)
        last_call = self._last_call(messages)
        controller = context.get("controller")
        state = context.get("state") or {}
        selected = state.get("selected_case")
        latest = context.get("latest_evidence") or {}
        diagnostics = latest.get("diagnostics") or {}
        if controller is None:
            self.writes += 1
            return _decision("write_file", {"path": "controller.py", "content":
                "def run(robot):\n    robot.act({'type':'set_value','value':1})\n"
                "    return robot.verify('target', {})\n"})
        if last_call == "write_file":
            return _decision("run_controller", {})
        observed = diagnostics.get("observed_value")
        target = diagnostics.get("target_value")
        if observed is not None and observed != target:
            if selected == "case-002": self.saw_second_failure = True
            if last_call != "read_file":
                return _decision("read_file", {"path": "controller.py"})
            assert self.saw_second_failure
            self.writes += 1
            return _decision("write_file", {"path": "controller.py", "content":
                "def run(robot):\n    state=robot.observe('proprioception', {})\n"
                "    robot.act({'type':'set_value','value':state['proprioception']['target']})\n"
                "    return robot.verify('target', {})\n"})
        if observed is not None and observed == target:
            if selected == "case-001" and not self.first_passed:
                self.first_passed = True
                return _decision("select_case", {"case_id": "case-002"})
            if selected == "case-002" and self.saw_second_failure:
                self.second_passed = True
                return _decision("select_case", {"case_id": "case-001"})
            if selected == "case-001" and self.second_passed:
                self.regression_passed = True
        if self.regression_passed:
            return _decision("finish", {"summary": "one Controller passed every case"})
        return _decision("run_controller", {})


def test_campaign_converges_one_controller_after_second_case_failure(tmp_path):
    run_root = tmp_path / "run"
    workspace = PersistentWorkspace(run_root / "workspace")
    case_a = _CampaignCase("set the case target", run_root / "cases/A",
                           case="A", expected=1)
    case_b = _CampaignCase("set the case target", run_root / "cases/B",
                           case="B", expected=2)
    adapter = CampaignAdapter([("A", case_a), ("B", case_b)])
    manager = CapabilityManager(asset_root=tmp_path / "assets",
                                workspace=workspace, adapter=adapter)
    model = _EvidenceReactiveCampaignModel()
    loop = CampaignRunner(model=model, workspace=workspace, adapter=adapter,
        context_builder=ContextBuilder(adapter_index=adapter.sdk_index,
            asset_registry=manager, workspace=workspace,
            initial_observation=adapter.initial_observation()),
        capability_manager=manager, runtime=ControllerRuntime(timeout_seconds=20),
        event_store=EventStore(run_root / "events"), root=run_root,
        budget=LoopBudget(max_steps=20, max_executions=10), resume=False)
    try:
        result = loop.run(adapter.instruction)
    finally:
        adapter.close()
    assert result["finished"] is True
    assert model.saw_second_failure is True and model.writes == 2
    assert len(case_a.actions) == 2
    assert len(case_b.actions) == 2
    assert model.first_passed and model.second_passed and model.regression_passed
    assert result["selected_case"] == "case-001"
    assert "campaign" not in loop.state
