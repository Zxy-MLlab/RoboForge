import base64
import csv
import hashlib
import io
import json
import platform
from pathlib import Path
import sys
import sysconfig
import zipfile

import pytest

from embodied_codex.kernel.assets import AssetError, CapabilityLibrary
from embodied_codex.kernel.cas import ContentAddressedStore
from embodied_codex.kernel.runtime_environment import RuntimeEnvironmentManager
from embodied_codex.kernel.sandbox import default_sandbox


def _wheel(root: Path, name: str, version: str) -> tuple[Path, str]:
    normalized = name.replace("-", "_")
    filename = f"{normalized}-{version}-py3-none-any.whl"
    path = root / filename
    files = {
        f"{normalized}/__init__.py": f"__version__ = {version!r}\n",
        f"{normalized}-{version}.dist-info/METADATA":
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        f"{normalized}-{version}.dist-info/WHEEL":
            "Wheel-Version: 1.0\nGenerator: roboforge-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    rows = []
    for relative, content in files.items():
        payload = content.encode()
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        rows.append((relative, f"sha256={digest}", str(len(payload))))
    record = f"{normalized}-{version}.dist-info/RECORD"
    rows.append((record, "", ""))
    stream = io.StringIO()
    csv.writer(stream, lineterminator="\n").writerows(rows)
    files[record] = stream.getvalue()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, content in files.items():
            archive.writestr(relative, content)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _spec(workspace: Path, wheel: Path, digest: str, name: str, version: str):
    return {
        "python": {"implementation": platform.python_implementation().casefold(),
                   "version": platform.python_version(),
                   "abi": str(sysconfig.get_config_var("SOABI") or "none")},
        "dependencies": [{"name": name, "version": version,
            "artifact": {"path": wheel.relative_to(workspace).as_posix(),
                         "filename": wheel.name, "kind": "wheel", "sha256": digest}}],
        "accelerator": "cpu",
        "platform": {"system": platform.system().casefold(),
                     "machine": platform.machine().casefold()},
    }


def _schemas():
    return ({"type": "object", "properties": {}, "additionalProperties": False},
            {"type": "object", "properties": {"version": {"type": "string"}},
             "required": ["version"], "additionalProperties": False})


def test_conflicting_dependencies_get_distinct_reusable_runtimes(tmp_path):
    workspace = tmp_path / "workspace"
    wheels = workspace / "wheels"
    wheels.mkdir(parents=True)
    first_wheel, first_sha = _wheel(wheels, "conflictdep", "1.0")
    second_wheel, second_sha = _wheel(wheels, "conflictdep", "2.0")
    (workspace / "one.py").write_text(
        "def run(payload):\n import conflictdep\n return {'version': conflictdep.__version__}\n")
    (workspace / "two.py").write_text(
        "def run(payload):\n import conflictdep\n return {'version': conflictdep.__version__}\n")
    cas = ContentAddressedStore(tmp_path / "assets/_cas")
    manager = RuntimeEnvironmentManager(tmp_path / "assets/runtimes", cas=cas,
                                        python=sys.executable, sandbox=default_sandbox())
    library = CapabilityLibrary(tmp_path / "assets/tools", workspace,
        python=sys.executable, sandbox=default_sandbox(), cas=cas,
        runtime_environment_manager=manager)
    input_schema, output_schema = _schemas()
    one = library.register_tool(name="one", source_path="one.py", description="one",
        input_schema=input_schema, output_schema=output_schema,
        runtime_spec=_spec(workspace, first_wheel, first_sha, "conflictdep", "1.0"))
    two = library.register_tool(name="two", source_path="two.py", description="two",
        input_schema=input_schema, output_schema=output_schema,
        runtime_spec=_spec(workspace, second_wheel, second_sha, "conflictdep", "2.0"))
    assert library.run(one["tool_id"], {}) == {"version": "1.0"}
    assert library.run(two["tool_id"], {}) == {"version": "2.0"}
    manifests = [library.inspect(item["tool_id"])["manifest"] for item in (one, two)]
    assert (manifests[0]["runtime_environment"]["runtime_id"]
            != manifests[1]["runtime_environment"]["runtime_id"])
    assert len(list((tmp_path / "assets/runtimes").glob("sha256-*"))) == 2
    assert "conflictdep" not in sys.modules

    # A fresh manager resolves the same immutable environment without rebuilding it.
    receipts = {path.parent.name: path.stat().st_mtime_ns
                for path in (tmp_path / "assets/runtimes").glob("sha256-*/runtime.json")}
    restarted = RuntimeEnvironmentManager(tmp_path / "assets/runtimes", cas=cas,
                                           python=sys.executable, sandbox=default_sandbox())
    for manifest in manifests:
        environment = restarted.ensure(manifest["runtime_environment"])
        assert environment.reused is True
    assert receipts == {path.parent.name: path.stat().st_mtime_ns
                        for path in (tmp_path / "assets/runtimes").glob("sha256-*/runtime.json")}


def test_same_runtime_spec_builds_one_environment_and_isolates_harness(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = ("def run(payload):\n"
              " import socket\n"
              " try:\n  import embodied_codex; private=False\n"
              " except ModuleNotFoundError:\n  private=True\n"
              " try:\n  socket.socket(); network=False\n"
              " except PermissionError:\n  network=True\n"
              " return {'version': str(private and network)}\n")
    (workspace / "one.py").write_text(source)
    (workspace / "two.py").write_text(source)
    cas = ContentAddressedStore(tmp_path / "assets/_cas")
    manager = RuntimeEnvironmentManager(tmp_path / "assets/runtimes", cas=cas,
                                        python=sys.executable, sandbox=default_sandbox())
    library = CapabilityLibrary(tmp_path / "assets/tools", workspace,
        python=sys.executable, sandbox=default_sandbox(), cas=cas,
        runtime_environment_manager=manager)
    input_schema, output_schema = _schemas()
    spec = manager.default_spec()
    tools = [library.register_tool(name=name, source_path=f"{name}.py", description=name,
        input_schema=input_schema, output_schema=output_schema, runtime_spec=spec)
        for name in ("one", "two")]
    assert [library.run(item["tool_id"], {}) for item in tools] == [
        {"version": "True"}, {"version": "True"}]
    ids = {library.inspect(item["tool_id"])["manifest"]["runtime_environment"]["runtime_id"]
           for item in tools}
    assert len(ids) == 1
    assert len(list((tmp_path / "assets/runtimes").glob("sha256-*"))) == 1


@pytest.mark.parametrize("mutation,match", [
    ("unpinned", "exact version"),
    ("missing", "artifact"),
    ("mismatch", "checksum"),
])
def test_runtime_dependency_spec_fails_closed(tmp_path, mutation, match):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    wheel, digest = _wheel(workspace, "strictdep", "1.0")
    spec = _spec(workspace, wheel, digest, "strictdep", "1.0")
    if mutation == "unpinned":
        spec["dependencies"][0]["version"] = "*"
    elif mutation == "missing":
        spec["dependencies"][0]["artifact"].pop("path")
    else:
        spec["dependencies"][0]["artifact"]["sha256"] = "0" * 64
    (workspace / "tool.py").write_text("def run(payload):\n return {'version': 'x'}\n")
    library = CapabilityLibrary(tmp_path / "assets/tools", workspace,
                                python=sys.executable, sandbox=default_sandbox())
    input_schema, output_schema = _schemas()
    with pytest.raises(AssetError, match=match):
        library.register_tool(name="strict", source_path="tool.py", description="strict",
            input_schema=input_schema, output_schema=output_schema, runtime_spec=spec)
