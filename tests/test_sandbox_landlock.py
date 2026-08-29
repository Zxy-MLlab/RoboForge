import json
import sys

import pytest

from embodied_codex.kernel.sandbox import (
    PosixSandboxBackend,
    _LL_MAKE_DIR,
    _LL_READ_DIR,
    _LL_REMOVE_DIR,
    _LL_TRUNCATE,
    _LL_WRITE_FILE,
    _landlock_path_rights,
    _landlock_rights,
)


def _run(backend, script, cwd, *, read_only=(), read_write=()):
    result = backend.run(
        [sys.executable, "-c", script],
        cwd=cwd,
        read_only_paths=read_only,
        read_write_paths=read_write,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.fixture
def backend():
    value = PosixSandboxBackend()
    if value.landlock_abi < 1:
        pytest.skip("Landlock ABI 1 is required")
    return value


def test_abi1_file_rules_exclude_directory_rights(tmp_path):
    path = tmp_path / "file"
    path.write_text("x")
    rights = _landlock_path_rights(path, _landlock_rights(1), 1)
    assert rights & _LL_WRITE_FILE
    assert not rights & (_LL_READ_DIR | _LL_MAKE_DIR | _LL_REMOVE_DIR | _LL_TRUNCATE)


def test_read_only_regular_file_is_readable_but_not_writable(backend, tmp_path):
    path = tmp_path / "read-only"
    path.write_text("secret")
    result = _run(backend, """
import json
out = {}
try:
    out['read'] = open('read-only').read() == 'secret'
except Exception:
    out['read'] = False
try:
    open('read-only', 'w').write('changed')
    out['write'] = True
except Exception:
    out['write'] = False
print(json.dumps(out))
""", tmp_path, read_only=(path,))
    assert result == {"read": True, "write": False}


def test_read_write_regular_file_is_writable(backend, tmp_path):
    path = tmp_path / "read-write"
    path.write_text("before")
    result = _run(backend, """
import json
open('read-write', 'w').write('after')
print(json.dumps(open('read-write').read()))
""", tmp_path, read_write=(path,))
    assert result == "after"


def test_read_only_directory_allows_reads_but_not_writes(backend, tmp_path):
    directory = tmp_path / "read-only-dir"
    directory.mkdir()
    (directory / "input").write_text("input")
    result = _run(backend, """
import json
out = {'read': open('read-only-dir/input').read() == 'input'}
try:
    open('read-only-dir/output', 'w').write('x')
    out['write'] = True
except Exception:
    out['write'] = False
print(json.dumps(out))
""", tmp_path, read_only=(directory,))
    assert result == {"read": True, "write": False}


def test_read_write_directory_allows_writes(backend, tmp_path):
    directory = tmp_path / "read-write-dir"
    directory.mkdir()
    result = _run(backend, """
import json
open('read-write-dir/output', 'w').write('x')
print(json.dumps(open('read-write-dir/output').read()))
""", tmp_path, read_write=(directory,))
    assert result == "x"


def test_unauthorized_file_is_denied(backend, tmp_path):
    allowed = tmp_path / "allowed"
    forbidden = tmp_path / "forbidden"
    allowed.write_text("allowed")
    forbidden.write_text("secret")
    result = _run(backend, """
import json
out = {'allowed': open('allowed').read()}
try:
    open('forbidden').read()
    out['forbidden_read'] = True
except Exception:
    out['forbidden_read'] = False
print(json.dumps(out))
""", tmp_path, read_only=(allowed,))
    assert result == {"allowed": "allowed", "forbidden_read": False}
