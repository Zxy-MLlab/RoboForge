import io
import json
import os
from pathlib import Path
import stat
import tarfile
import zipfile

import pytest

from embodied_codex.kernel.capability_manager import CapabilityError, CapabilityManager, ExtractionLimits
from embodied_codex.kernel.cas import ContentAddressedStore
from embodied_codex.kernel.workspace import PersistentWorkspace


class _Adapter:
    pass


def _manager(tmp_path, limits=None):
    workspace = PersistentWorkspace(tmp_path / "run" / "workspace", require_sandbox=False)
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=workspace,
                                adapter=_Adapter(), extraction_limits=limits)
    return manager, workspace


def _zip(path: Path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            archive.writestr(name, data)


def _tar(path: Path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _make(kind, path, entries):
    (_zip if kind == "zip" else _tar)(path, entries)


@pytest.mark.parametrize("kind", ["zip", "tar"])
@pytest.mark.parametrize("case", ["ratio", "total", "single", "count"])
def test_archive_rejects_each_resource_limit(tmp_path, kind, case):
    settings = {
        "ratio": (ExtractionLimits(max_files=10, max_total_bytes=10000,
                                   max_file_bytes=10000, max_compression_ratio=2),
                  [("dense.txt", b"x" * 4096)]),
        "total": (ExtractionLimits(max_files=10, max_total_bytes=10,
                                   max_file_bytes=10, max_compression_ratio=1000),
                  [("one.bin", os.urandom(6)), ("two.bin", os.urandom(6))]),
        "single": (ExtractionLimits(max_files=10, max_total_bytes=100,
                                    max_file_bytes=10, max_compression_ratio=1000),
                   [("large.bin", os.urandom(11))]),
        "count": (ExtractionLimits(max_files=2, max_total_bytes=100,
                                   max_file_bytes=100, max_compression_ratio=1000),
                  [("one", b"1"), ("two", b"2"), ("three", b"3")]),
    }
    limits, entries = settings[case]
    manager, workspace = _manager(tmp_path, limits)
    archive = workspace.root / f"{case}.{kind}"
    _make(kind, archive, entries)
    destination = workspace.root / "out"
    destination.mkdir()
    (destination / "keep.txt").write_text("original")
    with pytest.raises(CapabilityError):
        manager.unpack(archive.name, "out")
    assert [path.name for path in destination.iterdir()] == ["keep.txt"]
    assert (destination / "keep.txt").read_text() == "original"
    assert not list(workspace.root.glob(".out.staging-*"))


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_archive_limits_stream_and_preserve_destination_on_failure(tmp_path, kind):
    limits = ExtractionLimits(max_files=2, max_total_bytes=8, max_file_bytes=6,
                              max_compression_ratio=20)
    manager, workspace = _manager(tmp_path, limits)
    archive = workspace.root / f"bundle.{kind}"
    make = _zip if kind == "zip" else _tar
    make(archive, [("one.txt", b"1234567")])
    destination = workspace.root / "out"
    destination.mkdir()
    (destination / "keep.txt").write_text("original")
    with pytest.raises(CapabilityError):
        manager.unpack(archive.name, "out")
    assert (destination / "keep.txt").read_text() == "original"
    assert not list(workspace.root.glob(".out.staging-*"))


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_archive_normal_and_path_link_rejection(tmp_path, kind):
    limits = ExtractionLimits(max_files=10, max_total_bytes=1024,
                              max_file_bytes=512, max_compression_ratio=100)
    manager, workspace = _manager(tmp_path, limits)
    make = _zip if kind == "zip" else _tar
    archive = workspace.root / f"normal.{kind}"
    make(archive, [("nested/file.txt", b"ok"), ("other.txt", b"longer")])
    result = manager.unpack(archive.name, "normal-out")
    assert sorted(result["files"]) == ["nested/file.txt", "other.txt"]
    assert (workspace.root / "normal-out/nested/file.txt").read_text() == "ok"

    traversal = workspace.root / f"traversal.{kind}"
    make(traversal, [("../escape.txt", b"no")])
    with pytest.raises(CapabilityError):
        manager.unpack(traversal.name, "traversal-out")
    assert not (workspace.root / "escape.txt").exists()

    absolute = workspace.root / f"absolute.{kind}"
    make(absolute, [("/absolute.txt", b"no")])
    with pytest.raises(CapabilityError):
        manager.unpack(absolute.name, "absolute-out")
    assert not (workspace.root / "absolute-out").exists()


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_archive_actual_written_bytes_are_limited(tmp_path, kind, monkeypatch):
    manager, workspace = _manager(tmp_path, ExtractionLimits(
        max_files=10, max_total_bytes=10, max_file_bytes=10,
        max_compression_ratio=1000))
    archive_path = workspace.root / f"lying.{kind}"
    _make(kind, archive_path, [("small.bin", b"x")])
    if kind == "zip":
        monkeypatch.setattr(zipfile.ZipFile, "open",
            lambda *args, **kwargs: io.BytesIO(b"x" * 11))
    else:
        monkeypatch.setattr(tarfile.TarFile, "extractfile",
            lambda *args, **kwargs: io.BytesIO(b"x" * 11))
    with pytest.raises(CapabilityError):
        manager.unpack(archive_path.name, "out")
    assert not (workspace.root / "out").exists()


def test_archive_atomic_commit_failure_restores_original_directory(tmp_path, monkeypatch):
    manager, workspace = _manager(tmp_path)
    destination = workspace.root / "out"
    destination.mkdir()
    (destination / "keep.txt").write_text("original")
    staging = workspace.root / ".out.staging-test"
    staging.mkdir()
    (staging / "new.txt").write_text("new")
    monkeypatch.setattr(os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("fsync failed")))
    with pytest.raises(OSError, match="fsync failed"):
        manager._replace_directory(staging, destination)
    assert (destination / "keep.txt").read_text() == "original"
    assert not (destination / "new.txt").exists()
    assert not list(workspace.root.glob(".out.rollback-*"))


@pytest.mark.parametrize("kind,special", [
    ("zip", "symlink"), ("zip", "fifo"),
    ("tar", "symlink"), ("tar", "hardlink"), ("tar", "fifo"),
])
def test_archive_rejects_links_and_special_files(tmp_path, kind, special):
    manager, workspace = _manager(tmp_path, ExtractionLimits(max_compression_ratio=1000))
    archive_path = workspace.root / f"special.{kind}"
    if kind == "zip":
        with zipfile.ZipFile(archive_path, "w") as archive:
            info = zipfile.ZipInfo("unsafe")
            file_type = stat.S_IFLNK if special == "symlink" else stat.S_IFIFO
            info.create_system = 3
            info.external_attr = (file_type | 0o600) << 16
            archive.writestr(info, "target")
    else:
        with tarfile.open(archive_path, "w:gz") as archive:
            info = tarfile.TarInfo("unsafe")
            if special == "symlink":
                info.type, info.linkname = tarfile.SYMTYPE, "target"
            elif special == "hardlink":
                info.type, info.linkname = tarfile.LNKTYPE, "target"
            else:
                info.type = tarfile.FIFOTYPE
            archive.addfile(info)
    with pytest.raises(CapabilityError):
        manager.unpack(archive_path.name, "out")
    assert not (workspace.root / "out").exists()


def test_workspace_index_is_metadata_only_and_changed_file_is_batched(tmp_path, monkeypatch):
    cas = ContentAddressedStore(tmp_path / "shared-cas")
    workspace = PersistentWorkspace(tmp_path / "run/workspace", require_sandbox=False, cas=cas)
    workspace.write_file("a.txt", "a")
    workspace.write_file("b.txt", "b")
    calls = {"put": 0, "put_many": 0, "save": 0, "digest": 0}
    original_put, original_put_many = cas.put, cas.put_many
    original_save, original_digest = cas._save_index, cas.digest

    def put(*args, **kwargs):
        calls["put"] += 1
        return original_put(*args, **kwargs)

    def put_many(*args, **kwargs):
        calls["put_many"] += 1
        return original_put_many(*args, **kwargs)

    def save(*args, **kwargs):
        calls["save"] += 1
        return original_save(*args, **kwargs)

    def digest(*args, **kwargs):
        calls["digest"] += 1
        return original_digest(*args, **kwargs)

    monkeypatch.setattr(cas, "put", put)
    monkeypatch.setattr(cas, "put_many", put_many)
    monkeypatch.setattr(cas, "_save_index", save)
    monkeypatch.setattr(cas, "digest", digest)
    workspace.index()
    workspace.index()
    assert calls == {"put": 0, "put_many": 0, "save": 0, "digest": 0}

    workspace.write_file("a.txt", "changed")
    assert calls["put"] == 0
    assert calls["put_many"] == 1
    assert calls["save"] == 1
    assert calls["digest"] <= 3  # changed source, copy verification, no unchanged files

    before = dict(calls)
    workspace.snapshot()
    workspace.snapshot()
    assert calls["put"] == before["put"]
    assert calls["put_many"] == before["put_many"]
    assert calls["save"] == before["save"]
    assert calls["digest"] == before["digest"]


def test_many_file_commit_has_one_cas_index_save(tmp_path, monkeypatch):
    cas = ContentAddressedStore(tmp_path / "shared-cas")
    workspace = PersistentWorkspace(tmp_path / "run/workspace", require_sandbox=False, cas=cas)
    saves = []
    original = cas._save_index
    monkeypatch.setattr(cas, "_save_index", lambda value: (saves.append(len(value)), original(value)))
    workspace.apply({f"files/{index}.txt": str(index) for index in range(1000)})
    assert len(saves) == 1
    assert len(list(cas.blob_root.glob("*/*"))) == 1000

