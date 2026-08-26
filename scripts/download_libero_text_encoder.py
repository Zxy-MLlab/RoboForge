#!/usr/bin/env python3
"""Download the pinned GroundingDINO text encoder without Hub metadata APIs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen
import uuid


REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
BASE_URL = f"https://huggingface.co/google-bert/bert-base-uncased/resolve/{REVISION}"
ASSETS = {
    "config.json": {"url": f"{BASE_URL}/config.json",
        "sha256": "7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20"},
    "pytorch_model.bin": {"url": f"{BASE_URL}/pytorch_model.bin",
        "sha256": "097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a"},
    "tokenizer.json": {"url": f"{BASE_URL}/tokenizer.json",
        "sha256": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98"},
    "tokenizer_config.json": {"url": f"{BASE_URL}/tokenizer_config.json",
        "sha256": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10"},
    "vocab.txt": {"url": f"{BASE_URL}/vocab.txt",
        "sha256": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path) -> tuple[str, int]:
    request = Request(url, headers={"User-Agent": "RoboForge-LIBERO-installer/1"})
    digest = hashlib.sha256()
    written = 0
    with urlopen(request, timeout=120) as response, target.open("xb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
            digest.update(chunk)
            written += len(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    return digest.hexdigest(), written


def _verified(destination: Path) -> list[dict] | None:
    results = []
    for filename, asset in ASSETS.items():
        target = destination / filename
        if not target.is_file() or sha256(target) != asset["sha256"]:
            return None
        results.append({"filename": filename, "sha256": asset["sha256"],
                        "status": "verified", "bytes": target.stat().st_size})
    return results


def acquire_text_encoder(destination: Path, verify_only: bool = False) -> list[dict]:
    destination = Path(destination).expanduser().resolve()
    existing = _verified(destination) if destination.is_dir() else None
    if existing is not None:
        return existing
    if verify_only:
        raise RuntimeError(f"missing or invalid text encoder: {destination}")
    if destination.exists() and not destination.is_dir():
        raise RuntimeError(f"text encoder destination is not a directory: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".",
                                    suffix=".staging", dir=destination.parent))
    results = []
    try:
        for filename, asset in ASSETS.items():
            target = staging / filename
            actual, written = _download(asset["url"], target)
            if actual != asset["sha256"]:
                raise RuntimeError(
                    f"text encoder checksum mismatch for {filename}: {actual}")
            results.append({"filename": filename, "sha256": actual,
                            "status": "downloaded", "bytes": written})

        backup = None
        if destination.exists():
            backup = destination.with_name(
                destination.name + ".backup-" + uuid.uuid4().hex)
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except Exception:
            if backup is not None and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return results
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(acquire_text_encoder(args.destination, args.verify_only), indent=2))


if __name__ == "__main__":
    main()
