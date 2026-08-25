#!/usr/bin/env python3
"""Download and verify the exact checkpoint set used by the LIBERO Adapter."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


ASSETS = {
    "groundingdino": {
        "filename": "groundingdino_swint_ogc.pth",
        "url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
        "sha256": "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
    },
    "sam": {
        "filename": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sha256": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    },
    "graspnet": {
        "filename": "graspnet-checkpoint-rs.tar",
        "gdrive_id": "1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk",
        "sha256": "60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868",
    },
}


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_http(url: str, target: Path):
    request = Request(url, headers={"User-Agent": "RoboForge-checkpoint-installer/1"})
    with urlopen(request, timeout=120) as response, target.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)


def acquire(name: str, destination: Path, verify_only: bool):
    asset = ASSETS[name]; target = destination / asset["filename"]
    if target.is_file() and sha256(target) == asset["sha256"]:
        return {"name": name, "path": str(target), "sha256": asset["sha256"], "status": "verified"}
    if verify_only:
        raise RuntimeError(f"missing or invalid checkpoint: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".partial",
                                                   dir=destination)
    os.close(descriptor); temporary = Path(temporary_name)
    try:
        if "url" in asset:
            download_http(asset["url"], temporary)
        else:
            import gdown
            result = gdown.download(id=asset["gdrive_id"], output=str(temporary), quiet=False)
            if not result:
                raise RuntimeError(f"Google Drive download failed: {name}")
        actual = sha256(temporary)
        if actual != asset["sha256"]:
            raise RuntimeError(f"checkpoint checksum mismatch for {name}: {actual}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"name": name, "path": str(target), "sha256": asset["sha256"], "status": "downloaded"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path,
                        default=Path(__file__).resolve().parents[1] / "checkpoints")
    parser.add_argument("--only", choices=tuple(ASSETS), action="append")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    for name in args.only or ASSETS:
        print(acquire(name, args.destination.resolve(), args.verify_only))


if __name__ == "__main__":
    main()
