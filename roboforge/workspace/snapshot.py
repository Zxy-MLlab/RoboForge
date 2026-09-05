from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..candidate_bundle import CandidateBundleStore


def snapshot_workspace(workspace: str | Path, entrypoint: str | Path,
                       repository: str | Path, *, runtime_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Freeze the complete editable Controller/Robot Stack into a Bundle."""
    store = CandidateBundleStore(repository)
    return store.freeze(workspace=workspace, entrypoint=entrypoint,
                        runtime_metadata=runtime_metadata)
