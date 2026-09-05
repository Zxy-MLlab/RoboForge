"""OpenHands lifecycle adapter for the canonical development plane."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def build_conversation(*, llm: Any, workspace: str | Path, persistence_dir: str | Path,
                       max_iterations: int = 80, max_budget_per_run: float | None = None,
                       hook_config: Any = None, callbacks: list[Any] | None = None):
    """Build exactly one official LocalConversation with generic tools.

    Trial execution remains an ordinary Terminal command.  No robot-specific
    tool or secondary model loop is introduced here.
    """
    from .. import create_openhands_conversation
    root = Path(workspace).resolve()
    entrypoint = root / "controllers" / "controller.py"
    from ..workspace.project import ProjectWorkspace
    ProjectWorkspace(root).initialize()
    return create_openhands_conversation(
        llm=llm, workspace=root, persistence_dir=Path(persistence_dir),
        service=None, controller_path=entrypoint, asset_root=None,
        max_iterations=max_iterations, max_budget_per_run=max_budget_per_run,
        hook_config=hook_config, callbacks=callbacks,
        terminal_env={"ROBOFORGE_WORKSPACE": str(root), "PYTHONPATH": str(root)},
    )
