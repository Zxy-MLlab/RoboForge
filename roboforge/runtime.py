from __future__ import annotations

import os
from pathlib import Path

from openhands.sdk import Tool
from openhands.sdk.tool import ToolAnnotations, register_tool, list_registered_tools
from openhands.tools.file_editor import (
    FileEditorAction,
    FileEditorObservation,
    FileEditorTool,
)
from openhands.tools.file_editor.definition import TOOL_DESCRIPTION
from openhands.tools.file_editor.impl import FileEditorExecutor
from openhands.tools.terminal import TerminalTool
from openhands.tools.grep import GrepTool
from openhands.tools.glob import GlobTool

from .openhands_tools import create_embodied_tools
from .service import ExperimentService
from .assets import AssetLibrary
from .asset_tools import create_asset_tools

class ConfinedFileEditorExecutor(FileEditorExecutor):
    """Keep all reads and writes inside the OpenHands workspace."""
    def __init__(self, workspace_root: str, **kwargs):
        super().__init__(workspace_root=workspace_root, **kwargs)
        self.workspace_root = Path(workspace_root).resolve()
    def __call__(self, action, conversation=None):
        from openhands.tools.file_editor import FileEditorObservation
        path = Path(action.path).resolve()
        try: path.relative_to(self.workspace_root)
        except ValueError:
            return FileEditorObservation.from_text(
                text=f"Path is outside the OpenHands workspace: {path}",
                command=action.command, is_error=True)
        return super().__call__(action, conversation)


def register_spike_tools(
    service: ExperimentService,
    *,
    workspace: str | Path,
    controller_path: str | Path,
    asset_root: str | Path | None = None,
) -> list[Tool]:
    """Register the frozen spike surface with OpenHands.

    The file editor is the upstream OpenHands implementation, confined to the
    configured workspace. This permits normal coding work such as creating a
    reusable capability module while physical execution remains outside the
    coding workspace.
    """

    workspace_path = Path(workspace).resolve()
    controller = Path(controller_path).resolve()
    try:
        controller.relative_to(workspace_path)
    except ValueError as exc:
        raise ValueError("Controller must be inside the OpenHands workspace") from exc

    file_editor = FileEditorTool(
        action_type=FileEditorAction,
        observation_type=FileEditorObservation,
        description=(
            f"{TOOL_DESCRIPTION}\n\n"
            f"Workspace: {workspace_path}\n"
            "All reads and writes must remain inside this workspace."
        ),
        annotations=ToolAnnotations(
            title="file_editor",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=False,
        ),
        executor=ConfinedFileEditorExecutor(
            workspace_root=str(workspace_path),
        ),
    )
    registered = set(list_registered_tools())
    # Per-conversation tools capture service/workspace state and must replace
    # any prior process-global registration rather than retaining stale state.
    register_tool("file_editor", file_editor)
    # These are upstream OpenHands generic coding tools; RoboForge does not
    # reimplement shell, grep, or glob semantics.
    if "terminal" not in registered: register_tool("terminal", TerminalTool)
    if "grep" not in registered: register_tool("grep", GrepTool)
    if "glob" not in registered: register_tool("glob", GlobTool)

    library = AssetLibrary(asset_root) if asset_root else None
    embodied = create_embodied_tools(
        service,
        controller,
        asset_library=library,
        artifact_dir=workspace_path,
    )
    asset_tools = create_asset_tools(library, service, str(workspace_path)) if library else []
    for tool in [*embodied, *asset_tools]: register_tool(tool.name, tool)

    return [
        Tool(name="file_editor"),
        Tool(name="terminal", params={"env": {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PYTHONNOUSERSITE": "1",
        }}), Tool(name="grep"), Tool(name="glob"),
        *(Tool(name=tool.name) for tool in [*embodied, *asset_tools]),
    ]
