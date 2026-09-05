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
from openhands.tools.preset.default import register_builtins_agents
from openhands.tools.preset.planning import get_planning_tools
from openhands.tools.task import TaskToolSet

try:
    # BrowserToolSet is an upstream optional extension.  Importing it is kept
    # best-effort so a headless install still has a working coding agent.
    from openhands.tools.browser_use.definition import BrowserToolSet
except Exception:  # pragma: no cover - exercised by minimal SDK installs
    BrowserToolSet = None  # type: ignore[assignment,misc]

from .service import ExperimentService

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
    terminal_env: dict[str, str] | None = None,
) -> list[Tool]:
    """Register only OpenHands' public, generic software-engineering tools.

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
            "All reads and writes must remain inside this workspace.\n"
            "The upstream `view` command renders PNG, JPEG, GIF, and WebP "
            "files as multimodal image content. Use it directly on trial "
            "keyframes and diagnostic crops instead of converting images to "
            "ASCII or inferring appearance from pixel statistics alone."
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
    # Register only public OpenHands extensions. Browser-backed agents are
    # enabled only when the installed SDK can actually start Chromium; this
    # avoids both a fake Web capability and an unconditional hard-disable.
    browser_usable = False
    if BrowserToolSet is not None:
        try:
            browser_usable = bool(BrowserToolSet.is_usable())
        except Exception:
            browser_usable = False
    register_builtins_agents(enable_browser=browser_usable)
    planning = get_planning_tools(str(workspace_path / "PLAN.md"))

    # ``service``, ``controller`` and ``asset_root`` remain accepted for API
    # compatibility, but are deliberately not captured by LLM tools. Physical
    # work happens through the ordinary Terminal and ``python -m roboforge
    # trial``; artifacts are then ordinary Workspace files.
    del service, controller, asset_root
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PYTHONNOUSERSITE": "1",
        **dict(terminal_env or {}),
    }

    return [
        Tool(name="file_editor"),
        Tool(name="terminal", params={"env": environment}),
        Tool(name="grep"), Tool(name="glob"),
        *(tool for tool in planning if tool.name not in {"grep", "glob"}),
        Tool(name=TaskToolSet.name),
        *( [Tool(name=BrowserToolSet.name)]
           if browser_usable and BrowserToolSet is not None else []),
    ]


# Canonical name; the compatibility alias above is retained for existing
# callers while all new entry points describe this as OpenHands-native tools.
register_native_tools = register_spike_tools
