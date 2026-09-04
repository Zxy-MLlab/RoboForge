from pathlib import Path

from roboforge.path_gate import evaluate


def test_path_gate_blocks_history_and_ancestor_listing(tmp_path: Path) -> None:
    workspace = tmp_path / "current" / "workspace"
    history = tmp_path / "history"
    workspace.mkdir(parents=True)
    history.mkdir()
    direct = evaluate(
        {"working_dir": str(workspace), "tool_input": {"command": f"cat {history}/old/controller.py"}},
        workspace=workspace,
        forbidden=[history],
    )
    ancestor = evaluate(
        {"working_dir": str(workspace), "tool_input": {"command": f"find {tmp_path}"}},
        workspace=workspace,
        forbidden=[history],
    )
    assert direct["decision"] == "deny"
    assert ancestor["decision"] == "deny"


def test_path_gate_allows_workspace_and_source_reads(tmp_path: Path) -> None:
    workspace = tmp_path / "current" / "workspace"
    history = tmp_path / "history"
    workspace.mkdir(parents=True)
    history.mkdir()
    result = evaluate(
        {"working_dir": str(workspace), "tool_input": {"command": "cat controllers/controller.py"}},
        workspace=workspace,
        forbidden=[history],
    )
    assert result["decision"] == "allow"


def test_path_gate_blocks_credential_environment_reads(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    commands = [
        'env | grep -i "APEX_API_KEY"',
        "printenv",
        "echo $MODEL_API_TOKEN",
        "python -c 'import os; print(os.environ)'",
        "tr '\\0' '\\n' < /proc/self/environ",
        "export -p",
    ]
    for command in commands:
        result = evaluate(
            {"working_dir": str(workspace), "tool_input": {"command": command}},
            workspace=workspace,
            forbidden=[],
        )
        assert result["decision"] == "deny", command


def test_path_gate_allows_scoped_non_secret_environment_assignments(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = evaluate(
        {
            "working_dir": str(workspace),
            "tool_input": {
                "command": "env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q"
            },
        },
        workspace=workspace,
        forbidden=[],
    )
    assert result["decision"] == "allow"

    socket_read = evaluate(
        {
            "working_dir": str(workspace),
            "tool_input": {
                "command": (
                    "python -c \"import os; "
                    "print(os.environ.get('ROBOFORGE_RPC_SOCKET'))\""
                )
            },
        },
        workspace=workspace,
        forbidden=[],
    )
    assert socket_read["decision"] == "allow"
