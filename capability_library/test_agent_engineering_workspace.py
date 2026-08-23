from pathlib import Path

import pytest

from agent_engineering_workspace import (
    AgentEngineeringWorkspace,
    EngineeringWorkspaceError,
)


def test_engineering_workspace_writes_runs_tests_and_reads_only_mounted_evidence(tmp_path):
    evidence = tmp_path / "sensor_evidence"
    evidence.mkdir()
    (evidence / "trace.json").write_text('{"reward_hidden": true, "rgb": "frame-1"}')
    workspace = AgentEngineeringWorkspace(
        tmp_path / "work", read_roots={"runs": evidence}, timeout_sec=20,
    )
    workspace.write("planner.py", "def next_gain(progress):\n    return .3 if progress < .002 else .5\n")
    workspace.write(
        "test_planner.py",
        "from planner import next_gain\n\n"
        "def test_stall_gain():\n    assert next_gain(.0003) == .3\n",
    )
    result = workspace.run(["pytest", "-q", "test_planner.py"])
    assert result["success"] is True
    assert "1 passed" in result["stdout"]
    artifact = workspace.inspect_artifact("runs", "trace.json")
    assert '"rgb"' in artifact["content"]
    listing = workspace.list_files("workspace", "*.py")
    assert {item["path"] for item in listing["files"]} == {
        "planner.py", "test_planner.py",
    }


def test_engineering_workspace_hides_host_repo_secrets_and_blocks_escape(tmp_path):
    workspace = AgentEngineeringWorkspace(tmp_path / "work")
    result = workspace.run([
        "python", "-c",
        "import os; print('key=' + str('APEX_API_KEY' in os.environ)); "
        "print('repo=' + str(os.path.exists('/data/zxy/embodied_frontier/evaluation')))",
    ])
    assert result["success"] is True
    assert "key=False" in result["stdout"]
    assert "repo=False" in result["stdout"]
    with pytest.raises(EngineeringWorkspaceError, match="escapes"):
        workspace.write("../outside.py", "bad")
    with pytest.raises(EngineeringWorkspaceError, match="first argv"):
        workspace.run(["bash", "-lc", "whoami"])


def test_engineering_workspace_binary_artifact_returns_metadata(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "rollout.mp4").write_bytes(b"video")
    workspace = AgentEngineeringWorkspace(
        tmp_path / "work", read_roots={"runs": evidence},
    )
    report = workspace.inspect_artifact("runs", "rollout.mp4")
    assert report["binary"] is True
    assert report["bytes"] == 5
