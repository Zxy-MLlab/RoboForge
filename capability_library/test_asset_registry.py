import json

from asset_registry import (
    asset_id_for,
    find_assets,
    record_asset_reuse,
    register_asset,
    validate_asset_manifest,
)


def _asset():
    return {
        "asset_id": asset_id_for("tool", "RGB-D backprojection"),
        "kind": "tool",
        "name": "RGB-D backprojection",
        "status": "development_validated",
        "source_urls": ["https://example.org/rgbd"],
        "tested_tasks": ["libero_goal:task_1"],
        "reused_tasks": [],
        "sensors": ["RGB-D"],
        "current_task_data_used": False,
        "privileged_state_used": False,
    }


def test_manifest_rejects_current_task_or_privileged_state():
    asset = _asset()
    asset["current_task_data_used"] = True
    assert validate_asset_manifest(asset)


def test_register_find_and_record_reuse(tmp_path):
    path = tmp_path / "library.json"
    asset = _asset()
    assert register_asset(asset, library_path=str(path))["success"]
    assert find_assets(kind="tool", sensor="RGB-D", library_path=str(path))[0]["asset_id"] == asset["asset_id"]
    result = record_asset_reuse(asset["asset_id"], "libero_goal:task_2", outcome="success", evidence="episode.json", library_path=str(path))
    assert result["success"]
    payload = json.loads(path.read_text())
    assert payload["assets"][0]["status"] == "cross_task_reused"
    assert payload["events"][-1]["event"] == "asset_reused"
