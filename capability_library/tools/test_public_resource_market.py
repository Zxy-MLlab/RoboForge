import json

from public_resource_market import (
    record_acquisition_event,
    register_public_resource_tools,
    search_public_resources,
)


def test_search_records_sources_and_results(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        if "github" in url:
            return {"items": [{"full_name": "org/tool", "html_url": "https://github.com/org/tool", "default_branch": "main"}]}
        return [{"id": "org/model", "sha": "abc", "pipeline_tag": "image-text-to-text"}]

    ledger = tmp_path / "events.jsonl"
    result = search_public_resources("general grasp policy", ledger_path=str(ledger), fetch_json=fetch)
    assert result["success"]
    assert len(result["results"]) == 2
    assert len(calls) >= 2
    assert any("duckduckgo" in url for url in calls)
    assert any("arxiv" in url for url in calls)
    event = json.loads(ledger.read_text())
    assert event["event"] == "public_resource_search"
    assert event["query"] == "general grasp policy"


def test_search_failure_is_audited(tmp_path):
    def fetch(_url):
        raise OSError("offline")

    ledger = tmp_path / "events.jsonl"
    result = search_public_resources("sam", ledger_path=str(ledger), fetch_json=fetch)
    assert not result["success"]
    assert "all public searches failed" in result["reason"]
    assert len(ledger.read_text().splitlines()) == 1


def test_button_failure_preserves_specific_query_then_expands_domain(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        if "github" in url:
            return {"items": [{"full_name": "org/visual-servo", "html_url": "https://github.com/org/visual-servo"}]}
        return []

    result = search_public_resources(
        "button press visual servo",
        ledger_path=str(tmp_path / "events.jsonl"),
        fetch_json=fetch,
    )
    assert result["attempted_queries"][0] == "button press visual servo"
    assert "visual servo robot" in result["attempted_queries"]
    assert "button+press+visual+servo" in calls[0]


def test_record_event_and_thea_registration(tmp_path):
    ledger = tmp_path / "events.jsonl"
    result = record_acquisition_event({"event": "asset_rejected", "asset_id": "x"}, ledger_path=str(ledger))
    assert result["success"]
    assert json.loads(ledger.read_text())["asset_id"] == "x"

    class Registry:
        def __init__(self):
            self.tools = []

        def tool(self, **kwargs):
            def decorate(fn):
                self.tools.append((kwargs["name"], fn))
                return fn

            return decorate

    registry = Registry()
    register_public_resource_tools(registry)
    assert {name for name, _fn in registry.tools} == {
        "search_public_embodied_resources",
        "record_capability_acquisition_event",
    }
