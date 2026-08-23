import json

from web_research_broker import consult_external_model


def test_external_answer_is_unverified_and_audited(tmp_path):
    ledger = tmp_path / "events.jsonl"
    result = consult_external_model(
        "How can RGB-D be transformed into robot coordinates?",
        provider="test-model",
        ask_fn=lambda _q: "Use calibrated intrinsics and extrinsics.",
        ledger_path=str(ledger),
    )
    assert result["success"]
    assert result["verified"] is False
    assert result["action_selection_allowed"] is False
    event = json.loads(ledger.read_text())
    assert event["event"] == "external_model_consultation"
    assert event["answer_sha256"]


def test_unconfigured_provider_does_not_fail_open(tmp_path):
    result = consult_external_model("find a tool", ledger_path=str(tmp_path / "events"))
    assert result["success"] is False
