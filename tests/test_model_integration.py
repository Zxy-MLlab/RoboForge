import json

import pytest

from embodied_codex.model import OpenAIModel, ResponsesHistory


class _Responses:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        return value


class _Client:
    def __init__(self, responses):
        self.responses = _Responses(responses)


def _model(client, *, max_history_chars=300_000):
    model = object.__new__(OpenAIModel)
    model.api_key = "test"
    model.base_url = "https://api.openai.com/v1"
    model.model = "gpt-5.6-sol"
    model.reasoning_effort = "high"
    model.reasoning_context = "all_turns"
    model.max_tokens = 8000
    model.timeout = 120
    model.total_response_timeout = 120
    model.retry_delays = ()
    model.provider = "openai"
    model.max_history_chars = max_history_chars
    model.client = client
    model.history = ResponsesHistory(max_chars=max_history_chars)
    model.audit_log = []
    return model


def _reasoning(identifier, encrypted="ciphertext", *, status="completed"):
    return {"type": "reasoning", "id": identifier, "status": status,
            "summary": [], "content": [], "encrypted_content": encrypted}


def _call(identifier, name):
    return {"type": "function_call", "id": f"fc-{identifier}",
            "call_id": identifier, "name": name, "arguments": "{}",
            "status": "completed", "caller": "assistant"}


def _response(identifier, output, **extra):
    return {"id": identifier, "model": "gpt-5.6-sol-2026-08-01",
            "status": "completed", "output": output, **extra}


def test_responses_request_and_audit_are_stateless():
    client = _Client([_response("resp-1", [_call("call-1", "inspect_execution")],
                                        usage={"input_tokens": 12,
                                               "output_tokens": 8})])
    model = _model(client)
    result = model.decide(
        messages=[{"role": "system", "content": "system"},
                  {"role": "user", "content": "task"}],
        tools=[{"type": "function", "function": {
            "name": "inspect_execution", "description": "inspect",
            "parameters": {"type": "object"}}}],
    )
    request = client.responses.calls[0]
    assert "previous_response_id" not in request
    assert request["reasoning"] == {"effort": "high", "context": "all_turns"}
    assert request["include"] == ["reasoning.encrypted_content"]
    assert request["tools"][0]["name"] == "inspect_execution"
    assert result["tool_calls"][0]["id"] == "call-1"
    audit = result["audit"]
    assert audit["requested_model"] == "gpt-5.6-sol"
    assert audit["effective_model"] == "gpt-5.6-sol-2026-08-01"
    assert audit["reasoning_context"] == "all_turns"
    assert audit["response_status"] == "completed"
    assert audit["previous_response_id_used"] is None
    assert audit["history_item_count"] == 2
    assert audit["serialized_history_bytes"] > 0
    assert audit["reasoning_items_replayed"] == 0


def test_three_turn_stateless_encrypted_reasoning_replay():
    client = _Client([
        _response("resp-1", [_reasoning("rs-1", "encrypted-one"),
                              _call("call-a", "get_value")]),
        _response("resp-2", [_reasoning("rs-2", "encrypted-two"),
                              _call("call-b", "get_value")]),
        _response("resp-3", [_reasoning("rs-3", "encrypted-three"),
                              {"type": "message", "id": "msg-3",
                               "status": "completed", "phase": "final_answer",
                               "role": "assistant", "content": [
                                   {"type": "output_text", "text": "42"}]}]),
    ])
    model = _model(client)
    model.decide(messages=[{"role": "system", "content": "system"},
                           {"role": "user", "content": "state-1"}], tools=[])
    model.record_tool_output("call-a", '{"value": 20}')
    model.decide(messages=[{"role": "system", "content": "system"},
                           {"role": "user", "content": "state-2"}], tools=[])
    model.record_tool_output("call-b", '{"value": 22}')
    result = model.decide(
        messages=[{"role": "system", "content": "system"},
                  {"role": "user", "content": "state-3"}], tools=[])

    second = client.responses.calls[1]
    third = client.responses.calls[2]
    assert all("previous_response_id" not in request
               for request in client.responses.calls)
    assert second["input"][1] == {"role": "user", "content": "state-2"}
    assert third["input"][1] == {"role": "user", "content": "state-3"}
    assert "state-1" not in json.dumps(third["input"])
    replayed_reasoning = [item for item in third["input"]
                          if item.get("type") == "reasoning"]
    assert [item["encrypted_content"] for item in replayed_reasoning] == [
        "encrypted-one", "encrypted-two"]
    assert all("status" not in item for item in replayed_reasoning)
    calls = [item for item in third["input"]
             if item.get("type") == "function_call"]
    outputs = [item for item in third["input"]
               if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in calls] == ["call-a", "call-b"]
    assert [item["call_id"] for item in outputs] == ["call-a", "call-b"]
    assert result["content"] == "42"
    assert result["audit"]["reasoning_items_replayed"] == 2
    assert result["audit"]["function_calls_replayed"] == 2
    assert result["audit"]["function_outputs_replayed"] == 2


def test_output_to_input_normalization_drops_output_only_metadata():
    reasoning = ResponsesHistory.normalize_output_item(
        _reasoning("reason-1", "opaque"))
    function = ResponsesHistory.normalize_output_item(
        _call("call-1", "read_file"))
    message = ResponsesHistory.normalize_output_item({
        "type": "message", "id": "message-output-id", "status": "completed",
        "phase": "final_answer", "role": "assistant",
        "content": [{"type": "output_text", "text": "done"}]})
    assert reasoning == {"type": "reasoning", "id": "reason-1",
                         "summary": [], "content": [],
                         "encrypted_content": "opaque"}
    assert function == {"type": "function_call", "id": "fc-call-1",
                        "call_id": "call-1", "name": "read_file",
                        "arguments": "{}", "caller": "assistant"}
    assert message == {"type": "message", "role": "assistant",
                       "content": [{"type": "output_text", "text": "done"}]}
    assert "status" not in json.dumps([reasoning, function, message])


def test_user_state_and_replayed_assistant_text_use_distinct_input_shapes():
    history = ResponsesHistory()
    current = history.set_authoritative_messages([
        {"role": "user", "content": "authoritative state"}])
    history.append_response(response_id="response", output=[{
        "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "tool preamble"}]}])
    replay = history.serialize(current)
    assert replay[0] == {"role": "user", "content": "authoritative state"}
    assert replay[1] == {"type": "message", "role": "assistant",
                         "content": [{"type": "output_text",
                                      "text": "tool preamble"}]}


def test_unknown_output_item_fails_instead_of_being_silently_dropped():
    with pytest.raises(RuntimeError, match="Unsupported Responses output item"):
        ResponsesHistory.normalize_output_item({"type": "unknown_future_item"})


def test_every_function_call_requires_exactly_one_output():
    model = _model(_Client([
        _response("resp-many", [_call("a", "first"), _call("b", "second")]),
        _response("resp-next", []),
    ]))
    model.decide(messages=[{"role": "user", "content": "state"}], tools=[])
    model.record_tool_output("a", "ok")
    with pytest.raises(RuntimeError, match="unexecuted function calls: b"):
        model.decide(messages=[{"role": "user", "content": "new"}], tools=[])
    model.record_tool_output("b", "not run", failed=True)
    with pytest.raises(RuntimeError, match="already resolved"):
        model.record_tool_output("b", "duplicate")
    model.decide(messages=[{"role": "user", "content": "new"}], tools=[])
    replay = model.client.responses.calls[1]["input"]
    assert [item["call_id"] for item in replay
            if item.get("type") == "function_call_output"] == ["a", "b"]


def test_current_state_is_replaced_after_chat_transcript_compaction():
    client = _Client([_response("resp-a", []), _response("resp-b", [])])
    model = _model(client)
    model.decide(messages=[{"role": "system", "content": "s"},
                           {"role": "user", "content": "old-state"}], tools=[])
    model.decide(messages=[{"role": "system", "content": "s"},
                           {"role": "user", "content": "new-state"}], tools=[])
    request = client.responses.calls[1]["input"]
    assert sum(item.get("role") == "user" and item.get("content") == "new-state"
               for item in request) == 1
    assert "old-state" not in json.dumps(request)


def test_compaction_preserves_recent_complete_causal_chain():
    history = ResponsesHistory(max_chars=2_000)
    history.system = {"role": "system", "content": "system"}
    for index in range(3):
        history.append_response(
            response_id=f"resp-{index}",
            output=[_reasoning(f"rs-{index}", "x" * 900),
                    _call(f"call-{index}", f"tool-{index}")])
        history.record_tool_output(f"call-{index}", f"output-{index}")
    serialized = history.serialize({"role": "user", "content": "latest-state"})
    encoded = json.dumps(serialized)
    assert "call-2" in encoded and "output-2" in encoded
    assert history.compacted_turns
    assert all("call-0" not in json.dumps(turn) for turn in history.turns)
    for turn in history.turns:
        call_ids = set(turn["calls"])
        output_ids = {item["call_id"] for item in turn["continuation_items"]
                      if item.get("type") == "function_call_output"}
        assert call_ids == output_ids


def test_multimodal_history_is_kept_as_part_of_its_causal_turn():
    history = ResponsesHistory(max_chars=1_000)
    history.append_response(response_id="old", output=[
        _reasoning("old-r", "x" * 1_000), _call("old-call", "view")])
    history.record_tool_output("old-call", "artifact://sensor/old",
                               multimodal_inputs=["data:image/png;base64,OLD"])
    history.append_response(response_id="new", output=[
        _reasoning("new-r", "new"), _call("new-call", "view")])
    history.record_tool_output("new-call", "artifact://sensor/new",
                               multimodal_inputs=["data:image/png;base64,NEW"])
    serialized = history.serialize({"role": "user", "content": "state"})
    encoded = json.dumps(serialized)
    assert "new-call" in encoded and "base64,NEW" in encoded
    assert "base64,OLD" not in encoded
    assert all("old-call" not in json.dumps(turn) for turn in history.turns)
    assert history.compacted_turns[0]["references"] == [
        "artifact://sensor/old"]
    assert history.replay_counts(serialized)["multimodal_inputs_replayed"] == 1


def test_transport_checkpoint_restores_canonical_encrypted_history():
    original = _model(_Client([_response("resp-save", [
        _reasoning("rs-save", "opaque-encrypted-blob"),
        _call("call-save", "inspect_execution")])]))
    original.decide(messages=[{"role": "user", "content": "state"}], tools=[])
    original.record_tool_output("call-save", "result")
    state = original.transport_state()
    encoded = json.dumps(state)
    assert "opaque-encrypted-blob" in encoded
    assert "previous_response_id" not in encoded
    assert "hidden_reasoning_text" not in encoded

    restored = _model(_Client([_response("resp-resumed", [])]))
    restored.restore_transport_state(state)
    restored.decide(messages=[{"role": "user", "content": "resumed-state"}],
                    tools=[])
    replay = restored.client.responses.calls[0]["input"]
    assert any(item.get("encrypted_content") == "opaque-encrypted-blob"
               for item in replay)
    assert any(item.get("call_id") == "call-save"
               and item.get("type") == "function_call_output"
               for item in replay)


def test_checkpoint_with_pending_call_fails_clearly_on_resume():
    history = ResponsesHistory()
    history.append_response(response_id="resp", output=[_call("pending", "tool")])
    restored = ResponsesHistory()
    with pytest.raises(RuntimeError, match="unresolved function calls: pending"):
        restored.restore(history.state())


def test_non_completed_response_is_audited_and_rejected():
    model = _model(_Client([{"id": "resp-incomplete", "model": "gpt-5.6-sol",
                             "status": "incomplete", "output": []}]))
    with pytest.raises(RuntimeError, match="non-completed status: incomplete"):
        model.decide(messages=[{"role": "user", "content": "x"}], tools=[])
    assert model.audit_log[-1]["finish_status"] == "incomplete"
    assert model.history.turns == []


def test_transient_retry_reuses_identical_stateless_history():
    class APIConnectionError(Exception):
        pass

    client = _Client([APIConnectionError("temporary"),
                      _response("resp-retry", [])])
    model = _model(client)
    model.retry_delays = (0,)
    model.decide(messages=[{"role": "user", "content": "state"}], tools=[])
    assert client.responses.calls[0]["input"] == client.responses.calls[1]["input"]
    assert all("previous_response_id" not in request
               for request in client.responses.calls)


def test_responses_client_without_api_is_rejected_without_chat_fallback():
    model = _model(type("NoResponses", (), {})())
    with pytest.raises(RuntimeError, match="does not support the Responses API"):
        model.decide(messages=[{"role": "user", "content": "x"}], tools=[])


def test_responses_missing_id_fails_history_audit():
    model = _model(_Client([{"model": "gpt-5.6-sol", "status": "completed",
                             "output": []}]))
    with pytest.raises(RuntimeError, match="no response id"):
        model.decide(messages=[{"role": "user", "content": "x"}], tools=[])


def test_usage_details_are_retained_in_audit():
    class Usage:
        def model_dump(self):
            return {"input_tokens": 10, "output_tokens": 4,
                    "output_tokens_details": {"reasoning_tokens": 2}}

    model = _model(_Client([_response("resp-usage", [], usage=Usage())]))
    result = model.decide(messages=[{"role": "user", "content": "x"}], tools=[])
    assert result["audit"]["usage"]["output_tokens_details"][
        "reasoning_tokens"] == 2
    assert result["audit"]["reasoning_tokens"] == 2
