import pytest

from embodied_codex.model import OpenAIModel


class _Responses:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


class _Client:
    def __init__(self, responses):
        self.responses = _Responses(responses)


def _model(client):
    model = object.__new__(OpenAIModel)
    model.api_key = "test"
    model.base_url = "https://api.openai.com/v1"
    model.model = "gpt-5.6-sol"
    model.reasoning_effort = "high"
    model.max_tokens = 8000
    model.timeout = 120
    model.total_response_timeout = 120
    model.retry_delays = ()
    model.provider = "openai"
    model.client = client
    model.previous_response_id = None
    model._sent_message_count = 0
    model.audit_log = []
    return model


def test_responses_function_call_and_audit_metadata():
    client = _Client([{
        "id": "resp-1", "model": "gpt-5.6-sol-2026-08-01",
        "status": "completed", "usage": {"input_tokens": 12, "output_tokens": 8},
        "output": [{"type": "function_call", "call_id": "call-1",
                     "name": "inspect_execution", "arguments": "{\"ref\":\"evidence://x\"}"}],
    }])
    model = _model(client)
    result = model.decide(
        messages=[{"role": "system", "content": "system"},
                  {"role": "user", "content": "task"}],
        tools=[{"type": "function", "function": {
            "name": "inspect_execution", "description": "inspect",
            "parameters": {"type": "object", "properties": {"ref": {"type": "string"}}}}}],
    )
    request = client.responses.calls[0]
    assert request["model"] == "gpt-5.6-sol"
    assert request["reasoning"] == {"effort": "high"}
    assert request["tools"][0]["name"] == "inspect_execution"
    assert request["tools"][0]["parameters"]["type"] == "object"
    assert result["tool_calls"] == [{"id": "call-1", "name": "inspect_execution",
                                     "arguments": "{\"ref\":\"evidence://x\"}"}]
    assert result["audit"] == {
        "provider": "openai", "requested_model": "gpt-5.6-sol",
        "effective_model": "gpt-5.6-sol-2026-08-01", "reasoning_effort": "high",
        "response_id": "resp-1", "usage": {"input_tokens": 12, "output_tokens": 8},
        "finish_status": "completed", "tool_call_count": 1}
    assert model.audit_log == [result["audit"]]


def test_responses_previous_response_continuation_uses_tool_output():
    client = _Client([
        {"id": "resp-1", "model": "gpt-5.6-sol", "status": "completed",
         "output": [{"type": "function_call", "call_id": "call-1",
                      "name": "list_files", "arguments": "{}"}]},
        {"id": "resp-2", "model": "gpt-5.6-sol", "status": "completed",
         "output": [{"type": "message", "content":
                     [{"type": "output_text", "text": "done"}]}]},
    ])
    model = _model(client)
    first = [{"role": "system", "content": "system"},
             {"role": "user", "content": "state-1"}]
    model.decide(messages=first, tools=[])
    second = [*first, {"role": "assistant", "content": "", "tool_calls":
                       [{"id": "call-1", "function": {"name": "list_files", "arguments": "{}"}}]},
              {"role": "tool", "tool_call_id": "call-1", "content": "{\"files\": []}"}]
    result = model.decide(messages=second, tools=[])
    request = client.responses.calls[1]
    assert request["previous_response_id"] == "resp-1"
    assert request["input"] == [
        {"role": "user", "content": "state-1"},
        {"type": "function_call_output", "call_id": "call-1",
         "output": "{\"files\": []}"},
    ]
    assert result["content"] == "done"
    assert model.previous_response_id == "resp-2"


def test_responses_client_without_api_is_rejected_without_chat_fallback():
    model = _model(type("NoResponses", (), {})())
    with pytest.raises(RuntimeError, match="does not support the Responses API"):
        model.decide(messages=[{"role": "user", "content": "x"}], tools=[])


def test_responses_missing_id_fails_continuation_audit():
    model = _model(_Client([{"model": "gpt-5.6-sol", "status": "completed", "output": []}]))
    with pytest.raises(RuntimeError, match="no response id"):
        model.decide(messages=[{"role": "user", "content": "x"}], tools=[])


def test_usage_details_are_retained_in_audit():
    class Usage:
        def model_dump(self):
            return {"input_tokens": 10, "output_tokens": 4,
                    "output_tokens_details": {"reasoning_tokens": 2}}

    model = _model(_Client([{"id": "resp-usage", "model": "gpt-5.6-sol",
                             "status": "completed", "usage": Usage(), "output": []}]))
    result = model.decide(messages=[{"role": "user", "content": "x"}], tools=[])
    assert result["audit"]["usage"]["output_tokens_details"]["reasoning_tokens"] == 2
