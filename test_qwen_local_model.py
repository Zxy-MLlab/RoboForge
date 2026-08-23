from qwen_local_model import parse_qwen_decision


def test_parse_tool_call():
    response = parse_qwen_decision(
        '{"text":"searching", "tool_calls":[{"id":"1","name":"search_public_embodied_resources","arguments":{"query":"grasp"}}]}'
    )
    assert response.text == "searching"
    assert response.tool_calls[0].name == "search_public_embodied_resources"
    assert response.tool_calls[0].arguments == {"query": "grasp"}


def test_malformed_output_stays_text():
    response = parse_qwen_decision("not json")
    assert response.text == "not json"
    assert response.tool_calls == []
