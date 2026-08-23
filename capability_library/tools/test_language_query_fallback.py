import json
import importlib.util
from pathlib import Path


_CONTROLLER = Path("/data/zxy/vla_agentic_harness_pi0_libero/scripts/run_groundingdino_controller.py")
_SPEC = importlib.util.spec_from_file_location("groundingdino_controller_for_test", _CONTROLLER)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
extract_language_detector_queries = _MODULE.extract_language_detector_queries


def test_language_query_fallback_uses_only_instruction_when_provider_disconnects(monkeypatch, tmp_path):
    def disconnected(*args, **kwargs):
        raise OSError("remote provider disconnected")

    monkeypatch.setenv("APEX_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", disconnected)
    language = "pick up the black bowl next to the cookie box and place it on the plate"
    queries = extract_language_detector_queries(language, tmp_path)
    assert queries == ["black bowl", "cookie box", "plate"]
    audit = json.loads((tmp_path / "language_detector_queries.json").read_text())
    assert audit["fallback_used"] is True
    assert audit["privileged_inputs_used"] == []
