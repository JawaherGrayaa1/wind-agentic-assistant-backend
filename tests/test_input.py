import json

from erp_agent import input as input_module
from erp_agent.input import OllamaTranscriptionNormalizer


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": self.content}}


def test_ollama_normalizer_returns_structured_text(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(json.dumps({"text": "Quel est le stock de P-100 ?"}))

    monkeypatch.setattr(input_module.httpx, "post", fake_post)
    normalizer = OllamaTranscriptionNormalizer("http://ollama:11434", "qwen3:8b")

    result = normalizer.normalize("chnowa stock mta3 P-100")

    assert result == "Quel est le stock de P-100 ?"
    assert calls[0][1]["json"]["format"] == "json"
    assert calls[0][1]["json"]["messages"][1]["content"] == "chnowa stock mta3 P-100"


def test_ollama_normalizer_falls_back_to_raw_transcript(monkeypatch):
    def failing_post(*args, **kwargs):
        raise input_module.httpx.ConnectError("offline")

    monkeypatch.setattr(input_module.httpx, "post", failing_post)
    normalizer = OllamaTranscriptionNormalizer("http://ollama:11434", "qwen3:8b")

    assert normalizer.normalize("chnowa stock mta3 P-100") == "chnowa stock mta3 P-100"