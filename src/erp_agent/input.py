from __future__ import annotations

import json

import httpx


from .llm_logging import log_llm_error, log_llm_request, log_llm_response


class OllamaTranscriptionNormalizer:
    """Correct noisy ASR output while preserving the user's language and intent."""

    def __init__(self, base_url: str, model: str, timeout: float = 30.0):
        self.url = base_url.rstrip("/") + "/api/chat"
        self.model = model
        self.timeout = timeout

    def normalize(self, transcript: str) -> str:
        if not transcript.strip():
            return transcript

        system = (
            "You clean speech-to-text output for an ERP assistant. "
            "The input may be Tunisian Arabic written in Arabic or Latin characters, "
            "French, English, or a mixture. Correct likely transcription errors and "
            "add only minimal punctuation. Preserve the user's language, intent, names, "
            "numbers, product IDs, and document IDs. Do not answer the request, translate "
            "it, or invent missing information. Return only a JSON object with one key: "
            "text."
        )
        payload = {
            "model": self.model,
            "think": False,
            "stream": False,
            "format": "json",
            "keep_alive": -1,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": transcript},
            ],
        }

        headers = {
            "ngrok-skip-browser-warning": "1",
            "User-Agent": "ERP-Agent/1.0",
        }
        start_time = log_llm_request("ASR Normalizer", self.url, self.model, transcript)
        try:
            response = httpx.post(self.url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            parsed = json.loads(content)
            log_llm_response("ASR Normalizer", parsed, start_time)
            normalized = parsed.get("text", "")
            if isinstance(normalized, str) and normalized.strip():
                return normalized.strip()
        except Exception as exc:
            log_llm_error("ASR Normalizer", self.url, exc)

        return transcript


class NoOpTranscriptionNormalizer:
    def normalize(self, transcript: str) -> str:
        return transcript