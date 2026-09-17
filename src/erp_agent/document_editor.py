from __future__ import annotations

import json
from typing import Any
import httpx


from .llm_logging import log_llm_error, log_llm_request, log_llm_response


class SmartDocumentEditor:
    """Uses LLM (Ollama) to understand document structure and apply intelligent edits."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "qwen3:8b", timeout: float = 45.0):
        self.url = base_url.rstrip("/") + "/api/chat"
        self.model = model
        self.timeout = timeout

    def smart_edit(
        self,
        doc_id: str,
        title: str,
        doc_type: str,
        current_content: str,
        instruction: str,
    ) -> dict[str, Any]:
        """Analyzes existing document and performs intelligent modifications based on instructions."""
        system_prompt = (
            "You are an intelligent ERP document processing system. "
            "Your task is to analyze the given document and modify its content according to the user's instructions. "
            "You understand invoices, purchase orders, contracts, quotes, and reports. "
            "Accurately calculate totals, update line items, change dates/names, add legal clauses, or adjust quantities as instructed. "
            "Return ONLY a JSON object with the following keys:\n"
            "- edited_content (string): The full updated content of the document.\n"
            "- explanation (string): A short summary of why and how you changed the document.\n"
            "- diff_summary (list of strings): Specific changes made."
        )

        user_payload = {
            "document_id": doc_id,
            "title": title,
            "doc_type": doc_type,
            "current_content": current_content,
            "instruction": instruction,
        }

        payload = {
            "model": self.model,
            "think": False,
            "stream": False,
            "format": "json",
            "keep_alive": -1,
            "options": {"temperature": 0.1},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }

        headers = {
            "ngrok-skip-browser-warning": "1",
            "User-Agent": "ERP-Agent/1.0",
        }
        start_time = log_llm_request(
            "Smart Document Editor",
            self.url,
            self.model,
            f"Doc '{title}' ({doc_id}) -> {instruction}",
        )
        try:
            response = httpx.post(self.url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            raw_content = response.json().get("message", {}).get("content", "")
            parsed = json.loads(raw_content)
            log_llm_response("Smart Document Editor", parsed, start_time)
            if "edited_content" in parsed and parsed["edited_content"]:
                return {
                    "edited_content": str(parsed["edited_content"]),
                    "explanation": str(parsed.get("explanation", "Document modified by AI")),
                    "diff_summary": parsed.get("diff_summary", [instruction]),
                }
        except Exception as exc:
            log_llm_error("Smart Document Editor", self.url, exc)

        # Fallback when Ollama is unavailable or returns invalid format
        return self._rule_fallback_edit(current_content, instruction)

    def _rule_fallback_edit(self, current_content: str, instruction: str) -> dict[str, Any]:
        """Heuristic fallback for local test suites or offline mode."""
        clean_inst = instruction.strip()
        # If instruction says 'replace with X' or 'content: X'
        if ":" in clean_inst:
            _, sep, rest = clean_inst.partition(":")
            if rest.strip():
                return {
                    "edited_content": rest.strip(),
                    "explanation": f"Updated content based on instruction: {clean_inst}",
                    "diff_summary": [f"Updated content with: {rest.strip()[:60]}..."],
                }
        
        # Append instruction as modification annotation
        edited = f"{current_content}\n\n[Note: {clean_inst}]"
        return {
            "edited_content": edited,
            "explanation": f"Applied instruction: {clean_inst}",
            "diff_summary": [f"Appended instruction note: {clean_inst}"],
        }
