from __future__ import annotations

import re
from typing import Any
from .db import Database


class DynamicContextManager:
    """Token-budgeted dynamic memory manager for multi-turn conversation context and entity tracking."""

    def __init__(
        self,
        db: Database,
        max_history_chars: int = 10000,  # ~2,500 tokens budget for conversation history
        max_turns: int = 20,
    ):
        self.db = db
        self.max_history_chars = max_history_chars
        self.max_turns = max_turns

    def context(
        self,
        session_id: str,
        user_id: str,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        """Assembles a budget-constrained context payload containing entities, trimmed history, and user memories."""
        # 1. Session Entities
        session_state = self.db.get_session_state(session_id)
        entities = {
            "active_invoice_id": session_state.get("active_invoice_id"),
            "active_bc_id": session_state.get("active_bc_id"),
            "active_product_id": session_state.get("active_product_id"),
            "active_doc_id": doc_id or session_state.get("active_doc_id"),
            "active_client": session_state.get("active_client"),
        }

        # 2. Token-Budgeted Conversation History
        raw_messages = self.db.recent_messages(session_id, limit=self.max_turns * 2)
        budgeted_messages = self._budget_messages(raw_messages)

        # 3. Active Document (if any)
        active_doc = None
        target_doc_id = doc_id or entities.get("active_doc_id")
        if target_doc_id:
            active_doc = self.db.get_document(target_doc_id)

        # 4. Long-term user memories
        memories = self.db.memories(user_id)

        return {
            "session_entities": entities,
            "messages": budgeted_messages,
            "memories": memories,
            "active_document": active_doc,
        }

    def _budget_messages(self, raw_messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Filters and cleans recent messages to fit safely inside the token/character budget."""
        accumulated_chars = 0
        selected: list[dict[str, str]] = []

        # Iterate in reverse (newest turns first)
        for msg in reversed(raw_messages):
            role = msg.get("role", "user")
            content = str(msg.get("content", "")).strip()

            # Clean oversized raw tool dumps if present in history
            cleaned_content = self._sanitize_message_content(content)
            char_count = len(cleaned_content)

            if accumulated_chars + char_count > self.max_history_chars and len(selected) >= 4:
                # Budget reached, stop adding older turns
                break

            selected.append({"role": role, "content": cleaned_content})
            accumulated_chars += char_count

            if len(selected) >= self.max_turns:
                break

        # Re-order chronologically (oldest to newest)
        return list(reversed(selected))

    @staticmethod
    def _sanitize_message_content(content: str) -> str:
        """Sanitizes past messages to avoid bloat from raw JSON or oversized text."""
        # Truncate single past message if it exceeds 1,500 chars
        if len(content) > 1500:
            return content[:1400] + " ...[truncated in history]"
        return content


class Memory(DynamicContextManager):
    """Backward-compatible wrapper for DynamicContextManager."""
    pass
