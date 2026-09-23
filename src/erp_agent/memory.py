from __future__ import annotations

import re
from typing import Any
from .db import Database


class DynamicContextManager:
    """Token-budgeted dynamic memory manager for multi-turn conversation context and entity tracking."""

    LONG_TERM_MEMORY_TRIGGERS = (
        "remember", "memory", "memories", "preference", "prefer", "preferred",
        "usual", "as always", "as usual", "default", "what do you know about me",
        "what did i tell you", "you know my", "mon choix", "ma préférence",
        "mes préférences", "comme d'habitude", "souviens-toi",
    )
    DOCUMENT_CONTEXT_TRIGGERS = (
        "content", "text", "contract", "what is in", "tell me about", "show me",
        "read", "lire", "summarize", "résume", "edit", "modifier", "change",
        "this file", "that file", "ce fichier", "ce document", "cette facture",
    )

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
        query: str | None = None,
    ) -> dict[str, Any]:
        """Build context while injecting expensive memory only when the query needs it."""
        # 1. Session Entities
        session_state = self.db.get_session_state(session_id)
        entities = {
            "active_invoice_id": session_state.get("active_invoice_id"),
            "active_bc_id": session_state.get("active_bc_id"),
            "active_product_id": session_state.get("active_product_id"),
            "active_doc_id": doc_id or session_state.get("active_doc_id"),
            "active_client": session_state.get("active_client"),
            "active_financial_document_id": session_state.get("active_financial_document_id"),
            "active_financial_document_type": session_state.get("active_financial_document_type"),
        }

        # 2. Token-Budgeted Conversation History
        raw_messages = self.db.recent_messages(session_id, limit=self.max_turns * 2)
        budgeted_messages = self._budget_messages(raw_messages)

        # 3. Active Document (if any)
        active_doc = None
        target_doc_id = doc_id or entities.get("active_doc_id")
        if target_doc_id and self.requires_document_context(query, doc_id):
            active_doc = self._compact_document(self.db.get_document(target_doc_id))

        # 4. Long-term user memories
        memories = self._bounded_memories(self.db.memories(user_id)) if self.requires_long_term_memory(query) else {}

        return {
            "session_entities": entities,
            "messages": budgeted_messages,
            "memories": memories,
            "active_document": active_doc,
        }

    @classmethod
    def requires_long_term_memory(cls, query: str | None) -> bool:
        lower = (query or "").strip().lower()
        return bool(lower) and any(trigger in lower for trigger in cls.LONG_TERM_MEMORY_TRIGGERS)

    @classmethod
    def requires_document_context(cls, query: str | None, doc_id: str | None = None) -> bool:
        lower = (query or "").strip().lower()
        return bool(lower) and any(trigger in lower for trigger in cls.DOCUMENT_CONTEXT_TRIGGERS)

    @staticmethod
    def _bounded_memories(memories: dict[str, str], max_chars: int = 3000) -> dict[str, str]:
        selected: dict[str, str] = {}
        used = 0
        for key, value in memories.items():
            item_size = len(str(key)) + len(str(value)) + 2
            if selected and used + item_size > max_chars:
                break
            selected[str(key)] = str(value)
            used += item_size
        return selected

    @staticmethod
    def _compact_document(document: dict[str, Any] | None, max_content_chars: int = 6000) -> dict[str, Any] | None:
        if not document:
            return None
        compact = dict(document)
        content = str(compact.get("content") or "")
        if len(content) > max_content_chars:
            compact["content"] = content[:max_content_chars] + " ...[document content truncated]"
        return compact

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
