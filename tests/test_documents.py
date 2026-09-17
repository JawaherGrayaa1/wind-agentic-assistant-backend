"""
tests/test_documents.py
Integration tests for the document editing pipeline.
"""
from __future__ import annotations

import os
import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient

_tmp = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_PATH"] = _tmp
os.environ["AGENT_PLANNER"] = "rules"

from erp_agent.api import app, runtime  # noqa: E402

client = TestClient(app)


class TestDatabase:
    def test_seed_documents_exist(self):
        docs = runtime.db.list_documents()
        # Seed guarantees at least 3 docs; other tests may add more
        assert len(docs) >= 3
        ids = {d["doc_id"] for d in docs}
        assert "DOC-101" in ids and "DOC-102" in ids and "DOC-103" in ids

    def test_get_document_found(self):
        doc = runtime.db.get_document("DOC-101")
        assert doc is not None
        assert doc["title"] == "Client Invoice 101"

    def test_get_document_not_found(self):
        assert runtime.db.get_document("DOC-999") is None

    def test_search_by_title(self):
        results = runtime.db.search_documents("Invoice")
        assert any(d["doc_id"] == "DOC-101" for d in results)

    def test_search_by_content(self):
        results = runtime.db.search_documents("SLA")
        assert any(d["doc_id"] == "DOC-103" for d in results)

    def test_search_no_match(self):
        assert runtime.db.search_documents("XYZNOTEXIST") == []

    def test_update_document(self):
        original = runtime.db.get_document("DOC-103")
        updated = runtime.db.update_document("DOC-103", content="Revised SLA terms.")
        assert updated["content"] == "Revised SLA terms."
        assert updated["previous_content"] == original["content"]
        assert runtime.db.get_document("DOC-103")["content"] == "Revised SLA terms."

    def test_update_not_found(self):
        assert runtime.db.update_document("DOC-999", content="ghost") is None


class TestDocumentTools:
    def test_get_document_found(self):
        r = runtime.tools.get("get_document").handler(doc_id="DOC-101")
        assert r["found"] is True and r["document"]["doc_id"] == "DOC-101"

    def test_get_document_not_found(self):
        r = runtime.tools.get("get_document").handler(doc_id="DOC-000")
        assert r["found"] is False

    def test_search_documents(self):
        r = runtime.tools.get("search_documents").handler(query="invoice")
        assert r["count"] >= 1

    def test_edit_document_tool(self):
        r = runtime.tools.get("edit_document").handler(doc_id="DOC-102", content="Updated PO.")
        assert r["updated"] is True and r["document"]["content"] == "Updated PO."

    def test_edit_not_found(self):
        r = runtime.tools.get("edit_document").handler(doc_id="DOC-888", content="ghost")
        assert r["updated"] is False


class TestAgentRouting:
    def _run(self, text):
        return runtime.run("test-s", "test-u", text)

    def test_get_document(self):
        r = self._run("show me document DOC-101")
        assert r["status"] == "completed"
        assert "DOC-101" in r["reply"] or "Invoice" in r["reply"]

    def test_search_documents(self):
        r = self._run("search documents for invoice")
        assert r["status"] == "completed"

    def test_edit_requires_approval(self):
        r = self._run("edit document DOC-101 with content: new invoice text")
        assert r["status"] == "approval_required"
        assert r["approval_token"].startswith("appr-")

    def test_edit_approval_flow(self):
        r = self._run("edit document DOC-102 with content: Fresh PO for next quarter.")
        token = r["approval_token"]
        approved = runtime.approve(token)
        assert approved["status"] == "completed"
        assert runtime.db.get_document("DOC-102")["content"] == "Fresh PO for next quarter."

    def test_double_approve_fails(self):
        import uuid
        # Use a unique doc ID per run to avoid UNIQUE constraint conflicts
        unique_id = f"DOC-T{uuid.uuid4().hex[:6].upper()}"
        runtime.db.create_document(unique_id, "Temp Doc", "memo", "original content", "draft")
        r = self._run(f"edit document {unique_id} with content: one-time change.")
        token = r["approval_token"]
        first = runtime.approve(token)
        assert first["status"] == "completed"
        second = runtime.approve(token)
        assert second["status"] == "error"


class TestDocumentEndpoints:
    def test_list(self):
        resp = client.get("/v1/documents")
        assert resp.status_code == 200 and len(resp.json()) >= 3

    def test_list_search(self):
        resp = client.get("/v1/documents?q=purchase")
        assert resp.status_code == 200
        assert any(d["doc_id"] == "DOC-102" for d in resp.json())

    def test_get_found(self):
        resp = client.get("/v1/documents/DOC-101")
        assert resp.status_code == 200
        assert resp.json()["doc_id"] == "DOC-101"

    def test_get_not_found(self):
        assert client.get("/v1/documents/DOC-999").status_code == 404

    def test_patch(self):
        resp = client.patch("/v1/documents/DOC-101", json={"content": "Patched from Angular."})
        assert resp.status_code == 200
        assert resp.json()["content"] == "Patched from Angular."

    def test_patch_not_found(self):
        assert client.patch("/v1/documents/DOC-777", json={"content": "ghost"}).status_code == 404

    def test_chat_get_document(self):
        resp = client.post("/v1/chat", json={"session_id": "s1", "user_id": "u1", "text": "show document DOC-102"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert "DOC-102" in data["reply"] or "Purchase" in data["reply"]

    def test_create_document_endpoint(self):
        resp = client.post("/v1/documents", json={
            "title": "Real Supplier Invoice",
            "doc_type": "invoice",
            "content": "Invoice #9876: 10x Server Racks @ 500 TND = 5000 TND",
            "doc_id": "DOC-701",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["doc_id"] == "DOC-701"
        assert data["title"] == "Real Supplier Invoice"

        # Verify it can be retrieved
        get_resp = client.get("/v1/documents/DOC-701")
        assert get_resp.status_code == 200
        assert get_resp.json()["content"] == "Invoice #9876: 10x Server Racks @ 500 TND = 5000 TND"

    def test_upload_document_endpoint(self):
        content = b"Contract terms:\n1. 1-year warranty\n2. 24/7 technical assistance."
        resp = client.post(
            "/v1/documents/upload?title=Custom+Warranty+Contract&doc_type=contract",
            files={"file": ("contract.txt", content, "text/plain")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "Custom Warranty Contract"
        assert "1-year warranty" in data["content"]

    def test_smart_edit_preview_and_apply(self):
        unique_id = f"DOC-SE{uuid.uuid4().hex[:6].upper()}"
        # Create a document first
        client.post("/v1/documents", json={
            "doc_id": unique_id,
            "title": "Quarterly Quote",
            "doc_type": "quote",
            "content": "Quote for 5x Laptops: 10000 TND.",
        })

        # Test preview mode (apply_immediately=False)
        preview_resp = client.post(f"/v1/documents/{unique_id}/smart-edit", json={
            "instruction": "content: Quote for 5x Laptops: 9000 TND with 10% discount.",
            "apply_immediately": False,
        })
        assert preview_resp.status_code == 200
        preview_data = preview_resp.json()
        assert preview_data["applied"] is False
        assert "9000 TND" in preview_data["edited_content"]
        # Ensure DB was not changed yet
        assert client.get(f"/v1/documents/{unique_id}").json()["content"] == "Quote for 5x Laptops: 10000 TND."

        # Test apply mode (apply_immediately=True)
        apply_resp = client.post(f"/v1/documents/{unique_id}/smart-edit", json={
            "instruction": "content: Quote for 5x Laptops: 9000 TND with 10% discount.",
            "apply_immediately": True,
        })
        assert apply_resp.status_code == 200
        assert apply_resp.json()["applied"] is True
        assert "9000 TND" in client.get(f"/v1/documents/{unique_id}").json()["content"]

    def test_delete_document_endpoint(self):
        client.post("/v1/documents", json={
            "doc_id": "DOC-703",
            "title": "To Delete",
            "doc_type": "note",
            "content": "Temporary content",
        })
        del_resp = client.delete("/v1/documents/DOC-703")
        assert del_resp.status_code == 200
        assert del_resp.json()["deleted"] is True

        # Verify 404 on get
        assert client.get("/v1/documents/DOC-703").status_code == 404

    def test_chat_smart_edit_approval_flow(self):
        client.post("/v1/documents", json={
            "doc_id": "DOC-704",
            "title": "Client Bill",
            "doc_type": "invoice",
            "content": "Invoice Total: 500 TND.",
        })
        resp = client.post("/v1/chat", json={
            "session_id": "s-smart", "user_id": "u1",
            "text": "smart edit DOC-704 with content: Invoice Total: 450 TND after loyalty reduction."
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "approval_required"
        token = data["approval_token"]

        approve_resp = client.post(f"/v1/approvals/{token}")
        assert approve_resp.status_code == 200
        assert approve_resp.json()["status"] == "completed"

        doc = client.get("/v1/documents/DOC-704").json()
        assert "450 TND" in doc["content"]



