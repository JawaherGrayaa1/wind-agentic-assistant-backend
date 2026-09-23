from fastapi.testclient import TestClient
from erp_agent.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "planner" in data


def test_invoice_extraction_proxy_stages_and_cleans_file(monkeypatch):
    import erp_agent.api as api_module

    captured = {}

    class FakeTool:
        def handler(self, **kwargs):
            from pathlib import Path

            captured.update(kwargs)
            assert Path(kwargs["file_path"]).is_file()
            return {
                "found": True,
                "source": "test-extractor",
                "invoice_data": {"invoice_id": "FAC-1", "items": []},
            }

    monkeypatch.setattr(api_module.runtime.tools, "get", lambda name: FakeTool())
    response = client.post(
        "/v1/invoices/extract",
        data={
            "tenant_id": "wind-erp",
            "invoice_layout": "auto",
            "document_id": "DOC-1",
        },
        files={"file": ("invoice.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["invoice_data"]["invoice_id"] == "FAC-1"
    assert captured["tenant_id"] == "wind-erp"

    from pathlib import Path

    assert not Path(captured["file_path"]).exists()


def test_chat_inventory_query():
    response = client.post(
        "/v1/chat",
        json={"session_id": "test-session-1", "user_id": "user-1", "text": "What is the stock of P-100?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test-session-1"
    assert data["status"] == "completed"
    assert "Laptop Pro" in data["reply"] or "12" in data["reply"]

def test_chat_create_order_runs_without_approval():
    response = client.post(
        "/v1/chat",
        json={"session_id": "test-session-2", "user_id": "user-1", "text": "Create order for customer"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["approval_token"] is None

def test_session_events():
    session_id = "test-session-3"
    client.post(
        "/v1/chat",
        json={"session_id": session_id, "user_id": "user-1", "text": "Disponibilité P-200"}
    )
    events_res = client.get(f"/v1/sessions/{session_id}/events")
    assert events_res.status_code == 200
    events = events_res.json()
    assert len(events) > 0
    assert any(e.get("event_type") == "user_message" for e in events)


def test_voice_chat_uses_normalized_transcript(monkeypatch):
    import erp_agent.api as api_module

    class FakeASR:
        def transcribe(self, input_path):
            return "chnowa stock mta3 P-100"

    class FakeNormalizer:
        def normalize(self, transcript):
            assert transcript == "chnowa stock mta3 P-100"
            return "What is the stock of P-100?"

    monkeypatch.setattr(api_module, "asr", FakeASR())
    monkeypatch.setattr(api_module, "transcription_normalizer", FakeNormalizer())
    session_id = "test-voice-normalized"

    response = client.post(
        "/v1/voice/chat",
        params={"session_id": session_id, "user_id": "user-1"},
        files={"audio": ("voice.wav", b"not-a-real-audio-file", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    events = client.get(f"/v1/sessions/{session_id}/events").json()
    transcription = next(event for event in events if event["event_type"] == "transcription")
    assert transcription["payload"] == {
        "raw": "chnowa stock mta3 P-100",
        "normalized": "What is the stock of P-100?",
    }


def test_chat_with_document_upload():
    doc_content = b"Supplier Agreement 2026\nPayment terms: 60 days net.\nPenalty: 1.5% per month."
    resp = client.post(
        "/v1/chat/upload",
        data={"session_id": "s-up-1", "user_id": "u1", "title": "Supplier SLA", "text": ""},
        files={"file": ("sla.txt", doc_content, "text/plain")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["doc_id"] is not None
    assert data["document"] is not None
    assert data["document"]["title"] == "Supplier SLA"
    assert "Payment terms" in data["document"]["content"]

    # Verify saved in DB and queryable by editor
    doc_id = data["doc_id"]
    get_res = client.get(f"/v1/documents/{doc_id}")
    assert get_res.status_code == 200
    assert get_res.json()["content"] == data["document"]["content"]


def test_chat_with_document_upload_and_edit_prompt():
    invoice_content = b"Invoice #999\nTotal: 1000 TND."
    resp = client.post(
        "/v1/chat/upload",
        data={
            "session_id": "s-up-2",
            "user_id": "u1",
            "title": "Invoice 999",
            "text": "edit document with content: Invoice #999\nTotal: 850 TND after 15% discount.",
        },
        files={"file": ("inv999.txt", invoice_content, "text/plain")},
    )
    assert resp.status_code == 200
    data = resp.json()
    doc_id = data["doc_id"]
    assert doc_id is not None
    assert data["status"] == "approval_required"
    token = data["approval_token"]
    assert token is not None

    # Approve the edit
    appr_res = client.post(f"/v1/approvals/{token}")
    assert appr_res.status_code == 200
    appr_data = appr_res.json()
    assert appr_data["status"] == "completed"
    assert appr_data["document"] is not None
    assert "850 TND" in appr_data["document"]["content"]

    # Verify DB has updated content for editor
    latest = client.get(f"/v1/documents/{doc_id}").json()
    assert "850 TND" in latest["content"]


def test_download_export_file(tmp_path):
    from pathlib import Path
    export_dir = Path("data/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    test_file = export_dir / "test_invoice_export.pdf"
    test_file.write_bytes(b"%PDF-1.4 test pdf content")

    resp = client.get("/v1/exports/download/test_invoice_export.pdf")
    assert resp.status_code == 200
    assert resp.content == b"%PDF-1.4 test pdf content"
    assert resp.headers["content-type"] == "application/pdf"

    # Clean up
    test_file.unlink(missing_ok=True)
