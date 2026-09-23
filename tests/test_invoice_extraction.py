from __future__ import annotations

from erp_agent.invoice_extraction import InvoiceExtractionClient, normalize_extraction


def test_normalize_extraction_maps_lines_and_discount_amount():
    normalized = normalize_extraction(
        {
            "invoiceNumber": "FAC-42",
            "invoiceDate": "2026-09-22",
            "customerName": "Acme",
            "devise": "TND",
            "timbre": 1.0,
            "lineModels": [
                {
                    "reference": "SCR-7",
                    "productname": "Screen",
                    "quantite": 7,
                    "prixunitaire": 714.29,
                    "tvaAmount": 950.01,
                    "remise": 250.0,
                    "unit": "piece",
                    "isService": False,
                }
            ],
        }
    )

    assert normalized["invoice_id"] == "FAC-42"
    assert normalized["client_name"] == "Acme"
    assert normalized["currency"] == "TND"
    assert normalized["items"][0]["product_id"] == "SCR-7"
    assert normalized["items"][0]["quantity"] == 7.0
    assert normalized["items"][0]["source_discount_amount"] == 250.0
    assert normalized["items"][0]["discount_pct"] == 5.0
    assert normalized["items"][0]["tva_amount"] == 950.01
    assert normalized["financials"]["total_brut_ht"] == 5000.03
    assert normalized["financials"]["total_remises"] == 250.0
    assert normalized["financials"]["total_net_ht"] == 4750.03
    assert normalized["financials"]["total_tva"] == 950.01
    assert normalized["financials"]["total_ttc"] == 5701.04


def test_extraction_client_posts_file_and_normalizes_response(tmp_path, monkeypatch):
    invoice_path = tmp_path / "invoice.png"
    invoice_path.write_bytes(b"fake-image")
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "request_id": "req-1",
                "document_id": "DOC-1",
                "model_used": "local-test",
                "result": {
                    "invoiceNumber": "FAC-1",
                    "customerName": "Test Client",
                    "devise": "TND",
                    "lineModels": [],
                },
            }

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr("erp_agent.invoice_extraction.httpx.post", fake_post)
    result = InvoiceExtractionClient(
        "http://127.0.0.1:8010",
        timeout=12,
        api_token="secret",
    ).extract(
        str(invoice_path),
        tenant_id="wind-erp",
        invoice_layout="auto",
        document_id="DOC-1",
        force_langue="fr",
        debug=True,
    )

    assert result["found"] is True
    assert result["source"] == "wind-invoice-extraction"
    assert result["invoice_data"]["invoice_id"] == "FAC-1"
    assert captured["url"] == "http://127.0.0.1:8010/v1/invoices/extract"
    assert captured["kwargs"]["data"] == {
        "tenant_id": "wind-erp",
        "invoice_layout": "auto",
        "debug": "true",
        "document_id": "DOC-1",
        "force_langue": "fr",
    }
    assert captured["kwargs"]["headers"] == {"Authorization": "Bearer secret"}
