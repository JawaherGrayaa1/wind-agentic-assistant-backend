"""
tests/test_invoice_skill.py
Unit and integration tests for the Invoice (Facturation) Skill and Tools.
"""
from __future__ import annotations

import os
import tempfile
import pytest
from fastapi.testclient import TestClient

_tmp = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_PATH"] = _tmp
os.environ["AGENT_PLANNER"] = "rules"

from erp_agent.api import app, runtime  # noqa: E402
from erp_agent.skills.invoice.tools import (
    calculate_invoice_financials,
    format_invoice_markdown,
    parse_invoice_markdown,
)

client = TestClient(app)


def test_calculate_invoice_financials():
    items = [
        {"name": "Laptop Pro", "quantity": 2, "unit_price": 2500.0, "discount_pct": 0.0},
        {"name": "Wireless Mouse", "quantity": 5, "unit_price": 75.0, "discount_pct": 10.0},
    ]
    fin = calculate_invoice_financials(items, tax_rate=19.0, global_discount_pct=5.0, timbre_fiscal=1.0, currency="TND")

    assert fin["total_brut_ht"] == 5375.0
    # Line 2 discount: 5 * 75 * 0.10 = 37.50
    # Net after item remises: 5375 - 37.50 = 5337.50
    # Global discount: 5% of 5337.50 = 266.88
    # Total net HT: 5337.50 - 266.88 = 5070.62
    assert fin["total_net_ht"] == 5070.62
    # TVA: 19% of 5070.62 = 963.42
    assert fin["total_tva"] == 963.42
    # Timbre: 1.000 TND
    assert fin["timbre_fiscal"] == 1.0
    # Total TTC: 5070.62 + 963.42 + 1.0 = 6035.04
    assert fin["total_ttc"] == 6035.04


def test_markdown_format_and_parse_roundtrip():
    fin = calculate_invoice_financials([
        {"product_id": "P-100", "name": "Laptop Pro", "quantity": 2, "unit_price": 2499.0, "discount_pct": 0.0},
    ])
    md = format_invoice_markdown(
        invoice_id="INV-2026-001",
        client_name="Alpha Tech SARL",
        client_tax_id="1234567/A/P/000",
        financials=fin,
        status="draft",
    )

    assert "# FACTURE N° INV-2026-001" in md
    assert "Alpha Tech SARL" in md
    assert "Laptop Pro" in md

    parsed = parse_invoice_markdown(md)
    assert parsed["client_name"] == "Alpha Tech SARL"
    assert parsed["client_tax_id"] == "1234567/A/P/000"
    assert len(parsed["items"]) == 1
    assert parsed["items"][0]["name"] == "Laptop Pro"
    assert parsed["items"][0]["quantity"] == 2.0


def test_create_and_manage_invoice_tools():
    # 1. Create Invoice
    create_tool = runtime.tools.get("create_invoice")
    res = create_tool.handler(
        client_name="Beta Logistics",
        client_tax_id="9876543/B/M/000",
        invoice_id="INV-TEST-001",
        items=[
            {"name": "Transit International", "quantity": 1, "unit_price": 1500.0, "discount_pct": 0.0},
        ],
    )
    assert res["found"] is True
    assert res["invoice_id"] == "INV-TEST-001"
    assert res["financials"]["total_brut_ht"] == 1500.0

    # 2. Add Item
    add_tool = runtime.tools.get("add_invoice_item")
    add_res = add_tool.handler(
        invoice_id="INV-TEST-001",
        name="Assurance Fret",
        quantity=1,
        unit_price=200.0,
        discount_pct=0.0,
    )
    assert add_res["found"] is True
    assert add_res["financials"]["total_brut_ht"] == 1700.0

    # 3. Update Item
    update_tool = runtime.tools.get("update_invoice_item")
    up_res = update_tool.handler(
        invoice_id="INV-TEST-001",
        line_number=2,
        unit_price=250.0,
    )
    assert up_res["found"] is True
    assert up_res["financials"]["total_brut_ht"] == 1750.0

    # 4. Get Summary
    summary_tool = runtime.tools.get("get_invoice_summary")
    sum_res = summary_tool.handler(invoice_id="INV-TEST-001")
    assert sum_res["found"] is True
    assert sum_res["financials"]["total_brut_ht"] == 1750.0

    # 5. Validate Invoice
    val_tool = runtime.tools.get("validate_invoice")
    val_res = val_tool.handler(invoice_id="INV-TEST-001")
    assert val_res["valid"] is True
    assert val_res["status"] == "approved"

    # Verify document in DB is updated to approved
    db_doc = runtime.db.get_document("INV-TEST-001")
    assert db_doc["status"] == "approved"
    assert "Assurance Fret" in db_doc["content"]


def test_chat_invoice_lifecycle():
    # Chat creation
    resp = client.post("/v1/chat", json={
        "session_id": "s-inv-1",
        "user_id": "u1",
        "text": "créer une facture pour client Gamma Solutions avec 3x Wireless Mouse",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "approval_required"
    token = data["approval_token"]

    appr_resp = client.post(f"/v1/approvals/{token}")
    assert appr_resp.status_code == 200
    appr_data = appr_resp.json()
    assert appr_data["status"] == "completed"
    doc_id = appr_data["doc_id"]
    assert doc_id is not None

    # Summary lookup
    sum_resp = client.post("/v1/chat", json={
        "session_id": "s-inv-1",
        "user_id": "u1",
        "text": f"résumé de la facture {doc_id}",
    })
    assert sum_resp.status_code == 200
    sum_data = sum_resp.json()
    assert sum_data["status"] == "completed"
    assert "TOTAL TTC" in sum_data["reply"]
