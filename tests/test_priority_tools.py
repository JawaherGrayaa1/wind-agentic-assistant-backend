"""
tests/test_priority_tools.py
Comprehensive unit tests for the 8 new priority tools and the LLM response synthesis workflow.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
import pytest

from erp_agent.config import Settings
from erp_agent.db import Database
from erp_agent.tools import ToolRegistry
from erp_agent.agent import AgentRuntime


@pytest.fixture
def temp_db():
    tmp = tempfile.mktemp(suffix=".db")
    return Database(tmp)


@pytest.fixture
def registry(temp_db):
    return ToolRegistry(temp_db)


class TestProductPriorityTools:
    def test_create_and_get_product(self, registry, temp_db):
        create_tool = registry.get("create_product")
        res = create_tool.handler(
            product_id="P-NEW-01",
            name="Ergonomic Keyboard",
            stock=25,
            price=149.99,
        )
        assert res["created"] is True
        assert res["product"]["product_id"] == "P-NEW-01"
        assert res["product"]["price"] == 149.99

        get_tool = registry.get("get_product")
        get_res = get_tool.handler(product_id="P-NEW-01")
        assert get_res["found"] is True
        assert get_res["product"]["name"] == "Ergonomic Keyboard"
        assert get_res["product"]["stock"] == 25

    def test_update_product_stock(self, registry, temp_db):
        update_tool = registry.get("update_product_stock")
        res = update_tool.handler(product_id="P-100", stock=50, price=2200.0)
        assert res["updated"] is True
        assert res["product"]["stock"] == 50
        assert res["product"]["price"] == 2200.0

        get_res = registry.get("get_product").handler(product_id="P-100")
        assert get_res["product"]["stock"] == 50


class TestInvoicePriorityTools:
    def test_list_invoices(self, registry, temp_db):
        list_tool = registry.get("list_invoices")
        res = list_tool.handler()
        assert res["found"] is True
        assert res["count"] >= 1
        assert any(inv["invoice_id"] == "DOC-101" for inv in res["invoices"])

    def test_duplicate_invoice(self, registry, temp_db):
        # Create an invoice first
        create_tool = registry.get("create_invoice")
        create_res = create_tool.handler(
            client_name="Client Beta",
            invoice_id="INV-SRC-01",
            items=[{"name": "Consulting", "quantity": 10, "unit_price": 100.0}],
        )
        assert create_res["found"] is True

        dup_tool = registry.get("duplicate_invoice")
        dup_res = dup_tool.handler(
            invoice_id="INV-SRC-01",
            new_client_name="Client Gamma",
            new_invoice_id="INV-DUP-01",
        )
        assert dup_res["found"] is True
        assert dup_res["invoice_id"] == "INV-DUP-01"
        assert dup_res["client_name"] == "Client Gamma"
        assert dup_res["financials"]["total_brut_ht"] == 1000.0

    def test_delete_invoice(self, registry, temp_db):
        create_tool = registry.get("create_invoice")
        create_tool.handler(client_name="Client To Delete", invoice_id="INV-DEL-01")

        del_tool = registry.get("delete_invoice")
        del_res = del_tool.handler(invoice_id="INV-DEL-01")
        assert del_res["deleted"] is True

        # Check it is gone
        get_res = registry.get("get_invoice_summary").handler(invoice_id="INV-DEL-01")
        assert get_res["found"] is False

    def test_export_invoice_pdf(self, registry, temp_db, tmp_path):
        create_tool = registry.get("create_invoice")
        create_tool.handler(
            client_name="Client PDF Test",
            invoice_id="INV-PDF-01",
            items=[{"name": "Server Setup", "quantity": 1, "unit_price": 850.0}],
        )

        out_pdf = str(tmp_path / "invoice_test.pdf")
        export_tool = registry.get("export_invoice_pdf")
        res = export_tool.handler(invoice_id="INV-PDF-01", output_path=out_pdf)
        assert res["found"] is True
        assert Path(out_pdf).exists()
        assert Path(out_pdf).stat().st_size > 0


class TestBonDeCommandePriorityTools:
    def test_list_bon_de_commandes(self, registry, temp_db):
        create_tool = registry.get("create_bon_de_commande")
        create_tool.handler(
            client_name="Enterprise Corp",
            order_id="BC-LIST-01",
            items=[{"name": "Monitors", "quantity": 5, "unit_price": 400.0}],
        )

        list_tool = registry.get("list_bon_de_commandes")
        res = list_tool.handler(client_name="Enterprise")
        assert res["found"] is True
        assert res["count"] >= 1
        assert any(o["order_id"] == "BC-LIST-01" for o in res["orders"])


class TestAgentSynthesisWorkflow:
    def test_agent_synthesize_fallback(self):
        db_path = tempfile.mktemp(suffix=".db")
        settings = Settings(database_path=db_path, planner="rules")
        agent = AgentRuntime(settings)

        res = agent.run("s1", "u1", "lister factures")
        assert res["status"] == "completed"
        assert "Factures enregistrées" in res["reply"] or "DOC-101" in res["reply"]

    def test_agent_product_detail_flow(self):
        db_path = tempfile.mktemp(suffix=".db")
        settings = Settings(database_path=db_path, planner="rules")
        agent = AgentRuntime(settings)

        res = agent.run("s2", "u1", "detail produit P-100")
        assert res["status"] == "completed"
        assert "Laptop Pro" in res["reply"] or "P-100" in res["reply"]
