"""
tests/test_bon_de_commande_skill.py
Tests for the Bon de Commande financial skill, calculation engine, and agent integration.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from erp_agent.config import Settings
from erp_agent.db import Database
from erp_agent.skills.loader import SkillLoader
from erp_agent.skills.bon_de_commande.tools import calculate_financials
from erp_agent.tools import ToolRegistry
from erp_agent.agent import AgentRuntime


@pytest.fixture
def temp_db():
    tmp = tempfile.mktemp(suffix=".db")
    return Database(tmp)


@pytest.fixture
def registry(temp_db):
    return ToolRegistry(temp_db)


class TestFinancialCalculations:
    def test_calculate_financials_basic(self):
        items = [
            {"name": "Laptop Pro", "quantity": 2, "unit_price": 2000.0, "discount_pct": 10.0},
            {"name": "Mouse", "quantity": 5, "unit_price": 50.0, "discount_pct": 0.0},
        ]
        res = calculate_financials(items, tax_rate=19.0)
        assert res["item_count"] == 2
        # Laptop: 2 * 2000 = 4000 brut, 400 remise, 3600 net
        # Mouse: 5 * 50 = 250 brut, 0 remise, 250 net
        assert res["total_brut_ht"] == 4250.0
        assert res["total_remises"] == 400.0
        assert res["total_net_ht"] == 3850.0
        # TVA 19% on 3850 = 731.5
        assert res["total_tva"] == 731.5
        # TTC = 3850 + 731.5 = 4581.5
        assert res["total_ttc"] == 4581.5


class TestBonDeCommandeTools:
    def test_discovery_and_registration(self, registry):
        tools = [
            "create_bon_de_commande",
            "add_order_item",
            "update_order_item",
            "remove_order_item",
            "get_order_summary",
            "export_bon_de_commande_pdf",
        ]
        for t in tools:
            assert t in registry.tools

        skill_instructions = registry.skill_instructions()
        assert "bon_de_commande" in skill_instructions
        assert "Bon de Commande & Financial Management Guide" in skill_instructions["bon_de_commande"]

    def test_create_bon_de_commande_with_catalog_lookup(self, registry):
        handler = registry.get("create_bon_de_commande").handler
        res = handler(
            client_name="Alpha Corp",
            order_id="BC-TEST-01",
            items=[{"product_id": "P-100", "quantity": 2}],  # P-100 is Laptop Pro @ 2499.0 in seed
        )
        assert res["found"] is True
        assert res["order_id"] == "BC-TEST-01"
        assert res["financials"]["item_count"] == 1
        item = res["financials"]["items"][0]
        assert item["name"] == "Laptop Pro"
        assert item["unit_price"] == 2499.0
        assert res["financials"]["total_brut_ht"] == 4998.0

    def test_add_item_to_order(self, registry):
        create_handler = registry.get("create_bon_de_commande").handler
        create_handler(client_name="Beta SARL", order_id="BC-TEST-02", items=[])

        add_handler = registry.get("add_order_item").handler
        # Add P-200 (Wireless Mouse @ 75.0)
        res = add_handler(order_id="BC-TEST-02", product_id="P-200", quantity=4, discount_pct=10.0)
        assert res["found"] is True
        assert res["financials"]["item_count"] == 1
        # 4 * 75 = 300 brut, 30 remise, 270 net HT
        assert res["financials"]["total_brut_ht"] == 300.0
        assert res["financials"]["total_remises"] == 30.0
        assert res["financials"]["total_net_ht"] == 270.0

    def test_update_order_item(self, registry):
        create_handler = registry.get("create_bon_de_commande").handler
        create_handler(
            client_name="Gamma Industries",
            order_id="BC-TEST-03",
            items=[{"name": "Server Maintenance", "quantity": 1, "unit_price": 1000.0}],
        )

        update_handler = registry.get("update_order_item").handler
        res = update_handler(order_id="BC-TEST-03", item_index=1, quantity=3, unit_price=900.0, discount_pct=5.0)
        assert res["found"] is True
        # 3 * 900 = 2700 brut, 135 remise, 2565 net HT
        assert res["financials"]["total_brut_ht"] == 2700.0
        assert res["financials"]["total_remises"] == 135.0
        assert res["financials"]["total_net_ht"] == 2565.0

    def test_remove_order_item(self, registry):
        create_handler = registry.get("create_bon_de_commande").handler
        create_handler(
            client_name="Delta Tech",
            order_id="BC-TEST-04",
            items=[
                {"name": "Item A", "quantity": 1, "unit_price": 100.0},
                {"name": "Item B", "quantity": 1, "unit_price": 200.0},
            ],
        )

        remove_handler = registry.get("remove_order_item").handler
        res = remove_handler(order_id="BC-TEST-04", item_index=1)
        assert res["found"] is True
        assert res["remaining_count"] == 1
        assert res["financials"]["total_brut_ht"] == 200.0

    def test_get_order_summary(self, registry):
        create_handler = registry.get("create_bon_de_commande").handler
        create_handler(
            client_name="Epsilon Corp",
            order_id="BC-TEST-05",
            items=[{"name": "Consulting", "quantity": 10, "unit_price": 150.0}],
        )

        summary_handler = registry.get("get_order_summary").handler
        res = summary_handler(order_id="BC-TEST-05")
        assert res["found"] is True
        assert res["client_name"] == "Epsilon Corp"
        assert res["financials"]["total_net_ht"] == 1500.0
        assert res["financials"]["total_tva"] == 285.0
        assert res["financials"]["total_ttc"] == 1785.0

    def test_export_bon_de_commande_pdf(self, registry, tmp_path):
        create_handler = registry.get("create_bon_de_commande").handler
        create_handler(
            client_name="Omega Logistics",
            order_id="BC-TEST-06",
            items=[
                {"name": "Forklift Service", "quantity": 2, "unit_price": 1200.0},
                {"name": "Safety Inspection", "quantity": 1, "unit_price": 400.0, "discount_pct": 10.0},
            ],
        )

        pdf_out = str(tmp_path / "BC_TEST_06.pdf")
        export_handler = registry.get("export_bon_de_commande_pdf").handler
        res = export_handler(order_id="BC-TEST-06", output_path=pdf_out)
        assert res["found"] is True
        assert Path(pdf_out).exists()
        assert Path(pdf_out).stat().st_size > 0


class TestAgentBonDeCommandeInteraction:
    def test_agent_create_and_summary_flow(self, tmp_path):
        db_path = tempfile.mktemp(suffix=".db")
        settings = Settings(database_path=db_path, planner="rules")
        agent = AgentRuntime(settings)

        # 1. Ask agent to create Bon de Commande (no confirmation needed)
        create_res = agent.run("s-bc", "user-1", "Crée un bon de commande BC-900 pour Société Tunisienne")
        assert create_res["status"] == "completed"
        assert create_res["approval_token"] is None
        assert "BC-900" in create_res["reply"]

        # 3. Ask agent to calculate/show summary
        sum_res = agent.run("s-bc", "user-1", "Affiche le total du bon de commande BC-900")
        assert sum_res["status"] == "completed"
        assert "TOTAL TTC" in sum_res["reply"]
        assert "BC-900" in sum_res["reply"]
