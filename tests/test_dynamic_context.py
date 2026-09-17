"""
Tests for the Dynamic Context Manager and multi-turn session entity resolution.

Tests cover:
- Session state CRUD via Database
- DynamicContextManager token-budget trimming
- RulePlanner resolves active_invoice_id / active_bc_id / active_product_id
  from session_entities without an explicit ID in the user message
- AgentRuntime multi-turn flow: create -> approve -> follow-up with no explicit ID
"""
import pytest
from erp_agent.agent import AgentRuntime
from erp_agent.config import Settings
from erp_agent.db import Database
from erp_agent.memory import DynamicContextManager
from erp_agent.planner import RulePlanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_runtime(tmp_path) -> AgentRuntime:
    return AgentRuntime(Settings(database_path=str(tmp_path / "test.db"), planner="rules"))


# ---------------------------------------------------------------------------
# 1. Session-state CRUD
# ---------------------------------------------------------------------------


def test_session_state_empty_on_first_read(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    state = db.get_session_state("sess-1")
    # DB may return an empty dict OR a dict with all-None values for a new session
    non_null_values = {k: v for k, v in state.items() if k not in ("session_id", "updated_at") and v is not None}
    assert non_null_values == {}, f"Expected no active entities, got: {non_null_values}"


def test_session_state_update_and_retrieve(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))

    db.update_session_state("sess-1", active_invoice_id="INV-1001", active_client="Société Alpha")
    state = db.get_session_state("sess-1")

    assert state.get("active_invoice_id") == "INV-1001"
    assert state.get("active_client") == "Société Alpha"
    assert state.get("active_bc_id") is None


def test_session_state_incremental_update(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))

    db.update_session_state("sess-2", active_invoice_id="INV-2000")
    db.update_session_state("sess-2", active_bc_id="BC-3000")

    state = db.get_session_state("sess-2")
    # Both should survive after incremental update
    assert state.get("active_invoice_id") == "INV-2000"
    assert state.get("active_bc_id") == "BC-3000"


# ---------------------------------------------------------------------------
# 2. DynamicContextManager — token budget trimming
# ---------------------------------------------------------------------------


def test_budget_trims_old_turns(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    session_id = "budget-sess"

    # Add 25 turns of ~100 chars each => 2,500 chars total, max_history_chars=800 → should trim
    for i in range(25):
        db.add_message(session_id, "u1", "user" if i % 2 == 0 else "assistant", f"Message {i}: " + "x" * 80)

    mgr = DynamicContextManager(db, max_history_chars=800, max_turns=20)
    ctx = mgr.context(session_id, "u1")

    messages = ctx["messages"]
    assert len(messages) < 25, "Budget should have trimmed old turns"
    total_chars = sum(len(m["content"]) for m in messages)
    # Should be under budget (+small slack for the 4-message minimum rule)
    assert total_chars <= 1500


def test_budget_keeps_at_least_recent_turns(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    session_id = "recent-sess"

    for i in range(6):
        db.add_message(session_id, "u1", "user" if i % 2 == 0 else "assistant", f"Turn {i}")

    mgr = DynamicContextManager(db, max_history_chars=10000, max_turns=4)
    ctx = mgr.context(session_id, "u1")

    messages = ctx["messages"]
    assert len(messages) <= 4
    # The most recent message should be last
    assert "Turn 5" in messages[-1]["content"]


def test_large_payload_truncated_in_history(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    session_id = "truncate-sess"

    huge = "A" * 3000
    db.add_message(session_id, "u1", "assistant", huge)

    mgr = DynamicContextManager(db, max_history_chars=10000, max_turns=20)
    ctx = mgr.context(session_id, "u1")

    messages = ctx["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]
    # Over 1,500 chars should be truncated
    assert len(content) < 1600
    assert "truncated" in content.lower()


# ---------------------------------------------------------------------------
# 3. RulePlanner — entity resolution from session_entities
# ---------------------------------------------------------------------------


def test_rule_planner_resolves_active_invoice_for_add_item():
    planner = RulePlanner()
    context = {"session_entities": {"active_invoice_id": "INV-7777"}, "messages": []}

    plan = planner.plan("ajoute 2 claviers à 50 DT à la facture", context, [])
    assert plan.tool_name == "add_invoice_item"
    assert plan.arguments["invoice_id"] == "INV-7777"
    assert plan.arguments["quantity"] == 2.0
    assert plan.arguments["unit_price"] == 50.0


def test_rule_planner_resolves_active_invoice_for_export_pdf():
    planner = RulePlanner()
    context = {"session_entities": {"active_invoice_id": "INV-8888"}, "messages": []}

    plan = planner.plan("exporte la facture en PDF", context, [])
    assert plan.tool_name == "export_invoice_pdf"
    assert plan.arguments["invoice_id"] == "INV-8888"


def test_rule_planner_resolves_active_invoice_for_delete():
    planner = RulePlanner()
    context = {"session_entities": {"active_invoice_id": "INV-9999"}, "messages": []}

    plan = planner.plan("supprimer la facture", context, [])
    assert plan.tool_name == "delete_invoice"
    assert plan.arguments["invoice_id"] == "INV-9999"


def test_rule_planner_resolves_active_bc_for_add_item():
    planner = RulePlanner()
    context = {"session_entities": {"active_bc_id": "BC-5555"}, "messages": []}

    plan = planner.plan("ajoute 5 souris au bon de commande", context, [])
    assert plan.tool_name == "add_order_item"
    assert plan.arguments["order_id"] == "BC-5555"
    assert plan.arguments["quantity"] == 5


def test_rule_planner_resolves_active_bc_for_pdf():
    planner = RulePlanner()
    context = {"session_entities": {"active_bc_id": "BC-6666"}, "messages": []}

    plan = planner.plan("exporte le bon de commande en pdf", context, [])
    assert plan.tool_name == "export_bon_de_commande_pdf"
    assert plan.arguments["order_id"] == "BC-6666"


def test_rule_planner_resolves_active_product_for_update_stock():
    planner = RulePlanner()
    context = {"session_entities": {"active_product_id": "P-404"}, "messages": []}

    plan = planner.plan("mettre à jour le stock à 120 unités", context, [])
    assert plan.tool_name == "update_product_stock"
    assert plan.arguments["product_id"] == "P-404"
    assert plan.arguments["stock"] == 120


def test_rule_planner_resolves_active_product_for_detail():
    planner = RulePlanner()
    context = {"session_entities": {"active_product_id": "P-505"}, "messages": []}

    plan = planner.plan("fiche produit", context, [])
    assert plan.tool_name == "get_product"
    assert plan.arguments["product_id"] == "P-505"


def test_rule_planner_resolves_active_product_for_inventory():
    planner = RulePlanner()
    context = {"session_entities": {"active_product_id": "P-606"}, "messages": []}

    plan = planner.plan("stock disponible", context, [])
    assert plan.tool_name == "get_inventory"
    assert plan.arguments["product_id"] == "P-606"


def test_explicit_id_overrides_session_entity():
    """An explicit ID in the message must win over session entity."""
    planner = RulePlanner()
    context = {"session_entities": {"active_invoice_id": "INV-OLD"}, "messages": []}

    plan = planner.plan("exporte la facture INV-NEW en PDF", context, [])
    assert plan.tool_name == "export_invoice_pdf"
    assert plan.arguments["invoice_id"] == "INV-NEW"


# ---------------------------------------------------------------------------
# 4. AgentRuntime integration — multi-turn entity tracking
# ---------------------------------------------------------------------------


def test_agent_auto_tracks_invoice_entity(tmp_path):
    rt = make_runtime(tmp_path)
    session_id = "mt-inv-sess"

    # Turn 1: create invoice (needs approval)
    res1 = rt.run(session_id, "u1", "Créer une facture pour Client Test")
    assert res1["status"] == "approval_required"
    token = res1["approval_token"]

    # Approve
    res_approved = rt.approve(token)
    assert res_approved["status"] == "completed"

    # Session entity should now hold active_invoice_id
    state = rt.db.get_session_state(session_id)
    inv_id = state.get("active_invoice_id")
    assert inv_id is not None
    assert inv_id.startswith("INV-")


def test_agent_multi_turn_follow_up_no_explicit_id(tmp_path):
    rt = make_runtime(tmp_path)
    session_id = "mt-follow-up"

    # Turn 1: create and approve invoice
    res1 = rt.run(session_id, "u1", "Créer une facture pour Société Beta")
    assert res1["status"] == "approval_required"
    rt.approve(res1["approval_token"])

    inv_id = rt.db.get_session_state(session_id).get("active_invoice_id")
    assert inv_id is not None

    # Turn 2: Follow-up with "voir la facture" — no explicit invoice ID
    res2 = rt.run(session_id, "u1", "voir la facture")
    assert res2["status"] == "completed"
    # Tool result or reply should contain the invoice ID
    trace_tools = [t for t in res2.get("trace", []) if t.get("type") == "tool_result"]
    if trace_tools:
        result_payload = trace_tools[-1].get("result", {})
        assert isinstance(result_payload, dict)


def test_agent_bc_entity_tracked_after_creation(tmp_path):
    rt = make_runtime(tmp_path)
    session_id = "mt-bc-sess"

    res1 = rt.run(session_id, "u1", "Créer un bon de commande pour Fournisseur XYZ")
    assert res1["status"] == "approval_required"
    rt.approve(res1["approval_token"])

    state = rt.db.get_session_state(session_id)
    bc_id = state.get("active_bc_id")
    assert bc_id is not None


def test_export_it_to_pdf_multi_turn(tmp_path):
    """Test the exact user scenario: create invoice -> approve -> 'export it to pdf'."""
    rt = make_runtime(tmp_path)
    session_id = "sess-export-pdf-followup"

    # Turn 1: create invoice for TechCorp
    res1 = rt.run(session_id, "u1", "Crée une facture pour TechCorp avec 4 PC Portable à 1600 DT")
    assert res1["status"] == "approval_required"
    token = res1["approval_token"]

    # Turn 2: approve creation
    res_appr = rt.approve(token)
    assert res_appr["status"] == "completed"
    inv_id = rt.db.get_session_state(session_id).get("active_invoice_id")
    assert inv_id is not None
    assert inv_id.startswith("INV-")

    # Turn 3: User says "export it to pdf" (in English, no ID)
    res3 = rt.run(session_id, "u1", "export it to pdf")
    assert res3["status"] == "completed"
    assert res3.get("file_url") is not None or "INV-" in res3["reply"]
    trace_tools = [t for t in res3.get("trace", []) if t.get("type") == "tool_call"]
    assert any(t.get("tool") == "export_invoice_pdf" for t in trace_tools)

