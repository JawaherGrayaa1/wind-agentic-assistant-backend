from erp_agent.agent import AgentRuntime
from erp_agent.config import Settings
from erp_agent.planner import Plan

def make_runtime(tmp_path):
    return AgentRuntime(Settings(database_path=str(tmp_path / "test.db"), planner="rules"))

def test_inventory_lookup(tmp_path):
    runtime = make_runtime(tmp_path)
    result = runtime.run("s1", "u1", "How many P-100 are in stock?")
    assert result["status"] == "completed"
    assert "12" in result["reply"]
    assert any(item["type"] == "tool_result" for item in result["trace"])


def test_plural_product_request_stays_in_catalog_without_approval(tmp_path):
    runtime = make_runtime(tmp_path)
    result = runtime.run("s-products", "u1", "créer des produits random juste pour mock test")
    assert result["status"] == "completed"
    assert result["approval_token"] is None
    tool_calls = [item for item in result["trace"] if item.get("type") == "tool_call"]
    assert tool_calls[-1]["tool"] == "create_product"
    product_id = runtime.db.get_session_state("s-products")["active_product_id"]
    assert product_id and runtime.db.get_product(product_id)["name"] == "Random Product"


def test_document_creation_does_not_require_approval(tmp_path):
    runtime = make_runtime(tmp_path)
    result = runtime.run("s-doc-create", "u1", "create document title: Mock Note content: hello")
    assert result["status"] == "completed"
    assert result["approval_token"] is None

def test_create_order_runs_without_approval(tmp_path):
    runtime = make_runtime(tmp_path)
    result = runtime.run("s1", "u1", "Create a sales order")
    assert result["status"] == "completed"
    assert result["approval_token"] is None


def test_remote_no_tool_sentinel_does_not_crash(tmp_path):
    runtime = make_runtime(tmp_path)

    class SentinelPlanner:
        def plan(self, *args, **kwargs):
            return Plan(intent="greeting", reply="Hello!", tool_name="none", summary="Greeting")

    runtime.planner = SentinelPlanner()
    result = runtime.run("s-greeting", "u1", "hello")
    assert result["status"] == "completed"
    assert result["reply"] == "Hello!"
    assert not any(item.get("type") == "tool_call" for item in result["trace"])
