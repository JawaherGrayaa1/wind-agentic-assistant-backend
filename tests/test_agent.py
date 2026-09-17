from erp_agent.agent import AgentRuntime
from erp_agent.config import Settings

def make_runtime(tmp_path):
    return AgentRuntime(Settings(database_path=str(tmp_path / "test.db"), planner="rules"))

def test_inventory_lookup(tmp_path):
    runtime = make_runtime(tmp_path)
    result = runtime.run("s1", "u1", "How many P-100 are in stock?")
    assert result["status"] == "completed"
    assert "12" in result["reply"]
    assert any(item["type"] == "tool_result" for item in result["trace"])

def test_write_action_requires_approval(tmp_path):
    runtime = make_runtime(tmp_path)
    result = runtime.run("s1", "u1", "Create a sales order")
    assert result["status"] == "approval_required"
    approved = runtime.approve(result["approval_token"])
    assert approved["status"] == "completed"
