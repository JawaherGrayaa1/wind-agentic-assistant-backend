from erp_agent.config import settings
from erp_agent.agent import AgentRuntime

runtime = AgentRuntime(settings)  # AGENT_PLANNER defaults to "rules", no LLM needed

print("=" * 60)
print("PHASE 1 - ERP + Document pipeline smoke tests")
print("=" * 60)

# 1. Inventory read
r1 = runtime.run("s1", "u1", "what's the stock on P-100?")
print(f"\n[1] Inventory: {r1['reply']} | status={r1['status']}")
assert r1["status"] == "completed", f"Expected completed, got {r1['status']}"

# 2. Product search
r2 = runtime.run("s1", "u1", "search for mouse")
print(f"\n[2] Search: {r2['reply']}")

# 3. Write path - should pause for approval
r3 = runtime.run("s1", "u1", "create order for customer C1")
print(f"\n[3] Order approval: status={r3['status']} | token={r3['approval_token']}")
assert r3["status"] == "approval_required"

# 4. Approve order
r4 = runtime.approve(r3["approval_token"])
print(f"\n[4] Order executed: status={r4['status']} | result={r4['result']}")
assert r4["status"] == "completed"

# 5. Memory
r5 = runtime.run("s1", "u1", "remember that I prefer email notifications")
print(f"\n[5] Memory: {r5}")

# 6. Document lookup
r6 = runtime.run("s1", "u1", "show me document DOC-101")
print(f"\n[6] Document read: {r6['reply'][:120]}... | status={r6['status']}")
assert r6["status"] == "completed", f"Expected completed, got {r6['status']}"

# 7. Document search
r7 = runtime.run("s1", "u1", "search documents for purchase")
print(f"\n[7] Document search: {r7['reply']} | status={r7['status']}")
assert r7["status"] == "completed"

# 8. Document edit - approval required
r8 = runtime.run("s1", "u1", "edit document DOC-102 with content: Updated purchase order for Q4 2026.")
print(f"\n[8] Doc edit approval: status={r8['status']} | token={r8['approval_token']}")
assert r8["status"] == "approval_required"

# 9. Approve the edit
r9 = runtime.approve(r8["approval_token"])
print(f"\n[9] Doc edit executed: status={r9['status']} | updated={r9['result'].get('updated')}")
assert r9["status"] == "completed"
assert r9["result"]["updated"] is True

print("\n" + "=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
