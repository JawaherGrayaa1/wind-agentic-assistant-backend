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


def test_planner_reply_is_returned_after_tool_execution(tmp_path):
    runtime = make_runtime(tmp_path)

    class PlannerWithReply:
        def plan(self, *args, **kwargs):
            return Plan(
                intent="create_product",
                reply="MODEL_REPLY",
                tool_name="create_product",
                arguments={"product_id": "P-REPLY", "name": "Reply Product", "stock": 1, "price": 10},
                summary="Create a product.",
            )

    runtime.planner = PlannerWithReply()
    result = runtime.run("s-reply", "u1", "create product")
    assert "MODEL_REPLY" in result["reply"]
    assert "Reply Product" in result["reply"]


def test_planner_heading_includes_list_tool_result(tmp_path):
    runtime = make_runtime(tmp_path)
    created = runtime.run("s-list-reply", "u1", "create invoice for Acme")

    class ListPlanner:
        def plan(self, *args, **kwargs):
            return Plan(
                intent="list_invoices",
                reply="Here is the most recently created invoice.",
                tool_name="list_invoices",
                arguments={"limit": 1},
                summary="List the most recent invoice.",
            )

    runtime.planner = ListPlanner()
    result = runtime.run("s-list-reply", "u1", "what is the last created invoice")
    assert "Here is the most recently created invoice." in result["reply"]
    assert created["doc_id"] in result["reply"]
    assert result["open_editor_doc_id"] == created["doc_id"]
    assert runtime.db.get_session_state("s-list-reply").get("active_invoice_id") == created["doc_id"]


def test_latest_purchase_order_export_chains_list_and_export(tmp_path):
    runtime = make_runtime(tmp_path)
    created = runtime.tools.get("create_bon_de_commande").handler(
        client_name="jawaher",
        order_id="BC-CHAIN-01",
        items=[{"name": "Desk", "quantity": 1, "unit_price": 200}],
        tax_rate=0,
    )

    class ChainedExportPlanner:
        def plan(self, *args, **kwargs):
            return Plan(
                intent="export_purchase_order_pdf",
                reply="I will first list the most recent purchase order to get its ID, and then export it to PDF.",
                tool_name="list_bon_de_commandes",
                arguments={"limit": 1},
                summary="List the most recent purchase order before exporting it.",
            )

    runtime.planner = ChainedExportPlanner()
    result = runtime.run("s-chain-export", "u1", "export the last created purchase order to pdf")
    tool_calls = [item for item in result["trace"] if item.get("type") == "tool_call"]
    assert [item["tool"] for item in tool_calls] == [
        "list_bon_de_commandes",
        "export_bon_de_commande_pdf",
    ]
    assert created["order_id"] in result["reply"]
    assert result["file_url"]


def test_create_invoice_and_export_chains_both_tools(tmp_path):
    runtime = make_runtime(tmp_path)

    class ChainedInvoicePlanner:
        def plan(self, *args, **kwargs):
            return Plan(
                intent="create_invoice_and_export_pdf",
                reply="Invoice created successfully. Now exporting to PDF...",
                tool_name="create_invoice",
                arguments={
                    "client_name": "jawaher",
                    "items": [{"name": "Laptop", "quantity": 4, "unit_price": 12000}],
                    "currency": "TND",
                },
                summary="Create an invoice and export it to PDF.",
            )

    runtime.planner = ChainedInvoicePlanner()
    result = runtime.run(
        "s-chain-invoice-export",
        "u1",
        "create a new invoice for jawaher with 4 laptops for 12000 TND then export it to pdf",
    )
    tool_calls = [item for item in result["trace"] if item.get("type") == "tool_call"]
    assert [item["tool"] for item in tool_calls] == [
        "create_invoice",
        "export_invoice_pdf",
    ]
    assert "INV-" in result["reply"]
    assert result["file_url"]


def test_create_purchase_order_from_latest_invoice_chains_all_steps(tmp_path):
    runtime = make_runtime(tmp_path)
    source = runtime.tools.get("create_invoice").handler(
        client_name="jawaher",
        invoice_id="INV-CHAIN-PO",
        items=[{"name": "Laptop", "quantity": 2, "unit_price": 1200}],
        currency="TND",
    )
    assert source["found"] is True

    class ChainedPurchaseOrderPlanner:
        def plan(self, *args, **kwargs):
            return Plan(
                intent="create_purchase_order_from_invoice",
                reply="I will create the purchase order from the latest invoice.",
                tool_name="list_invoices",
                arguments={"limit": 1},
                summary="Find the latest invoice, read its items, then create a purchase order.",
            )

    runtime.planner = ChainedPurchaseOrderPlanner()
    result = runtime.run(
        "s-chain-invoice-po",
        "u1",
        "now create a purchase order with the same products as this invoice",
    )
    tool_calls = [item for item in result["trace"] if item.get("type") == "tool_call"]
    assert [item["tool"] for item in tool_calls] == [
        "list_invoices",
        "get_invoice_summary",
        "create_bon_de_commande",
    ]
    assert result["status"] == "completed"
    assert result["doc_id"].startswith("BC-")
    assert result["document"]["document_type"] == "bon_de_commande"
    assert result["document"]["items"][0]["name"] == "Laptop"


def test_remote_planner_can_choose_a_follow_up_tool(tmp_path):
    runtime = AgentRuntime(Settings(
        database_path=str(tmp_path / "test.db"),
        planner="ollama",
    ))

    class FlexiblePlanner:
        def plan(self, text, context, tools, skills=None):
            history = context.get("tool_history") or []
            if not history:
                return Plan(
                    intent="catalog_lookup_workflow",
                    tool_name="create_product",
                    arguments={
                        "product_id": "P-FLEX-01",
                        "name": "Flexible Product",
                        "stock": 3,
                        "price": 25,
                    },
                    summary="Create the product before looking it up.",
                )
            if len(history) == 1:
                return Plan(
                    intent="catalog_lookup_workflow",
                    tool_name="get_product",
                    arguments={"product_id": "P-FLEX-01"},
                    summary="Now retrieve the created product.",
                )
            return Plan(
                intent="catalog_lookup_workflow",
                tool_name=None,
                summary="Workflow complete.",
            )

    runtime.planner = FlexiblePlanner()
    runtime._synthesize_reply = lambda **kwargs: ("Workflow complete.", False)
    result = runtime.run("s-flexible-chain", "u1", "create and then inspect a product")
    tool_calls = [item for item in result["trace"] if item.get("type") == "tool_call"]
    assert [item["tool"] for item in tool_calls] == ["create_product", "get_product"]
    assert result["status"] == "completed"


def test_invoice_follow_up_uses_active_invoice_not_purchase_order(tmp_path):
    runtime = make_runtime(tmp_path)
    created = runtime.run("s-invoice-follow-up", "u1", "create invoice for Acme")
    assert created["status"] == "completed"

    result = runtime.run(
        "s-invoice-follow-up",
        "u1",
        "yes the client name is wind-consulting and change device to tnd instead",
    )
    assert result["status"] == "approval_required"
    assert result["approval_token"]
    approval = runtime.db.get_approval(result["approval_token"])
    assert approval["tool_name"] == "update_invoice"
    assert approval["arguments"]["invoice_id"] == created["doc_id"]
    assert approval["arguments"]["client_name"] == "wind-consulting"
    assert approval["arguments"]["new_item_name"] == "tnd"


def test_invoice_listing_preserves_saved_discount_and_total(tmp_path):
    runtime = make_runtime(tmp_path)
    create_invoice = runtime.tools.get("create_invoice").handler
    create_invoice(
        client_name="Zind Consulting",
        invoice_id="INV-DISCOUNT",
        discount_pct=20,
        items=[
            {"name": "Keyboards", "quantity": 4, "unit_price": 1000},
            {"name": "Computer Screens", "quantity": 8, "unit_price": 500},
        ],
    )

    listed = runtime.tools.get("list_invoices").handler(limit=1)
    assert listed["invoices"][0]["invoice_id"] == "INV-DISCOUNT"
    assert listed["invoices"][0]["total_ttc"] == 7617.0


def test_file_modality_routes_images_to_ocr_and_pdfs_to_pdf_reader():
    image_plan = Plan(
        intent="read_document",
        tool_name="read_pdf",
        arguments={"file_path": r"C:	empscanned-invoice.webp"},
    )
    image_routed = AgentRuntime._enforce_file_modality(image_plan)
    assert image_routed.tool_name == "extract_invoice_from_file"
    assert image_routed.arguments["file_path"].endswith("scanned-invoice.webp")

    pdf_plan = Plan(
        intent="read_document",
        tool_name="extract_invoice_from_file",
        arguments={"file_path": r"C:	empinvoice.pdf"},
    )
    pdf_routed = AgentRuntime._enforce_file_modality(pdf_plan)
    assert pdf_routed.tool_name == "read_pdf"
    assert pdf_routed.arguments["file_path"].endswith("invoice.pdf")
