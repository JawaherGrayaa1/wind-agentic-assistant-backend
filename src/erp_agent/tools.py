from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from .db import Database, utc_now
from .document_editor import SmartDocumentEditor
from .skills import SkillLoader


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict[str, Any]]
    requires_confirmation: bool = False


class ToolRegistry:
    def __init__(self, db: Database, editor: SmartDocumentEditor | None = None):
        self.db = db
        self.editor = editor or SmartDocumentEditor()
        self.skill_loader = SkillLoader()
        self.tools = {
            "get_inventory": Tool(
                "get_inventory", "Look up current stock for a product.",
                {"product_id": "string"}, self.get_inventory
            ),
            "get_product": Tool(
                "get_product", "Get full product details (name, stock, unit price) from the catalog.",
                {"product_id": "string"}, self.get_product
            ),
            "search_products": Tool(
                "search_products", "Search products by name or ID.",
                {"query": "string"}, self.search_products
            ),
            "create_product": Tool(
                "create_product", "Create a new product in the catalog with initial stock and price. Requires user approval.",
                {"product_id": "string", "name": "string", "stock": "integer", "price": "number"},
                self.create_product, True
            ),
            "update_product_stock": Tool(
                "update_product_stock", "Update stock level, price, or name for an existing product. Requires user approval.",
                {"product_id": "string", "stock": "integer", "price": "number", "name": "string"},
                self.update_product_stock, True
            ),
            "create_sales_order": Tool(
                "create_sales_order", "Create a sales order for a customer.",
                {"customer_id": "string", "items": "array"},
                self.create_sales_order, True
            ),
            "remember_fact": Tool(
                "remember_fact", "Save an explicit user preference or fact.",
                {"key": "string", "value": "string"}, self.remember_fact, True
            ),
            "get_document": Tool(
                "get_document", "Retrieve a document by its ID (e.g. DOC-101).",
                {"doc_id": "string"}, self.get_document
            ),
            "search_documents": Tool(
                "search_documents", "Search documents by title, type, or content.",
                {"query": "string"}, self.search_documents
            ),
            "create_document": Tool(
                "create_document",
                "Create a new document (invoice, quote, contract, purchase order, etc). Requires user approval.",
                {"title": "string", "doc_type": "string", "content": "string", "doc_id": "string"},
                self.create_document, True
            ),
            "edit_document": Tool(
                "edit_document",
                "Directly update an existing document's content. Requires user approval.",
                {"doc_id": "string", "content": "string"},
                self.edit_document, True
            ),
            "smart_edit_document": Tool(
                "smart_edit_document",
                "Intelligently edit a document using AI understanding based on natural language instructions. Requires user approval.",
                {"doc_id": "string", "instruction": "string"},
                self.smart_edit_document, True
            ),
            "delete_document": Tool(
                "delete_document",
                "Delete a document by its ID. Requires user approval.",
                {"doc_id": "string"},
                self.delete_document, True
            ),
        }

        # Dynamically discover and register tools from skills (e.g. PDF skill)
        self.skills = self.skill_loader.load_all(db_instance=self.db)
        for skill in self.skills.values():
            for st in skill.tools:
                self.tools[st.name] = Tool(
                    name=st.name,
                    description=st.description,
                    parameters=st.parameters,
                    handler=st.handler,
                    requires_confirmation=st.requires_confirmation,
                )

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")
        return self.tools[name]

    def descriptions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "requires_confirmation": tool.requires_confirmation,
            }
            for tool in self.tools.values()
        ]

    def skill_instructions(self) -> dict[str, str]:
        """Returns mapping of skill names to their instructions/guidelines from SKILL.md."""
        return {
            name: skill.instructions
            for name, skill in self.skills.items()
            if skill.instructions
        }

    # --- Inventory & Product Catalog ---

    def get_inventory(self, product_id: str, **_: Any) -> dict[str, Any]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM products WHERE product_id=?", (product_id,)).fetchone()
        return {"found": bool(row), "product": dict(row) if row else None}

    def get_product(self, product_id: str, **_: Any) -> dict[str, Any]:
        prod = self.db.get_product(product_id)
        return {"found": bool(prod), "product": prod}

    def create_product(self, product_id: str, name: str, stock: int = 0, price: float = 0.0, **_: Any) -> dict[str, Any]:
        prod = self.db.create_product(product_id=product_id, name=name, stock=int(stock), price=float(price))
        return {"created": True, "product": prod, "message": f"Produit '{name}' ({product_id}) créé avec un stock de {stock} au prix de {price} TND."}

    def update_product_stock(self, product_id: str, stock: int | None = None, price: float | None = None, name: str | None = None, **_: Any) -> dict[str, Any]:
        stk = int(stock) if stock is not None else None
        prc = float(price) if price is not None else None
        prod = self.db.update_product(product_id=product_id, name=name, stock=stk, price=prc)
        if not prod:
            return {"updated": False, "error": f"Produit '{product_id}' introuvable."}
        return {"updated": True, "product": prod, "message": f"Produit '{product_id}' mis à jour (Stock: {prod['stock']}, Prix: {prod['price']} TND)."}

    def search_products(self, query: str, **_: Any) -> dict[str, Any]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM products WHERE product_id LIKE ? OR name LIKE ?",
                (f"%{query}%", f"%{query}%")
            ).fetchall()
        return {"products": [dict(row) for row in rows]}

    def create_sales_order(self, customer_id: str, items: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        order_id = f"SO-{uuid.uuid4().hex[:8].upper()}"
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO orders(order_id,customer_id,items,status,created_at) VALUES (?,?,?,?,?)",
                (order_id, customer_id, json.dumps(items), "created", utc_now())
            )
        return {"order_id": order_id, "customer_id": customer_id, "items": items, "status": "created"}

    def remember_fact(self, key: str, value: str, user_id: str = "anonymous", **_: Any) -> dict[str, Any]:
        self.db.remember(user_id, key, value)
        return {"saved": True, "key": key, "value": value}

    # --- Documents ---

    def get_document(self, doc_id: str, **_: Any) -> dict[str, Any]:
        doc = self.db.get_document(doc_id)
        return {"found": bool(doc), "document": doc}

    def search_documents(self, query: str, **_: Any) -> dict[str, Any]:
        docs = self.db.search_documents(query)
        return {"documents": docs, "count": len(docs)}

    def create_document(self, title: str, doc_type: str, content: str, doc_id: str | None = None, status: str = "draft", **_: Any) -> dict[str, Any]:
        doc = self.db.create_document(doc_id=doc_id, title=title, doc_type=doc_type, content=content, status=status)
        return {"created": True, "document": doc}

    def edit_document(self, doc_id: str, content: str, title: str | None = None, status: str | None = None, **_: Any) -> dict[str, Any]:
        updated = self.db.update_document(doc_id, content=content, title=title, status=status)
        if not updated:
            return {"updated": False, "error": f"Document {doc_id} not found"}
        return {"updated": True, "document": updated}

    def smart_edit_document(self, doc_id: str, instruction: str, **_: Any) -> dict[str, Any]:
        doc = self.db.get_document(doc_id)
        if not doc:
            return {"updated": False, "error": f"Document {doc_id} not found"}
        edit_result = self.editor.smart_edit(
            doc_id=doc_id,
            title=doc["title"],
            doc_type=doc["doc_type"],
            current_content=doc["content"],
            instruction=instruction,
        )
        updated = self.db.update_document(doc_id, content=edit_result["edited_content"])
        return {
            "updated": True,
            "document": updated,
            "explanation": edit_result.get("explanation"),
            "diff_summary": edit_result.get("diff_summary"),
        }

    def delete_document(self, doc_id: str, **_: Any) -> dict[str, Any]:
        deleted = self.db.delete_document(doc_id)
        return {"deleted": deleted, "doc_id": doc_id}
