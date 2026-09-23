from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

class Database:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def init_schema(self) -> None:
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(user_id, key)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    token TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS products (
                    product_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    stock INTEGER NOT NULL,
                    price REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    items TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    currency TEXT NOT NULL DEFAULT 'TND',
                    tax_rate REAL NOT NULL DEFAULT 19.0,
                    discount_pct REAL NOT NULL DEFAULT 0.0
                );
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_states (
                    session_id TEXT PRIMARY KEY,
                    active_invoice_id TEXT,
                    active_bc_id TEXT,
                    active_product_id TEXT,
                    active_doc_id TEXT,
                    active_client TEXT,
                    active_financial_document_id TEXT,
                    active_financial_document_type TEXT,
                    updated_at TEXT NOT NULL
                );
            """)
            order_columns = {row["name"] for row in db.execute("PRAGMA table_info(orders)").fetchall()}
            if "currency" not in order_columns:
                db.execute("ALTER TABLE orders ADD COLUMN currency TEXT NOT NULL DEFAULT 'TND'")
            if "tax_rate" not in order_columns:
                db.execute("ALTER TABLE orders ADD COLUMN tax_rate REAL NOT NULL DEFAULT 19.0")
            if "discount_pct" not in order_columns:
                db.execute("ALTER TABLE orders ADD COLUMN discount_pct REAL NOT NULL DEFAULT 0.0")
            session_state_columns = {row["name"] for row in db.execute("PRAGMA table_info(session_states)").fetchall()}
            if "active_financial_document_id" not in session_state_columns:
                db.execute("ALTER TABLE session_states ADD COLUMN active_financial_document_id TEXT")
            if "active_financial_document_type" not in session_state_columns:
                db.execute("ALTER TABLE session_states ADD COLUMN active_financial_document_type TEXT")
            if db.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0:
                db.executemany(
                    "INSERT INTO products(product_id,name,stock,price) VALUES (?,?,?,?)",
                    [("P-100", "Laptop Pro", 12, 2499.0),
                     ("P-200", "Wireless Mouse", 48, 75.0),
                     ("P-300", "USB-C Dock", 7, 199.0)]
                )
            if db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0:
                now = utc_now()
                db.executemany(
                    "INSERT INTO documents(doc_id,title,doc_type,content,status,updated_at,created_at) VALUES (?,?,?,?,?,?,?)",
                    [
                        ("DOC-101", "Client Invoice 101", "invoice", "Invoice for Client Alpha - 12x Wireless Mouse @ 75 TND. Total: 900 TND. Payment due in 30 days.", "draft", now, now),
                        ("DOC-102", "Purchase Order 102", "purchase_order", "PO for Supplier TechCorp - 5x Laptop Pro @ 2200 TND. Delivery scheduled for next week.", "approved", now, now),
                        ("DOC-103", "Service Agreement Alpha", "contract", "Standard SLA: 99.9% uptime and 24/7 emergency response support.", "active", now, now),
                    ]
                )

    def add_message(self, session_id: str, user_id: str, role: str, content: str) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO messages(session_id,user_id,role,content,created_at) VALUES (?,?,?,?,?)",
                (session_id, user_id, role, content, utc_now())
            )

    def recent_messages(self, session_id: str, limit: int = 30) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT role,content,created_at FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit)
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_session_state(self, session_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM session_states WHERE session_id=?", (session_id,)).fetchone()
        if not row:
            return {
                "session_id": session_id,
                "active_invoice_id": None,
                "active_bc_id": None,
                "active_product_id": None,
                "active_doc_id": None,
                "active_client": None,
                "active_financial_document_id": None,
                "active_financial_document_type": None,
                "updated_at": utc_now(),
            }
        return dict(row)

    def update_session_state(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        current = self.get_session_state(session_id)
        for key, val in kwargs.items():
            if val is not None and key in current:
                current[key] = val
        now = utc_now()
        current["updated_at"] = now

        with self.connect() as db:
            db.execute(
                "INSERT INTO session_states(session_id, active_invoice_id, active_bc_id, active_product_id, active_doc_id, active_client, active_financial_document_id, active_financial_document_type, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "active_invoice_id=excluded.active_invoice_id, "
                "active_bc_id=excluded.active_bc_id, "
                "active_product_id=excluded.active_product_id, "
                "active_doc_id=excluded.active_doc_id, "
                "active_client=excluded.active_client, "
                "active_financial_document_id=excluded.active_financial_document_id, "
                "active_financial_document_type=excluded.active_financial_document_type, "
                "updated_at=excluded.updated_at",
                (
                    session_id,
                    current.get("active_invoice_id"),
                    current.get("active_bc_id"),
                    current.get("active_product_id"),
                    current.get("active_doc_id"),
                    current.get("active_client"),
                    current.get("active_financial_document_id"),
                    current.get("active_financial_document_type"),
                    now,
                ),
            )
        return current

    def add_event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO events(session_id,event_type,payload,created_at) VALUES (?,?,?,?)",
                (session_id, event_type, json.dumps(payload, ensure_ascii=False), utc_now())
            )

    def events(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM events WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def remember(self, user_id: str, key: str, value: str) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO memories(user_id,key,value,created_at) VALUES (?,?,?,?) "
                "ON CONFLICT(user_id,key) DO UPDATE SET value=excluded.value,created_at=excluded.created_at",
                (user_id, key, value, utc_now())
            )

    def memories(self, user_id: str) -> dict[str, str]:
        with self.connect() as db:
            rows = db.execute("SELECT key,value FROM memories WHERE user_id=?", (user_id,)).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def create_approval(self, token: str, session_id: str, user_id: str, tool_name: str, arguments: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO approvals(token,session_id,user_id,tool_name,arguments,status,created_at) VALUES (?,?,?,?,?,?,?)",
                (token, session_id, user_id, tool_name, json.dumps(arguments), "pending", utc_now())
            )

    def get_approval(self, token: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM approvals WHERE token=?", (token,)).fetchone()
        return self._approval_row(row)

    def claim_approval(self, token: str) -> dict[str, Any] | None:
        """Atomically move a pending approval to executing. Returns None if not pending."""
        with self.connect() as db:
            row = db.execute("SELECT * FROM approvals WHERE token=?", (token,)).fetchone()
            if not row:
                return None
            if row["status"] != "pending":
                # Already executed, executing, or failed — do not re-run
                return None
            updated = db.execute(
                "UPDATE approvals SET status='executing' "
                "WHERE token=? AND status='pending'",
                (token,),
            ).rowcount
            if updated == 1:
                row = db.execute("SELECT * FROM approvals WHERE token=?", (token,)).fetchone()
            else:
                return None  # Lost race with concurrent claim
        return self._approval_row(row)

    @staticmethod
    def _approval_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if not row:
            return None
        result = dict(row)
        result["arguments"] = json.loads(result["arguments"])
        return result

    def finish_approval(self, token: str, status: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE approvals SET status=? WHERE token=?", (status, token))

    # --- Products Catalog Management ---

    def get_product(self, product_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM products WHERE product_id=?", (product_id,)).fetchone()
        return dict(row) if row else None

    def create_product(self, product_id: str, name: str, stock: int = 0, price: float = 0.0) -> dict[str, Any]:
        with self.connect() as db:
            db.execute(
                "INSERT INTO products(product_id,name,stock,price) VALUES (?,?,?,?)",
                (product_id, name, stock, price),
            )
        return {"product_id": product_id, "name": name, "stock": stock, "price": price}

    def update_product(
        self,
        product_id: str,
        name: str | None = None,
        stock: int | None = None,
        price: float | None = None,
    ) -> dict[str, Any] | None:
        prod = self.get_product(product_id)
        if not prod:
            return None
        new_name = name if name is not None else prod["name"]
        new_stock = stock if stock is not None else prod["stock"]
        new_price = price if price is not None else prod["price"]
        with self.connect() as db:
            db.execute(
                "UPDATE products SET name=?, stock=?, price=? WHERE product_id=?",
                (new_name, new_stock, new_price, product_id),
            )
        return {"product_id": product_id, "name": new_name, "stock": new_stock, "price": new_price}

    # --- Orders / Bon de Commande Management ---

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM orders WHERE order_id=?", (order_id,)).fetchone()
        if not row:
            return None
        res = dict(row)
        try:
            res["items"] = json.loads(res["items"])
        except Exception:
            res["items"] = []
        return res

    def list_orders(self, limit: int = 50, customer_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM orders WHERE 1=1"
        params: list[Any] = []
        if customer_id:
            query += " AND customer_id LIKE ?"
            params.append(f"%{customer_id}%")
        if status:
            query += " AND status=?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self.connect() as db:
            rows = db.execute(query, params).fetchall()
        orders = []
        for r in rows:
            d = dict(r)
            try:
                d["items"] = json.loads(d["items"])
            except Exception:
                d["items"] = []
            orders.append(d)
        return orders

    def save_order(
        self,
        order_id: str,
        customer_id: str,
        items: list[dict[str, Any]],
        status: str = "draft",
        currency: str = "TND",
        tax_rate: float = 19.0,
        discount_pct: float = 0.0,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO orders(order_id,customer_id,items,status,created_at,currency,tax_rate,discount_pct) VALUES (?,?,?,?,?,?,?,?)",
                (order_id, customer_id, json.dumps(items, ensure_ascii=False), status, now, currency, tax_rate, discount_pct)
            )
        return {
            "order_id": order_id, "customer_id": customer_id, "items": items, "status": status,
            "created_at": now, "currency": currency, "tax_rate": tax_rate, "discount_pct": discount_pct,
        }

    def update_order(
        self,
        order_id: str,
        items: list[dict[str, Any]] | None = None,
        status: str | None = None,
        customer_id: str | None = None,
        currency: str | None = None,
        tax_rate: float | None = None,
        discount_pct: float | None = None,
    ) -> dict[str, Any] | None:
        order = self.get_order(order_id)
        if not order:
            return None
        new_items = items if items is not None else order["items"]
        new_status = status if status is not None else order["status"]
        new_customer = customer_id if customer_id is not None else order["customer_id"]
        new_currency = currency if currency is not None else order.get("currency", "TND")
        new_tax_rate = tax_rate if tax_rate is not None else order.get("tax_rate", 19.0)
        new_discount_pct = discount_pct if discount_pct is not None else order.get("discount_pct", 0.0)
        with self.connect() as db:
            db.execute(
                "UPDATE orders SET items=?, status=?, customer_id=?, currency=?, tax_rate=?, discount_pct=? WHERE order_id=?",
                (json.dumps(new_items, ensure_ascii=False), new_status, new_customer, new_currency, new_tax_rate, new_discount_pct, order_id)
            )
        order["items"] = new_items
        order["status"] = new_status
        order["customer_id"] = new_customer
        order["currency"] = new_currency
        order["tax_rate"] = new_tax_rate
        order["discount_pct"] = new_discount_pct
        return order

    def delete_order(self, order_id: str) -> bool:
        with self.connect() as db:
            cur = db.execute("DELETE FROM orders WHERE order_id=?", (order_id,))
            return cur.rowcount > 0

    # --- Document Management ---

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self, limit: int = 50, doc_type: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM documents WHERE 1=1"
        params: list[Any] = []
        if doc_type:
            query += " AND doc_type=?"
            params.append(doc_type)
        if status:
            query += " AND status=?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with self.connect() as db:
            rows = db.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def search_documents(self, query: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM documents WHERE doc_id LIKE ? OR title LIKE ? OR content LIKE ? OR doc_type LIKE ?",
                (f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%")
            ).fetchall()
        return [dict(row) for row in rows]

    def create_document(self, doc_id: str | None, title: str, doc_type: str, content: str, status: str = "draft") -> dict[str, Any]:
        import uuid
        actual_doc_id = doc_id if doc_id and doc_id.strip() else f"DOC-{uuid.uuid4().hex[:4].upper()}"
        now = utc_now()
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO documents(doc_id,title,doc_type,content,status,updated_at,created_at) VALUES (?,?,?,?,?,?,?)",
                (actual_doc_id, title, doc_type, content, status, now, now)
            )
        return {"doc_id": actual_doc_id, "title": title, "doc_type": doc_type, "content": content, "status": status, "updated_at": now, "created_at": now}

    def delete_document(self, doc_id: str) -> bool:
        with self.connect() as db:
            cur = db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            return cur.rowcount > 0

    def update_document(self, doc_id: str, content: str | None = None, title: str | None = None, status: str | None = None) -> dict[str, Any] | None:
        doc = self.get_document(doc_id)
        if not doc:
            return None
        new_content = content if content is not None else doc["content"]
        new_title = title if title is not None else doc["title"]
        new_status = status if status is not None else doc["status"]
        now = utc_now()
        with self.connect() as db:
            db.execute(
                "UPDATE documents SET content=?, title=?, status=?, updated_at=? WHERE doc_id=?",
                (new_content, new_title, new_status, now, doc_id)
            )
        return {
            "doc_id": doc_id,
            "title": new_title,
            "doc_type": doc["doc_type"],
            "content": new_content,
            "status": new_status,
            "updated_at": now,
            "previous_content": doc["content"],
        }
