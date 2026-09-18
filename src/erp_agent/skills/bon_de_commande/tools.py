from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from erp_agent.document_rendering import render_and_export_pdf
from erp_agent.skills.loader import SkillTool


def calculate_financials(
    items: list[dict[str, Any]],
    tax_rate: float = 19.0,
    global_discount_pct: float = 0.0,
    currency: str = "TND",
) -> dict[str, Any]:
    """Calculates all line item financials and document totals (HT, TVA, Remises, TTC)."""
    processed_items: list[dict[str, Any]] = []
    total_brut_ht = 0.0
    total_item_remises = 0.0

    for idx, item in enumerate(items, start=1):
        name = str(item.get("name") or item.get("product_id") or f"Article #{idx}")
        product_id = item.get("product_id")
        qty = float(item.get("quantity") or item.get("qty") or 1)
        unit_price = float(item.get("unit_price") or item.get("price") or 0.0)
        disc_pct = float(item.get("discount_pct") or item.get("remise") or 0.0)

        line_brut = round(qty * unit_price, 2)
        line_remise = round(line_brut * (disc_pct / 100.0), 2)
        line_net = round(line_brut - line_remise, 2)

        total_brut_ht += line_brut
        total_item_remises += line_remise

        processed_items.append({
            "line": idx,
            "product_id": product_id,
            "name": name,
            "quantity": qty,
            "unit_price": unit_price,
            "discount_pct": disc_pct,
            "line_brut_ht": line_brut,
            "line_remise": line_remise,
            "line_net_ht": line_net,
        })

    total_brut_ht = round(total_brut_ht, 2)
    total_item_remises = round(total_item_remises, 2)
    net_after_items = round(total_brut_ht - total_item_remises, 2)

    global_remise = round(net_after_items * (global_discount_pct / 100.0), 2) if global_discount_pct > 0 else 0.0
    total_net_ht = round(net_after_items - global_remise, 2)
    total_remises = round(total_item_remises + global_remise, 2)

    total_tva = round(total_net_ht * (tax_rate / 100.0), 2)
    total_ttc = round(total_net_ht + total_tva, 2)

    return {
        "items": processed_items,
        "item_count": len(processed_items),
        "tax_rate": tax_rate,
        "global_discount_pct": global_discount_pct,
        "total_brut_ht": total_brut_ht,
        "total_remises": total_remises,
        "total_net_ht": total_net_ht,
        "total_tva": total_tva,
        "total_ttc": total_ttc,
        "currency": currency,
    }


def build_bon_de_commande_payload(
    order: dict[str, Any],
    financials: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical structured representation used for BC rendering."""
    currency = order.get("currency", "TND")
    financials = financials or calculate_financials(
        order.get("items", []),
        tax_rate=float(order.get("tax_rate", 19.0)),
        global_discount_pct=float(order.get("discount_pct", 0.0)),
        currency=currency,
    )
    return {
        "schema_version": "1.0",
        "document_type": "bon_de_commande",
        "document_id": order["order_id"],
        "title": f"Bon de commande {order['order_id']} - {order['customer_id']}",
        "status": order.get("status", "draft"),
        "document_date": order.get("created_at", "")[:10],
        "client": {"name": order.get("customer_id", "")},
        "currency": currency,
        "items": financials.get("items", []),
        "financials": financials,
    }


def calculate_order_financials(order: dict[str, Any]) -> dict[str, Any]:
    """Calculate an order with its persisted document-level settings."""
    return calculate_financials(
        order.get("items", []),
        tax_rate=float(order.get("tax_rate", 19.0)),
        global_discount_pct=float(order.get("discount_pct", 0.0)),
        currency=order.get("currency", "TND"),
    )


def register_tools(db: Any = None, skill_dir: Path | None = None) -> list[SkillTool]:
    """Registers financial tools for Bon de Commande management."""

    def _lookup_catalog_product(pid_or_name: str) -> dict[str, Any] | None:
        if not db:
            return None
        with db.connect() as conn:
            row = conn.execute(
                "SELECT product_id, name, stock, price FROM products WHERE product_id=? OR name LIKE ? LIMIT 1",
                (pid_or_name, f"%{pid_or_name}%")
            ).fetchone()
        return dict(row) if row else None

    def create_bon_de_commande(
        client_name: str,
        items: list[dict[str, Any]] | None = None,
        tax_rate: float = 19.0,
        discount_pct: float = 0.0,
        order_id: str | None = None,
        currency: str = "TND",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Creates a new Bon de Commande (Purchase Order) with financial calculations."""
        if not db:
            return {"found": False, "error": "Database not available"}

        order_code = order_id if order_id and order_id.strip() else f"BC-{uuid.uuid4().hex[:6].upper()}"
        initial_items = items or []

        # Enhance items with catalog data if needed
        enriched_items = []
        for it in initial_items:
            p_id = it.get("product_id")
            p_name = it.get("name")
            p_info = None
            if p_id:
                p_info = _lookup_catalog_product(p_id)
            elif p_name:
                p_info = _lookup_catalog_product(p_name)

            name = it.get("name") or (p_info["name"] if p_info else p_id or "Article")
            price = float(it.get("unit_price") or it.get("price") or (p_info["price"] if p_info else 0.0))
            qty = float(it.get("quantity") or it.get("qty") or 1)
            disc = float(it.get("discount_pct") or it.get("remise") or 0.0)

            enriched_items.append({
                "product_id": p_info["product_id"] if p_info else p_id,
                "name": name,
                "quantity": qty,
                "unit_price": price,
                "discount_pct": disc,
            })

        financials = calculate_financials(
            enriched_items,
            tax_rate=tax_rate,
            global_discount_pct=discount_pct,
            currency=currency,
        )
        
        saved = db.save_order(
            order_id=order_code,
            customer_id=client_name,
            items=enriched_items,
            status="draft",
            currency=currency,
            tax_rate=tax_rate,
            discount_pct=discount_pct,
        )

        return {
            "found": True,
            "order_id": order_code,
            "client_name": client_name,
            "currency": currency,
            "status": "draft",
            "created_at": saved.get("created_at"),
            "financials": financials,
            "document": build_bon_de_commande_payload(saved, financials),
            "message": f"Bon de commande '{order_code}' créé avec succès pour le client '{client_name}' (Total TTC: {financials['total_ttc']} {currency}).",
        }

    def add_order_item(
        order_id: str,
        product_id: str | None = None,
        name: str | None = None,
        quantity: int | float = 1,
        unit_price: float | None = None,
        discount_pct: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Adds a new line item or product to an existing Bon de Commande."""
        if not db:
            return {"found": False, "error": "Database not available"}

        order = db.get_order(order_id)
        if not order:
            return {"found": False, "error": f"Bon de commande '{order_id}' introuvable"}

        # Lookup in catalog if price or name is missing
        p_info = None
        if product_id:
            p_info = _lookup_catalog_product(product_id)
        elif name:
            p_info = _lookup_catalog_product(name)

        item_name = name or (p_info["name"] if p_info else product_id or "Article")
        item_price = float(unit_price) if unit_price is not None else (p_info["price"] if p_info else 0.0)
        item_pid = p_info["product_id"] if p_info else product_id

        current_items = list(order.get("items", []))
        
        # Check if already in order -> update quantity
        matched = False
        for item in current_items:
            if (item_pid and item.get("product_id") == item_pid) or (item.get("name", "").lower() == item_name.lower()):
                item["quantity"] = float(item.get("quantity", 1)) + float(quantity)
                if unit_price is not None:
                    item["unit_price"] = item_price
                if discount_pct > 0:
                    item["discount_pct"] = discount_pct
                matched = True
                break

        if not matched:
            current_items.append({
                "product_id": item_pid,
                "name": item_name,
                "quantity": float(quantity),
                "unit_price": item_price,
                "discount_pct": float(discount_pct),
            })

        db.update_order(order_id, items=current_items)
        financials = calculate_order_financials({**order, "items": current_items})

        return {
            "found": True,
            "order_id": order_id,
            "client_name": order.get("customer_id"),
            "added_item": {"name": item_name, "quantity": quantity, "unit_price": item_price, "discount_pct": discount_pct},
            "financials": financials,
            "message": f"Article '{item_name}' (x{quantity}) ajouté au bon de commande '{order_id}'. Nouveau Total TTC: {financials['total_ttc']} TND.",
        }

    def update_order_item(
        order_id: str,
        item_index: int | None = None,
        product_id: str | None = None,
        name: str | None = None,
        quantity: int | float | None = None,
        unit_price: float | None = None,
        discount_pct: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Updates quantity, unit price, or discount on a specific line item in a Bon de Commande."""
        if not db:
            return {"found": False, "error": "Database not available"}

        order = db.get_order(order_id)
        if not order:
            return {"found": False, "error": f"Bon de commande '{order_id}' introuvable"}

        current_items = list(order.get("items", []))
        if not current_items:
            return {"found": False, "error": f"Le bon de commande '{order_id}' ne contient aucun article"}

        target_idx = None
        if item_index is not None and 1 <= item_index <= len(current_items):
            target_idx = item_index - 1
        else:
            for idx, it in enumerate(current_items):
                if (product_id and it.get("product_id") == product_id) or (name and name.lower() in it.get("name", "").lower()):
                    target_idx = idx
                    break

        if target_idx is None:
            return {"found": False, "error": f"Article cible introuvable dans le bon de commande '{order_id}'"}

        item = current_items[target_idx]
        if quantity is not None:
            item["quantity"] = float(quantity)
        if unit_price is not None:
            item["unit_price"] = float(unit_price)
        if discount_pct is not None:
            item["discount_pct"] = float(discount_pct)

        db.update_order(order_id, items=current_items)
        financials = calculate_order_financials({**order, "items": current_items})

        return {
            "found": True,
            "order_id": order_id,
            "updated_line": target_idx + 1,
            "item": item,
            "financials": financials,
            "message": f"Ligne #{target_idx + 1} ({item['name']}) mise à jour dans '{order_id}'. Nouveau Total TTC: {financials['total_ttc']} TND.",
        }

    def remove_order_item(
        order_id: str,
        item_index: int | None = None,
        product_id: str | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Removes a line item from a Bon de Commande."""
        if not db:
            return {"found": False, "error": "Database not available"}

        order = db.get_order(order_id)
        if not order:
            return {"found": False, "error": f"Bon de commande '{order_id}' introuvable"}

        current_items = list(order.get("items", []))
        target_idx = None
        if item_index is not None and 1 <= item_index <= len(current_items):
            target_idx = item_index - 1
        else:
            for idx, it in enumerate(current_items):
                if (product_id and it.get("product_id") == product_id) or (name and name.lower() in it.get("name", "").lower()):
                    target_idx = idx
                    break

        if target_idx is None:
            return {"found": False, "error": f"Article cible introuvable dans le bon de commande '{order_id}'"}

        removed = current_items.pop(target_idx)
        db.update_order(order_id, items=current_items)
        financials = calculate_order_financials({**order, "items": current_items})

        return {
            "found": True,
            "order_id": order_id,
            "removed_item": removed,
            "remaining_count": len(current_items),
            "financials": financials,
            "message": f"Article '{removed.get('name')}' retiré du bon de commande '{order_id}'. Nouveau Total TTC: {financials['total_ttc']} TND.",
        }

    def get_order_summary(order_id: str, **kwargs: Any) -> dict[str, Any]:
        """Retrieves and calculates financial summary, items, and tax totals for a Bon de Commande."""
        if not db:
            return {"found": False, "error": "Database not available"}

        order = db.get_order(order_id)
        if not order:
            return {"found": False, "error": f"Bon de commande '{order_id}' introuvable"}

        items = order.get("items", [])
        financials = calculate_order_financials(order)

        return {
            "found": True,
            "order_id": order["order_id"],
            "client_name": order["customer_id"],
            "status": order["status"],
            "created_at": order["created_at"],
            "financials": financials,
        }

    def export_bon_de_commande_pdf(order_id: str, output_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Render the canonical order JSON to HTML and pipe that HTML to WeasyPrint."""
        if not db:
            return {"found": False, "error": "Database not available"}
        order = db.get_order(order_id)
        if not order:
            return {"found": False, "error": f"Bon de commande '{order_id}' introuvable"}

        financials = calculate_order_financials(order)
        export_dir = Path("data/exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        target_path = Path(output_path) if output_path else export_dir / f"{order_id}_Bon_De_Commande.pdf"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            render_and_export_pdf(build_bon_de_commande_payload(order, financials), target_path)
            return {
                "found": True,
                "order_id": order_id,
                "pdf_path": str(target_path.resolve()),
                "filename": target_path.name,
                "download_url": f"/v1/exports/download/{target_path.name}",
                "financials": financials,
                "message": f"Bon de commande '{order_id}' exporte avec succes en PDF.",
            }
        except RuntimeError as exc:
            return {"found": False, "error": str(exc)}
        except Exception as exc:
            return {"found": False, "error": f"Failed to export Bon de Commande PDF: {exc}"}

    def _legacy_export_bon_de_commande_pdf(order_id: str, output_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Generates a professional Bon de Commande PDF with table layout and totals."""
        if not db:
            return {"found": False, "error": "Database not available"}

        order = db.get_order(order_id)
        if not order:
            return {"found": False, "error": f"Bon de commande '{order_id}' introuvable"}

        export_dir = Path("data/exports")
        export_dir.mkdir(parents=True, exist_ok=True)

        if not output_path:
            clean_filename = f"{order_id}_Bon_De_Commande.pdf"
            target_path = export_dir / clean_filename
        else:
            target_path = Path(output_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors

            doc = SimpleDocTemplate(
                str(target_path),
                pagesize=letter,
                rightMargin=36,
                leftMargin=36,
                topMargin=36,
                bottomMargin=36,
            )
            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                'BCTitle',
                parent=styles['Heading1'],
                fontSize=22,
                leading=26,
                textColor=colors.HexColor('#0f172a'),
                spaceAfter=6,
            )
            meta_style = ParagraphStyle(
                'BCMeta',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#475569'),
                spaceAfter=12,
            )
            cell_style = ParagraphStyle(
                'BCCell',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.HexColor('#1e293b'),
            )
            cell_bold = ParagraphStyle(
                'BCCellBold',
                parent=styles['Normal'],
                fontSize=9,
                leading=11,
                fontName="Helvetica-Bold",
                textColor=colors.HexColor('#0f172a'),
            )

            story = []
            story.append(Paragraph("<b>BON DE COMMANDE</b>", title_style))
            story.append(Paragraph(
                f"<b>Numéro:</b> {order_id} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Client:</b> {order['customer_id']} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Date:</b> {order['created_at'][:10]} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Statut:</b> {order['status'].upper()}",
                meta_style,
            ))
            story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3b82f6'), spaceAfter=15))

            # Financial calculations
            financials = calculate_financials(order.get("items", []))

            # Table Header
            table_data = [[
                Paragraph("<b>Réf</b>", cell_bold),
                Paragraph("<b>Désignation</b>", cell_bold),
                Paragraph("<b>Qté</b>", cell_bold),
                Paragraph("<b>P.U HT</b>", cell_bold),
                Paragraph("<b>Remise</b>", cell_bold),
                Paragraph("<b>Total Net HT</b>", cell_bold),
            ]]

            for item in financials["items"]:
                table_data.append([
                    Paragraph(str(item.get("product_id") or "-"), cell_style),
                    Paragraph(str(item.get("name")), cell_style),
                    Paragraph(f"{item['quantity']:g}", cell_style),
                    Paragraph(f"{item['unit_price']:.2f} TND", cell_style),
                    Paragraph(f"{item['discount_pct']:g}%" if item['discount_pct'] > 0 else "-", cell_style),
                    Paragraph(f"{item['line_net_ht']:.2f} TND", cell_style),
                ])

            # Table styling
            item_table = Table(table_data, colWidths=[65, 200, 45, 75, 55, 90])
            item_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(item_table)
            story.append(Spacer(1, 15))

            # Summary Totals Box
            totals_data = [
                [Paragraph("<b>Total Brut HT:</b>", cell_style), Paragraph(f"{financials['total_brut_ht']:.2f} TND", cell_bold)],
                [Paragraph("<b>Total Remises:</b>", cell_style), Paragraph(f"-{financials['total_remises']:.2f} TND", cell_style)],
                [Paragraph("<b>Total Net HT:</b>", cell_style), Paragraph(f"{financials['total_net_ht']:.2f} TND", cell_bold)],
                [Paragraph(f"<b>TVA ({financials['tax_rate']:g}%):</b>", cell_style), Paragraph(f"{financials['total_tva']:.2f} TND", cell_style)],
                [Paragraph("<b>TOTAL TTC NET:</b>", cell_bold), Paragraph(f"<b>{financials['total_ttc']:.2f} TND</b>", cell_bold)],
            ]
            totals_table = Table(totals_data, colWidths=[120, 100], hAlign='RIGHT')
            totals_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#94a3b8')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(totals_table)

            doc.build(story)

            return {
                "found": True,
                "order_id": order_id,
                "pdf_path": str(target_path.resolve()),
                "financials": financials,
                "message": f"Bon de commande '{order_id}' exporté avec succès en PDF à {target_path}",
            }
        except ImportError:
            return {"found": False, "error": "Export PDF requires reportlab (pip install reportlab)"}
        except Exception as exc:
            return {"found": False, "error": f"Failed to export Bon de Commande PDF: {exc}"}

    def list_bon_de_commandes(
        client_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Lists Bon de Commandes with client info, financial totals, and status."""
        if not db:
            return {"found": False, "error": "Database not available"}

        # Normalize 'all' / 'tous' / '' to None so no SQL status filter is applied
        _status_filter: str | None = status
        if not _status_filter or _status_filter.strip().lower() in ("all", "tous", "toutes", "*"):
            _status_filter = None

        orders = db.list_orders(limit=limit, customer_id=client_name, status=_status_filter)
        summary_list = []
        for o in orders:
            fin = calculate_order_financials(o)
            summary_list.append({
                "order_id": o["order_id"],
                "client_name": o.get("customer_id"),
                "status": o.get("status"),
                "created_at": o.get("created_at"),
                "item_count": fin["item_count"],
                "total_ttc": fin["total_ttc"],
            })

        return {
            "found": True,
            "count": len(summary_list),
            "orders": summary_list,
            "message": f"{len(summary_list)} bon(s) de commande trouvé(s).",
        }

    return [
        SkillTool(
            name="create_bon_de_commande",
            description="Create a new Bon de Commande (Purchase/Sales Order) with client name, line items, and financial tax/discount calculation.",
            parameters={
                "client_name": "string",
                "items": "array",
                "tax_rate": "number",
                "discount_pct": "number",
                "order_id": "string",
            },
            handler=create_bon_de_commande,
            requires_confirmation=False,
        ),
        SkillTool(
            name="list_bon_de_commandes",
            description="List all Bon de Commandes with client name, item count, status, and Total TTC.",
            parameters={
                "client_name": "string",
                "status": "string",
                "limit": "integer",
            },
            handler=list_bon_de_commandes,
            requires_confirmation=False,
        ),
        SkillTool(
            name="add_order_item",
            description="Add a new line item or product to an existing Bon de Commande. Checks catalog price and stock if product_id is given.",
            parameters={
                "order_id": "string",
                "product_id": "string",
                "name": "string",
                "quantity": "number",
                "unit_price": "number",
                "discount_pct": "number",
            },
            handler=add_order_item,
            requires_confirmation=False,
        ),
        SkillTool(
            name="update_order_item",
            description="Update an existing line item's quantity, unit price, or discount in a Bon de Commande. Requires approval.",
            parameters={
                "order_id": "string",
                "item_index": "integer",
                "product_id": "string",
                "name": "string",
                "quantity": "number",
                "unit_price": "number",
                "discount_pct": "number",
            },
            handler=update_order_item,
            requires_confirmation=True,
        ),
        SkillTool(
            name="remove_order_item",
            description="Remove a line item from an existing Bon de Commande. Requires approval.",
            parameters={
                "order_id": "string",
                "item_index": "integer",
                "product_id": "string",
                "name": "string",
            },
            handler=remove_order_item,
            requires_confirmation=True,
        ),
        SkillTool(
            name="get_order_summary",
            description="Retrieve and calculate financial summary, itemized line totals, VAT (TVA), discounts, and net total (TTC) for a Bon de Commande.",
            parameters={"order_id": "string"},
            handler=get_order_summary,
            requires_confirmation=False,
        ),
        SkillTool(
            name="export_bon_de_commande_pdf",
            description="Export a Bon de Commande into a formatted PDF document with company layout, table grid, and financial summary box.",
            parameters={"order_id": "string", "output_path": "string"},
            handler=export_bon_de_commande_pdf,
            requires_confirmation=False,
        ),
    ]
