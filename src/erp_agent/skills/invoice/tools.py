from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from erp_agent.document_rendering import (
    parse_json_content,
    render_and_export_pdf,
)
from erp_agent.skills.loader import SkillTool


def calculate_invoice_financials(
    items: list[dict[str, Any]],
    tax_rate: float = 19.0,
    global_discount_pct: float = 0.0,
    timbre_fiscal: float = 1.0,
    currency: str = "TND",
) -> dict[str, Any]:
    """Calculates all line item financials and document totals (HT, TVA, Remises, Timbre, TTC)."""
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
    applicable_timbre = timbre_fiscal if currency.upper() == "TND" else 0.0
    total_ttc = round(total_net_ht + total_tva + applicable_timbre, 2)

    return {
        "items": processed_items,
        "item_count": len(processed_items),
        "tax_rate": tax_rate,
        "global_discount_pct": global_discount_pct,
        "total_brut_ht": total_brut_ht,
        "total_remises": total_remises,
        "total_net_ht": total_net_ht,
        "total_tva": total_tva,
        "timbre_fiscal": applicable_timbre,
        "total_ttc": total_ttc,
        "currency": currency,
    }


def format_invoice_markdown(
    invoice_id: str,
    client_name: str,
    client_tax_id: str | None,
    financials: dict[str, Any],
    invoice_date: str | None = None,
    due_date: str | None = None,
    payment_terms: str = "30 jours fin de mois",
    status: str = "draft",
) -> str:
    """Formats an invoice as structured Markdown for direct live view and editing."""
    cur = financials.get("currency", "TND")
    inv_date = invoice_date or datetime.now(timezone.utc).strftime("%d/%m/%Y")
    d_date = due_date or "À réception"
    tax_id_str = client_tax_id or "Non spécifié"

    lines_table = [
        "| # | Réf / Produit | Désignation | Qté | Prix Unit. HT | Remise % | Total Net HT |",
        "|---|---|---|---|---|---|---|",
    ]

    for item in financials.get("items", []):
        ref = item.get("product_id") or "-"
        name = item.get("name", "Article")
        qty = item.get("quantity", 1)
        price = f"{item.get('unit_price', 0.0):.2f} {cur}"
        disc = f"{item.get('discount_pct', 0.0):.1f}%" if item.get("discount_pct") else "0%"
        total = f"{item.get('line_net_ht', 0.0):.2f} {cur}"
        lines_table.append(f"| {item.get('line', 1)} | {ref} | {name} | {qty} | {price} | {disc} | {total} |")

    if not financials.get("items"):
        lines_table.append("| - | - | *Aucun article spécifié* | - | - | - | 0.00 TND |")

    timbre_line = ""
    if financials.get("timbre_fiscal", 0) > 0:
        timbre_line = f"\n| **Timbre Fiscal** | {financials.get('timbre_fiscal', 1.0):.3f} {cur} |"

    md = f"""# FACTURE N° {invoice_id}

**Date d'émission :** {inv_date}  
**Date d'échéance :** {d_date}  
**Client :** {client_name}  
**Matricule Fiscal / Tax ID :** {tax_id_str}  
**Conditions de règlement :** {payment_terms}  
**Statut :** {status.upper()}

---

### Détail des Articles & Prestations

{chr(10).join(lines_table)}

---

### Récapitulatif Financier

| Libellé | Montant ({cur}) |
|---|---|
| **Total Brut HT** | {financials.get('total_brut_ht', 0.0):.2f} {cur} |
| **Total Remises** | -{financials.get('total_remises', 0.0):.2f} {cur} |
| **Total Net HT** | {financials.get('total_net_ht', 0.0):.2f} {cur} |
| **TVA ({financials.get('tax_rate', 19.0):.1f}%)** | {financials.get('total_tva', 0.0):.2f} {cur} |{timbre_line}
| **TOTAL TTC À PAYER** | **{financials.get('total_ttc', 0.0):.2f} {cur}** |

---
*Document généré par WIND ERP Assistant Agentique*
"""
    return md.strip()


def build_invoice_payload(
    invoice_id: str,
    client_name: str,
    client_tax_id: str | None,
    financials: dict[str, Any],
    invoice_date: str | None = None,
    due_date: str | None = None,
    payment_terms: str = "30 jours fin de mois",
    status: str = "draft",
) -> dict[str, Any]:
    """Build the canonical structured representation used across the backend."""
    return {
        "schema_version": "1.0",
        "document_type": "invoice",
        "document_id": invoice_id,
        "title": f"Facture {invoice_id} - {client_name}",
        "status": status,
        "invoice_date": invoice_date or datetime.now(timezone.utc).strftime("%d/%m/%Y"),
        "due_date": due_date or "A reception",
        "payment_terms": payment_terms,
        "client": {"name": client_name, "tax_id": client_tax_id},
        "currency": financials.get("currency", "TND"),
        "items": financials.get("items", []),
        "financials": financials,
    }


def format_invoice_json(
    invoice_id: str,
    client_name: str,
    client_tax_id: str | None,
    financials: dict[str, Any],
    invoice_date: str | None = None,
    due_date: str | None = None,
    payment_terms: str = "30 jours fin de mois",
    status: str = "draft",
) -> str:
    """Serialize the canonical invoice payload for persistence and API responses."""
    return json.dumps(
        build_invoice_payload(
            invoice_id, client_name, client_tax_id, financials,
            invoice_date, due_date, payment_terms, status,
        ),
        ensure_ascii=False,
        indent=2,
    )


def parse_invoice_markdown(content: str) -> dict[str, Any]:
    """Parses an invoice Markdown document to extract metadata, client info, and line items."""
    payload = parse_json_content(content)
    if payload and payload.get("document_type") == "invoice":
        client = payload.get("client") or {}
        return {
            **payload,
            "client_name": client.get("name") or payload.get("client_name") or "Client Inconnu",
            "client_tax_id": client.get("tax_id") or payload.get("client_tax_id"),
            "invoice_date": payload.get("invoice_date"),
            "due_date": payload.get("due_date"),
            "payment_terms": payload.get("payment_terms", "30 jours fin de mois"),
            "status": payload.get("status", "draft"),
            "items": payload.get("items") or payload.get("financials", {}).get("items", []),
        }
    client_match = re.search(r"\*\*Client\s*:\*\*\s*(.+)", content, re.I)
    client_name = client_match.group(1).strip() if client_match else "Client Inconnu"

    tax_id_match = re.search(r"\*\*(?:Matricule Fiscal\s*/\s*Tax ID|Matricule Fiscal|Tax ID|MF)\s*:\*\*\s*(.+)", content, re.I)
    tax_id = tax_id_match.group(1).strip() if tax_id_match else None
    if tax_id == "Non spécifié":
        tax_id = None

    date_match = re.search(r"\*\*Date d'émission\s*:\*\*\s*(.+)", content, re.I)
    invoice_date = date_match.group(1).strip() if date_match else None

    due_match = re.search(r"\*\*Date d'échéance\s*:\*\*\s*(.+)", content, re.I)
    due_date = due_match.group(1).strip() if due_match else None

    status_match = re.search(r"\*\*Statut\s*:\*\*\s*(.+)", content, re.I)
    status = status_match.group(1).strip().lower() if status_match else "draft"

    items: list[dict[str, Any]] = []
    table_lines = [line.strip() for line in content.splitlines() if line.strip().startswith("|")]
    for line in table_lines:
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) >= 7 and parts[0].isdigit():
            line_no = int(parts[0])
            ref = parts[1] if parts[1] != "-" else None
            name = parts[2]
            try:
                qty = float(parts[3])
            except ValueError:
                qty = 1.0
            price_match = re.search(r"([\d\s.,]+)", parts[4])
            price = float(price_match.group(1).replace(" ", "").replace(",", ".")) if price_match else 0.0
            disc_match = re.search(r"([\d.]+)", parts[5])
            disc = float(disc_match.group(1)) if disc_match else 0.0

            items.append({
                "line": line_no,
                "product_id": ref,
                "name": name,
                "quantity": qty,
                "unit_price": price,
                "discount_pct": disc,
            })

    return {
        "client_name": client_name,
        "client_tax_id": tax_id,
        "invoice_date": invoice_date,
        "due_date": due_date,
        "status": status,
        "items": items,
    }


def recalculate_invoice_financials(parsed: dict[str, Any]) -> dict[str, Any]:
    """Recalculate an invoice while preserving its saved financial settings."""
    saved_financials = parsed.get("financials") or {}
    return calculate_invoice_financials(
        parsed.get("items", []),
        tax_rate=float(saved_financials.get("tax_rate", 19.0)),
        global_discount_pct=float(saved_financials.get("global_discount_pct", 0.0)),
        currency=saved_financials.get("currency") or parsed.get("currency") or "TND",
    )


def register_tools(db: Any = None, skill_dir: Path | None = None) -> list[SkillTool]:
    """Registers financial and compliance tools for Invoice management."""

    def _lookup_catalog_product(pid_or_name: str) -> dict[str, Any] | None:
        if not db:
            return None
        with db.connect() as conn:
            row = conn.execute(
                "SELECT product_id, name, stock, price FROM products WHERE product_id=? OR name LIKE ? LIMIT 1",
                (pid_or_name, f"%{pid_or_name}%")
            ).fetchone()
        return dict(row) if row else None

    def create_invoice(
        client_name: str,
        client_tax_id: str | None = None,
        items: list[dict[str, Any]] | None = None,
        invoice_id: str | None = None,
        due_date: str | None = None,
        payment_terms: str = "30 jours fin de mois",
        tax_rate: float = 19.0,
        discount_pct: float = 0.0,
        currency: str = "TND",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Creates a new invoice document with automatic financial and tax calculations."""
        if not db:
            return {"found": False, "error": "Database not available"}

        inv_code = invoice_id if invoice_id and invoice_id.strip() else f"INV-{uuid.uuid4().hex[:6].upper()}"
        initial_items = items or []

        enriched_items = []
        for it in initial_items:
            p_id = it.get("product_id")
            p_name = it.get("name")
            p_info = None
            if p_id:
                p_info = _lookup_catalog_product(p_id)
            elif p_name:
                p_info = _lookup_catalog_product(p_name)

            name = it.get("name") or (p_info["name"] if p_info else p_id or "Prestation")
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

        fin = calculate_invoice_financials(
            enriched_items,
            tax_rate=tax_rate,
            global_discount_pct=discount_pct,
            currency=currency,
        )

        md_content = format_invoice_json(
            invoice_id=inv_code,
            client_name=client_name,
            client_tax_id=client_tax_id,
            financials=fin,
            due_date=due_date,
            payment_terms=payment_terms,
            status="draft",
        )

        doc = db.create_document(
            doc_id=inv_code,
            title=f"Facture {inv_code} - {client_name}",
            doc_type="invoice",
            content=md_content,
            status="draft",
        )

        return {
            "found": True,
            "invoice_id": inv_code,
            "doc_id": inv_code,
            "client_name": client_name,
            "financials": fin,
            "document": doc,
            "message": f"Facture {inv_code} créée avec succès pour {client_name} (Total TTC: {fin['total_ttc']} {currency}).",
        }

    def get_invoice_summary(invoice_id: str, **kwargs: Any) -> dict[str, Any]:
        """Calculates and returns the complete breakdown and financial totals for an invoice."""
        if not db:
            return {"found": False, "error": "Database not available"}

        doc = db.get_document(invoice_id)
        if not doc:
            return {"found": False, "error": f"Facture {invoice_id} introuvable."}

        parsed = parse_invoice_markdown(doc["content"])
        fin = recalculate_invoice_financials(parsed)

        return {
            "found": True,
            "invoice_id": invoice_id,
            "doc_id": invoice_id,
            "title": doc["title"],
            "client_name": parsed["client_name"],
            "client_tax_id": parsed["client_tax_id"],
            "status": doc["status"],
            "financials": fin,
            "document": doc,
        }

    def add_invoice_item(
        invoice_id: str,
        name: str,
        quantity: float = 1,
        unit_price: float = 0.0,
        discount_pct: float = 0.0,
        product_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Adds a new line item to an existing invoice and recalculates totals."""
        if not db:
            return {"found": False, "error": "Database not available"}

        doc = db.get_document(invoice_id)
        if not doc:
            return {"found": False, "error": f"Facture {invoice_id} introuvable."}

        parsed = parse_invoice_markdown(doc["content"])
        p_info = _lookup_catalog_product(product_id or name)

        item_name = name or (p_info["name"] if p_info else "Article")
        item_price = float(unit_price if unit_price > 0 else (p_info["price"] if p_info else 0.0))
        item_pid = product_id or (p_info["product_id"] if p_info else None)

        parsed["items"].append({
            "product_id": item_pid,
            "name": item_name,
            "quantity": float(quantity),
            "unit_price": item_price,
            "discount_pct": float(discount_pct),
        })

        fin = recalculate_invoice_financials(parsed)
        new_md = format_invoice_json(
            invoice_id=invoice_id,
            client_name=parsed["client_name"],
            client_tax_id=parsed["client_tax_id"],
            financials=fin,
            invoice_date=parsed["invoice_date"],
            due_date=parsed["due_date"],
            status=doc["status"],
        )

        updated_doc = db.update_document(invoice_id, content=new_md)

        return {
            "found": True,
            "invoice_id": invoice_id,
            "doc_id": invoice_id,
            "financials": fin,
            "document": updated_doc,
            "message": f"Article '{item_name}' (x{quantity}) ajouté à la facture {invoice_id}. Nouveau Total TTC : {fin['total_ttc']} TND.",
        }

    def update_invoice_item(
        invoice_id: str,
        line_number: int | None = None,
        product_id: str | None = None,
        name: str | None = None,
        quantity: float | None = None,
        unit_price: float | None = None,
        discount_pct: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Updates an existing line item in an invoice."""
        if not db:
            return {"found": False, "error": "Database not available"}

        doc = db.get_document(invoice_id)
        if not doc:
            return {"found": False, "error": f"Facture {invoice_id} introuvable."}

        parsed = parse_invoice_markdown(doc["content"])
        items = parsed["items"]

        target_idx = None
        if line_number is not None and 1 <= line_number <= len(items):
            target_idx = line_number - 1
        elif product_id:
            for i, it in enumerate(items):
                if it.get("product_id") == product_id:
                    target_idx = i
                    break
        elif name:
            for i, it in enumerate(items):
                if name.lower() in it.get("name", "").lower():
                    target_idx = i
                    break

        if target_idx is None:
            return {"found": False, "error": f"Ligne d'article introuvable dans la facture {invoice_id}."}

        it = items[target_idx]
        if quantity is not None:
            it["quantity"] = float(quantity)
        if unit_price is not None:
            it["unit_price"] = float(unit_price)
        if discount_pct is not None:
            it["discount_pct"] = float(discount_pct)

        fin = recalculate_invoice_financials(parsed)
        new_md = format_invoice_json(
            invoice_id=invoice_id,
            client_name=parsed["client_name"],
            client_tax_id=parsed["client_tax_id"],
            financials=fin,
            invoice_date=parsed["invoice_date"],
            due_date=parsed["due_date"],
            status=doc["status"],
        )

        updated_doc = db.update_document(invoice_id, content=new_md)
        return {
            "found": True,
            "invoice_id": invoice_id,
            "doc_id": invoice_id,
            "financials": fin,
            "document": updated_doc,
            "message": f"Ligne #{target_idx + 1} mise à jour dans la facture {invoice_id}. Nouveau Total TTC : {fin['total_ttc']} TND.",
        }

    def update_invoice(
        invoice_id: str,
        client_name: str | None = None,
        new_item_name: str | None = None,
        currency: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update invoice-level details from a confirmed conversational follow-up."""
        if not db:
            return {"found": False, "error": "Database not available"}

        doc = db.get_document(invoice_id)
        if not doc:
            return {"found": False, "error": f"Facture {invoice_id} introuvable."}

        parsed = parse_invoice_markdown(doc["content"])
        old_client = parsed["client_name"]
        old_currency = (parsed.get("financials") or {}).get("currency", "TND")
        changes: list[str] = []

        if client_name:
            parsed["client_name"] = client_name
            changes.append(f"client = {client_name}")

        if new_item_name:
            target = next(
                (item for item in parsed["items"] if "device" in str(item.get("name", "")).lower()),
                parsed["items"][0] if parsed["items"] else None,
            )
            if target is None:
                return {"found": False, "error": f"Aucune ligne à renommer dans la facture {invoice_id}."}
            old_name = target.get("name", "Article")
            target["name"] = new_item_name
            changes.append(f"article = {new_item_name}")

        selected_currency = (currency or old_currency).upper()
        if currency:
            changes.append(f"devise = {selected_currency}")

        financials = parsed.get("financials") or {}
        fin = calculate_invoice_financials(
            parsed["items"],
            tax_rate=float(financials.get("tax_rate", 19.0)),
            global_discount_pct=float(financials.get("global_discount_pct", 0.0)),
            currency=selected_currency,
        )
        new_content = format_invoice_json(
            invoice_id=invoice_id,
            client_name=parsed["client_name"],
            client_tax_id=parsed["client_tax_id"],
            financials=fin,
            invoice_date=parsed.get("invoice_date"),
            due_date=parsed.get("due_date"),
            payment_terms=parsed.get("payment_terms", "30 jours fin de mois"),
            status=parsed.get("status", doc["status"]),
        )
        updated_doc = db.update_document(
            invoice_id,
            content=new_content,
            title=f"Facture {invoice_id} - {parsed['client_name']}",
        )
        return {
            "found": True,
            "invoice_id": invoice_id,
            "doc_id": invoice_id,
            "client_name": parsed["client_name"],
            "financials": fin,
            "document": updated_doc,
            "message": f"Facture {invoice_id} mise à jour ({'; '.join(changes) or 'aucun changement'}).",
        }

    def remove_invoice_item(
        invoice_id: str,
        line_number: int | None = None,
        product_id: str | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Removes a line item from an invoice."""
        if not db:
            return {"found": False, "error": "Database not available"}

        doc = db.get_document(invoice_id)
        if not doc:
            return {"found": False, "error": f"Facture {invoice_id} introuvable."}

        parsed = parse_invoice_markdown(doc["content"])
        items = parsed["items"]

        target_idx = None
        if line_number is not None and 1 <= line_number <= len(items):
            target_idx = line_number - 1
        elif product_id:
            for i, it in enumerate(items):
                if it.get("product_id") == product_id:
                    target_idx = i
                    break
        elif name:
            for i, it in enumerate(items):
                if name.lower() in it.get("name", "").lower():
                    target_idx = i
                    break

        if target_idx is None:
            return {"found": False, "error": f"Ligne d'article introuvable dans la facture {invoice_id}."}

        removed = items.pop(target_idx)
        fin = recalculate_invoice_financials(parsed)
        new_md = format_invoice_json(
            invoice_id=invoice_id,
            client_name=parsed["client_name"],
            client_tax_id=parsed["client_tax_id"],
            financials=fin,
            invoice_date=parsed["invoice_date"],
            due_date=parsed["due_date"],
            status=doc["status"],
        )

        updated_doc = db.update_document(invoice_id, content=new_md)
        return {
            "found": True,
            "invoice_id": invoice_id,
            "doc_id": invoice_id,
            "financials": fin,
            "document": updated_doc,
            "message": f"Article '{removed.get('name')}' supprimé de la facture {invoice_id}. Nouveau Total TTC : {fin['total_ttc']} TND.",
        }

    def validate_invoice(invoice_id: str, **kwargs: Any) -> dict[str, Any]:
        """Validates legal and accounting compliance of an invoice, transitioning status to 'approved'."""
        if not db:
            return {"found": False, "error": "Database not available"}

        doc = db.get_document(invoice_id)
        if not doc:
            return {"found": False, "error": f"Facture {invoice_id} introuvable."}

        parsed = parse_invoice_markdown(doc["content"])
        items = parsed.get("items", [])

        errors = []
        warnings = []

        if not parsed.get("client_name") or parsed["client_name"] == "Client Inconnu":
            errors.append("Le nom du client est obligatoire.")

        if not items:
            errors.append("La facture doit comporter au moins une ligne d'article ou prestation.")

        for it in items:
            if it.get("quantity", 0) <= 0:
                errors.append(f"Quantité invalide sur l'article '{it.get('name')}' ({it.get('quantity')}).")
            if it.get("unit_price", 0) < 0:
                errors.append(f"Prix unitaire négatif sur l'article '{it.get('name')}'.")

        if not parsed.get("client_tax_id"):
            warnings.append("Matricule Fiscal / Tax ID client manquant (recommandé pour les factures professionnelles).")

        if errors:
            return {
                "found": True,
                "valid": False,
                "invoice_id": invoice_id,
                "doc_id": invoice_id,
                "errors": errors,
                "warnings": warnings,
                "message": f"Validation échouée pour la facture {invoice_id} : " + "; ".join(errors),
            }

        # Compliance passed -> Update status to approved
        fin = recalculate_invoice_financials(parsed)
        new_md = format_invoice_json(
            invoice_id=invoice_id,
            client_name=parsed["client_name"],
            client_tax_id=parsed["client_tax_id"],
            financials=fin,
            invoice_date=parsed["invoice_date"],
            due_date=parsed["due_date"],
            status="approved",
        )

        updated_doc = db.update_document(invoice_id, content=new_md, status="approved")
        return {
            "found": True,
            "valid": True,
            "invoice_id": invoice_id,
            "doc_id": invoice_id,
            "status": "approved",
            "warnings": warnings,
            "financials": fin,
            "document": updated_doc,
            "message": f"Facture {invoice_id} vérifiée et validée avec succès ! Statut passé à 'approved' (Total TTC: {fin['total_ttc']} TND).",
        }

    def list_invoices(
        status: str | None = None,
        client_name: str | None = None,
        limit: int = 50,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Lists invoices with financial summaries, status, and client info."""
        if not db:
            return {"found": False, "error": "Database not available"}

        # Normalize 'all' / 'tous' / '' to None so no SQL status filter is applied
        _status_filter: str | None = status
        if not _status_filter or _status_filter.strip().lower() in ("all", "tous", "toutes", "*"):
            _status_filter = None

        docs = db.list_documents(limit=limit, doc_type="invoice", status=_status_filter)
        invoices = []
        for d in docs:
            parsed = parse_invoice_markdown(d.get("content", ""))
            c_name = parsed.get("client_name") or d.get("title", "")
            if client_name and client_name.lower() not in c_name.lower():
                continue
            fin = recalculate_invoice_financials(parsed)
            invoices.append({
                "invoice_id": d["doc_id"],
                "title": d["title"],
                "client_name": c_name,
                "status": d["status"],
                "total_ttc": fin["total_ttc"],
                "currency": fin["currency"],
                "item_count": fin["item_count"],
                "invoice_date": parsed.get("invoice_date"),
                "updated_at": d.get("updated_at"),
            })

        return {
            "found": True,
            "count": len(invoices),
            "invoices": invoices,
            "message": f"{len(invoices)} facture(s) trouvée(s).",
        }

    def delete_invoice(invoice_id: str, **kwargs: Any) -> dict[str, Any]:
        """Deletes an invoice from the system."""
        if not db:
            return {"found": False, "error": "Database not available"}

        doc = db.get_document(invoice_id)
        if not doc:
            return {"found": False, "error": f"Facture '{invoice_id}' introuvable."}

        deleted = db.delete_document(invoice_id)
        return {
            "found": True,
            "deleted": deleted,
            "invoice_id": invoice_id,
            "message": f"Facture '{invoice_id}' ({doc.get('title')}) supprimée avec succès.",
        }

    def duplicate_invoice(
        invoice_id: str,
        new_client_name: str | None = None,
        new_invoice_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Duplicates an existing invoice, optionally for a new client."""
        if not db:
            return {"found": False, "error": "Database not available"}

        doc = db.get_document(invoice_id)
        if not doc:
            return {"found": False, "error": f"Facture source '{invoice_id}' introuvable."}

        parsed = parse_invoice_markdown(doc["content"])
        target_inv_id = new_invoice_id if new_invoice_id and new_invoice_id.strip() else f"INV-{uuid.uuid4().hex[:6].upper()}"
        target_client = new_client_name or parsed.get("client_name", "Client Inconnu")

        fin = recalculate_invoice_financials(parsed)
        new_md = format_invoice_json(
            invoice_id=target_inv_id,
            client_name=target_client,
            client_tax_id=parsed.get("client_tax_id"),
            financials=fin,
            status="draft",
        )

        new_doc = db.create_document(
            doc_id=target_inv_id,
            title=f"Facture {target_inv_id} - {target_client}",
            doc_type="invoice",
            content=new_md,
            status="draft",
        )

        return {
            "found": True,
            "source_invoice_id": invoice_id,
            "invoice_id": target_inv_id,
            "doc_id": target_inv_id,
            "client_name": target_client,
            "financials": fin,
            "document": new_doc,
            "message": f"Facture {invoice_id} dupliquée vers {target_inv_id} pour {target_client} (Total TTC: {fin['total_ttc']} {fin['currency']}).",
        }

    def export_invoice_pdf(invoice_id: str, output_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Render the canonical invoice JSON to HTML and pipe that HTML to WeasyPrint."""
        if not db:
            return {"found": False, "error": "Database not available"}
        doc = db.get_document(invoice_id)
        if not doc:
            return {"found": False, "error": f"Facture '{invoice_id}' introuvable."}

        parsed = parse_invoice_markdown(doc["content"])
        stored_financials = parsed.get("financials") or {}
        fin = calculate_invoice_financials(
            parsed.get("items", []),
            tax_rate=float(stored_financials.get("tax_rate", 19.0)),
            global_discount_pct=float(stored_financials.get("global_discount_pct", 0.0)),
            timbre_fiscal=float(stored_financials.get("timbre_fiscal", 1.0)),
            currency=stored_financials.get("currency", "TND"),
        )

        export_dir = Path("data/exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        target_path = Path(output_path) if output_path else export_dir / f"{invoice_id}_Facture.pdf"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            payload = build_invoice_payload(
                invoice_id=invoice_id,
                client_name=parsed.get("client_name", "Client Inconnu"),
                client_tax_id=parsed.get("client_tax_id"),
                financials=fin,
                invoice_date=parsed.get("invoice_date"),
                due_date=parsed.get("due_date"),
                payment_terms=parsed.get("payment_terms", "30 jours fin de mois"),
                status=doc["status"],
            )
            render_and_export_pdf(payload, target_path)
            return {
                "found": True,
                "invoice_id": invoice_id,
                "pdf_path": str(target_path.resolve()),
                "filename": target_path.name,
                "download_url": f"/v1/exports/download/{target_path.name}",
                "financials": fin,
                "message": f"Facture '{invoice_id}' exportee avec succes en PDF.",
            }
        except RuntimeError as exc:
            return {"found": False, "error": str(exc)}
        except Exception as exc:
            return {"found": False, "error": f"Failed to export Invoice PDF: {exc}"}

    def _legacy_export_invoice_pdf(invoice_id: str, output_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Generates a professional Invoice PDF with table layout, tax breakdown, timbre fiscal, and totals."""
        if not db:
            return {"found": False, "error": "Database not available"}

        doc = db.get_document(invoice_id)
        if not doc:
            return {"found": False, "error": f"Facture '{invoice_id}' introuvable."}

        parsed = parse_invoice_markdown(doc["content"])
        fin = recalculate_invoice_financials(parsed)

        export_dir = Path("data/exports")
        export_dir.mkdir(parents=True, exist_ok=True)

        if not output_path:
            clean_filename = f"{invoice_id}_Facture.pdf"
            target_path = export_dir / clean_filename
        else:
            target_path = Path(output_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors

            pdf_doc = SimpleDocTemplate(
                str(target_path),
                pagesize=letter,
                rightMargin=36,
                leftMargin=36,
                topMargin=36,
                bottomMargin=36,
            )
            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                'InvTitle',
                parent=styles['Heading1'],
                fontSize=22,
                leading=26,
                textColor=colors.HexColor('#0f172a'),
                spaceAfter=6,
            )
            meta_style = ParagraphStyle(
                'InvMeta',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#475569'),
                spaceAfter=12,
            )
            cell_style = ParagraphStyle(
                'InvCell',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.HexColor('#1e293b'),
            )
            cell_bold = ParagraphStyle(
                'InvCellBold',
                parent=styles['Normal'],
                fontSize=9,
                leading=11,
                fontName="Helvetica-Bold",
                textColor=colors.HexColor('#0f172a'),
            )

            story = []
            story.append(Paragraph(f"<b>FACTURE N° {invoice_id}</b>", title_style))
            tax_id_str = parsed.get('client_tax_id') or "Non spécifié"
            inv_date = parsed.get('invoice_date') or datetime.now(timezone.utc).strftime("%d/%m/%Y")
            due_d = parsed.get('due_date') or "À réception"
            story.append(Paragraph(
                f"<b>Client:</b> {parsed.get('client_name')} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>MF:</b> {tax_id_str} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Date:</b> {inv_date} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Échéance:</b> {due_d} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Statut:</b> {doc['status'].upper()}",
                meta_style,
            ))
            story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563eb'), spaceAfter=15))

            # Table Header
            cur = fin.get("currency", "TND")
            table_data = [[
                Paragraph("<b>Réf</b>", cell_bold),
                Paragraph("<b>Désignation</b>", cell_bold),
                Paragraph("<b>Qté</b>", cell_bold),
                Paragraph(f"<b>P.U HT ({cur})</b>", cell_bold),
                Paragraph("<b>Remise</b>", cell_bold),
                Paragraph(f"<b>Total Net HT ({cur})</b>", cell_bold),
            ]]

            for item in fin["items"]:
                table_data.append([
                    Paragraph(str(item.get("product_id") or "-"), cell_style),
                    Paragraph(str(item.get("name")), cell_style),
                    Paragraph(f"{item['quantity']:g}", cell_style),
                    Paragraph(f"{item['unit_price']:.2f}", cell_style),
                    Paragraph(f"{item['discount_pct']:g}%" if item['discount_pct'] > 0 else "-", cell_style),
                    Paragraph(f"{item['line_net_ht']:.2f}", cell_style),
                ])

            if not fin["items"]:
                table_data.append([
                    Paragraph("-", cell_style),
                    Paragraph("<i>Aucun article</i>", cell_style),
                    Paragraph("-", cell_style),
                    Paragraph("-", cell_style),
                    Paragraph("-", cell_style),
                    Paragraph("0.00", cell_style),
                ])

            item_table = Table(table_data, colWidths=[65, 200, 45, 80, 50, 100])
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
                [Paragraph("<b>Total Brut HT:</b>", cell_style), Paragraph(f"{fin['total_brut_ht']:.2f} {cur}", cell_bold)],
                [Paragraph("<b>Total Remises:</b>", cell_style), Paragraph(f"-{fin['total_remises']:.2f} {cur}", cell_style)],
                [Paragraph("<b>Total Net HT:</b>", cell_style), Paragraph(f"{fin['total_net_ht']:.2f} {cur}", cell_bold)],
                [Paragraph(f"<b>TVA ({fin['tax_rate']:g}%):</b>", cell_style), Paragraph(f"{fin['total_tva']:.2f} {cur}", cell_style)],
            ]
            if fin.get("timbre_fiscal", 0) > 0:
                totals_data.append([Paragraph("<b>Timbre Fiscal:</b>", cell_style), Paragraph(f"{fin['timbre_fiscal']:.3f} {cur}", cell_style)])
            totals_data.append([Paragraph("<b>TOTAL TTC À PAYER:</b>", cell_bold), Paragraph(f"<b>{fin['total_ttc']:.2f} {cur}</b>", cell_bold)])

            totals_table = Table(totals_data, colWidths=[140, 110], hAlign='RIGHT')
            totals_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#94a3b8')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(totals_table)

            pdf_doc.build(story)

            return {
                "found": True,
                "invoice_id": invoice_id,
                "pdf_path": str(target_path.resolve()),
                "filename": target_path.name,
                "download_url": f"/v1/exports/download/{target_path.name}",
                "financials": fin,
                "message": f"Facture '{invoice_id}' exportée avec succès en PDF.",
            }
        except ImportError:
            return {"found": False, "error": "Export PDF requires reportlab (pip install reportlab)"}
        except Exception as exc:
            return {"found": False, "error": f"Failed to export Invoice PDF: {exc}"}

    return [
        SkillTool(
            name="create_invoice",
            description="Create a new invoice with client info, items, tax calculations, and Markdown formatting.",
            parameters={
                "client_name": "string",
                "client_tax_id": "string",
                "items": "array",
                "invoice_id": "string",
                "due_date": "string",
                "payment_terms": "string",
                "tax_rate": "number",
                "discount_pct": "number",
                "currency": "string",
            },
            handler=create_invoice,
            requires_confirmation=False,
        ),
        SkillTool(
            name="get_invoice_summary",
            description="Retrieve and calculate itemized breakdown, tax details, and totals for an invoice.",
            parameters={"invoice_id": "string"},
            handler=get_invoice_summary,
            requires_confirmation=False,
        ),
        SkillTool(
            name="list_invoices",
            description="List all invoices with status, client name, item count, and Total TTC summary.",
            parameters={
                "status": "string",
                "client_name": "string",
                "limit": "integer",
            },
            handler=list_invoices,
            requires_confirmation=False,
        ),
        SkillTool(
            name="add_invoice_item",
            description="Add a product or service line to an existing invoice.",
            parameters={
                "invoice_id": "string",
                "name": "string",
                "quantity": "number",
                "unit_price": "number",
                "discount_pct": "number",
                "product_id": "string",
            },
            handler=add_invoice_item,
            requires_confirmation=False,
        ),
        SkillTool(
            name="update_invoice_item",
            description="Update an existing line item in an invoice. Requires confirmation.",
            parameters={
                "invoice_id": "string",
                "line_number": "integer",
                "product_id": "string",
                "name": "string",
                "quantity": "number",
                "unit_price": "number",
                "discount_pct": "number",
            },
            handler=update_invoice_item,
            requires_confirmation=True,
        ),
        SkillTool(
            name="update_invoice",
            description="Update invoice-level details such as the client name, currency, or a line name. Requires confirmation.",
            parameters={
                "invoice_id": "string",
                "client_name": "string",
                "new_item_name": "string",
                "currency": "string",
            },
            handler=update_invoice,
            requires_confirmation=True,
        ),
        SkillTool(
            name="remove_invoice_item",
            description="Remove a line item from an invoice. Requires confirmation.",
            parameters={
                "invoice_id": "string",
                "line_number": "integer",
                "product_id": "string",
                "name": "string",
            },
            handler=remove_invoice_item,
            requires_confirmation=True,
        ),
        SkillTool(
            name="validate_invoice",
            description="Validate legal/accounting rules of an invoice and set its status to approved.",
            parameters={"invoice_id": "string"},
            handler=validate_invoice,
            requires_confirmation=False,
        ),
        SkillTool(
            name="duplicate_invoice",
            description="Duplicate an existing invoice for a new client or period.",
            parameters={
                "invoice_id": "string",
                "new_client_name": "string",
                "new_invoice_id": "string",
            },
            handler=duplicate_invoice,
            requires_confirmation=False,
        ),
        SkillTool(
            name="delete_invoice",
            description="Delete an invoice document. Requires confirmation.",
            parameters={"invoice_id": "string"},
            handler=delete_invoice,
            requires_confirmation=True,
        ),
        SkillTool(
            name="export_invoice_pdf",
            description="Export an invoice into a formatted PDF document with legal/tax layout and totals.",
            parameters={"invoice_id": "string", "output_path": "string"},
            handler=export_invoice_pdf,
            requires_confirmation=False,
        ),
    ]
