from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx


class InvoiceExtractionClient:
    """HTTP adapter for the WIND Invoice Extraction FastAPI service."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 180.0,
        api_token: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_token = api_token

    def extract(
        self,
        file_path: str,
        tenant_id: str,
        invoice_layout: str,
        document_id: str | None = None,
        force_langue: str | None = None,
        debug: bool = False,
    ) -> dict[str, Any]:
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            return {"found": False, "error": f"Invoice file not found: {path}"}

        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        data: dict[str, str] = {
            "tenant_id": tenant_id,
            "invoice_layout": invoice_layout,
            "debug": str(debug).lower(),
        }
        if document_id:
            data["document_id"] = document_id
        if force_langue:
            data["force_langue"] = force_langue

        try:
            with path.open("rb") as invoice_file:
                response = httpx.post(
                    f"{self.base_url}/v1/invoices/extract",
                    files={
                        "file": (
                            path.name,
                            invoice_file,
                            _content_type(path),
                        )
                    },
                    data=data,
                    headers=headers,
                    timeout=self.timeout,
                )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail: Any
            try:
                detail = exc.response.json()
            except ValueError:
                detail = exc.response.text
            return {
                "found": False,
                "error": f"Invoice extraction service returned HTTP {exc.response.status_code}.",
                "details": detail,
            }
        except (httpx.HTTPError, OSError, ValueError) as exc:
            return {
                "found": False,
                "error": f"Invoice extraction service unavailable: {exc}",
            }

        extracted = payload.get("result") or {}
        normalized = normalize_extraction(extracted)
        return {
            "found": True,
            "source": "wind-invoice-extraction",
            "request_id": payload.get("request_id"),
            "document_id": payload.get("document_id") or document_id,
            "model_used": payload.get("model_used"),
            "warnings": payload.get("warnings") or [],
            "validation_errors": payload.get("validation_errors") or [],
            "extraction": extracted,
            "invoice_data": normalized,
            "message": "Invoice extracted successfully.",
        }


def normalize_extraction(extracted: dict[str, Any]) -> dict[str, Any]:
    """Map extractor schema fields to the ERP invoice tool schema."""
    items: list[dict[str, Any]] = []
    for line in extracted.get("lineModels") or []:
        if not isinstance(line, dict):
            continue

        quantity = float(line.get("quantite") or 1)
        unit_price = float(line.get("prixunitaire") or 0)
        discount_amount = float(line.get("remise") or 0)
        gross = quantity * unit_price
        discount_pct = (discount_amount / gross * 100) if gross else 0.0

        items.append({
            "product_id": line.get("reference"),
            "name": line.get("productname") or "Article",
            "quantity": quantity,
            "unit_price": unit_price,
            "discount_pct": round(discount_pct, 4),
            "source_discount_amount": discount_amount,
            "tva_amount": float(line.get("tvaAmount") or 0),
            "unit": line.get("unit"),
            "is_service": bool(line.get("isService", False)),
        })

    timbre_fiscal = float(extracted.get("timbre") or 0)
    fodec = float(extracted.get("fodec") or 0)
    derived_brut_ht = round(sum(item["quantity"] * item["unit_price"] for item in items), 3)
    derived_remises = round(sum(item["source_discount_amount"] for item in items), 3)
    derived_net_ht = round(derived_brut_ht - derived_remises, 3)
    derived_tva = round(sum(item["tva_amount"] for item in items), 3)
    derived_ttc = round(derived_net_ht + derived_tva + timbre_fiscal + fodec, 3)

    # The current extractor schema is line-based and normally does not expose
    # invoice totals. Preserve reported totals when a newer extractor version
    # provides them; otherwise derive them from the extracted lines.
    reported_financials = extracted.get("financials")
    if not isinstance(reported_financials, dict):
        reported_financials = extracted.get("totals")
    if not isinstance(reported_financials, dict):
        reported_financials = {}

    def reported_number(*keys: str) -> float | None:
        for key in keys:
            value = extracted.get(key)
            if value is None:
                value = reported_financials.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return None

    total_brut_ht = reported_number("total_brut_ht", "totalBrutHt", "totalBrutHT", "subtotal") or derived_brut_ht
    total_remises = reported_number("total_remises", "totalRemises", "totalDiscount", "totalDiscounts") or derived_remises
    total_net_ht = reported_number("total_net_ht", "totalNetHt", "totalNetHT", "netTotal") or derived_net_ht
    total_tva = reported_number("total_tva", "totalTva", "totalTVA", "taxAmount") or derived_tva
    total_ttc = reported_number("total_ttc", "totalTtc", "totalTTC", "grandTotal", "totalAmount", "amountDue") or round(
        total_net_ht + total_tva + timbre_fiscal + fodec,
        3,
    )
    tax_rate = reported_number("tax_rate", "taxRate", "tvaRate", "tvaPercent")

    financials = {
        "total_brut_ht": round(total_brut_ht, 3),
        "total_remises": round(total_remises, 3),
        "total_net_ht": round(total_net_ht, 3),
        "total_tva": round(total_tva, 3),
        "timbre_fiscal": timbre_fiscal,
        "fodec": fodec,
        "total_ttc": round(total_ttc, 3),
        "tax_rate": tax_rate,
        "currency": extracted.get("devise") or "TND",
        "items": items,
    }

    return {
        "invoice_id": extracted.get("invoiceNumber"),
        "invoice_date": extracted.get("invoiceDate"),
        "client_name": extracted.get("customerName") or extracted.get("partnerName") or "Client Inconnu",
        "supplier_name": extracted.get("partnerName"),
        "currency": extracted.get("devise") or "TND",
        "timbre_fiscal": timbre_fiscal,
        "fodec": fodec,
        "items": items,
        "financials": financials,
    }


def _content_type(path: Path) -> str:
    return {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")
