"""Canonical document contract and server-side HTML/PDF rendering."""

from __future__ import annotations

import json
from html import escape
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .schemas import DocumentSchema

TEMPLATE_DIR = Path(__file__).with_name("templates")


def _environment() -> Environment:
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True, undefined=StrictUndefined)
    env.filters.update({
        "money": lambda value: f"{float(value or 0):,.2f}".replace(",", " "),
        "money3": lambda value: f"{float(value or 0):,.3f}".replace(",", " "),
        "number": lambda value: f"{float(value or 0):g}",
    })
    return env


def _coerce_document(doc: DocumentSchema | Mapping[str, Any]) -> DocumentSchema:
    return doc if isinstance(doc, DocumentSchema) else DocumentSchema.model_validate(doc)


def _document_from_record(record: Mapping[str, Any]) -> DocumentSchema:
    """Normalize one persisted document row without involving an LLM."""
    content = str(record.get("content") or "")
    payload = parse_json_content(content)
    if payload and payload.get("document_type") in {"invoice", "bon_de_commande", "purchase_order"}:
        return DocumentSchema.model_validate({
            **payload,
            "document_id": payload.get("document_id") or record.get("doc_id"),
            "title": record.get("title") or payload.get("title") or record.get("doc_id"),
            "status": record.get("status") or payload.get("status", "draft"),
            "content": content,
        })

    if str(record.get("doc_type") or "") == "invoice":
        from .skills.invoice.tools import build_invoice_payload, calculate_invoice_financials, parse_invoice_markdown
        parsed = parse_invoice_markdown(content)
        financials = calculate_invoice_financials(parsed.get("items", []))
        return DocumentSchema.model_validate({
            **build_invoice_payload(
                invoice_id=str(record.get("doc_id")),
                client_name=parsed.get("client_name", record.get("title", "")),
                client_tax_id=parsed.get("client_tax_id"),
                financials=financials,
                invoice_date=parsed.get("invoice_date"),
                due_date=parsed.get("due_date"),
                payment_terms=parsed.get("payment_terms", ""),
                status=str(record.get("status") or "draft"),
            ),
            "title": record.get("title") or str(record.get("doc_id")),
            "content": content,
        })

    return DocumentSchema(
        document_id=str(record.get("doc_id")),
        title=str(record.get("title") or record.get("doc_id")),
        document_type=str(record.get("doc_type") or "general"),
        status=str(record.get("status") or "draft"),
        content=content,
    )


def _default_database() -> Any:
    from .config import settings
    from .db import Database
    return Database(settings.database_path)


def generate_document(entity_id: str) -> DocumentSchema:
    """Build a typed document from the session store/database.

    Commercial documents are read from structured JSON or an order row.
    Product lookup and arithmetic are deterministic backend operations; the
    LLM is not called and no table markup is generated here.
    """
    if not entity_id or not entity_id.strip():
        raise ValueError("entity_id is required")
    db = _default_database()
    return document_from_database(db, entity_id)


def document_from_database(db: Any, entity_id: str) -> DocumentSchema:
    """Normalize a document using an explicitly injected database."""
    record = db.get_document(entity_id)
    if record:
        return _document_from_record(record)
    order = db.get_order(entity_id)
    if order:
        from .skills.bon_de_commande.tools import build_bon_de_commande_payload
        return DocumentSchema.model_validate(build_bon_de_commande_payload(order))
    raise KeyError(f"Document '{entity_id}' not found")


def render_document_html(doc: DocumentSchema) -> str:
    """Render a validated document with its server-side Jinja2 template."""
    document = _coerce_document(doc)
    template_name = {
        "invoice": "invoice.html.j2",
        "bon_de_commande": "bon_de_commande.html.j2",
        "purchase_order": "bon_de_commande.html.j2",
    }.get(document.document_type, "generic_document.html.j2")
    return _environment().get_template(template_name).render(document=document.model_dump(mode="json"))


def export_document_pdf(doc: DocumentSchema) -> bytes:
    """Render a document and return raw PDF bytes from WeasyPrint.

    The ReportLab branch is only a local Windows fallback for machines where
    WeasyPrint's optional native text libraries are not installed. Production
    export always uses WeasyPrint and never a headless browser.
    """
    html = render_document_html(_coerce_document(doc))
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as exc:
        return _fallback_pdf_bytes(html, exc)
    try:
        return HTML(string=html, base_url=str(TEMPLATE_DIR)).write_pdf()
    except (ImportError, OSError) as exc:
        return _fallback_pdf_bytes(html, exc)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = " ".join(data.split())
        if value:
            self.parts.append(value)


def _fallback_pdf_bytes(html: str, cause: Exception) -> bytes:
    """Keep local exports usable when GTK/Pango is absent on Windows."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise RuntimeError("PDF export requires WeasyPrint and its native runtime") from exc
    parser = _TextExtractor()
    parser.feed(html)
    output = BytesIO()
    styles = getSampleStyleSheet()
    story = []
    for value in parser.parts:
        story.append(Paragraph(escape(value), styles["BodyText"]))
        story.append(Spacer(1, 5))
    SimpleDocTemplate(output, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36).build(story)
    return output.getvalue()


def parse_json_content(content: str) -> dict[str, Any] | None:
    """Return JSON document content when content is a JSON object."""
    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def export_html_to_pdf(html: str, output_path: str | Path) -> None:
    """Compatibility helper for existing skill tools."""
    try:
        from weasyprint import HTML
        Path(output_path).write_bytes(HTML(string=html, base_url=str(TEMPLATE_DIR)).write_pdf())
    except (ImportError, OSError) as exc:
        Path(output_path).write_bytes(_fallback_pdf_bytes(html, exc))


def render_and_export_pdf(document: DocumentSchema | Mapping[str, Any], output_path: str | Path) -> str:
    """Compatibility wrapper for existing skill tools."""
    typed_document = _coerce_document(document)
    html = render_document_html(typed_document)
    Path(output_path).write_bytes(export_document_pdf(typed_document))
    return html
