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
    document = _coerce_document(doc)
    html = render_document_html(document)
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as exc:
        return _fallback_pdf_bytes(document, exc)
    try:
        return HTML(string=html, base_url=str(TEMPLATE_DIR)).write_pdf()
    except (ImportError, OSError) as exc:
        return _fallback_pdf_bytes(document, exc)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = " ".join(data.split())
        if value:
            self.parts.append(value)


def _fallback_pdf_bytes(document: DocumentSchema | str, cause: Exception) -> bytes:
    """Keep local exports usable when GTK/Pango is absent on Windows.

    The old fallback flattened the rendered HTML into text paragraphs. That
    made invoices readable but discarded the metadata header, table structure,
    totals, and payment terms. Rebuild the important invoice layout directly
    with ReportLab when WeasyPrint is unavailable.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("PDF export requires WeasyPrint and its native runtime") from exc

    output = BytesIO()
    styles = getSampleStyleSheet()

    def money(value: Any, decimals: int = 2) -> str:
        return f"{float(value or 0):,.{decimals}f}".replace(",", " ")

    def text(value: Any, fallback: str = "-") -> str:
        return escape(str(value if value not in (None, "") else fallback))

    body_style = ParagraphStyle(
        "FallbackBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9, leading=12, textColor=colors.HexColor("#334155"), spaceAfter=6,
    )
    small_style = ParagraphStyle(
        "FallbackSmall", parent=body_style, fontSize=8, leading=10,
        textColor=colors.HexColor("#64748b"),
    )
    cell_style = ParagraphStyle("FallbackCell", parent=body_style, fontSize=8, leading=10)
    header_cell_style = ParagraphStyle(
        "FallbackHeaderCell", parent=cell_style, fontName="Helvetica-Bold",
        fontSize=7, leading=9, textColor=colors.HexColor("#1e3a8a"),
    )
    title_style = ParagraphStyle(
        "FallbackTitle", parent=body_style, fontName="Helvetica-Bold",
        fontSize=19, leading=22, textColor=colors.HexColor("#0f172a"), spaceAfter=0,
    )
    status_style = ParagraphStyle(
        "FallbackStatus", parent=body_style, fontName="Helvetica-Bold",
        fontSize=8, leading=10, alignment=1, textColor=colors.HexColor("#1d4ed8"),
    )
    total_style = ParagraphStyle(
        "FallbackTotal", parent=status_style, textColor=colors.white,
        fontSize=9, leading=11,
    )

    story: list[Any] = []
    if isinstance(document, DocumentSchema) and document.document_type == "invoice":
        client = document.client
        financials = document.financials
        currency = text(document.currency, "TND")

        header = Table([
            [
                Paragraph(
                    '<font color="#64748b" size="8"><b>DOCUMENT COMMERCIAL</b></font><br/>'
                    f'<font size="19"><b>FACTURE N. {text(document.document_id)}</b></font>',
                    title_style,
                ),
                Paragraph(text(document.status, "draft").upper(), status_style),
            ]
        ], colWidths=[140 * mm, 36 * mm])
        header.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#dbeafe")),
            ("BOX", (1, 0), (1, 0), 0.5, colors.HexColor("#93c5fd")),
            ("LEFTPADDING", (1, 0), (1, 0), 8),
            ("RIGHTPADDING", (1, 0), (1, 0), 8),
            ("TOPPADDING", (1, 0), (1, 0), 6),
            ("BOTTOMPADDING", (1, 0), (1, 0), 6),
        ]))
        story.extend([
            header, Spacer(1, 4),
            HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2563eb")),
            Spacer(1, 12),
        ])

        metadata = Table([
            [
                Paragraph(
                    f'<font color="#64748b" size="7"><b>CLIENT</b></font><br/>'
                    f'<b>{text(client.name)}</b><br/>'
                    f'<font color="#64748b">MF / Tax ID : {text(client.tax_id)}</font>',
                    cell_style,
                ),
                Paragraph(
                    f'<font color="#64748b" size="7"><b>DATES</b></font><br/>'
                    f'<b>Emission : {text(document.invoice_date)}</b><br/>'
                    f'<font color="#64748b">Echeance : {text(document.due_date)}</font>',
                    cell_style,
                ),
            ],
            [
                Paragraph(
                    f'<font color="#64748b" size="7"><b>CONDITIONS DE REGLEMENT</b></font><br/>'
                    f'<b>{text(document.payment_terms)}</b>',
                    cell_style,
                ),
                Paragraph(
                    f'<font color="#64748b" size="7"><b>DEVISE</b></font><br/><b>{currency}</b>',
                    cell_style,
                ),
            ],
        ], colWidths=[88 * mm, 88 * mm])
        metadata.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        section_style = ParagraphStyle(
            "FallbackSection", parent=body_style, fontName="Helvetica-Bold",
            fontSize=10, textColor=colors.HexColor("#0f172a"),
        )
        story.extend([metadata, Spacer(1, 18), Paragraph("DETAIL DES ARTICLES ET PRESTATIONS", section_style), Spacer(1, 6)])

        item_rows: list[list[Any]] = [[
            Paragraph("#", header_cell_style),
            Paragraph("REF / PRODUIT", header_cell_style),
            Paragraph("DESIGNATION", header_cell_style),
            Paragraph("QTE", header_cell_style),
            Paragraph(f"P.U. HT ({currency})", header_cell_style),
            Paragraph("REMISE", header_cell_style),
            Paragraph(f"TOTAL NET HT ({currency})", header_cell_style),
        ]]
        for item in document.items:
            quantity_decimals = 0 if float(item.quantity).is_integer() else 2
            item_rows.append([
                Paragraph(text(item.line), cell_style),
                Paragraph(text(item.product_id), cell_style),
                Paragraph(text(item.name), cell_style),
                Paragraph(money(item.quantity, quantity_decimals), cell_style),
                Paragraph(money(item.unit_price), cell_style),
                Paragraph(f"{money(item.discount_pct, 1)}%" if item.discount_pct else "-", cell_style),
                Paragraph(money(item.line_net_ht), cell_style),
            ])
        if not document.items:
            item_rows.append([Paragraph("Aucun article specifie", cell_style)] + [""] * 6)

        items_table = Table(
            item_rows,
            colWidths=[9 * mm, 23 * mm, 54 * mm, 13 * mm, 27 * mm, 20 * mm, 30 * mm],
            repeatRows=1,
        )
        items_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eff6ff")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (3, 1), (-1, -1), "RIGHT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([items_table, Spacer(1, 16)])

        summary_rows = [
            [Paragraph("Total brut HT", cell_style), Paragraph(f"{money(financials.total_brut_ht)} {currency}", cell_style)],
            [Paragraph("Total remises", cell_style), Paragraph(f"-{money(financials.total_remises)} {currency}", cell_style)],
            [Paragraph("Total net HT", cell_style), Paragraph(f"{money(financials.total_net_ht)} {currency}", cell_style)],
            [Paragraph(f"TVA ({money(financials.tax_rate, 1)}%)", cell_style), Paragraph(f"{money(financials.total_tva)} {currency}", cell_style)],
        ]
        if financials.timbre_fiscal:
            summary_rows.append([Paragraph("Timbre fiscal", cell_style), Paragraph(f"{money(financials.timbre_fiscal, 3)} {currency}", cell_style)])
        summary_rows.append([Paragraph("TOTAL TTC A PAYER", total_style), Paragraph(f"{money(financials.total_ttc)} {currency}", total_style)])
        summary = Table(summary_rows, colWidths=[58 * mm, 38 * mm], hAlign="RIGHT")
        summary.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#1d4ed8")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([
            summary, Spacer(1, 18),
            Paragraph(
                f"Conditions de reglement : {text(document.payment_terms)} - Document genere par WIND ERP Assistant Agentique",
                small_style,
            ),
        ])
    else:
        story.extend([
            Paragraph(text(document.title if isinstance(document, DocumentSchema) else "Document"), title_style),
            Spacer(1, 8),
        ])
        parser = _TextExtractor()
        parser.feed(document.content if isinstance(document, DocumentSchema) else document)
        for value in parser.parts:
            story.extend([Paragraph(escape(value), body_style), Spacer(1, 5)])

    def draw_page_number(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(
        output, pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=15 * mm, bottomMargin=18 * mm,
    ).build(story, onFirstPage=draw_page_number, onLaterPages=draw_page_number)
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
