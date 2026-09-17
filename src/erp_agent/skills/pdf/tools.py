from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from erp_agent.document_rendering import parse_json_content, render_and_export_pdf
from erp_agent.skills.loader import SkillTool


def register_tools(db: Any = None, skill_dir: Path | None = None) -> list[SkillTool]:
    """Registers and returns all PDF tools for the agent."""
    
    def read_pdf(file_path: str, max_pages: int = 10, **kwargs: Any) -> dict[str, Any]:
        """Extracts text and metadata from a PDF file."""
        p = Path(file_path)
        if not p.is_file():
            return {"found": False, "error": f"File not found: {file_path}"}

        text_content: list[str] = []
        page_count = 0
        metadata: dict[str, Any] = {}

        try:
            # 1. Try PyMuPDF (fitz)
            import fitz
            doc = fitz.open(str(p))
            page_count = len(doc)
            metadata = dict(doc.metadata or {})
            for idx in range(min(page_count, max_pages)):
                page = doc[idx]
                text_content.append(page.get_text("text").strip())
            doc.close()
        except (ImportError, Exception):
            try:
                # 2. Try pdfplumber
                import pdfplumber
                with pdfplumber.open(str(p)) as pdf:
                    page_count = len(pdf.pages)
                    metadata = dict(pdf.metadata or {})
                    for idx in range(min(page_count, max_pages)):
                        page_text = pdf.pages[idx].extract_text() or ""
                        text_content.append(page_text.strip())
            except (ImportError, Exception):
                try:
                    # 3. Fallback to pypdf
                    from pypdf import PdfReader
                    reader = PdfReader(str(p))
                    page_count = len(reader.pages)
                    metadata = dict(reader.metadata or {})
                    for idx in range(min(page_count, max_pages)):
                        page_text = reader.pages[idx].extract_text() or ""
                        text_content.append(page_text.strip())
                except Exception as exc:
                    return {"found": False, "error": f"Failed to read PDF with fitz/pdfplumber/pypdf: {exc}"}

        full_text = "\n\n--- Page Break ---\n\n".join(text_content)
        return {
            "found": True,
            "file_path": str(p),
            "total_pages": page_count,
            "pages_read": min(page_count, max_pages),
            "metadata": metadata,
            "text": full_text[:4000],  # Return up to 4000 chars for context
        }

    def extract_pdf_tables(file_path: str, page_number: int | None = None, **kwargs: Any) -> dict[str, Any]:
        """Extracts tables from a PDF using pdfplumber."""
        p = Path(file_path)
        if not p.is_file():
            return {"found": False, "error": f"File not found: {file_path}"}

        try:
            import pdfplumber
            extracted_tables: list[dict[str, Any]] = []
            with pdfplumber.open(str(p)) as pdf:
                pages_to_check = [pdf.pages[page_number - 1]] if page_number and 1 <= page_number <= len(pdf.pages) else pdf.pages
                for idx, page in enumerate(pages_to_check):
                    tables = page.extract_tables()
                    for t_idx, table in enumerate(tables):
                        extracted_tables.append({
                            "page": (page_number if page_number else idx + 1),
                            "table_index": t_idx + 1,
                            "rows": table,
                        })
            return {
                "found": True,
                "file_path": str(p),
                "table_count": len(extracted_tables),
                "tables": extracted_tables,
            }
        except ImportError:
            return {"found": False, "error": "Table extraction requires pdfplumber library (pip install pdfplumber)"}
        except Exception as exc:
            return {"found": False, "error": f"Table extraction failed: {exc}"}

    def export_document_to_pdf(doc_id: str, output_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Exports an ERP document to a styled PDF using ReportLab."""
        if not db:
            return {"found": False, "error": "Database not configured"}

        doc = db.get_document(doc_id)
        if not doc:
            return {"found": False, "error": f"Document '{doc_id}' not found"}

        export_dir = Path("data/exports")
        export_dir.mkdir(parents=True, exist_ok=True)

        if not output_path or output_path.startswith("/") or output_path.startswith("\\"):
            clean_filename = Path(output_path).name if output_path else f"{doc_id}_{doc['title'].replace(' ', '_')}.pdf"
            target_path = export_dir / clean_filename
        else:
            target_path = Path(output_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = parse_json_content(doc["content"])
        if payload and payload.get("document_type") in {"invoice", "bon_de_commande", "purchase_order"}:
            try:
                render_and_export_pdf(payload, target_path)
                return {
                    "found": True,
                    "doc_id": doc_id,
                    "title": doc["title"],
                    "pdf_path": str(target_path.resolve()),
                    "filename": target_path.name,
                    "download_url": f"/v1/exports/download/{target_path.name}",
                    "message": f"Document '{doc_id}' exporte avec succes en PDF.",
                }
            except RuntimeError as exc:
                return {"found": False, "error": str(exc)}
            except Exception as exc:
                return {"found": False, "error": f"Failed to generate PDF: {exc}"}

        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors

            pdf = SimpleDocTemplate(str(target_path), pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle(
                'DocTitle',
                parent=styles['Heading1'],
                fontSize=20,
                leading=24,
                textColor=colors.HexColor('#1e293b'),
                spaceAfter=10
            )
            meta_style = ParagraphStyle(
                'DocMeta',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#64748b'),
                spaceAfter=15
            )
            body_style = ParagraphStyle(
                'DocBody',
                parent=styles['Normal'],
                fontSize=11,
                leading=16,
                textColor=colors.HexColor('#334155'),
                spaceAfter=12
            )

            story = []
            story.append(Paragraph(f"<b>{doc['title']}</b>", title_style))
            story.append(Paragraph(f"Document ID: {doc['doc_id']} | Type: {doc['doc_type']} | Status: {doc['status']}", meta_style))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceAfter=15))

            for para in doc['content'].split("\n\n"):
                clean_p = para.replace("\n", "<br/>")
                if clean_p.strip():
                    story.append(Paragraph(clean_p, body_style))
                    story.append(Spacer(1, 8))

            pdf.build(story)

            return {
                "found": True,
                "doc_id": doc_id,
                "title": doc["title"],
                "pdf_path": str(target_path.resolve()),
                "filename": target_path.name,
                "download_url": f"/v1/exports/download/{target_path.name}",
                "message": f"Document '{doc_id}' exporté avec succès en PDF.",
            }
        except ImportError:
            return {"found": False, "error": "PDF export requires reportlab library (pip install reportlab)"}
        except Exception as exc:
            return {"found": False, "error": f"Failed to generate PDF: {exc}"}

    def merge_pdfs(file_paths: list[str], output_path: str, **kwargs: Any) -> dict[str, Any]:
        """Merges multiple PDF files into one output PDF."""
        for fp in file_paths:
            p = Path(fp)
            if not p.is_file():
                return {"found": False, "error": f"File not found: {fp}"}

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Try fitz (PyMuPDF)
            import fitz
            merged = fitz.open()
            for fp in file_paths:
                with fitz.open(fp) as src:
                    merged.insert_pdf(src)
            merged.save(str(out))
            merged.close()
            return {
                "found": True,
                "output_path": str(out.resolve()),
                "files_merged": file_paths,
                "message": f"Successfully merged {len(file_paths)} PDFs into {out}",
            }
        except (ImportError, Exception):
            try:
                # 2. Fallback to pypdf
                from pypdf import PdfWriter, PdfReader
                writer = PdfWriter()
                for fp in file_paths:
                    reader = PdfReader(fp)
                    for page in reader.pages:
                        writer.add_page(page)
                with open(out, "wb") as f_out:
                    writer.write(f_out)
                return {
                    "found": True,
                    "output_path": str(out.resolve()),
                    "files_merged": file_paths,
                    "message": f"Successfully merged {len(file_paths)} PDFs into {out}",
                }
            except Exception as exc:
                return {"found": False, "error": f"Failed to merge PDFs with fitz or pypdf: {exc}"}

    return [
        SkillTool(
            name="read_pdf",
            description="Extract and read text and metadata from a PDF file path.",
            parameters={"file_path": "string", "max_pages": "integer"},
            handler=read_pdf,
            requires_confirmation=False,
        ),
        SkillTool(
            name="extract_pdf_tables",
            description="Extract tables and structured data from a PDF file.",
            parameters={"file_path": "string", "page_number": "integer"},
            handler=extract_pdf_tables,
            requires_confirmation=False,
        ),
        SkillTool(
            name="export_document_to_pdf",
        description="Export an ERP document to PDF; commercial JSON documents use their Jinja2/WeasyPrint template.",
            parameters={"doc_id": "string", "output_path": "string"},
            handler=export_document_to_pdf,
            requires_confirmation=True,
        ),
        SkillTool(
            name="merge_pdfs",
            description="Merge multiple PDF files into a single destination PDF file.",
            parameters={"file_paths": "array", "output_path": "string"},
            handler=merge_pdfs,
            requires_confirmation=True,
        ),
    ]
