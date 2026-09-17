from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .agent import AgentRuntime
from .asr import VoskTunisianASR
from .config import settings
from .input import NoOpTranscriptionNormalizer, OllamaTranscriptionNormalizer
from .document_rendering import document_from_database, export_document_pdf
from .schemas import ApprovalResponse, ChatRequest, ChatResponse, DocumentClient, DocumentLineItem


def _extract_text_from_upload(filename: str, file_bytes: bytes) -> str:
    """Extracts text content from uploaded files, parsing PDFs with pypdf."""
    if filename.lower().endswith(".pdf"):
        try:
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            pages_text = [p.extract_text() or "" for p in reader.pages]
            full_text = "\n\n--- Page Break ---\n\n".join(t.strip() for t in pages_text if t.strip())
            return full_text or f"[PDF {filename} without readable text layer]"
        except Exception:
            pass
    return file_bytes.decode("utf-8", errors="replace")


app = FastAPI(title="ERP Agentic Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

runtime = AgentRuntime(settings)
vosk_model_dir = settings.resolved_vosk_model_dir()
asr = VoskTunisianASR(str(vosk_model_dir) if vosk_model_dir else None)
if settings.transcription_normalizer == "ollama":
    transcription_normalizer = OllamaTranscriptionNormalizer(
        settings.ollama_base_url,
        settings.ollama_model,
        settings.transcription_normalizer_timeout,
    )
else:
    transcription_normalizer = NoOpTranscriptionNormalizer()


@app.get("/health")
def health() -> dict[str, str]:
    model_configured = bool(vosk_model_dir and vosk_model_dir.is_dir())
    return {
        "status": "ok",
        "planner": settings.planner,
        "asr": "configured" if model_configured else "not_configured",
        "transcription_normalizer": settings.transcription_normalizer,
    }


@app.post("/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict:
    return runtime.run(request.session_id, request.user_id, request.text, doc_id=request.doc_id)


@app.post("/v1/chat/upload", response_model=ChatResponse)
async def chat_with_document_upload(
    file: UploadFile = File(...),
    text: str = Form(default=""),
    session_id: str = Form(...),
    user_id: str = Form(default="anonymous"),
    title: str | None = Form(default=None),
    doc_type: str = Form(default="general"),
) -> dict:
    """Uploads a document (PDF/text/markdown), saves it to the database, and processes prompt instructions on it."""
    file_bytes = await file.read()
    filename = file.filename or "Uploaded Document"
    content = _extract_text_from_upload(filename, file_bytes)
    doc_title = title or filename

    # 1. Save document to DB
    saved_doc = runtime.db.create_document(
        doc_id=None,
        title=doc_title,
        doc_type=doc_type,
        content=content,
        status="draft",
    )
    new_doc_id = saved_doc["doc_id"]

    # 2. If prompt instruction given, execute agent against this document
    prompt_text = text.strip()
    if prompt_text:
        agent_prompt = f"In document {new_doc_id} ('{doc_title}'): {prompt_text}"
        res = runtime.run(session_id, user_id, agent_prompt, doc_id=new_doc_id)
        res["doc_id"] = new_doc_id
        res["document"] = runtime.db.get_document(new_doc_id) or saved_doc
        return res

    # 3. If no extra prompt given, return document confirmation for frontend editor
    reply_msg = f"Document '{doc_title}' ({new_doc_id}) téléchargé et enregistré avec succès. Vous pouvez maintenant le modifier dans l'éditeur."
    runtime.db.add_message(session_id, user_id, "assistant", reply_msg)
    return {
        "session_id": session_id,
        "reply": reply_msg,
        "status": "completed",
        "approval_token": None,
        "trace": [{"type": "upload", "doc_id": new_doc_id, "title": doc_title}],
        "doc_id": new_doc_id,
        "document": saved_doc,
    }


@app.post("/v1/voice/chat", response_model=ChatResponse)
async def voice_chat(session_id: str, user_id: str = "anonymous", audio: UploadFile = File(...)) -> dict:
    suffix = Path(audio.filename or ".wav").suffix or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with open(path, "wb") as output:
            output.write(await audio.read())
        transcript = asr.transcribe(path)
        if not transcript:
            raise HTTPException(status_code=422, detail="ASR returned an empty transcript")
        normalized_transcript = transcription_normalizer.normalize(transcript)
        runtime.db.add_event(
            session_id,
            "transcription",
            {"raw": transcript, "normalized": normalized_transcript},
        )
        return runtime.run(session_id, user_id, normalized_transcript)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        Path(path).unlink(missing_ok=True)


@app.post("/v1/approvals/{token}", response_model=ApprovalResponse)
def approve(token: str) -> dict:
    return runtime.approve(token)


@app.get("/v1/sessions/{session_id}/events")
def events(session_id: str) -> list[dict]:
    return runtime.db.events(session_id)


# ---------------------------------------------------------------------------
# Document endpoints
# ---------------------------------------------------------------------------

class DocumentCreateRequest(BaseModel):
    title: str
    doc_type: str = "general"
    content: str
    doc_id: str | None = None
    status: str = "draft"


class DocumentUpdateRequest(BaseModel):
    content: str | None = None
    title: str | None = None
    status: str | None = None
    items: list[DocumentLineItem] | None = None
    client: DocumentClient | None = None
    currency: str | None = None


class SmartEditRequest(BaseModel):
    instruction: str
    apply_immediately: bool = False


def _document_contract(doc_id: str) -> dict[str, Any]:
    """Return the JSON document contract plus persistence metadata."""
    record = runtime.db.get_document(doc_id)
    if record:
        document = document_from_database(runtime.db, doc_id)
        response = document.model_dump(mode="json")
        response.update({"doc_id": doc_id, "doc_type": document.document_type, "created_at": record.get("created_at"), "updated_at": record.get("updated_at"), "content": record.get("content", "")})
        return response
    order = runtime.db.get_order(doc_id)
    if order:
        document = document_from_database(runtime.db, doc_id)
        response = document.model_dump(mode="json")
        response.update({"doc_id": doc_id, "doc_type": document.document_type, "created_at": order.get("created_at"), "updated_at": order.get("created_at"), "content": None})
        return response
    raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")


@app.get("/v1/documents")
def list_documents(
    q: str | None = Query(default=None, description="Optional search query"),
) -> list[dict[str, Any]]:
    """List all documents, or search by passing ?q=<term>."""
    if q:
        return runtime.db.search_documents(q)
    return runtime.db.list_documents()


@app.post("/v1/documents", status_code=201)
def create_document(body: DocumentCreateRequest) -> dict[str, Any]:
    """Create a new document."""
    return runtime.db.create_document(
        doc_id=body.doc_id,
        title=body.title,
        doc_type=body.doc_type,
        content=body.content,
        status=body.status,
    )


@app.post("/v1/documents/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Query(default=None),
    doc_type: str = Query(default="general"),
) -> dict[str, Any]:
    """Upload a raw text/markdown/csv/pdf file as a real ERP document in the database."""
    file_bytes = await file.read()
    filename = file.filename or "Uploaded Document"
    content = _extract_text_from_upload(filename, file_bytes)
    doc_title = title or filename
    return runtime.db.create_document(
        doc_id=None,
        title=doc_title,
        doc_type=doc_type,
        content=content,
        status="draft",
    )


@app.get("/v1/documents/{doc_id}")
def get_document(doc_id: str) -> dict[str, Any]:
    """Fetch a single document by ID."""
    doc = runtime.db.get_document(doc_id)
    if not doc and not runtime.db.get_order(doc_id):
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return _document_contract(doc_id)


@app.get("/documents/{doc_id}")
def get_document_contract(doc_id: str) -> dict[str, Any]:
    return _document_contract(doc_id)


def _update_structured_document(doc_id: str, body: DocumentUpdateRequest) -> dict[str, Any] | None:
    """Persist line items and rebuild server-owned financial totals."""
    current = document_from_database(runtime.db, doc_id)
    record = runtime.db.get_document(doc_id)
    order = runtime.db.get_order(doc_id)
    items = [item.model_dump(mode="json") for item in (body.items if body.items is not None else current.items)]
    client = body.client or current.client
    currency = body.currency or current.currency
    tax_rate = float(current.financials.tax_rate)
    global_discount = float(current.financials.global_discount_pct)

    if order:
        runtime.db.update_order(doc_id, items=items, status=body.status, customer_id=client.name, currency=currency)
        return _document_contract(doc_id)

    if not record:
        return None
    if current.document_type == "invoice":
        from .skills.invoice.tools import build_invoice_payload, calculate_invoice_financials
        financials = calculate_invoice_financials(items, tax_rate=tax_rate, global_discount_pct=global_discount, timbre_fiscal=float(current.financials.timbre_fiscal or 1.0), currency=currency)
        payload = build_invoice_payload(invoice_id=doc_id, client_name=client.name, client_tax_id=client.tax_id, financials=financials, invoice_date=current.invoice_date, due_date=current.due_date, payment_terms=current.payment_terms, status=body.status or current.status)
    else:
        from .skills.bon_de_commande.tools import calculate_financials
        financials = calculate_financials(items, tax_rate=tax_rate, global_discount_pct=global_discount, currency=currency)
        payload = {**current.model_dump(mode="json"), "document_id": doc_id, "title": body.title or current.title, "status": body.status or current.status, "client": client.model_dump(mode="json"), "currency": currency, "items": financials["items"], "financials": financials}
    import json
    runtime.db.update_document(doc_id, content=json.dumps(payload, ensure_ascii=False, indent=2), title=body.title, status=body.status)
    return _document_contract(doc_id)


@app.patch("/documents/{doc_id}")
@app.patch("/v1/documents/{doc_id}")
def update_document(doc_id: str, body: DocumentUpdateRequest) -> dict[str, Any]:
    """Directly update a document (bypasses agent approval — use for editor saves)."""
    if body.items is not None or body.client is not None or body.currency is not None:
        try:
            structured = _update_structured_document(doc_id, body)
        except KeyError:
            structured = None
        if structured:
            return structured

    updated = runtime.db.update_document(
        doc_id,
        content=body.content,
        title=body.title,
        status=body.status,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return _document_contract(doc_id)


@app.get("/documents/{doc_id}/pdf")
@app.get("/v1/documents/{doc_id}/pdf")
def export_document_pdf_endpoint(doc_id: str) -> Response:
    """Generate a PDF on the server and return raw PDF bytes."""
    try:
        document = document_from_database(runtime.db, doc_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        pdf_bytes = export_document_pdf(document)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in document.document_id)
    return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{safe_name}.pdf"'})


@app.delete("/v1/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, Any]:
    """Delete a document by ID."""
    deleted = runtime.db.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"deleted": True, "doc_id": doc_id}


@app.post("/v1/documents/{doc_id}/smart-edit")
def smart_edit_document(doc_id: str, body: SmartEditRequest) -> dict[str, Any]:
    """AI smart-edit for editor: analyzes document and transforms content based on natural language instructions."""
    doc = runtime.db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    edit_res = runtime.editor.smart_edit(
        doc_id=doc_id,
        title=doc["title"],
        doc_type=doc["doc_type"],
        current_content=doc["content"],
        instruction=body.instruction,
    )

    if body.apply_immediately:
        updated = runtime.db.update_document(doc_id, content=edit_res["edited_content"])
        return {
            "doc_id": doc_id,
            "original_content": doc["content"],
            "edited_content": edit_res["edited_content"],
            "explanation": edit_res["explanation"],
            "diff_summary": edit_res["diff_summary"],
            "applied": True,
            "document": updated,
        }

    return {
        "doc_id": doc_id,
        "original_content": doc["content"],
        "edited_content": edit_res["edited_content"],
        "explanation": edit_res["explanation"],
        "diff_summary": edit_res["diff_summary"],
        "applied": False,
    }


# ---------------------------------------------------------------------------
# File Export & Download endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/exports/download/{filename}")
@app.get("/v1/exports/{filename}")
def download_export_file(filename: str) -> FileResponse:
    """Download an exported PDF or data file."""
    export_dir = Path("data/exports").resolve()
    target_file = (export_dir / filename).resolve()
    if not str(target_file).startswith(str(export_dir)) or not target_file.is_file():
        raise HTTPException(status_code=404, detail=f"Fichier exporté '{filename}' introuvable.")
    media_type = "application/pdf" if filename.lower().endswith(".pdf") else "application/octet-stream"
    return FileResponse(
        path=str(target_file),
        filename=filename,
        media_type=media_type,
    )
