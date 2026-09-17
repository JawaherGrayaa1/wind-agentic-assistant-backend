from typing import Any, Literal
from pydantic import BaseModel, Field


class DocumentClient(BaseModel):
    name: str = ""
    tax_id: str | None = None


class DocumentLineItem(BaseModel):
    line: int = 0
    product_id: str | None = None
    name: str = ""
    quantity: float = 1
    unit_price: float = 0
    discount_pct: float = 0
    line_brut_ht: float = 0
    line_remise: float = 0
    line_net_ht: float = 0


class DocumentFinancials(BaseModel):
    items: list[DocumentLineItem] = Field(default_factory=list)
    item_count: int = 0
    tax_rate: float = 19
    global_discount_pct: float = 0
    total_brut_ht: float = 0
    total_remises: float = 0
    total_net_ht: float = 0
    total_tva: float = 0
    timbre_fiscal: float = 0
    total_ttc: float = 0
    currency: str = "TND"


class DocumentSchema(BaseModel):
    """Renderer/API contract for every ERP document."""

    schema_version: str = "1.0"
    document_type: str = "general"
    document_id: str
    title: str
    status: str = "draft"
    invoice_date: str | None = None
    due_date: str | None = None
    document_date: str | None = None
    payment_terms: str = ""
    client: DocumentClient = Field(default_factory=DocumentClient)
    currency: str = "TND"
    items: list[DocumentLineItem] = Field(default_factory=list)
    financials: DocumentFinancials = Field(default_factory=DocumentFinancials)
    # Kept for legacy generic documents; commercial tables never render it.
    content: str | None = None

    model_config = {"extra": "allow"}

class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=20000)
    doc_id: str | None = Field(default=None, description="Optional active document ID being edited")

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    status: Literal["completed", "approval_required", "error"]
    approval_token: str | None = None
    trace: list[dict[str, Any]] = []
    doc_id: str | None = None
    document: dict[str, Any] | None = None
    file_url: str | None = None
    file_name: str | None = None

class ApprovalResponse(BaseModel):
    approval_token: str
    status: Literal["completed", "error"]
    result: dict[str, Any] | None = None
    message: str
    doc_id: str | None = None
    document: dict[str, Any] | None = None
    file_url: str | None = None
    file_name: str | None = None
