from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx


class SpecializedDocumentExtractionClient:
    """HTTP adapter for non-invoice WIND document extraction endpoints."""

    ENDPOINTS = {
        "bank_statement": "/v1/bank-statements/extract",
        "cheque": "/v1/cheques/extract",
        "bill_of_exchange": "/v1/bills-of-exchange/extract",
    }

    def __init__(self, base_url: str, timeout: float = 180.0, api_token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_token = api_token

    def extract(
        self,
        document_type: str,
        file_path: str,
        tenant_id: str,
        bank_layout: str | None = None,
        document_id: str | None = None,
        debug: bool = False,
    ) -> dict[str, Any]:
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            return {"found": False, "error": f"Document file not found: {path}"}

        endpoint = self.ENDPOINTS.get(document_type)
        if not endpoint:
            return {"found": False, "error": f"Unsupported extraction document type: {document_type}"}

        data: dict[str, str] = {
            "tenant_id": tenant_id,
            "debug": str(debug).lower(),
        }
        if document_type == "bank_statement":
            data["bank_layout"] = bank_layout or "auto"
        if document_id:
            data["document_id"] = document_id

        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        try:
            with path.open("rb") as document_file:
                response = httpx.post(
                    f"{self.base_url}{endpoint}",
                    files={"file": (path.name, document_file, _content_type(path))},
                    data=data,
                    headers=headers,
                    timeout=self.timeout,
                )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            try:
                detail: Any = exc.response.json()
            except ValueError:
                detail = exc.response.text
            return {
                "found": False,
                "error": f"Document extraction service returned HTTP {exc.response.status_code}.",
                "details": detail,
            }
        except (httpx.HTTPError, OSError, ValueError) as exc:
            return {"found": False, "error": f"Document extraction service unavailable: {exc}"}

        extracted = payload.get("result") or {}
        return {
            "found": True,
            "source": "wind-document-extraction",
            "document_type": document_type,
            "request_id": payload.get("request_id"),
            "document_id": payload.get("document_id") or document_id,
            "model_used": payload.get("model_used"),
            "warnings": payload.get("warnings") or [],
            "validation_errors": payload.get("validation_errors") or [],
            "extraction": extracted,
            "document_data": extracted,
            "message": f"{document_type.replace('_', ' ').title()} extracted successfully.",
        }


def _content_type(path: Path) -> str:
    return {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(path.suffix.lower(), "application/octet-stream")
