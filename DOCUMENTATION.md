# WIND ERP — Agentic System Documentation

> **Version:** 1.0  
> **Stack:** Python 3.12 · FastAPI · SQLite · Vosk ASR · Ollama LLM  
> **Frontend:** Angular 18 (TypeScript)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Models & AI Components](#3-models--ai-components)
4. [Planner](#4-planner)
5. [Core Tools](#5-core-tools)
6. [Skills](#6-skills)
   - [Invoice Skill](#61-invoice-skill)
   - [Bon de Commande Skill](#62-bon-de-commande-skill)
   - [PDF Skill](#63-pdf-skill)
   - [XLSX Skill](#64-xlsx-skill)
7. [Database](#7-database)
8. [REST API](#8-rest-api)
9. [ASR — Voice Input](#9-asr--voice-input)
10. [Frontend](#10-frontend)
11. [Configuration Reference](#11-configuration-reference)
12. [Approval Flow](#12-approval-flow)

---

## 1. System Overview

WIND ERP is a **conversational agentic ERP assistant** that accepts natural language (text or voice) from users and maps requests to structured ERP actions: creating and managing invoices, purchase orders, documents, searching inventory, and reading/exporting files.

The system is multilingual — it understands **French**, **Arabic (Tunisian dialect via voice)**, and **English** in the same session.

```
User (text / voice)
        │
        ▼
  [FastAPI Backend]  ──────────────────► [SQLite DB]
        │
        ├── [Planner] ──► RulePlanner (deterministic)
        │              └► OllamaPlanner (LLM, remote/local)
        │
        └── [ToolRegistry]
                ├── Core Tools  (inventory, documents, orders)
                └── Skill Tools (invoice, bon_de_commande, pdf, xlsx)
```

---

## 2. Architecture

### Key Modules

| File | Role |
|------|------|
| `api.py` | FastAPI entry point; defines all HTTP endpoints |
| `agent.py` | `AgentRuntime` — orchestrates planner + tool execution |
| `planner.py` | `RulePlanner` + `OllamaPlanner` — intent detection and tool selection |
| `tools.py` | `ToolRegistry` — registers and dispatches all tools |
| `db.py` | `Database` — SQLite layer (documents, orders, products, events) |
| `document_editor.py` | `SmartDocumentEditor` — AI-powered document content editing via Ollama |
| `memory.py` | `Memory` — per-user conversation context and preference recall |
| `asr.py` | `VoskTunisianASR` — offline voice-to-text transcription |
| `input.py` | Text normalization after ASR (via Ollama) |
| `config.py` | `Settings` dataclass — all env-var driven configuration |
| `llm_logging.py` | Structured logging for all LLM requests/responses/errors |
| `skills/loader.py` | `SkillLoader` — auto-discovers and loads all skill directories |

### Skill Auto-Discovery

The `SkillLoader` scans `src/erp_agent/skills/*/` at startup. Any directory containing a `tools.py` with a `register_tools(db, skill_dir)` function is automatically loaded and its tools registered into the `ToolRegistry`.

```
skills/
├── invoice/          ← Invoice management skill
│   ├── SKILL.md
│   └── tools.py
├── bon_de_commande/  ← Purchase order skill
│   ├── SKILL.md
│   └── tools.py
├── pdf/              ← PDF reading/export/merge skill
│   ├── SKILL.md
│   └── tools.py
└── xlsx/             ← Excel read/export skill
    ├── SKILL.md
    └── tools.py
```

---

## 3. Models & AI Components

### 3.1 LLM — Ollama

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Local Ollama or remote (ngrok tunnel to Kaggle GPU) |
| `OLLAMA_MODEL` | `qwen3:8b` | Primary LLM for planning and document editing |
| `OLLAMA_TIMEOUT` | `180s` | Max wait time per LLM call |

**Supported deployment modes:**
- **Local:** Ollama running on `localhost:11434`
- **Remote (Kaggle/Colab GPU):** Expose via ngrok tunnel, set `NGROK_URL` or `OLLAMA_BASE_URL`

The LLM is used in two places:
1. **OllamaPlanner** — selects the right tool and extracts arguments from natural language
2. **SmartDocumentEditor** — rewrites/edits document content based on natural language instructions

### 3.2 ASR — Vosk (Tunisian Arabic)

| Setting | Default | Description |
|---------|---------|-------------|
| `VOSK_MODEL_DIR` | _(unset)_ | Path to LinTO Tunisian Vosk model directory |

- **Model:** LinTO Tunisian Arabic model distributed in Vosk format
- **Audio format:** WAV, 16kHz mono 16-bit (auto-converted via `pydub` if needed)
- **Transcription normalizer:** A second Ollama call can clean up raw ASR output (`TRANSCRIPTION_NORMALIZER=ollama`)

### 3.3 PDF Libraries (multi-fallback)

The PDF skill tries three libraries in order:
1. **PyMuPDF (`fitz`)** — preferred, fastest
2. **pdfplumber** — fallback, needed for table extraction
3. **pypdf** — last resort

PDF generation uses **ReportLab**.

---

## 4. Planner

The planner receives the user's text, conversation context, and the list of all available tools, then returns a `Plan` object:

```python
@dataclass
class Plan:
    intent: str               # e.g. "create_invoice", "conversation"
    reply: str | None         # Direct reply if no tool needed
    tool_name: str | None     # Tool to call
    arguments: dict | None    # Tool arguments
    summary: str              # Human-readable explanation of what the agent plans to do
```

### 4.1 RulePlanner (default)

Deterministic, regex-based planner. No LLM needed. Handles the full set of ERP intents:

| Intent trigger | Mapped tool |
|----------------|-------------|
| "stock", "inventory", "disponible" | `get_inventory` / `search_products` |
| "create document", "nouveau document" | `create_document` |
| "edit document", "modifier document" | `edit_document` / `smart_edit_document` |
| "delete document" | `delete_document` |
| "search documents", "find" | `search_documents` |
| "pdf" + "export/read/merge/table" | `export_document_to_pdf` / `read_pdf` / `merge_pdfs` / `extract_pdf_tables` |
| "bon de commande", "bc", "bd", "bl", "purchase order" | `create_bon_de_commande` / `add_order_item` / `get_order_summary` / `export_bon_de_commande_pdf` |
| "facture", "invoice", "facturation" | `create_invoice` / `add_invoice_item` / `validate_invoice` / `get_invoice_summary` |
| "remember", "prefers" | `remember_fact` |
| "create order", "commande" | `create_sales_order` |

Also extracts structured data from natural language:
- **Document IDs**: `DOC-101`, `DOC_XYZ`, `DOC42`
- **Order IDs**: `BC-001`, `BC-DEMO`
- **Invoice IDs**: `INV-001`, `INV-XYZ`
- **Product IDs**: `P-01`, `P123`
- **Item/quantity parsing**: "2 laptops and 1 mouse" → `[{name: "laptops", quantity: 2}, {name: "mouse", quantity: 1}]`

### 4.2 OllamaPlanner (optional)

Activated with `AGENT_PLANNER=ollama`. Sends the user message, context, and tool schema to the LLM which returns a JSON plan. Features:

- JSON mode (`format: json`) for structured output
- Automatic fallback to `RulePlanner` if the LLM returns an invalid or unknown tool name
- Markdown code fence stripping (handles LLMs that wrap JSON in ` ```json ``` `)
- Skill instructions injected into the system prompt as a knowledge base

---

## 5. Core Tools

These tools are always available, regardless of which skills are loaded.

| Tool | Confirmation Required | Description |
|------|-----------------------|-------------|
| `get_inventory` | No | Look up stock level for a product by ID |
| `get_product` | No | Get complete product catalog details (name, stock, price) |
| `search_products` | No | Search product catalog by name or ID |
| `create_product` | **Yes** | Add a new product to the catalog with initial stock and price |
| `update_product_stock` | **Yes** | Update stock quantity, price, or name for an existing product |
| `get_document` | No | Retrieve a document by its ID |
| `search_documents` | No | Full-text search across documents |
| `create_document` | **Yes** | Create a new document (any type) |
| `edit_document` | **Yes** | Directly overwrite a document's content |
| `smart_edit_document` | **Yes** | Intelligently edit a document using AI given a natural language instruction |
| `delete_document` | **Yes** | Permanently delete a document |
| `create_sales_order` | **Yes** | Create a generic sales order |
| `remember_fact` | **Yes** | Save a user preference or note to memory |

---

## 6. Skills

Skills are pluggable modules that extend the agent with domain-specific tools.

### 6.1 Invoice Skill

**Directory:** `skills/invoice/`  
**Purpose:** Full lifecycle management of invoices in Tunisian ERP format (HT / TVA 19% / Timbre Fiscal / TTC)

Documents are stored as **Markdown** in the database for live editing and rendering in the frontend.

#### Financial Calculation Logic

```
Total Brut HT   = Σ (qty × unit_price)
Line Remise     = line_brut × (discount_pct / 100)
Net after Items = Total Brut HT − Σ Line Remises
Global Remise   = Net after Items × (global_discount_pct / 100)
Total Net HT    = Net after Items − Global Remise
TVA             = Total Net HT × (tax_rate / 100)   [default 19%]
Timbre Fiscal   = 1.000 TND                          [TND only]
Total TTC       = Total Net HT + TVA + Timbre Fiscal
```

#### Tools

| Tool | Confirmation | Parameters | Description |
|------|-------------|------------|-------------|
| `create_invoice` | **Yes** | `client_name`, `client_tax_id`, `items[]`, `invoice_id`, `due_date`, `payment_terms`, `tax_rate`, `discount_pct`, `currency` | Creates a new invoice with auto-calculation and Markdown formatting |
| `get_invoice_summary` | No | `invoice_id` | Parses and recalculates totals from stored invoice Markdown |
| `list_invoices` | No | `status`, `client_name`, `limit` | Lists all invoices with client name, status, and Total TTC summary |
| `add_invoice_item` | **Yes** | `invoice_id`, `name`, `quantity`, `unit_price`, `discount_pct`, `product_id` | Adds a line item; auto-looks up price from product catalog |
| `update_invoice_item` | **Yes** | `invoice_id`, `line_number`, `product_id`, `name`, `quantity`, `unit_price`, `discount_pct` | Updates a specific line by line number, product ID, or name |
| `remove_invoice_item` | **Yes** | `invoice_id`, `line_number`, `product_id`, `name` | Removes a line item by line number, product ID, or name |
| `validate_invoice` | **Yes** | `invoice_id` | Runs compliance checks and sets status to `approved` |
| `duplicate_invoice` | **Yes** | `invoice_id`, `new_client_name`, `new_invoice_id` | Clones an existing invoice for a new client/period |
| `delete_invoice` | **Yes** | `invoice_id` | Deletes an invoice document |
| `export_invoice_pdf` | **Yes** | `invoice_id`, `output_path` | Generates a styled PDF invoice with legal breakdown |

#### Validation Rules (checked by `validate_invoice`)

- Client name must be set
- At least one line item must exist
- All quantities must be > 0
- All unit prices must be >= 0
- Warning (non-blocking): Missing Tax ID / Matricule Fiscal

#### Invoice Markdown Format

Invoices are stored as Markdown with these sections:
- Header: `# FACTURE N° {id}`
- Metadata block: date, due date, client, tax ID, payment terms, status
- Line items table (# | Réf | Désignation | Qté | Prix Unit HT | Remise% | Total Net HT)
- Financial summary table (Brut HT, Remises, Net HT, TVA, Timbre, TTC)

---

### 6.2 Bon de Commande Skill

**Directory:** `skills/bon_de_commande/`  
**Purpose:** Purchase/Sales order management with financial calculations and PDF export

#### Financial Calculation Logic

Same as invoice except no Timbre Fiscal:
```
Total TTC = Total Net HT + TVA
```

#### Tools

| Tool | Confirmation | Parameters | Description |
|------|-------------|------------|-------------|
| `create_bon_de_commande` | **Yes** | `client_name`, `items[]`, `tax_rate`, `discount_pct`, `order_id`, `currency` | Creates a BC with auto-ID (e.g. `BC-A3F2B1`), persists to DB |
| `add_order_item` | **Yes** | `order_id`, `product_id`, `name`, `quantity`, `unit_price`, `discount_pct` | Adds item; merges with existing line if same product |
| `update_order_item` | **Yes** | `order_id`, `item_index`, `product_id`, `name`, `quantity`, `unit_price`, `discount_pct` | Updates a specific line by index, product ID, or name |
| `remove_order_item` | **Yes** | `order_id`, `item_index`, `product_id`, `name` | Removes a specific line item |
| `get_order_summary` | No | `order_id` | Returns itemized financials and full totals |
| `export_bon_de_commande_pdf` | **Yes** | `order_id`, `output_path` | Generates a styled PDF (ReportLab) saved to `data/exports/` |

---

### 6.3 PDF Skill

**Directory:** `skills/pdf/`  
**Purpose:** Read, extract, export, and merge PDF files

#### Tools

| Tool | Confirmation | Parameters | Description |
|------|-------------|------------|-------------|
| `read_pdf` | No | `file_path`, `max_pages` (default 10) | Extracts text + metadata; tries fitz → pdfplumber → pypdf |
| `extract_pdf_tables` | No | `file_path`, `page_number` | Extracts structured tables from PDF via pdfplumber |
| `export_document_to_pdf` | **Yes** | `doc_id`, `output_path` | Exports a stored ERP document to styled PDF; saved to `data/exports/` |
| `merge_pdfs` | **Yes** | `file_paths[]`, `output_path` | Merges multiple PDFs; tries fitz → pypdf |

---

### 6.4 XLSX Skill

**Directory:** `skills/xlsx/`  
**Purpose:** Read from and export to Excel spreadsheet files

#### Tools

| Tool | Confirmation | Parameters | Description |
|------|-------------|------------|-------------|
| `read_xlsx` | No | `file_path`, `sheet_name` | Reads all rows from a sheet using openpyxl |
| `export_to_excel` | No | `data[]`, `output_path` | Exports a list of dicts to `.xlsx` with auto-headers |

---

## 7. Database

**Engine:** SQLite  
**Path:** `data/erp_agent.db` (configurable via `DATABASE_PATH`)

### Tables

| Table | Description |
|-------|-------------|
| `documents` | Stores all ERP documents (invoices, contracts, quotes). Content is Markdown. |
| `orders` | Bon de Commande and sales orders. Items stored as JSON array. |
| `products` | Product catalog with `product_id`, `name`, `stock`, `price` |
| `messages` | Full conversation history per session |
| `events` | Audit log of all agent actions per session |
| `approvals` | Pending approval tokens for write operations |
| `user_memory` | Key-value store per user for preferences and facts |

### Document Schema

```
doc_id      TEXT PRIMARY KEY    -- e.g. "INV-001", "DOC-42", "BC-XYZ"
title       TEXT
doc_type    TEXT                -- "invoice", "contract", "bon_de_commande", "general"
content     TEXT                -- Full Markdown
status      TEXT                -- "draft" | "approved" | "paid" | "cancelled"
created_at  TEXT
updated_at  TEXT
```

---

## 8. REST API

**Base URL:** `http://localhost:`  
**Framework:** FastAPI (auto-docs at `/docs`)

### Endpoints

#### `POST /chat`
Main conversational endpoint.

**Request:**
```json
{
  "session_id": "sess-abc",
  "user_id": "user-1",
  "text": "create an invoice for Client Alpha with 2 laptops at 1200 TND",
  "doc_id": "INV-001"
}
```

**Response:**
```json
{
  "session_id": "sess-abc",
  "reply": "...",
  "status": "ok | approval_required",
  "approval_token": "appr-xxxx",
  "trace": [...],
  "doc_id": "INV-001",
  "document": { ... }
}
```

#### `POST /approve/{token}`
Executes a previously pending write action.

#### `POST /transcribe`
Submits a WAV audio file for ASR transcription.

**Request:** `multipart/form-data` with `audio` field  
**Response:** `{ "text": "transcribed text" }`

#### `GET /documents`
Lists all documents (with optional `?type=invoice&status=draft` filters).

#### `GET /documents/{doc_id}`
Retrieves a single document by ID.

#### `POST /documents/upload`
Uploads a document file (PDF, DOCX, TXT, MD) which is processed and stored.

#### `GET /invoices`
Returns all documents of type `invoice`.

#### `GET /history/{session_id}`
Returns full message history for a session.

#### `GET /health`
Health check endpoint.

---

## 9. ASR — Voice Input

**Class:** `VoskTunisianASR`  
**Model:** LinTO Tunisian Arabic (Vosk format)

### Pipeline

```
WAV upload (any format)
      │
      ▼
  pydub normalization  ──► 16kHz mono 16-bit WAV
      │
      ▼
  Vosk KaldiRecognizer ──► raw Arabic transcript
      │
      ▼
  (optional) Ollama normalizer ──► cleaned, punctuated text
      │
      ▼
  /chat endpoint
```

**Setup:** Set `VOSK_MODEL_DIR` in `.env` to the path of the extracted LinTO model directory.

---

## 10. Frontend

**Directory:** `Wind-Agentic-Interface/wind-agentic-ui/`  
**Framework:** Angular 18  
**Dev server:** `ng serve` → `http://localhost:4200`

### Key Components

| Component | Description |
|-----------|-------------|
| `app.component` | Root shell: sidebar, chat panel, document editor |
| `document-editor` | Live Markdown editor/viewer for invoices and documents |
| `erp-agent.service` | HTTP service communicating with the FastAPI backend |
| `audio-recorder.service` | Browser MediaRecorder → WAV → `/transcribe` |

### Communication Flow

```
User types or speaks
        │
        ▼
Angular [erp-agent.service] ─► POST /chat ─► FastAPI backend
        │
        ◄─────────────── reply + document (Markdown)
        │
        ▼
document-editor renders Markdown inline
```

---

## 11. Configuration Reference

All configuration is via environment variables (`.env` file at project root).

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_PATH` | `data/erp_agent.db` | SQLite database file path |
| `AGENT_PLANNER` | `rules` | `rules` = deterministic, `ollama` = LLM-powered |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API base URL (local or ngrok) |
| `NGROK_URL` | _(unset)_ | Alias for remote Ollama URL (Kaggle GPU tunnel) |
| `OLLAMA_MODEL` | `qwen3:8b` | LLM model name |
| `OLLAMA_TIMEOUT` | `180` | Ollama request timeout in seconds |
| `VOSK_MODEL_DIR` | _(unset)_ | Path to Vosk ASR model directory |
| `TRANSCRIPTION_NORMALIZER` | `ollama` | Post-ASR text cleanup mode (`ollama` or `none`) |
| `TRANSCRIPTION_NORMALIZER_TIMEOUT` | `60` | Timeout for ASR normalization LLM call |
| `MAX_AGENT_STEPS` | `4` | Max tool calls per agent turn |

### Example `.env`

```env
DATABASE_PATH=data/erp_agent.db
AGENT_PLANNER=ollama
OLLAMA_BASE_URL=https://your-ngrok-url.ngrok-free.app
OLLAMA_MODEL=qwen3:8b
OLLAMA_TIMEOUT=180
VOSK_MODEL_DIR=models/vosk-tunisian-ar
TRANSCRIPTION_NORMALIZER=ollama
```

---

## 12. Approval Flow

Write operations require explicit user confirmation before execution.

```
User: "create invoice for Client Alpha"
         │
         ▼
  Agent plans → create_invoice (requires_confirmation=True)
         │
         ▼
  Agent returns:
    status: "approval_required"
    approval_token: "appr-abc123"
    reply: "L'action 'create_invoice' nécessite une confirmation..."
         │
         ▼
  User clicks "Approve" in UI
         │
         ▼
  POST /approve/appr-abc123
         │
         ▼
  Tool executes → document created → returned to frontend
```

### Tools Requiring Confirmation

| Skill | Tools |
|-------|-------|
| Core | `create_document`, `edit_document`, `smart_edit_document`, `delete_document`, `create_sales_order`, `remember_fact` |
| Invoice | `create_invoice`, `add_invoice_item`, `update_invoice_item`, `remove_invoice_item`, `validate_invoice` |
| Bon de Commande | `create_bon_de_commande`, `add_order_item`, `update_order_item`, `remove_order_item`, `export_bon_de_commande_pdf` |
| PDF | `export_document_to_pdf`, `merge_pdfs` |
| XLSX | _(none — read-only operations)_ |

---

*Generated for WIND ERP Agentic System — September 2026*
