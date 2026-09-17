"""
tests/test_pdf_skill.py
Tests for the PDF skill, tool registration, SKILL.md loading, and fallbacks.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from erp_agent.config import Settings
from erp_agent.db import Database
from erp_agent.skills.loader import SkillLoader
from erp_agent.tools import ToolRegistry
from erp_agent.agent import AgentRuntime


@pytest.fixture
def temp_db():
    tmp = tempfile.mktemp(suffix=".db")
    db = Database(tmp)
    return db


@pytest.fixture
def registry(temp_db):
    return ToolRegistry(temp_db)


class TestPdfSkillLoader:
    def test_skill_loader_discovers_pdf_skill(self, temp_db):
        loader = SkillLoader()
        skills = loader.load_all(db_instance=temp_db)
        assert "pdf" in skills
        pdf_skill = skills["pdf"]
        assert pdf_skill.name == "pdf"
        assert len(pdf_skill.instructions) > 50
        assert "PDF Processing Guide" in pdf_skill.instructions
        tool_names = {t.name for t in pdf_skill.tools}
        assert {"read_pdf", "extract_pdf_tables", "export_document_to_pdf", "merge_pdfs"}.issubset(tool_names)

    def test_registry_has_pdf_tools_and_instructions(self, registry):
        assert "read_pdf" in registry.tools
        assert "extract_pdf_tables" in registry.tools
        assert "export_document_to_pdf" in registry.tools
        assert "merge_pdfs" in registry.tools

        instructions = registry.skill_instructions()
        assert "pdf" in instructions
        assert "PDF Processing Guide" in instructions["pdf"]


class TestPdfToolExecution:
    def test_export_document_to_pdf_default_path(self, temp_db, registry):
        # DOC-101 exists in seed
        handler = registry.get("export_document_to_pdf").handler
        res = handler(doc_id="DOC-101")
        assert res["found"] is True
        pdf_path = Path(res["pdf_path"])
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0

    def test_export_document_to_pdf_custom_path(self, temp_db, registry, tmp_path):
        custom_out = str(tmp_path / "custom_doc.pdf")
        handler = registry.get("export_document_to_pdf").handler
        res = handler(doc_id="DOC-101", output_path=custom_out)
        assert res["found"] is True
        assert Path(custom_out).exists()

    def test_read_pdf(self, temp_db, registry, tmp_path):
        # First export a document
        pdf_out = str(tmp_path / "invoice.pdf")
        registry.get("export_document_to_pdf").handler(doc_id="DOC-101", output_path=pdf_out)

        read_res = registry.get("read_pdf").handler(file_path=pdf_out)
        assert read_res["found"] is True
        assert read_res["total_pages"] >= 1
        assert "Client Invoice 101" in read_res["text"] or "Wireless Mouse" in read_res["text"]

    def test_merge_pdfs(self, temp_db, registry, tmp_path):
        pdf1 = str(tmp_path / "doc1.pdf")
        pdf2 = str(tmp_path / "doc2.pdf")
        registry.get("export_document_to_pdf").handler(doc_id="DOC-101", output_path=pdf1)
        registry.get("export_document_to_pdf").handler(doc_id="DOC-102", output_path=pdf2)

        merged_out = str(tmp_path / "merged.pdf")
        merge_res = registry.get("merge_pdfs").handler(file_paths=[pdf1, pdf2], output_path=merged_out)
        assert merge_res["found"] is True
        assert Path(merged_out).exists()

        # Read back merged PDF to verify 2 pages
        read_merged = registry.get("read_pdf").handler(file_path=merged_out)
        assert read_merged["found"] is True
        assert read_merged["total_pages"] == 2


class TestAgentPdfInteraction:
    def test_agent_read_pdf_flow(self, tmp_path):
        db_path = tempfile.mktemp(suffix=".db")
        settings = Settings(database_path=db_path, planner="rules")
        agent = AgentRuntime(settings)

        # Export a PDF first
        pdf_path = str(tmp_path / "report.pdf")
        agent.tools.get("export_document_to_pdf").handler(doc_id="DOC-101", output_path=pdf_path)

        res = agent.run("s-pdf", "user-1", f"read pdf '{pdf_path}'")
        assert res["status"] == "completed"
        assert "PDF Content" in res["reply"] or "Client Invoice 101" in res["reply"]
