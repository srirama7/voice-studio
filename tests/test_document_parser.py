"""Unit tests for DocumentParser module."""

import os
from pathlib import Path
import pytest

from document_parser import DocumentParser, ParsedDocument, SlideData


def test_parse_txt_file(tmp_path):
    txt_file = tmp_path / "sample_presentation.txt"
    content = """# Slide 1: Introduction
Welcome to the Voice Studio presentation.

---

# Slide 2: Technical Architecture
This section details the modular pipeline architecture.
Here are the key points to understand.
"""
    txt_file.write_text(content, encoding="utf-8")

    doc = DocumentParser.parse(txt_file)
    assert isinstance(doc, ParsedDocument)
    assert doc.doc_type == "txt"
    assert doc.total_slides >= 2
    assert len(doc.slides) >= 2

    s1 = doc.slides[0]
    assert isinstance(s1, SlideData)
    assert "Introduction" in s1.title
    assert "Welcome" in s1.get_narrative_text()


def test_parse_markdown_file(tmp_path):
    md_file = tmp_path / "notes.md"
    content = """# Executive Summary
The system provides zero-shot voice cloning.

# Performance Benchmarks
Latency is under 50 milliseconds.
"""
    md_file.write_text(content, encoding="utf-8")

    doc = DocumentParser.parse(md_file)
    assert doc.doc_type == "txt"
    assert doc.total_slides == 2
    assert "Executive Summary" in doc.slides[0].title


def test_parse_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        DocumentParser.parse("nonexistent_document_file.xyz")


def test_parse_unsupported_format(tmp_path):
    invalid_file = tmp_path / "data.xyz"
    invalid_file.write_text("dummy data", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported document format"):
        DocumentParser.parse(invalid_file)
