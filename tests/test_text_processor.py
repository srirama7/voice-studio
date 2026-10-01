"""Unit tests for TextProcessor and CriticalTokenProtection modules."""

import pytest

from text_processor import (
    CriticalTokenProtection,
    CriticalTokenReport,
    TextProcessor,
    TextProcessingMode,
)


def test_clean_exact_mode():
    tp = TextProcessor()
    raw = "“Hello world!” — This is a test  with   multiple   spaces."
    cleaned = tp.process(raw, mode=TextProcessingMode.EXACT)
    assert cleaned == '"Hello world!" - This is a test with multiple spaces.'


def test_clean_narrative_regex_fallback():
    tp = TextProcessor()
    raw = """
    # Section Header
    **NARRATOR:** Welcome [sighs] to the demo!
    Please visit https://example.com for $100 bonus (50% increase).
    """
    cleaned = tp.clean_narrative_regex(raw)
    assert "Section Header" in cleaned
    assert "NARRATOR:" not in cleaned
    assert "[sighs]" not in cleaned
    assert "$100" in cleaned
    assert "50%" in cleaned


def test_critical_token_extraction_and_verification():
    raw = "On Jan 15th 2026, John Smith bought 50 shares for $1,250.50 (a 12.5% yield)."
    tokens = CriticalTokenProtection.extract_tokens(raw)
    
    assert "$1,250.50" in tokens["currency"] or "$1,250.50" in [t for t in tokens["currency"]]
    assert "12.5%" in tokens["percentages"]
    assert "Jan 15th 2026" in tokens["dates"] or "2026" in " ".join(tokens["dates"])
    assert "John Smith" in tokens["names"]

    # Verify matching processed text passes
    report = CriticalTokenProtection.verify(raw, raw)
    assert report.passed is True
    assert len(report.missing_tokens) == 0

    # Verify text missing $1,250.50 fails
    modified = "On Jan 15th 2026, John Smith bought 50 shares for free (a 12.5% yield)."
    report_fail = CriticalTokenProtection.verify(raw, modified)
    assert report_fail.passed is False
    assert "currency" in report_fail.missing_tokens or "numbers" in report_fail.missing_tokens


def test_chunk_text():
    tp = TextProcessor()
    long_text = (
        "First sentence is short. "
        "Second sentence is also quite reasonable. "
        "Third sentence contains important technical details about speech synthesis and audio processing. "
        "Fourth sentence finishes up the document."
    )
    chunks = tp.chunk_text(long_text, max_chars=80)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 120  # Allows sentence boundary headroom
