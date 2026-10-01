"""Unit tests for i18n Translation & Narration Text Resizing features."""

import pytest
from translator import translate_text, translate_chunk, SUPPORTED_LANGUAGES
from app import handle_narration_translation, handle_text_size_change


def test_supported_languages():
    assert "kn" in SUPPORTED_LANGUAGES
    assert "hi" in SUPPORTED_LANGUAGES
    assert "ta" in SUPPORTED_LANGUAGES
    assert "te" in SUPPORTED_LANGUAGES
    assert "en" in SUPPORTED_LANGUAGES


def test_translator_empty_text():
    assert translate_text("", target_lang="hi") == ""
    assert translate_chunk("", target_lang="kn") == ""


def test_translator_english_to_kannada():
    input_text = "Welcome to Voice Studio"
    translated = translate_text(input_text, target_lang="kn")
    assert isinstance(translated, str)
    assert len(translated) > 0


def test_handle_narration_translation():
    text = "Hello world presentation narration"
    translated, html_preview = handle_narration_translation(text, target_lang="hi", text_size="Large (22px)")
    assert isinstance(translated, str)
    assert len(translated) > 0
    assert "22px" in html_preview
    assert "Preview" in html_preview


def test_handle_text_size_change():
    text = "Sample translated narration text"
    preview_small = handle_text_size_change(text, "Small (14px)")
    assert "14px" in preview_small
    assert "Sample translated narration text" in preview_small

    preview_xl = handle_text_size_change(text, "Extra Large (26px)")
    assert "26px" in preview_xl
    assert "Sample translated narration text" in preview_xl
