"""Unit tests for Audio Download components in app.py UI."""

import pytest
from app import handle_document_generation, handle_job_monitor, handle_voice_enrollment, build_app


def test_app_build():
    demo = build_app()
    assert demo is not None


def test_handle_voice_enrollment_signature():
    # Calling with empty audio file should return status, dict, None, None, Dropdown
    status, metrics, ref_wav, ref_download, dropdown = handle_voice_enrollment(
        audio_file=None,
        voice_id="test_voice",
        version="v001",
        strict_qc=True,
    )
    assert "Error" in status
    assert metrics == {}
    assert ref_wav is None
    assert ref_download is None


def test_handle_document_generation_signature():
    # Calling with no document file should return status, audio, audio_download, video, srt, manifest
    status, audio_res, audio_download, video_res, srt_res, manifest = handle_document_generation(
        document_file=None,
        voice_id="speaker_01",
        text_mode="exact",
        language="en-in",
    )
    assert "Please upload" in status
    assert audio_res is None
    assert audio_download is None
    assert video_res is None
    assert srt_res is None
    assert manifest == {}


def test_handle_job_monitor_signature():
    status, manifest_json, audio_res, audio_download, video_res, srt_res = handle_job_monitor(job_id_search="")
    assert "Please enter a valid Job ID" in status
    assert audio_res is None
    assert audio_download is None
    assert video_res is None
    assert srt_res is None
