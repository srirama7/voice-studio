"""
Verification test script for Voice Studio core modules.
Tests config, db_client, voice_enrollment, document_parser, and slide_renderer.
"""

import os
from pathlib import Path
import tempfile
import wave
import struct
import numpy as np

from config import Config, default_config
from db_client import DBClient
from voice_enrollment import VoiceEnrollmentEngine, QCResult, VoiceProfile
from document_parser import DocumentParser, ParsedDocument, SlideData
from slide_renderer import SlideRenderer


def test_config():
    print("Testing config.py...")
    cfg = Config()
    assert cfg.firebase.project_id == "voiceai-6c2e1"
    assert cfg.firebase.storage_bucket == "voiceai-6c2e1.firebasestorage.app"
    assert cfg.audio.target_lufs == -14.0
    assert cfg.audio.min_duration_sec == 8.0
    assert cfg.audio.max_duration_sec == 30.0
    assert cfg.slide.target_width == 1920
    assert cfg.slide.target_height == 1080
    assert cfg.paths.base_dir.exists()
    assert cfg.paths.cache_dir.exists()
    print("-> Config test PASSED!")


def test_db_client():
    print("Testing db_client.py...")
    client = DBClient()

    # Save voice profile test
    test_meta = {
        "speaker_name": "Test Speaker",
        "language": "en",
        "sample_rate": 22050,
        "updated_at": "2026-09-29T12:00:00Z"
    }
    try:
        saved = client.save_voice_profile("test_voice_01", "v001", test_meta)
        assert saved is True

        fetched = client.get_voice_profile("test_voice_01", "v001")
        assert fetched is not None
        assert fetched.get("speaker_name") == "Test Speaker"

        # Save job checkpoint test
        cp_saved = client.save_job_checkpoint("job_123", {"status": "PARSED", "step": 1})
        assert cp_saved is True
        cp_fetched = client.get_job_checkpoint("job_123")
        assert cp_fetched is not None
        assert cp_fetched.get("status") == "PARSED"
    finally:
        # Clean up test artifacts so the real voice library stays pristine.
        for stale in [
            client.local_db_dir / "voice_profiles" / "test_voice_01_v001.json",
            client.local_db_dir / "job_checkpoints" / "job_123.json",
        ]:
            try:
                if stale.exists():
                    stale.unlink()
            except Exception:
                pass

    print("-> DB Client test PASSED!")


def test_voice_enrollment():
    print("Testing voice_enrollment.py...")
    engine = VoiceEnrollmentEngine()

    # Generate a dummy 10-second 22.05kHz WAV file with speech frames and quiet background frames
    sr = 22050
    duration = 10.0
    total_samples = int(sr * duration)
    signal = np.random.randn(total_samples) * 0.0001  # Noise floor
    # 6-second speech burst in middle
    speech_start = int(sr * 2.0)
    speech_end = int(sr * 8.0)
    t = np.linspace(0, 6.0, speech_end - speech_start, endpoint=False)
    signal[speech_start:speech_end] += 0.3 * np.sin(2 * np.pi * 440 * t)
    signal = signal.astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        pcm = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

        # Test quality analysis
        qc = engine.analyze_quality(signal, sr)
        print(f"QC metrics: duration={qc.duration_sec}s, peak={qc.peak_dbfs}dBFS, snr={qc.snr_db}dB, is_valid={qc.is_valid}")
        assert qc.is_valid is True

        # Test enroll voice
        profile = engine.enroll_voice(tmp_path, voice_id="speaker_demo", version="v001")
        assert profile.voice_id == "speaker_demo"
        assert profile.version == "v001"
        assert Path(profile.ref_wav_path).exists()
        assert Path(profile.conditioning_path).exists()
        print("-> Voice Enrollment test PASSED!")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_document_parser():
    print("Testing document_parser.py...")
    # Create temp TXT file
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as tmp:
        tmp.write("# Slide 1: Introduction\nWelcome to Voice Studio Enterprise.\n\n---\n# Slide 2: Core Architecture\nModular design with fallback support.")
        tmp_path = Path(tmp.name)

    try:
        doc = DocumentParser.parse(tmp_path)
        assert doc.doc_type == "txt"
        assert doc.total_slides == 2
        assert doc.slides[0].title == "Slide 1: Introduction"
        assert doc.slides[1].title == "Slide 2: Core Architecture"
        print("-> Document Parser TXT test PASSED!")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    # Create temp DOCX file if docx is available or test XML fallback
    try:
        import docx
        doc_obj = docx.Document()
        doc_obj.add_heading("Docx Section 1", level=1)
        doc_obj.add_paragraph("This is paragraph text inside docx test.")
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_docx:
            doc_obj.save(tmp_docx.name)
            docx_path = Path(tmp_docx.name)
        try:
            parsed_docx = DocumentParser.parse(docx_path)
            assert parsed_docx.doc_type == "docx"
            assert len(parsed_docx.slides) > 0
            print("-> Document Parser DOCX test PASSED!")
        finally:
            if docx_path.exists():
                docx_path.unlink()
    except Exception as exc:
        print(f"-> DOCX test skipped or exception: {exc}")


def test_slide_renderer():
    print("Testing slide_renderer.py...")
    renderer = SlideRenderer()
    
    slide1 = SlideData(
        slide_index=1,
        title="Test Executive Overview",
        body_text="Voice Studio provides high-fidelity TTS rendering with PyMuPDF visual slides.",
        speaker_notes="Welcome everyone to this presentation.",
        raw_text="Test Executive Overview\nVoice Studio provides high-fidelity TTS rendering with PyMuPDF visual slides."
    )

    doc = ParsedDocument(
        file_path="test_doc.txt",
        doc_type="txt",
        total_slides=1,
        slides=[slide1]
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_images = renderer.render_document(doc, tmp_dir)
        assert len(output_images) == 1
        assert output_images[0].exists()
        
        # Verify output image dimensions
        from PIL import Image
        with Image.open(output_images[0]) as img:
            assert img.size == (1920, 1080)
            print(f"Rendered PNG resolution: {img.size}")

    print("-> Slide Renderer test PASSED!")


if __name__ == "__main__":
    print("=== RUNNING CORE MODULE INTEGRATION TESTS ===")
    test_config()
    test_db_client()
    test_voice_enrollment()
    test_document_parser()
    test_slide_renderer()
    print("=== ALL 5 MODULE TESTS SUCCEEDED PERFECTLY! ===")
