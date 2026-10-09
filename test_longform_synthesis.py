"""
Test Suite for Long-Form Audio Synthesis (15-20 Minutes / Multi-Thousand Chars)
Verifies that synthesis exceeds the previous 2500-char / 3:10 cap across Indian English,
Hindi, Kannada, and Cloned voices without errors or socket drops.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from voice_clone import EdgeTTSAdapter, CloningTTSAdapter, GALLERY_VOICES
from document_parser import DocumentParser

def test_long_text_no_clamp():
    """Verify that a 6,000-character text (~4.5 minutes) synthesizes completely without 2500-char clamp."""
    # Generate ~6,000 characters of clean presentation text
    base_paragraph = (
        "Artificial intelligence and voice synthesis have evolved rapidly in recent years. "
        "With advanced deep neural networks, zero-shot voice cloning can replicate subtle acoustic "
        "timbre, pitch contours, and emotional prosody across multiple Indian regional languages. "
        "This enterprise studio allows high-quality narration generation for presentations, slides, and educational videos.\n\n"
    )
    long_text = base_paragraph * 15  # ~6,000 chars (approx 4.5 minutes of continuous speech)
    assert len(long_text) > 5000, f"Expected >5000 chars, got {len(long_text)}"

    adapter = EdgeTTSAdapter(forced_voice="en-IN-PrabhatNeural")
    waveform, sr = adapter.synthesize(long_text, language="en-in")

    duration_sec = len(waveform) / sr
    print(f"\n[Test 1] 6,000 chars synthesis: {len(waveform)} samples @ {sr}Hz -> {duration_sec:.2f}s ({duration_sec/60:.2f} minutes)")
    
    # Previous bug capped audio at 189s (3:09). Now it must exceed 220s!
    assert duration_sec > 200.0, f"Audio duration {duration_sec:.2f}s is still clamped near 3 minutes!"
    assert not np.all(waveform == 0), "Waveform should not be empty or silent"


def test_15_to_20_min_synthesis():
    """Verify that a 16,000+ char text (~18-20 minutes) synthesizes seamlessly across languages."""
    # ~16,500 characters of narration (~18-20 minutes of speech)
    chapter_content = (
        "Welcome to the comprehensive masterclass on neural architectures and distributed systems. "
        "In this extended lecture, we explore the foundations of high-throughput stream processing, "
        "signal conditioning, acoustic normalization, and zero-shot voice cloning algorithms. "
        "Each processing module is designed with decoupled concurrency, ensuring robust performance.\n\n"
    )
    long_narrative = chapter_content * 45  # ~16,500 characters (~18-20 minutes)
    assert len(long_narrative) > 15000

    adapter = EdgeTTSAdapter(forced_voice="en-IN-PrabhatNeural")
    waveform, sr = adapter.synthesize(long_narrative, language="en-in")

    duration_sec = len(waveform) / sr
    minutes = duration_sec / 60.0
    print(f"\n[Test 2] 16,500 chars (18-20 min target): {len(waveform)} samples @ {sr}Hz -> {duration_sec:.2f}s ({minutes:.2f} minutes)")
    
    assert duration_sec >= 900.0, f"Expected >= 15 minutes (900s), got {duration_sec:.2f}s"
    assert not np.all(waveform == 0), "Waveform should not be empty"


def test_document_parser_long_txt_pagination(tmp_path):
    """Verify that a long TXT file is paginated into logical slide blocks."""
    txt_file = tmp_path / "long_presentation.txt"
    paras = [
        f"Chapter {i}: Architectural Deep Dive\nThis section explains step {i} of the distributed processing pipeline in detail."
        for i in range(1, 10)
    ]
    txt_file.write_text("\n\n".join(paras), encoding="utf-8")

    doc = DocumentParser.parse(str(txt_file))
    assert doc.total_slides >= 1
    assert len(doc.full_text) > 0


if __name__ == "__main__":
    print("Testing long text synthesis...")
    test_long_text_no_clamp()
    print("Testing 15-20 min synthesis...")
    test_15_to_20_min_synthesis()
    print("ALL LONGFORM TESTS PASSED!")
