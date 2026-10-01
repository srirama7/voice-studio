"""Unit tests for VideoAssembly and SRTGenerator modules."""

import wave
from pathlib import Path
import numpy as np
import pytest

from video_assembly import (
    SRTGenerator,
    SlideImageGenerator,
    VideoAssembly,
    format_srt_timestamp,
    pick_video_encoder,
)


def test_pick_video_encoder_cached_and_valid():
    enc, args = pick_video_encoder()
    assert enc in ("h264_mf", "h264_qsv", "h264_amf", "libx264")
    enc2, _ = pick_video_encoder()
    assert enc2 == enc  # probed once, then cached


def test_format_srt_timestamp():
    assert format_srt_timestamp(0.0) == "00:00:00,000"
    assert format_srt_timestamp(125.456) == "00:02:05,456"
    assert format_srt_timestamp(3661.005) == "01:01:01,005"


def test_srt_generator(tmp_path):
    srt_out = tmp_path / "test.srt"
    chunks = ["First subtitle chunk.", "Second subtitle chunk."]
    durations = [2.5, 3.0]

    res_path = SRTGenerator.generate_srt(chunks, durations, srt_out)
    assert res_path.exists()
    content = res_path.read_text(encoding="utf-8")

    assert "1\n00:00:00,000 --> 00:00:02,500" in content
    assert "First subtitle chunk." in content
    assert "Second subtitle chunk." in content


def test_slide_image_generator(tmp_path):
    img_out = tmp_path / "slide_01.png"
    res_path = SlideImageGenerator.create_slide_image(
        title="Test Architecture",
        body_text="Line 1 details.\nLine 2 specs.",
        output_path=img_out,
        slide_num=1,
        total_slides=3,
    )
    assert res_path.exists()
    assert res_path.stat().st_size > 0


def test_video_assembly_pipeline(tmp_path):
    # Create dummy WAV audio file
    audio_path = tmp_path / "audio.wav"
    sr = 22050
    duration_sec = 4.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    signal = 0.2 * np.sin(2 * np.pi * 440 * t)
    pcm_data = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_data.tobytes())

    slides = [
        {"title": "Slide 1", "body_text": "Content for slide 1"},
        {"title": "Slide 2", "body_text": "Content for slide 2"},
    ]
    cleaned_chunks = ["Content for slide 1", "Content for slide 2"]

    res = VideoAssembly.assemble(
        slides_data=slides,
        cleaned_chunks=cleaned_chunks,
        audio_path=audio_path,
        output_dir=tmp_path,
        job_id="test_assembly_job",
        padding_sec=0.5,
    )

    assert "video_path" in res
    assert "subtitle_path" in res
    assert Path(res["subtitle_path"]).exists()


def test_video_assembly_hardened_render_with_progress(tmp_path):
    # 6s audio: exercises the hardened ffmpeg runner (nostdin, file logging,
    # -progress) and proves the progress callback is wired.
    audio_path = tmp_path / "audio6.wav"
    sr = 22050
    t = np.linspace(0, 6.0, int(sr * 6.0), endpoint=False)
    signal = 0.2 * np.sin(2 * np.pi * 440 * t)
    pcm_data = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_data.tobytes())

    slides = [
        {"title": "S1", "body_text": "Alpha"},
        {"title": "S2", "body_text": "Beta"},
        {"title": "S3", "body_text": "Gamma"},
    ]
    seen = []
    res = VideoAssembly.assemble(
        slides_data=slides,
        cleaned_chunks=["Alpha", "Beta", "Gamma"],
        audio_path=audio_path,
        output_dir=tmp_path,
        job_id="test_progress_job",
        progress_callback=lambda frac, msg: seen.append(frac),
    )

    assert Path(res["video_path"]).exists()
    assert Path(res["video_path"]).stat().st_size > 0
    assert len(seen) >= 1  # progress hook fired during encode
    assert all(0.0 <= f <= 1.0 for f in seen)
