"""Unit tests for VoiceEnrollmentEngine module."""

import wave
from pathlib import Path
import numpy as np
import pytest

from config import Config
from voice_enrollment import QCResult, VoiceEnrollmentEngine, VoiceProfile


def create_dummy_wav(path: Path, duration_sec: float = 10.0, sample_rate: int = 22050, amplitude: float = 0.3) -> Path:
    """Helper to generate a valid 16-bit PCM WAV file with speech bursts and silence for QC testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    
    # 80% speech signal (burst), 20% background silence
    speech_mask = (t % 1.0) < 0.8
    signal = np.where(speech_mask, amplitude * np.sin(2 * np.pi * 440 * t), 0.0)
    pcm_data = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data.tobytes())

    return path


def test_analyze_quality_valid(tmp_path):
    cfg = Config()
    cfg.paths.base_dir = tmp_path
    cfg.ensure_directories()
    engine = VoiceEnrollmentEngine(config=cfg)

    wav_path = tmp_path / "good_voice.wav"
    create_dummy_wav(wav_path, duration_sec=10.0, amplitude=0.3)  # ~ -10.4 dBFS peak

    signal, sr = engine.load_audio_signal(wav_path)
    qc = engine.analyze_quality(signal, sr)

    assert isinstance(qc, QCResult)
    assert qc.is_valid is True
    assert 9.9 <= qc.duration_sec <= 10.1
    assert -18.0 <= qc.peak_dbfs <= -3.0
    assert qc.snr_db >= 10.0


def test_analyze_quality_duration_fail(tmp_path):
    cfg = Config()
    cfg.paths.base_dir = tmp_path
    cfg.ensure_directories()
    engine = VoiceEnrollmentEngine(config=cfg)

    # Short audio 3.0s < min 8.0s
    short_wav = tmp_path / "short_voice.wav"
    create_dummy_wav(short_wav, duration_sec=3.0, amplitude=0.3)

    signal, sr = engine.load_audio_signal(short_wav)
    qc = engine.analyze_quality(signal, sr)

    assert qc.is_valid is False
    assert any("duration" in err.lower() for err in qc.errors)


def test_enroll_voice_workflow(tmp_path):
    cfg = Config()
    cfg.paths.base_dir = tmp_path
    cfg.ensure_directories()
    engine = VoiceEnrollmentEngine(config=cfg)

    ref_wav = tmp_path / "input_ref.wav"
    create_dummy_wav(ref_wav, duration_sec=10.0, amplitude=0.3)

    profile = engine.enroll_voice(
        audio_path=ref_wav,
        voice_id="test_speaker_01",
        version="v001",
        strict_qc=True,
    )

    assert isinstance(profile, VoiceProfile)
    assert profile.voice_id == "test_speaker_01"
    assert profile.version == "v001"
    assert Path(profile.ref_wav_path).exists()
    assert Path(profile.conditioning_path).exists()
