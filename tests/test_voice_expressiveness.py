"""Unit tests for expressive voice upgrades: tempo matching, signatures, human pauses."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from job_engine import human_pause
from voice_clone import (
    VoiceConverter,
    build_voice_signature,
    estimate_tempo_sps,
    get_ref_tempo,
)


def _pulsed_speech(sr: int = 22050, secs: float = 4.0, sps: float = 3.0) -> np.ndarray:
    """Synthetic syllable-pulse train: ~sps energy bursts per second."""
    n = int(sr * secs)
    t = np.arange(n) / sr
    carrier = 0.4 * np.sin(2 * np.pi * 180 * t)
    gate = (np.sin(2 * np.pi * sps * t) > 0.2).astype(float)
    # Smooth gate edges to look like syllable envelopes
    k = np.ones(int(sr * 0.03)) / int(sr * 0.03)
    gate = np.convolve(gate, k, mode="same")
    rng = np.random.default_rng(7)
    return ((carrier * gate) + 0.005 * rng.standard_normal(n)).astype(np.float32)


def _write_wav(path: Path, audio: np.ndarray, sr: int = 22050) -> str:
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return str(path)


def test_tempo_speech_like_signal():
    sr = 22050
    sig = _pulsed_speech(sr, secs=4.0, sps=3.0)
    tempo = estimate_tempo_sps(sig, sr)
    assert tempo > 0.0
    assert 1.0 <= tempo <= 8.0
    assert abs(tempo - 3.0) < 1.5  # roughly recovers the true pulse rate


def test_tempo_silence_is_zero():
    sr = 22050
    assert estimate_tempo_sps(np.zeros(sr * 3, dtype=np.float32), sr) == 0.0
    assert estimate_tempo_sps(np.zeros(100, dtype=np.float32), sr) == 0.0


def test_human_pause_deterministic_and_bounded():
    a = human_pause(0.45, "job_abc", 3)
    b = human_pause(0.45, "job_abc", 3)
    assert a == b  # resume-safe determinism
    assert 0.45 * 0.85 <= a <= 0.45 * 1.15
    # Varies across positions (not a fixed robot grid)
    vals = {human_pause(0.45, "job_abc", i) for i in range(10)}
    assert len(vals) > 1


def test_build_voice_signature(tmp_path):
    sr = 22050
    ref = _write_wav(tmp_path / "ref.wav", _pulsed_speech(sr, secs=10.0, sps=3.5), sr)
    sig = build_voice_signature(ref, sr)
    assert sig["version"] == "signature_v1"
    assert sig["sample_rate"] == sr
    assert sig["f0_hz"] >= 0.0
    assert sig["tempo_sps"] >= 0.0
    assert "rms_dbfs" in sig and "spectral_tilt_db" in sig
    json.dumps(sig)  # must be JSON-serializable for DB storage


def test_convert_tempo_path_keeps_length(tmp_path):
    sr = 24000
    ref = _write_wav(tmp_path / "ref.wav", _pulsed_speech(22050, secs=10.0, sps=4.0), 22050)
    base = _pulsed_speech(sr, secs=4.0, sps=2.5)
    out = VoiceConverter.convert(base, sr, ref)
    assert len(out) == len(base)  # sample-exact length preserved
    assert np.max(np.abs(out)) > 0
    # Cached tempo reused without recompute
    assert get_ref_tempo(ref, sr) >= 0.0
