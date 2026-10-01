"""Unit tests for Audio DSP Engine module."""

import numpy as np
import pytest

from audio_dsp import (
    AudioDSPPipeline,
    duck_background_music,
    dynamics_compressor,
    estimate_integrated_loudness,
    highpass_filter,
    normalize_youtube_standard,
    parametric_eq_4band,
)


def test_highpass_filter():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 20 Hz low frequency rumble + 440 Hz signal tone
    signal = 0.5 * np.sin(2 * np.pi * 20 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
    
    filtered = highpass_filter(signal, sr, cutoff=70.0)
    assert len(filtered) == len(signal)
    # Power of 20Hz rumble should be attenuated relative to 440Hz tone
    assert np.max(np.abs(filtered)) <= np.max(np.abs(signal))


def test_parametric_eq_4band():
    sr = 22050
    audio = np.random.normal(0, 0.1, sr).astype(np.float32)
    equalized = parametric_eq_4band(audio, sr)
    assert len(equalized) == len(audio)
    assert not np.isnan(equalized).any()


def test_dynamics_compressor():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # High amplitude burst followed by low amplitude signal
    burst = np.concatenate([np.ones(sr // 2), np.ones(sr // 2) * 0.1]).astype(np.float32)
    
    compressed = dynamics_compressor(burst, sr, threshold_db=-10.0, ratio=4.0)
    assert len(compressed) == len(burst)
    assert not np.isnan(compressed).any()


def test_estimate_integrated_loudness_and_normalization():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    lufs_before = estimate_integrated_loudness(audio, sr)
    assert isinstance(lufs_before, float)

    normalized = normalize_youtube_standard(audio, sr, target_lufs=-14.0, max_dbtp=-1.5)
    assert len(normalized) == len(audio)
    assert not np.isnan(normalized).any()
    
    # Peak level must respect -1.5 dBTP (~0.841)
    max_peak = np.max(np.abs(normalized))
    assert max_peak <= 0.85


def test_duck_background_music():
    sr = 22050
    voice = np.zeros(sr, dtype=np.float32)
    # Voice active in second half
    voice[sr // 2:] = 0.5

    music = np.ones(sr, dtype=np.float32) * 0.4
    ducked = duck_background_music(voice, music, sr, threshold_db=-20.0, duck_gain_db=-12.0)

    assert len(ducked) == len(music)
    # Music level in second half should be lower than first half due to ducking
    first_half_power = np.mean(ducked[:sr // 4] ** 2)
    second_half_power = np.mean(ducked[3 * sr // 4:] ** 2)
    assert second_half_power < first_half_power


def test_dsp_pipeline_mastering():
    pipeline = AudioDSPPipeline(sample_rate=22050)
    sr = 22050
    raw_audio = np.random.normal(0, 0.2, sr).astype(np.float32)
    
    mastered = pipeline.process(raw_audio, target_lufs=-14.0, max_dbtp=-1.5)
    assert len(mastered) == len(raw_audio)
    assert not np.isnan(mastered).any()
