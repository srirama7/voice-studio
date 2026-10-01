import sys
import numpy as np
import scipy.signal as signal
from pathlib import Path
sys.path.insert(0, '.')
from voice_clone import EdgeTTSAdapter

def ultra_human_dsp(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio is None or len(audio) == 0:
        return audio
    
    x = audio.astype(np.float64)
    n = len(x)

    # 1. Harmonic Warmth (Subtle 2nd-order vocal cord warmth saturation)
    sat = x + 0.04 * (x ** 2) - 0.01 * (x ** 3)

    # 2. Dual-Band Chest Resonance & Vocal Clarity FFT Filter
    fft_data = np.fft.rfft(sat)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    gain = np.ones_like(freqs)

    # Chest warmth (140-380 Hz)
    chest_mask = (freqs >= 140) & (freqs <= 380)
    gain[chest_mask] *= 1.35

    # Vocal clarity & presence (2200-3800 Hz)
    clarity_mask = (freqs >= 2200) & (freqs <= 3800)
    gain[clarity_mask] *= 1.15

    # De-harshness filter (> 6200 Hz)
    deharsh_mask = freqs > 6200
    gain[deharsh_mask] *= 0.88

    fft_data *= gain
    morphed = np.fft.irfft(fft_data, n=n)

    # 3. Micro-jitter / Vocal cord vibrato (0.15% subtle pitch fluctuation at 5.5 Hz)
    t = np.arange(n) / sr
    vibrato_phase = 0.0003 * np.sin(2.0 * np.pi * 5.5 * t)
    indices = np.clip(t * sr + vibrato_phase * sr, 0, n - 1)
    morphed = np.interp(indices, np.arange(n), morphed)

    # 4. Soft Limiting & Broadcast Peak Normalization
    max_val = np.max(np.abs(morphed))
    if max_val > 1e-4:
        morphed = (morphed / max_val) * 0.94

    return morphed.astype(np.float32)

adapter = EdgeTTSAdapter(forced_voice="en-IN-PrabhatNeural")
raw_audio, sr = adapter.synthesize("Hello! This is an advanced humanized speech test with vocal warmth, chest resonance, and conversational prosody.")
humanized = ultra_human_dsp(raw_audio, sr)

print("Raw Audio Len:", len(raw_audio))
print("Humanized Audio Len:", len(humanized))
print("Peak Max:", np.max(np.abs(humanized)))
