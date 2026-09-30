"""TTS Adapter Module for Voice Studio.

Provides abstract TTSAdapter base class, Coqui XTTS-v2 adapter,
Kokoro-82M fallback adapter, and ThreadPoolExecutor parallel batch synthesis.
"""

from __future__ import annotations

import abc
import concurrent.futures
import importlib.util
import logging
import math
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D float audio array from orig_sr to target_sr.

    Uses scipy.signal.resample if available, otherwise linear interpolation.

    Args:
        audio: 1D numpy array of audio samples.
        orig_sr: Source sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled 1D numpy float32 array.
    """
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32)

    num_target_samples = int(round(len(audio) * target_sr / orig_sr))

    # Try scipy.signal if installed
    try:
        from scipy import signal
        resampled = signal.resample(audio, num_target_samples)
        return resampled.astype(np.float32)
    except ImportError:
        # Fallback to linear interpolation using numpy
        orig_indices = np.linspace(0, len(audio) - 1, num=len(audio))
        target_indices = np.linspace(0, len(audio) - 1, num=num_target_samples)
        resampled = np.interp(target_indices, orig_indices, audio)
        return resampled.astype(np.float32)


class TTSAdapter(abc.ABC):
    """Abstract base class for Text-to-Speech synthesis adapters."""

    @property
    @abc.abstractmethod
    def adapter_name(self) -> str:
        """Return human-readable identifier for the adapter."""
        pass

    @property
    @abc.abstractmethod
    def sample_rate(self) -> int:
        """Default target sample rate for this adapter (e.g. 24000)."""
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the underlying dependencies and models are installed/ready."""
        pass

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ) -> Tuple[np.ndarray, int]:
        """Synthesize text string to audio waveform array.

        Args:
            text: Cleaned text string to synthesize.
            speaker_wav: Optional path to reference speaker audio for voice cloning.
            language: ISO language code (e.g. 'en', 'es').

        Returns:
            Tuple of (audio_waveform, sample_rate), where audio_waveform is a 1D float32 array in [-1.0, 1.0].
        """
        pass

    def _generate_fallback_audio(self, text: str, duration_per_char: float = 0.06) -> Tuple[np.ndarray, int]:
        """Generate a clean synthetic placeholder audio tone sequence.

        WARNING: this is a sine-beep placeholder, NOT speech and NOT the
        enrolled voice. It is only used when no real TTS engine is
        available. Callers should surface this loudly instead of silently
        delivering beeps as if they were voice output.

        Args:
            text: Text being synthesized.
            duration_per_char: Approximate duration per character in seconds.

        Returns:
            Tuple of (synthetic_audio_array, sample_rate).
        """
        logger.error(
            "NO TTS ENGINE AVAILABLE -- generating sine-beep placeholder for "
            "'%s...'. Install edge-tts (online) or use SAPI (offline) for real speech.",
            text[:40],
        )
        sr = self.sample_rate
        duration = max(0.5, min(10.0, len(text) * duration_per_char))
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Soft multi-harmonic tone (A440 + harmonics with decay envelope)
        audio = 0.2 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)
        envelope = np.exp(-t / duration)
        audio = audio * envelope
        return audio.astype(np.float32), sr


class CoquiXTTSAdapter(TTSAdapter):
    """Adapter for Coqui XTTS-v2 voice synthesis engine."""

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "cpu",
        use_deepspeed: bool = False,
    ) -> None:
        """Initialize Coqui XTTS Adapter.

        Args:
            model_name: Model identifier or local directory path.
            device: Computing device ('cpu' or 'cuda').
            use_deepspeed: Enable DeepSpeed acceleration if available.
        """
        self._model_name = model_name
        self._device = device
        self._use_deepspeed = use_deepspeed
        self._model = None
        self._sr = 24000

    @property
    def adapter_name(self) -> str:
        return "Coqui-XTTS-v2"

    @property
    def sample_rate(self) -> int:
        return self._sr

    def is_available(self) -> bool:
        """Check whether Coqui TTS framework is installed."""
        return importlib.util.find_spec("TTS") is not None and importlib.util.find_spec("torch") is not None

    def _load_model(self) -> None:
        """Lazily load Coqui TTS model into memory."""
        if self._model is not None:
            return

        if not self.is_available():
            raise ImportError("Coqui TTS or PyTorch package is not installed.")

        try:
            from TTS.api import TTS
            logger.info("Loading Coqui XTTS-v2 model: %s on %s...", self._model_name, self._device)
            self._model = TTS(model_name=self._model_name, progress_bar=False).to(self._device)
            logger.info("Coqui XTTS-v2 model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load Coqui XTTS model: %s", exc)
            raise RuntimeError(f"Coqui XTTS model loading error: {exc}") from exc

    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ) -> Tuple[np.ndarray, int]:
        """Synthesize text using Coqui XTTS-v2.

        Args:
            text: Input text string.
            speaker_wav: Path to voice cloning sample audio file.
            language: ISO language code.

        Returns:
            Tuple of (audio_waveform, sample_rate).
        """
        if not text.strip():
            return np.zeros(0, dtype=np.float32), self.sample_rate

        try:
            self._load_model()
            wav = self._model.tts(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
            )
            audio = np.array(wav, dtype=np.float32)
            # Normalize peak if necessary
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val
            return audio, self.sample_rate
        except Exception as exc:
            logger.warning("Coqui XTTS synthesis failed (%s). Falling back to synthetic stub audio.", exc)
            return self._generate_fallback_audio(text)


class KokoroTTSAdapter(TTSAdapter):
    """Adapter for Kokoro-82M lightweight fallback synthesis engine."""

    def __init__(self, model_path: Optional[str] = None, voices_path: Optional[str] = None) -> None:
        """Initialize Kokoro TTS Adapter.

        Args:
            model_path: Path to kokoro ONNX or PyTorch weights file.
            voices_path: Path to voice embedding JSON/npy file.
        """
        self.model_path = model_path
        self.voices_path = voices_path
        self._sr = 24000
        self._kokoro_pipeline = None

    @property
    def adapter_name(self) -> str:
        return "Kokoro-82M"

    @property
    def sample_rate(self) -> int:
        return self._sr

    def is_available(self) -> bool:
        """Check whether Kokoro module or kokoro_onnx is available."""
        return (
            importlib.util.find_spec("kokoro") is not None
            or importlib.util.find_spec("kokoro_onnx") is not None
        )

    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ) -> Tuple[np.ndarray, int]:
        """Synthesize text using Kokoro-82M.

        Args:
            text: Input text string.
            speaker_wav: Optional reference speaker audio or voice identifier.
            language: ISO language code.

        Returns:
            Tuple of (audio_waveform, sample_rate).
        """
        if not text.strip():
            return np.zeros(0, dtype=np.float32), self.sample_rate

        if self.is_available():
            try:
                # Attempt kokoro_onnx or kokoro package
                if importlib.util.find_spec("kokoro_onnx") is not None:
                    from kokoro_onnx import Kokoro
                    kokoro = Kokoro(self.model_path or "kokoro-v0_19.onnx", self.voices_path or "voices.json")
                    samples, sr = kokoro.create(text, voice="af_sarah", speed=1.0, lang=language)
                    return np.array(samples, dtype=np.float32), int(sr)
                elif importlib.util.find_spec("kokoro") is not None:
                    from kokoro import KPipeline
                    pipeline = KPipeline(lang_code=language[0] if language else "a")
                    generator = pipeline(text, voice="af_heart", speed=1.0)
                    all_audio = []
                    for _, _, audio in generator:
                        all_audio.append(audio)
                    if all_audio:
                        combined = np.concatenate(all_audio)
                        return combined.astype(np.float32), self.sample_rate
            except Exception as exc:
                logger.warning("Kokoro TTS synthesis execution error: %s. Falling back to stub.", exc)

        # Standalone fallback if package not present or runtime error occurred
        return self._generate_fallback_audio(text)


class BatchTTSSynthesizer:
    """Batch TTS Synthesizer using ThreadPoolExecutor for concurrent generation."""

    def __init__(
        self,
        primary_adapter: TTSAdapter,
        fallback_adapter: Optional[TTSAdapter] = None,
        max_workers: int = 4,
    ) -> None:
        """Initialize BatchTTSSynthesizer.

        Args:
            primary_adapter: Primary TTSAdapter instance.
            fallback_adapter: Optional secondary fallback TTSAdapter instance.
            max_workers: Maximum parallel workers in ThreadPoolExecutor.
        """
        self.primary_adapter = primary_adapter
        self.fallback_adapter = fallback_adapter or KokoroTTSAdapter()
        self.max_workers = max_workers

    def _synthesize_chunk_with_fallback(
        self,
        index: int,
        chunk_text: str,
        speaker_wav: Optional[str],
        language: str,
    ) -> Tuple[int, np.ndarray, int]:
        """Synthesize a single chunk using primary adapter, with automatic fallback.

        Args:
            index: Chunk sequence index.
            chunk_text: Text string of the chunk.
            speaker_wav: Optional speaker reference WAV.
            language: Language code.

        Returns:
            Tuple of (index, audio_array, sample_rate).
        """
        try:
            audio, sr = self.primary_adapter.synthesize(chunk_text, speaker_wav=speaker_wav, language=language)
            if len(audio) > 0:
                return index, audio, sr
            raise ValueError("Primary adapter produced empty audio output.")
        except Exception as exc:
            logger.warning(
                "Primary adapter %s failed for chunk %d ('%s...'): %s. Using fallback adapter %s.",
                self.primary_adapter.adapter_name,
                index,
                chunk_text[:30],
                exc,
                self.fallback_adapter.adapter_name,
            )
            audio, sr = self.fallback_adapter.synthesize(chunk_text, speaker_wav=speaker_wav, language=language)
            return index, audio, sr

    def synthesize_batch(
        self,
        chunks: List[str],
        speaker_wav: Optional[str] = None,
        language: str = "en",
        pause_duration_s: float = 0.2,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize multiple text chunks in parallel and concatenate results.

        Args:
            chunks: List of text string chunks to synthesize.
            speaker_wav: Optional path to reference speaker WAV file.
            language: Language code.
            pause_duration_s: Silence duration between chunks in seconds.

        Returns:
            Tuple of (concatenated_audio_array, target_sample_rate).
        """
        if not chunks:
            target_sr = self.primary_adapter.sample_rate
            return np.zeros(0, dtype=np.float32), target_sr

        target_sr = self.primary_adapter.sample_rate
        pause_samples = int(round(pause_duration_s * target_sr))
        silence_padding = np.zeros(pause_samples, dtype=np.float32)

        results: List[Tuple[int, np.ndarray, int]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(
                    self._synthesize_chunk_with_fallback, i, chunk, speaker_wav, language
                ): i
                for i, chunk in enumerate(chunks)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                res = future.result()
                results.append(res)

        # Sort by original chunk index to maintain text order
        results.sort(key=lambda x: x[0])

        # Resample and concatenate all audio segments
        audio_segments: List[np.ndarray] = []
        for i, (_, audio, sr) in enumerate(results):
            if len(audio) == 0:
                continue
            if sr != target_sr:
                audio = resample_audio(audio, sr, target_sr)

            audio_segments.append(audio)
            if i < len(results) - 1 and pause_samples > 0:
                audio_segments.append(silence_padding)

        if not audio_segments:
            return np.zeros(0, dtype=np.float32), target_sr

        full_audio = np.concatenate(audio_segments)
        return full_audio.astype(np.float32), target_sr
