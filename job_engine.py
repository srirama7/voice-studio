"""Job Engine State Machine & Checkpointing Module for Voice Studio.

Implements state transitions:
QUEUED -> PARSED -> CLEANED -> SYNTHESIZING -> MASTERING -> RENDERING -> COMPLETED

Features:
- Deterministic SHA256 chunk audio cache lookup.
- Persistent checkpoint.json state tracking and manifest.json final output packaging.
- Crash recovery algorithm resuming from last verified state without re-running finished steps.
- Progress reporting callbacks for web UI integration.
"""

from __future__ import annotations

import concurrent.futures
import enum
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import wave

from config import Config, default_config
from document_parser import DocumentParser, ParsedDocument, SlideData
from text_processor import TextProcessor, TextProcessingMode
from audio_dsp import AudioDSPPipeline, normalize_youtube_standard
from tts_adapter import BatchTTSSynthesizer, CoquiXTTSAdapter, KokoroTTSAdapter, TTSAdapter

logger = logging.getLogger(__name__)


class JobStatus(str, enum.Enum):
    """Execution state machine statuses for voice generation jobs."""
    QUEUED = "QUEUED"
    PARSED = "PARSED"
    CLEANED = "CLEANED"
    SYNTHESIZING = "SYNTHESIZING"
    MASTERING = "MASTERING"
    RENDERING = "RENDERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


ProgressCallback = Callable[[JobStatus, float, str], None]


@dataclass
class JobCheckpoint:
    """Serializable job state checkpoint representation."""
    job_id: str
    document_path: str
    voice_id: str
    text_mode: str = "exact"
    language: str = "en-in"
    gallery_voice: str = ""  # gallery voice id (e.g. 'prabhat_in_m'); '' = cloned voice
    status: JobStatus = JobStatus.QUEUED
    current_step_index: int = 0
    completed_steps: List[str] = field(default_factory=list)
    parsed_slides: List[Dict] = field(default_factory=list)
    cleaned_chunks: List[str] = field(default_factory=list)
    synthesized_chunk_files: List[str] = field(default_factory=list)
    mastered_audio_path: Optional[str] = None
    subtitle_path: Optional[str] = None
    video_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["status"] = self.status.value if isinstance(self.status, enum.Enum) else str(self.status)
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> JobCheckpoint:
        data_copy = dict(data)
        if "status" in data_copy:
            data_copy["status"] = JobStatus(data_copy["status"])
        return cls(**data_copy)


@dataclass
class JobManifest:
    """Final packaged output manifest representation."""
    job_id: str
    status: JobStatus = JobStatus.COMPLETED
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    total_duration_sec: float = 0.0
    slide_count: int = 0
    chunk_count: int = 0
    gallery_voice: str = ""
    document_file: str = ""
    voice_id: str = ""
    mastered_audio_file: str = ""
    subtitle_file: str = ""
    video_file: str = ""

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["status"] = self.status.value if isinstance(self.status, enum.Enum) else str(self.status)
        return data


def human_pause(base_gap: float, job_id: str, pos: int, jitter: float = 0.15) -> float:
    """Humanize a fixed inter-chunk pause with deterministic jitter.

    Real speakers never pause exactly 0.45s every sentence; fixed gaps sound
    metronomic. The jitter (±15% default) is hashed from (job_id, pos) so a
    resumed/re-run job produces byte-identical output (cache-safe).
    """
    h = int(hashlib.sha256(f"{job_id}|{pos}".encode("utf-8")).hexdigest()[:8], 16)
    factor = 1.0 - jitter + (h % 1000) / 1000.0 * (2.0 * jitter)
    return base_gap * factor


class ChunkCache:
    """SHA256 deterministic chunk audio cache manager."""

    def __init__(self, cache_dir: Union[str, Path]) -> None:
        self.cache_dir = Path(cache_dir)
        self.chunks_dir = self.cache_dir / "audio_chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

    def compute_hash(self, text: str, voice_id: str, extra_key: str = "") -> str:
        """Compute SHA256 deterministic hash for a given chunk of text and voice profile."""
        payload = f"{text.strip()}|{voice_id.strip()}|{extra_key}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get_chunk_path(self, chunk_hash: str) -> Path:
        """Get file path for a cached audio chunk hash."""
        return self.chunks_dir / f"{chunk_hash}.wav"

    def is_cached(self, chunk_hash: str) -> bool:
        """Check if audio chunk is already synthesized and stored in cache."""
        path = self.get_chunk_path(chunk_hash)
        return path.exists() and path.stat().st_size > 0

    def put_chunk(self, chunk_hash: str, audio: np.ndarray, sample_rate: int) -> Path:
        """Save chunk audio waveform float array into WAV file in cache."""
        out_path = self.get_chunk_path(chunk_hash)
        pcm_data = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data.tobytes())
        return out_path

    def get_chunk(self, chunk_hash: str) -> Tuple[np.ndarray, int]:
        """Read cached chunk WAV file into float32 numpy array and sample rate."""
        chunk_path = self.get_chunk_path(chunk_hash)
        if not chunk_path.exists():
            raise FileNotFoundError(f"Cached chunk audio not found: {chunk_path}")

        with wave.open(str(chunk_path), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)
            audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return audio, sr


class JobEngine:
    """Stateful workflow orchestrator managing parsing, synthesis, DSP, video, and recovery."""

    def __init__(
        self,
        config: Optional[Config] = None,
        tts_synthesizer: Optional[BatchTTSSynthesizer] = None,
    ) -> None:
        self.config = config or default_config
        self.chunk_cache = ChunkCache(self.config.paths.cache_dir)
        self.text_processor = TextProcessor()
        self.dsp_pipeline = AudioDSPPipeline(sample_rate=self.config.audio.sample_rate)

        if tts_synthesizer is None:
            try:
                from voice_clone import CloningTTSAdapter, SapiTTSAdapter
                primary = CloningTTSAdapter()  # Edge neural + DSP morph to ref voice
                fallback = SapiTTSAdapter()    # offline Windows SAPI backup
            except Exception:
                primary = CoquiXTTSAdapter()
                fallback = KokoroTTSAdapter()
            self.tts_synthesizer = BatchTTSSynthesizer(primary, fallback)
        else:
            self.tts_synthesizer = tts_synthesizer

    def resolve_voice_ref(self, voice_id: str) -> Optional[str]:
        """Resolve the enrolled ref.wav for a voice_id across ALL versions.

        Previously only ``voices/{id}/v001/ref.wav`` was checked, so voices
        enrolled under any other version silently fell back to ``None``
        (no cloning). Now the newest version containing ref.wav wins.
        """
        base = self.config.paths.voices_dir / voice_id
        if not base.exists():
            return None
        candidates = sorted(base.glob("*/ref.wav"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return str(candidates[0])
        legacy = base / "ref.wav"
        if legacy.exists():
            return str(legacy)
        return None

    def engine_cache_key(self, voice_ref_wav: Optional[str], language: str = "en",
                           gallery_voice: str = "") -> str:
        """Cache-busting key: engine name + ref mtime + language.

        The old key (text|voice_id) reused stale sine-beep chunks even after
        the cloning engine was fixed. Including the engine + ref mtime forces
        re-synthesis with the real voice.
        """
        try:
            engine = self.tts_synthesizer.primary_adapter.adapter_name
        except Exception:
            engine = "unknown"
        mtime = ""
        if voice_ref_wav and Path(voice_ref_wav).exists():
            try:
                mtime = str(int(Path(voice_ref_wav).stat().st_mtime))
            except Exception:
                pass
        return f"{engine}|{mtime}|{language}|{gallery_voice}"

    def get_job_dir(self, job_id: str) -> Path:
        """Get or create directory for job checkpoints and intermediate artifacts."""
        job_dir = self.config.paths.checkpoint_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def get_checkpoint_path(self, job_id: str) -> Path:
        """Get path to job checkpoint JSON file."""
        return self.get_job_dir(job_id) / "checkpoint.json"

    def get_manifest_path(self, job_id: str) -> Path:
        """Get path to job manifest JSON file."""
        return self.get_job_dir(job_id) / "manifest.json"

    def load_checkpoint(self, job_id: str) -> Optional[JobCheckpoint]:
        """Load job checkpoint if it exists on disk."""
        ckpt_path = self.get_checkpoint_path(job_id)
        if not ckpt_path.exists():
            return None
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return JobCheckpoint.from_dict(data)
        except Exception as exc:
            logger.error("Failed to load checkpoint for job %s: %s", job_id, exc)
            return None

    def save_checkpoint(self, checkpoint: JobCheckpoint) -> None:
        """Save job checkpoint atomically to disk."""
        checkpoint.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        ckpt_path = self.get_checkpoint_path(checkpoint.job_id)
        temp_path = ckpt_path.with_suffix(".tmp")

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        for attempt in range(5):
            try:
                temp_path.replace(ckpt_path)
                break
            except (PermissionError, OSError):
                if attempt == 4:
                    with open(ckpt_path, "w", encoding="utf-8") as f:
                        json.dump(checkpoint.to_dict(), f, indent=2)
                    if temp_path.exists():
                        try:
                            temp_path.unlink()
                        except Exception:
                            pass
                else:
                    time.sleep(0.02)

    def save_manifest(self, manifest: JobManifest) -> Path:
        """Save final job manifest JSON file to disk."""
        path = self.get_manifest_path(manifest.job_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        # Also write to output directory for distribution
        output_manifest_path = self.config.paths.output_dir / manifest.job_id / "manifest.json"
        output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        return path

    def create_job(
        self,
        job_id: str,
        document_path: str,
        voice_id: str,
        text_mode: str = "exact",
        language: str = "en-in",
        gallery_voice: str = "",
    ) -> JobCheckpoint:
        """Initialize a new job checkpoint in QUEUED state."""
        checkpoint = JobCheckpoint(
            job_id=job_id,
            document_path=str(document_path),
            voice_id=voice_id,
            text_mode=text_mode,
            language=language,
            gallery_voice=gallery_voice,
            status=JobStatus.QUEUED,
        )
        self.save_checkpoint(checkpoint)
        return checkpoint

    def run_job(
        self,
        job_id: str,
        document_path: Optional[str] = None,
        voice_id: Optional[str] = None,
        text_mode: str = "exact",
        language: str = "en-in",
        gallery_voice: str = "",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> JobManifest:
        """Run or resume a voice studio document processing job.

        Executes state machine:
        QUEUED -> PARSED -> CLEANED -> SYNTHESIZING -> MASTERING -> RENDERING -> COMPLETED
        Resumes seamlessly if interrupted or crashed.
        """
        checkpoint = self.load_checkpoint(job_id)
        if checkpoint is None:
            if not document_path or not voice_id:
                raise ValueError(f"Job {job_id} does not exist and document_path/voice_id were not supplied.")
            checkpoint = self.create_job(job_id, document_path, voice_id, text_mode, language, gallery_voice)

        if checkpoint.status == JobStatus.COMPLETED:
            manifest_path = self.get_manifest_path(job_id)
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    return JobManifest(**json.load(f))

        def notify(status: JobStatus, pct: float, msg: str) -> None:
            checkpoint.status = status
            self.save_checkpoint(checkpoint)
            logger.info("Job [%s] Status: %s (%.1f%%) - %s", job_id, status.value, pct * 100, msg)
            if progress_callback:
                try:
                    progress_callback(status, pct, msg)
                except Exception as cb_err:
                    logger.warning("Progress callback error: %s", cb_err)

        job_dir = self.get_job_dir(job_id)
        t_job_start = time.time()

        try:
            # Stage 1: PARSED
            if JobStatus.PARSED.value not in checkpoint.completed_steps:
                notify(JobStatus.QUEUED, 0.05, "Parsing input document...")
                parsed_doc = DocumentParser.parse(checkpoint.document_path)
                checkpoint.parsed_slides = [s.to_dict() for s in parsed_doc.slides]
                checkpoint.completed_steps.append(JobStatus.PARSED.value)
                notify(JobStatus.PARSED, 0.20, f"Parsed {len(parsed_doc.slides)} slides from document.")

            # Stage 2: CLEANED
            if JobStatus.CLEANED.value not in checkpoint.completed_steps:
                notify(JobStatus.CLEANED, 0.25, "Cleaning slide text...")
                mode = TextProcessingMode.NARRATIVE if checkpoint.text_mode == "narrative" else TextProcessingMode.EXACT
                cleaned_chunks: List[str] = []

                for slide in checkpoint.parsed_slides:
                    # Construct text for slide
                    notes = slide.get("speaker_notes", "").strip()
                    title = slide.get("title", "").strip()
                    body = slide.get("body_text", "").strip()

                    raw_text = notes if notes else f"{title}. {body}".strip()
                    cleaned_text = self.text_processor.process(raw_text, mode=mode, use_llm=False)
                    chunks = self.text_processor.chunk_text(cleaned_text, max_chars=800)
                    cleaned_chunks.extend(chunks if chunks else [cleaned_text])

                checkpoint.cleaned_chunks = [c for c in cleaned_chunks if c.strip()]
                # Merge tiny consecutive chunks (each costs a full TTS round-trip):
                # pack to ~200-800 chars so small docs don't pay N x 3s latency.
                merged, buf = [], ""
                for c in checkpoint.cleaned_chunks:
                    if len(buf) + len(c) + 1 <= 800:
                        buf = (buf + " " + c).strip()
                    else:
                        if buf:
                            merged.append(buf)
                        buf = c
                if buf:
                    merged.append(buf)
                checkpoint.cleaned_chunks = merged
                checkpoint.completed_steps.append(JobStatus.CLEANED.value)
                notify(JobStatus.CLEANED, 0.40, f"Cleaned text into {len(checkpoint.cleaned_chunks)} chunks.")

            # Stage 3: SYNTHESIZING
            if JobStatus.SYNTHESIZING.value not in checkpoint.completed_steps:
                notify(JobStatus.SYNTHESIZING, 0.45, "Synthesizing audio chunks...")
                chunk_files: List[str] = list(checkpoint.synthesized_chunk_files)

                # Look up voice reference file if available (any enrolled version)
                voice_ref_wav: Optional[str] = self.resolve_voice_ref(checkpoint.voice_id)
                if voice_ref_wav is None:
                    logger.warning(
                        "No enrolled ref.wav found for voice '%s' -- output will use "
                        "base voice without cloning. Enroll the voice in Tab 1 first.",
                        checkpoint.voice_id,
                    )
                engine_key = self.engine_cache_key(voice_ref_wav, checkpoint.language,
                                                    checkpoint.gallery_voice)

                # Gallery mode: direct stock voice, NO morph DSP (max naturalness).
                gallery_adapter = None
                if checkpoint.gallery_voice:
                    from voice_clone import GALLERY_VOICES, EdgeTTSAdapter
                    info = GALLERY_VOICES.get(checkpoint.gallery_voice, {})
                    if not info:
                        raise ValueError(f"Unknown gallery voice '{checkpoint.gallery_voice}'.")
                    gallery_adapter = EdgeTTSAdapter(
                        forced_voice=info["voice"],
                        forced_rate=info.get("rate", "+0%"),
                        forced_pitch=info.get("pitch", "+0Hz"),
                    )

                total_chunks = len(checkpoint.cleaned_chunks)
                while len(chunk_files) < total_chunks:
                    chunk_files.append("")

                # Cache-hit pass (cheap, serial); collect misses for parallel synthesis.
                pending: Dict[int, str] = {}  # chunk idx -> hash needing synthesis
                for idx, chunk_text in enumerate(checkpoint.cleaned_chunks):
                    if chunk_files[idx] and Path(chunk_files[idx]).exists():
                        continue  # Already synthesized
                    chunk_hash = self.chunk_cache.compute_hash(
                        chunk_text, checkpoint.voice_id, engine_key)
                    if self.chunk_cache.is_cached(chunk_hash):
                        chunk_files[idx] = str(self.chunk_cache.get_chunk_path(chunk_hash))
                    else:
                        pending[idx] = chunk_hash

                def _synth_one(idx: int, text: str):
                    # Retry transient EdgeTTS network blips (2 retries, backoff)
                    # so one failed chunk doesn't kill a 60-chunk job.
                    last_exc: Optional[Exception] = None
                    for attempt in range(3):
                        try:
                            if gallery_adapter is not None:
                                audio, sr = gallery_adapter.synthesize(
                                    text, speaker_wav=None, language=checkpoint.language
                                )
                            else:
                                audio, sr = self.tts_synthesizer.primary_adapter.synthesize(
                                    text, speaker_wav=voice_ref_wav, language=checkpoint.language
                                )
                            if len(audio) > 0:
                                return audio, sr
                            raise ValueError("empty audio")
                        except Exception as exc:
                            last_exc = exc
                            logger.warning("Chunk %d attempt %d failed: %s",
                                           idx, attempt + 1, exc)
                            time.sleep(2 * (attempt + 1))
                    audio, sr = self.tts_synthesizer.fallback_adapter.synthesize(
                        text, speaker_wav=voice_ref_wav, language=checkpoint.language
                    )
                    if len(audio) == 0:
                        raise RuntimeError(f"Chunk {idx} failed: {last_exc}")
                    return audio, sr

                if pending:
                    synth_t0 = time.time()
                    workers = max(1, self.config.engine.max_parallel_workers)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                        futs = {ex.submit(_synth_one, i, checkpoint.cleaned_chunks[i]): i
                                for i in pending}
                        done = 0
                        for fut in concurrent.futures.as_completed(futs):
                            idx = futs[fut]
                            audio, sr = fut.result()
                            chunk_files[idx] = str(
                                self.chunk_cache.put_chunk(pending[idx], audio, sr))
                            done += 1
                            checkpoint.synthesized_chunk_files = chunk_files
                            self.save_checkpoint(checkpoint)
                            finished = (total_chunks - len(pending)) + done
                            pct = 0.45 + (0.25 * finished / max(1, total_chunks))
                            notify(JobStatus.SYNTHESIZING, pct,
                                   f"Synthesized chunk {finished}/{total_chunks}")
                    logger.info("Synthesized %d chunks in %.1fs (%d workers).",
                                len(pending), time.time() - synth_t0, workers)
                else:
                    checkpoint.synthesized_chunk_files = chunk_files
                    self.save_checkpoint(checkpoint)

                checkpoint.completed_steps.append(JobStatus.SYNTHESIZING.value)
                notify(JobStatus.SYNTHESIZING, 0.70, "All chunks synthesized successfully.")

            # Stage 4: MASTERING
            if JobStatus.MASTERING.value not in checkpoint.completed_steps:
                notify(JobStatus.MASTERING, 0.72, "Mastering audio pipeline...")
                t_master = time.time()
                all_audio_segments = []
                target_sr = self.config.audio.sample_rate

                for pos, chunk_file in enumerate(checkpoint.synthesized_chunk_files):
                    path = Path(chunk_file)
                    if not path.exists():
                        continue
                    with wave.open(str(path), "rb") as wf:
                        n_frames = wf.getnframes()
                        raw_bytes = wf.readframes(n_frames)
                        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        # 25ms raised-cosine fades kill clicks/chop at joins
                        nf = min(len(audio) // 4, int(target_sr * 0.025))
                        if nf > 8:
                            ramp = (0.5 * (1.0 - np.cos(np.pi * np.arange(nf) / nf))).astype(np.float32)
                            audio[:nf] *= ramp
                            audio[-nf:] *= ramp[::-1]
                        all_audio_segments.append(audio)
                        # Natural pause from boundary punctuation: full stop
                        # breathes longer, comma shorter (kills robot staccato).
                        # Jittered deterministically per (job, position) so no
                        # two pauses are identical -- like a human speaker.
                        txt = checkpoint.cleaned_chunks[pos].strip() if pos < len(checkpoint.cleaned_chunks) else ""
                        if txt and txt[-1] in ".!?\u0964":
                            gap = 0.45
                        elif txt and txt[-1] in ",;:\u2014-":
                            gap = 0.20
                        else:
                            gap = 0.30
                        gap = human_pause(gap, job_id, pos)
                        silence = np.zeros(int(target_sr * gap), dtype=np.float32)
                        all_audio_segments.append(silence)

                if all_audio_segments:
                    raw_full_audio = np.concatenate(all_audio_segments)
                else:
                    raw_full_audio = np.zeros(int(target_sr * 1.0), dtype=np.float32)

                mastered_audio = self.dsp_pipeline.process(
                    raw_full_audio,
                    target_lufs=self.config.audio.target_lufs,
                    max_dbtp=self.config.audio.max_true_peak_dbtp,
                )

                mastered_file = job_dir / "mastered.wav"
                pcm_data = np.clip(mastered_audio * 32767.0, -32768, 32767).astype(np.int16)
                with wave.open(str(mastered_file), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(target_sr)
                    wf.writeframes(pcm_data.tobytes())

                checkpoint.mastered_audio_path = str(mastered_file)
                checkpoint.completed_steps.append(JobStatus.MASTERING.value)
                logger.info("Mastering finished in %.1fs.", time.time() - t_master)
                notify(JobStatus.MASTERING, 0.85, "Mastered audio saved to WAV.")

            # Stage 5: RENDERING
            if JobStatus.RENDERING.value not in checkpoint.completed_steps:
                notify(JobStatus.RENDERING, 0.87, "Assembling video and generating SRT subtitles...")
                t_render = time.time()

                from video_assembly import VideoAssembly

                def _render_progress(frac: float, msg: str) -> None:
                    frac = max(0.0, min(1.0, frac))
                    notify(JobStatus.RENDERING, 0.87 + 0.08 * frac,
                           f"Rendering video... {msg}")

                assembly_result = VideoAssembly.assemble(
                    slides_data=checkpoint.parsed_slides,
                    cleaned_chunks=checkpoint.cleaned_chunks,
                    audio_path=checkpoint.mastered_audio_path,
                    output_dir=job_dir,
                    job_id=job_id,
                    progress_callback=_render_progress,
                )

                checkpoint.subtitle_path = assembly_result.get("subtitle_path")
                checkpoint.video_path = assembly_result.get("video_path")
                checkpoint.completed_steps.append(JobStatus.RENDERING.value)
                logger.info("Rendering finished in %.1fs.", time.time() - t_render)
                notify(JobStatus.RENDERING, 0.95, "Video rendering and subtitle generation completed.")

            # Stage 6: COMPLETED
            checkpoint.status = JobStatus.COMPLETED
            if JobStatus.COMPLETED.value not in checkpoint.completed_steps:
                checkpoint.completed_steps.append(JobStatus.COMPLETED.value)
            self.save_checkpoint(checkpoint)

            total_duration = 0.0
            if checkpoint.mastered_audio_path and Path(checkpoint.mastered_audio_path).exists():
                with wave.open(checkpoint.mastered_audio_path, "rb") as wf:
                    total_duration = wf.getnframes() / float(wf.getframerate())

            manifest = JobManifest(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                total_duration_sec=round(total_duration, 2),
                slide_count=len(checkpoint.parsed_slides),
                chunk_count=len(checkpoint.cleaned_chunks),
                document_file=checkpoint.document_path,
                voice_id=checkpoint.voice_id,
                gallery_voice=checkpoint.gallery_voice,
                mastered_audio_file=checkpoint.mastered_audio_path or "",
                subtitle_file=checkpoint.subtitle_path or "",
                video_file=checkpoint.video_path or "",
            )
            self.save_manifest(manifest)
            logger.info("Job %s finished in %.1fs total (%d slides, %d chunks).",
                        job_id, time.time() - t_job_start,
                        len(checkpoint.parsed_slides), len(checkpoint.cleaned_chunks))
            notify(JobStatus.COMPLETED, 1.0, "Job completed successfully.")
            return manifest

        except Exception as exc:
            logger.error("Job %s failed with exception: %s", job_id, exc, exc_info=True)
            checkpoint.status = JobStatus.FAILED
            checkpoint.error_message = str(exc)
            self.save_checkpoint(checkpoint)
            if progress_callback:
                try:
                    progress_callback(JobStatus.FAILED, 0.0, f"Error: {exc}")
                except Exception:
                    pass
            raise RuntimeError(f"Job execution failed for {job_id}: {exc}") from exc
