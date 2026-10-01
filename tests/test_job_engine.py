"""Unit tests for JobEngine and state machine crash recovery."""

import json
from pathlib import Path
import numpy as np
import pytest

from config import Config
from job_engine import ChunkCache, JobCheckpoint, JobEngine, JobManifest, JobStatus


def test_chunk_cache_sha256(tmp_path):
    cache = ChunkCache(tmp_path)
    h1 = cache.compute_hash("Hello world", "speaker_01")
    h2 = cache.compute_hash("Hello world", "speaker_01")
    h3 = cache.compute_hash("Different text", "speaker_01")

    assert h1 == h2
    assert h1 != h3

    assert not cache.is_cached(h1)

    audio = np.ones(22050, dtype=np.float32) * 0.1
    p = cache.put_chunk(h1, audio, 22050)
    assert p.exists()
    assert cache.is_cached(h1)

    read_audio, sr = cache.get_chunk(h1)
    assert sr == 22050
    assert len(read_audio) == 22050


def test_job_checkpoint_serialization():
    ckpt = JobCheckpoint(
        job_id="job_123",
        document_path="doc.pptx",
        voice_id="speaker_01",
        status=JobStatus.PARSED,
        completed_steps=["PARSED"],
    )
    data = ckpt.to_dict()
    assert data["status"] == "PARSED"

    restored = JobCheckpoint.from_dict(data)
    assert restored.job_id == "job_123"
    assert restored.status == JobStatus.PARSED


def test_job_engine_execution_and_crash_recovery(tmp_path):
    cfg = Config()
    cfg.paths.base_dir = tmp_path
    cfg.ensure_directories()

    doc_path = tmp_path / "test_presentation.txt"
    doc_path.write_text("# Slide 1\nThis is slide one text.\n", encoding="utf-8")

    engine = JobEngine(config=cfg)
    job_id = "job_recovery_test"

    # 1. First Run: Complete full job execution
    manifest = engine.run_job(
        job_id=job_id,
        document_path=str(doc_path),
        voice_id="speaker_01",
        text_mode="exact",
    )

    assert isinstance(manifest, JobManifest)
    assert manifest.status == JobStatus.COMPLETED
    assert manifest.slide_count >= 1
    assert Path(manifest.mastered_audio_file).exists()

    # 2. Crash Recovery Check: Running job_id again should immediately return manifest without re-running steps!
    manifest_resumed = engine.run_job(job_id=job_id)
    assert manifest_resumed.status == JobStatus.COMPLETED
    assert manifest_resumed.job_id == job_id
