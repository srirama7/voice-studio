"""
Test script test_voices_batch2.py
Tests voice synthesis for voices 21 to 39 out of 39 on sample text extracted from 5 document types:
- QA_REPORT.md (.md)
- Roblox_AI_Testing_MCP_Specification_v2.pdf (.pdf)
- Amogha_Bhat_Cover_Letter.txt (.txt)
- SYNOPSIS ANAGHA(2).docx (.docx)
- HEART FAILURE PPT(3)_v2_backup.pptx (.pptx)

Outputs detailed results to test_batch2_results.json.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import docx
import pptx
import pymupdf as fitz

from document_parser import DocumentParser
from job_engine import JobEngine, JobStatus, JobManifest
from voice_clone import GALLERY_VOICES

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_voices_batch2")

VOICES_BATCH2 = [
    (21, "gallery:gagan_kn_m", "gagan_kn_m"),
    (22, "gallery:gagan_deep_kn_m", "gagan_deep_kn_m"),
    (23, "gallery:gagan_young_kn_m", "gagan_young_kn_m"),
    (24, "gallery:sapna_kn_f", "sapna_kn_f"),
    (25, "gallery:sapna_soft_kn_f", "sapna_soft_kn_f"),
    (26, "gallery:sapna_lively_kn_f", "sapna_lively_kn_f"),
    (27, "gallery:valluvar_ta_m", "valluvar_ta_m"),
    (28, "gallery:pallavi_ta_f", "pallavi_ta_f"),
    (29, "gallery:mohan_te_m", "mohan_te_m"),
    (30, "gallery:shruti_te_f", "shruti_te_f"),
    (31, "gallery:midhun_ml_m", "midhun_ml_m"),
    (32, "gallery:sobhana_ml_f", "sobhana_ml_f"),
    (33, "gallery:manohar_mr_m", "manohar_mr_m"),
    (34, "gallery:aarohi_mr_f", "aarohi_mr_f"),
    (35, "gallery:bashkar_bn_m", "bashkar_bn_m"),
    (36, "gallery:tanishaa_bn_f", "tanishaa_bn_f"),
    (37, "gallery:niranjan_gu_m", "niranjan_gu_m"),
    (38, "gallery:dhwani_gu_f", "dhwani_gu_f"),
    (39, "gallery:gurpreet_pa_m", "gurpreet_pa_m"),
]

TEST_FILES_SOURCE = [
    ("md", r"C:\Users\amogh\Downloads\QA_REPORT.md"),
    ("pdf", r"C:\Users\amogh\Downloads\Roblox_AI_Testing_MCP_Specification_v2.pdf"),
    ("txt", r"C:\Users\amogh\Downloads\Amogha_Bhat_Cover_Letter.txt"),
    ("docx", r"C:\Users\amogh\Downloads\SYNOPSIS ANAGHA(2).docx"),
    ("pptx", r"C:\Users\amogh\Downloads\HEART FAILURE PPT(3)_v2_backup.pptx"),
]


def prepare_sample_documents(work_dir: Path) -> Dict[str, Path]:
    """Extract sample text from source files and create sample documents for each file type."""
    samples_dir = work_dir / "batch2_sample_docs"
    samples_dir.mkdir(exist_ok=True, parents=True)
    sample_paths: Dict[str, Path] = {}

    logger.info("Extracting sample texts from source files...")

    for file_type, src_path_str in TEST_FILES_SOURCE:
        src_path = Path(src_path_str)
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")

        parsed = DocumentParser.parse(src_path)
        extracted_text = ""
        for slide in parsed.slides[:2]:
            narrative = slide.get_narrative_text().strip()
            if narrative:
                extracted_text += narrative + " "

        extracted_text = extracted_text.strip()
        if len(extracted_text) > 300:
            extracted_text = extracted_text[:300] + "."
        if not extracted_text:
            extracted_text = f"Sample text for {file_type} document synthesis testing."

        out_path = samples_dir / f"sample_{file_type}.{file_type}"

        if file_type == "md":
            out_path.write_text(f"# Markdown Sample\n{extracted_text}\n", encoding="utf-8")
        elif file_type == "txt":
            out_path.write_text(extracted_text, encoding="utf-8")
        elif file_type == "docx":
            doc = docx.Document()
            doc.add_heading("DOCX Sample", level=1)
            doc.add_paragraph(extracted_text)
            doc.save(str(out_path))
        elif file_type == "pptx":
            prs = pptx.Presentation()
            slide = prs.slides.add_slide(prs.slide_layouts[0])
            slide.shapes.title.text = "PPTX Sample"
            slide.placeholders[1].text = extracted_text
            prs.save(str(out_path))
        elif file_type == "pdf":
            pdf_doc = fitz.open()
            page = pdf_doc.new_page()
            page.insert_text((50, 50), f"PDF Sample Document\n\n{extracted_text}")
            pdf_doc.save(str(out_path))
            pdf_doc.close()

        sample_paths[file_type] = out_path
        logger.info("Prepared %s sample: %s (%d chars)", file_type, out_path, len(extracted_text))

    return sample_paths


def verify_audio_file(audio_path: str) -> Tuple[bool, float, int, int, str]:
    """Verify audio file exist, can be parsed, duration > 0, non-zero amplitude."""
    if not audio_path or not Path(audio_path).exists():
        return False, 0.0, 0, 0, f"Audio file does not exist: {audio_path}"

    try:
        with wave.open(audio_path, "rb") as wf:
            n_frames = wf.getnframes()
            sr = wf.getframerate()
            if sr <= 0 or n_frames <= 0:
                return False, 0.0, sr, 0, f"Invalid audio header (sr={sr}, frames={n_frames})"
            duration = n_frames / float(sr)
            raw = wf.readframes(n_frames)
            audio = np.frombuffer(raw, dtype=np.int16)
            if len(audio) == 0:
                return False, duration, sr, 0, "Audio buffer is empty"
            max_amp = int(np.max(np.abs(audio)))
            if max_amp == 0:
                return False, duration, sr, max_amp, "Audio signal is completely silent (amplitude 0)"
            if duration <= 0.0:
                return False, duration, sr, max_amp, "Audio duration is 0"

            return True, duration, sr, max_amp, ""
    except Exception as exc:
        return False, 0.0, 0, 0, f"Failed to read WAV file: {exc}"


def run_batch2_tests():
    start_time = time.time()
    work_dir = Path(os.getcwd())
    logger.info("Starting Voice Synthesis Test Suite for Batch 2 (Voices 21-39)...")

    sample_docs = prepare_sample_documents(work_dir)
    engine = JobEngine()

    results: List[Dict[str, Any]] = []
    passed_count = 0
    failed_count = 0
    total_tests = len(VOICES_BATCH2) * len(TEST_FILES_SOURCE)

    test_idx = 0
    for voice_idx, voice_full, gid in VOICES_BATCH2:
        voice_info = GALLERY_VOICES.get(gid, {})
        lang = voice_info.get("lang", "en-in")

        logger.info("==================================================")
        logger.info("Testing Voice %d/39: %s (lang=%s, label=%s)", voice_idx, voice_full, lang, voice_info.get("label", ""))

        for file_type, src_path_str in TEST_FILES_SOURCE:
            test_idx += 1
            doc_path = sample_docs[file_type]
            job_id = f"test_b2_v{voice_idx}_{file_type}_{int(time.time()*1000)}"

            logger.info("[%d/%d] Testing voice %d (%s) on doc type '%s'...", test_idx, total_tests, voice_idx, gid, file_type)

            test_record: Dict[str, Any] = {
                "test_index": test_idx,
                "voice_index": voice_idx,
                "voice_id": voice_full,
                "gallery_voice": gid,
                "language": lang,
                "file_type": file_type,
                "source_file": str(src_path_str),
                "sample_doc_file": str(doc_path),
                "job_id": job_id,
                "status": "FAILED",
                "audio_path": None,
                "duration_sec": 0.0,
                "sample_rate": 0,
                "max_amplitude": 0,
                "error": None,
            }

            try:
                manifest: JobManifest = engine.run_job(
                    job_id=job_id,
                    document_path=str(doc_path),
                    voice_id=voice_full,
                    text_mode="exact",
                    language=lang,
                    gallery_voice=gid,
                )

                if manifest.status != JobStatus.COMPLETED:
                    test_record["error"] = f"Job status was {manifest.status} instead of COMPLETED"
                else:
                    audio_file = manifest.mastered_audio_file
                    test_record["audio_path"] = audio_file
                    is_valid, duration, sr, max_amp, err = verify_audio_file(audio_file)
                    test_record["duration_sec"] = round(duration, 3)
                    test_record["sample_rate"] = sr
                    test_record["max_amplitude"] = max_amp

                    if is_valid:
                        test_record["status"] = "PASSED"
                        passed_count += 1
                        logger.info(" -> PASSED! Duration: %.2fs, SR: %dHz, MaxAmp: %d", duration, sr, max_amp)
                    else:
                        test_record["error"] = err
                        failed_count += 1
                        logger.error(" -> FAILED audio verification: %s", err)

            except Exception as exc:
                test_record["error"] = str(exc)
                failed_count += 1
                logger.error(" -> FAILED with exception: %s", exc)

            results.append(test_record)

    elapsed_time = round(time.time() - start_time, 2)
    pass_rate = round((passed_count / total_tests) * 100.0, 2) if total_tests > 0 else 0.0

    summary = {
        "batch": "Batch 2 (Voices 21-39)",
        "voice_range": "21 to 39",
        "total_voices_tested": len(VOICES_BATCH2),
        "document_types_tested": [ft for ft, _ in TEST_FILES_SOURCE],
        "total_tests": total_tests,
        "passed": passed_count,
        "failed": failed_count,
        "pass_rate_percent": pass_rate,
        "elapsed_seconds": elapsed_time,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,
    }

    out_json = work_dir / "test_batch2_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("==================================================")
    logger.info("TEST RUN COMPLETE!")
    logger.info("Total Tests: %d", total_tests)
    logger.info("Passed: %d", passed_count)
    logger.info("Failed: %d", failed_count)
    logger.info("Pass Rate: %.2f%%", pass_rate)
    logger.info("Execution Time: %.2fs", elapsed_time)
    logger.info("Results saved to: %s", out_json)

    return summary


if __name__ == "__main__":
    summary = run_batch2_tests()
    if summary["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)
