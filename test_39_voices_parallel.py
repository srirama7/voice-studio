"""
Fast Parallel 39-Voice Verification Script.
Executes voice synthesis across all 39 voices concurrently (6 worker threads)
for all 5 document types: PDF, PPTX, TXT, DOCX, MD.
"""

import sys
import os
import json
import time
import concurrent.futures
from pathlib import Path
import numpy as np

# UTF-8 stdout
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app import all_voice_choices, GALLERY_VOICES
from document_parser import DocumentParser
from voice_clone import EdgeTTSAdapter, CloningTTSAdapter
from job_engine import JobEngine
from config import default_config

files = {
    "MD": r"C:\Users\amogh\Downloads\QA_REPORT.md",
    "PDF": r"C:\Users\amogh\Downloads\Roblox_AI_Testing_MCP_Specification_v2.pdf",
    "TXT": r"C:\Users\amogh\Downloads\Amogha_Bhat_Cover_Letter.txt",
    "DOCX": r"C:\Users\amogh\Downloads\SYNOPSIS ANAGHA(2).docx",
    "PPTX": r"C:\Users\amogh\Downloads\HEART FAILURE PPT(3)_v2_backup.pptx"
}

def clean_str(s: str) -> str:
    return s.encode("ascii", errors="ignore").decode("ascii")

def test_single_voice(item, doc_samples, job_eng):
    idx, (label, voice_val) = item
    clean_lbl = clean_str(label)
    doc_types = list(doc_samples.keys())
    doc_type = doc_types[(idx - 1) % len(doc_types)]
    sample_text = doc_samples[doc_type]

    result = {
        "index": idx,
        "label": label,
        "clean_label": clean_lbl,
        "voice_val": voice_val,
        "doc_type_tested": doc_type,
        "status": "PENDING",
        "duration_sec": 0.0,
        "elapsed_sec": 0.0,
        "error": None
    }

    start_t = time.time()
    try:
        if voice_val.startswith("gallery:"):
            gid = voice_val.split(":", 1)[1]
            info = GALLERY_VOICES[gid]
            adapter = EdgeTTSAdapter(
                forced_voice=info["voice"],
                forced_rate=info.get("rate", "+0%"),
                forced_pitch=info.get("pitch", "+0Hz"),
            )
            waveform, sr = adapter.synthesize(sample_text, language=info.get("lang", "en"))
        else:
            ref_wav = job_eng.resolve_voice_ref(voice_val)
            adapter = CloningTTSAdapter()
            waveform, sr = adapter.synthesize(sample_text, speaker_wav=ref_wav, language="en")

        elapsed = round(time.time() - start_t, 2)
        dur = round(len(waveform) / sr, 2) if sr > 0 else 0.0
        
        if len(waveform) > 0 and dur > 0.0 and not np.isnan(waveform).any():
            result["status"] = "PASSED"
            result["duration_sec"] = dur
            result["elapsed_sec"] = elapsed
            print(f"[PASS] Voice {idx:02d}/39 ({clean_lbl}) | Doc: {doc_type} | Audio: {dur}s | Time: {elapsed}s")
        else:
            result["status"] = "FAILED"
            result["error"] = "Empty or NaN waveform generated"
            print(f"[FAIL] Voice {idx:02d}/39 ({clean_lbl}) | Doc: {doc_type} | Error: Empty audio")
    except Exception as exc:
        elapsed = round(time.time() - start_t, 2)
        result["status"] = "FAILED"
        result["elapsed_sec"] = elapsed
        result["error"] = str(exc)
        print(f"[FAIL] Voice {idx:02d}/39 ({clean_lbl}) | Doc: {doc_type} | Exception: {exc}")

    return result

def main():
    print("=== STARTING PARALLEL 39-VOICE VERIFICATION ===")
    
    # Parse doc samples
    doc_samples = {}
    for doc_type, path_str in files.items():
        try:
            doc = DocumentParser.parse(path_str)
            text = doc.slides[0].get_narrative_text() if doc.slides else "Verification text sample."
            if len(text) > 100:
                text = text[:100] + "."
            doc_samples[doc_type] = text
        except Exception as exc:
            doc_samples[doc_type] = "Fallback slide text sample for testing."

    choices = all_voice_choices()
    print(f"Testing all {len(choices)} voices concurrently with 6 worker threads...\n")

    job_eng = JobEngine(config=default_config)
    items = list(enumerate(choices, start=1))
    
    results = []
    start_total = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(test_single_voice, item, doc_samples, job_eng) for item in items]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    results.sort(key=lambda r: r["index"])
    passed = [r for r in results if r["status"] == "PASSED"]
    failed = [r for r in results if r["status"] == "FAILED"]
    total_t = round(time.time() - start_total, 2)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_voices": len(choices),
        "passed_count": len(passed),
        "failed_count": len(failed),
        "success_rate": f"{(len(passed) / len(choices)) * 100:.1f}%",
        "total_time_seconds": total_t,
        "results": results
    }

    out_file = Path("VOICE_VERIFICATION_MATRIX.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print(f"VERIFICATION SUMMARY: {len(passed)}/{len(choices)} VOICES PASSED ({summary['success_rate']}) in {total_t}s")
    if failed:
        print(f"FAILED VOICES ({len(failed)}):")
        for f_item in failed:
            print(f" - Voice {f_item['index']}: {f_item['clean_label']} ({f_item['voice_val']}) -> Error: {f_item['error']}")
    print(f"Matrix report written to: {out_file.resolve()}")
    print("="*60)

if __name__ == "__main__":
    main()
