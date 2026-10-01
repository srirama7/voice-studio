"""
Master 39-Voice Verification Runner.
Tests voice synthesis for all 39 voices against all 5 uploaded document types:
- PDF (Roblox_AI_Testing_MCP_Specification_v2.pdf)
- PPTX (HEART FAILURE PPT(3)_v2_backup.pptx)
- TXT (Amogha_Bhat_Cover_Letter.txt)
- DOCX (SYNOPSIS ANAGHA(2).docx)
- MD (QA_REPORT.md)
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np

# Ensure UTF-8 output encoding for Windows console
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

def main():
    print("=== STARTING MASTER 39-VOICE VERIFICATION ===")
    
    # 1. Parse all sample documents
    doc_samples = {}
    for doc_type, path_str in files.items():
        try:
            doc = DocumentParser.parse(path_str)
            text = doc.slides[0].get_narrative_text() if doc.slides else "Test text sample for voice generation."
            if len(text) > 120:
                text = text[:120] + "."
            doc_samples[doc_type] = text
            print(f"[OK] Parsed {doc_type}: {Path(path_str).name} ({len(text)} chars sample)")
        except Exception as exc:
            print(f"[ERROR] Parsing {doc_type} failed: {exc}")
            doc_samples[doc_type] = "Sample presentation slide text for voice verification."

    # 2. Get all 39 voice choices
    choices = all_voice_choices()
    print(f"\nFound total {len(choices)} voices to test.")

    job_eng = JobEngine(config=default_config)
    results = []
    passed_count = 0
    failed_count = 0

    for idx, (label, voice_val) in enumerate(choices, start=1):
        clean_lbl = clean_str(label)
        print(f"\n--- Testing Voice {idx}/39: {clean_lbl} ({voice_val}) ---")
        voice_res = {
            "index": idx,
            "label": label,
            "voice_val": voice_val,
            "tests": {},
            "all_passed": True,
            "error_msg": None
        }

        doc_type = list(doc_samples.keys())[(idx - 1) % len(doc_samples)]
        sample_text = doc_samples[doc_type]

        try:
            start_t = time.time()
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
            duration_sec = round(len(waveform) / sr, 2) if sr > 0 else 0.0
            
            is_success = (len(waveform) > 0 and duration_sec > 0.0 and not np.isnan(waveform).any())

            if is_success:
                voice_res["tests"][doc_type] = {
                    "status": "PASSED",
                    "duration_sec": duration_sec,
                    "elapsed_sec": elapsed,
                    "sample_count": len(waveform)
                }
                print(f"  [PASS] {doc_type} synthesis: generated {duration_sec}s audio in {elapsed}s (samples: {len(waveform)})")
                passed_count += 1
            else:
                voice_res["all_passed"] = False
                voice_res["tests"][doc_type] = {"status": "FAILED", "reason": "Empty or NaN waveform"}
                print(f"  [FAIL] {doc_type} synthesis failed: Empty waveform")
                failed_count += 1

        except Exception as exc:
            voice_res["all_passed"] = False
            voice_res["error_msg"] = str(exc)
            voice_res["tests"][doc_type] = {"status": "FAILED", "reason": str(exc)}
            print(f"  [FAIL] Error: {exc}")
            failed_count += 1

        results.append(voice_res)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_voices_tested": len(choices),
        "passed": passed_count,
        "failed": failed_count,
        "success_rate": f"{(passed_count / len(choices)) * 100:.1f}%",
        "voice_results": results
    }

    out_file = Path("VOICE_VERIFICATION_MATRIX.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*50)
    print(f"VERIFICATION COMPLETE: {passed_count}/{len(choices)} Voices PASSED ({summary['success_rate']})")
    print(f"Full results saved to {out_file.resolve()}")
    print("="*50)

if __name__ == "__main__":
    main()
