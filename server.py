"""
Voice Studio & Audiogram Unified Local High-Performance Server.

Serves:
1. Audiogram Web Studio (public/index.html) at http://127.0.0.1:7860/
2. Local High-Speed API endpoints (/api/voices, /api/synthesize, /api/parse, /api/translate)
   - Zero Vercel timeouts (handles unlimited length 10-20 min narrations)
   - Real-time EdgeTTS + VoiceClone synthesis with 5-stream concurrency
3. Gradio Enterprise Studio mounted at http://127.0.0.1:7860/studio
"""

import os
import sys
import io
import wave
import json
import tempfile
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import gradio as gr

# Ensure root dir is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from voice_clone import GALLERY_VOICES, EdgeTTSAdapter, CloningTTSAdapter
from document_parser import DocumentParser
from translator import translate_text
from app import build_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("voice_studio_server")

app = FastAPI(title="Voice Studio & Audiogram Server")

# Enable CORS for seamless fetch requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API Endpoints (Local High-Performance Implementations)
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def api_health():
    return {
        "status": "online",
        "service": "Voice Studio Enterprise (Local High-Performance)",
        "total_voices": len(GALLERY_VOICES) + 7,
        "supported_formats": [".pdf", ".pptx", ".txt", ".md", ".docx"]
    }


@app.get("/api/voices")
async def api_voices():
    choices = []
    # Enrolled voice profiles
    enrolled = ["speaker_01", "amogh", "nethuman", "netvoice2", "speaker_demo", "test_speaker_01", "voice3"]
    voices_cache_dir = root_dir / "cache" / "voices"
    if voices_cache_dir.exists():
        for d in voices_cache_dir.iterdir():
            if d.is_dir() and d.name not in enrolled:
                enrolled.append(d.name)

    for v in enrolled:
        choices.append({"label": f"🎙️ {v} — my clone", "value": v})

    for gid, info in GALLERY_VOICES.items():
        choices.append({"label": f"⭐ {info['label']}", "value": f"gallery:{gid}"})

    return {
        "total_count": len(choices),
        "choices": choices
    }


@app.post("/api/synthesize")
async def api_synthesize(request: Request):
    content_type = request.headers.get("content-type", "")
    text = ""
    voice_id = "gallery:prabhat_in_m"
    speed = 1.0

    if "application/json" in content_type:
        try:
            body = await request.json()
            text = body.get("text", "")
            voice_id = body.get("voice_id", "gallery:prabhat_in_m")
            speed = float(body.get("speed", 1.0))
        except Exception:
            pass
    else:
        try:
            form = await request.form()
            text = form.get("text", "")
            voice_id = form.get("voice_id", "gallery:prabhat_in_m")
            if "speed" in form:
                speed = float(form.get("speed", 1.0))
        except Exception:
            pass

    if not text or not str(text).strip():
        return Response(content="Text parameter required", status_code=400, media_type="text/plain")

    text = str(text).strip()
    logger.info("Local Synthesize Request: %d chars, voice=%s, speed=%.2f", len(text), voice_id, speed)

    try:
        waveform = np.zeros(0, dtype=np.float32)
        sr = 24000

        if voice_id.startswith("gallery:"):
            gid = voice_id.split(":", 1)[1]
            info = GALLERY_VOICES.get(gid, GALLERY_VOICES.get("prabhat_in_m", {}))
            voice_name = info.get("voice", "en-IN-PrabhatNeural")
            rate = info.get("rate", "+0%")
            pitch = info.get("pitch", "+0Hz")
            lang = info.get("lang", "en")
            adapter = EdgeTTSAdapter(
                forced_voice=voice_name,
                forced_rate=rate,
                forced_pitch=pitch
            )
            waveform, sr = adapter.synthesize(text, language=lang, speed=speed)

        elif voice_id.startswith("stock:"):
            v_name = voice_id.split(":", 1)[1]
            lang_code = "en"
            if "-" in v_name:
                lang_code = v_name.split("-")[0]
            adapter = EdgeTTSAdapter(forced_voice=v_name)
            waveform, sr = adapter.synthesize(text, language=lang_code, speed=speed)

        else:
            # Enrolled clone voice ID (e.g. "amogh", "speaker_01")
            ref_dir = root_dir / "cache" / "voices" / voice_id
            if not ref_dir.exists():
                ref_dir = root_dir / "voices" / voice_id

            ref_wav = None
            if ref_dir.exists():
                cands = sorted(ref_dir.glob("*/ref.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
                if cands:
                    ref_wav = str(cands[0])
                else:
                    for cand in ["ref.wav", "ref(1).wav", "speaker.wav"]:
                        if (ref_dir / cand).exists():
                            ref_wav = str(ref_dir / cand)
                            break

            adapter = CloningTTSAdapter()
            waveform, sr = adapter.synthesize(text, speaker_wav=ref_wav, language="en", speed=speed)

        if len(waveform) == 0:
            adapter = EdgeTTSAdapter(forced_voice="en-IN-PrabhatNeural")
            waveform, sr = adapter.synthesize(text, language="en", speed=speed)

        # Convert to 16-bit PCM WAV bytes
        pcm = np.clip(waveform * 32767.0, -32768, 32767).astype(np.int16)
        out_buffer = io.BytesIO()
        with wave.open(out_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

        audio_bytes = out_buffer.getvalue()
        logger.info("Synthesis complete: %d bytes (%.2fs)", len(audio_bytes), len(waveform) / sr)

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'inline; filename="audiogram_speech_{os.urandom(4).hex()}.wav"',
                "Access-Control-Allow-Origin": "*",
            }
        )

    except Exception as exc:
        logger.error("Synthesis error: %s", exc, exc_info=True)
        return Response(content=f"Synthesis failed: {exc}", status_code=500, media_type="text/plain")


@app.post("/api/parse")
async def api_parse(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower() if file.filename else ".txt"
    if ext not in [".pptx", ".pdf", ".txt", ".md", ".docx"]:
        ext = ".txt"

    rand_id = os.urandom(4).hex()
    temp_path = Path(tempfile.gettempdir()) / f"upload_doc_{rand_id}{ext}"
    try:
        content = await file.read()
        temp_path.write_bytes(content)
        parsed = DocumentParser.parse(str(temp_path))
        return parsed.to_dict()
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.post("/api/translate")
async def api_translate(request: Request):
    try:
        body = await request.json()
        text = body.get("text", "").strip()
        target_lang = body.get("target_lang", "kn").strip().lower()

        if not text:
            return Response(content=json.dumps({"error": "text is required"}), status_code=400, media_type="application/json")

        translated_text = translate_text(text, target_lang=target_lang)
        return {
            "original_text": text,
            "target_lang": target_lang,
            "translated_text": translated_text
        }
    except Exception as exc:
        return Response(content=json.dumps({"error": str(exc)}), status_code=500, media_type="application/json")


# ---------------------------------------------------------------------------
# Static Web App (Audiogram at Root '/')
# ---------------------------------------------------------------------------

public_dir = root_dir / "public"
textures_dir = public_dir / "textures"

if textures_dir.exists():
    app.mount("/textures", StaticFiles(directory=str(textures_dir)), name="textures")


@app.get("/")
async def serve_index():
    index_file = public_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return Response(content="Audiogram index.html not found.", media_type="text/plain")


# ---------------------------------------------------------------------------
# Mount Gradio Enterprise Studio at '/studio'
# ---------------------------------------------------------------------------

demo = build_app()
app = gr.mount_gradio_app(app, demo, path="/studio")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    logger.info("Starting Unified Voice Studio & Audiogram Server on http://127.0.0.1:%d", port)
    logger.info("  -> Audiogram Web Studio: http://127.0.0.1:%d/", port)
    logger.info("  -> Enterprise Studio:   http://127.0.0.1:%d/studio", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
