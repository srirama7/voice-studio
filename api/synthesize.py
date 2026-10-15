from http.server import BaseHTTPRequestHandler
import json
import logging
import os
import sys
import io
import wave
import urllib.parse
import re
from pathlib import Path

#: Cloud safety cap: the serverless function has ~1GB RAM and a hard
#: execution limit. Long narrations OOM-kill the instance
#: (FUNCTION_INVOCATION_FAILED). Refuse oversized texts with a clear
#: message instead of crashing; unlimited length stays available on the
#: local Studio server.
SERVERLESS_MAX_CHARS = int(os.getenv("SERVERLESS_MAX_CHARS", "2500"))

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from voice_clone import GALLERY_VOICES, EdgeTTSAdapter, CloningTTSAdapter
import numpy as np

logger = logging.getLogger(__name__)

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', self.headers.get('content-length', 0)))
        raw_body = self.rfile.read(content_length)
        content_type = self.headers.get('Content-Type', self.headers.get('content-type', ''))

        text = ""
        voice_id = "gallery:prabhat_in_m"
        speed = 1.0

        if 'application/json' in content_type:
            try:
                data = json.loads(raw_body.decode('utf-8'))
                text = data.get('text', '')
                voice_id = data.get('voice_id', 'gallery:prabhat_in_m')
                speed = float(data.get('speed', 1.0))
            except Exception:
                pass

        if not text:
            try:
                body_str = raw_body.decode('utf-8', errors='ignore')
                params = urllib.parse.parse_qs(body_str)
                text = params.get('text', [''])[0]
                voice_id = params.get('voice_id', ['gallery:prabhat_in_m'])[0]
                if 'speed' in params:
                    try:
                        speed = float(params.get('speed', ['1.0'])[0])
                    except Exception:
                        pass
                
                if not text:
                    m = re.search(r'name=["\']text["\']\r?\n\r?\n([^\r\n]+)', body_str)
                    if m:
                        text = m.group(1).strip()
                    m_v = re.search(r'name=["\']voice_id["\']\r?\n\r?\n([^\r\n]+)', body_str)
                    if m_v:
                        voice_id = m_v.group(1).strip()
            except Exception:
                pass

        if not text or not text.strip():
            self.send_response(400)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b"Text parameter required")
            return

        if SERVERLESS_MAX_CHARS > 0 and len(text.strip()) > SERVERLESS_MAX_CHARS:
            self.send_response(400)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(
                f"Cloud limit: text is {len(text.strip())} characters "
                f"(max {SERVERLESS_MAX_CHARS} per cloud request) -- it would "
                f"exhaust the serverless function memory. Shorten the text, "
                f"or synthesize long documents on the local Studio "
                f"(http://127.0.0.1:7860/studio), which has no length limit.".encode('utf-8'))
            return

        try:
            waveform = np.zeros(0, dtype=np.float32)
            sr = 24000

            if voice_id.startswith("gallery:"):
                gid = voice_id.split(":", 1)[1]
                info = GALLERY_VOICES.get(gid, GALLERY_VOICES.get('prabhat_in_m', {}))
                voice_name = info.get('voice', 'en-IN-PrabhatNeural')
                rate = info.get('rate', '+0%')
                pitch = info.get('pitch', '+0Hz')
                lang = info.get('lang', 'en')
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
                ref_wav = None
                for base_dir in [Path(root_dir) / "cache" / "voices" / voice_id, Path(root_dir) / "voices" / voice_id]:
                    if base_dir.exists():
                        candidates = sorted(base_dir.glob("*/ref.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
                        if candidates:
                            ref_wav = str(candidates[0])
                            break
                        elif (base_dir / "ref.wav").exists():
                            ref_wav = str(base_dir / "ref.wav")
                            break
                        else:
                            wavs = list(base_dir.rglob("*.wav"))
                            if wavs:
                                ref_wav = str(wavs[0])
                                break

                adapter = CloningTTSAdapter()
                waveform, sr = adapter.synthesize(text, speaker_wav=ref_wav, language="en-in", speed=speed)

            if len(waveform) == 0:
                adapter = EdgeTTSAdapter(forced_voice="en-IN-PrabhatNeural")
                waveform, sr = adapter.synthesize(text, language="en", speed=speed)

            pcm = np.clip(waveform * 32767.0, -32768, 32767).astype(np.int16)
            out_buffer = io.BytesIO()
            with wave.open(out_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm.tobytes())

            audio_bytes = out_buffer.getvalue()
            self.send_response(200)
            self.send_header('Content-type', 'audio/wav')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(audio_bytes)

        except Exception as exc:
            logger.warning("Primary synthesis failed (%s); trying fallback...", exc)
            try:
                adapter = EdgeTTSAdapter(forced_voice="en-IN-PrabhatNeural")
                waveform, sr = adapter.synthesize(text, language="en")
                pcm = np.clip(waveform * 32767.0, -32768, 32767).astype(np.int16)
                out_buffer = io.BytesIO()
                with wave.open(out_buffer, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(pcm.tobytes())
                audio_bytes = out_buffer.getvalue()
                self.send_response(200)
                self.send_header('Content-type', 'audio/wav')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(audio_bytes)
            except Exception as fallback_exc:
                self.send_response(500)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(f"Synthesis failed: {fallback_exc}".encode('utf-8'))
