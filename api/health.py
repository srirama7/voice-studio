from http.server import BaseHTTPRequestHandler
import json
import sys
import traceback
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        diag = {}
        try:
            import miniaudio
            diag["miniaudio"] = "available"
        except Exception as e:
            diag["miniaudio"] = f"error: {e}"

        try:
            import asyncio
            import tempfile
            import os
            import edge_tts
            diag["edge_tts"] = "available"

            async def _test_tts():
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    out_path = tmp.name
                try:
                    comm = edge_tts.Communicate("Hello test", "en-IN-PrabhatNeural")
                    await comm.save(out_path)
                    sz = os.path.getsize(out_path) if os.path.exists(out_path) else 0
                    diag["direct_mp3_size"] = sz
                    if sz > 0:
                        import miniaudio
                        dec = miniaudio.decode_file(out_path)
                        diag["miniaudio_samples"] = len(dec.samples)
                        diag["miniaudio_sr"] = dec.sample_rate
                finally:
                    if os.path.exists(out_path):
                        os.unlink(out_path)

            asyncio.run(_test_tts())
            diag["direct_tts"] = "success"
        except Exception as e:
            diag["direct_tts_error"] = f"{type(e).__name__}: {e}"

        try:
            from voice_clone import EdgeTTSAdapter
            adapter = EdgeTTSAdapter(forced_voice="en-IN-PrabhatNeural")
            audio, sr = adapter.synthesize("Welcome to Voice Studio", language="en")
            diag["adapter_samples"] = len(audio)
            diag["adapter_sr"] = sr
            diag["adapter_is_tone"] = (len(audio) == int(sr * 1.5))
        except Exception as e:
            diag["adapter_error"] = f"{type(e).__name__}: {e}"

        response = {
            "status": "online",
            "service": "Voice Studio Enterprise",
            "total_voices": 39,
            "supported_formats": [".pdf", ".pptx", ".txt", ".md", ".docx"],
            "diagnostics": diag
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))
