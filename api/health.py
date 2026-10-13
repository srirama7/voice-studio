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
            import edge_tts
            diag["edge_tts"] = "available"
        except Exception as e:
            diag["edge_tts"] = f"error: {e}"

        try:
            from voice_clone import EdgeTTSAdapter
            adapter = EdgeTTSAdapter(forced_voice="en-IN-PrabhatNeural")
            audio, sr = adapter.synthesize("test", language="en")
            diag["synth_samples"] = len(audio)
            diag["synth_sr"] = sr
            diag["is_fallback_tone"] = (len(audio) == int(sr * 1.5))
        except Exception as e:
            diag["synth_error"] = f"{e}\n{traceback.format_exc()}"

        response = {
            "status": "online",
            "service": "Voice Studio Enterprise",
            "total_voices": 39,
            "supported_formats": [".pdf", ".pptx", ".txt", ".md", ".docx"],
            "diagnostics": diag
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))
