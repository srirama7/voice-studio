from http.server import BaseHTTPRequestHandler
import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from voice_clone import GALLERY_VOICES

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        choices = []
        # Enrolled clones
        enrolled = ["speaker_01", "amogh", "nethuman", "netvoice2", "speaker_demo", "test_speaker_01", "voice3"]
        for v in enrolled:
            choices.append({"label": f"🎙️ {v} — my clone", "value": v})

        for gid, info in GALLERY_VOICES.items():
            choices.append({"label": f"⭐ {info['label']}", "value": f"gallery:{gid}"})
        
        response = {
            "total_count": len(choices),
            "choices": choices
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))
