from http.server import BaseHTTPRequestHandler
import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from translator import translate_text, SUPPORTED_LANGUAGES


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', self.headers.get('content-length', 0)))
            raw_body = self.rfile.read(content_length)
            data = json.loads(raw_body.decode('utf-8'))
            text = data.get('text', '').strip()
            target_lang = data.get('target_lang', 'kn').strip().lower()

            if not text:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "text parameter is required"}).encode('utf-8'))
                return

            translated_text = translate_text(text, target_lang=target_lang)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "original_text": text,
                "target_lang": target_lang,
                "translated_text": translated_text
            }, ensure_ascii=False).encode('utf-8'))

        except Exception as exc:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode('utf-8'))
