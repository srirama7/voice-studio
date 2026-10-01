from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = {
            "status": "online",
            "service": "Voice Studio Enterprise",
            "total_voices": 39,
            "supported_formats": [".pdf", ".pptx", ".txt", ".md", ".docx"]
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))
