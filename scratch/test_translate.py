import io
import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from api.translate import handler

class ReqHandler(handler):
    def __init__(self, data):
        self.rfile = io.BytesIO(json.dumps(data).encode('utf-8'))
        self.wfile = io.BytesIO()
        self.headers = {'Content-Length': str(len(self.rfile.getvalue())), 'Content-Type': 'application/json'}
    def send_response(self, code):
        self.code = code
    def send_header(self, k, v):
        pass
    def end_headers(self):
        pass

h = ReqHandler({'text': 'Hello and welcome to Voice Studio Enterprise.', 'target_lang': 'kn'})
h.do_POST()
res = json.loads(h.wfile.getvalue().decode('utf-8'))
print('STATUS:', getattr(h, 'code', 200))
print('TRANSLATED KANNADA:', repr(res.get('translated_text')))

h2 = ReqHandler({'text': 'The Sun is a glowing star at the center of the solar system.', 'target_lang': 'hi'})
h2.do_POST()
res2 = json.loads(h2.wfile.getvalue().decode('utf-8'))
print('TRANSLATED HINDI:', repr(res2.get('translated_text')))
