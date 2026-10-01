import io
import json
import sys
from pathlib import Path
sys.path.insert(0, '.')
from api.parse import handler

ppt_bytes = Path(r"C:\Users\amogh\Downloads\HEART FAILURE PPT(3)_v2_backup.pptx").read_bytes()
boundary = "----Boundary1234"
body = b"--" + boundary.encode() + b'\r\nContent-Disposition: form-data; name="file"; filename="HEART FAILURE PPT(3)_v2_backup.pptx"\r\n\r\n' + ppt_bytes + b"\r\n--" + boundary.encode() + b"--\r\n"

class ReqHandler(handler):
    def __init__(self, body_data, boundary_str):
        self.rfile = io.BytesIO(body_data)
        self.wfile = io.BytesIO()
        self.headers = {'Content-Length': str(len(body_data)), 'Content-Type': f'multipart/form-data; boundary={boundary_str}'}
    def send_response(self, code):
        self.code = code
    def send_header(self, k, v):
        pass
    def end_headers(self):
        pass

h = ReqHandler(body, boundary)
h.do_POST()
res = json.loads(h.wfile.getvalue().decode('utf-8'))
print('STATUS:', getattr(h, 'code', 200))
print('DOC_TYPE:', res.get('doc_type'))
print('SLIDES:', res.get('total_slides'))
print('FULL_TEXT_LEN:', len(res.get('full_text', '')))
print('SAMPLE TEXT:', res.get('full_text', '')[:150])
