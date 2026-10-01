from http.server import BaseHTTPRequestHandler
import json
import os
import re
import sys
import tempfile
import zipfile
import io
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from document_parser import DocumentParser

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        temp_path = None
        try:
            content_length = int(self.headers.get('Content-Length', self.headers.get('content-length', 0)))
            body_bytes = self.rfile.read(content_length)
            content_type = self.headers.get('Content-Type', self.headers.get('content-type', ''))

            ext = ""
            file_data = body_bytes

            # 1. Parse Multipart Form Data if present
            if 'multipart/form-data' in content_type and 'boundary=' in content_type:
                boundary = content_type.split('boundary=')[1].strip()
                if boundary.startswith('"') and boundary.endswith('"'):
                    boundary = boundary[1:-1]
                b_boundary = ('--' + boundary).encode('utf-8')
                parts = body_bytes.split(b_boundary)

                for part in parts:
                    if b'filename=' in part or b'Content-Disposition' in part:
                        header_part, _, data_part = part.partition(b'\r\n\r\n')
                        header_str = header_part.decode('utf-8', errors='ignore')
                        match = re.search(r'filename=["\']?([^"\'\r\n;]+)["\']?', header_str, re.IGNORECASE)
                        if match:
                            filename = match.group(1).strip()
                            ext = Path(filename).suffix.lower()

                        # Strip trailing \r\n or -- from data_part
                        while data_part.endswith(b'\r\n') or data_part.endswith(b'--'):
                            if data_part.endswith(b'\r\n'):
                                data_part = data_part[:-2]
                            elif data_part.endswith(b'--'):
                                data_part = data_part[:-2]
                        file_data = data_part
                        break

            # 2. Magic byte / zip structure inspection fallback
            if ext not in (".pptx", ".pdf", ".docx", ".txt", ".md"):
                if file_data.startswith(b'%PDF'):
                    ext = ".pdf"
                elif file_data.startswith(b'PK\x03\x04'):
                    try:
                        with zipfile.ZipFile(io.BytesIO(file_data)) as z:
                            names = z.namelist()
                            if any(n.startswith("ppt/") for n in names):
                                ext = ".pptx"
                            elif any(n.startswith("word/") for n in names):
                                ext = ".docx"
                            else:
                                ext = ".pptx"
                    except Exception:
                        ext = ".pptx"
                else:
                    ext = ".txt"

            rand_id = os.urandom(4).hex()
            temp_path = Path(tempfile.gettempdir()) / f"upload_doc_{rand_id}{ext}"
            with open(temp_path, "wb") as f:
                f.write(file_data)

            parsed = DocumentParser.parse(temp_path)
            res_data = parsed.to_dict()

            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(res_data, ensure_ascii=False).encode('utf-8'))

        except Exception as exc:
            self.send_response(500)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode('utf-8'))
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
