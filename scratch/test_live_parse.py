import json
import urllib.request
from pathlib import Path

p = Path(r"C:\Users\amogh\Downloads\HEART FAILURE PPT(3)_v2_backup.pptx")
ppt_bytes = p.read_bytes()
boundary = "----Boundary1234"
body = b"--" + boundary.encode() + b'\r\nContent-Disposition: form-data; name="file"; filename="HEART FAILURE PPT(3)_v2_backup.pptx"\r\n\r\n' + ppt_bytes + b"\r\n--" + boundary.encode() + b"--\r\n"

req = urllib.request.Request(
    'https://my-voice-studio-zeta.vercel.app/api/parse',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)

res = urllib.request.urlopen(req)
data = json.loads(res.read())

print("Vercel Live Parse Status:", res.status)
print("Doc Type:", data.get('doc_type'))
print("Total Slides:", data.get('total_slides'))
print("Full Text Length:", len(data.get('full_text', '')))
print("Extracted Sample:", data.get('full_text', '')[:200])
