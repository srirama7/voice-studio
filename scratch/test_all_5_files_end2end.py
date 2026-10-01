import json
import urllib.request
from pathlib import Path

files = [
    Path(r"C:\Users\amogh\Downloads\QA_REPORT.md"),
    Path(r"C:\Users\amogh\Downloads\Roblox_AI_Testing_MCP_Specification_v2.pdf"),
    Path(r"C:\Users\amogh\Downloads\Amogha_Bhat_Cover_Letter.txt"),
    Path(r"C:\Users\amogh\Downloads\SYNOPSIS ANAGHA(2).docx"),
    Path(r"C:\Users\amogh\Downloads\HEART FAILURE PPT(3)_v2_backup.pptx"),
]

boundary = "----Boundary12345"

for fpath in files:
    print(f"\n--- Testing {fpath.name} ({fpath.suffix}) ---")
    file_bytes = fpath.read_bytes()
    body = b"--" + boundary.encode() + b'\r\nContent-Disposition: form-data; name="file"; filename="' + fpath.name.encode('utf-8') + b'"\r\n\r\n' + file_bytes + b"\r\n--" + boundary.encode() + b"--\r\n"

    # 1. Parse Document
    req_parse = urllib.request.Request(
        'https://my-voice-studio-zeta.vercel.app/api/parse',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
    )
    res_parse = urllib.request.urlopen(req_parse)
    parse_data = json.loads(res_parse.read())
    doc_type = parse_data.get('doc_type')
    slides_cnt = parse_data.get('total_slides')
    full_text = parse_data.get('full_text', '')
    print(f"Parse Status: {res_parse.status} | doc_type: {doc_type} | slides: {slides_cnt} | text_len: {len(full_text)}")

    # 2. Synthesize Extracted Document Text
    synth_payload = json.dumps({
        'text': full_text[:1500], # Send first section/narrative chunk
        'voice_id': 'stock:en-US-AvaNeural'
    }).encode('utf-8')

    req_synth = urllib.request.Request(
        'https://my-voice-studio-zeta.vercel.app/api/synthesize',
        data=synth_payload,
        headers={'Content-Type': 'application/json'}
    )
    res_synth = urllib.request.urlopen(req_synth)
    audio_bytes = res_synth.read()
    print(f"Synthesize Status: {res_synth.status} | Content-Type: {res_synth.headers.get('Content-Type')} | audio_bytes: {len(audio_bytes)}")
