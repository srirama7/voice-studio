"""i18n Translation Engine for Voice Studio.

Provides multi-provider fallback translation for presentation narration text into
Indian & International languages (Kannada, Hindi, Tamil, Telugu, Malayalam, Marathi,
Bengali, Gujarati, English, Spanish, French, German, etc.).
"""

import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Dict, List

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: Dict[str, str] = {
    "kn": "Kannada (ಕನ್ನಡ)",
    "hi": "Hindi (हिंदी)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "ml": "Malayalam (മലയാളം)",
    "mr": "Marathi (मराठी)",
    "bn": "Bengali (বাংলা)",
    "gu": "Gujarati (ગુજરાતી)",
    "en": "English",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
}


def translate_chunk(c_text: str, target_lang: str = "hi") -> str:
    """Translate a single text chunk using multi-provider fallback."""
    if not c_text or not c_text.strip():
        return c_text

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        )
    }

    # Provider 1: dict-chrome-ex (fastest, high quota)
    try:
        url = (
            f"https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=auto&tl={target_lang}&q="
            + urllib.parse.quote(c_text)
        )
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as res:
            res_json = json.loads(res.read().decode('utf-8'))
            if isinstance(res_json, list) and len(res_json) > 0:
                if isinstance(res_json[0], list) and len(res_json[0]) > 0:
                    return res_json[0][0]
                elif isinstance(res_json[0], str):
                    return "".join(res_json)
    except Exception as err:
        logger.debug("Provider dict-chrome-ex failed: %s", err)

    # Provider 2: translate.googleapis.com (gtx)
    try:
        url = (
            f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={target_lang}&dt=t&q="
            + urllib.parse.quote(c_text)
        )
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as res:
            res_json = json.loads(res.read().decode('utf-8'))
            return "".join([item[0] for item in res_json[0] if item and item[0]])
    except Exception as err:
        logger.debug("Provider gtx failed: %s", err)

    # Provider 3: MyMemory API fallback
    try:
        url = (
            "https://api.mymemory.translated.net/get?q="
            + urllib.parse.quote(c_text[:400])
            + f"&langpair=auto|{target_lang}"
        )
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as res:
            res_json = json.loads(res.read().decode('utf-8'))
            if 'responseData' in res_json and 'translatedText' in res_json['responseData']:
                return res_json['responseData']['translatedText']
    except Exception as err:
        logger.debug("Provider MyMemory failed: %s", err)

    return c_text


def translate_text(text: str, target_lang: str = "hi") -> str:
    """Translate full presentation narration text with smart chunking.

    Args:
        text: Input text in any language.
        target_lang: Language code (e.g. 'hi', 'kn', 'ta', 'te', 'en', 'es', 'fr', etc.).

    Returns:
        Translated text string in target_lang.
    """
    if not text or not text.strip():
        return ""

    target_lang = target_lang.strip().lower()
    clean_text = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]', ' ', text).strip()

    # Smart chunking (~1000 chars per chunk)
    max_chunk_size = 1000
    paragraphs = clean_text.split('\n\n')
    chunks: List[str] = []
    current_chunk = ""

    for p in paragraphs:
        p_str = p.strip()
        if not p_str:
            continue
        if len(current_chunk) + len(p_str) + 2 <= max_chunk_size:
            current_chunk = f"{current_chunk}\n\n{p_str}" if current_chunk else p_str
        else:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            if len(p_str) > max_chunk_size:
                lines = p_str.split('\n')
                for line in lines:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    if len(current_chunk) + len(line_str) + 1 <= max_chunk_size:
                        current_chunk = f"{current_chunk}\n{line_str}" if current_chunk else line_str
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                            current_chunk = ""
                        if len(line_str) > max_chunk_size:
                            sentences = re.split(r'(?<=[.!?])\s+', line_str)
                            for s in sentences:
                                s_str = s.strip()
                                if not s_str:
                                    continue
                                if len(current_chunk) + len(s_str) + 1 <= max_chunk_size:
                                    current_chunk = f"{current_chunk} {s_str}" if current_chunk else s_str
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk)
                                    current_chunk = s_str
                        else:
                            current_chunk = line_str
            else:
                current_chunk = p_str

    if current_chunk:
        chunks.append(current_chunk)

    translated_parts = [translate_chunk(chunk, target_lang) for chunk in chunks]
    return "\n\n".join(translated_parts)
