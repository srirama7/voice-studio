import asyncio
import re
import tempfile
import edge_tts
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '.')
from voice_clone import VoiceConverter

# User's exact text with \u000b
sample_text = """Slide 1. THE SOLAR SYSTEM
5 KEY FACTS PER TOPIC • COMPLETE GUIDE TO OUR PLANETARY NEIGHBORHOOD
20 Clear, Easy-to-Read Slides • Simple Concepts & Mind-Blowing Facts\u000bThe Sun, Rocky Planets, Gas Giants, Ocean Moons & Outer Space Frontiers

Slide 2. TABLE OF CONTENTS
A QUICK TOUR OF OUR 20 SLIDES AND 18 FASCINATING TOPICS
ORGANIZATION: 20 Curated Slides  |  FORMAT: Exactly 5 Key Facts Per Topic  |  SIMPLE & ACCURATE
01  THE SUN & INNER WORLDS
• Topic 01: Solar System Overview (5 Facts)
• Topic 02: The Sun — Our Central Star (5 Facts)
• Topic 03: Mercury — The Fastest Planet (5 Facts)
• Topic 04: Venus — The Hottest Planet (5 Facts)
• Topic 05: Earth — Our Living Home (5 Facts)
• Topic 06: The Moon — Earth's Partner (5 Facts)
02  MARS, ASTEROIDS & JUPITER
• Topic 07: Mars — The Red Planet (5 Facts)
• Topic 08: The Asteroid Belt & Ceres (5 Facts)
• Topic 09: Jupiter — King of Planets (5 Facts)
• Topic 10: The Galilean Moons (5 Facts)
• Highlights: Io's Volcanoes & Europa's Ocean
• Highlights: Ganymede & Cratered Callisto"""

# 1. Clean control chars
clean_text = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]', ' ', sample_text)

# 2. Chunking function
def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_chars:
            current = f"{current}\n\n{p}" if current else p
        else:
            if current:
                chunks.append(current)
            if len(p) > max_chars:
                # Split by sentence
                sentences = re.split(r'(?<=[.!?])\s+', p)
                sub_curr = ""
                for s in sentences:
                    if len(sub_curr) + len(s) + 1 <= max_chars:
                        sub_curr = f"{sub_curr} {s}" if sub_curr else s
                    else:
                        if sub_curr:
                            chunks.append(sub_curr)
                        sub_curr = s
                if sub_curr:
                    current = sub_curr
                else:
                    current = ""
            else:
                current = p
    if current:
        chunks.append(current)
    return chunks

chunks = chunk_text(clean_text)
print("Total Chunks:", len(chunks))

async def synth_all():
    async def synth_one(c):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            p = tmp.name
        comm = edge_tts.Communicate(c, "en-US-AvaNeural")
        await comm.save(p)
        audio, sr = VoiceConverter.load_mono(p, 24000)
        Path(p).unlink(missing_ok=True)
        return audio
    
    audios = await asyncio.gather(*[synth_one(c) for c in chunks])
    full_audio = np.concatenate(audios) if audios else np.zeros(0, dtype=np.float32)
    print("Full Audio Samples:", len(full_audio), "Duration:", len(full_audio)/24000, "sec")

asyncio.run(synth_all())
