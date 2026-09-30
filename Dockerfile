# Render free-tier deploy (no card required). Uses system ffmpeg + Python.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py config.py voice_clone.py job_engine.py tts_adapter.py audio_dsp.py \
     text_processor.py document_parser.py slide_renderer.py video_assembly.py \
     voice_enrollment.py db_client.py README.md ./
# Render injects $PORT; /data keeps paths writable (disk is ephemeral on free tier).
ENV PORT=7860
CMD ["sh", "-c", "python app.py"]
