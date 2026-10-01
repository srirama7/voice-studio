"""Video Assembly & SRT Subtitle Generation Module for Voice Studio.

Provides:
- FFmpeg slide image + audio merging with slide duration + 0.5s padding.
- Standard SRT subtitle generator from audio chunk timing.
- H.264 MP4 video rendering with fallback Pillow slide frame generator.
"""

from __future__ import annotations

import datetime
import logging
import math
import os
from pathlib import Path
import subprocess
import time
import wave
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds float into standard SRT timestamp string format: HH:MM:SS,mmm.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Formatted SRT timecode string (e.g., '00:01:23,456').
    """
    if seconds < 0:
        seconds = 0.0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))

    if millis >= 1000:
        secs += 1
        millis = 0
    if secs >= 60:
        minutes += 1
        secs = 0
    if minutes >= 60:
        hours += 1
        minutes = 0

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class SRTGenerator:
    """SRT Subtitle File Generator for chunked speech narration."""

    @staticmethod
    def generate_srt(
        chunks: List[str],
        chunk_durations: List[float],
        output_srt_path: Union[str, Path],
        start_delay: float = 0.0,
        pause_between: float = 0.2,
    ) -> Path:
        """Generate formatted SRT subtitle file from text chunks and their durations.

        Args:
            chunks: List of clean text chunk strings.
            chunk_durations: List of durations in seconds for each chunk.
            output_srt_path: Target path for .srt file.
            start_delay: Initial offset in seconds.
            pause_between: Pause duration between consecutive chunks.

        Returns:
            Path object pointing to written SRT file.
        """
        output_srt_path = Path(output_srt_path)
        output_srt_path.parent.mkdir(parents=True, exist_ok=True)

        srt_entries = []
        current_time = start_delay

        for i, (chunk, duration) in enumerate(zip(chunks, chunk_durations), start=1):
            chunk_text = chunk.strip()
            if not chunk_text:
                continue

            dur = max(0.5, duration)
            start_str = format_srt_timestamp(current_time)
            end_time = current_time + dur
            end_str = format_srt_timestamp(end_time)

            entry = f"{i}\n{start_str} --> {end_str}\n{chunk_text}\n"
            srt_entries.append(entry)

            current_time = end_time + pause_between

        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_entries))

        return output_srt_path


class SlideImageGenerator:
    """Generates high-definition slide images (1920x1080) for video assembly."""

    @staticmethod
    def create_slide_image(
        title: str,
        body_text: str,
        output_path: Union[str, Path],
        slide_num: int = 1,
        total_slides: int = 1,
        width: int = 1920,
        height: int = 1080,
    ) -> Path:
        """Create a professional presentation slide image card using Pillow.

        Args:
            title: Title text for the slide.
            body_text: Body text content.
            output_path: File path to write the PNG image.
            slide_num: Current slide number.
            total_slides: Total slides count.
            width: Image canvas width in pixels.
            height: Image canvas height in pixels.

        Returns:
            Path object of generated PNG file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create dark gradient background image canvas
        img = Image.new("RGB", (width, height), color=(15, 23, 42))  # Slate dark background
        draw = ImageDraw.Draw(img)

        # Draw decorative banner line
        draw.rectangle([80, 140, width - 80, 144], fill=(59, 130, 246))  # Accent blue

        # Load font (fallback to default font)
        try:
            font_title = ImageFont.truetype("arial.ttf", 60)
            font_body = ImageFont.truetype("arial.ttf", 36)
            font_footer = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            font_title = ImageFont.load_default()
            font_body = ImageFont.load_default()
            font_footer = ImageFont.load_default()

        # Draw Title
        title_text = title if title else f"Slide {slide_num}"
        draw.text((80, 60), title_text, fill=(248, 250, 252), font=font_title)

        # Draw Body Text (wrapped)
        if body_text:
            lines = body_text.splitlines()
            wrapped_lines = []
            for line in lines:
                words = line.split()
                curr_line = ""
                for word in words:
                    if len(curr_line) + len(word) + 1 <= 65:
                        curr_line = f"{curr_line} {word}".strip()
                    else:
                        wrapped_lines.append(curr_line)
                        curr_line = word
                if curr_line:
                    wrapped_lines.append(curr_line)

            y = 180
            for line in wrapped_lines[:14]:  # Max lines
                draw.text((80, y), line, fill=(203, 213, 225), font=font_body)
                y += 50

        # Draw Footer
        footer_text = f"Voice Studio | Slide {slide_num} of {total_slides}"
        draw.text((80, height - 60), footer_text, fill=(100, 116, 139), font=font_footer)

        img.save(str(output_path), "PNG")
        return output_path


def _read_ffmpeg_progress(progress_path: Union[str, Path], total_sec: float) -> float:
    """Parse an ffmpeg ``-progress`` file into a 0.0..1.0 completion fraction.

    Returns 0.0 when the file is missing/unparseable (encoder just started).
    """
    try:
        lines = Path(progress_path).read_text(encoding="utf-8", errors="ignore").splitlines()
        ms = 0
        for line in reversed(lines):
            if line.startswith("out_time_ms="):
                try:
                    ms = int(line.split("=", 1)[1])
                except (ValueError, IndexError):
                    pass
                break
        if total_sec > 0:
            return max(0.0, min(1.0, (ms / 1000000.0) / total_sec))
    except Exception:
        pass
    return 0.0


# Cached hardware-encoder probe result: (encoder_name, extra_args).
_ENCODER_CACHE: Optional[Tuple[str, List[str]]] = None


def pick_video_encoder() -> Tuple[str, List[str]]:
    """Pick the fastest WORKING H.264 encoder on this machine.

    Probes Windows hardware encoders with a 1s test render and caches the
    winner; falls back to libx264 (always present). Typical result on an
    Intel/AMD Windows box: ``h264_mf`` at 3-5x libx264 speed for 1080p.
    Returns (encoder_name, extra_codec_args).
    """
    global _ENCODER_CACHE
    if _ENCODER_CACHE is not None:
        return _ENCODER_CACHE
    candidates = [
        ("h264_mf", ["-b:v", "6M"]),    # Windows MediaFoundation (GPU/driver)
        ("h264_qsv", ["-b:v", "6M"]),   # Intel Quick Sync
        ("h264_amf", ["-b:v", "6M"]),   # AMD AMF
    ]
    for name, args in candidates:
        try:
            cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                   "-f", "lavfi", "-i", "testsrc=duration=1:size=1280x720:rate=10",
                   "-c:v", name] + args + ["-pix_fmt", "yuv420p", "-f", "null", "-"]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, timeout=60)
            if res.returncode == 0:
                _ENCODER_CACHE = (name, args)
                logger.info("Hardware video encoder available: %s", name)
                return _ENCODER_CACHE
        except Exception as exc:
            logger.debug("Encoder %s probe failed: %s", name, exc)
            continue
    _ENCODER_CACHE = ("libx264", [])
    logger.info("No hardware encoder usable; falling back to libx264.")
    return _ENCODER_CACHE


class VideoAssembly:
    """FFmpeg-based Video Assembly Engine for merging slides, audio, and subtitles."""

    @staticmethod
    def get_audio_duration(audio_path: Union[str, Path]) -> float:
        """Read exact duration in seconds from WAV audio file."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            return 0.0

        try:
            with wave.open(str(audio_path), "rb") as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            return 0.0

    @staticmethod
    def assemble(
        slides_data: List[Dict],
        cleaned_chunks: List[str],
        audio_path: Union[str, Path],
        output_dir: Union[str, Path],
        job_id: str,
        padding_sec: float = 0.5,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, str]:
        """Assemble complete MP4 presentation video with audio and subtitles.

        Args:
            slides_data: List of slide dictionaries (from DocumentParser).
            cleaned_chunks: List of text chunk strings.
            audio_path: Path to mastered WAV audio file.
            output_dir: Output directory for video and subtitles.
            job_id: Unique job identifier.
            padding_sec: Extra padding time in seconds per slide (default 0.5s).
            progress_callback: Optional ``(fraction 0..1, message)`` hook fired
                ~every 2s while ffmpeg encodes, so the UI never looks stuck.

        Returns:
            Dictionary containing 'video_path' and 'subtitle_path'.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = Path(audio_path)
        total_audio_duration = VideoAssembly.get_audio_duration(audio_path)
        if total_audio_duration <= 0.0:
            total_audio_duration = 5.0  # Fallback duration

        num_slides = max(1, len(slides_data))
        # Compute duration per slide with +0.5s padding rule applied
        base_slide_duration = (total_audio_duration / float(num_slides)) + padding_sec

        # 1. Generate Slide PNG Images
        slide_image_paths: List[Path] = []
        slides_dir = output_dir / "slides"
        slides_dir.mkdir(parents=True, exist_ok=True)

        for idx, slide in enumerate(slides_data, start=1):
            img_path = slides_dir / f"slide_{idx:03d}.png"
            title = slide.get("title", f"Slide {idx}")
            body = slide.get("body_text", "")
            SlideImageGenerator.create_slide_image(
                title=title,
                body_text=body,
                output_path=img_path,
                slide_num=idx,
                total_slides=num_slides,
            )
            slide_image_paths.append(img_path)

        if not slide_image_paths:
            img_path = slides_dir / "slide_001.png"
            SlideImageGenerator.create_slide_image(
                title="Presentation",
                body_text="",
                output_path=img_path,
                slide_num=1,
                total_slides=1,
            )
            slide_image_paths.append(img_path)

        # 2. Generate SRT Subtitle file
        chunk_dur = total_audio_duration / float(max(1, len(cleaned_chunks)))
        chunk_durations = [chunk_dur] * len(cleaned_chunks)
        srt_path = output_dir / f"{job_id}.srt"
        SRTGenerator.generate_srt(
            chunks=cleaned_chunks,
            chunk_durations=chunk_durations,
            output_srt_path=srt_path,
        )

        # 3. Create FFmpeg concat input file list
        concat_list_path = output_dir / "slides_concat.txt"
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for img_p in slide_image_paths:
                # Escape backslashes for FFmpeg file path format
                escaped_path = str(img_p.resolve()).replace("\\", "/")
                f.write(f"file '{escaped_path}'\n")
                f.write(f"duration {base_slide_duration:.3f}\n")
            # Repeat last file for FFmpeg concat duration bug workaround
            escaped_path = str(slide_image_paths[-1].resolve()).replace("\\", "/")
            f.write(f"file '{escaped_path}'\n")

        # 4. Render H.264 MP4 using FFmpeg CLI (hardened + live progress).
        # Hardening vs the old code that stalled forever on long videos:
        # -nostdin (never block on console input), -loglevel error (no
        # chatter), stdout/stderr appended to a log FILE (a full OS pipe
        # buffer wedged the encoder on 20+ min videos), -progress file so
        # the UI bar keeps moving. Videos over 10 min use ultrafast
        # (~2x quicker encode, still H.264) since still slides compress well.
        output_mp4_path = output_dir / f"{job_id}.mp4"
        ffmpeg_log = output_dir / "ffmpeg.log"
        progress_path = output_dir / "ffmpeg_progress.txt"

        # Speed recipe for the 88%+ stage (still slides need few frames):
        # - Long videos (>10 min): 5 fps instead of 15 -> 3x fewer frames
        #   (slide cuts look identical), ultrafast + fastest HW encoder.
        # - Short videos: keep 1080p15 veryfast for maximum quality.
        # - Hardware encoder (MediaFoundation/QuickSync/AMF) is typically
        #   3-5x libx264 for 1080p; probed once, then cached.
        long_video = total_audio_duration > 600
        fps = 5 if long_video else 15
        # Always use libx264 for image concat rendering to avoid h264_mf GPU hangs
        vcodec_args = ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage", "-crf", "23"]

        def _build_cmd(first_image: Optional[Path] = None) -> List[str]:
            if first_image is None:
                return [
                    "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-f", "concat", "-safe", "0",
                    "-i", str(concat_list_path),
                    "-i", str(audio_path),
                    "-t", f"{total_audio_duration:.3f}",
                ] + vcodec_args + [
                    "-r", str(fps), "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k",
                    "-progress", str(progress_path), "-nostats",
                    "-shortest", str(output_mp4_path),
                ]
            return [
                "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-loop", "1", "-i", str(first_image),
                "-i", str(audio_path),
                "-t", f"{total_audio_duration:.3f}",
            ] + vcodec_args + [
                "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
                "-progress", str(progress_path), "-nostats",
                "-shortest", str(output_mp4_path),
            ]

        def _run(cmd: List[str], timeout: int, tag: str) -> bool:
            try:
                progress_path.unlink(missing_ok=True)
            except Exception:
                pass
            logger.info("Executing FFmpeg %s command: %s", tag, " ".join(cmd))
            try:
                with open(ffmpeg_log, "ab") as logf:
                    proc = subprocess.Popen(
                        cmd, stdin=subprocess.DEVNULL,
                        stdout=logf, stderr=subprocess.STDOUT,
                    )
                    t0 = time.time()
                    while True:
                        rc = proc.poll()
                        frac = _read_ffmpeg_progress(progress_path, total_audio_duration)
                        if progress_callback is not None:
                            try:
                                progress_callback(frac, f"Encoding video ({frac * 100:.0f}%)")
                            except Exception:
                                pass
                        if rc is not None:
                            return rc == 0
                        if time.time() - t0 > timeout:
                            logger.warning("FFmpeg %s timed out after %ds; killing.", tag, timeout)
                            try:
                                proc.kill()
                            except Exception:
                                pass
                            return False
                        time.sleep(2)
            except FileNotFoundError:
                logger.error("ffmpeg binary not found on PATH.")
                return False
            except Exception as exc:
                logger.error("FFmpeg %s execution error: %s", tag, exc)
                return False

        ok = _run(_build_cmd(), timeout=1800, tag="render")
        if not ok:
            logger.warning("Main FFmpeg render failed; trying single-image fallback.")
            ok = _run(_build_cmd(slide_image_paths[0]), timeout=900, tag="fallback")
        if not ok:
            logger.error("All FFmpeg renders failed; continuing without video.")

        return {
            "video_path": str(output_mp4_path) if output_mp4_path.exists() else "",
            "subtitle_path": str(srt_path) if srt_path.exists() else "",
        }
