"""
Voice Studio Enterprise Web Application.

ElevenLabs & Sarvam AI Inspired Interface:
1. Voice Cloning Studio: Zero-shot voice profile enrollment & quality control gates.
2. Presentation & Speech Studio: Document-to-voice & video presentation synthesis with voice selector.
3. Job Monitor Studio: Checkpoint state tracking & crash recovery.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any

import gradio as gr

from config import Config, default_config
from voice_enrollment import VoiceEnrollmentEngine, QCResult, VoiceProfile
from voice_clone import GALLERY_VOICES
from job_engine import JobEngine, JobStatus, JobCheckpoint, JobManifest
from document_parser import DocumentParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Initialize master singletons
config = default_config
enrollment_engine = VoiceEnrollmentEngine(config=config)
job_engine = JobEngine(config=config)

# Register the 31 stock gallery voices in the database (Firestore + local
# cache) so the full library is stored, not just user-enrolled clones.
# Skips docs already present; never breaks startup if the DB is unreachable.
try:
    from voice_clone import sync_gallery_to_db as _sync_gallery
    _gallery_sync = _sync_gallery(enrollment_engine.db_client)
    logger.info("Gallery voices in database: %s", _gallery_sync)
except Exception as _sync_err:
    logger.warning("Gallery DB sync skipped at startup: %s", _sync_err)


def list_available_voices() -> List[str]:
    """List enrolled voice profile IDs found in voices cache directory."""
    voices_dir = config.paths.voices_dir
    if not voices_dir.exists():
        return ["speaker_01"]
    voices = [d.name for d in voices_dir.iterdir() if d.is_dir()]
    return voices if voices else ["speaker_01"]


def all_voice_choices() -> List[tuple]:
    """Every saved voice for the single website dropdown (38 total).

    Enrolled clones keep plain voice-ID values; gallery stock voices use
    ``gallery:<id>`` values so the generation handler auto-routes them.
    All 38 are backed by database documents.
    """
    enrolled = list_available_voices()
    choices: List[tuple] = [(f"🎙️ {v} — my clone", v) for v in enrolled]
    for gid, info in GALLERY_VOICES.items():
        choices.append((f"⭐ {info['label']}", f"gallery:{gid}"))
    return choices


def default_voice_value() -> str:
    enrolled = list_available_voices()
    return enrolled[0] if enrolled else "gallery:prabhat_in_m"


def handle_voice_enrollment(
    audio_file: Optional[str],
    voice_id: str,
    version: str,
    strict_qc: bool,
) -> Tuple[str, Dict, Optional[str], Any]:
    """Handler for Tab 1: Voice Enrollment & QC Verification."""
    if not audio_file:
        return "### ❌ Error: Please upload or record a reference audio file.", {}, None, gr.Dropdown()

    if not voice_id or not voice_id.strip():
        voice_id = f"voice_{uuid.uuid4().hex[:8]}"
    if not version or not version.strip():
        version = "v001"

    try:
        profile: VoiceProfile = enrollment_engine.enroll_voice(
            audio_path=audio_file,
            voice_id=voice_id.strip(),
            version=version.strip(),
            strict_qc=strict_qc,
        )

        qc = profile.qc_result
        if qc.is_valid:
            status_md = (
                f"### 🎉 Voice Profile `{profile.voice_id}` Saved Successfully!\n\n"
                f"• **Voice ID:** `{profile.voice_id}`  \n"
                f"• **Version:** `{profile.version}`  \n"
                f"• **Database Status:** Saved to Firestore & Storage (`voiceai-6c2e1`)  \n"
                f"• **Canonical Audio:** `{profile.ref_wav_path}` (22.05kHz Mono)  \n"
                f"• **Conditioning Cache:** Zero-Shot Embeddings Pre-computed (`speaker_embedding.pth`)  \n\n"
                f"👉 **Voice Selected:** `{profile.voice_id}` is now available in the **Presentation Studio** dropdown!"
            )
        else:
            status_md = (
                f"### ⚠️ Voice Profile `{profile.voice_id}` Enrolled with QC Warnings\n\n"
                f"• **Errors:** {', '.join(qc.errors)}\n"
                f"• **Warnings:** {', '.join(qc.warnings)}"
            )

        voices = list_available_voices()
        dropdown_update = gr.Dropdown(choices=all_voice_choices(), value=profile.voice_id)
        return status_md, profile.to_dict(), profile.ref_wav_path, dropdown_update

    except Exception as exc:
        logger.error("Enrollment failed: %s", exc)
        return f"### ❌ Enrollment Failed\n**Error:** `{exc}`", {"error": str(exc)}, None, gr.Dropdown()


def handle_document_generation(
    document_file: Optional[str],
    voice_id: str,
    text_mode: str,
    language: str,
    progress=gr.Progress(),
) -> Tuple[str, Optional[str], Optional[str], Optional[str], Dict]:
    """Handler for Tab 2: Document Generation & Video Assembly."""
    if not document_file:
        return "❌ Please upload a presentation document file (.pptx, .pdf, .txt, .md).", None, None, None, {}

    # Single-dropdown routing: "gallery:<id>" values speak via the stock
    # gallery voice (no enrollment needed); anything else is an enrolled
    # clone from the database.
    gallery_voice = ""
    if voice_id and voice_id.startswith("gallery:"):
        gallery_voice = voice_id.split(":", 1)[1]
        if gallery_voice not in GALLERY_VOICES:
            return f"❌ Unknown gallery voice `{gallery_voice}`.", None, None, None, {}
        voice_id = f"gallery:{gallery_voice}"
    else:
        if not voice_id or not voice_id.strip():
            voice_id = "speaker_01"

    if not language or not language.strip():
        language = "en-in"

    job_id = f"job_{uuid.uuid4().hex[:8]}"

    def ui_progress_callback(status: JobStatus, pct: float, msg: str) -> None:
        progress(pct, desc=f"[{status.value}] {msg}")

    try:
        manifest: JobManifest = job_engine.run_job(
            job_id=job_id,
            document_path=document_file,
            voice_id=voice_id.strip(),
            text_mode=text_mode,
            language=language.strip(),
            gallery_voice=gallery_voice,
            progress_callback=ui_progress_callback,
        )

        status_msg = (
            f"✅ Presentation Generation Complete!\n"
            f"• Job ID: {manifest.job_id}\n"
            f"• Voice Profile Used: {manifest.voice_id}\n"
            f"• Language/Accent: {language.strip()}\n"
            f"• Cloning Engine: {job_engine.tts_synthesizer.primary_adapter.adapter_name}\n"
            f"• Voice Reference: {job_engine.resolve_voice_ref(manifest.voice_id) or '⚠️ NOT FOUND (base voice used)'}\n"
            f"• Total Duration: {manifest.total_duration_sec:.2f}s\n"
            f"• Slides Processed: {manifest.slide_count}\n"
            f"• Text Chunks: {manifest.chunk_count}"
        )

        audio_res = manifest.mastered_audio_file if Path(manifest.mastered_audio_file).exists() else None
        video_res = manifest.video_file if Path(manifest.video_file).exists() else None
        srt_res = manifest.subtitle_file if Path(manifest.subtitle_file).exists() else None

        return status_msg, audio_res, video_res, srt_res, manifest.to_dict()

    except Exception as exc:
        logger.error("Document generation error: %s", exc)
        return f"❌ Generation Failed: {exc}", None, None, None, {"error": str(exc)}


def handle_job_monitor(job_id_search: str) -> Tuple[str, Dict, Optional[str], Optional[str], Optional[str]]:
    """Handler for Tab 3: Job Monitor & Crash Recovery Inspection."""
    if not job_id_search or not job_id_search.strip():
        return "⚠️ Please enter a valid Job ID.", {}, None, None, None

    job_id = job_id_search.strip()
    checkpoint = job_engine.load_checkpoint(job_id)

    if checkpoint is None:
        return f"❌ Job `{job_id}` not found in checkpoints repository.", {}, None, None, None

    status_summary = (
        f"### Job Details: `{checkpoint.job_id}`\n"
        f"- **Status:** `{checkpoint.status.value}`\n"
        f"- **Document:** `{checkpoint.document_path}`\n"
        f"- **Voice ID:** `{checkpoint.voice_id}`\n"
        f"- **Mode:** `{checkpoint.text_mode}`\n"
        f"- **Completed Steps:** {', '.join(checkpoint.completed_steps) or 'None'}\n"
        f"- **Updated At:** `{checkpoint.updated_at}`"
    )

    audio_res = checkpoint.mastered_audio_path if checkpoint.mastered_audio_path and Path(checkpoint.mastered_audio_path).exists() else None
    video_res = checkpoint.video_path if checkpoint.video_path and Path(checkpoint.video_path).exists() else None
    srt_res = checkpoint.subtitle_path if checkpoint.subtitle_path and Path(checkpoint.subtitle_path).exists() else None

    return status_summary, checkpoint.to_dict(), audio_res, video_res, srt_res


def refresh_voice_list() -> Any:
    return gr.Dropdown(choices=all_voice_choices(), value=default_voice_value())


def handle_voice_delete(voice_id: str, confirm: bool) -> Tuple[str, Any, Any]:
    """Handler: delete a voice profile from the library (both tabs refresh)."""
    if not voice_id or not voice_id.strip():
        return "⚠️ Select a voice to delete.", gr.Dropdown(), gr.Dropdown()
    if not confirm:
        return f"⚠️ Tick **Confirm delete** to permanently remove `{voice_id.strip()}`.", gr.Dropdown(), gr.Dropdown()
    try:
        removed = enrollment_engine.delete_voice(voice_id.strip())
        dd = gr.Dropdown(choices=all_voice_choices(), value=default_voice_value())
        if removed:
            return f"🗑️ Voice `{voice_id.strip()}` deleted from library.", dd, dd
        return f"⚠️ Voice `{voice_id.strip()}` was not found.", dd, dd
    except Exception as exc:
        logger.error("Delete failed: %s", exc)
        return f"❌ Delete failed: `{exc}`", gr.Dropdown(), gr.Dropdown()


def handle_voice_rename(old_id: str, new_id: str) -> Tuple[str, Any, Any]:
    """Handler: rename a voice profile ID (both tabs refresh)."""
    if not old_id or not old_id.strip() or not new_id or not new_id.strip():
        return "⚠️ Select a voice and type a new ID.", gr.Dropdown(), gr.Dropdown()
    try:
        final = enrollment_engine.rename_voice(old_id.strip(), new_id.strip())
        dd = gr.Dropdown(choices=all_voice_choices(), value=final)
        return f"✏️ Renamed `{old_id.strip()}` → `{final}`.", dd, dd
    except Exception as exc:
        logger.error("Rename failed: %s", exc)
        return f"❌ Rename failed: `{exc}`", gr.Dropdown(), gr.Dropdown()


# Build Studio App
def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
    )

    with gr.Blocks(title="Voice Studio Enterprise") as demo:
        gr.Markdown(
            """
            # 🎙️ Voice Studio Enterprise
            ### Next-Gen Zero-Shot Voice Cloning & Presentation Synthesis (ElevenLabs & Sarvam AI Powered)
            """
        )

        with gr.Tabs():
            # --- TAB 1: INSTANT VOICE CLONING STUDIO ---
            with gr.Tab("🎙️ Instant Voice Cloning & Library"):
                gr.Markdown("### Create and Manage Custom Zero-Shot Voice Profiles")
                
                gr.Markdown(
                    """
                    > 📜 **Reading Passage Prompt (Read Aloud While Recording):**  
                    > *"When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon."*
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="🎙️ Record Voice Sample via Microphone or Upload WAV File (8s - 30s)",
                        )
                        voice_id_input = gr.Textbox(
                            label="Voice Profile ID",
                            value="speaker_01",
                            placeholder="e.g. narrator_alex",
                        )
                        version_input = gr.Textbox(
                            label="Profile Version",
                            value="v001",
                        )
                        strict_qc_checkbox = gr.Checkbox(
                            label="Enforce Quality Control Gates (Duration, Peak, SNR)",
                            value=True,
                        )
                        enroll_button = gr.Button("✨ Save Voice Profile to Database", variant="primary")

                    with gr.Column(scale=1):
                        enroll_status_output = gr.Markdown("Ready to capture voice sample.")
                        qc_metrics_json = gr.JSON(label="QC Metrics & Profile Data")
                        ref_audio_output = gr.Audio(label="Canonical Reference WAV (22.05kHz Mono)", interactive=False)

                gr.Markdown("### 📚 Voice Library — Edit / Delete Profiles")
                with gr.Row():
                    with gr.Column(scale=2):
                        lib_voice_select = gr.Dropdown(
                            choices=list_available_voices(),
                            value=(list_available_voices()[0] if list_available_voices() else "speaker_01"),
                            label="Select Voice to Manage",
                        )
                        new_name_input = gr.Textbox(
                            label="New Voice ID (for Rename)",
                            placeholder="e.g. narrator_alex",
                        )
                    with gr.Column(scale=2):
                        confirm_delete = gr.Checkbox(label="Confirm delete (permanent)", value=False)
                        with gr.Row():
                            rename_button = gr.Button("✏️ Rename", variant="secondary")
                            delete_button = gr.Button("🗑️ Delete", variant="stop")
                        lib_status_output = gr.Markdown("Pick a voice to rename or delete.")

            # --- TAB 2: PRESENTATION & SPEECH STUDIO ---
            with gr.Tab("📄 Presentation & Speech Studio"):
                gr.Markdown("### Synthesize Presentations into Audio & Video with Any Enrolled Voice")
                with gr.Row():
                    with gr.Column(scale=1):
                        doc_file_input = gr.File(
                            label="Upload Presentation Document (.pptx, .pdf, .txt, .md)",
                            file_types=[".pptx", ".pdf", ".txt", ".md"],
                        )
                        
                        with gr.Row():
                            voice_id_select = gr.Dropdown(
                                choices=all_voice_choices(),
                                value=default_voice_value(),
                                label=f"🗣️ Voice — all {len(all_voice_choices())} saved voices",
                                info="🎙️ = your enrolled clone from the database · ⭐ = stock gallery voice (no enrollment needed). Every voice can read English, Hindi, Kannada — pick language below.",
                                scale=4,
                            )
                            refresh_voices_btn = gr.Button("🔄 Refresh List", scale=1)

                        text_mode_radio = gr.Radio(
                            choices=["exact", "narrative"],
                            value="exact",
                            label="Text Processing Mode",
                            info="'exact' = verbatim text; 'narrative' = cleans markdown, stage directions, speaker labels.",
                        )
                        language_select = gr.Dropdown(
                            choices=["en-in", "hi", "kn", "ta", "te", "ml", "mr", "bn", "gu", "en"],
                            value="en-in",
                            label="🌍 Language / Accent",
                            info="Text language to speak. Gallery voices read any of these; cloned voices morph your timbre onto it.",
                        )
                        generate_button = gr.Button("⚡ Generate Voice Presentation", variant="primary")

                    with gr.Column(scale=1):
                        gen_status_output = gr.Markdown("Ready to generate presentation.")
                        mastered_audio_player = gr.Audio(label="Mastered Speech Audio (-14 LUFS)", interactive=False)
                        rendered_video_player = gr.Video(label="H.264 MP4 Presentation Video")
                        srt_file_download = gr.File(label="Download Subtitle (.srt)")
                        manifest_json_output = gr.JSON(label="Job Manifest Details")

            # --- TAB 3: ENTERPRISE JOB MONITOR ---
            with gr.Tab("🔍 Job Monitor & Crash Recovery"):
                gr.Markdown("### Inspect State Machine Execution & Checkpoints")
                with gr.Row():
                    with gr.Column(scale=1):
                        job_id_input = gr.Textbox(
                            label="Job ID",
                            placeholder="e.g. job_a1b2c3d4",
                        )
                        inspect_button = gr.Button("Inspect Job Checkpoint", variant="secondary")

                    with gr.Column(scale=1):
                        monitor_status_output = gr.Markdown("Enter Job ID to view state.")
                        checkpoint_json_output = gr.JSON(label="Checkpoint State JSON")
                        monitored_audio = gr.Audio(label="Mastered Audio", interactive=False)
                        monitored_video = gr.Video(label="Rendered Video")
                        monitored_srt = gr.File(label="SRT Subtitle File")

        # Wire Event Handlers
        enroll_button.click(
            fn=handle_voice_enrollment,
            inputs=[audio_input, voice_id_input, version_input, strict_qc_checkbox],
            outputs=[enroll_status_output, qc_metrics_json, ref_audio_output, voice_id_select],
        )

        refresh_voices_btn.click(
            fn=refresh_voice_list,
            inputs=[],
            outputs=[voice_id_select],
        )

        delete_button.click(
            fn=handle_voice_delete,
            inputs=[lib_voice_select, confirm_delete],
            outputs=[lib_status_output, lib_voice_select, voice_id_select],
        )

        rename_button.click(
            fn=handle_voice_rename,
            inputs=[lib_voice_select, new_name_input],
            outputs=[lib_status_output, lib_voice_select, voice_id_select],
        )

        generate_button.click(
            fn=handle_document_generation,
            inputs=[doc_file_input, voice_id_select, text_mode_radio, language_select],
            outputs=[
                gen_status_output,
                mastered_audio_player,
                rendered_video_player,
                srt_file_download,
                manifest_json_output,
            ],
        )

        inspect_button.click(
            fn=handle_job_monitor,
            inputs=[job_id_input],
            outputs=[
                monitor_status_output,
                checkpoint_json_output,
                monitored_audio,
                monitored_video,
                monitored_srt,
            ],
        )

    return demo


app = build_app()

if __name__ == "__main__":
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")
    # 0.0.0.0 + $PORT: runs on localhost AND on cloud hosts
    # (e.g. Hugging Face Spaces). Local behavior is unchanged.
    port = int(os.environ.get("PORT", "7860"))
    # SHARE_PUBLIC=1 -> also creates a temporary public link (no account
    # needed) for sharing straight from this PC. It lasts 72h and works
    # only while this PC stays on.
    share = os.environ.get("SHARE_PUBLIC", "0") == "1"
    app.launch(server_name="0.0.0.0", server_port=port, share=share, theme=theme)
