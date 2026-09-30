"""
Unified Voice Studio Configuration Module.

Provides standard configuration parameters, environment variable loading,
Firebase settings, audio audio mastering targets, slide rendering dimensions,
and workspace path management matching UNIFIED_VOICE_STUDIO_MASTER_SPEC.md.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False


@dataclass
class FirebaseConfig:
    """Firebase Credentials & Project Details for voiceai-6c2e1."""
    project_id: str = "voiceai-6c2e1"
    storage_bucket: str = "voiceai-6c2e1.firebasestorage.app"
    auth_domain: str = "voiceai-6c2e1.firebaseapp.com"
    api_key: str = "AIzaSyBjYPWHb1FHs_yiztf7EV2vj8_N5uTCq_A"
    messaging_sender_id: str = "151528570993"
    app_id: str = "1:151528570993:web:dc7f27f76fc267d0509bd0"
    measurement_id: str = "G-6CDHNY3DM9"
    service_account_path: Optional[str] = None


@dataclass
class AudioQualityConfig:
    """Audio Quality Control (QC) Gates and DSP Loudness Normalization targets."""
    min_duration_sec: float = 8.0
    max_duration_sec: float = 30.0
    min_peak_dbfs: float = -18.0
    max_peak_dbfs: float = -3.0
    min_snr_db: float = 10.0
    
    # DSP Loudness Mastering targets
    target_lufs: float = -14.0
    lufs_tolerance: float = 0.5
    max_true_peak_dbtp: float = -1.5
    sample_rate: int = 22050
    channels: int = 1


@dataclass
class SlideRenderConfig:
    """Slide rendering parameters for visual visual asset generation."""
    target_width: int = 1920
    target_height: int = 1080
    dpi: int = 150
    output_format: str = "PNG"
    theme: str = "dark"  # dark / light template


@dataclass
class EngineConfig:
    """TTS engine default selection and operational modes."""
    primary_engine: str = "edge-tts-voiceclone"
    fallback_engine: str = "sapi-offline"
    default_profile_version: str = "v001"
    default_text_mode: str = "exact"  # exact vs llm_clean
    max_parallel_workers: int = 8


@dataclass
class PathConfig:
    """Workspace directories and cache path definitions."""
    base_dir: Path = field(default_factory=lambda: Path(
        os.getenv("VOICE_STUDIO_ROOT", str(Path(__file__).resolve().parent))))
    cache_dir: Path = field(init=False)
    voices_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.cache_dir = self.base_dir / "cache"
        self.voices_dir = self.cache_dir / "voices"
        self.output_dir = self.base_dir / "outputs"
        self.checkpoint_dir = self.base_dir / "checkpoints"


class Config:
    """
    Central Master Configuration Singleton/Container for Voice Studio.
    
    Loads configuration settings from environment variables with safe defaults
    for Firebase project voiceai-6c2e1, audio quality specifications,
    rendering specifications, and local folder management.
    """

    def __init__(self, env_file: Optional[str] = None) -> None:
        if _HAS_DOTENV and env_file and os.path.exists(env_file):
            load_dotenv(env_file)
        elif _HAS_DOTENV:
            load_dotenv()

        self.firebase = FirebaseConfig(
            project_id=os.getenv("FIREBASE_PROJECT_ID", "voiceai-6c2e1"),
            storage_bucket=os.getenv("FIREBASE_STORAGE_BUCKET", "voiceai-6c2e1.firebasestorage.app"),
            auth_domain=os.getenv("FIREBASE_AUTH_DOMAIN", "voiceai-6c2e1.firebaseapp.com"),
            api_key=os.getenv("FIREBASE_API_KEY", "AIzaSyBjYPWHb1FHs_yiztf7EV2vj8_N5uTCq_A"),
            messaging_sender_id=os.getenv("FIREBASE_MESSAGING_SENDER_ID", "151528570993"),
            app_id=os.getenv("FIREBASE_APP_ID", "1:151528570993:web:dc7f27f76fc267d0509bd0"),
            measurement_id=os.getenv("FIREBASE_MEASUREMENT_ID", "G-6CDHNY3DM9"),
            service_account_path=os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", None)
        )

        self.audio = AudioQualityConfig(
            min_duration_sec=float(os.getenv("QC_MIN_DURATION", "8.0")),
            max_duration_sec=float(os.getenv("QC_MAX_DURATION", "30.0")),
            min_peak_dbfs=float(os.getenv("QC_MIN_PEAK_DBFS", "-18.0")),
            max_peak_dbfs=float(os.getenv("QC_MAX_PEAK_DBFS", "-3.0")),
            min_snr_db=float(os.getenv("QC_MIN_SNR_DB", "10.0")),
            target_lufs=float(os.getenv("DSP_TARGET_LUFS", "-14.0")),
            max_true_peak_dbtp=float(os.getenv("DSP_MAX_TRUE_PEAK", "-1.5"))
        )

        self.slide = SlideRenderConfig(
            target_width=int(os.getenv("SLIDE_WIDTH", "1920")),
            target_height=int(os.getenv("SLIDE_HEIGHT", "1080")),
            dpi=int(os.getenv("SLIDE_DPI", "150"))
        )

        self.engine = EngineConfig(
            primary_engine=os.getenv("PRIMARY_TTS_ENGINE", "edge-tts-voiceclone"),
            fallback_engine=os.getenv("FALLBACK_TTS_ENGINE", "sapi-offline"),
            default_profile_version=os.getenv("DEFAULT_PROFILE_VERSION", "v001"),
            default_text_mode=os.getenv("DEFAULT_TEXT_MODE", "exact")
        )

        base_path = Path(os.getenv(
            "VOICE_STUDIO_ROOT", str(Path(__file__).resolve().parent)))
        self.paths = PathConfig(base_dir=base_path)
        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Create workspace directories if they do not exist."""
        for path in [
            self.paths.base_dir,
            self.paths.cache_dir,
            self.paths.voices_dir,
            self.paths.output_dir,
            self.paths.checkpoint_dir
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a serializable dictionary."""
        return {
            "firebase": {
                "project_id": self.firebase.project_id,
                "storage_bucket": self.firebase.storage_bucket,
                "auth_domain": self.firebase.auth_domain,
            },
            "audio": {
                "min_duration_sec": self.audio.min_duration_sec,
                "max_duration_sec": self.audio.max_duration_sec,
                "min_peak_dbfs": self.audio.min_peak_dbfs,
                "max_peak_dbfs": self.audio.max_peak_dbfs,
                "min_snr_db": self.audio.min_snr_db,
                "target_lufs": self.audio.target_lufs,
                "max_true_peak_dbtp": self.audio.max_true_peak_dbtp,
            },
            "slide": {
                "target_width": self.slide.target_width,
                "target_height": self.slide.slide_height if hasattr(self.slide, "slide_height") else self.slide.target_height,
                "dpi": self.slide.dpi,
            },
            "engine": {
                "primary_engine": self.engine.primary_engine,
                "fallback_engine": self.engine.fallback_engine,
                "default_profile_version": self.engine.default_profile_version,
                "default_text_mode": self.engine.default_text_mode,
            },
            "paths": {
                "base_dir": str(self.paths.base_dir),
                "cache_dir": str(self.paths.cache_dir),
                "voices_dir": str(self.paths.voices_dir),
                "output_dir": str(self.paths.output_dir),
                "checkpoint_dir": str(self.paths.checkpoint_dir),
            }
        }


# Default global instance
default_config = Config()
