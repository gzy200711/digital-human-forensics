from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    app_name: str = "数字人鉴别平台"
    app_base_url: str = "http://localhost:8000"
    web_base_url: str = "http://localhost:3000"
    secret_key: str = "change-me"

    database_url: str = "sqlite+aiosqlite:///./data/forensics.db"
    queue_backend: str = "inline"
    redis_url: str = "redis://localhost:6379/0"
    # inline 队列下同时执行的检测任务数上限；批量提交时防止任务互相抢 CPU 拖慢整体。
    job_parallelism: int = Field(default=4, ge=1, le=32)

    storage_backend: str = "local"
    local_storage_root: str = "./data/derived/object-store"
    s3_endpoint_url: str | None = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "forensics"
    s3_region: str = "us-east-1"

    max_upload_bytes: int = 500 * 1024 * 1024
    allowed_media_types: str = (
        "video/mp4,video/quicktime,video/x-matroska,video/webm,video/x-msvideo,"
        "video/avi,video/x-m4v,audio/wav,audio/x-wav,audio/wave,audio/mpeg,"
        "audio/mp4,audio/aac,audio/flac"
    )
    raw_retention_days: int = Field(default=7, ge=1, le=365)
    retention_sweep_interval_seconds: int = Field(default=3600, ge=60, le=86400)
    allow_private_urls: bool = False
    url_download_timeout_seconds: int = Field(default=45, ge=5, le=300)
    max_media_duration_seconds: int = Field(default=900, ge=1, le=7200)

    sample_fps: float = Field(default=2.0, ge=0.25, le=12)
    temporal_fps: float = Field(default=4.0, ge=0.5, le=24)
    max_sampled_frames: int = Field(default=600, ge=8, le=5000)
    quality_pass_threshold: float = Field(default=0.62, ge=0, le=1)
    quality_marginal_threshold: float = Field(default=0.42, ge=0, le=1)

    calibration_path: str = "configs/calibration.json"
    fusion_config_path: str = "configs/fusion.json"
    visual_onnx_path: str | None = None
    onnx_input_size: int = Field(default=224, ge=64, le=1024)
    visual_onnx_license: str = "Apache-2.0"
    visual_onnx_source: str = "dima806/deepfake_vs_real_image_detection"

    audio_spoof_onnx_path: str | None = None
    audio_spoof_input_mode: str = "waveform"
    audio_spoof_sample_rate: int = Field(default=16000, ge=8000, le=48000)
    audio_spoof_segment_seconds: float = Field(default=4.04, gt=0.2, le=30)
    audio_spoof_max_segments: int = Field(default=8, ge=1, le=64)
    audio_spoof_aggregation: str = "max"
    audio_spoof_license: str = "MIT"
    audio_spoof_source: str = "AASIST"

    review_provider_mode: str = "rules"
    review_provider_endpoint: str | None = None
    review_provider_api_key: str | None = None
    review_provider_model: str | None = None
    review_provider_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    review_prompt_version: str = "evidence-review-v1"
    review_provider_license: str | None = None

    # 隐式水印核验（SynthID Detector / 联盟水印核验服务）。
    # 默认 off：不把原始媒体提交给任何外部服务。
    watermark_provider: str = "off"
    watermark_provider_label: str | None = None
    watermark_endpoint: str | None = None
    watermark_api_key: str | None = None
    watermark_app_id: str | None = None
    watermark_secret: str | None = None
    watermark_media_field: str = "file"
    watermark_extra_fields: dict[str, str] = Field(default_factory=dict)
    watermark_auth_header: str = "Authorization"
    watermark_auth_scheme: str = "Bearer"
    watermark_sign_requests: bool = True
    watermark_verify_tls: bool = True
    watermark_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    watermark_allow_media_upload: bool = False

    cors_origins: str = "http://localhost:3000"
    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def allowed_media_type_set(self) -> set[str]:
        return {
            item.strip().lower() for item in self.allowed_media_types.split(",") if item.strip()
        }

    @property
    def storage_root_path(self) -> Path:
        return Path(self.local_storage_root).expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
