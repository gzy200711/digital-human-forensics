from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import AnyHttpUrl, BaseModel, Field


class JobCreateOptions(BaseModel):
    retention_days: int | None = Field(default=None, ge=1, le=365)
    consent_ack: bool = False


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    status_url: str
    report_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    progress: int
    stage: str
    original_filename: str
    mime_type: str | None = None
    size_bytes: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    expires_at: datetime | None = None
    report_ready: bool


class AppealCreate(BaseModel):
    reason: str = Field(min_length=10, max_length=5000)
    contact: str | None = Field(default=None, max_length=320)


class AppealResponse(BaseModel):
    appeal_id: str
    job_id: str
    status: str
    created_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app: str
    environment: str
    database: str
    storage: str
    queue_backend: str
    storage_backend: str
    version: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class URLJobCreate(BaseModel):
    source_url: AnyHttpUrl
    retention_days: int | None = Field(default=None, ge=1, le=365)
    consent_ack: bool = False
