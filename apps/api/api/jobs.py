from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path
from typing import BinaryIO
from urllib.parse import unquote, urlparse
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, Response
from pydantic import AnyHttpUrl, TypeAdapter, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.api.dependencies import get_object_store, get_session
from apps.api.config import get_settings
from apps.api.repository import JobRepository
from apps.api.schemas import AppealCreate, AppealResponse, JobCreateResponse, JobStatusResponse
from apps.api.services.pipeline import validate_remote_url
from apps.api.services.queue import enqueue_job
from apps.api.services.storage import (
    ObjectStore,
    StorageError,
    job_prefix,
    raw_key,
    safe_filename,
)

router = APIRouter(prefix="/v1", tags=["jobs"])

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
}


class UploadTooLargeError(RuntimeError):
    pass


class LimitedReader:
    def __init__(self, stream: BinaryIO, max_bytes: int) -> None:
        self.stream = stream
        self.max_bytes = max_bytes
        self.total = 0

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.total += len(data)
        if self.total > self.max_bytes:
            raise UploadTooLargeError
        return data


@router.post(
    "/jobs",
    response_model=JobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_job(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    source_url: str | None = Form(default=None),
    retention_days: int | None = Form(default=None, ge=1, le=365),
    consent_ack: bool = Form(default=False),
    x_actor: str | None = Header(default=None, alias="X-Actor"),
    session: AsyncSession = Depends(get_session),
    storage: ObjectStore = Depends(get_object_store),
) -> JobCreateResponse:
    settings = get_settings()
    repository = JobRepository(session)
    if not consent_ack:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "CONSENT_REQUIRED",
                "message": "提交前必须确认拥有素材处理权限。",
            },
        )
    if file is None and not source_url:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "SOURCE_REQUIRED",
                "message": "必须上传文件或提供 source_url。",
            },
        )
    if file is not None and source_url:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "MULTIPLE_SOURCES",
                "message": "文件上传与 URL 只能二选一。",
            },
        )

    resolved_retention = retention_days or settings.raw_retention_days
    filename: str
    mime_type: str | None
    size_bytes: int | None
    object_key: str
    source_type: str
    job_id = str(uuid4())

    if file is not None:
        filename = _safe_filename(file.filename or "upload.bin")
        mime_type = (file.content_type or mimetypes.guess_type(filename)[0] or "").lower() or None
        _validate_media_type(filename, mime_type, settings.allowed_media_type_set)
        object_key = raw_key(job_id, filename)
        try:
            limited = LimitedReader(file.file, settings.max_upload_bytes)
            size_bytes = await asyncio.to_thread(
                storage.put_stream,
                object_key,
                limited,
                content_type=mime_type,
            )
        except UploadTooLargeError as exc:
            storage.delete_prefix(job_prefix(job_id))
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "MEDIA_TOO_LARGE",
                    "message": f"上传文件超过 {settings.max_upload_bytes} 字节限制。",
                },
            ) from exc
        except Exception as exc:
            storage.delete_prefix(job_prefix(job_id))
            raise HTTPException(
                status_code=502,
                detail={"error": "STORAGE_WRITE_FAILED", "message": str(exc)},
            ) from exc
        source_type = "upload"
        source_url_value = None
    else:
        try:
            validated_url = TypeAdapter(AnyHttpUrl).validate_python(source_url)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail={"error": "URL_INVALID", "message": "source_url 格式无效。"},
            ) from exc
        validated_url_string = str(validated_url)
        await asyncio.to_thread(
            validate_remote_url,
            validated_url_string,
            allow_private=settings.allow_private_urls,
        )
        parsed_name = (
            Path(unquote(urlparse(validated_url_string).path)).name or "remote-media"
        )
        filename = _safe_filename(parsed_name)
        mime_type = mimetypes.guess_type(filename)[0]
        object_key = raw_key(job_id, filename)
        size_bytes = None
        source_type = "url"
        source_url_value = validated_url_string

    job = await repository.create_job(
        job_id=job_id,
        original_filename=filename,
        object_key=object_key,
        source_type=source_type,
        source_url=source_url_value,
        mime_type=mime_type,
        size_bytes=size_bytes,
        retention_days=resolved_retention,
    )
    await repository.add_audit_log(
        actor=x_actor or "anonymous",
        action="create_job",
        resource_type="job",
        resource_id=job.id,
        details={
            "source_type": source_type,
            "filename": filename,
            "size_bytes": size_bytes,
            "retention_days": resolved_retention,
        },
        ip_address=_client_ip(request),
    )
    await session.commit()
    try:
        enqueue_job(job.id, background_tasks)
    except Exception as exc:
        await repository.mark_failed(
            job.id,
            "QUEUE_ENQUEUE_FAILED",
            f"任务已创建，但入队失败: {type(exc).__name__}",
        )
        await session.commit()
        raise HTTPException(
            status_code=503,
            detail={
                "error": "QUEUE_ENQUEUE_FAILED",
                "message": "任务暂时无法入队，请稍后重试。",
            },
        ) from exc
    return JobCreateResponse(
        job_id=job.id,
        status="queued",
        status_url=f"/v1/jobs/{job.id}",
        report_url=f"/v1/jobs/{job.id}/report",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: str,
    session: AsyncSession = Depends(get_session),
) -> JobStatusResponse:
    repository = JobRepository(session)
    job = await repository.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "JOB_NOT_FOUND", "message": "任务不存在。"},
        )
    report = await repository.get_report(job_id)
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,  # type: ignore[arg-type]
        progress=job.progress,
        stage=job.stage,
        original_filename=job.original_filename,
        mime_type=job.mime_type,
        size_bytes=job.size_bytes,
        error_code=job.error_code,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        expires_at=job.expires_at,
        report_ready=report is not None,
    )


@router.get("/jobs/{job_id}/report")
async def get_report(
    job_id: str,
    request: Request,
    x_actor: str | None = Header(default=None, alias="X-Actor"),
    session: AsyncSession = Depends(get_session),
) -> JSONResponse:
    repository = JobRepository(session)
    job = await repository.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "JOB_NOT_FOUND", "message": "任务不存在。"},
        )
    report = await repository.get_report(job_id)
    if report is None:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "REPORT_NOT_READY",
                "message": f"任务当前状态为 {job.status}，报告尚未生成。",
            },
        )
    await repository.add_audit_log(
        actor=x_actor or "anonymous",
        action="view_report",
        resource_type="job",
        resource_id=job_id,
        details={"verdict": report.verdict, "schema_version": report.schema_version},
        ip_address=_client_ip(request),
    )
    await session.commit()
    return JSONResponse(
        content=report.payload,
        headers={"Cache-Control": "private, no-store"},
    )


@router.post(
    "/jobs/{job_id}/appeal",
    response_model=AppealResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_appeal(
    job_id: str,
    payload: AppealCreate,
    request: Request,
    x_actor: str | None = Header(default=None, alias="X-Actor"),
    session: AsyncSession = Depends(get_session),
) -> AppealResponse:
    repository = JobRepository(session)
    job = await repository.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "JOB_NOT_FOUND", "message": "任务不存在。"},
        )
    report = await repository.get_report(job_id)
    if report is None:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "REPORT_NOT_READY",
                "message": "报告生成后才能提交申诉。",
            },
        )
    appeal = await repository.create_appeal(
        job_id=job_id,
        reason=payload.reason,
        contact=payload.contact,
    )
    await repository.add_audit_log(
        actor=x_actor or "anonymous",
        action="submit_appeal",
        resource_type="job",
        resource_id=job_id,
        details={"appeal_id": appeal.id, "has_contact": bool(payload.contact)},
        ip_address=_client_ip(request),
    )
    await session.commit()
    return AppealResponse(
        appeal_id=appeal.id,
        job_id=job_id,
        status=appeal.status,
        created_at=appeal.created_at,
    )


@router.get("/assets/{key:path}", response_class=Response)
async def get_asset(
    key: str,
    request: Request,
    storage: ObjectStore = Depends(get_object_store),
) -> Response:
    try:
        total_size = await asyncio.to_thread(storage.size, key)
    except (StorageError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": "ASSET_NOT_FOUND", "message": "派生产物不存在。"},
        ) from exc
    content_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
    headers = {
        "Cache-Control": "private, max-age=300",
        "Accept-Ranges": "bytes",
    }
    range_header = request.headers.get("range")
    if range_header:
        try:
            byte_range = _parse_byte_range(range_header, total_size)
        except ValueError as exc:
            raise HTTPException(
                status_code=416,
                detail="Range 格式无效",
                headers={"Content-Range": f"bytes */{total_size}"},
            ) from exc
        start, end = byte_range
        payload = await asyncio.to_thread(storage.read_range, key, start, end)
        headers["Content-Range"] = f"bytes {start}-{end}/{total_size}"
        headers["Content-Length"] = str(len(payload))
        return Response(
            content=payload,
            status_code=206,
            media_type=content_type,
            headers=headers,
        )
    payload = await asyncio.to_thread(storage.read_bytes, key)
    headers["Content-Length"] = str(len(payload))
    if content_type == "text/html":
        headers["Content-Security-Policy"] = (
            "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
            "font-src 'none'; frame-ancestors 'none'"
        )
    return Response(content=payload, media_type=content_type, headers=headers)


def _validate_media_type(
    filename: str,
    mime_type: str | None,
    allowed_types: set[str],
) -> None:
    suffix = Path(filename).suffix.lower()
    mime_allowed = mime_type in allowed_types if mime_type else False
    extension_allowed = suffix in ALLOWED_EXTENSIONS
    if mime_type == "application/octet-stream" and extension_allowed:
        return
    if not mime_allowed or not extension_allowed:
        raise HTTPException(
            status_code=415,
            detail={
                "error": "MEDIA_TYPE_UNSUPPORTED",
                "message": f"不支持的文件类型: {mime_type or suffix or 'unknown'}",
            },
        )


def _parse_byte_range(value: str, total_size: int) -> tuple[int, int]:
    unit, separator, raw_range = value.partition("=")
    normalized = raw_range.strip()
    if (
        unit.strip().lower() != "bytes"
        or not separator
        or total_size <= 0
        or "," in normalized
        or normalized.count("-") != 1
    ):
        raise ValueError("unsupported range")
    start_text, end_text = (part.strip() for part in normalized.split("-", maxsplit=1))
    if not start_text:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise ValueError("invalid suffix range")
        start = max(0, total_size - suffix_length)
        end = total_size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else total_size - 1
        end = min(end, total_size - 1)
    if start < 0 or start > end:
        raise ValueError("range is outside the resource")
    return start, end


def _safe_filename(filename: str) -> str:
    return safe_filename(filename)


def _client_ip(request: Request) -> str | None:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", maxsplit=1)[0].strip()
    return request.client.host if request.client else None
