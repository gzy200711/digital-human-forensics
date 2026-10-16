from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Appeal, AuditLog, Job, MediaRetentionRecord, ReportRecord


class JobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_job(
        self,
        *,
        job_id: str,
        original_filename: str,
        object_key: str,
        source_type: str,
        source_url: str | None,
        mime_type: str | None,
        size_bytes: int | None,
        retention_days: int,
    ) -> Job:
        job = Job(
            id=job_id,
            original_filename=original_filename,
            object_key=object_key,
            source_type=source_type,
            source_url=source_url,
            mime_type=mime_type,
            size_bytes=size_bytes,
            expires_at=datetime.now(UTC) + timedelta(days=retention_days),
        )
        self.session.add(job)
        await self.session.flush()
        return job

    async def get_job(self, job_id: str) -> Job | None:
        result = await self.session.execute(select(Job).where(Job.id == job_id))
        return result.scalar_one_or_none()

    async def update_job(self, job_id: str, **values: Any) -> Job | None:
        values["updated_at"] = datetime.now(UTC)
        await self.session.execute(update(Job).where(Job.id == job_id).values(**values))
        await self.session.flush()
        return await self.get_job(job_id)

    async def mark_processing(self, job_id: str, stage: str) -> None:
        now = datetime.now(UTC)
        await self.session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(
                status="processing",
                progress=max(1, 1),
                stage=stage,
                started_at=now,
                updated_at=now,
                error_code=None,
                error_message=None,
            )
        )
        await self.session.flush()

    async def mark_failed(self, job_id: str, code: str, message: str) -> None:
        now = datetime.now(UTC)
        await self.session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(
                status="failed",
                stage="处理失败",
                error_code=code,
                error_message=message[:2000],
                completed_at=now,
                updated_at=now,
            )
        )
        await self.session.flush()

    async def save_report(self, job_id: str, payload: dict[str, Any]) -> ReportRecord:
        existing = await self.get_report(job_id)
        if existing is not None:
            await self.session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(
                    status="completed",
                    progress=100,
                    stage="检测完成",
                    completed_at=existing.created_at,
                    updated_at=datetime.now(UTC),
                )
            )
            await self.session.flush()
            return existing

        report = ReportRecord(
            job_id=job_id,
            schema_version=payload.get("schema_version", "1.0"),
            verdict=payload["verdict"],
            calibrated_score=float(payload["calibrated_score"]),
            payload=payload,
        )
        self.session.add(report)
        await self.session.flush()
        await self.session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(
                status="completed",
                progress=100,
                stage="检测完成",
                completed_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
        return report

    async def get_report(self, job_id: str) -> ReportRecord | None:
        result = await self.session.execute(
            select(ReportRecord).where(ReportRecord.job_id == job_id)
        )
        return result.scalar_one_or_none()

    async def create_appeal(
        self,
        *,
        job_id: str,
        reason: str,
        contact: str | None,
    ) -> Appeal:
        appeal = Appeal(job_id=job_id, reason=reason, contact=contact)
        self.session.add(appeal)
        await self.session.flush()
        return appeal

    async def add_audit_log(
        self,
        *,
        actor: str,
        action: str,
        resource_type: str,
        resource_id: str | None,
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
    ) -> AuditLog:
        audit_log = AuditLog(
            actor=actor,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
        )
        self.session.add(audit_log)
        await self.session.flush()
        return audit_log

    async def list_expired_media_jobs(self, *, limit: int = 100) -> list[Job]:
        now = datetime.now(UTC)
        result = await self.session.execute(
            select(Job)
            .outerjoin(MediaRetentionRecord, MediaRetentionRecord.job_id == Job.id)
            .where(
                Job.expires_at.is_not(None),
                Job.expires_at <= now,
                Job.status.in_(("completed", "failed")),
                MediaRetentionRecord.job_id.is_(None),
            )
            .order_by(Job.expires_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_media_purged(
        self,
        job_id: str,
        *,
        raw_deleted_at: datetime,
        derived_deleted_at: datetime,
        details: dict[str, Any],
    ) -> dict[str, Any] | None:
        self.session.add(
            MediaRetentionRecord(
                job_id=job_id,
                raw_deleted_at=raw_deleted_at,
                derived_deleted_at=derived_deleted_at,
                details=details,
            )
        )
        report = await self.get_report(job_id)
        updated_payload = None
        if report is not None:
            payload = dict(report.payload)
            artifacts = dict(payload.get("artifacts") or {})
            artifacts["media_url"] = None
            artifacts["heatmap_url"] = None
            artifacts["keyframe_url"] = None
            payload["artifacts"] = artifacts
            processing = dict(payload.get("processing") or {})
            processing["media_retention"] = details
            payload["processing"] = processing
            report.payload = payload
            updated_payload = payload
        await self.session.flush()
        return updated_payload

    async def delete_jobs(self, job_ids: list[str]) -> None:
        if not job_ids:
            return
        await self.session.execute(delete(Job).where(Job.id.in_(job_ids)))
