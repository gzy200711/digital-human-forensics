from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError

from apps.api.config import Settings, get_settings
from apps.api.db import get_session_factory
from apps.api.repository import JobRepository
from apps.api.services.reporting import render_static_report
from apps.api.services.storage import ObjectStore, build_object_store, derived_key
from dhfp.domain import ForensicReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetentionSweepResult:
    scanned: int = 0
    purged: int = 0
    skipped: int = 0
    failed_job_ids: list[str] = field(default_factory=list)


async def purge_expired_media(
    *,
    settings: Settings | None = None,
    storage: ObjectStore | None = None,
    limit: int = 100,
) -> RetentionSweepResult:
    """Delete expired raw media and thumbnails while preserving reports and audits."""
    current_settings = settings or get_settings()
    object_store = storage or build_object_store(current_settings)
    session_factory = get_session_factory()

    async with session_factory() as session:
        jobs = await JobRepository(session).list_expired_media_jobs(limit=limit)

    purged = 0
    skipped = 0
    failed: list[str] = []
    for job in jobs:
        deleted_at = datetime.now(UTC)
        keys = [
            job.object_key,
            derived_key(job.id, "heatmap.png"),
            derived_key(job.id, "keyframe.jpg"),
        ]
        try:
            await asyncio.gather(
                *(
                    asyncio.to_thread(object_store.delete_prefix, key)
                    for key in dict.fromkeys(keys)
                )
            )
        except Exception:
            logger.exception("Failed to purge expired media for job %s", job.id)
            failed.append(job.id)
            continue

        details = {
            "policy": "raw_media_retention",
            "retention_days": _retention_days(job.created_at, job.expires_at),
            "raw_media_deleted": True,
            "derived_thumbnails_deleted": True,
            "reports_preserved": True,
            "purged_at": deleted_at.isoformat(),
        }
        try:
            async with session_factory() as session:
                repository = JobRepository(session)
                updated_payload = await repository.mark_media_purged(
                    job.id,
                    raw_deleted_at=deleted_at,
                    derived_deleted_at=deleted_at,
                    details=details,
                )
                await repository.add_audit_log(
                    actor="system",
                    action="purge_expired_media",
                    resource_type="job",
                    resource_id=job.id,
                    details=details,
                )
                if updated_payload is not None:
                    report = ForensicReport.model_validate(updated_payload)
                    await asyncio.gather(
                        asyncio.to_thread(
                            object_store.put_bytes,
                            derived_key(job.id, "report.json"),
                            report.model_dump_json(indent=2).encode("utf-8"),
                            content_type="application/json",
                        ),
                        asyncio.to_thread(
                            object_store.put_bytes,
                            derived_key(job.id, "report.html"),
                            render_static_report(report).encode("utf-8"),
                            content_type="text/html",
                        ),
                    )
                await session.commit()
        except IntegrityError:
            skipped += 1
            continue
        except Exception:
            logger.exception("Failed to record media retention for job %s", job.id)
            failed.append(job.id)
            continue
        purged += 1

    return RetentionSweepResult(
        scanned=len(jobs),
        purged=purged,
        skipped=skipped,
        failed_job_ids=failed,
    )


def _retention_days(created_at: datetime | None, expires_at: datetime | None) -> int | None:
    if created_at is None or expires_at is None:
        return None
    delta = expires_at - created_at
    return max(0, int(round(delta.total_seconds() / 86400)))
