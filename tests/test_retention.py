from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from apps.api.config import get_settings
from apps.api.db import dispose_db, get_session_factory, init_db
from apps.api.models import AuditLog, MediaRetentionRecord
from apps.api.repository import JobRepository
from apps.api.services.retention import purge_expired_media
from apps.api.services.storage import build_object_store, derived_key, raw_key
from dhfp.domain import (
    ArtifactLinks,
    Assessability,
    ForensicReport,
    MediaInfo,
    Provenance,
)


def _report_payload(job_id: str) -> dict:
    report = ForensicReport(
        report_id=f"report-{job_id}",
        job_id=job_id,
        media=MediaInfo(filename="original.mp4", size_bytes=4),
        verdict="无法确定",
        confidence=0.5,
        calibrated_score=0.5,
        summary="测试报告",
        assessability=Assessability(
            score=0.0,
            status="不可鉴定",
            reasons=["测试"],
        ),
        provenance=Provenance(),
        recommended_action="人工复核",
        artifacts=ArtifactLinks(
            media_url=f"/v1/assets/jobs/{job_id}/raw/original.mp4",
            heatmap_url=f"/v1/assets/jobs/{job_id}/derived/heatmap.png",
            keyframe_url=f"/v1/assets/jobs/{job_id}/derived/keyframe.jpg",
            json_report_url=f"/v1/assets/jobs/{job_id}/derived/report.json",
            html_report_url=f"/v1/assets/jobs/{job_id}/derived/report.html",
        ),
    )
    return report.model_dump(mode="json")


def test_expired_media_is_purged_but_report_is_preserved(isolated_env) -> None:
    async def run() -> None:
        await init_db()
        settings = get_settings()
        storage = build_object_store(settings)
        session_factory = get_session_factory()
        raw_object = raw_key("expired-job", "original.mp4")
        heatmap_object = derived_key("expired-job", "heatmap.png")
        keyframe_object = derived_key("expired-job", "keyframe.jpg")
        report_json_object = derived_key("expired-job", "report.json")
        report_html_object = derived_key("expired-job", "report.html")
        for key in (
            raw_object,
            heatmap_object,
            keyframe_object,
            report_json_object,
            report_html_object,
        ):
            storage.put_bytes(key, b"data")

        async with session_factory() as session:
            repository = JobRepository(session)
            await repository.create_job(
                job_id="expired-job",
                original_filename="original.mp4",
                object_key=raw_object,
                source_type="upload",
                source_url=None,
                mime_type="video/mp4",
                size_bytes=4,
                retention_days=1,
            )
            await repository.save_report("expired-job", _report_payload("expired-job"))
            await repository.update_job(
                "expired-job",
                expires_at=datetime.now(UTC) - timedelta(minutes=1),
            )
            await session.commit()

        result = await purge_expired_media(settings=settings, storage=storage)

        assert result.purged == 1
        assert not storage.exists(raw_object)
        assert not storage.exists(heatmap_object)
        assert not storage.exists(keyframe_object)
        assert storage.exists(report_json_object)
        assert storage.exists(report_html_object)

        async with session_factory() as session:
            report = await JobRepository(session).get_report("expired-job")
            assert report is not None
            assert report.payload["artifacts"]["media_url"] is None
            assert report.payload["artifacts"]["heatmap_url"] is None
            assert report.payload["artifacts"]["keyframe_url"] is None
            assert report.payload["processing"]["media_retention"]["raw_media_deleted"] is True
            retention_count = await session.scalar(
                select(func.count()).select_from(MediaRetentionRecord)
            )
            audit_count = await session.scalar(
                select(func.count())
                .select_from(AuditLog)
                .where(AuditLog.action == "purge_expired_media")
            )
            assert retention_count == 1
            assert audit_count == 1

        # 第二次扫描不会重复清理或重复写审计。
        second = await purge_expired_media(settings=settings, storage=storage)
        assert second.scanned == 0
        await dispose_db()

    asyncio.run(run())
