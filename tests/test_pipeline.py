from __future__ import annotations

import asyncio

from apps.api.config import get_settings
from apps.api.db import dispose_db, get_session_factory, init_db
from apps.api.repository import JobRepository
from apps.api.services.pipeline import DetectionPipeline
from apps.api.services.storage import build_object_store, raw_key


def test_pipeline_returns_unknown_for_no_face_video(
    isolated_env,
    synthetic_video,
) -> None:
    async def run() -> dict:
        await init_db()
        settings = get_settings()
        storage = build_object_store(settings)
        key = raw_key("test-job", synthetic_video.name)
        with synthetic_video.open("rb") as handle:
            storage.put_stream(key, handle, content_type="video/mp4")
        session_factory = get_session_factory()
        async with session_factory() as session:
            repository = JobRepository(session)
            await repository.create_job(
                job_id="test-job",
                original_filename=synthetic_video.name,
                object_key=key,
                source_type="upload",
                source_url=None,
                mime_type="video/mp4",
                size_bytes=synthetic_video.stat().st_size,
                retention_days=1,
            )
            await session.commit()

        await DetectionPipeline(settings=settings, storage=storage).run("test-job")

        async with session_factory() as session:
            repository = JobRepository(session)
            job = await repository.get_job("test-job")
            report = await repository.get_report("test-job")
            assert job is not None
            assert report is not None
            assert job.status == "completed"
            assert job.started_at is not None
            return report.payload

    payload = asyncio.run(run())
    assert payload["verdict"] == "无法确定"
    assert payload["assessability"]["status"] == "不可鉴定"
    assert payload["review"]["status"] == "not_triggered"
    assert any(item["signal"] == "visual_artifact" for item in payload["evidence"])
    asyncio.run(dispose_db())
