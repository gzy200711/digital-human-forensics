from __future__ import annotations

import asyncio

from celery import Celery

from apps.api.config import get_settings

settings = get_settings()
celery_app = Celery(
    "digital_human_forensics",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)


@celery_app.task(name="forensics.process_job", autoretry_for=(Exception,), max_retries=1)
def process_job(job_id: str) -> None:
    from apps.api.services.pipeline import DetectionPipeline

    asyncio.run(DetectionPipeline().run(job_id))
