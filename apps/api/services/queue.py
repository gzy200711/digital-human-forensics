from __future__ import annotations

import asyncio

from fastapi import BackgroundTasks

from apps.api.config import get_settings

_semaphore: asyncio.Semaphore | None = None
_semaphore_size: int | None = None


def _get_semaphore(size: int) -> asyncio.Semaphore:
    """按配置并发上限复用同一信号量，限制 inline 队列同时运行的检测任务数。"""
    global _semaphore, _semaphore_size
    if _semaphore is None or _semaphore_size != size:
        _semaphore = asyncio.Semaphore(size)
        _semaphore_size = size
    return _semaphore


async def _run_inline(job_id: str) -> None:
    from apps.api.services.pipeline import DetectionPipeline

    limit = max(1, get_settings().job_parallelism)
    async with _get_semaphore(limit):
        await DetectionPipeline().run(job_id)


def enqueue_job(job_id: str, background_tasks: BackgroundTasks) -> None:
    settings = get_settings()
    if settings.queue_backend.lower() == "celery":
        from apps.api.tasks import process_job

        process_job.delay(job_id)
        return
    background_tasks.add_task(_run_inline, job_id)
