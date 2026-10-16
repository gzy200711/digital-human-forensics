from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from apps.api.api.dependencies import get_object_store
from apps.api.api.jobs import router as jobs_router
from apps.api.config import get_settings
from apps.api.db import dispose_db, get_session_factory, init_db
from apps.api.schemas import HealthResponse
from apps.api.services.pipeline import PipelineError
from apps.api.services.retention import purge_expired_media
from dhfp import __version__

logger = logging.getLogger(__name__)
settings = get_settings()


async def _retention_sweep_loop(interval_seconds: int) -> None:
    while True:
        try:
            result = await purge_expired_media()
            if result.purged or result.failed_job_ids:
                logger.info(
                    "Media retention sweep: scanned=%s purged=%s failed=%s",
                    result.scanned,
                    result.purged,
                    len(result.failed_job_ids),
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Media retention sweep failed")
        await asyncio.sleep(interval_seconds)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    retention_task = asyncio.create_task(
        _retention_sweep_loop(settings.retention_sweep_interval_seconds)
    )
    try:
        yield
    finally:
        retention_task.cancel()
        with suppress(asyncio.CancelledError):
            await retention_task
        await dispose_db()


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="证据优先的数字人 / 生成合成内容鉴别 API。",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Actor"],
)
app.include_router(jobs_router)


@app.get("/healthz", response_model=HealthResponse, tags=["system"])
async def healthz() -> HealthResponse:
    database_status = "ok"
    storage_status = "ok"
    overall = "ok"
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(text("SELECT 1"))
    except Exception:
        database_status = "error"
        overall = "degraded"
    try:
        await asyncio.to_thread(get_object_store().healthcheck)
    except Exception:
        storage_status = "error"
        overall = "degraded"
    return HealthResponse(
        status=overall,  # type: ignore[arg-type]
        app=settings.app_name,
        environment=settings.app_env,
        database=database_status,
        storage=storage_status,
        queue_backend=settings.queue_backend,
        storage_backend=settings.storage_backend,
        version=__version__,
    )


@app.exception_handler(PipelineError)
async def pipeline_error_handler(_: Request, exc: PipelineError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": exc.code,
            "message": exc.message,
            "detail": {},
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "REQUEST_VALIDATION_ERROR",
            "message": "请求参数不符合接口约束。",
            "detail": {"errors": exc.errors()},
        },
    )
