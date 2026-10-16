from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("media") / "synthetic.mp4"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=12",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=16000",
        "-t",
        "1.5",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        "-y",
        str(output),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.skip(f"FFmpeg 无法生成测试视频: {completed.stderr}")
    return output


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    database_path = tmp_path / "test.db"
    storage_path = tmp_path / "storage"
    monkeypatch.setenv(
        "DATABASE_URL",
        f"sqlite+aiosqlite:///{database_path}",
    )
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_ROOT", str(storage_path))
    monkeypatch.setenv("QUEUE_BACKEND", "inline")
    monkeypatch.setenv("CALIBRATION_PATH", str(Path.cwd() / "configs/calibration.json"))
    monkeypatch.setenv("FUSION_CONFIG_PATH", str(Path.cwd() / "configs/fusion.json"))
    monkeypatch.setenv("SAMPLE_FPS", "2")
    monkeypatch.setenv("MAX_SAMPLED_FRAMES", "20")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000")
    # 单元测试必须与开发机的模型配置隔离，保证基线结果可复现（空值 = 未启用）。
    monkeypatch.setenv("VISUAL_ONNX_PATH", "")
    monkeypatch.setenv("AUDIO_SPOOF_ONNX_PATH", "")

    from apps.api.api.dependencies import get_object_store
    from apps.api.config import get_settings

    get_settings.cache_clear()
    get_object_store.cache_clear()
    return tmp_path


@pytest.fixture()
def api_client(isolated_env, monkeypatch: pytest.MonkeyPatch):
    import asyncio

    from fastapi.testclient import TestClient

    from apps.api.db import dispose_db
    from apps.api.main import app

    async def clean_database() -> None:
        await dispose_db()

    asyncio.run(clean_database())
    with TestClient(app) as client:
        yield client
    asyncio.run(clean_database())
