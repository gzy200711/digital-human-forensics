from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def media_parser(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--media", required=True)
    return parser.parse_args()


def emit(score: float, details: dict[str, Any]) -> None:
    score = max(0.0, min(1.0, float(score)))
    print(
        json.dumps(
            {"score": score, "details": details},
            ensure_ascii=False,
        )
    )


def fail(message: str, code: int = 2) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def resolve_repo(env_name: str, default: str) -> Path:
    path = Path(os.environ.get(env_name, default)).expanduser().resolve()
    if not path.exists():
        fail(f"仓库不存在: {path}")
    return path


def decode_audio_mono(path: str, sample_rate: int = 16000) -> list[float]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, timeout=300)
    if completed.returncode != 0:
        fail(completed.stderr.decode("utf-8", errors="replace"))
    import array

    values = array.array("f")
    values.frombytes(completed.stdout)
    return values.tolist()
