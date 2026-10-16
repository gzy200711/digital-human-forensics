from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from dhfp.evaluation.schema import DatasetRecord
from dhfp.plugins.base import DetectorUnavailable, EvaluationPrediction


class ExternalCommandDetector:
    def __init__(self, manifest: dict) -> None:
        self.detector_id = manifest["id"]
        self.display_name = manifest["display_name"]
        self.version = manifest.get("version", "external")
        self.license = manifest.get("license", "unknown")
        self.description = manifest.get("description", "")
        self.command = manifest["command"]
        self.timeout_seconds = int(manifest.get("timeout_seconds", 600))
        self.required_paths = [
            Path(path).expanduser() for path in manifest.get("required_paths", [])
        ]
        self.env = {str(key): str(value) for key, value in manifest.get("env", {}).items()}

    def predict(self, record: DatasetRecord) -> EvaluationPrediction:
        missing = [str(path) for path in self.required_paths if not path.exists()]
        if missing:
            raise DetectorUnavailable("缺少模型或仓库路径: " + ", ".join(missing))
        arguments = [
            part.replace("{media_path}", record.media_path) for part in self.command
        ]
        executable = arguments[0]
        if shutil.which(executable) is None and not Path(executable).exists():
            raise DetectorUnavailable(f"命令不存在: {executable}")
        started = time.perf_counter()
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env={**os.environ, **self.env} if self.env else None,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        if completed.returncode != 0:
            raise DetectorUnavailable(
                completed.stderr.strip()[-2000:] or f"退出码 {completed.returncode}"
            )
        try:
            payload = json.loads(completed.stdout.strip())
        except json.JSONDecodeError as exc:
            raise DetectorUnavailable("外部检测器未返回 JSON") from exc
        if "score" not in payload:
            raise DetectorUnavailable("外部检测器 JSON 缺少 score 字段")
        return EvaluationPrediction(
            sample_id=record.sample_id,
            score=float(payload["score"]),
            available=True,
            latency_ms=latency_ms,
            details=payload.get("details", {}),
        )


def command_text(command: list[str]) -> str:
    return shlex.join(command)
