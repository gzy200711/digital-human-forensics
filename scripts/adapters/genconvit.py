from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from scripts.adapters._common import emit, fail, media_parser, resolve_repo


def main() -> None:
    args = media_parser("GenConViT 外部命令适配器")
    repo = resolve_repo("GENCONVIT_REPO", "third_party/GenConViT")
    media = Path(args.media).resolve()
    if not media.exists():
        fail(f"媒体不存在: {media}")
    with tempfile.TemporaryDirectory(prefix="genconvit-") as temporary:
        input_dir = Path(temporary) / "input"
        input_dir.mkdir()
        media_link = input_dir / media.name
        media_link.symlink_to(media)
        command = [
            os.environ.get("GENCONVIT_PYTHON", sys.executable),
            str(repo / "prediction.py"),
            "--p",
            str(input_dir),
            "--f",
            os.environ.get("GENCONVIT_FRAMES", "15"),
            "--e",
            "--v",
        ]
        started_at = time.time()
        completed = subprocess.run(
            command,
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("GENCONVIT_TIMEOUT", "900")),
        )
        if completed.returncode != 0:
            fail(completed.stderr[-4000:] or "GenConViT 推理失败")
        result_dir = repo / "result"
        candidates = sorted(
            (
                path
                for path in result_dir.glob("prediction_*.json")
                if path.stat().st_mtime >= started_at - 1
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            fail("GenConViT 未生成结果 JSON")
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
        video_payload = payload.get("video", payload)
        predictions = video_payload.get("pred", [])
        if not predictions:
            fail("GenConViT 结果缺少 pred 字段")
        score = max(float(value) for value in predictions)
        emit(
            score,
            {
                "result_file": str(candidates[0]),
                "frame_predictions": predictions,
            },
        )


if __name__ == "__main__":
    main()
