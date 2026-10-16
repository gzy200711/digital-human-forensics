"""对单个媒体文件运行溯源核验：c2patool 清单检查 + 隐式水印核验服务。

用法：

    uv run python scripts/verify_provenance.py sample.mp4
    uv run python scripts/verify_provenance.py sample.mp4 --json

阴性结果只说明本次未检出，不作为内容真实的证据；服务不可用会明确标注。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from apps.api.config import get_settings
from apps.api.detector_plugins.base import PluginContext
from apps.api.detector_plugins.provenance import ProvenanceAnalyzer
from apps.api.services.media import MediaProbeError, probe_media


def build_context(media_path: Path) -> PluginContext:
    settings = get_settings()
    probe = probe_media(media_path)
    return PluginContext(
        job_id="provenance-cli",
        original_filename=media_path.name,
        source_object_key=f"provenance/{media_path.name}",
        media_path=media_path,
        probe=probe,
        settings=settings,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="运行 c2patool 与隐式水印核验，输出溯源检查结果。"
    )
    parser.add_argument("media", type=Path, help="待核验的媒体文件")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而不是文本表格")
    args = parser.parse_args(argv)

    media_path: Path = args.media
    if not media_path.exists():
        print(f"媒体文件不存在：{media_path}", file=sys.stderr)
        return 2

    settings = get_settings()
    tool = shutil.which("c2patool")
    try:
        context = build_context(media_path)
    except MediaProbeError as exc:
        print(f"媒体探测失败：{exc}", file=sys.stderr)
        return 2

    result = ProvenanceAnalyzer().analyze(context)
    if args.json:
        print(
            json.dumps(
                {
                    "media": str(media_path),
                    "c2patool": tool,
                    "watermark_provider": settings.watermark_provider,
                    "watermark_media_upload_allowed": settings.watermark_allow_media_upload,
                    "explicit_ai_indicators": result.explicit_ai_indicators,
                    "checks": result.checks,
                    "render_chain_hits": result.render_chain_hits,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(f"媒体：{media_path}")
    print(f"c2patool：{tool or '未安装（C2PA 检查将标记为不可用）'}")
    print(f"水印核验服务：{settings.watermark_provider}")
    print(f"允许上传原始媒体：{settings.watermark_allow_media_upload}")
    print("")
    for check in result.checks:
        flag = "阳性" if check["positive"] else "非阳性"
        print(f"[{check['status']}/{flag}] {check['name']}")
        print(f"  {check['explanation']}")
    print("")
    if result.explicit_ai_indicators:
        print("显式生成合成标识：")
        for indicator in result.explicit_ai_indicators:
            print(f"  - {indicator}")
    else:
        print("显式生成合成标识：未命中（阴性结果不证明内容真实）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
