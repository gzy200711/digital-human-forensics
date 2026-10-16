from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from apps.api.services.media import MediaProbe, MediaProbeError, probe_media
from dhfp.evaluation.schema import (
    AuthorizationStatus,
    DatasetLabel,
    DatasetRecord,
    ForgeryType,
    index_summary,
    load_index,
    write_index,
)

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
VARIANT_KEYS = {
    "avatar": "avatar",
    "voice": "voice",
    "ratio": "ratio",
    "aspect": "ratio",
    "bg": "background",
    "background": "background",
}


def parse_variant_tags(
    relative_path: Path,
    manifest_entry: dict | None = None,
) -> dict[str, str]:
    """从目录名解析变体标签，例如 `avatar=a1,voice=v2/ratio=9x16/bg=on`。"""
    tags: dict[str, str] = {}
    for part in relative_path.parts[:-1]:
        for token in part.split(","):
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            normalized = VARIANT_KEYS.get(key.strip().lower())
            if normalized and value.strip():
                tags[normalized] = value.strip()
    if manifest_entry:
        for key, value in manifest_entry.items():
            normalized = VARIANT_KEYS.get(str(key).strip().lower())
            if normalized and str(value).strip():
                tags[normalized] = str(value).strip()
    return tags


def variant_subset(tags: dict[str, str], *, prefix: str = "chatcut") -> str:
    if not tags:
        return f"{prefix}/default"
    ordered = [f"{key}={tags[key]}" for key in sorted(tags)]
    return f"{prefix}/" + "/".join(ordered)


def build_record(
    media_path: Path,
    probe: MediaProbe,
    *,
    sample_id: str,
    label: DatasetLabel,
    generator: str,
    generator_version: str | None,
    forgery_type: ForgeryType,
    tags: dict[str, str],
    subset_prefix: str,
    notes: str | None = None,
) -> DatasetRecord:
    video = probe.video_stream
    if video is None:
        raise ValueError(f"{media_path} 不含视频轨，无法加入评测集")
    duration = probe.duration_seconds or video.duration_seconds
    if not duration or duration <= 0:
        raise ValueError(f"{media_path} 缺少可用时长")
    width = video.width or 0
    height = video.height or 0
    frame_rate = video.frame_rate or 0
    if width <= 0 or height <= 0 or frame_rate <= 0:
        raise ValueError(f"{media_path} 缺少分辨率或帧率")
    created = datetime.fromtimestamp(media_path.stat().st_mtime, tz=UTC).date().isoformat()
    return DatasetRecord(
        sample_id=sample_id,
        media_path=str(media_path.resolve()),
        label=label,
        generator=generator,
        generator_version=generator_version,
        forgery_type=forgery_type,
        authorization_status=AuthorizationStatus.SELF_OWNED,
        duration_seconds=float(duration),
        width=int(width),
        height=int(height),
        frame_rate=float(frame_rate),
        has_audio=probe.has_audio,
        compression=video.codec_name,
        subset=variant_subset(tags, prefix=subset_prefix),
        generation_date=created,
        prompt_or_driver=json.dumps(tags, ensure_ascii=False) if tags else None,
        notes=notes,
    )


def collect_media(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def _manifest_entry(manifest: dict, relative_path: Path) -> dict | None:
    for key in (relative_path.as_posix(), relative_path.name, relative_path.stem):
        entry = manifest.get(key)
        if isinstance(entry, dict):
            return entry
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 ChatCut 量产输出整理成独立生成器族的评测集索引。"
    )
    parser.add_argument("--input", required=True, help="ChatCut 输出目录（可含变体子目录）")
    parser.add_argument("--output", required=True, help="输出 JSONL 索引路径")
    parser.add_argument("--manifest", help="可选变体清单 JSON：相对路径 -> 变体字段")
    parser.add_argument("--generator", default="chatcut")
    parser.add_argument("--generator-version")
    parser.add_argument(
        "--forgery-type",
        default="combined",
        choices=[item.value for item in ForgeryType],
    )
    parser.add_argument("--sample-prefix")
    parser.add_argument("--real-dir", help="可选真人对照组目录")
    parser.add_argument("--real-generator", default="authorized-real")
    parser.add_argument("--extra-index", help="可选已有索引，合并写入输出")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_root = Path(args.input).expanduser().resolve()
    if not input_root.exists():
        raise SystemExit(f"输入目录不存在: {input_root}")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8")) if args.manifest else {}
    prefix = args.sample_prefix or args.generator
    records: list[DatasetRecord] = []
    skipped: list[str] = []

    for media_path in collect_media(input_root):
        relative = media_path.relative_to(input_root)
        tags = parse_variant_tags(relative, _manifest_entry(manifest, relative))
        try:
            probe = probe_media(media_path)
            records.append(
                build_record(
                    media_path,
                    probe,
                    sample_id=f"{prefix}-{relative.with_suffix('').as_posix().replace('/', '-')}",
                    label=DatasetLabel.GENERATED,
                    generator=args.generator,
                    generator_version=args.generator_version,
                    forgery_type=ForgeryType(args.forgery_type),
                    tags=tags,
                    subset_prefix=args.generator,
                    notes="ChatCut 专属评测样本；按独立生成器族参与切分。",
                )
            )
        except (MediaProbeError, ValueError) as exc:
            skipped.append(f"{relative}: {exc}")

    if args.real_dir:
        real_root = Path(args.real_dir).expanduser().resolve()
        for media_path in collect_media(real_root):
            relative = media_path.relative_to(real_root)
            try:
                probe = probe_media(media_path)
                records.append(
                    build_record(
                        media_path,
                        probe,
                        sample_id=f"{args.real_generator}-{relative.with_suffix('').as_posix().replace('/', '-')}",
                        label=DatasetLabel.REAL,
                        generator=args.real_generator,
                        generator_version=None,
                        forgery_type=ForgeryType.NONE,
                        tags={},
                        subset_prefix="real",
                        notes="授权真人对照样本，用于与 ChatCut 生成族同强度重编码配对。",
                    )
                )
            except (MediaProbeError, ValueError) as exc:
                skipped.append(f"{relative}: {exc}")

    if args.extra_index:
        records = [*load_index(args.extra_index), *records]

    output = Path(args.output)
    write_index(records, output)
    summary = index_summary(records)
    print(f"索引已写入: {output.resolve()}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        "提示: ChatCut 必须在 split 中保持为独立生成器族，"
        "并配套同强度重编码增广（dhfp augment-reencode）后再训练。"
    )
    if skipped:
        print(f"跳过 {len(skipped)} 个文件:")
        for item in skipped[:20]:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
