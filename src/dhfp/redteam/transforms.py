from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_PERTURBATIONS = (
    "jpeg_compression",
    "low_resolution",
    "screen_recording",
    "subtitle_overlay",
    "color_grading",
    "noise_perturbation",
)


@dataclass(frozen=True)
class RedTeamJob:
    case_id: str
    family: str
    description: str
    output_path: Path
    command: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    subtitle_text: str | None = None


@dataclass(frozen=True)
class RedTeamResult:
    case_id: str
    family: str
    output_path: str
    success: bool
    return_code: int
    error: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_redteam_plan(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    families: list[str] | None = None,
    subtitle_text: str = "红队测试字幕",
) -> list[RedTeamJob]:
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"输入媒体不存在: {source}")
    selected = families or list(SUPPORTED_PERTURBATIONS)
    invalid = sorted(set(selected) - set(SUPPORTED_PERTURBATIONS))
    if invalid:
        raise ValueError(f"不支持的红队扰动: {', '.join(invalid)}")

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs: list[RedTeamJob] = []
    for family in selected:
        jobs.append(_build_job(source, output_root, family, subtitle_text))
    return jobs


def apply_redteam_plan(
    jobs: list[RedTeamJob],
    *,
    ffmpeg_binary: str = "ffmpeg",
    timeout_seconds: int = 300,
    continue_on_error: bool = True,
) -> list[RedTeamResult]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds 必须大于 0")
    results: list[RedTeamResult] = []
    for job in jobs:
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        if job.subtitle_text:
            _write_subtitle(job.output_path.parent / "redteam.srt", job.subtitle_text)
        try:
            completed = subprocess.run(
                [ffmpeg_binary, *job.command],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            success = completed.returncode == 0 and job.output_path.exists()
            error = None if success else _tail(completed.stderr)
        except (OSError, subprocess.TimeoutExpired) as exc:
            completed = None
            success = False
            error = str(exc)
        results.append(
            RedTeamResult(
                case_id=job.case_id,
                family=job.family,
                output_path=str(job.output_path),
                success=success,
                return_code=completed.returncode if completed is not None else -1,
                error=error,
                metadata=job.metadata,
            )
        )
        if not success and not continue_on_error:
            break
    return results


def run_redteam_suite(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    families: list[str] | None = None,
    ffmpeg_binary: str = "ffmpeg",
    timeout_seconds: int = 300,
    subtitle_text: str = "红队测试字幕",
) -> list[RedTeamResult]:
    jobs = build_redteam_plan(
        input_path,
        output_dir,
        families=families,
        subtitle_text=subtitle_text,
    )
    return apply_redteam_plan(
        jobs,
        ffmpeg_binary=ffmpeg_binary,
        timeout_seconds=timeout_seconds,
    )


def write_redteam_manifest(
    results: list[RedTeamResult],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "supported_perturbations": list(SUPPORTED_PERTURBATIONS),
        "results": [result.as_dict() for result in results],
        "success_count": sum(1 for result in results if result.success),
        "failure_count": sum(1 for result in results if not result.success),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _build_job(
    source: Path,
    output_root: Path,
    family: str,
    subtitle_text: str,
) -> RedTeamJob:
    stem = source.stem
    common = ("-hide_banner", "-loglevel", "error", "-y", "-i", str(source))
    if family == "jpeg_compression":
        output = output_root / f"{stem}_jpeg.mkv"
        return RedTeamJob(
            case_id="jpeg_compression",
            family=family,
            description="MJPEG 重编码，模拟强 JPEG 压缩链路。",
            output_path=output,
            command=(
                *common,
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "mjpeg",
                "-q:v",
                "10",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                str(output),
            ),
            metadata={"codec": "mjpeg", "quality": 10},
        )
    if family == "low_resolution":
        output = output_root / f"{stem}_lowres.mp4"
        return RedTeamJob(
            case_id="low_resolution",
            family=family,
            description="缩放到 320 像素宽，模拟转发和低清播放。",
            output_path=output,
            command=(
                *common,
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-vf",
                "scale=320:-2:flags=bicubic",
                *_h264_args(),
                str(output),
            ),
            metadata={"width": 320, "filter": "bicubic"},
        )
    if family == "screen_recording":
        output = output_root / f"{stem}_screen.mp4"
        return RedTeamJob(
            case_id="screen_recording",
            family=family,
            description="降帧率、缩放到 640 像素宽并限制码率，模拟录屏链路。",
            output_path=output,
            command=(
                *common,
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-vf",
                "fps=12,scale=640:-2:flags=bilinear,eq=contrast=1.04:saturation=0.96",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-b:v",
                "900k",
                "-maxrate",
                "1100k",
                "-bufsize",
                "1800k",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-movflags",
                "+faststart",
                str(output),
            ),
            metadata={"fps": 12, "width": 640, "video_bitrate": "900k"},
        )
    if family == "subtitle_overlay":
        output = output_root / f"{stem}_subtitle.mp4"
        subtitle_path = output_root / "redteam.srt"
        escaped_path = _escape_filter_path(subtitle_path)
        return RedTeamJob(
            case_id="subtitle_overlay",
            family=family,
            description="烧录显式字幕，测试文字区域对定位和视觉分支的干扰。",
            output_path=output,
            command=(
                *common,
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-vf",
                f"subtitles={escaped_path}:force_style='FontName=Arial,FontSize=18,Outline=2'",
                *_h264_args(),
                str(output),
            ),
            metadata={"subtitle_text": subtitle_text},
            subtitle_text=subtitle_text,
        )
    if family == "color_grading":
        output = output_root / f"{stem}_color.mp4"
        return RedTeamJob(
            case_id="color_grading",
            family=family,
            description="提高对比度和饱和度，模拟调色和美颜后处理。",
            output_path=output,
            command=(
                *common,
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-vf",
                "eq=contrast=1.15:saturation=1.25:brightness=0.03",
                *_h264_args(),
                str(output),
            ),
            metadata={"contrast": 1.15, "saturation": 1.25, "brightness": 0.03},
        )
    if family == "noise_perturbation":
        output = output_root / f"{stem}_noise.mp4"
        return RedTeamJob(
            case_id="noise_perturbation",
            family=family,
            description="加入确定性轻量噪声；这是代理攻击，不宣称等同于梯度对抗样本。",
            output_path=output,
            command=(
                *common,
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-vf",
                "noise=alls=4:allf=t+u,eq=contrast=1.03:saturation=0.98",
                *_h264_args(),
                str(output),
            ),
            metadata={
                "noise_strength": 4,
                "attack_type": "surrogate",
                "limitation": "not gradient-based adversarial perturbation",
            },
        )
    raise ValueError(f"不支持的红队扰动: {family}")


def _h264_args() -> tuple[str, ...]:
    return (
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
    )


def _write_subtitle(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "1\n00:00:00,200 --> 00:00:01,500\n" + text.strip() + "\n",
        encoding="utf-8",
    )


def _escape_filter_path(path: Path) -> str:
    value = str(path)
    return (
        value.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(",", "\\,")
    )


def _tail(value: str, limit: int = 2000) -> str:
    return value[-limit:].strip() if value else "FFmpeg 执行失败"
