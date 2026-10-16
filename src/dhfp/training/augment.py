from __future__ import annotations

import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dhfp.evaluation.schema import DatasetRecord


@dataclass(frozen=True)
class ReencodeSpec:
    """一次同强度重编码：正负样本必须使用完全相同的参数集合。"""

    name: str
    crf: int = 28
    preset: str = "veryfast"
    max_width: int | None = None
    fps: float | None = None
    audio_bitrate: str = "96k"

    def filters(self) -> list[str]:
        filters: list[str] = []
        if self.max_width:
            filters.append(f"scale='min({self.max_width},iw)':-2")
        if self.fps:
            filters.append(f"fps={self.fps}")
        return filters

    def compression_tag(self) -> str:
        details = [f"crf{self.crf}"]
        if self.max_width:
            details.append(f"w{self.max_width}")
        if self.fps:
            details.append(f"fps{self.fps:g}")
        return f"reencode:{self.name}:" + "-".join(details)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_SPECS: dict[str, ReencodeSpec] = {
    "crf23": ReencodeSpec(name="crf23", crf=23),
    "crf28": ReencodeSpec(name="crf28", crf=28),
    "crf32": ReencodeSpec(name="crf32", crf=32),
    "crf28-720p": ReencodeSpec(name="crf28-720p", crf=28, max_width=720),
}


@dataclass(frozen=True)
class AugmentJob:
    case_id: str
    sample_id: str
    label: str
    spec: ReencodeSpec
    source_path: Path
    output_path: Path
    command: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spec"] = self.spec.as_dict()
        payload["source_path"] = str(self.source_path)
        payload["output_path"] = str(self.output_path)
        payload["command"] = list(self.command)
        return payload


@dataclass
class AugmentResult:
    case_id: str
    sample_id: str
    spec_name: str
    output_path: str
    success: bool
    return_code: int
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_specs(value: str | None) -> list[ReencodeSpec]:
    if not value:
        return [DEFAULT_SPECS["crf28"]]
    specs: list[ReencodeSpec] = []
    for name in (item.strip() for item in value.split(",")):
        if not name:
            continue
        spec = DEFAULT_SPECS.get(name)
        if spec is None:
            raise ValueError(f"未知的重编码强度: {name}（可选 {', '.join(DEFAULT_SPECS)}）")
        specs.append(spec)
    if not specs:
        raise ValueError("至少需要一个重编码强度")
    return specs


def build_plan(
    records: list[DatasetRecord],
    output_dir: str | Path,
    specs: list[ReencodeSpec],
) -> list[AugmentJob]:
    """为每条记录生成同样的重编码强度集合，避免编码强度成为类别捷径。"""
    if not specs:
        raise ValueError("specs 不能为空")
    root = Path(output_dir).expanduser().resolve()
    jobs: list[AugmentJob] = []
    for record in records:
        source = Path(record.media_path).expanduser()
        for spec in specs:
            suffix = source.suffix or ".mp4"
            output = root / spec.name / f"{record.sample_id}__{spec.name}{suffix}"
            jobs.append(
                AugmentJob(
                    case_id=f"{record.sample_id}:{spec.name}",
                    sample_id=record.sample_id,
                    label=record.label.value,
                    spec=spec,
                    source_path=source,
                    output_path=output,
                    command=_ffmpeg_command(source, output, spec),
                )
            )
    return jobs


def apply_plan(
    jobs: list[AugmentJob],
    *,
    ffmpeg_binary: str = "ffmpeg",
    timeout_seconds: int = 900,
    continue_on_error: bool = True,
) -> list[AugmentResult]:
    results: list[AugmentResult] = []
    for job in jobs:
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not job.source_path.exists():
            results.append(
                AugmentResult(
                    case_id=job.case_id,
                    sample_id=job.sample_id,
                    spec_name=job.spec.name,
                    output_path=str(job.output_path),
                    success=False,
                    return_code=-1,
                    error=f"源文件不存在: {job.source_path}",
                )
            )
            if not continue_on_error:
                break
            continue
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
            return_code = completed.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            success = False
            error = str(exc)
            return_code = -1
        results.append(
            AugmentResult(
                case_id=job.case_id,
                sample_id=job.sample_id,
                spec_name=job.spec.name,
                output_path=str(job.output_path),
                success=success,
                return_code=return_code,
                error=error,
                metadata={"spec": job.spec.as_dict(), "label": job.label},
            )
        )
        if not success and not continue_on_error:
            break
    return results


def augmented_records(
    records: list[DatasetRecord],
    specs: list[ReencodeSpec],
    output_dir: str | Path,
) -> list[DatasetRecord]:
    """把重编码产物写回索引：标签与授权状态不变，只标记增广子集。"""
    root = Path(output_dir).expanduser().resolve()
    augmented: list[DatasetRecord] = []
    for record in records:
        source = Path(record.media_path).expanduser()
        for spec in specs:
            suffix = source.suffix or ".mp4"
            output = root / spec.name / f"{record.sample_id}__{spec.name}{suffix}"
            payload = record.model_dump()
            payload["sample_id"] = f"{record.sample_id}__{spec.name}"
            payload["media_path"] = str(output)
            payload["subset"] = f"{record.subset}+{spec.name}"
            payload["compression"] = spec.compression_tag()
            payload["notes"] = _append_note(
                record.notes,
                f"同强度重编码增广（{spec.name}，所有标签使用同一参数）。",
            )
            augmented.append(DatasetRecord.model_validate(payload))
    return augmented


def plan_summary(jobs: list[AugmentJob]) -> dict[str, Any]:
    per_spec = Counter(job.spec.name for job in jobs)
    per_label = Counter(job.label for job in jobs)
    specs_by_label: dict[str, set[str]] = {}
    for job in jobs:
        specs_by_label.setdefault(job.label, set()).add(job.spec.name)
    matched = len(set(map(frozenset, specs_by_label.values()))) <= 1
    return {
        "job_count": len(jobs),
        "per_spec": dict(sorted(per_spec.items())),
        "per_label": dict(sorted(per_label.items())),
        "specs_by_label": {
            label: sorted(values) for label, values in sorted(specs_by_label.items())
        },
        "same_strength_across_labels": matched,
    }


def _ffmpeg_command(source: Path, output: Path, spec: ReencodeSpec) -> tuple[str, ...]:
    command = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
    ]
    filters = spec.filters()
    if filters:
        command.extend(["-vf", ",".join(filters)])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            spec.preset,
            "-crf",
            str(spec.crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            spec.audio_bitrate,
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    return tuple(command)


def _append_note(existing: str | None, addition: str) -> str:
    return f"{existing} {addition}".strip() if existing else addition


def _tail(value: str, limit: int = 2000) -> str:
    return value[-limit:].strip() if value else "FFmpeg 执行失败"
