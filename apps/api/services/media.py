from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


class MediaProbeError(RuntimeError):
    pass


@dataclass(frozen=True)
class MediaStream:
    index: int
    codec_type: str
    codec_name: str | None
    width: int | None
    height: int | None
    frame_rate: float | None
    sample_rate: int | None
    channels: int | None
    duration_seconds: float | None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MediaProbe:
    path: Path
    format_name: str | None
    format_long_name: str | None
    duration_seconds: float | None
    size_bytes: int | None
    bit_rate: int | None
    tags: dict[str, str]
    streams: list[MediaStream]

    @property
    def video_stream(self) -> MediaStream | None:
        return next((stream for stream in self.streams if stream.codec_type == "video"), None)

    @property
    def audio_stream(self) -> MediaStream | None:
        return next((stream for stream in self.streams if stream.codec_type == "audio"), None)

    @property
    def has_audio(self) -> bool:
        return self.audio_stream is not None

    @property
    def width(self) -> int | None:
        stream = self.video_stream
        return stream.width if stream else None

    @property
    def height(self) -> int | None:
        stream = self.video_stream
        return stream.height if stream else None

    @property
    def frame_rate(self) -> float | None:
        stream = self.video_stream
        return stream.frame_rate if stream else None

    @property
    def video_codec(self) -> str | None:
        stream = self.video_stream
        return stream.codec_name if stream else None

    @property
    def audio_codec(self) -> str | None:
        stream = self.audio_stream
        return stream.codec_name if stream else None


@dataclass
class DecodedFrame:
    index: int
    timestamp: float
    image: np.ndarray


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def _parse_fraction(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" not in value:
        try:
            return float(value)
        except ValueError:
            return None
    numerator, denominator = value.split("/", maxsplit=1)
    try:
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else None
    except ValueError:
        return None


def probe_media(path: str | Path) -> MediaProbe:
    media_path = Path(path)
    if not media_path.exists():
        raise MediaProbeError(f"媒体文件不存在: {media_path}")
    if not ffprobe_available():
        raise MediaProbeError("ffprobe 不可用，请安装 FFmpeg")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(media_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaProbeError(f"ffprobe 执行失败: {exc}") from exc
    if completed.returncode != 0:
        raise MediaProbeError(completed.stderr.strip() or "ffprobe 读取失败")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError("ffprobe 返回了无效 JSON") from exc

    streams: list[MediaStream] = []
    for stream in payload.get("streams", []):
        tags = {str(key): str(value) for key, value in (stream.get("tags") or {}).items()}
        streams.append(
            MediaStream(
                index=_optional_int(stream.get("index")) or 0,
                codec_type=str(stream.get("codec_type", "unknown")),
                codec_name=stream.get("codec_name"),
                width=_optional_int(stream.get("width")),
                height=_optional_int(stream.get("height")),
                frame_rate=_parse_fraction(
                    stream.get("avg_frame_rate") or stream.get("r_frame_rate")
                ),
                sample_rate=_optional_int(stream.get("sample_rate")),
                channels=_optional_int(stream.get("channels")),
                duration_seconds=_optional_float(stream.get("duration")),
                tags=tags,
            )
        )

    format_info = payload.get("format") or {}
    format_tags = {str(key): str(value) for key, value in (format_info.get("tags") or {}).items()}
    format_size = _optional_int(format_info.get("size"))
    return MediaProbe(
        path=media_path,
        format_name=format_info.get("format_name"),
        format_long_name=format_info.get("format_long_name"),
        duration_seconds=_optional_float(format_info.get("duration")),
        size_bytes=format_size if format_size is not None else media_path.stat().st_size,
        bit_rate=_optional_int(format_info.get("bit_rate")),
        tags=format_tags,
        streams=streams,
    )


def _optional_float(value: object) -> float | None:
    if value is None or value == "N/A" or value == "":
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_int(value: object) -> int | None:
    if value is None or value == "N/A" or value == "":
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sample_video_frames(
    path: str | Path,
    *,
    sample_fps: float,
    max_frames: int,
    duration_seconds: float | None = None,
) -> list[DecodedFrame]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise MediaProbeError("OpenCV 无法打开视频")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = 25.0
    effective_duration = duration_seconds
    if not effective_duration and frame_count > 0:
        effective_duration = frame_count / source_fps
    if not effective_duration:
        effective_duration = 0.0

    target_count = max(1, min(max_frames, int(math.ceil(effective_duration * sample_fps))))
    if frame_count > 0:
        indices = np.linspace(0, max(0, frame_count - 1), target_count, dtype=int).tolist()
    else:
        step = max(1, int(round(source_fps / sample_fps)))
        indices = list(range(0, target_count * step, step))

    frames: list[DecodedFrame] = []
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, image = capture.read()
            if not ok or image is None:
                continue
            timestamp = frame_index / source_fps
            frames.append(DecodedFrame(index=int(frame_index), timestamp=timestamp, image=image))
    finally:
        capture.release()
    return frames


def decode_audio_mono(
    path: str | Path,
    *,
    sample_rate: int = 16000,
    max_duration_seconds: int = 900,
) -> np.ndarray:
    if not ffmpeg_available():
        raise MediaProbeError("ffmpeg 不可用，无法解码音轨")
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-t",
        str(max_duration_seconds),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=max(60, max_duration_seconds * 2),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaProbeError(f"音频解码失败: {exc}") from exc
    if completed.returncode != 0:
        raise MediaProbeError(
            completed.stderr.decode("utf-8", errors="replace").strip() or "音频解码失败"
        )
    return np.frombuffer(completed.stdout, dtype=np.float32).copy()


def safe_ffmpeg_stderr(data: bytes) -> str:
    return data.decode("utf-8", errors="replace").strip()
