from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from apps.api.detector_plugins.base import PluginContext
from apps.api.detector_plugins.face_quality import FaceSample
from apps.api.detector_plugins.temporal import TemporalConsistencyPlugin
from apps.api.services.media import MediaProbe, MediaStream


def _sample(timestamp: float, frame: np.ndarray) -> FaceSample:
    return FaceSample(
        timestamp=timestamp,
        frame_index=int(timestamp * 10),
        frame=frame,
        bbox=(40, 30, 80, 80),
        crop=frame,
        blur_score=0.8,
        brightness_score=0.8,
        pose_score=0.8,
        occlusion_score=0.8,
        sharpness_raw=120.0,
        skin_ratio=0.3,
    )


def _context(tmp_path: Path, samples: list[FaceSample]) -> PluginContext:
    probe = MediaProbe(
        path=tmp_path / "media.mp4",
        format_name="mp4",
        format_long_name="MP4",
        duration_seconds=2.0,
        size_bytes=1024,
        bit_rate=None,
        tags={},
        streams=[
            MediaStream(
                index=0,
                codec_type="video",
                codec_name="h264",
                width=320,
                height=240,
                frame_rate=25.0,
                sample_rate=None,
                channels=None,
                duration_seconds=2.0,
                tags={},
            )
        ],
    )
    return PluginContext(
        job_id="temporal",
        original_filename="media.mp4",
        source_object_key="media.mp4",
        media_path=probe.path,
        probe=probe,
        settings=None,
        face_analysis=SimpleNamespace(samples=samples),
    )


def test_temporal_ignores_background_changes(tmp_path: Path) -> None:
    face_patch = np.zeros((80, 80, 3), dtype=np.uint8)
    face_patch[20:60, 20:60] = 180
    rng = np.random.default_rng(11)
    samples = []
    for index in range(12):
        frame = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
        frame[30:110, 40:120] = face_patch
        samples.append(_sample(index / 2, frame))

    result = TemporalConsistencyPlugin().analyze(_context(tmp_path, samples))

    assert result.available
    assert result.score is not None
    # 人脸框内完全一致，背景剧烈变化不应抬高时序分数。
    assert result.score < 0.3
    assert result.raw_value["region"] == "face_bbox"
    assert result.raw_value["histogram_drift_median"] == 0.0
