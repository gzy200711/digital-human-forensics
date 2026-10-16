from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from apps.api.detector_plugins.base import PluginContext
from apps.api.detector_plugins.digital_human import (
    DigitalHumanPlugin,
    _feature_components,
    _hand_like_blob_present,
    _head_micro_motion_features,
)
from apps.api.detector_plugins.face_quality import FaceSample
from apps.api.services.media import MediaProbe, MediaStream


def _sample(timestamp: float, bbox: tuple[int, int, int, int], frame: np.ndarray) -> FaceSample:
    return FaceSample(
        timestamp=timestamp,
        frame_index=int(timestamp * 10),
        frame=frame,
        bbox=bbox,
        crop=frame,
        blur_score=0.8,
        brightness_score=0.8,
        pose_score=0.8,
        occlusion_score=0.8,
        sharpness_raw=120.0,
        skin_ratio=0.3,
    )


def _context(tmp_path: Path, samples: list[FaceSample] | None = None) -> PluginContext:
    probe = MediaProbe(
        path=tmp_path / "media.mp4",
        format_name="mp4",
        format_long_name="MP4",
        duration_seconds=12.0,
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
                duration_seconds=12.0,
                tags={},
            )
        ],
    )
    return PluginContext(
        job_id="digital-human",
        original_filename="media.mp4",
        source_object_key="media.mp4",
        media_path=probe.path,
        probe=probe,
        settings=None,
        face_analysis=SimpleNamespace(samples=samples) if samples is not None else None,
    )


def test_static_head_reports_frozen_motion() -> None:
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    samples = [_sample(index / 4, (120, 40, 120, 120), frame) for index in range(40)]

    features = _head_micro_motion_features(samples, 4.0)

    assert features["frozen_ratio"] == 1.0
    assert features["high_band_ratio"] == 0.0


def test_head_micro_motion_spectrum_detects_oscillation() -> None:
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    samples = []
    for index in range(40):
        timestamp = index / 4
        offset = int(round(3 * np.sin(2 * np.pi * 0.5 * timestamp)))
        samples.append(_sample(timestamp, (120 + offset, 40, 120, 120), frame))

    features = _head_micro_motion_features(samples, 4.0)

    assert features["frozen_ratio"] < 0.2
    assert features["high_band_ratio"] > 0.5


def test_suspicious_features_build_strong_components() -> None:
    components = _feature_components(
        blink={
            "eye_visibility_ratio": 0.9,
            "blink_rate_per_minute": 0.0,
            "eye_motion_median": 0.001,
            "observed_seconds": 12.0,
            "eye_detection_available": True,
        },
        motion={
            "high_band_ratio": 0.02,
            "frozen_ratio": 0.95,
            "sample_count": 20,
            "observed_seconds": 12.0,
        },
        hands={"presence_ratio": 0.0, "frames_checked": 12},
    )

    assert components["eye"] > 0.6
    assert components["head"] > 0.6
    assert components["hand"] == 1.0


def test_plugin_requires_enough_observation_time(tmp_path: Path) -> None:
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    samples = [_sample(index * 0.1, (120, 40, 120, 120), frame) for index in range(12)]

    result = DigitalHumanPlugin().analyze(_context(tmp_path, samples))

    assert not result.available
    assert "时长不足" in result.explanation


def test_hand_like_blob_detection() -> None:
    frame = np.full((240, 320, 3), 90, dtype=np.uint8)
    bbox = (120, 20, 80, 80)
    assert not _hand_like_blob_present(frame, bbox)

    cv2.rectangle(frame, (10, 160), (70, 220), (140, 170, 220), thickness=-1)
    assert _hand_like_blob_present(frame, bbox)


def test_plugin_is_unavailable_without_face_samples(tmp_path: Path) -> None:
    result = DigitalHumanPlugin().analyze(_context(tmp_path))

    assert not result.available
    assert result.signal_id == "digital_human"


def test_plugin_runs_on_long_enough_clip(tmp_path: Path) -> None:
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    samples = [_sample(index / 2, (120, 40, 120, 120), frame) for index in range(16)]

    result = DigitalHumanPlugin(sample_fps=2.0).analyze(_context(tmp_path, samples))

    assert result.available
    assert result.score is not None
    assert "components" in result.raw_value


def test_plugin_uses_effective_sampling_rate_from_timestamps(tmp_path: Path) -> None:
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    samples = [_sample(index * 2.0, (120, 40, 120, 120), frame) for index in range(16)]

    result = DigitalHumanPlugin(sample_fps=2.0).analyze(_context(tmp_path, samples))

    assert result.available
    assert result.raw_value["sample_fps"] == pytest.approx(0.5)
