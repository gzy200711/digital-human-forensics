from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from apps.api.detector_plugins.base import PluginContext
from apps.api.detector_plugins.face_quality import FaceSample
from apps.api.detector_plugins.visual_onnx import OnnxVisualArtifactPlugin
from apps.api.services.media import MediaProbe, MediaStream


def test_configured_missing_visual_model_fails_closed(tmp_path: Path) -> None:
    frame = np.full((96, 96, 3), 128, dtype=np.uint8)
    sample = FaceSample(
        timestamp=0.0,
        frame_index=0,
        frame=frame,
        bbox=(12, 12, 72, 72),
        crop=frame[12:84, 12:84],
        blur_score=0.8,
        brightness_score=0.8,
        pose_score=0.8,
        occlusion_score=0.8,
        sharpness_raw=100.0,
        skin_ratio=0.3,
    )
    stream = MediaStream(
        index=0,
        codec_type="video",
        codec_name="h264",
        width=96,
        height=96,
        frame_rate=25.0,
        sample_rate=None,
        channels=None,
        duration_seconds=1.0,
        tags={},
    )
    probe = MediaProbe(
        path=tmp_path / "sample.mp4",
        format_name="mp4",
        format_long_name="MP4",
        duration_seconds=1.0,
        size_bytes=1024,
        bit_rate=None,
        tags={},
        streams=[stream],
    )
    context = PluginContext(
        job_id="visual",
        original_filename="sample.mp4",
        source_object_key="sample.mp4",
        media_path=probe.path,
        probe=probe,
        settings=SimpleNamespace(),
        face_analysis=SimpleNamespace(samples=[sample]),
    )
    plugin = OnnxVisualArtifactPlugin(
        tmp_path / "missing.onnx",
        input_size=224,
        source="test-model",
        license_name="Apache-2.0",
    )

    result = plugin.analyze(context)

    assert not result.signal.available
    assert result.signal.score is None
    assert "未回退" in result.signal.explanation
