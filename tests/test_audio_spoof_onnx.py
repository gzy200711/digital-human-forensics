from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apps.api.detector_plugins.audio_onnx import (
    OnnxAudioSpoofPlugin,
    _aggregate,
    logmel_segments,
    segment_waveform,
)
from apps.api.detector_plugins.base import PluginContext
from apps.api.services.media import MediaProbe, MediaStream


def test_segment_waveform_pads_short_audio() -> None:
    audio = np.ones(16000, dtype=np.float32)

    batch, starts = segment_waveform(
        audio,
        sample_rate=16000,
        segment_seconds=4.04,
        max_segments=8,
    )

    assert batch.shape == (1, 64640)
    assert starts == [0.0]
    assert batch[0, -1] == 0.0


def test_segment_waveform_splits_long_audio() -> None:
    audio = np.arange(64640 * 3, dtype=np.float32)

    batch, starts = segment_waveform(
        audio,
        sample_rate=16000,
        segment_seconds=4.04,
        max_segments=8,
    )

    assert batch.shape == (3, 64640)
    assert starts[1] == pytest.approx(4.04)
    assert starts[2] == pytest.approx(8.08)


def test_logmel_segments_shape_and_normalization() -> None:
    audio = np.random.default_rng(7).normal(0, 0.1, 16000 * 5).astype(np.float32)

    features, starts = logmel_segments(
        audio,
        sample_rate=16000,
        segment_seconds=1.0,
        max_segments=4,
        n_mels=40,
    )

    assert features.ndim == 3
    assert features.shape[0] == len(starts) == 4
    assert features.shape[2] == 40
    # 片段级均值-方差归一化（CMVN），整体均值接近 0、标准差接近 1。
    assert float(np.mean(features)) == pytest.approx(0.0, abs=1e-3)
    assert float(np.std(features)) == pytest.approx(1.0, abs=1e-3)


def test_missing_model_reports_unavailable_without_fallback(tmp_path: Path) -> None:
    plugin = OnnxAudioSpoofPlugin(tmp_path / "missing.onnx")
    audio_stream = MediaStream(
        index=1,
        codec_type="audio",
        codec_name="aac",
        width=None,
        height=None,
        frame_rate=None,
        sample_rate=16000,
        channels=1,
        duration_seconds=10.0,
        tags={},
    )
    probe = MediaProbe(
        path=tmp_path / "media.mp4",
        format_name="mp4",
        format_long_name="MP4",
        duration_seconds=10.0,
        size_bytes=1024,
        bit_rate=None,
        tags={},
        streams=[audio_stream],
    )
    context = PluginContext(
        job_id="audio",
        original_filename="media.mp4",
        source_object_key="media.mp4",
        media_path=probe.path,
        probe=probe,
        settings=SimpleNamespace(max_media_duration_seconds=60),
    )

    result = plugin.analyze(context)

    assert not result.available
    assert result.score is None
    assert "不存在" in result.explanation


def test_aggregate_modes() -> None:
    probabilities = np.asarray([0.1, 0.9, 0.6, 0.4])

    assert _aggregate(probabilities, "max") == pytest.approx(0.9)
    assert _aggregate(probabilities, "mean") == pytest.approx(0.5)
    assert _aggregate(probabilities, "trimmed") == pytest.approx(0.75)


class _StubSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.input_name = "waveform"

    def get_inputs(self):
        return [SimpleNamespace(name=self.input_name)]

    def run(self, _outputs, _feed):
        return [self.output]


def test_predict_handles_two_class_logits_and_single_logit(tmp_path: Path) -> None:
    plugin = OnnxAudioSpoofPlugin(tmp_path / "model.onnx")
    batch = np.zeros((3, 16), dtype=np.float32)

    two_class = np.asarray([[0.0, 2.0], [2.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    probabilities = plugin._predict(_StubSession(two_class), batch)
    assert probabilities.shape == (3,)
    assert probabilities[0] > 0.85
    assert probabilities[1] < 0.15
    assert probabilities[2] == pytest.approx(0.5)

    single = np.asarray([[2.0], [-2.0]], dtype=np.float32)
    probabilities = plugin._predict(_StubSession(single), batch[:2])
    assert probabilities[0] > 0.85
    assert probabilities[1] < 0.15

    single_flat = np.asarray([-2.0, 2.0], dtype=np.float32)
    probabilities = plugin._predict(_StubSession(single_flat), batch[:1])
    assert probabilities.shape == (1,)
    assert probabilities[0] > 0.85
