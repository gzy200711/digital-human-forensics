from __future__ import annotations

import pytest

from dhfp.evaluation.localization import (
    SpatialBox,
    TimeSegment,
    evaluate_spatial_localization,
    evaluate_temporal_localization,
    spatial_iou,
    temporal_iou,
)
from dhfp.evaluation.runner import EvaluationRunner
from dhfp.evaluation.schema import (
    AuthorizationStatus,
    DatasetLabel,
    DatasetRecord,
    SpatialAnnotation,
    TemporalAnnotation,
)
from dhfp.plugins.base import EvaluationPrediction


def test_temporal_iou_and_average_precision() -> None:
    target = TimeSegment(0, 10, 0)
    prediction = TimeSegment(1, 9, 0.9)
    false_positive = TimeSegment(20, 30, 0.8)

    assert temporal_iou(prediction, target) == pytest.approx(0.8)
    metrics = evaluate_temporal_localization(
        {"sample": [prediction, false_positive]},
        {"sample": [target]},
    )

    assert metrics.average_precision == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.matched_count == 1


def test_spatial_iou_and_average_precision() -> None:
    target = SpatialBox(0, 0, 10, 10, 0)
    prediction = SpatialBox(1, 1, 9, 9, 0.9)

    assert spatial_iou(prediction, target) == pytest.approx(0.64)
    metrics = evaluate_spatial_localization(
        {"sample": [prediction]},
        {"sample": [target]},
    )

    assert metrics.average_precision == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.calibration_error == pytest.approx(0.1)


def test_segment_calibration_error_uses_match_labels() -> None:
    target = TimeSegment(0, 10, 0)
    metrics = evaluate_temporal_localization(
        {
            "sample": [
                TimeSegment(0, 10, 1.0),
                TimeSegment(20, 30, 0.0),
            ]
        },
        {"sample": [target]},
    )

    assert metrics.calibration_error is not None
    assert metrics.calibration_error == pytest.approx(0.0)


def test_localization_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError):
        TimeSegment(2, 2, 0.5)
    with pytest.raises(ValueError):
        TimeSegment(0, 2, 1.1)
    with pytest.raises(ValueError):
        SpatialBox(2, 0, 1, 1, 0.5)


def test_spatial_matching_respects_timestamp() -> None:
    target = SpatialBox(0, 0, 10, 10, 0, timestamp=1.0)
    wrong_time = SpatialBox(0, 0, 10, 10, 1.0, timestamp=2.0)
    right_time = SpatialBox(0, 0, 10, 10, 0.5, timestamp=1.05)

    metrics = evaluate_spatial_localization(
        {"sample": [wrong_time]},
        {"sample": [target]},
        time_tolerance=0.1,
    )

    assert metrics.matched_count == 0

    metrics = evaluate_spatial_localization(
        {"sample": [right_time]},
        {"sample": [target]},
        time_tolerance=0.1,
    )
    assert metrics.matched_count == 1


def test_runner_localization_contract() -> None:
    record = DatasetRecord(
        sample_id="generated-1",
        media_path="/tmp/generated-1.mp4",
        label=DatasetLabel.GENERATED,
        generator="generator-a",
        authorization_status=AuthorizationStatus.VERIFIED,
        duration_seconds=4,
        width=640,
        height=360,
        frame_rate=25,
        has_audio=True,
        temporal_annotations=[TemporalAnnotation(start=1, end=2)],
        spatial_annotations=[
            SpatialAnnotation(timestamp=1.2, x1=10, y1=10, x2=110, y2=110)
        ],
    )
    prediction = EvaluationPrediction(
        sample_id=record.sample_id,
        score=0.8,
        available=True,
        latency_ms=10,
        temporal_segments=[TimeSegment(start=1, end=2, score=0.9)],
        spatial_boxes=[
            SpatialBox(
                10,
                10,
                110,
                110,
                score=0.9,
                timestamp=1.2,
            )
        ],
    )

    payload = EvaluationRunner._localization_metrics([record], [prediction])

    assert payload is not None
    assert payload["temporal"]["average_precision"] == pytest.approx(1.0)
    assert payload["spatial"]["recall"] == pytest.approx(1.0)
