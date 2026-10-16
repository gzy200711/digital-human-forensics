from __future__ import annotations

import numpy as np
import pytest

from dhfp.evaluation.metrics import (
    auc_score,
    binary_metrics,
    expected_calibration_error,
    tpr_at_fpr,
)


def test_perfect_ranking_metrics() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    scores = np.asarray([0.05, 0.11, 0.23, 0.71, 0.86, 0.97])
    assert auc_score(scores, labels) == pytest.approx(1.0)
    assert tpr_at_fpr(scores, labels, 0.01) == pytest.approx(1.0)
    metrics = binary_metrics(scores, labels)
    assert metrics.eer == pytest.approx(0.0)
    assert metrics.accuracy == pytest.approx(1.0)


def test_expected_calibration_error() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.01, 0.01, 0.99, 0.99])
    assert expected_calibration_error(probabilities, labels) < 0.02


def test_metrics_reject_invalid_input() -> None:
    with pytest.raises(ValueError):
        binary_metrics(np.asarray([0.2, 0.4]), np.asarray([0, 0]))


def test_operational_fpr_is_measured_at_reported_threshold() -> None:
    labels = np.asarray([0] * 20 + [1] * 20)
    scores = np.asarray([0.9] * 2 + [0.1] * 18 + [0.8] * 20)

    metrics = binary_metrics(scores, labels)

    predictions = scores >= metrics.threshold
    expected = float(
        np.mean(predictions[labels == 0])
    )
    assert metrics.operational_fpr == pytest.approx(expected)
