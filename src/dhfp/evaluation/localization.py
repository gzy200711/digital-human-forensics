from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class TimeSegment:
    start: float
    end: float
    score: float = 1.0

    def __post_init__(self) -> None:
        _validate_time(self.start, "start")
        _validate_time(self.end, "end")
        if self.start >= self.end:
            raise ValueError("start 必须小于 end")
        _validate_score(self.score)


@dataclass(frozen=True)
class SpatialBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    timestamp: float | None = None

    def __post_init__(self) -> None:
        _validate_time(self.x1, "x1")
        _validate_time(self.y1, "y1")
        _validate_time(self.x2, "x2")
        _validate_time(self.y2, "y2")
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("空间框必须满足 x1 < x2 且 y1 < y2")
        _validate_score(self.score)
        if self.timestamp is not None:
            _validate_time(self.timestamp, "timestamp")
            if self.timestamp < 0:
                raise ValueError("timestamp 不能为负数")


@dataclass(frozen=True)
class LocalizationMetrics:
    average_precision: float
    precision: float
    recall: float
    matched_count: int
    predicted_count: int
    target_count: int
    calibration_error: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "average_precision": self.average_precision,
            "precision": self.precision,
            "recall": self.recall,
            "matched_count": self.matched_count,
            "predicted_count": self.predicted_count,
            "target_count": self.target_count,
            "calibration_error": self.calibration_error,
        }


def temporal_iou(left: TimeSegment, right: TimeSegment) -> float:
    intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    union = (left.end - left.start) + (right.end - right.start) - intersection
    return float(intersection / union) if union > 0 else 0.0


def spatial_iou(left: SpatialBox, right: SpatialBox) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    union = left_area + right_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def evaluate_temporal_localization(
    predictions: Mapping[str, Sequence[TimeSegment]],
    targets: Mapping[str, Sequence[TimeSegment]],
    *,
    iou_threshold: float = 0.5,
    calibration_bins: int = 10,
) -> LocalizationMetrics:
    _validate_threshold(iou_threshold)
    ordered = _ordered_predictions(predictions)
    matched: dict[str, set[int]] = {sample_id: set() for sample_id in targets}
    hits: list[bool] = []
    scores: list[float] = []
    for sample_id, prediction in ordered:
        sample_targets = targets.get(sample_id, ())
        matched_index, best_iou = _best_temporal_match(
            prediction,
            sample_targets,
            matched.setdefault(sample_id, set()),
        )
        hit = best_iou >= iou_threshold
        if hit:
            matched[sample_id].add(matched_index)
        hits.append(hit)
        scores.append(prediction.score)
    return _build_metrics(
        hits=hits,
        scores=scores,
        target_count=sum(len(items) for items in targets.values()),
        calibration_bins=calibration_bins,
    )


def evaluate_spatial_localization(
    predictions: Mapping[str, Sequence[SpatialBox]],
    targets: Mapping[str, Sequence[SpatialBox]],
    *,
    iou_threshold: float = 0.5,
    calibration_bins: int = 10,
    time_tolerance: float = 0.1,
) -> LocalizationMetrics:
    _validate_threshold(iou_threshold)
    if time_tolerance < 0:
        raise ValueError("time_tolerance 不能为负数")
    ordered = _ordered_predictions(predictions)
    matched: dict[str, set[int]] = {sample_id: set() for sample_id in targets}
    hits: list[bool] = []
    scores: list[float] = []
    for sample_id, prediction in ordered:
        sample_targets = targets.get(sample_id, ())
        matched_index, best_iou = _best_spatial_match(
            prediction,
            sample_targets,
            matched.setdefault(sample_id, set()),
            time_tolerance=time_tolerance,
        )
        hit = best_iou >= iou_threshold
        if hit:
            matched[sample_id].add(matched_index)
        hits.append(hit)
        scores.append(prediction.score)
    return _build_metrics(
        hits=hits,
        scores=scores,
        target_count=sum(len(items) for items in targets.values()),
        calibration_bins=calibration_bins,
    )


def _ordered_predictions(
    predictions: Mapping[str, Sequence[TimeSegment | SpatialBox]],
) -> list[tuple[str, TimeSegment | SpatialBox]]:
    ordered: list[tuple[str, TimeSegment | SpatialBox]] = []
    for sample_id, items in predictions.items():
        ordered.extend((sample_id, item) for item in items)
    return sorted(
        ordered,
        key=lambda item: (-item[1].score, item[0]),
    )


def _best_temporal_match(
    prediction: TimeSegment,
    targets: Sequence[TimeSegment],
    matched: set[int],
) -> tuple[int, float]:
    best_index = -1
    best_iou = 0.0
    for index, target in enumerate(targets):
        if index in matched:
            continue
        value = temporal_iou(prediction, target)
        if value > best_iou:
            best_index = index
            best_iou = value
    return best_index, best_iou


def _best_spatial_match(
    prediction: SpatialBox,
    targets: Sequence[SpatialBox],
    matched: set[int],
    *,
    time_tolerance: float,
) -> tuple[int, float]:
    best_index = -1
    best_iou = 0.0
    for index, target in enumerate(targets):
        if index in matched:
            continue
        if not _times_compatible(prediction.timestamp, target.timestamp, time_tolerance):
            continue
        value = spatial_iou(prediction, target)
        if value > best_iou:
            best_index = index
            best_iou = value
    return best_index, best_iou


def _times_compatible(
    left: float | None,
    right: float | None,
    tolerance: float,
) -> bool:
    if left is None or right is None:
        return True
    return abs(left - right) <= tolerance


def _build_metrics(
    *,
    hits: list[bool],
    scores: list[float],
    target_count: int,
    calibration_bins: int,
) -> LocalizationMetrics:
    predicted_count = len(hits)
    matched_count = sum(hits)
    precision = matched_count / predicted_count if predicted_count else 0.0
    recall = matched_count / target_count if target_count else 0.0
    return LocalizationMetrics(
        average_precision=_average_precision(hits, target_count),
        precision=float(precision),
        recall=float(recall),
        matched_count=matched_count,
        predicted_count=predicted_count,
        target_count=target_count,
        calibration_error=_calibration_error(scores, hits, bins=calibration_bins),
    )


def _average_precision(hits: list[bool], target_count: int) -> float:
    if not hits or target_count <= 0:
        return 0.0
    hit_array = np.asarray(hits, dtype=float)
    true_positives = np.cumsum(hit_array)
    precision = true_positives / np.arange(1, len(hits) + 1)
    recall = true_positives / target_count
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    previous_recall = 0.0
    average_precision = 0.0
    for current_precision, current_recall in zip(
        precision_envelope,
        recall,
        strict=True,
    ):
        average_precision += (current_recall - previous_recall) * current_precision
        previous_recall = current_recall
    return float(average_precision)


def _calibration_error(
    scores: list[float],
    hits: list[bool],
    *,
    bins: int,
) -> float | None:
    if not scores:
        return None
    if bins < 1:
        raise ValueError("calibration_bins 必须至少为 1")
    score_array = np.asarray(scores, dtype=float)
    hit_array = np.asarray(hits, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        if upper == 1:
            mask = (score_array >= lower) & (score_array <= upper)
        else:
            mask = (score_array >= lower) & (score_array < upper)
        if not np.any(mask):
            continue
        confidence = float(np.mean(score_array[mask]))
        observed = float(np.mean(hit_array[mask]))
        error += float(np.mean(mask)) * abs(confidence - observed)
    return float(error)


def _validate_time(value: float, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} 必须是有限数值")


def _validate_score(value: float) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("score 必须在 [0, 1] 范围内")


def _validate_threshold(value: float) -> None:
    if not isfinite(value) or not 0 < value <= 1:
        raise ValueError("iou_threshold 必须在 (0, 1] 范围内")
