from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    auc: float
    eer: float
    accuracy: float
    tpr_at_fpr_1pct: float
    tpr_at_fpr_0_1pct: float
    ece: float
    threshold: float
    operational_fpr: float
    count: int
    positive_count: int
    negative_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "auc": self.auc,
            "eer": self.eer,
            "accuracy": self.accuracy,
            "tpr_at_fpr_1pct": self.tpr_at_fpr_1pct,
            "tpr_at_fpr_0.1pct": self.tpr_at_fpr_0_1pct,
            "ece": self.ece,
            "threshold": self.threshold,
            "operational_fpr": self.operational_fpr,
            "count": self.count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
        }


def _validate_inputs(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("scores 和 labels 必须是一维且等长")
    if scores.size == 0:
        raise ValueError("scores 和 labels 不能为空")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores 包含非有限值")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels 只能包含 0 和 1")
    if len(np.unique(labels)) < 2:
        raise ValueError("计算二分类指标至少需要正负样本各一条")
    return scores, labels


def roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores, labels = _validate_inputs(scores, labels)
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    positives = sorted_labels.sum()
    negatives = len(sorted_labels) - positives

    tps = np.cumsum(sorted_labels)
    fps = np.cumsum(1 - sorted_labels)
    tpr = np.concatenate([[0.0], tps / positives])
    fpr = np.concatenate([[0.0], fps / negatives])
    return fpr, tpr


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr = roc_points(scores, labels)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    return float(np.trapz(tpr, fpr))


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    fpr, tpr = roc_points(scores, labels)
    eligible = np.where(fpr <= target_fpr + 1e-12)[0]
    if eligible.size == 0:
        return 0.0
    return float(np.max(tpr[eligible]))


def eer_score(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr = roc_points(scores, labels)
    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2)


def threshold_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    scores, labels = _validate_inputs(scores, labels)
    negative_scores = np.sort(scores[labels == 0])
    allowed_false_positives = int(np.floor(target_fpr * negative_scores.size))
    if allowed_false_positives <= 0:
        return float(np.nextafter(negative_scores.max(), np.inf))
    index = max(0, negative_scores.size - allowed_false_positives - 1)
    return float(negative_scores[index])


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    probabilities, labels = _validate_inputs(probabilities, labels)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        if upper == 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        confidence = float(np.mean(probabilities[mask]))
        accuracy = float(np.mean(labels[mask]))
        ece += float(np.mean(mask)) * abs(confidence - accuracy)
    return float(ece)


def reliability_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 15,
) -> dict[str, list[float]]:
    probabilities, labels = _validate_inputs(probabilities, labels)
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers: list[float] = []
    observed: list[float] = []
    counts: list[float] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        if upper == 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        centers.append(float((lower + upper) / 2))
        observed.append(float(np.mean(labels[mask])))
        counts.append(float(np.mean(mask)))
    return {"confidence": centers, "observed_positive_rate": observed, "sample_fraction": counts}


def score_quantiles(
    scores: np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
) -> dict[str, float]:
    """输出分数分位数，供按分位数确定灰区阈值使用。"""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores 必须是一维非空数组")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores 包含非有限值")
    return {f"{quantile:g}": float(np.quantile(values, quantile)) for quantile in quantiles}


def binary_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float | None = None,
) -> BinaryMetrics:
    scores, labels = _validate_inputs(scores, labels)
    if threshold is None:
        threshold = threshold_at_fpr(scores, labels, 0.01)
    predictions = (scores >= threshold).astype(int)
    accuracy = float(np.mean(predictions == labels))
    false_positives = int(np.sum((predictions == 1) & (labels == 0)))
    operational_fpr = false_positives / int(np.sum(labels == 0))
    return BinaryMetrics(
        auc=auc_score(scores, labels),
        eer=eer_score(scores, labels),
        accuracy=accuracy,
        tpr_at_fpr_1pct=tpr_at_fpr(scores, labels, 0.01),
        tpr_at_fpr_0_1pct=tpr_at_fpr(scores, labels, 0.001),
        ece=expected_calibration_error(scores, labels),
        threshold=float(threshold),
        operational_fpr=float(operational_fpr),
        count=int(scores.size),
        positive_count=int(labels.sum()),
        negative_count=int((1 - labels).sum()),
    )


def grouped_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
) -> dict[str, dict[str, float | int]]:
    if len(scores) != len(groups) or len(labels) != len(groups):
        raise ValueError("scores、labels、groups 必须等长")
    result: dict[str, dict[str, float | int]] = {}
    for group in sorted(set(groups)):
        mask = np.array([value == group for value in groups], dtype=bool)
        group_scores = np.asarray(scores)[mask]
        group_labels = np.asarray(labels)[mask]
        if np.unique(group_labels).size < 2:
            result[group] = {
                "count": int(mask.sum()),
                "positive_count": int(group_labels.sum()),
                "negative_count": int((1 - group_labels).sum()),
                "insufficient_classes": True,
            }
            continue
        result[group] = binary_metrics(group_scores, group_labels).as_dict()
    return result
