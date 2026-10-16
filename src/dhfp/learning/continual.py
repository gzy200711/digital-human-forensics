from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dhfp.evaluation.schema import DatasetRecord


@dataclass(frozen=True)
class ReplaySelection:
    seed: int
    max_samples: int
    sample_ids: list[str]
    label_counts: dict[str, int]
    generator_counts: dict[str, int]
    excluded_generators: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContinualGateThresholds:
    min_tpr_at_fpr_1pct: float = 0.65
    max_ece: float = 0.06
    max_tpr_drop: float = 0.03
    max_ece_increase: float = 0.01
    max_auc_drop: float = 0.02
    max_localization_ap_drop: float = 0.05


@dataclass(frozen=True)
class RegressionFinding:
    control: str
    passed: bool
    details: str
    values: dict[str, float | None] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegressionGateResult:
    detector_id: str
    passed: bool
    findings: list[RegressionFinding]

    def as_dict(self) -> dict[str, Any]:
        return {
            "detector_id": self.detector_id,
            "passed": self.passed,
            "findings": [finding.as_dict() for finding in self.findings],
        }


def select_replay_samples(
    records: list[DatasetRecord],
    *,
    max_samples: int = 200,
    seed: int = 20260918,
    excluded_generators: set[str] | None = None,
    per_generator_limit: int | None = None,
) -> ReplaySelection:
    if max_samples <= 0:
        raise ValueError("max_samples 必须大于 0")
    if per_generator_limit is not None and per_generator_limit <= 0:
        raise ValueError("per_generator_limit 必须大于 0")
    excluded = excluded_generators or set()
    buckets: dict[tuple[str, str], list[DatasetRecord]] = defaultdict(list)
    for record in records:
        if not record.usable or record.generator in excluded:
            continue
        buckets[(record.label.value, record.generator)].append(record)
    if not buckets:
        raise ValueError("没有可用于经验回放的记录")

    rng = random.Random(seed)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item.sample_id)
        rng.shuffle(bucket)
    bucket_generators = {
        key: key[1]
        for key in sorted(buckets, key=lambda item: (item[0], item[1]))
    }
    generator_usage: Counter[str] = Counter()
    selected: list[DatasetRecord] = []
    keys = sorted(buckets, key=lambda item: (item[0], item[1]))
    while len(selected) < max_samples:
        progressed = False
        for key in keys:
            bucket = buckets[key]
            generator = bucket_generators[key]
            if not bucket:
                continue
            if (
                per_generator_limit is not None
                and generator_usage[generator] >= per_generator_limit
            ):
                continue
            selected.append(bucket.pop())
            generator_usage[generator] += 1
            progressed = True
            if len(selected) >= max_samples:
                break
        if not progressed:
            break

    labels = Counter(record.label.value for record in selected)
    generators = Counter(record.generator for record in selected)
    return ReplaySelection(
        seed=seed,
        max_samples=max_samples,
        sample_ids=[record.sample_id for record in selected],
        label_counts=dict(sorted(labels.items())),
        generator_counts=dict(sorted(generators.items())),
        excluded_generators=sorted(excluded),
    )


def evaluate_regression_gate(
    previous_report: dict[str, Any],
    candidate_report: dict[str, Any],
    *,
    detector_id: str,
    thresholds: ContinualGateThresholds | None = None,
) -> RegressionGateResult:
    limits = thresholds or ContinualGateThresholds()
    previous = _detector_payload(previous_report, detector_id)
    candidate = _detector_payload(candidate_report, detector_id)
    findings: list[RegressionFinding] = []
    if not candidate or not candidate.get("available"):
        return RegressionGateResult(
            detector_id=detector_id,
            passed=False,
            findings=[
                RegressionFinding(
                    control="candidate_available",
                    passed=False,
                    details="候选模型在评测报告中不可用。",
                )
            ],
        )

    candidate_metrics = candidate.get("metrics") or {}
    tpr = _metric(candidate_metrics, "tpr_at_fpr_1pct")
    ece = _metric(candidate_metrics, "ece")
    findings.append(
        _threshold_finding(
            "absolute_tpr_at_fpr_1pct",
            tpr,
            minimum=limits.min_tpr_at_fpr_1pct,
        )
    )
    findings.append(
        _threshold_finding(
            "absolute_ece",
            ece,
            maximum=limits.max_ece,
        )
    )

    if previous and previous.get("available"):
        previous_metrics = previous.get("metrics") or {}
        findings.extend(
            [
                _regression_finding(
                    "tpr_regression",
                    previous_metrics.get("tpr_at_fpr_1pct"),
                    candidate_metrics.get("tpr_at_fpr_1pct"),
                    max_drop=limits.max_tpr_drop,
                ),
                _regression_finding(
                    "ece_regression",
                    previous_metrics.get("ece"),
                    candidate_metrics.get("ece"),
                    max_increase=limits.max_ece_increase,
                ),
                _regression_finding(
                    "auc_regression",
                    previous_metrics.get("auc"),
                    candidate_metrics.get("auc"),
                    max_drop=limits.max_auc_drop,
                ),
            ]
        )
        previous_ap = _localization_ap(previous)
        candidate_ap = _localization_ap(candidate)
        if previous_ap is not None or candidate_ap is not None:
            findings.append(
                _regression_finding(
                    "localization_ap_regression",
                    previous_ap,
                    candidate_ap,
                    max_drop=limits.max_localization_ap_drop,
                )
            )
    else:
        findings.append(
            RegressionFinding(
                control="baseline_available",
                passed=True,
                details="没有可用的旧模型基线，仅执行绝对门禁。",
            )
        )

    return RegressionGateResult(
        detector_id=detector_id,
        passed=all(finding.passed for finding in findings),
        findings=findings,
    )


def write_continual_update_manifest(
    path: str | Path,
    *,
    candidate_detector_id: str,
    replay: ReplaySelection,
    gate: RegressionGateResult,
    notes: str = "",
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "candidate_detector_id": candidate_detector_id,
        "replay": replay.as_dict(),
        "gate": gate.as_dict(),
        "release_eligible": gate.passed,
        "notes": notes,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _detector_payload(report: dict[str, Any], detector_id: str) -> dict[str, Any] | None:
    detectors = report.get("detectors")
    if not isinstance(detectors, dict):
        return None
    payload = detectors.get(detector_id)
    return payload if isinstance(payload, dict) else None


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _localization_ap(payload: dict[str, Any]) -> float | None:
    localization = payload.get("localization")
    if not isinstance(localization, dict):
        return None
    values = []
    for item in localization.values():
        if isinstance(item, dict):
            value = item.get("average_precision")
            if isinstance(value, int | float):
                values.append(float(value))
    return max(values) if values else None


def _threshold_finding(
    control: str,
    value: float | None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> RegressionFinding:
    if value is None:
        return RegressionFinding(
            control=control,
            passed=False,
            details="指标缺失，不能通过发布门禁。",
        )
    passed = True
    if minimum is not None:
        passed = passed and value >= minimum
    if maximum is not None:
        passed = passed and value <= maximum
    return RegressionFinding(
        control=control,
        passed=passed,
        details="满足阈值。" if passed else "未满足发布阈值。",
        values={"value": value, "minimum": minimum, "maximum": maximum},
    )


def _regression_finding(
    control: str,
    previous: float | None,
    candidate: float | None,
    *,
    max_drop: float | None = None,
    max_increase: float | None = None,
) -> RegressionFinding:
    if not isinstance(previous, int | float) or not isinstance(candidate, int | float):
        return RegressionFinding(
            control=control,
            passed=False,
            details="旧模型或候选模型指标缺失，无法判断回归。",
            values={"previous": None, "candidate": None},
        )
    previous_value = float(previous)
    candidate_value = float(candidate)
    delta = candidate_value - previous_value
    passed = True
    if max_drop is not None:
        passed = passed and delta >= -max_drop
    if max_increase is not None:
        passed = passed and delta <= max_increase
    return RegressionFinding(
        control=control,
        passed=passed,
        details="未检测到超限回归。" if passed else "检测到超限回归。",
        values={
            "previous": previous_value,
            "candidate": candidate_value,
            "delta": delta,
        },
    )
