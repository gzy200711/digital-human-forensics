from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dhfp.calibration import CalibrationProfile, fit_temperature
from dhfp.evaluation.audit import AuditFinding, audit_model, audit_split
from dhfp.evaluation.localization import (
    SpatialBox,
    TimeSegment,
    evaluate_spatial_localization,
    evaluate_temporal_localization,
)
from dhfp.evaluation.metrics import (
    binary_metrics,
    grouped_metrics,
    reliability_curve,
    score_quantiles,
)
from dhfp.evaluation.reports import plot_reliability_curve, render_markdown_report
from dhfp.evaluation.schema import DatasetRecord, index_summary
from dhfp.evaluation.split import split_map
from dhfp.plugins.base import (
    DetectorUnavailable,
    EvaluationDetector,
    EvaluationPrediction,
)


@dataclass(frozen=True)
class EvaluationSettings:
    seed: int = 20260918
    target_fpr: float = 0.01
    top_failures: int = 10


class EvaluationRunner:
    def __init__(
        self,
        *,
        records: list[DatasetRecord],
        splits: dict[str, Any],
        detectors: list[EvaluationDetector],
        output_dir: str | Path,
        settings: EvaluationSettings | None = None,
    ) -> None:
        self.records = records
        self.splits = splits
        self.detectors = detectors
        self.output_dir = Path(output_dir)
        self.settings = settings or EvaluationSettings()
        self.lookup = {record.sample_id: record for record in records}

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        split_lookup = split_map(self.splits)
        audit_findings = audit_split(self.records, self.splits["splits"])
        trivial_metrics = self._evaluate_trivial_baseline(split_lookup)
        detector_payloads: dict[str, Any] = {}
        for detector in self.detectors:
            detector_payloads[detector.detector_id] = self._evaluate_detector(
                detector,
                split_lookup,
                trivial_metrics,
            )
        coherence_findings = self._coherence_audits(detector_payloads, split_lookup)
        cross_encoding_findings = self._cross_encoding_audits(detector_payloads)
        payload = {
            "schema_version": "1.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "settings": asdict(self.settings),
            "dataset_summary": index_summary(self.records),
            "split_metadata": self.splits["metadata"],
            "split_counts": self.splits["metadata"].get(
                "counts",
                {name: len(values) for name, values in self.splits["splits"].items()},
            ),
            "audits": [
                item.as_dict()
                for item in [
                    *audit_findings,
                    *coherence_findings,
                    *cross_encoding_findings,
                ]
            ],
            "detectors": detector_payloads,
        }
        json_path = self.output_dir / "stage0_report.json"
        markdown_path = self.output_dir / "stage0_report.md"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(render_markdown_report(payload), encoding="utf-8")
        return payload

    def _evaluate_detector(
        self,
        detector: EvaluationDetector,
        split_lookup: dict[str, str],
        trivial_metrics,
    ) -> dict[str, Any]:
        train_records = self._records_for_split(split_lookup, "train")
        validation_records = self._records_for_split(split_lookup, "validation")
        test_records = self._records_for_split(split_lookup, "test")
        if hasattr(detector, "fit"):
            try:
                detector.fit(train_records)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                return self._unavailable_detector(detector, f"训练失败: {exc}")

        validation_predictions = self._predict_records(detector, validation_records)
        if not validation_predictions["available"]:
            return self._unavailable_detector(
                detector,
                validation_predictions["error"] or "验证集没有可用预测",
            )
        test_predictions = self._predict_records(detector, test_records)
        if not test_predictions["available"]:
            return self._unavailable_detector(
                detector,
                test_predictions["error"] or "测试集没有可用预测",
            )

        validation_scores = np.asarray(validation_predictions["scores"], dtype=float)
        validation_labels = np.asarray(validation_predictions["labels"], dtype=int)
        calibration = CalibrationProfile(version="identity", temperature=1.0, bias=0.0)
        if np.unique(validation_labels).size >= 2:
            calibration = fit_temperature(validation_scores, validation_labels)
        validation_calibrated_scores = np.asarray(
            [calibration.calibrate(score) for score in validation_scores],
            dtype=float,
        )

        raw_test_scores = np.asarray(test_predictions["scores"], dtype=float)
        test_scores = np.asarray(
            [calibration.calibrate(score) for score in raw_test_scores],
            dtype=float,
        )
        labels = np.asarray(test_predictions["labels"], dtype=int)
        metrics = binary_metrics(test_scores, labels)
        groups = {
            "generator": grouped_metrics(
                test_scores,
                labels,
                [record.generator for record in test_records],
            ),
            "forgery_type": grouped_metrics(
                test_scores,
                labels,
                [record.forgery_type.value for record in test_records],
            ),
            "resolution": grouped_metrics(
                test_scores,
                labels,
                [_resolution_bucket(record) for record in test_records],
            ),
            "compression": grouped_metrics(
                test_scores,
                labels,
                [record.compression or "unknown" for record in test_records],
            ),
        }
        curve = reliability_curve(test_scores, labels)
        curve_path = self.output_dir / "figures" / f"{detector.detector_id}_reliability.png"
        plot_reliability_curve(
            curve,
            detector_name=detector.display_name,
            output_path=curve_path,
        )
        failures = self._failure_cases(
            detector,
            test_records,
            test_scores,
            labels,
            threshold=metrics.threshold,
        )
        detector_audits = audit_model(
            metrics,
            trivial_metrics,
            max_fpr=self.settings.target_fpr,
            min_unseen_tpr=0.65,
            max_ece=0.06,
        )
        return {
            "detector_id": detector.detector_id,
            "display_name": detector.display_name,
            "version": detector.version,
            "license": detector.license,
            "description": detector.description,
            "available": True,
            "metrics": metrics.as_dict(),
            "groups": groups,
            "localization": self._localization_metrics(
                test_records,
                test_predictions["predictions"],
            ),
            "reliability_curve": curve,
            "reliability_plot": str(curve_path),
            "calibration": calibration.model_dump(),
            "validation_score_quantiles": score_quantiles(validation_calibrated_scores),
            "test_score_quantiles": score_quantiles(test_scores),
            "validation_count": int(validation_scores.size),
            "validation_negative_count": int(np.sum(validation_labels == 0)),
            "failure_cases": failures,
            "audits": [item.as_dict() for item in detector_audits],
            "latency_ms_mean": float(
                np.mean(
                    [
                        prediction.latency_ms
                        for prediction in test_predictions["predictions"]
                        if prediction.available
                    ]
                )
            ),
        }

    def _predict_records(
        self,
        detector: EvaluationDetector,
        records: list[DatasetRecord],
    ) -> dict[str, Any]:
        scores: list[float] = []
        labels: list[int] = []
        predictions: list[EvaluationPrediction] = []
        first_error = None
        for record in records:
            try:
                prediction = detector.predict(record)
            except DetectorUnavailable as exc:
                first_error = str(exc)
                break
            except Exception as exc:  # noqa: BLE001
                first_error = f"{type(exc).__name__}: {exc}"
                break
            if not prediction.available or prediction.score is None:
                continue
            scores.append(float(np.clip(prediction.score, 0, 1)))
            labels.append(int(record.is_positive))
            predictions.append(prediction)
        available = bool(scores) and len(scores) == len(records)
        return {
            "available": available,
            "scores": scores,
            "labels": labels,
            "predictions": predictions,
            "error": first_error,
        }

    def _evaluate_trivial_baseline(
        self,
        split_lookup: dict[str, str],
    ):
        from dhfp.plugins.builtin import TrivialMetadataBaseline

        detector = TrivialMetadataBaseline()
        try:
            detector.fit(self._records_for_split(split_lookup, "train"))
            test_records = self._records_for_split(split_lookup, "test")
            predictions = [detector.predict(record) for record in test_records]
            scores = np.asarray([prediction.score for prediction in predictions], dtype=float)
            labels = np.asarray([record.is_positive for record in test_records], dtype=int)
            return binary_metrics(scores, labels)
        except Exception:
            return binary_metrics(
                np.asarray([0.5, 0.5], dtype=float),
                np.asarray([0, 1], dtype=int),
            )

    def _failure_cases(
        self,
        detector: EvaluationDetector,
        records: list[DatasetRecord],
        scores: np.ndarray,
        labels: np.ndarray,
        *,
        threshold: float,
    ) -> list[dict[str, Any]]:
        false_positives = [
            index
            for index in range(len(records))
            if labels[index] == 0 and scores[index] >= threshold
        ]
        false_negatives = [
            index
            for index in range(len(records))
            if labels[index] == 1 and scores[index] < threshold
        ]
        false_positives.sort(key=lambda index: scores[index], reverse=True)
        false_negatives.sort(key=lambda index: abs(scores[index] - threshold), reverse=False)
        selected: list[tuple[str, int]] = [
            ("误报", index) for index in false_positives[: self.settings.top_failures]
        ]
        selected.extend(
            ("漏报", index)
            for index in false_negatives[: max(0, self.settings.top_failures - len(selected))]
        )
        failures: list[dict[str, Any]] = []
        failure_dir = self.output_dir / "failure_cases" / detector.detector_id
        failure_dir.mkdir(parents=True, exist_ok=True)
        for failure_type, index in selected:
            record = records[index]
            thumbnail_path = failure_dir / f"{record.sample_id}.jpg"
            thumbnail = _extract_thumbnail(Path(record.media_path), thumbnail_path)
            failures.append(
                {
                    "sample_id": record.sample_id,
                    "generator": record.generator,
                    "forgery_type": record.forgery_type.value,
                    "failure_type": failure_type,
                    "score": float(scores[index]),
                    "label": record.label.value,
                    "thumbnail": str(thumbnail) if thumbnail else None,
                }
            )
        return failures

    @staticmethod
    def _localization_metrics(
        records: list[DatasetRecord],
        predictions: list[EvaluationPrediction],
    ) -> dict[str, dict[str, float | int | None]] | None:
        temporal_targets = {
            record.sample_id: [
                TimeSegment(
                    start=annotation.start,
                    end=annotation.end,
                    score=1.0,
                )
                for annotation in record.temporal_annotations
            ]
            for record in records
        }
        spatial_targets = {
            record.sample_id: [
                SpatialBox(
                    x1=annotation.x1,
                    y1=annotation.y1,
                    x2=annotation.x2,
                    y2=annotation.y2,
                    score=1.0,
                    timestamp=annotation.timestamp,
                )
                for annotation in record.spatial_annotations
            ]
            for record in records
        }
        temporal_predictions = {
            prediction.sample_id: prediction.temporal_segments
            for prediction in predictions
            if prediction.temporal_segments
        }
        spatial_predictions = {
            prediction.sample_id: prediction.spatial_boxes
            for prediction in predictions
            if prediction.spatial_boxes
        }
        result: dict[str, dict[str, float | int | None]] = {}
        if sum(len(items) for items in temporal_targets.values()) > 0:
            result["temporal"] = evaluate_temporal_localization(
                temporal_predictions,
                temporal_targets,
            ).as_dict()
        if sum(len(items) for items in spatial_targets.values()) > 0:
            result["spatial"] = evaluate_spatial_localization(
                spatial_predictions,
                spatial_targets,
            ).as_dict()
        return result or None

    def _coherence_audits(
        self,
        detector_payloads: dict[str, Any],
        split_lookup: dict[str, str],
    ) -> list[AuditFinding]:
        findings = []
        test_records = self._records_for_split(split_lookup, "test")
        for detector_id, payload in detector_payloads.items():
            if not payload.get("available"):
                continue
            groups = payload["groups"].get("generator", {})
            real_groups = {
                key: value for key, value in groups.items() if key in {"real", "hard_negative"}
            }
            passed = len(real_groups) < 2
            findings.append(
                AuditFinding(
                    control=f"real_vs_real_coherence::{detector_id}",
                    passed=passed,
                    details=(
                        "该评测集没有独立真实/硬负样本组，未形成 coherence 判断。"
                        if passed
                        else "真实与硬负样本已分别分组，需要检查分组性能差异。"
                    ),
                    values={"test_count": len(test_records)},
                )
            )
        return findings

    def _cross_encoding_audits(
        self,
        detector_payloads: dict[str, Any],
    ) -> list[AuditFinding]:
        findings = []
        for detector_id, payload in detector_payloads.items():
            if not payload.get("available"):
                continue
            metrics = payload["groups"].get("compression", {})
            available = [
                value for value in metrics.values() if not value.get("insufficient_classes")
            ]
            if len(available) >= 2:
                tpr_values = [float(value["tpr_at_fpr_1pct"]) for value in available]
                drop = max(tpr_values) - min(tpr_values)
                findings.append(
                    AuditFinding(
                        control=f"cross_encoding_robustness::{detector_id}",
                        passed=drop <= 0.20,
                        details=f"不同压缩条件下 TPR@FPR=1% 最大差异为 {drop:.3f}。",
                        values={"max_drop": drop},
                    )
                )
        return findings

    def _records_for_split(
        self,
        split_lookup: dict[str, str],
        split_name: str,
    ) -> list[DatasetRecord]:
        return [
            record
            for record in self.records
            if split_lookup.get(record.sample_id) == split_name and record.usable
        ]

    @staticmethod
    def _unavailable_detector(detector: EvaluationDetector, error: str) -> dict[str, Any]:
        return {
            "detector_id": detector.detector_id,
            "display_name": detector.display_name,
            "version": detector.version,
            "license": detector.license,
            "description": detector.description,
            "available": False,
            "error": error,
            "metrics": None,
            "groups": {},
            "localization": None,
            "failure_cases": [],
            "audits": [],
        }


def _resolution_bucket(record: DatasetRecord) -> str:
    smallest_side = min(record.width, record.height)
    if smallest_side >= 1080:
        return ">=1080p"
    if smallest_side >= 720:
        return "720p"
    if smallest_side >= 480:
        return "480p"
    return "<480p"


def _extract_thumbnail(media_path: Path, output_path: Path) -> Path | None:
    if not media_path.exists():
        return None
    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_POS_MSEC, 1000)
    success, frame = capture.read()
    capture.release()
    if not success or frame is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(str(output_path), frame):
        return output_path
    return None
