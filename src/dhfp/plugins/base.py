from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from dhfp.evaluation.localization import SpatialBox, TimeSegment
from dhfp.evaluation.schema import DatasetRecord


class DetectorUnavailable(RuntimeError):
    pass


@dataclass
class EvaluationPrediction:
    sample_id: str
    score: float | None
    available: bool
    latency_ms: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    temporal_segments: list[TimeSegment] = field(default_factory=list)
    spatial_boxes: list[SpatialBox] = field(default_factory=list)


class EvaluationDetector(Protocol):
    detector_id: str
    display_name: str
    version: str
    license: str
    description: str

    def predict(self, record: DatasetRecord) -> EvaluationPrediction: ...
