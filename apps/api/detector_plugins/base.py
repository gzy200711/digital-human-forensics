from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from apps.api.services.media import MediaProbe


@dataclass
class SignalWindow:
    start: float
    end: float
    score: float
    reason: str


@dataclass
class SignalResult:
    signal_id: str
    display_name: str
    score: float | None
    available: bool
    direction: str
    raw_value: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    time_windows: list[SignalWindow] = field(default_factory=list)
    model_name: str = "heuristic"
    model_version: str = "bootstrap-1"
    license: str = "internal"
    latency_ms: int = 0


@dataclass
class ProvenanceResult:
    explicit_ai_indicators: list[str]
    checks: list[dict[str, Any]]
    render_chain_hits: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PluginContext:
    job_id: str
    original_filename: str
    source_object_key: str
    media_path: Path
    probe: MediaProbe
    settings: Any
    face_analysis: Any = None


class DetectorPlugin(Protocol):
    signal_id: str
    display_name: str
    model_name: str
    model_version: str
    license: str

    def analyze(self, context: PluginContext) -> SignalResult: ...
