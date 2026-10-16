from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Verdict(StrEnum):
    TRUSTED = "可信"
    UNCERTAIN = "无法确定"
    SUSPECTED = "疑似生成合成"


class SignalDirection(StrEnum):
    GENERATED = "指向生成"
    REAL = "指向真实"
    NEUTRAL = "中性"


class SignalStrength(StrEnum):
    LOW = "弱"
    MEDIUM = "中"
    HIGH = "强"


class AssessabilityStatus(StrEnum):
    PASS = "可鉴定"
    MARGINAL = "勉强可鉴定"
    FAIL = "不可鉴定"


class ModalityKey(StrEnum):
    VISUAL = "visual"
    AUDIO = "audio"
    CROSS_MODAL = "cross_modal"
    PROVENANCE = "provenance"


class ModalityStatus(StrEnum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class ProvenanceStatus(StrEnum):
    POSITIVE = "阳性"
    NEGATIVE = "阴性"
    NOT_CHECKED = "未检查"
    UNAVAILABLE = "不可用"


class MediaInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    mime_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    has_audio: bool = False


class TimeRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    score: float = Field(ge=0, le=1)
    reason: str


class EvidenceRegion(BaseModel):
    """证据作用的空间区域。坐标为归一化比例，未知时留空。"""

    model_config = ConfigDict(extra="forbid")

    label: str
    x: float | None = Field(default=None, ge=0, le=1)
    y: float | None = Field(default=None, ge=0, le=1)
    width: float | None = Field(default=None, ge=0, le=1)
    height: float | None = Field(default=None, ge=0, le=1)


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    license: str | None = None


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    signal: str
    display_name: str
    layer: str | None = None
    source: str | None = None
    direction: SignalDirection
    strength: SignalStrength
    score: float | None = Field(default=None, ge=0, le=1)
    unit: str | None = None
    raw_value: dict[str, Any] = Field(default_factory=dict)
    explanation: str
    time_range: TimeRange | None = None
    region: EvidenceRegion | None = None
    model: EvidenceModel | None = None
    uncertainty: str | None = None
    limitations: list[str] = Field(default_factory=list)
    available: bool = True


class ModalityCoverage(BaseModel):
    """实际被有效分析的媒体比例；None 表示当前版本无法估计。"""

    model_config = ConfigDict(extra="forbid")

    full_frame: float | None = Field(default=None, ge=0, le=1)
    face: float | None = Field(default=None, ge=0, le=1)
    audio: float | None = Field(default=None, ge=0, le=1)
    cross_modal: float | None = Field(default=None, ge=0, le=1)
    notes: list[str] = Field(default_factory=list)


class ModalityVerdict(BaseModel):
    """模态级结论。模态之间方向不同不构成冲突。"""

    model_config = ConfigDict(extra="forbid")

    modality: ModalityKey
    label: str
    verdict: Verdict
    score: float | None = Field(default=None, ge=0, le=1)
    coverage: float | None = Field(default=None, ge=0, le=1)
    status: ModalityStatus = ModalityStatus.AVAILABLE
    signal_ids: list[str] = Field(default_factory=list)
    summary: str
    limitations: list[str] = Field(default_factory=list)


class GrayZone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum: float = Field(ge=0, le=1)
    maximum: float = Field(ge=0, le=1)
    score: float | None = Field(default=None, ge=0, le=1)
    in_gray_zone: bool = False
    note: str = ""


class CalibratedProbability(BaseModel):
    """只在声明的评测分布与阈值下成立的校准概率，不是事实确定性。"""

    model_config = ConfigDict(extra="forbid")

    value: float | None = Field(default=None, ge=0, le=1)
    is_calibrated: bool = False
    calibration_version: str = "unknown"
    calibration_group: str = "unknown"
    distribution: str = "未声明的评测分布"
    confidence_interval: tuple[float, float] | None = None
    semantics: str = (
        "该数值只表示在声明的评测分布和阈值下的生成合成风险，不代表事实确定性。"
    )


class ThresholdPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str
    source: str
    gray_zone: tuple[float, float]
    target_fprs: list[float] = Field(default_factory=list)
    real_negative_samples: int | None = Field(default=None, ge=0)
    fpr_evidence_status: Literal["verified", "insufficient", "unknown"] = "unknown"
    statement: str


class ModelSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registry_version: str
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_research_only: bool = True
    note: str = ""


class AssessabilityFactor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | str | bool | None
    status: Literal["good", "warning", "bad", "unknown"]
    explanation: str


class Assessability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=1)
    status: AssessabilityStatus
    face_detected_ratio: float | None = Field(default=None, ge=0, le=1)
    factors: list[AssessabilityFactor] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class ProvenanceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: ProvenanceStatus
    positive: bool = False
    details: dict[str, Any] = Field(default_factory=dict)
    explanation: str


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checks: list[ProvenanceCheck] = Field(default_factory=list)
    explicit_ai_indicators: list[str] = Field(default_factory=list)
    status: Literal["阳性", "无声明", "未检查"] = "无声明"
    note: str = "元数据未命中不能作为真实性的阴性证据。"
    limitation: str = (
        "缺失、损坏或被清除的来源声明不能作为真实性证据；阳性声明也需要验证签名与信任链。"
    )


class ArtifactLinks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    media_url: str | None = None
    heatmap_url: str | None = None
    keyframe_url: str | None = None
    json_report_url: str | None = None
    html_report_url: str | None = None


class ModelVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    license: str | None = None


class ReviewStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    kind: Literal["生成风险", "真实性提示", "不确定性", "质量限制"]


class EvidenceReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["not_triggered", "completed", "unavailable"]
    triggered: bool
    provider: str
    model_version: str | None = None
    license: str | None = None
    summary: str
    statements: list[ReviewStatement] = Field(default_factory=list)
    rejected_claim_count: int = Field(default=0, ge=0)
    note: str

    @model_validator(mode="after")
    def validate_state(self) -> EvidenceReview:
        if not self.triggered and self.status != "not_triggered":
            raise ValueError("未触发复核时状态必须为 not_triggered")
        if self.status == "not_triggered" and self.statements:
            raise ValueError("未触发复核不能包含陈述")
        if self.status == "completed" and not self.statements:
            raise ValueError("完成的复核必须至少包含一条有效陈述")
        return self


class ForensicReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0", "1.1"] = "1.1"
    report_id: str
    job_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    media: MediaInfo
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    calibrated_score: float = Field(ge=0, le=1)
    calibrated_probability: CalibratedProbability | None = None
    summary: str
    modality_verdicts: list[ModalityVerdict] = Field(default_factory=list)
    coverage: ModalityCoverage = Field(default_factory=ModalityCoverage)
    gray_zone: GrayZone | None = None
    abstention_reasons: list[str] = Field(default_factory=list)
    abstention_codes: list[str] = Field(default_factory=list)
    time_ranges: list[TimeRange] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    review: EvidenceReview | None = None
    assessability: Assessability
    provenance: Provenance
    limitations: list[str] = Field(default_factory=list)
    recommended_action: Literal["通过", "人工复核", "拦截并允许申诉"]
    artifacts: ArtifactLinks = Field(default_factory=ArtifactLinks)
    model_versions: list[ModelVersion] = Field(default_factory=list)
    model_snapshot: ModelSnapshot | None = None
    threshold_policy: ThresholdPolicy | None = None
    audit_trace_id: str | None = None
    processing: dict[str, Any] = Field(default_factory=dict)
