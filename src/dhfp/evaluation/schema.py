from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetLabel(StrEnum):
    REAL = "real"
    GENERATED = "generated"
    HARD_NEGATIVE = "hard_negative"


class ForgeryType(StrEnum):
    NONE = "none"
    FULL_SYNTHESIS = "full_synthesis"
    FACE_SWAP = "face_swap"
    LIP_SYNC = "lip_sync"
    AUDIO_CLONE = "audio_clone"
    COMBINED = "combined"
    OTHER = "other"


class AuthorizationStatus(StrEnum):
    VERIFIED = "verified"
    SELF_OWNED = "self_owned"
    PUBLIC_DOMAIN = "public_domain"
    PENDING = "pending"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"


class TemporalAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> TemporalAnnotation:
        if self.start >= self.end:
            raise ValueError("时间标注必须满足 start < end")
        return self


class SpatialAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: float | None = Field(default=None, ge=0)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(gt=0)
    y2: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_box(self) -> SpatialAnnotation:
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("空间标注必须满足 x1 < x2 且 y1 < y2")
        return self


class DatasetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    media_path: str
    label: DatasetLabel
    generator: str
    generator_version: str | None = None
    forgery_type: ForgeryType = ForgeryType.NONE
    source_person_id: str | None = None
    authorization_status: AuthorizationStatus
    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_rate: float = Field(gt=0)
    has_audio: bool
    compression: str | None = None
    beauty_filter: bool = False
    subset: str = "default"
    source_dataset: str | None = None
    generation_date: str | None = None
    prompt_or_driver: str | None = None
    watermark: str | None = None
    temporal_annotations: list[TemporalAnnotation] = Field(default_factory=list)
    spatial_annotations: list[SpatialAnnotation] = Field(default_factory=list)
    usable: bool = True
    notes: str | None = None

    @model_validator(mode="after")
    def validate_authorization(self) -> DatasetRecord:
        if self.label in {DatasetLabel.REAL, DatasetLabel.HARD_NEGATIVE}:
            allowed = {
                AuthorizationStatus.VERIFIED,
                AuthorizationStatus.SELF_OWNED,
                AuthorizationStatus.PUBLIC_DOMAIN,
                AuthorizationStatus.NOT_APPLICABLE,
            }
            if self.authorization_status not in allowed:
                self.usable = False
        for annotation in self.temporal_annotations:
            if annotation.end > self.duration_seconds:
                raise ValueError("时间标注不能超出媒体时长")
        for annotation in self.spatial_annotations:
            if annotation.timestamp is not None and annotation.timestamp > self.duration_seconds:
                raise ValueError("空间标注时间不能超出媒体时长")
        return self

    @property
    def is_positive(self) -> bool:
        return self.label == DatasetLabel.GENERATED

    @property
    def group_key(self) -> str:
        return self.generator or self.source_dataset or self.sample_id


def load_index(path: str | Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(DatasetRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number} 数据记录无效: {exc}") from exc
    return records


def write_index(records: list[DatasetRecord], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json(exclude_none=True))
            handle.write("\n")


def index_summary(records: list[DatasetRecord]) -> dict:
    usable = [record for record in records if record.usable]
    labels: dict[str, int] = {}
    generators: dict[str, int] = {}
    forgery_types: dict[str, int] = {}
    for record in records:
        labels[record.label.value] = labels.get(record.label.value, 0) + 1
        generators[record.generator] = generators.get(record.generator, 0) + 1
        forgery_types[record.forgery_type.value] = (
            forgery_types.get(record.forgery_type.value, 0) + 1
        )
    return {
        "total": len(records),
        "usable": len(usable),
        "unusable": len(records) - len(usable),
        "labels": labels,
        "generators": generators,
        "forgery_types": forgery_types,
        "duration_seconds": round(sum(record.duration_seconds for record in records), 3),
    }


def dump_summary(records: list[DatasetRecord], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(index_summary(records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
