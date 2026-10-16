from __future__ import annotations

from pathlib import Path

import pytest

from dhfp.evaluation.schema import (
    AuthorizationStatus,
    DatasetLabel,
    DatasetRecord,
    ForgeryType,
)
from dhfp.training.augment import (
    DEFAULT_SPECS,
    augmented_records,
    build_plan,
    parse_specs,
    plan_summary,
)


def _record(sample_id: str, label: DatasetLabel, path: Path) -> DatasetRecord:
    return DatasetRecord(
        sample_id=sample_id,
        media_path=str(path),
        label=label,
        generator="chatcut" if label == DatasetLabel.GENERATED else "authorized-real",
        forgery_type=ForgeryType.COMBINED if label == DatasetLabel.GENERATED else ForgeryType.NONE,
        authorization_status=AuthorizationStatus.SELF_OWNED,
        duration_seconds=8.0,
        width=1080,
        height=1920,
        frame_rate=30.0,
        has_audio=True,
        compression="h264",
        subset="chatcut/avatar=a1",
    )


def _records(tmp_path: Path) -> list[DatasetRecord]:
    return [
        _record("gen-1", DatasetLabel.GENERATED, tmp_path / "gen.mp4"),
        _record("real-1", DatasetLabel.REAL, tmp_path / "real.mp4"),
    ]


def test_build_plan_applies_identical_strength_to_every_label(tmp_path: Path) -> None:
    specs = [DEFAULT_SPECS["crf23"], DEFAULT_SPECS["crf32"]]

    jobs = build_plan(_records(tmp_path), tmp_path / "aug", specs)

    summary = plan_summary(jobs)
    assert summary["same_strength_across_labels"] is True
    assert summary["per_label"] == {"generated": 2, "real": 2}
    generated = next(job for job in jobs if job.sample_id == "gen-1" and job.spec.name == "crf32")
    real = next(job for job in jobs if job.sample_id == "real-1" and job.spec.name == "crf32")
    codec_start = generated.command.index("-c:v")
    assert generated.command[codec_start:-1] == real.command[codec_start:-1]
    assert "-crf" in generated.command
    assert generated.command[generated.command.index("-crf") + 1] == "32"
    assert real.command[real.command.index("-crf") + 1] == "32"


def test_augmented_records_keep_label_and_mark_subset(tmp_path: Path) -> None:
    specs = [DEFAULT_SPECS["crf28"]]

    augmented = augmented_records(_records(tmp_path), specs, tmp_path / "aug")

    assert [record.label for record in augmented] == [
        DatasetLabel.GENERATED,
        DatasetLabel.REAL,
    ]
    assert all("+crf28" in record.subset for record in augmented)
    assert all("reencode:crf28" in (record.compression or "") for record in augmented)
    assert all("同强度重编码增广" in (record.notes or "") for record in augmented)
    assert augmented[0].sample_id == "gen-1__crf28"
    assert augmented[0].media_path.endswith("crf28/gen-1__crf28.mp4")


def test_parse_specs_rejects_unknown_strength() -> None:
    with pytest.raises(ValueError):
        parse_specs("crf99")


def test_parse_specs_defaults_to_medium_strength() -> None:
    specs = parse_specs(None)

    assert [spec.name for spec in specs] == ["crf28"]
