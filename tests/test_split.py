from __future__ import annotations

from dhfp.evaluation.audit import audit_split
from dhfp.evaluation.schema import (
    AuthorizationStatus,
    DatasetLabel,
    DatasetRecord,
    ForgeryType,
)
from dhfp.evaluation.split import build_splits


def make_record(
    sample_id: str,
    *,
    label: DatasetLabel,
    generator: str,
    person: str,
) -> DatasetRecord:
    return DatasetRecord(
        sample_id=sample_id,
        media_path=f"/tmp/{sample_id}.mp4",
        label=label,
        generator=generator,
        forgery_type=(
            ForgeryType.FULL_SYNTHESIS if label == DatasetLabel.GENERATED else ForgeryType.NONE
        ),
        source_person_id=person,
        authorization_status=AuthorizationStatus.VERIFIED,
        duration_seconds=5,
        width=720,
        height=1280,
        frame_rate=30,
        has_audio=True,
    )


def test_generator_and_identity_disjoint_split() -> None:
    records = []
    for generator_index in range(5):
        for person_index in range(4):
            records.append(
                make_record(
                    f"gen-{generator_index}-{person_index}",
                    label=DatasetLabel.GENERATED,
                    generator=f"generator-{generator_index}",
                    person=f"person-{generator_index}-{person_index}",
                )
            )
    for person_index in range(8):
        records.append(
            make_record(
                f"real-{person_index}",
                label=DatasetLabel.REAL,
                generator="iphone",
                person=f"real-person-{person_index}",
            )
        )

    payload = build_splits(records, seed=7, generator_test_count=2)
    findings = audit_split(records, payload["splits"])

    assert all(item.passed for item in findings), [item.as_dict() for item in findings]
    assert payload["metadata"]["counts"]["test"] > 0


def test_unusable_records_are_not_reported_as_missing_from_splits() -> None:
    records = [
        make_record(
            f"gen-{generator_index}-{person_index}",
            label=DatasetLabel.GENERATED,
            generator=f"generator-{generator_index}",
            person=f"person-{generator_index}-{person_index}",
        )
        for generator_index in range(4)
        for person_index in range(2)
    ]
    records.extend(
        make_record(
            f"real-{index}",
            label=DatasetLabel.REAL,
            generator="iphone",
            person=f"real-person-{index}",
        )
        for index in range(4)
    )
    unusable = make_record(
        "pending-authorization",
        label=DatasetLabel.REAL,
        generator="iphone",
        person="pending-person",
    ).model_copy(
        update={
            "authorization_status": AuthorizationStatus.MISSING,
            "usable": False,
        }
    )
    records.append(unusable)

    payload = build_splits(records, seed=7, generator_test_count=1)
    findings = audit_split(records, payload["splits"])
    partition = next(item for item in findings if item.control == "partition_integrity")

    assert partition.passed, partition.as_dict()
    assert "pending-authorization" not in payload["splits"]["train"]


def test_split_audit_reports_unknown_sample_ids() -> None:
    records = [
        make_record(
            f"gen-{generator_index}-{person_index}",
            label=DatasetLabel.GENERATED,
            generator=f"generator-{generator_index}",
            person=f"person-{generator_index}-{person_index}",
        )
        for generator_index in range(4)
        for person_index in range(2)
    ]
    records.extend(
        make_record(
            f"real-{index}",
            label=DatasetLabel.REAL,
            generator="iphone",
            person=f"real-person-{index}",
        )
        for index in range(4)
    )
    payload = build_splits(records, seed=7, generator_test_count=1)
    payload["splits"]["train"].append("missing-sample")

    partition = next(
        item
        for item in audit_split(records, payload["splits"])
        if item.control == "partition_integrity"
    )

    assert not partition.passed
    assert partition.values["unknown"] == ["missing-sample"]
