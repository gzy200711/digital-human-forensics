from __future__ import annotations

from dhfp.evaluation.schema import (
    AuthorizationStatus,
    DatasetLabel,
    DatasetRecord,
)
from dhfp.learning.continual import (
    ContinualGateThresholds,
    evaluate_regression_gate,
    select_replay_samples,
    write_continual_update_manifest,
)


def _record(sample_id: str, label: DatasetLabel, generator: str) -> DatasetRecord:
    return DatasetRecord(
        sample_id=sample_id,
        media_path=f"/tmp/{sample_id}.mp4",
        label=label,
        generator=generator,
        authorization_status=AuthorizationStatus.VERIFIED,
        duration_seconds=3,
        width=640,
        height=360,
        frame_rate=25,
        has_audio=True,
    )


def test_replay_selection_is_deterministic_and_stratified() -> None:
    records = [
        _record("real-1", DatasetLabel.REAL, "camera"),
        _record("real-2", DatasetLabel.REAL, "camera"),
        _record("negative-1", DatasetLabel.HARD_NEGATIVE, "filter"),
        _record("generated-1", DatasetLabel.GENERATED, "generator-a"),
        _record("generated-2", DatasetLabel.GENERATED, "generator-b"),
        _record("heldout", DatasetLabel.GENERATED, "generator-test"),
    ]

    first = select_replay_samples(
        records,
        max_samples=4,
        seed=7,
        excluded_generators={"generator-test"},
    )
    second = select_replay_samples(
        records,
        max_samples=4,
        seed=7,
        excluded_generators={"generator-test"},
    )

    assert first.sample_ids == second.sample_ids
    assert "heldout" not in first.sample_ids
    assert set(first.label_counts) == {"real", "hard_negative", "generated"}


def test_regression_gate_blocks_metric_drops(tmp_path) -> None:
    previous = {
        "detectors": {
            "fusion-pipeline": {
                "available": True,
                "metrics": {
                    "tpr_at_fpr_1pct": 0.72,
                    "ece": 0.04,
                    "auc": 0.90,
                },
                "localization": {"temporal": {"average_precision": 0.70}},
            }
        }
    }
    candidate = {
        "detectors": {
            "fusion-pipeline": {
                "available": True,
                "metrics": {
                    "tpr_at_fpr_1pct": 0.64,
                    "ece": 0.04,
                    "auc": 0.88,
                },
                "localization": {"temporal": {"average_precision": 0.50}},
            }
        }
    }

    result = evaluate_regression_gate(
        previous,
        candidate,
        detector_id="fusion-pipeline",
        thresholds=ContinualGateThresholds(),
    )

    assert result.passed is False
    failed = {finding.control for finding in result.findings if not finding.passed}
    assert "absolute_tpr_at_fpr_1pct" in failed
    assert "tpr_regression" in failed
    assert "localization_ap_regression" in failed


def test_continual_manifest_records_release_gate(tmp_path) -> None:
    records = [
        _record("generated-1", DatasetLabel.GENERATED, "generator-a"),
        _record("real-1", DatasetLabel.REAL, "camera"),
    ]
    replay = select_replay_samples(records, max_samples=2, seed=1)
    report = {
        "detectors": {
            "fusion-pipeline": {
                "available": True,
                "metrics": {
                    "tpr_at_fpr_1pct": 0.80,
                    "ece": 0.03,
                    "auc": 0.95,
                },
            }
        }
    }
    gate = evaluate_regression_gate(
        report,
        report,
        detector_id="fusion-pipeline",
    )
    manifest = write_continual_update_manifest(
        tmp_path / "manifest.json",
        candidate_detector_id="fusion-pipeline",
        replay=replay,
        gate=gate,
    )

    assert manifest.exists()
    assert '"release_eligible": true' in manifest.read_text(encoding="utf-8")
