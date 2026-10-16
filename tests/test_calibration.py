from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apps.api.detector_plugins.base import SignalResult
from apps.api.services.fusion import FusionConfig, FusionEngine
from dhfp.calibration import CalibrationProfile, fit_temperature, quantile_thresholds
from dhfp.evaluation.cli import main


def test_bootstrap_calibration_has_no_bias() -> None:
    profile = CalibrationProfile.load("configs/calibration.json")

    assert profile.bias == 0.0
    assert profile.calibrate(0.5) == pytest.approx(0.5, abs=1e-9)
    assert profile.thresholds.method == "quantile"
    assert profile.thresholds.trusted_max < profile.thresholds.suspect_min


def test_quantile_thresholds_follow_score_distribution() -> None:
    scores = np.linspace(0.0, 1.0, 101)

    thresholds = quantile_thresholds(scores, trusted_quantile=0.25, suspect_quantile=0.75)

    assert thresholds.trusted_max == pytest.approx(0.25, abs=1e-9)
    assert thresholds.suspect_min == pytest.approx(0.75, abs=1e-9)
    assert thresholds.method == "quantile"
    assert thresholds.calibration_count == 101
    assert thresholds.real_negative_count is None


def test_quantile_thresholds_reject_bad_quantiles() -> None:
    with pytest.raises(ValueError):
        quantile_thresholds(np.linspace(0, 1, 10), trusted_quantile=0.8, suspect_quantile=0.2)


def test_fit_temperature_disables_bias_by_default() -> None:
    profile = fit_temperature(
        np.asarray([0.2, 0.3, 0.7, 0.8]),
        np.asarray([0, 0, 1, 1]),
    )

    assert profile.bias == 0.0
    assert "禁用" in profile.notes


def test_calibrate_cli_writes_quantile_thresholds(tmp_path: Path) -> None:
    report_path = tmp_path / "stage0_report.json"
    report_path.write_text(
        json.dumps(
            {
                "detectors": {
                    "fusion-pipeline": {
                        "available": True,
                        "metrics": {"count": 4000, "negative_count": 3000},
                        "validation_score_quantiles": {"0.25": 0.31, "0.75": 0.68},
                        "test_score_quantiles": {"0.25": 0.40, "0.75": 0.90},
                        "validation_count": 3500,
                        "validation_negative_count": 3000,
                        "calibration": {
                            "version": "fitted",
                            "method": "temperature",
                            "temperature": 1.2,
                            "bias": -0.08,
                            "thresholds": {"trusted_max": 0.25, "suspect_min": 0.75},
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "calibration.json"

    code = main(
        [
            "calibrate",
            "--report",
            str(report_path),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    profile = json.loads(output.read_text(encoding="utf-8"))
    assert profile["bias"] == 0.0
    assert profile["thresholds"]["trusted_max"] == pytest.approx(0.31)
    assert profile["thresholds"]["suspect_min"] == pytest.approx(0.68)
    assert profile["thresholds"]["method"] == "quantile"
    assert profile["thresholds"]["calibration_count"] == 3500
    assert profile["thresholds"]["real_negative_count"] == 3000


def test_calibrate_cli_refuses_to_tune_on_test_quantiles(tmp_path: Path) -> None:
    report_path = tmp_path / "stage0_report.json"
    report_path.write_text(
        json.dumps(
            {
                "detectors": {
                    "fusion-pipeline": {
                        "available": True,
                        "calibration": {},
                        "test_score_quantiles": {"0.25": 0.2, "0.75": 0.8},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "calibrate",
            "--report",
            str(report_path),
            "--output",
            str(tmp_path / "calibration.json"),
        ]
    )

    assert code == 1


def test_quantile_thresholds_drive_gray_zone() -> None:
    config = FusionConfig(gray_zone_min=0.2, gray_zone_max=0.8)
    calibration = CalibrationProfile(
        thresholds=quantile_thresholds(
            np.linspace(0.0, 1.0, 101),
            trusted_quantile=0.4,
            suspect_quantile=0.6,
        )
    )
    engine = FusionEngine(config, calibration)

    assert engine.gray_zone_min == pytest.approx(0.4)
    assert engine.gray_zone_max == pytest.approx(0.6)
    decision = engine.decide(
        [SignalResult("visual_artifact", "帧级视觉伪影", 0.5, True, "中性")],
        provenance_positive=False,
        assessability_status="pass",
        beauty_filter_score=0.0,
    )
    assert decision.calibrated_score == pytest.approx(0.5)
    assert decision.verdict == "无法确定"


def test_manual_thresholds_fall_back_to_fusion_config() -> None:
    config = FusionConfig(gray_zone_min=0.2, gray_zone_max=0.8)
    calibration = CalibrationProfile(
        thresholds=quantile_thresholds(np.linspace(0.0, 1.0, 11)).model_copy(
            update={"method": "manual", "trusted_max": 0.45, "suspect_min": 0.55}
        )
    )

    engine = FusionEngine(config, calibration)

    assert engine.gray_zone_min == pytest.approx(0.2)
    assert engine.gray_zone_max == pytest.approx(0.8)
