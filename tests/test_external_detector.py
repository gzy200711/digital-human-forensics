from __future__ import annotations

import sys

from dhfp.evaluation.schema import (
    AuthorizationStatus,
    DatasetLabel,
    DatasetRecord,
    ForgeryType,
)
from dhfp.plugins.external import ExternalCommandDetector


def _record() -> DatasetRecord:
    return DatasetRecord(
        sample_id="external-1",
        media_path="/tmp/external.mp4",
        label=DatasetLabel.GENERATED,
        generator="external",
        forgery_type=ForgeryType.OTHER,
        authorization_status=AuthorizationStatus.NOT_APPLICABLE,
        duration_seconds=1.0,
        width=320,
        height=240,
        frame_rate=25.0,
        has_audio=False,
    )


def test_external_detector_keeps_parent_environment() -> None:
    detector = ExternalCommandDetector(
        {
            "id": "env-check",
            "display_name": "Environment check",
            "version": "1",
            "license": "internal",
            "command": [
                sys.executable,
                "-c",
                (
                    "import json, os; "
                    "print(json.dumps({'score': 0.75, 'details': {"
                    "'path': os.environ.get('PATH'), "
                    "'custom': os.environ.get('DETECTOR_CUSTOM')}}))"
                ),
                "{media_path}",
            ],
            "env": {"DETECTOR_CUSTOM": "enabled"},
        }
    )

    prediction = detector.predict(_record())

    assert prediction.available
    assert prediction.score == 0.75
    assert prediction.details["path"]
    assert prediction.details["custom"] == "enabled"
