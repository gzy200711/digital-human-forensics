from __future__ import annotations

import time

import pytest

from apps.api.api.jobs import _validate_media_type
from apps.api.config import Settings


@pytest.mark.parametrize(
    ("filename", "mime_type"),
    [
        ("sample.avi", "video/x-msvideo"),
        ("sample.m4v", "video/x-m4v"),
        ("sample.aac", "audio/aac"),
        ("sample.flac", "audio/flac"),
    ],
)
def test_supported_media_types_are_consistent(filename: str, mime_type: str) -> None:
    _validate_media_type(
        filename,
        mime_type,
        Settings(_env_file=None).allowed_media_type_set,
    )


def test_health_includes_storage_status(api_client) -> None:
    response = api_client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["storage"] == "ok"


def test_upload_requires_explicit_consent(api_client, synthetic_video) -> None:
    with synthetic_video.open("rb") as handle:
        response = api_client.post(
            "/v1/jobs",
            files={"file": (synthetic_video.name, handle, "video/mp4")},
            data={"retention_days": "1"},
        )
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "CONSENT_REQUIRED"


def test_upload_flow(api_client, synthetic_video) -> None:
    with synthetic_video.open("rb") as handle:
        response = api_client.post(
            "/v1/jobs",
            files={"file": (synthetic_video.name, handle, "video/mp4")},
            data={"retention_days": "1", "consent_ack": "true"},
        )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]

    status_payload = None
    for _ in range(60):
        status_response = api_client.get(f"/v1/jobs/{job_id}")
        assert status_response.status_code == 200
        status_payload = status_response.json()
        if status_payload["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert status_payload is not None
    assert status_payload["status"] == "completed", status_payload

    report_response = api_client.get(f"/v1/jobs/{job_id}/report")
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["schema_version"] == "1.1"
    assert report["verdict"] in {"可信", "无法确定", "疑似生成合成"}
    assert report["artifacts"]["media_url"]
    # 模态级结论必须分离输出，且溯源/取证结果不得混写。
    modalities = {item["modality"] for item in report["modality_verdicts"]}
    assert {"visual", "audio", "cross_modal", "provenance"} <= modalities
    assert report["calibrated_probability"]["is_calibrated"] is False
    assert report["calibrated_probability"]["confidence_interval"] is None
    assert report["threshold_policy"]["fpr_evidence_status"] == "unknown"
    assert report["audit_trace_id"] == job_id
    for item in report["evidence"]:
        assert item["id"]
        assert item["model"]["name"]

    media_response = api_client.get(
        report["artifacts"]["media_url"],
        headers={"Range": "bytes=-4"},
    )
    assert media_response.status_code == 206
    assert len(media_response.content) == 4
    unit, _, size_value = media_response.headers["content-range"].partition(" ")
    range_value, _, total_size = size_value.partition("/")
    start_value, end_value = range_value.split("-")
    assert unit == "bytes"
    assert int(end_value) - int(start_value) + 1 == 4
    assert int(total_size) > 4

    appeal_response = api_client.post(
        f"/v1/jobs/{job_id}/appeal",
        json={"reason": "测试申诉理由，需要人工复核具体证据。"},
    )
    assert appeal_response.status_code == 201

    history_response = api_client.get("/v1/jobs", params={"limit": 20, "offset": 0})
    assert history_response.status_code == 200
    assert history_response.headers["cache-control"] == "private, no-store"
    history = history_response.json()
    assert history["total"] == 1
    assert history["limit"] == 20
    assert history["offset"] == 0
    history_item = history["items"][0]
    assert history_item["job_id"] == job_id
    assert history_item["original_filename"] == synthetic_video.name
    assert history_item["status"] == "completed"
    assert history_item["report_ready"] is True
    assert history_item["verdict"] == report["verdict"]
    assert history_item["confidence"] == report["confidence"]

    search_response = api_client.get(
        "/v1/jobs",
        params={"q": synthetic_video.stem.upper(), "status": "completed"},
    )
    assert search_response.status_code == 200
    assert search_response.json()["total"] == 1

    verdict_filter_response = api_client.get(
        "/v1/jobs",
        params={"verdict": report["verdict"]},
    )
    assert verdict_filter_response.status_code == 200
    assert verdict_filter_response.json()["total"] == 1

    missing_filter_response = api_client.get(
        "/v1/jobs",
        params={"status": "failed"},
    )
    assert missing_filter_response.status_code == 200
    assert missing_filter_response.json()["total"] == 0

    invalid_filter_response = api_client.get(
        "/v1/jobs",
        params={"status": "unknown"},
    )
    assert invalid_filter_response.status_code == 422


def test_queue_failure_returns_503(
    api_client,
    synthetic_video,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apps.api.api import jobs as jobs_module

    def fail_enqueue(*_args, **_kwargs) -> None:
        raise RuntimeError("broker unavailable")

    monkeypatch.setattr(jobs_module, "enqueue_job", fail_enqueue)
    with synthetic_video.open("rb") as handle:
        response = api_client.post(
            "/v1/jobs",
            files={"file": (synthetic_video.name, handle, "video/mp4")},
            data={"retention_days": "1", "consent_ack": "true"},
        )

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "QUEUE_ENQUEUE_FAILED"
