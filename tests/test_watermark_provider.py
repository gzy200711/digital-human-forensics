from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from apps.api.detector_plugins import provenance as provenance_module
from apps.api.detector_plugins.base import PluginContext
from apps.api.detector_plugins.provenance import ProvenanceAnalyzer
from apps.api.detector_plugins.watermark import WatermarkOutcome, WatermarkVerifier
from apps.api.services.media import MediaProbe, MediaStream


def _settings(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "watermark_provider": "synthid",
        "watermark_provider_label": None,
        "watermark_endpoint": "https://verifier.example/api/detect",
        "watermark_api_key": "key-123",
        "watermark_app_id": None,
        "watermark_secret": None,
        "watermark_media_field": "file",
        "watermark_extra_fields": {},
        "watermark_auth_header": "Authorization",
        "watermark_auth_scheme": "Bearer",
        "watermark_sign_requests": True,
        "watermark_verify_tls": True,
        "watermark_timeout_seconds": 5.0,
        "watermark_allow_media_upload": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _media(tmp_path: Path) -> Path:
    path = tmp_path / "sample.mp4"
    path.write_bytes(b"digital-human-sample")
    return path


def _verifier(settings: SimpleNamespace, handler: Any) -> WatermarkVerifier:
    verifier = WatermarkVerifier.from_settings(settings)
    verifier.client = httpx.Client(transport=httpx.MockTransport(handler))
    return verifier


def test_disabled_provider_is_not_checked(tmp_path: Path) -> None:
    verifier = WatermarkVerifier.from_settings(_settings(watermark_provider="off"))
    outcome = verifier.verify(_media(tmp_path))

    assert outcome.status == "not_checked"
    assert not outcome.positive
    assert outcome.to_check()["status"] == "未检查"


def test_auto_without_credentials_is_unavailable(tmp_path: Path) -> None:
    settings = _settings(
        watermark_provider="auto",
        watermark_endpoint=None,
        watermark_api_key=None,
        watermark_app_id=None,
        watermark_secret=None,
    )
    outcome = WatermarkVerifier.from_settings(settings).verify(_media(tmp_path))

    assert outcome.status == "unavailable"
    assert "WATERMARK_PROVIDER=auto" in outcome.explanation


def test_media_upload_requires_explicit_consent(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("未开启上传同意时不应发起请求")

    settings = _settings(watermark_allow_media_upload=False)
    outcome = _verifier(settings, handler).verify(_media(tmp_path))

    assert outcome.status == "not_checked"
    assert outcome.details["consent_required"] is True


def test_synthid_positive_result(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer key-123"
        return httpx.Response(
            200,
            json={
                "detected": True,
                "confidence": 0.97,
                "watermark_id": "synthid-v1-8f2c",
            },
        )

    outcome = _verifier(_settings(), handler).verify(_media(tmp_path))

    assert outcome.status == "positive"
    assert outcome.detected is True
    assert outcome.score == 0.97
    assert outcome.watermark_id == "synthid-v1-8f2c"
    assert outcome.to_check()["status"] == "阳性"


def test_code_wrapped_negative_result(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 0, "data": {"watermark_detected": "false"}})

    outcome = _verifier(_settings(), handler).verify(_media(tmp_path))

    assert outcome.status == "negative"
    assert outcome.detected is False
    assert not outcome.positive


def test_service_error_is_unavailable(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 401, "message": "invalid signature"})

    outcome = _verifier(_settings(), handler).verify(_media(tmp_path))

    assert outcome.status == "unavailable"
    assert "invalid signature" in outcome.explanation


def test_unrecognized_response_is_unavailable(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": "ok"})

    outcome = _verifier(_settings(), handler).verify(_media(tmp_path))

    assert outcome.status == "unavailable"
    assert not outcome.positive


def test_alliance_request_is_signed(tmp_path: Path) -> None:
    media = _media(tmp_path)
    captured: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json={"detected": True, "watermark_id": "alliance-42"})

    settings = _settings(
        watermark_provider="alliance",
        watermark_api_key=None,
        watermark_app_id="app-7",
        watermark_secret="secret-9",
    )
    outcome = _verifier(settings, handler).verify(media)

    assert outcome.status == "positive"
    body = captured["body"]
    assert b"app-7" in body
    assert b"media_sha256" in body
    digest = hashlib.sha256(media.read_bytes()).hexdigest()
    assert digest.encode() in body
    assert b"sign" in body


class _StubVerifier:
    def __init__(self, outcome: WatermarkOutcome) -> None:
        self.outcome = outcome

    def verify(self, media_path: Path) -> WatermarkOutcome:
        return self.outcome


def _probe(tmp_path: Path) -> MediaProbe:
    video = MediaStream(
        index=0,
        codec_type="video",
        codec_name="h264",
        width=720,
        height=1280,
        frame_rate=25.0,
        sample_rate=None,
        channels=None,
        duration_seconds=8.0,
        tags={},
    )
    return MediaProbe(
        path=tmp_path / "sample.mp4",
        format_name="mov,mp4,m4a",
        format_long_name="QuickTime / MOV",
        duration_seconds=8.0,
        size_bytes=4096,
        bit_rate=800000,
        tags={},
        streams=[video],
    )


def test_provenance_analyzer_uses_watermark_result(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(provenance_module.shutil, "which", lambda name: None)
    outcome = WatermarkOutcome(
        status="positive",
        provider_id="synthid",
        provider_label="SynthID 水印核验服务",
        explanation="检出隐式水印",
        detected=True,
    )
    probe = _probe(tmp_path)
    context = PluginContext(
        job_id="watermark",
        original_filename="sample.mp4",
        source_object_key="samples/sample.mp4",
        media_path=probe.path,
        probe=probe,
        settings=None,
    )

    result = ProvenanceAnalyzer(watermark_verifier=_StubVerifier(outcome)).analyze(context)

    watermark_check = next(item for item in result.checks if item["name"] == "隐式水印")
    assert watermark_check["status"] == "阳性"
    assert watermark_check["positive"] is True
    assert "SynthID 水印核验服务水印阳性" in result.explicit_ai_indicators


def test_alliance_signature_matches_documented_rule(tmp_path: Path) -> None:
    media = _media(tmp_path)
    captured: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json={"data": {"detected": False}})

    settings = _settings(
        watermark_provider="alliance",
        watermark_api_key=None,
        watermark_app_id="app-7",
        watermark_secret="secret-9",
    )
    _verifier(settings, handler).verify(media)

    body = captured["body"].decode("utf-8", errors="ignore")
    timestamp = body.split('name="timestamp"\r\n\r\n', maxsplit=1)[1].split("\r\n", maxsplit=1)[0]
    nonce = body.split('name="nonce"\r\n\r\n', maxsplit=1)[1].split("\r\n", maxsplit=1)[0]
    signature = body.split('name="sign"\r\n\r\n', maxsplit=1)[1].split("\r\n", maxsplit=1)[0]
    digest = hashlib.sha256(media.read_bytes()).hexdigest()
    expected = hmac.new(
        b"secret-9", f"app-7{timestamp}{nonce}{digest}".encode(), hashlib.sha256
    ).hexdigest()

    assert signature == expected
