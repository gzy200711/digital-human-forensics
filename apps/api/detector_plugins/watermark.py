"""隐式水印核验接入（SynthID / 联盟水印核验服务）。

平台不自带 SynthID 检测算法：Google 的 SynthID Detector 只对有权限的账号
开放，视频与音频目前没有可离线部署的公开检测权重。因此本模块把本地媒体提交
给已授权的外部核验端点，并统一归一化返回结果。

设计约束：

- 默认不把原始媒体送出本机，必须显式设置 ``WATERMARK_ALLOW_MEDIA_UPLOAD=true``；
- 服务不可用、未配置或返回不可解析结果时输出“不可用”，绝不把失败当作阴性；
- 阴性结论只说明本次未检出，不作为内容真实的证据。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import mimetypes
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

WatermarkStatus = Literal["positive", "negative", "not_checked", "unavailable"]

STATUS_LABELS: dict[WatermarkStatus, str] = {
    "positive": "阳性",
    "negative": "阴性",
    "not_checked": "未检查",
    "unavailable": "不可用",
}

DETECTED_KEYS = (
    "detected",
    "watermark_detected",
    "watermarkDetected",
    "is_detected",
    "isDetected",
    "is_watermarked",
    "isWatermarked",
    "has_watermark",
    "hasWatermark",
    "found",
    "hit",
)
SCORE_KEYS = (
    "confidence",
    "score",
    "probability",
    "detection_score",
    "detectionScore",
    "watermark_score",
    "watermarkScore",
    "certainty",
)
IDENTIFIER_KEYS = (
    "watermark_id",
    "watermarkId",
    "payload_id",
    "payloadId",
    "trace_id",
    "traceId",
    "record_id",
    "recordId",
    "identifier",
)
WRAPPER_KEYS = ("data", "result", "payload", "body", "response")
OK_CODES = {0, 200, "0", "200", "ok", "success", "true", "True"}
TRUTHY_VALUES = {"true", "yes", "y", "1", "detected", "positive", "hit"}
FALSY_VALUES = {"false", "no", "n", "0", "not_detected", "negative", "miss", "none"}
MAX_SNIPPET_CHARS = 2000


@dataclass(frozen=True)
class WatermarkOutcome:
    status: WatermarkStatus
    provider_id: str
    provider_label: str
    explanation: str
    detected: bool | None = None
    score: float | None = None
    watermark_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def positive(self) -> bool:
        return self.status == "positive"

    def to_check(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider_id,
            "provider_label": self.provider_label,
        }
        if self.detected is not None:
            payload["detected"] = self.detected
        if self.score is not None:
            payload["score"] = self.score
        if self.watermark_id:
            payload["watermark_id"] = self.watermark_id
        payload.update(self.details)
        return {
            "name": "隐式水印",
            "status": STATUS_LABELS[self.status],
            "positive": self.positive,
            "details": payload,
            "explanation": self.explanation,
        }


@dataclass
class WatermarkVerifier:
    """把媒体提交给外部水印核验服务，并把结果归一化。

    ``provider_id`` 取值：``off`` / ``auto`` / ``synthid`` / ``alliance``。
    """

    provider_id: str = "off"
    provider_label: str = "未配置"
    endpoint: str | None = None
    api_key: str | None = None
    app_id: str | None = None
    secret: str | None = None
    media_field: str = "file"
    extra_fields: dict[str, str] = field(default_factory=dict)
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"
    sign_requests: bool = True
    verify_tls: bool = True
    timeout_seconds: float = 30.0
    allow_media_upload: bool = False
    client: httpx.Client | None = None
    config_error: str | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> WatermarkVerifier:
        kind = str(getattr(settings, "watermark_provider", "off") or "off").strip().lower()
        endpoint = _clean(getattr(settings, "watermark_endpoint", None))
        api_key = _clean(getattr(settings, "watermark_api_key", None))
        app_id = _clean(getattr(settings, "watermark_app_id", None))
        secret = _clean(getattr(settings, "watermark_secret", None))
        label = _clean(getattr(settings, "watermark_provider_label", None))
        common = {
            "endpoint": endpoint,
            "api_key": api_key,
            "app_id": app_id,
            "secret": secret,
            "media_field": str(getattr(settings, "watermark_media_field", "file") or "file"),
            "extra_fields": dict(getattr(settings, "watermark_extra_fields", {}) or {}),
            "auth_header": str(
                getattr(settings, "watermark_auth_header", "Authorization") or "Authorization"
            ),
            "auth_scheme": str(getattr(settings, "watermark_auth_scheme", "Bearer") or ""),
            "sign_requests": bool(getattr(settings, "watermark_sign_requests", True)),
            "verify_tls": bool(getattr(settings, "watermark_verify_tls", True)),
            "timeout_seconds": float(
                getattr(settings, "watermark_timeout_seconds", 30.0) or 30.0
            ),
            "allow_media_upload": bool(
                getattr(settings, "watermark_allow_media_upload", False)
            ),
        }

        if kind in {"off", "none", "disabled", ""}:
            return cls(
                provider_id="off",
                provider_label=label or "已关闭",
                **common,
                config_error="隐式水印核验未启用（WATERMARK_PROVIDER=off）。",
            )

        if kind == "auto":
            if app_id and secret:
                kind = "alliance"
            elif api_key or endpoint:
                kind = "synthid"
            else:
                return cls(
                    provider_id="auto",
                    provider_label=label or "自动选择",
                    **common,
                    config_error=(
                        "WATERMARK_PROVIDER=auto 需要先配置核验服务凭据："
                        "SynthID 填 WATERMARK_ENDPOINT + WATERMARK_API_KEY，"
                        "联盟服务填 WATERMARK_ENDPOINT + WATERMARK_APP_ID + WATERMARK_SECRET。"
                    ),
                )

        if kind == "synthid":
            return cls(
                provider_id="synthid",
                provider_label=label or "SynthID 水印核验服务",
                **common,
            )

        if kind == "alliance":
            missing = [
                name
                for name, value in (
                    ("WATERMARK_ENDPOINT", endpoint),
                    ("WATERMARK_APP_ID", app_id),
                    ("WATERMARK_SECRET", secret),
                )
                if not value
            ]
            error = f"联盟水印核验服务缺少配置项：{', '.join(missing)}。" if missing else None
            return cls(
                provider_id="alliance",
                provider_label=label or "联盟水印核验服务",
                **common,
                config_error=error,
            )

        return cls(
            provider_id=kind,
            provider_label=label or kind,
            **common,
            config_error=(
                f"不支持的 WATERMARK_PROVIDER={kind}；可选值：off、auto、synthid、alliance。"
            ),
        )

    def verify(self, media_path: Path) -> WatermarkOutcome:
        media_path = Path(media_path)
        if self.provider_id == "off":
            return self._outcome(
                "not_checked",
                self.config_error or "隐式水印核验未启用。",
            )
        if self.config_error:
            return self._outcome("unavailable", self.config_error)
        if not self.endpoint:
            return self._outcome(
                "unavailable",
                "未配置 WATERMARK_ENDPOINT，无法调用水印核验服务。",
            )
        if not media_path.exists():
            return self._outcome("unavailable", f"待核验媒体不存在：{media_path}")
        if not self.allow_media_upload:
            return self._outcome(
                "not_checked",
                "已配置水印核验服务，但未开启 WATERMARK_ALLOW_MEDIA_UPLOAD，"
                "因此没有把原始媒体提交给外部服务。开启前请确认素材授权与数据出境要求。",
                details={"endpoint_host": _host(self.endpoint), "consent_required": True},
            )

        try:
            payload, snippet = self._post(media_path)
        except httpx.HTTPError as exc:
            return self._outcome(
                "unavailable",
                f"水印核验服务请求失败：{type(exc).__name__}。",
                details={"endpoint_host": _host(self.endpoint), "error": str(exc)[:300]},
            )

        return self._normalize(payload, snippet)

    def _post(self, media_path: Path) -> tuple[dict[str, Any] | None, str]:
        digest = _sha256(media_path)
        content_type = mimetypes.guess_type(media_path.name)[0] or "application/octet-stream"
        form: dict[str, str] = {key: str(value) for key, value in self.extra_fields.items()}
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.api_key and self.auth_header:
            headers[self.auth_header] = f"{self.auth_scheme} {self.api_key}".strip()

        if self.provider_id == "alliance" and self.sign_requests:
            timestamp = str(int(time.time() * 1000))
            nonce = uuid.uuid4().hex
            form.setdefault("app_id", self.app_id or "")
            form.setdefault("timestamp", timestamp)
            form.setdefault("nonce", nonce)
            if self.secret:
                form.setdefault(
                    "sign",
                    hmac.new(
                        self.secret.encode("utf-8"),
                        f"{self.app_id or ''}{timestamp}{nonce}{digest}".encode(),
                        hashlib.sha256,
                    ).hexdigest(),
                )

        form.setdefault("media_sha256", digest)

        client = self.client
        owns_client = client is None
        if client is None:
            client = httpx.Client(verify=self.verify_tls, timeout=self.timeout_seconds)
        try:
            with media_path.open("rb") as handle:
                response = client.post(
                    self.endpoint or "",
                    data=form,
                    files={self.media_field: (media_path.name, handle, content_type)},
                    headers=headers,
                )
        finally:
            if owns_client:
                client.close()

        snippet = _snippet(response)
        if response.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"核验服务返回 HTTP {response.status_code}",
                request=response.request,
                response=response,
            )
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError):
            payload = None
        return (payload if isinstance(payload, dict) else None), snippet

    def _normalize(self, payload: dict[str, Any] | None, snippet: str) -> WatermarkOutcome:
        details: dict[str, Any] = {
            "endpoint_host": _host(self.endpoint),
            "media_sha256": None,
        }
        if payload is None:
            return self._outcome(
                "unavailable",
                "水印核验服务返回的内容不是可解析的 JSON 对象，无法给出结论。",
                details={**details, "response_snippet": snippet},
            )

        service_error = _service_error(payload)
        if service_error:
            return self._outcome(
                "unavailable",
                f"水印核验服务返回失败状态：{service_error}",
                details={**details, "response": payload},
            )

        body = _unwrap(payload)
        detected = _coerce_bool(_find_value(body, DETECTED_KEYS))
        score = _coerce_float(_find_value(body, SCORE_KEYS))
        identifier = _coerce_str(_find_value(body, IDENTIFIER_KEYS))

        if detected is None and score is not None:
            detected = score >= 0.5
        if detected is None:
            return self._outcome(
                "unavailable",
                "水印核验服务返回成功，但响应中没有可识别的检出结论字段，"
                "请按文档配置 WATERMARK_EXTRA_FIELDS 或适配响应字段。",
                details={**details, "response": payload},
            )

        details["score"] = score
        details["watermark_id"] = identifier
        details["response"] = payload
        if detected:
            return self._outcome(
                "positive",
                f"{self.provider_label}检出隐式水印"
                + (f"（标识：{identifier}）" if identifier else "")
                + "，作为显式生成合成标识参与判定。",
                detected=True,
                score=score,
                watermark_id=identifier,
                details=details,
            )
        return self._outcome(
            "negative",
            f"{self.provider_label}未检出可识别水印；该结果不证明内容真实，"
            "水印也可能被裁剪、重编码或破坏。",
            detected=False,
            score=score,
            details=details,
        )

    def _outcome(
        self,
        status: WatermarkStatus,
        explanation: str,
        *,
        detected: bool | None = None,
        score: float | None = None,
        watermark_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> WatermarkOutcome:
        return WatermarkOutcome(
            status=status,
            provider_id=self.provider_id,
            provider_label=self.provider_label,
            explanation=explanation,
            detected=detected,
            score=score,
            watermark_id=watermark_id,
            details=details or {},
        )


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _host(endpoint: str | None) -> str | None:
    if not endpoint:
        return None
    return urlsplit(endpoint).netloc or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snippet(response: httpx.Response) -> str:
    try:
        return response.text[:MAX_SNIPPET_CHARS]
    except Exception:  # pragma: no cover - 仅在响应体不可解码时触发
        return ""


def _service_error(payload: dict[str, Any]) -> str | None:
    success = payload.get("success")
    if isinstance(success, bool) and not success:
        return str(payload.get("message") or payload.get("msg") or "success=false")
    if "code" in payload:
        code = payload.get("code")
        if code not in OK_CODES:
            return f"code={code} {payload.get('message') or payload.get('msg') or ''}".strip()
    status = payload.get("status")
    if isinstance(status, str) and status.strip().lower() in {"error", "failed", "failure"}:
        return f"status={status}"
    return None


def _unwrap(payload: dict[str, Any]) -> dict[str, Any]:
    current = payload
    for _ in range(3):
        for key in WRAPPER_KEYS:
            value = current.get(key)
            if isinstance(value, dict):
                current = value
                break
        else:
            return current
    return current


def _find_value(payload: Any, keys: tuple[str, ...], depth: int = 4) -> Any:
    if depth < 0 or not isinstance(payload, dict):
        return None
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    for value in payload.values():
        if isinstance(value, dict):
            found = _find_value(value, keys, depth - 1)
            if found is not None:
                return found
        elif isinstance(value, list):
            for item in value:
                found = _find_value(item, keys, depth - 1)
                if found is not None:
                    return found
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUTHY_VALUES:
            return True
        if normalized in FALSY_VALUES:
            return False
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _coerce_str(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    text = str(value).strip()
    return text or None
