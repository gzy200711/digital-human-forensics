from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

import httpx

from dhfp.review.verifier import (
    ProviderReview,
    ProviderStatement,
    ReviewRequest,
)

SYSTEM_PROMPT = """你是数字合成内容鉴别报告的证据复核器。
只能引用用户提供的 evidence_ids。不得补充外部事实、身份判断、司法结论或确定性指控。
输出严格 JSON：
{"statements":[{"claim":"...","evidence_ids":["..."],"kind":"生成风险|真实性提示|不确定性|质量限制"}]}
没有可回溯证据时返回 {"statements":[]}。"""


class EvidenceReviewProviderError(RuntimeError):
    pass


class UnavailableEvidenceProvider:
    name = "unconfigured-mllm"
    model_version: str | None = None
    license: str | None = None

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def review(self, _request: ReviewRequest) -> ProviderReview:
        raise EvidenceReviewProviderError(self.reason)


class HttpEvidenceReviewProvider:
    name = "openai-compatible-mllm"

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
        prompt_version: str = "evidence-review-v1",
        license_name: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("复核 provider endpoint 必须是有效的 http/https URL")
        if not model.strip():
            raise ValueError("复核 provider model 不能为空")
        if timeout_seconds <= 0:
            raise ValueError("复核 provider timeout 必须大于 0")
        self.endpoint = endpoint
        self.model = model
        self.model_version = model
        self.api_key = api_key
        self.prompt_version = prompt_version
        self.license = license_name
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def review(self, request: ReviewRequest) -> ProviderReview:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        _review_payload(request),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
        }
        try:
            response = self._client.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EvidenceReviewProviderError(f"复核 provider 请求失败: {exc}") from exc

        try:
            response_payload = response.json()
        except ValueError as exc:
            raise EvidenceReviewProviderError("复核 provider 返回的不是 JSON") from exc
        content = _extract_message_content(response_payload)
        review_payload = _parse_json_content(content)
        raw_statements = review_payload.get("statements")
        if not isinstance(raw_statements, list):
            raise EvidenceReviewProviderError("复核 provider 缺少 statements 数组")

        statements: list[ProviderStatement] = []
        for item in raw_statements[:8]:
            if not isinstance(item, dict):
                raise EvidenceReviewProviderError("复核 statement 必须是对象")
            claim = item.get("claim")
            evidence_ids = item.get("evidence_ids")
            kind = item.get("kind")
            if not isinstance(claim, str) or not isinstance(kind, str):
                raise EvidenceReviewProviderError("复核 statement 缺少 claim 或 kind")
            if not isinstance(evidence_ids, list) or not all(
                isinstance(value, str) for value in evidence_ids
            ):
                raise EvidenceReviewProviderError("复核 statement 的 evidence_ids 无效")
            statements.append(
                ProviderStatement(
                    claim=claim,
                    evidence_ids=tuple(evidence_ids),
                    kind=kind,
                )
            )

        response_model = response_payload.get("model")
        return ProviderReview(
            statements=tuple(statements),
            provider=self.name,
            model_version=response_model if isinstance(response_model, str) else self.model,
            license=self.license,
            note=f"prompt={self.prompt_version}；输出已进入证据约束校验。",
        )


def _review_payload(request: ReviewRequest) -> dict[str, Any]:
    return {
        "calibrated_score": round(request.calibrated_score, 6),
        "assessability_status": request.assessability_status,
        "evidence": [
            {
                "id": item.id,
                "signal": item.signal,
                "display_name": item.display_name,
                "direction": item.direction,
                "strength": item.strength,
                "score": item.score,
                "available": item.available,
            }
            for item in request.evidence
        ],
    }


def _extract_message_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise EvidenceReviewProviderError("复核 provider 响应缺少 choices[0].message.content") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ]
        if texts:
            return "\n".join(str(text) for text in texts)
    raise EvidenceReviewProviderError("复核 provider 消息内容不是文本")


def _parse_json_content(content: str) -> dict[str, Any]:
    normalized = content.strip()
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        normalized = "\n".join(lines).strip()
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise EvidenceReviewProviderError("复核 provider 消息不是有效 JSON") from exc
    if not isinstance(payload, dict):
        raise EvidenceReviewProviderError("复核 provider JSON 顶层必须是对象")
    return payload
