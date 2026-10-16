from __future__ import annotations

import json

import httpx

from dhfp.domain import EvidenceItem
from dhfp.review.http_provider import (
    HttpEvidenceReviewProvider,
    UnavailableEvidenceProvider,
)
from dhfp.review.verifier import EvidenceReviewEngine


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id="visual_artifact",
            signal="visual_artifact",
            display_name="帧级视觉伪影",
            direction="指向生成",
            strength="中",
            score=0.7,
            raw_value={"private_feature": 123.456},
            explanation="测试证据",
        )
    ]


def test_http_provider_parses_evidence_bound_response() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "review-model-2026",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "statements": [
                                        {
                                            "claim": "视觉伪影达到风险提示区间。",
                                            "evidence_ids": ["visual_artifact"],
                                            "kind": "生成风险",
                                        }
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = HttpEvidenceReviewProvider(
        endpoint="https://review.example/v1/chat/completions",
        model="review-model",
        api_key="secret",
        client=client,
    )
    review = EvidenceReviewEngine(provider=provider).review(
        calibrated_score=0.5,
        assessability_status="pass",
        evidence=_evidence(),
    )

    assert review.status == "completed"
    assert review.provider == "openai-compatible-mllm"
    assert review.model_version == "review-model-2026"
    assert review.statements[0].evidence_ids == ["visual_artifact"]
    assert captured["headers"]["authorization"] == "Bearer secret"
    sent_evidence = captured["payload"]["messages"][1]["content"]
    assert "private_feature" not in sent_evidence
    assert "raw_value" not in sent_evidence


def test_http_provider_errors_are_not_turned_into_claims() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "not-json"}}]},
        )

    provider = HttpEvidenceReviewProvider(
        endpoint="https://review.example/v1/chat/completions",
        model="review-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    review = EvidenceReviewEngine(provider=provider).review(
        calibrated_score=0.5,
        assessability_status="pass",
        evidence=_evidence(),
    )

    assert review.status == "unavailable"
    assert not review.statements


def test_unconfigured_provider_fails_closed() -> None:
    provider = UnavailableEvidenceProvider("未配置外部复核服务")
    review = EvidenceReviewEngine(provider=provider).review(
        calibrated_score=0.5,
        assessability_status="pass",
        evidence=_evidence(),
    )

    assert review.status == "unavailable"
    assert review.provider == "unconfigured-mllm"
    assert "EvidenceReviewProviderError" in review.note
