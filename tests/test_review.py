from __future__ import annotations

from dhfp.domain import EvidenceItem
from dhfp.review.verifier import (
    EvidenceReviewEngine,
    ProviderReview,
    ProviderStatement,
)


class StubProvider:
    name = "stub-reviewer"
    model_version = "test-1"
    license = "internal"

    def __init__(self, statements: tuple[ProviderStatement, ...]) -> None:
        self.statements = statements
        self.calls = 0

    def review(self, _request) -> ProviderReview:
        self.calls += 1
        return ProviderReview(
            statements=self.statements,
            provider=self.name,
            model_version=self.model_version,
            license=self.license,
            note="stub",
        )


class FailingProvider:
    name = "failing-reviewer"
    model_version = "test-1"
    license = "internal"

    def review(self, _request) -> ProviderReview:
        raise RuntimeError("provider unavailable")


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            id="visual_artifact",
            signal="visual_artifact",
            display_name="帧级视觉伪影",
            direction="指向生成",
            strength="中",
            score=0.72,
            explanation="测试证据",
        ),
        EvidenceItem(
            id="quality-gate",
            signal="quality_gate",
            display_name="可鉴定性门控",
            direction="中性",
            strength="中",
            score=0.58,
            explanation="测试质量门控",
        ),
    ]


def test_review_only_runs_in_gray_zone() -> None:
    provider = StubProvider(())
    engine = EvidenceReviewEngine(provider=provider)

    review = engine.review(
        calibrated_score=0.91,
        assessability_status="pass",
        evidence=_evidence(),
    )

    assert review.status == "not_triggered"
    assert review.triggered is False
    assert provider.calls == 0


def test_reviewer_rejects_unbound_and_mismatched_claims() -> None:
    provider = StubProvider(
        (
            ProviderStatement(
                claim="有效风险陈述。",
                evidence_ids=("visual_artifact",),
                kind="生成风险",
            ),
            ProviderStatement(
                claim="引用不存在证据的陈述。",
                evidence_ids=("missing",),
                kind="生成风险",
            ),
            ProviderStatement(
                claim="证据类型不匹配的陈述。",
                evidence_ids=("quality-gate",),
                kind="生成风险",
            ),
        )
    )
    engine = EvidenceReviewEngine(provider=provider)

    review = engine.review(
        calibrated_score=0.55,
        assessability_status="pass",
        evidence=_evidence(),
    )

    assert review.status == "completed"
    assert review.triggered is True
    assert len(review.statements) == 1
    assert review.statements[0].evidence_ids == ["visual_artifact"]
    assert review.rejected_claim_count == 2


def test_review_unavailable_when_all_claims_are_rejected() -> None:
    provider = StubProvider(
        (
            ProviderStatement(
                claim="无证据陈述。",
                evidence_ids=("missing",),
                kind="生成风险",
            ),
        )
    )
    engine = EvidenceReviewEngine(provider=provider)

    review = engine.review(
        calibrated_score=0.5,
        assessability_status="marginal",
        evidence=_evidence(),
    )

    assert review.status == "unavailable"
    assert review.rejected_claim_count == 1
    assert not review.statements


def test_rules_provider_returns_only_cited_claims() -> None:
    engine = EvidenceReviewEngine()

    review = engine.review(
        calibrated_score=0.5,
        assessability_status="marginal",
        evidence=_evidence(),
    )

    assert review.status == "completed"
    assert review.statements
    known_ids = {item.id for item in _evidence()}
    assert all(set(statement.evidence_ids) <= known_ids for statement in review.statements)


def test_review_skips_failed_quality_and_handles_provider_errors() -> None:
    skipped = EvidenceReviewEngine(provider=FailingProvider()).review(
        calibrated_score=0.5,
        assessability_status="fail",
        evidence=_evidence(),
    )
    assert skipped.status == "unavailable"
    assert "质量门控失败" in skipped.summary

    failed = EvidenceReviewEngine(provider=FailingProvider()).review(
        calibrated_score=0.5,
        assessability_status="pass",
        evidence=_evidence(),
    )
    assert failed.status == "unavailable"
    assert "执行失败" in failed.summary
