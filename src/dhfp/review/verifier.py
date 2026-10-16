from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from dhfp.domain import EvidenceItem, EvidenceReview, ReviewStatement


@dataclass(frozen=True)
class ReviewRequest:
    calibrated_score: float
    assessability_status: str
    evidence: tuple[EvidenceItem, ...]


@dataclass(frozen=True)
class ProviderStatement:
    claim: str
    evidence_ids: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class ProviderReview:
    statements: tuple[ProviderStatement, ...]
    provider: str
    model_version: str
    license: str | None = None
    note: str = ""


class EvidenceReviewProvider(Protocol):
    name: str
    model_version: str
    license: str | None

    def review(self, request: ReviewRequest) -> ProviderReview: ...


class RulesEvidenceProvider:
    """Evidence-only fallback used when no external L6 model is configured."""

    name = "rules-evidence-summary"
    model_version = "bootstrap-1"
    license = "internal"

    def review(self, request: ReviewRequest) -> ProviderReview:
        statements: list[ProviderStatement] = []
        for item in request.evidence:
            if item.signal not in {
                "visual_artifact",
                "temporal_consistency",
                "lip_sync",
                "audio_spoof",
            }:
                continue
            if not item.available or item.score is None:
                continue
            if item.score >= 0.58:
                statements.append(
                    ProviderStatement(
                        claim=(
                            f"{item.display_name}分数为 {item.score:.3f}，"
                            "该分支已达到当前生成风险提示区间。"
                        ),
                        evidence_ids=(item.id,),
                        kind="生成风险",
                    )
                )
            elif item.score <= 0.32:
                statements.append(
                    ProviderStatement(
                        claim=(
                            f"{item.display_name}分数为 {item.score:.3f}，"
                            "该分支低于当前生成风险提示区间。"
                        ),
                        evidence_ids=(item.id,),
                        kind="真实性提示",
                    )
                )
            else:
                statements.append(
                    ProviderStatement(
                        claim=(
                            f"{item.display_name}分数为 {item.score:.3f}，"
                            "位于不确定区间，不能单独形成方向性结论。"
                        ),
                        evidence_ids=(item.id,),
                        kind="不确定性",
                    )
                )

        quality = next(
            (item for item in request.evidence if item.signal == "quality_gate"),
            None,
        )
        if quality is not None and request.assessability_status != "pass":
            statements.append(
                ProviderStatement(
                    claim="质量门控未完全通过，视觉和时序证据的适用范围受到限制。",
                    evidence_ids=(quality.id,),
                    kind="质量限制",
                )
            )

        provenance = next(
            (item for item in request.evidence if item.signal == "provenance"),
            None,
        )
        if provenance is not None and provenance.available:
            statements.append(
                ProviderStatement(
                    claim="溯源分支未命中显式生成标识；该结果不能证明内容真实。",
                    evidence_ids=(provenance.id,),
                    kind="不确定性",
                )
            )

        return ProviderReview(
            statements=tuple(statements[:8]),
            provider=self.name,
            model_version=self.model_version,
            license=self.license,
            note="规则摘要只复述已存证据，不新增事实或身份判断。",
        )


class EvidenceReviewEngine:
    def __init__(
        self,
        *,
        provider: EvidenceReviewProvider | None = None,
        gray_zone_min: float = 0.25,
        gray_zone_max: float = 0.75,
    ) -> None:
        if not 0 <= gray_zone_min < gray_zone_max <= 1:
            raise ValueError("复核灰区阈值无效")
        self.provider = provider or RulesEvidenceProvider()
        self.gray_zone_min = gray_zone_min
        self.gray_zone_max = gray_zone_max

    def review(
        self,
        *,
        calibrated_score: float,
        assessability_status: str,
        evidence: list[EvidenceItem],
    ) -> EvidenceReview:
        if not self.gray_zone_min <= calibrated_score <= self.gray_zone_max:
            return EvidenceReview(
                status="not_triggered",
                triggered=False,
                provider="none",
                summary="校准分数不在灰区，未启动证据复核。",
                note="证据复核只在融合分数位于灰区时运行。",
            )

        provider_name = getattr(self.provider, "name", "unknown")
        model_version = getattr(self.provider, "model_version", None)
        license_name = getattr(self.provider, "license", None)
        if assessability_status == "fail":
            return EvidenceReview(
                status="unavailable",
                triggered=True,
                provider=provider_name,
                model_version=model_version,
                license=license_name,
                summary="质量门控失败，未生成方向性解释。",
                note="低质量媒体不应通过自由文本解释补足缺失的检测证据。",
            )

        request = ReviewRequest(
            calibrated_score=calibrated_score,
            assessability_status=assessability_status,
            evidence=tuple(evidence),
        )
        try:
            provider_review = self.provider.review(request)
        except Exception as exc:  # noqa: BLE001
            return EvidenceReview(
                status="unavailable",
                triggered=True,
                provider=provider_name,
                model_version=model_version,
                license=license_name,
                summary="证据复核提供器执行失败，未采用未经验证的文本。",
                note=f"提供器错误: {type(exc).__name__}",
            )

        evidence_by_id = {item.id: item for item in evidence}
        accepted: list[ReviewStatement] = []
        rejected = 0
        for candidate in provider_review.statements:
            statement = _validate_statement(candidate, evidence_by_id)
            if statement is None:
                rejected += 1
            else:
                accepted.append(statement)

        if not accepted:
            return EvidenceReview(
                status="unavailable",
                triggered=True,
                provider=provider_review.provider,
                model_version=provider_review.model_version,
                license=provider_review.license,
                summary="复核输出没有任何可回溯到现有证据的陈述。",
                rejected_claim_count=rejected,
                note=provider_review.note or "全部复核陈述均被证据约束校验拒绝。",
            )

        return EvidenceReview(
            status="completed",
            triggered=True,
            provider=provider_review.provider,
            model_version=provider_review.model_version,
            license=provider_review.license,
            summary=f"已完成灰区证据复核，保留 {len(accepted)} 条可回溯陈述。",
            statements=accepted,
            rejected_claim_count=rejected,
            note=provider_review.note or "所有保留陈述均可回溯到报告中的证据 ID。",
        )


def _validate_statement(
    candidate: ProviderStatement,
    evidence_by_id: dict[str, EvidenceItem],
) -> ReviewStatement | None:
    claim = candidate.claim.strip()
    if not claim or len(claim) > 500:
        return None
    evidence_ids = list(dict.fromkeys(candidate.evidence_ids))
    if not evidence_ids or len(evidence_ids) > 8:
        return None
    cited = [evidence_by_id.get(evidence_id) for evidence_id in evidence_ids]
    if any(item is None for item in cited):
        return None
    concrete = [item for item in cited if item is not None]
    if not _kind_is_supported(candidate.kind, concrete):
        return None
    return ReviewStatement(
        claim=claim,
        evidence_ids=evidence_ids,
        kind=candidate.kind,  # type: ignore[arg-type]
    )


def _kind_is_supported(kind: str, evidence: list[EvidenceItem]) -> bool:
    if kind == "生成风险":
        return any(item.direction == "指向生成" for item in evidence)
    if kind == "真实性提示":
        return any(item.direction == "指向真实" for item in evidence)
    if kind == "质量限制":
        return any(item.signal == "quality_gate" for item in evidence)
    if kind == "不确定性":
        return any(item.direction == "中性" for item in evidence)
    return False
