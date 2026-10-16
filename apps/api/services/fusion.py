from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from apps.api.detector_plugins.base import SignalResult
from dhfp.calibration import CalibrationProfile

SIGNAL_LAYERS: dict[str, str] = {
    "provenance": "溯源层",
    "render_chain": "溯源层",
    "visual_artifact": "帧内视觉层",
    "digital_human": "数字人特征层",
    "temporal_consistency": "时序层",
    "lip_sync": "音视频一致性层",
    "audio_spoof": "音频层",
    "beauty_filter": "干扰层",
}

COMPETITIVE_SIGNAL_IDS = {
    "visual_artifact",
    "digital_human",
    "temporal_consistency",
    "lip_sync",
    "audio_spoof",
}

# 模态分组只用于“同模态冲突”判定和模态级结论。
# 不同模态指向不同方向是允许的组合，例如音频疑似合成而画面未见异常。
MODALITY_GROUPS: dict[str, tuple[str, ...]] = {
    "visual": ("visual_artifact", "digital_human", "temporal_consistency"),
    "audio": ("audio_spoof",),
    "cross_modal": ("lip_sync",),
}

MODALITY_LABELS: dict[str, str] = {
    "visual": "画面",
    "audio": "音频",
    "cross_modal": "跨模态",
}


class AudioEscalationConfig(BaseModel):
    """音频反欺骗强证据升级规则：多处高置信命中时升级综合判定。"""

    enabled: bool = True
    rule_id: str = "audio_antispoof_strong"
    min_score: float = Field(default=0.95, ge=0, le=1)
    segment_threshold: float = Field(default=0.8, ge=0, le=1)
    min_hit_ratio: float = Field(default=0.6, ge=0, le=1)
    min_segments: int = Field(default=2, ge=1, le=64)


class FusionConfig(BaseModel):
    version: str = "bootstrap-1"
    gray_zone_min: float = Field(default=0.25, ge=0, le=1)
    gray_zone_max: float = Field(default=0.75, ge=0, le=1)
    conflict_spread: float = Field(default=0.52, ge=0, le=1)
    modality_divergence: float = Field(default=0.35, ge=0, le=1)
    signals: dict[str, float] = Field(default_factory=dict)
    layers: dict[str, float] = Field(default_factory=dict)
    unavailable_signal_policy: str = "renormalize"
    audio_escalation: AudioEscalationConfig = Field(default_factory=AudioEscalationConfig)
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path) -> FusionConfig:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


@dataclass
class FusionDecision:
    verdict: str
    calibrated_score: float
    verdict_confidence: float
    summary: str
    conflict: bool
    conflict_note: str | None
    discounted_signals: dict[str, float]
    layer_scores: dict[str, float]
    modality_scores: dict[str, float]
    modality_divergence: list[str]
    abstention_codes: list[str]
    abstention_reasons: list[str]
    gray_zone_min: float
    gray_zone_max: float
    calibration_version: str
    calibration_group: str
    calibration_is_fitted: bool
    calibration_count: int | None
    calibration_negative_count: int | None
    escalation_rule: str | None = None


class FusionEngine:
    def __init__(
        self,
        config: FusionConfig,
        calibration: CalibrationProfile,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self.gray_zone_min, self.gray_zone_max = _effective_gray_zone(config, calibration)

    def decide(
        self,
        signals: list[SignalResult],
        *,
        provenance_positive: bool,
        assessability_status: str,
        beauty_filter_score: float,
    ) -> FusionDecision:
        available = {
            signal.signal_id: signal.score
            for signal in signals
            if signal.available and signal.score is not None
        }
        if provenance_positive:
            summary = "溯源信息命中显式生成合成标识，按合规规则作出疑似生成合成判定。"
            return FusionDecision(
                verdict="疑似生成合成",
                calibrated_score=0.99,
                verdict_confidence=0.99,
                summary=summary,
                conflict=False,
                conflict_note=None,
                discounted_signals={},
                layer_scores={},
                modality_scores={},
                modality_divergence=[],
                abstention_codes=[],
                abstention_reasons=[],
                gray_zone_min=self.gray_zone_min,
                gray_zone_max=self.gray_zone_max,
                calibration_version=self.calibration.version,
                calibration_group=self.calibration_group,
                calibration_is_fitted=self.calibration_is_fitted,
                calibration_count=self.calibration_count,
                calibration_negative_count=self.calibration_negative_count,
            )

        effective = {
            signal_id: score
            for signal_id, score in available.items()
            if signal_id != "beauty_filter"
        }
        discounted: dict[str, float] = {}
        if beauty_filter_score >= 0.45 and "visual_artifact" in effective:
            original = effective["visual_artifact"]
            shrink = float(np.clip(0.30 * beauty_filter_score, 0.0, 1.0))
            effective["visual_artifact"] = float(
                np.clip(original + (0.5 - original) * shrink, 0, 1)
            )
            discounted["visual_artifact"] = effective["visual_artifact"]

        weights = {
            key: max(0.0, self.config.signals.get(key, 0.0))
            for key in effective
            if key in self.config.signals
        }
        layer_scores: dict[str, float] = {}
        if not weights or sum(weights.values()) <= 0:
            raw_score = 0.5
        elif self.config.layers:
            raw_score = self._layered_score(effective, weights, layer_scores)
        else:
            total_weight = sum(weights.values())
            raw_score = (
                sum(effective[key] * weight for key, weight in weights.items()) / total_weight
            )
            layer_scores = self._describe_layers(effective, weights)
        calibrated_score = self.calibration.calibrate(raw_score)

        modality_scores = {}
        for modality, signal_ids in MODALITY_GROUPS.items():
            scores = [effective[signal_id] for signal_id in signal_ids if signal_id in effective]
            if scores:
                modality_scores[modality] = float(np.mean(scores))

        conflict, conflict_note = self._detect_conflict(effective)
        modality_divergence = self._describe_modality_divergence(modality_scores)
        suspect_signal_ids = [
            signal_id
            for signal_id, score in effective.items()
            if signal_id in COMPETITIVE_SIGNAL_IDS and score >= self.gray_zone_max
        ]

        abstention_codes: list[str] = []
        abstention_reasons: list[str] = []
        usable_non_visual = any(
            modality in modality_scores and modality != "visual" for modality in modality_scores
        )
        visual_dominant = self._visual_dominant(modality_scores)

        if not modality_scores:
            verdict = "无法确定"
            summary = "所有检测分支均未获得有效结果，无法给出方向性结论。"
            abstention_codes.append("no_usable_modality")
            abstention_reasons.append("画面、音频、跨模态与溯源分支都没有产生可用证据。")
        elif conflict:
            verdict = "无法确定"
            summary = conflict_note or "同一模态内的信号存在冲突，进入无法确定。"
            abstention_codes.append("evidence_conflict")
            abstention_reasons.append(conflict_note or "同一模态内的信号方向不一致。")
        elif assessability_status == "fail" and not usable_non_visual:
            verdict = "无法确定"
            summary = "质量门控未通过，且没有人脸以外的可用分支，无法给出方向性结论。"
            abstention_codes.append("quality_gate_failed")
            abstention_reasons.append("人脸视觉分支不可用，音频与跨模态分支也没有可用结果。")
        elif (
            assessability_status == "marginal"
            and calibrated_score >= self.gray_zone_max
            and visual_dominant
        ):
            verdict = "无法确定"
            summary = "视觉伪影信号偏高，但媒体质量处于边缘区，需人工复核。"
            abstention_codes.append("quality_marginal")
            abstention_reasons.append("结论主要来自质量处于边缘区的人脸视觉分支。")
        elif calibrated_score < self.gray_zone_min:
            verdict = "可信"
            summary = "未发现足够强的生成合成证据；该结论不等于对内容真实性作保证。"
        elif calibrated_score < self.gray_zone_max:
            verdict = "无法确定"
            summary = "综合分数位于灰区，当前证据不足以给出方向性结论。"
            abstention_codes.append("gray_zone")
            abstention_reasons.append(
                f"融合分数 {calibrated_score:.2f} 位于灰区 "
                f"[{self.gray_zone_min:.2f}, {self.gray_zone_max:.2f}]。"
            )
        elif len(suspect_signal_ids) < 2:
            verdict = "无法确定"
            summary = (
                "当前只有单个信号越过疑似阈值，未达到“多个独立信号互相佐证”的要求。"
            )
            abstention_codes.append("insufficient_corroboration")
            abstention_reasons.append(
                f"仅有 {len(suspect_signal_ids)} 个信号越过疑似阈值，"
                "疑似生成合成判定要求至少两个独立信号在同一时段或同一模态指向生成合成。"
            )
        else:
            verdict = "疑似生成合成"
            summary = "多个可用信号综合后越过疑似阈值，建议进入人工复核。"

        escalation = self._audio_escalation(signals)
        escalation_rule: str | None = None
        if escalation is not None:
            escalation_score, escalation_summary = escalation
            escalation_rule = self.config.audio_escalation.rule_id
            verdict = "疑似生成合成"
            summary = escalation_summary
            # 升级后不再属于拒判，清空拒判原因避免报告自相矛盾。
            abstention_codes = []
            abstention_reasons = []

        if escalation is not None:
            confidence = escalation_score
        elif verdict == "无法确定":
            confidence = float(np.clip(0.5 + abs(calibrated_score - 0.5) * 0.35, 0.5, 0.82))
        else:
            confidence = calibrated_score if verdict == "疑似生成合成" else 1 - calibrated_score
        if assessability_status != "pass" and verdict != "无法确定":
            abstention_reasons.append(
                "质量门控未完全通过，相关分支已按可用性单独降权，整体结论覆盖范围收窄。"
            )
        return FusionDecision(
            verdict=verdict,
            calibrated_score=float(calibrated_score),
            verdict_confidence=float(confidence),
            summary=summary,
            conflict=conflict,
            conflict_note=conflict_note,
            discounted_signals=discounted,
            layer_scores=layer_scores,
            modality_scores=modality_scores,
            modality_divergence=modality_divergence,
            abstention_codes=abstention_codes,
            abstention_reasons=abstention_reasons,
            gray_zone_min=self.gray_zone_min,
            gray_zone_max=self.gray_zone_max,
            calibration_version=self.calibration.version,
            calibration_group=self.calibration_group,
            calibration_is_fitted=self.calibration_is_fitted,
            calibration_count=self.calibration_count,
            calibration_negative_count=self.calibration_negative_count,
            escalation_rule=escalation_rule,
        )

    def _audio_escalation(self, signals: list[SignalResult]) -> tuple[float, str] | None:
        """音频反欺骗多点高置信命中时，按配置规则升级为疑似生成合成。"""
        rule = self.config.audio_escalation
        if not rule.enabled:
            return None
        signal = next((item for item in signals if item.signal_id == "audio_spoof"), None)
        if signal is None or not signal.available or signal.score is None:
            return None
        if signal.score < rule.min_score:
            return None
        segments = signal.raw_value.get("segment_probabilities")
        if not isinstance(segments, list) or len(segments) < rule.min_segments:
            return None
        hits = sum(
            1
            for value in segments
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= rule.segment_threshold
        )
        hit_ratio = hits / len(segments)
        if hit_ratio < rule.min_hit_ratio:
            return None
        summary = (
            f"音频反欺骗模型给出强证据（综合 {signal.score:.2f}，"
            f"{hits}/{len(segments)} 段不低于 {rule.segment_threshold:.2f}），"
            f"按融合策略 [{rule.rule_id}] 升级为疑似生成合成；"
            "该结论仅基于音频分支，建议人工复核并结合原始素材确认。"
        )
        return float(signal.score), summary

    def _detect_conflict(self, effective: dict[str, float]) -> tuple[bool, str | None]:
        """只在同一模态内判定冲突；跨模态方向不同不按冲突降级。"""
        for modality, signal_ids in MODALITY_GROUPS.items():
            scores = [effective[signal_id] for signal_id in signal_ids if signal_id in effective]
            if len(scores) < 2:
                continue
            spread = max(scores) - min(scores)
            if spread >= self.config.conflict_spread:
                return True, (
                    f"{MODALITY_LABELS[modality]}分支内部信号分歧达到 {spread:.2f}，"
                    "同一模态内无法可靠合并为单一方向。"
                )
        return False, None

    def _describe_modality_divergence(self, modality_scores: dict[str, float]) -> list[str]:
        notes: list[str] = []
        keys = sorted(modality_scores)
        for index, left in enumerate(keys):
            for right in keys[index + 1 :]:
                spread = abs(modality_scores[left] - modality_scores[right])
                if spread < self.config.modality_divergence:
                    continue
                notes.append(
                    f"{MODALITY_LABELS.get(left, left)}分支 "
                    f"{modality_scores[left]:.2f} 与 "
                    f"{MODALITY_LABELS.get(right, right)}分支 "
                    f"{modality_scores[right]:.2f} 指向不同方向。"
                    "不同模态对应不同问题，保留各自结论，不按信号冲突降级。"
                )
        return notes

    def _visual_dominant(self, modality_scores: dict[str, float]) -> bool:
        if "visual" not in modality_scores:
            return False
        visual = modality_scores["visual"]
        others = [score for key, score in modality_scores.items() if key != "visual"]
        return visual >= max(others) if others else True

    @property
    def calibration_group(self) -> str:
        if not self.calibration_is_fitted:
            return "uncalibrated_bootstrap"
        return f"temperature:{self.calibration.version}"

    @property
    def calibration_is_fitted(self) -> bool:
        thresholds = self.calibration.thresholds
        return bool(thresholds.calibration_count) and self.calibration.version != "bootstrap-1"

    @property
    def calibration_count(self) -> int | None:
        return self.calibration.thresholds.calibration_count

    @property
    def calibration_negative_count(self) -> int | None:
        return self.calibration.thresholds.real_negative_count

    def _layered_score(
        self,
        effective: dict[str, float],
        weights: dict[str, float],
        layer_scores: dict[str, float],
    ) -> float:
        """先层内加权平均，再按层权重融合，避免同层相关信号重复计权。"""
        layer_numerator: dict[str, float] = {}
        layer_denominator: dict[str, float] = {}
        for signal_id, weight in weights.items():
            if weight <= 0:
                continue
            layer = SIGNAL_LAYERS.get(signal_id, "其他层")
            layer_numerator[layer] = layer_numerator.get(layer, 0.0) + effective[signal_id] * weight
            layer_denominator[layer] = layer_denominator.get(layer, 0.0) + weight
        if not layer_denominator:
            return 0.5
        for layer, denominator in layer_denominator.items():
            layer_scores[layer] = layer_numerator[layer] / denominator

        configured = [max(0.0, weight) for weight in self.config.layers.values()]
        positive = [weight for weight in configured if weight > 0]
        default_weight = float(np.median(positive)) if positive else 1.0
        layer_weights = {
            layer: max(0.0, self.config.layers.get(layer, default_weight)) for layer in layer_scores
        }
        total_weight = sum(layer_weights.values())
        if total_weight <= 0:
            return float(np.mean(list(layer_scores.values())))
        return float(
            sum(layer_scores[layer] * weight for layer, weight in layer_weights.items())
            / total_weight
        )

    @staticmethod
    def _describe_layers(
        effective: dict[str, float],
        weights: dict[str, float],
    ) -> dict[str, float]:
        layer_numerator: dict[str, float] = {}
        layer_denominator: dict[str, float] = {}
        for signal_id, weight in weights.items():
            if weight <= 0:
                continue
            layer = SIGNAL_LAYERS.get(signal_id, "其他层")
            layer_numerator[layer] = layer_numerator.get(layer, 0.0) + effective[signal_id] * weight
            layer_denominator[layer] = layer_denominator.get(layer, 0.0) + weight
        return {
            layer: layer_numerator[layer] / denominator
            for layer, denominator in layer_denominator.items()
        }


def load_fusion_engine(
    fusion_path: str | Path,
    calibration_path: str | Path,
) -> FusionEngine:
    config_payload = json.loads(Path(fusion_path).read_text(encoding="utf-8"))
    return FusionEngine(
        config=FusionConfig.model_validate(config_payload),
        calibration=CalibrationProfile.load(calibration_path),
    )


def _effective_gray_zone(
    config: FusionConfig,
    calibration: CalibrationProfile,
) -> tuple[float, float]:
    """分位数标定阈值优先于 fusion.json 中的静态灰区。"""
    thresholds = calibration.thresholds
    if thresholds.method == "quantile":
        low = float(thresholds.trusted_max)
        high = float(thresholds.suspect_min)
        if 0.0 <= low < high <= 1.0:
            return low, high
    return float(config.gray_zone_min), float(config.gray_zone_max)


def signal_strength(score: float | None, available: bool) -> str:
    if not available or score is None:
        return "弱"
    if score >= 0.75 or score <= 0.25:
        return "强"
    if score >= 0.6 or score <= 0.4:
        return "中"
    return "弱"


def signal_direction(signal: SignalResult) -> str:
    if not signal.available or signal.score is None:
        return "中性"
    if signal.signal_id == "beauty_filter":
        return "中性"
    if signal.score >= 0.58:
        return "指向生成"
    if signal.score <= 0.32:
        return "指向真实"
    return "中性"


def build_limitations(
    *,
    signals: list[SignalResult],
    assessability_reasons: list[str],
) -> list[str]:
    limitations = [
        "本平台提供技术检测证据，不构成司法鉴定结论，也不能证明某一自然人身份。",
        "元数据和可验证凭证未命中不能证明内容真实，标识可能被合法或恶意移除。",
        "当前版本使用内部评测集校准前的 bootstrap 参数，不得直接用于高风险自动化拦截。",
    ]
    unavailable = [signal.display_name for signal in signals if not signal.available]
    if unavailable:
        limitations.append(f"本次未获得有效结果的分支：{'、'.join(unavailable)}。")
    if assessability_reasons:
        limitations.append("质量门控提示：" + "；".join(assessability_reasons))
    return limitations
