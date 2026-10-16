from __future__ import annotations

import pytest

from apps.api.detector_plugins.base import SignalResult
from apps.api.services.fusion import load_fusion_engine


def test_beauty_filter_is_not_fused_as_generation_risk() -> None:
    engine = load_fusion_engine("configs/fusion.json", "configs/calibration.json")
    base_signals = [
        SignalResult(
            signal_id="visual_artifact",
            display_name="帧级视觉伪影",
            score=0.2,
            available=True,
            direction="中性",
        ),
        SignalResult(
            signal_id="temporal_consistency",
            display_name="时序一致性",
            score=0.3,
            available=True,
            direction="中性",
        ),
    ]
    beauty_signal = SignalResult(
        signal_id="beauty_filter",
        display_name="美颜 / 滤镜干扰",
        score=1.0,
        available=True,
        direction="中性",
    )

    baseline = engine.decide(
        base_signals,
        provenance_positive=False,
        assessability_status="pass",
        beauty_filter_score=1.0,
    )
    with_beauty = engine.decide(
        [*base_signals, beauty_signal],
        provenance_positive=False,
        assessability_status="pass",
        beauty_filter_score=1.0,
    )

    assert with_beauty.calibrated_score == baseline.calibrated_score


def test_beauty_filter_shrinks_visual_score_toward_half() -> None:
    engine = load_fusion_engine("configs/fusion.json", "configs/calibration.json")
    signals = [
        SignalResult(
            signal_id="visual_artifact",
            display_name="帧级视觉伪影",
            score=0.95,
            available=True,
            direction="中性",
        ),
        SignalResult(
            signal_id="temporal_consistency",
            display_name="时序一致性",
            score=0.4,
            available=True,
            direction="中性",
        ),
    ]

    decision = engine.decide(
        signals,
        provenance_positive=False,
        assessability_status="pass",
        beauty_filter_score=1.0,
    )
    adjusted = decision.discounted_signals["visual_artifact"]
    assert 0.5 < adjusted < 0.95

    low_signal = [
        SignalResult(
            signal_id="visual_artifact",
            display_name="帧级视觉伪影",
            score=0.1,
            available=True,
            direction="中性",
        )
    ]
    low_decision = engine.decide(
        low_signal,
        provenance_positive=False,
        assessability_status="pass",
        beauty_filter_score=1.0,
    )
    # 拉向 0.5 而不是 0：低分不会被美颜干扰进一步推向“真实”。
    assert low_decision.discounted_signals["visual_artifact"] > 0.1


def test_layered_fusion_reports_layer_scores() -> None:
    engine = load_fusion_engine("configs/fusion.json", "configs/calibration.json")
    signals = [
        SignalResult("render_chain", "渲染链路指纹", 0.68, True, "指向生成"),
        SignalResult("visual_artifact", "帧级视觉伪影", 0.8, True, "指向生成"),
        SignalResult("digital_human", "数字人专属特征", 0.72, True, "指向生成"),
        SignalResult("temporal_consistency", "时序一致性", 0.7, True, "指向生成"),
        SignalResult("lip_sync", "唇音同步", 0.5, True, "中性"),
        SignalResult("audio_spoof", "音频反欺骗", 0.9, True, "指向生成"),
    ]

    decision = engine.decide(
        signals,
        provenance_positive=False,
        assessability_status="pass",
        beauty_filter_score=0.0,
    )

    assert decision.layer_scores["音频层"] == pytest.approx(0.9)
    assert decision.layer_scores["数字人特征层"] == pytest.approx(0.72)
    assert decision.calibrated_score > 0.5


def test_cross_modality_divergence_is_not_a_conflict() -> None:
    """音频疑似合成而画面未见异常，是成立的组合，不应按冲突降级。"""
    engine = load_fusion_engine("configs/fusion.json", "configs/calibration.json")
    signals = [
        SignalResult("visual_artifact", "帧级视觉伪影", 0.95, True, "指向生成"),
        SignalResult("temporal_consistency", "时序一致性", 0.92, True, "指向生成"),
        SignalResult("audio_spoof", "音频反欺骗", 0.05, True, "中性"),
    ]

    decision = engine.decide(
        signals,
        provenance_positive=False,
        assessability_status="pass",
        beauty_filter_score=0.0,
    )

    assert decision.conflict is False
    assert decision.modality_divergence
    assert "evidence_conflict" not in decision.abstention_codes


def test_face_failure_does_not_discard_the_audio_branch() -> None:
    """人脸分支不可用时，音频分支仍要独立评估，而不是被质量门控直接丢弃。"""
    engine = load_fusion_engine("configs/fusion.json", "configs/calibration.json")
    signals = [
        SignalResult("audio_spoof", "音频反欺骗", 0.95, True, "指向生成"),
    ]

    decision = engine.decide(
        signals,
        provenance_positive=False,
        assessability_status="fail",
        beauty_filter_score=0.0,
    )

    # 单个信号不足以支撑“疑似生成合成”，但拒判原因必须来自证据不足，
    # 而不是“人脸分支不可用”这一条与音频无关的质量门控。
    assert decision.verdict == "无法确定"
    assert "quality_gate_failed" not in decision.abstention_codes
    assert "insufficient_corroboration" in decision.abstention_codes


def test_multiple_independent_signals_can_reach_suspected() -> None:
    engine = load_fusion_engine("configs/fusion.json", "configs/calibration.json")
    signals = [
        SignalResult("visual_artifact", "帧级视觉伪影", 0.9, True, "指向生成"),
        SignalResult("audio_spoof", "音频反欺骗", 0.9, True, "指向生成"),
    ]

    decision = engine.decide(
        signals,
        provenance_positive=False,
        assessability_status="pass",
        beauty_filter_score=0.0,
    )

    assert decision.verdict == "疑似生成合成"
    assert decision.abstention_codes == []


def test_gray_zone_is_reported_before_corroboration() -> None:
    """灰区分数的拒判原因应当是灰区，而不是“越过阈值的信号不足”。"""
    engine = load_fusion_engine("configs/fusion.json", "configs/calibration.json")
    signals = [
        SignalResult("visual_artifact", "帧级视觉伪影", 0.45, True, "中性"),
        SignalResult("audio_spoof", "音频反欺骗", 0.45, True, "中性"),
    ]

    decision = engine.decide(
        signals,
        provenance_positive=False,
        assessability_status="pass",
        beauty_filter_score=0.0,
    )

    assert decision.verdict == "无法确定"
    assert decision.abstention_codes == ["gray_zone"]


def _audio_spoof(score: float, segments: list[float]) -> SignalResult:
    return SignalResult(
        signal_id="audio_spoof",
        display_name="音频反欺骗",
        score=score,
        available=True,
        direction="指向生成",
        raw_value={"segment_probabilities": segments},
    )


def test_strong_audio_evidence_escalates_even_with_visual_conflict() -> None:
    """画面内部冲突时，音频反欺骗多点强命中应按规则升级为疑似生成合成。"""
    engine = load_fusion_engine("configs/fusion.json", "configs/calibration.json")
    signals = [
        SignalResult("visual_artifact", "帧级视觉伪影", 0.07, True, "中性"),
        SignalResult("digital_human", "数字人专属特征", 0.90, True, "中性"),
        SignalResult("temporal_consistency", "时序一致性", 0.19, True, "中性"),
        _audio_spoof(
            0.999,
            [0.9999, 0.0181, 0.9912, 0.8443, 0.8637, 0.8822, 0.9856, 0.6457],
        ),
    ]

    decision = engine.decide(
        signals,
        provenance_positive=False,
        assessability_status="pass",
        beauty_filter_score=0.0,
    )

    assert decision.conflict is True
    assert decision.verdict == "疑似生成合成"
    assert decision.escalation_rule == "audio_antispoof_strong"
    assert decision.verdict_confidence == pytest.approx(0.999)
    assert decision.abstention_codes == []
    assert "音频" in decision.summary


def test_audio_escalation_requires_segment_majority() -> None:
    """整体分数再高，分段命中比例不足时不得升级。"""
    engine = load_fusion_engine("configs/fusion.json", "configs/calibration.json")
    signals = [
        SignalResult("visual_artifact", "帧级视觉伪影", 0.4, True, "中性"),
        _audio_spoof(0.99, [0.99, 0.1, 0.1, 0.1]),
    ]

    decision = engine.decide(
        signals,
        provenance_positive=False,
        assessability_status="pass",
        beauty_filter_score=0.0,
    )

    assert decision.verdict == "无法确定"
    assert decision.escalation_rule is None


def test_audio_escalation_can_be_disabled_by_config() -> None:
    engine = load_fusion_engine("configs/fusion.json", "configs/calibration.json")
    engine.config.audio_escalation.enabled = False
    signals = [
        SignalResult("visual_artifact", "帧级视觉伪影", 0.07, True, "中性"),
        _audio_spoof(0.999, [0.99, 0.99, 0.99, 0.99]),
    ]

    decision = engine.decide(
        signals,
        provenance_positive=False,
        assessability_status="pass",
        beauty_filter_score=0.0,
    )

    assert decision.verdict == "无法确定"
    assert decision.escalation_rule is None
