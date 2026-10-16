from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from jinja2 import BaseLoader, Environment, select_autoescape

from apps.api.detector_plugins.base import PluginContext, ProvenanceResult, SignalResult
from apps.api.detector_plugins.face_quality import FaceQualityResult
from apps.api.services.fusion import (
    MODALITY_GROUPS,
    MODALITY_LABELS,
    SIGNAL_LAYERS,
    FusionDecision,
    build_limitations,
    signal_direction,
    signal_strength,
)
from apps.api.services.media import sha256_file
from dhfp.domain import (
    ArtifactLinks,
    Assessability,
    AssessabilityFactor,
    CalibratedProbability,
    EvidenceItem,
    EvidenceModel,
    EvidenceRegion,
    ForensicReport,
    GrayZone,
    MediaInfo,
    ModalityCoverage,
    ModalityVerdict,
    ModelSnapshot,
    ModelVersion,
    Provenance,
    ProvenanceCheck,
    ThresholdPolicy,
    TimeRange,
    Verdict,
)
from dhfp.review.verifier import EvidenceReviewEngine

# FPR=0.1% 的统计门槛：真实阴性样本少于该数量时不能宣称已验证。
MIN_REAL_NEGATIVES_FOR_FPR_0_1 = 2995


def build_report(
    *,
    job_id: str,
    context: PluginContext,
    face_quality: FaceQualityResult,
    provenance: ProvenanceResult,
    signals: list[SignalResult],
    decision: FusionDecision,
    duration_seconds: float | None,
    artifacts: ArtifactLinks,
    review_engine: EvidenceReviewEngine | None = None,
    fusion_version: str = "unknown",
    calibration_version: str = "unknown",
) -> ForensicReport:
    status_map = {
        "pass": "可鉴定",
        "marginal": "勉强可鉴定",
        "fail": "不可鉴定",
    }
    assessability = Assessability(
        score=face_quality.score,
        status=status_map[face_quality.status],
        face_detected_ratio=face_quality.face_detected_ratio,
        factors=[AssessabilityFactor.model_validate(item) for item in face_quality.factors],
        reasons=face_quality.reasons,
    )
    coverage = _build_coverage(
        signals=signals,
        face_quality=face_quality,
        duration_seconds=duration_seconds,
        has_audio=context.probe.has_audio,
    )
    evidence = [
        _signal_to_evidence(signal)
        for signal in signals
        if signal.signal_id != "provenance" or signal.available
    ]
    evidence.append(
        EvidenceItem(
            id="quality-gate",
            signal="quality_gate",
            display_name="可鉴定性门控",
            layer="质量门控层",
            direction="中性",
            strength={
                "pass": "强",
                "marginal": "中",
                "fail": "弱",
            }[face_quality.status],
            score=face_quality.score,
            raw_value={
                "status": face_quality.status,
                "sampled_frames": face_quality.sampled_frames,
                "face_detected_ratio": face_quality.face_detected_ratio,
            },
            source="face-quality-gate",
            unit="score",
            model=EvidenceModel(
                name="haar-face-quality",
                version="bootstrap-1",
                license="Apache-2.0 (OpenCV)",
            ),
            uncertainty="仅覆盖被采样帧，未对未采样区间给出置信区间。",
            explanation=(
                "质量门控通过，视觉信号可信度较高。"
                if face_quality.status == "pass"
                else "质量门控未完全通过；相关结论已降级或建议人工复核。"
            ),
            available=True,
        )
    )
    time_ranges = _collect_time_ranges(signals)
    review = (
        review_engine.review(
            calibrated_score=decision.calibrated_score,
            assessability_status=face_quality.status,
            evidence=evidence,
        )
        if review_engine is not None
        else None
    )
    limitations = build_limitations(
        signals=signals,
        assessability_reasons=face_quality.reasons,
    )
    limitations.append(
        "视觉、时序与数字人特征分数含启发式特征，生产环境应替换为经过许可证与数据审计的模型。"
    )
    audio_spoof = next(
        (signal for signal in signals if signal.signal_id == "audio_spoof" and signal.available),
        None,
    )
    if audio_spoof is not None and audio_spoof.raw_value.get("model_path"):
        limitations.append(
            "音频反欺骗使用外部 ONNX 权重，跨语种、跨编解码和音乐场景必须用内部数据重新评测。"
        )
    else:
        limitations.append("音频反欺骗当前为轻量启发式基线，不覆盖音乐、多人交谈和强后期处理。")
    limitations.append("唇音同步为包络相关基线，不能替代 SyncNet/AuViRe 等专用模型。")
    recommended_action = (
        "拦截并允许申诉"
        if decision.verdict == "疑似生成合成"
        else ("人工复核" if decision.verdict == "无法确定" else "通过")
    )
    model_versions = [
        ModelVersion(
            name=signal.model_name,
            version=signal.model_version,
            license=signal.license,
        )
        for signal in signals
    ]
    model_versions.append(
        ModelVersion(
            name="fusion-config",
            version=fusion_version,
            license="internal",
        )
    )
    model_versions.append(
        ModelVersion(
            name="fusion-calibration",
            version=calibration_version,
            license="internal",
        )
    )
    if review is not None and review.model_version:
        model_versions.append(
            ModelVersion(
                name=review.provider,
                version=review.model_version,
                license=review.license,
            )
        )
    media = MediaInfo(
        filename=context.original_filename,
        mime_type=None,
        size_bytes=context.probe.size_bytes,
        sha256=sha256_file(context.probe.path),
        duration_seconds=duration_seconds,
        width=context.probe.width,
        height=context.probe.height,
        frame_rate=context.probe.frame_rate,
        video_codec=context.probe.video_codec,
        audio_codec=context.probe.audio_codec,
        has_audio=context.probe.has_audio,
    )
    provenance_model = Provenance(
        checks=[ProvenanceCheck.model_validate(item) for item in provenance.checks],
        explicit_ai_indicators=provenance.explicit_ai_indicators,
        status="阳性" if provenance.explicit_ai_indicators else "无声明",
        note="元数据未命中不能作为真实性的阴性证据。",
        limitation=(
            "缺失、损坏或被清除的来源声明不能作为真实性证据；"
            "阳性声明也需要验证签名与信任链后才能采信。"
        ),
    )
    modality_verdicts = _build_modality_verdicts(
        signals=signals,
        decision=decision,
        coverage=coverage,
        provenance=provenance_model,
    )
    calibrated_probability = CalibratedProbability(
        value=float(decision.calibrated_score),
        is_calibrated=decision.calibration_is_fitted,
        calibration_version=decision.calibration_version,
        calibration_group=decision.calibration_group,
        distribution=(
            f"内部评测分布（校准组 {decision.calibration_group}，"
            f"样本数 {decision.calibration_count}）"
            if decision.calibration_is_fitted
            else "未声明的评测分布：当前为 bootstrap 占位参数，未在内部评测集上拟合"
        ),
        confidence_interval=None,
        semantics=(
            "只表示在声明的评测分布和阈值下的生成合成风险，不代表事实确定性，"
            "也不能解释为“该内容是假的”的概率。"
        ),
    )
    gray_zone = GrayZone(
        minimum=decision.gray_zone_min,
        maximum=decision.gray_zone_max,
        score=float(decision.calibrated_score),
        in_gray_zone=(
            decision.gray_zone_min <= decision.calibrated_score < decision.gray_zone_max
        ),
        note=(
            "分数落在灰区时不下方向性结论。"
            + (
                "当前阈值来自占位校准配置，必须在内部评测集上重新锁定。"
                if not decision.calibration_is_fitted
                else "阈值已在内部校准集锁定，测试集不参与调参。"
            )
        ),
    )
    threshold_policy = _build_threshold_policy(
        decision=decision,
        fusion_version=fusion_version,
        calibration_version=calibration_version,
    )
    return ForensicReport(
        schema_version="1.1",
        report_id=str(uuid4()),
        job_id=job_id,
        generated_at=datetime.now(UTC),
        media=media,
        verdict=Verdict(decision.verdict),
        confidence=decision.verdict_confidence,
        calibrated_score=decision.calibrated_score,
        calibrated_probability=calibrated_probability,
        summary=_trim_summary(decision.summary),
        modality_verdicts=modality_verdicts,
        coverage=coverage,
        gray_zone=gray_zone,
        abstention_reasons=decision.abstention_reasons,
        abstention_codes=decision.abstention_codes,
        time_ranges=time_ranges,
        evidence=evidence,
        review=review,
        assessability=assessability,
        provenance=provenance_model,
        limitations=limitations,
        recommended_action=recommended_action,  # type: ignore[arg-type]
        artifacts=artifacts,
        model_versions=_deduplicate_models(model_versions),
        model_snapshot=ModelSnapshot(
            registry_version=f"fusion:{fusion_version}",
            captured_at=datetime.now(UTC),
            is_research_only=not decision.calibration_is_fitted,
            note=(
                "当前登记的检测分支均为研究/内部基线，权重来源与商用限制尚未逐项核验，"
                "不得接入高风险自动化处置路径。"
            ),
        ),
        threshold_policy=threshold_policy,
        audit_trace_id=job_id,
        processing={
            "conflict_detected": decision.conflict,
            "conflict_note": decision.conflict_note,
            "discounted_signals": decision.discounted_signals,
            "layer_scores": decision.layer_scores,
            "modality_scores": decision.modality_scores,
            "modality_divergence": decision.modality_divergence,
            "abstention_codes": decision.abstention_codes,
            "escalation_rule": decision.escalation_rule,
            "quality_status": face_quality.status,
            "review_status": review.status if review is not None else "disabled",
        },
    )


def _build_coverage(
    *,
    signals: list[SignalResult],
    face_quality: FaceQualityResult,
    duration_seconds: float | None,
    has_audio: bool,
) -> ModalityCoverage:
    """覆盖率只由已经执行的分支推导；无法估计时保留 None，不用 0 冒充阴性。"""
    notes: list[str] = []
    by_id = {signal.signal_id: signal for signal in signals}

    if face_quality.sampled_frames == 0:
        face_coverage = None
        notes.append("未采样到可用视频帧，人脸覆盖率无法估计。")
    else:
        face_coverage = face_quality.face_detected_ratio
        if face_coverage == 0:
            notes.append("采样帧中未检测到人脸，人脸分支不可用，其余分支独立给出结论。")

    visual_signal = by_id.get("visual_artifact")
    face_region_coverage: float | None = None
    if visual_signal is not None:
        sample_count = visual_signal.raw_value.get("sample_count")
        face_region_coverage = _ratio(sample_count, face_quality.sampled_frames)
    notes.append(
        "当前版本没有独立的全画面检测分支，画面结论只覆盖人脸区域；"
        "全画面覆盖率无法估计。"
    )

    audio_signal = by_id.get("audio_spoof")
    audio_coverage: float | None
    if audio_signal is not None and audio_signal.available:
        audio_coverage = _ratio(
            audio_signal.raw_value.get("analyzed_seconds"),
            duration_seconds,
        )
    elif not has_audio:
        audio_coverage = 0.0
        notes.append("输入不含音轨，音频与唇音分支关闭，覆盖范围下降。")
    else:
        audio_coverage = None
        notes.append("音轨存在但音频反欺骗分支未产出结果，音频覆盖率无法估计。")

    lip_sync = by_id.get("lip_sync")
    cross_modal_coverage: float | None
    if lip_sync is not None and lip_sync.available:
        cross_modal_coverage = _ratio(
            lip_sync.raw_value.get("audio_seconds"),
            duration_seconds,
        )
    elif not has_audio:
        cross_modal_coverage = 0.0
    else:
        cross_modal_coverage = None
        notes.append("唇音同步分支未产出结果，跨模态覆盖率无法估计。")

    if face_region_coverage is not None and face_region_coverage < 0.5:
        notes.append(
            "人脸分支只分析了部分采样帧，画面结论的适用区间随之收窄。"
        )
    return ModalityCoverage(
        full_frame=None,
        face=face_coverage,
        audio=audio_coverage,
        cross_modal=cross_modal_coverage,
        notes=list(dict.fromkeys(notes)),
    )


def _build_modality_verdicts(
    *,
    signals: list[SignalResult],
    decision: FusionDecision,
    coverage: ModalityCoverage,
    provenance: Provenance,
) -> list[ModalityVerdict]:
    by_id = {signal.signal_id: signal for signal in signals}
    coverage_by_modality = {
        "visual": coverage.face,
        "audio": coverage.audio,
        "cross_modal": coverage.cross_modal,
    }
    verdicts: list[ModalityVerdict] = []
    for modality, signal_ids in MODALITY_GROUPS.items():
        available = [
            by_id[signal_id]
            for signal_id in signal_ids
            if signal_id in by_id
            and by_id[signal_id].available
            and by_id[signal_id].score is not None
        ]
        if not available:
            reason = next(
                (
                    by_id[signal_id].explanation
                    for signal_id in signal_ids
                    if signal_id in by_id and not by_id[signal_id].available
                ),
                "该分支未运行。",
            )
            verdicts.append(
                ModalityVerdict(
                    modality=modality,  # type: ignore[arg-type]
                    label=MODALITY_LABELS[modality],
                    verdict=Verdict.UNCERTAIN,
                    score=None,
                    coverage=coverage_by_modality.get(modality),
                    status="unavailable",
                    signal_ids=list(signal_ids),
                    summary=f"该分支不可用：{reason}",
                    limitations=["分支不可用不等于该模态未发现异常。"],
                )
            )
            continue

        score = float(sum(signal.score or 0.0 for signal in available) / len(available))
        if score >= decision.gray_zone_max:
            modality_verdict = Verdict.SUSPECTED
        elif score < decision.gray_zone_min:
            modality_verdict = Verdict.TRUSTED
        else:
            modality_verdict = Verdict.UNCERTAIN
        described = "、".join(f"{signal.display_name} {signal.score:.2f}" for signal in available)
        status = "available" if len(available) == len(signal_ids) else "partial"
        verdicts.append(
            ModalityVerdict(
                modality=modality,  # type: ignore[arg-type]
                label=MODALITY_LABELS[modality],
                verdict=modality_verdict,
                score=score,
                coverage=coverage_by_modality.get(modality),
                status=status,
                signal_ids=[signal.signal_id for signal in available],
                summary=f"可用信号：{described}。该结论只在本模态覆盖范围内成立。",
                limitations=[
                    "模态结论与综合结论分开输出；不同模态指向不同方向不构成冲突。",
                ],
            )
        )

    provenance_positive = bool(provenance.explicit_ai_indicators)
    verdicts.append(
        ModalityVerdict(
            modality="provenance",
            label="溯源",
            verdict=Verdict.SUSPECTED if provenance_positive else Verdict.UNCERTAIN,
            score=0.99 if provenance_positive else None,
            coverage=1.0 if provenance.checks else None,
            status="available" if provenance.checks else "unavailable",
            signal_ids=["provenance"],
            summary=(
                "命中可验证的生成合成声明，作为阳性证据。"
                if provenance_positive
                else "未检出生成声明；缺失、损坏或被清除的标识不能作为真实性证据。"
            ),
            limitations=[provenance.limitation],
        )
    )
    return verdicts


def _build_threshold_policy(
    *,
    decision: FusionDecision,
    fusion_version: str,
    calibration_version: str,
) -> ThresholdPolicy:
    sample_count = decision.calibration_negative_count
    if sample_count is None:
        evidence_status = "unknown"
        statement = (
            "当前没有可核验的校准样本量记录，无法声明任何 FPR 门槛已经过验证。"
        )
    elif sample_count < MIN_REAL_NEGATIVES_FOR_FPR_0_1:
        evidence_status = "insufficient"
        statement = (
            f"FPR=0.1% 至少需要 {MIN_REAL_NEGATIVES_FOR_FPR_0_1} 条独立真实阴性样本，"
            f"当前校准样本数为 {sample_count}，因此不宣称已满足该门槛；"
            "零误报时只能报告单侧置信上界。"
        )
    else:
        evidence_status = "verified"
        statement = (
            f"校准样本数 {sample_count} 满足 FPR=0.1% 的最低样本量要求，"
            "仍需报告 bootstrap 置信区间与分组结果。"
        )
    return ThresholdPolicy(
        policy_id=(
            f"{calibration_version}:"
            f"{decision.gray_zone_min:.2f}-{decision.gray_zone_max:.2f}"
        ),
        source=f"fusion:{fusion_version} + calibration:{calibration_version}",
        gray_zone=(decision.gray_zone_min, decision.gray_zone_max),
        target_fprs=[0.01, 0.001],
        real_negative_samples=sample_count,
        fpr_evidence_status=evidence_status,
        statement=statement,
    )


def _ratio(numerator: object, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    try:
        value = float(numerator)
    except (TypeError, ValueError):
        return None
    return round(max(0.0, min(1.0, value / denominator)), 4)


def _signal_to_evidence(signal: SignalResult) -> EvidenceItem:
    time_range = None
    if signal.time_windows:
        best = max(signal.time_windows, key=lambda item: item.score)
        time_range = TimeRange(
            start=round(best.start, 3),
            end=round(best.end, 3),
            score=float(best.score),
            reason=best.reason,
        )
    region_label = signal.raw_value.get("region")
    if isinstance(region_label, str):
        region = EvidenceRegion(label=region_label)
    elif signal.signal_id in {"visual_artifact", "digital_human", "face_quality"}:
        region = EvidenceRegion(label="face_crop")
    else:
        region = None
    heuristic = any(
        marker in f"{signal.model_name} {signal.model_version}".lower()
        for marker in ("heuristic", "bootstrap", "baseline", "rule")
    )
    limitations: list[str] = []
    if heuristic:
        limitations.append(
            "该分支为内部启发式/基线实现，未在内部评测集上完成校准与置信区间验证。"
        )
    return EvidenceItem(
        id=signal.signal_id,
        signal=signal.signal_id,
        display_name=signal.display_name,
        layer=SIGNAL_LAYERS.get(signal.signal_id),
        source=signal.model_name,
        direction=signal_direction(signal),
        strength=signal_strength(signal.score, signal.available),
        score=signal.score,
        unit="probability" if signal.score is not None else None,
        raw_value={
            **signal.raw_value,
            "latency_ms": signal.latency_ms,
            "model": f"{signal.model_name}:{signal.model_version}",
        },
        explanation=signal.explanation,
        time_range=time_range,
        region=region,
        model=EvidenceModel(
            name=signal.model_name,
            version=signal.model_version,
            license=signal.license,
        ),
        uncertainty=(
            "未提供置信区间；分数来自未经校准的启发式基线。"
            if heuristic and signal.score is not None
            else None
        ),
        limitations=limitations,
        available=signal.available,
    )


def _collect_time_ranges(signals: list[SignalResult]) -> list[TimeRange]:
    ranges: list[TimeRange] = []
    for signal in signals:
        for window in signal.time_windows:
            ranges.append(
                TimeRange(
                    start=round(window.start, 3),
                    end=round(window.end, 3),
                    score=float(window.score),
                    reason=f"{signal.display_name}：{window.reason}",
                )
            )
    ranges.sort(key=lambda item: (item.start, item.end))
    merged: list[TimeRange] = []
    for item in ranges:
        if merged and item.start <= merged[-1].end + 0.25:
            previous = merged[-1]
            merged[-1] = TimeRange(
                start=previous.start,
                end=max(previous.end, item.end),
                score=max(previous.score, item.score),
                reason=previous.reason if previous.score >= item.score else item.reason,
            )
        else:
            merged.append(item)
    return merged[:20]


def _deduplicate_models(models: list[ModelVersion]) -> list[ModelVersion]:
    deduplicated: dict[tuple[str, str], ModelVersion] = {}
    for model in models:
        deduplicated[(model.name, model.version)] = model
    return list(deduplicated.values())


def _trim_summary(summary: str) -> str:
    return summary if len(summary) <= 150 else summary[:147] + "..."


def render_static_report(report: ForensicReport) -> str:
    environment = Environment(
        loader=BaseLoader(),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = environment.from_string(STATIC_REPORT_TEMPLATE)
    return template.render(report=report, payload=report.model_dump(mode="json"))


STATIC_REPORT_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>检测报告 {{ report.report_id }}</title>
  <style>
    :root { color-scheme: light; --ink:#172033; --muted:#667085; --line:#e4e7ec; --bg:#f5f7fa; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif; }
    main { max-width:1040px; margin:0 auto; padding:32px 20px 64px; }
    header { border-bottom:1px solid var(--line); padding-bottom:24px; margin-bottom:24px; }
    h1,h2,h3 { margin:0; }
    h1 { font-size:28px; }
    h2 { font-size:19px; margin:32px 0 14px; }
    p { line-height:1.7; }
    .meta { color:var(--muted); font-size:13px; margin-top:8px; }
    .verdict { display:grid; grid-template-columns:1fr auto; gap:18px; align-items:center; padding:24px; background:white; border:1px solid var(--line); border-radius:10px; }
    .verdict strong { display:block; font-size:30px; }
    .score { font-size:34px; font-variant-numeric:tabular-nums; }
    .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
    .grid-3 { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
    article { background:white; border:1px solid var(--line); border-radius:8px; padding:18px; }
    article p { margin:8px 0 0; color:var(--muted); font-size:14px; }
    article .verdict-tag { display:inline-block; font-size:12px; font-weight:700; padding:2px 8px; border-radius:999px; background:#f2f4f7; color:#475467; }
    table { width:100%; border-collapse:collapse; background:white; border:1px solid var(--line); }
    th,td { text-align:left; vertical-align:top; padding:11px 12px; border-bottom:1px solid var(--line); font-size:13px; }
    th { background:#f9fafb; color:#475467; }
    code { word-break:break-all; white-space:pre-wrap; font-size:12px; }
    ul { padding-left:20px; line-height:1.75; }
    .scope { margin-top:10px; color:var(--muted); font-size:13px; line-height:1.7; }
    @media (max-width:700px) { .grid, .grid-3 { grid-template-columns:1fr; } .verdict { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<main>
  <header>
    <h1>数字人 / 生成合成内容检测报告</h1>
    <div class="meta">报告 {{ report.report_id }} · 任务 {{ report.job_id }} · {{ report.generated_at }}</div>
  </header>
  <section class="verdict">
    <div><span>综合判定</span><strong>{{ report.verdict }}</strong><p>{{ report.summary }}</p></div>
    <div>
      <span>{{ "校准概率" if report.calibrated_probability and report.calibrated_probability.is_calibrated else "未校准风险分" }}</span>
      <div class="score">{{ "%.3f"|format(report.calibrated_score) }}</div>
    </div>
  </section>
  <p class="scope">
    “可信”只表示在本次检测覆盖的模态、时段和质量范围内未发现足够强的生成合成证据，不构成对原始真实性的证明。
  </p>

  {% if report.modality_verdicts %}
  <h2>模态级结论</h2>
  <div class="grid-3">
    {% for item in report.modality_verdicts %}
    <article>
      <strong>{{ item.label }}</strong>
      <div><span class="verdict-tag">{{ item.verdict }}</span></div>
      <p>{{ item.summary }}</p>
      <p>覆盖率：{{ "%.0f%%"|format(item.coverage * 100) if item.coverage is not none else "无法估计" }}</p>
    </article>
    {% endfor %}
  </div>
  {% endif %}

  {% if report.abstention_reasons %}
  <h2>无法确定的原因</h2>
  <ul>{% for item in report.abstention_reasons %}<li>{{ item }}</li>{% endfor %}</ul>
  {% endif %}

  {% if report.coverage %}
  <h2>覆盖率</h2>
  <table>
    <tr>
      <th>人脸</th><td>{{ "%.0f%%"|format(report.coverage.face * 100) if report.coverage.face is not none else "无法估计" }}</td>
      <th>音频</th><td>{{ "%.0f%%"|format(report.coverage.audio * 100) if report.coverage.audio is not none else "无法估计" }}</td>
    </tr>
    <tr>
      <th>跨模态</th><td>{{ "%.0f%%"|format(report.coverage.cross_modal * 100) if report.coverage.cross_modal is not none else "无法估计" }}</td>
      <th>全画面</th><td>{{ "%.0f%%"|format(report.coverage.full_frame * 100) if report.coverage.full_frame is not none else "当前版本未启用" }}</td>
    </tr>
  </table>
  {% if report.coverage.notes %}<ul>{% for item in report.coverage.notes %}<li>{{ item }}</li>{% endfor %}</ul>{% endif %}
  {% endif %}

  <h2>媒体信息</h2>
  <table>
    <tr><th>文件</th><td>{{ report.media.filename }}</td><th>SHA-256</th><td><code>{{ report.media.sha256 }}</code></td></tr>
    <tr><th>时长</th><td>{{ report.media.duration_seconds }}</td><th>分辨率</th><td>{{ report.media.width }} × {{ report.media.height }}</td></tr>
    <tr><th>视频编码</th><td>{{ report.media.video_codec }}</td><th>音轨</th><td>{{ report.media.audio_codec or "无" }}</td></tr>
  </table>

  <h2>证据链</h2>
  <table>
    <thead><tr><th>信号</th><th>取证层</th><th>方向</th><th>强度</th><th>分数</th><th>解释</th></tr></thead>
    <tbody>
    {% for item in report.evidence %}
      <tr>
        <td>{{ item.display_name }}</td>
        <td>{{ item.layer or "未分层" }}</td>
        <td>{{ item.direction }}</td>
        <td>{{ item.strength }}</td>
        <td>{{ "%.3f"|format(item.score) if item.score is not none else "不可用" }}</td>
        <td>{{ item.explanation }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>可鉴定性</h2>
  <article>
    <strong>{{ report.assessability.status }} · {{ "%.3f"|format(report.assessability.score) }}</strong>
    {% if report.assessability.reasons %}<ul>{% for reason in report.assessability.reasons %}<li>{{ reason }}</li>{% endfor %}</ul>{% endif %}
  </article>

  {% if report.review and report.review.triggered %}
  <h2>证据复核</h2>
  <article>
    <strong>{{ report.review.status }} · {{ report.review.provider }}</strong>
    <p>{{ report.review.summary }}</p>
    {% if report.review.statements %}
    <ul>
      {% for statement in report.review.statements %}
      <li>{{ statement.claim }} <small>证据：{{ statement.evidence_ids|join("、") }}</small></li>
      {% endfor %}
    </ul>
    {% endif %}
    <p>{{ report.review.note }}</p>
  </article>
  {% endif %}

  <h2>溯源</h2>
  <p class="scope">{{ report.provenance.limitation }}</p>
  {% for check in report.provenance.checks %}
    <article style="margin-bottom:10px"><strong>{{ check.name }} · {{ check.status }}</strong><p>{{ check.explanation }}</p></article>
  {% endfor %}

  {% if report.threshold_policy %}
  <h2>阈值与校准</h2>
  <article>
    <strong>{{ report.threshold_policy.policy_id }}</strong>
    <p>{{ report.threshold_policy.statement }}</p>
    <p>灰区：{{ "%.2f"|format(report.threshold_policy.gray_zone[0]) }} – {{ "%.2f"|format(report.threshold_policy.gray_zone[1]) }}</p>
  </article>
  {% endif %}

  {% if report.model_snapshot %}
  <h2>模型快照</h2>
  <article>
    <strong>{{ report.model_snapshot.registry_version }}</strong>
    <p>{{ report.model_snapshot.note }}</p>
    <p>审计追踪：{{ report.audit_trace_id or "未记录" }}</p>
  </article>
  {% endif %}

  <h2>局限声明</h2>
  <ul>{% for item in report.limitations %}<li>{{ item }}</li>{% endfor %}</ul>

  <h2>机器可读数据</h2>
  <details><summary>展开 JSON</summary><pre><code>{{ payload | tojson(indent=2) }}</code></pre></details>
</main>
</body>
</html>
"""
