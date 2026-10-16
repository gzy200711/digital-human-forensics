from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from apps.api.config import get_settings
from apps.api.detector_plugins.audio import AudioConsistencyPlugin, AudioSpoofPlugin
from apps.api.detector_plugins.audio_onnx import OnnxAudioSpoofPlugin
from apps.api.detector_plugins.base import PluginContext
from apps.api.detector_plugins.digital_human import DigitalHumanPlugin
from apps.api.detector_plugins.face_quality import FaceQualityAnalyzer
from apps.api.detector_plugins.provenance import (
    ProvenanceAnalyzer,
    build_render_chain_signal,
)
from apps.api.detector_plugins.temporal import TemporalConsistencyPlugin
from apps.api.detector_plugins.visual_artifact import VisualArtifactPlugin
from apps.api.detector_plugins.visual_onnx import OnnxVisualArtifactPlugin
from apps.api.services.fusion import load_fusion_engine
from apps.api.services.media import MediaProbeError, probe_media
from dhfp.evaluation.localization import SpatialBox, TimeSegment
from dhfp.evaluation.schema import DatasetRecord
from dhfp.plugins.base import DetectorUnavailable, EvaluationPrediction


class TrivialMetadataBaseline:
    detector_id = "trivial-metadata"
    display_name = "时长/分辨率/帧率平凡基线"
    version = "1.0"
    license = "internal"
    description = "只用容器级剪辑特征，用来证明模型没有依赖数据集 shortcut。"

    def __init__(self) -> None:
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=20260918),
        )
        self._fitted = False

    def fit(self, records: list[DatasetRecord]) -> None:
        labels = np.asarray([record.is_positive for record in records], dtype=int)
        if np.unique(labels).size < 2:
            raise DetectorUnavailable("平凡基线训练集缺少正负样本")
        features = np.asarray([self._features(record) for record in records], dtype=float)
        self.model.fit(features, labels)
        self._fitted = True

    def predict(self, record: DatasetRecord) -> EvaluationPrediction:
        if not self._fitted:
            raise DetectorUnavailable("平凡基线尚未在训练集上拟合")
        started = time.perf_counter()
        probability = float(self.model.predict_proba(self._features(record).reshape(1, -1))[0, 1])
        return EvaluationPrediction(
            sample_id=record.sample_id,
            score=probability,
            available=True,
            latency_ms=(time.perf_counter() - started) * 1000,
            details={"features": self._features(record).tolist()},
        )

    @staticmethod
    def _features(record: DatasetRecord) -> np.ndarray:
        aspect_ratio = record.width / max(1, record.height)
        return np.asarray(
            [
                record.duration_seconds,
                record.width,
                record.height,
                record.frame_rate,
                aspect_ratio,
                float(record.has_audio),
            ],
            dtype=float,
        )


class MetadataRuleBaseline:
    detector_id = "metadata-rule"
    display_name = "元数据阳性规则"
    version = "1.0"
    license = "internal"
    description = "只把显式生成标识当阳性证据，不把缺失元数据当真实性证据。"

    def predict(self, record: DatasetRecord) -> EvaluationPrediction:
        started = time.perf_counter()
        path = Path(record.media_path)
        if not path.exists():
            raise DetectorUnavailable(f"媒体不存在: {path}")
        try:
            probe = probe_media(path)
        except MediaProbeError as exc:
            raise DetectorUnavailable(str(exc)) from exc
        from apps.api.detector_plugins.provenance import ProvenanceAnalyzer

        result = ProvenanceAnalyzer()._metadata_hits(probe.tags, probe.streams)
        score = 0.99 if result else 0.05
        return EvaluationPrediction(
            sample_id=record.sample_id,
            score=score,
            available=True,
            latency_ms=(time.perf_counter() - started) * 1000,
            details={"metadata_hits": result},
        )


class FusionPipelineDetector:
    detector_id = "fusion-pipeline"
    display_name = "多信号融合基线"
    version = "2026-09-layered"
    license = "internal"
    description = (
        "质量门控 + 渲染链路指纹 + 视觉伪影 + 数字人特征 + 时序一致性 "
        "+ 唇音同步 + 音频反欺骗 + 分层融合与分位数校准。"
    )

    def __init__(self) -> None:
        self.settings = get_settings()
        self.face_analyzer = FaceQualityAnalyzer(
            pass_threshold=self.settings.quality_pass_threshold,
            marginal_threshold=self.settings.quality_marginal_threshold,
        )
        self.provenance_analyzer = ProvenanceAnalyzer()
        visual_onnx_path = (
            Path(self.settings.visual_onnx_path).expanduser()
            if self.settings.visual_onnx_path
            else None
        )
        if visual_onnx_path:
            self.visual_plugin = OnnxVisualArtifactPlugin(
                visual_onnx_path,
                input_size=self.settings.onnx_input_size,
                source=self.settings.visual_onnx_source,
                license_name=self.settings.visual_onnx_license,
            )
        else:
            self.visual_plugin = VisualArtifactPlugin()
        self.temporal_plugin = TemporalConsistencyPlugin()
        self.digital_human_plugin = DigitalHumanPlugin(sample_fps=self.settings.sample_fps)
        self.lip_sync_plugin = AudioConsistencyPlugin()
        self.audio_spoof_plugin = self._build_audio_spoof_plugin()
        self.fusion_engine = load_fusion_engine(
            self.settings.fusion_config_path,
            self.settings.calibration_path,
        )

    def predict(self, record: DatasetRecord) -> EvaluationPrediction:
        started = time.perf_counter()
        path = Path(record.media_path)
        if not path.exists():
            raise DetectorUnavailable(f"媒体不存在: {path}")
        try:
            probe = probe_media(path)
        except MediaProbeError as exc:
            raise DetectorUnavailable(str(exc)) from exc
        context = PluginContext(
            job_id=f"eval-{record.sample_id}",
            original_filename=path.name,
            source_object_key=f"eval/{record.sample_id}",
            media_path=path,
            probe=probe,
            settings=self.settings,
        )
        provenance = self.provenance_analyzer.analyze(context)
        face_quality = self.face_analyzer.analyze(
            probe,
            sample_fps=self.settings.sample_fps,
            max_frames=self.settings.max_sampled_frames,
        )
        context.face_analysis = face_quality
        signals = []
        temporal_segments: list[TimeSegment] = []
        spatial_boxes: list[SpatialBox] = []
        if provenance.explicit_ai_indicators:
            from apps.api.detector_plugins.provenance import ProvenanceSignalPlugin

            signals.append(ProvenanceSignalPlugin().analyze(context))
        signals.append(build_render_chain_signal(provenance))
        visual = None
        if face_quality.status != "fail":
            visual = self.visual_plugin.analyze(context)
            temporal_signal = self.temporal_plugin.analyze(context)
            signals.extend(
                [
                    visual.signal,
                    temporal_signal,
                    self.digital_human_plugin.analyze(context),
                    self.lip_sync_plugin.analyze(context),
                ]
            )
            temporal_segments = [
                TimeSegment(
                    start=window.start,
                    end=window.end,
                    score=window.score,
                )
                for window in temporal_signal.time_windows
            ]
            spatial_boxes = [
                SpatialBox(
                    x1=float(sample.bbox[0]),
                    y1=float(sample.bbox[1]),
                    x2=float(sample.bbox[0] + sample.bbox[2]),
                    y2=float(sample.bbox[1] + sample.bbox[3]),
                    score=1.0,
                    timestamp=sample.timestamp,
                )
                for sample in face_quality.samples
            ]
        if probe.has_audio:
            signals.append(self.audio_spoof_plugin.analyze(context))
        decision = self.fusion_engine.decide(
            signals,
            provenance_positive=bool(provenance.explicit_ai_indicators),
            assessability_status=face_quality.status,
            beauty_filter_score=visual.beauty_filter_score if visual else 0.0,
        )
        return EvaluationPrediction(
            sample_id=record.sample_id,
            score=decision.calibrated_score,
            available=True,
            latency_ms=(time.perf_counter() - started) * 1000,
            details={
                "verdict": decision.verdict,
                "quality_status": face_quality.status,
                "quality_score": face_quality.score,
                "signals": {
                    signal.signal_id: signal.score for signal in signals if signal.available
                },
                "conflict": decision.conflict,
            },
            temporal_segments=temporal_segments,
            spatial_boxes=spatial_boxes,
        )

    def _build_audio_spoof_plugin(self):
        configured = self.settings.audio_spoof_onnx_path
        if not configured:
            return AudioSpoofPlugin()
        return OnnxAudioSpoofPlugin(
            Path(configured).expanduser(),
            input_mode=self.settings.audio_spoof_input_mode,
            sample_rate=self.settings.audio_spoof_sample_rate,
            segment_seconds=self.settings.audio_spoof_segment_seconds,
            max_segments=self.settings.audio_spoof_max_segments,
            aggregation=self.settings.audio_spoof_aggregation,
            source=self.settings.audio_spoof_source,
            license_name=self.settings.audio_spoof_license,
        )
