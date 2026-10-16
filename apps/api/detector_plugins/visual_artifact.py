from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from apps.api.detector_plugins.base import PluginContext, SignalResult
from apps.api.detector_plugins.face_quality import FaceSample


@dataclass
class VisualArtifactResult:
    signal: SignalResult
    heatmap_overlay: np.ndarray | None
    representative_frame: np.ndarray | None
    beauty_filter_score: float
    beauty_filter_raw: dict


class VisualArtifactPlugin:
    signal_id = "visual_artifact"
    display_name = "帧级视觉伪影"
    model_name = "wavelet-srm-heuristic"
    model_version = "bootstrap-1"
    license = "internal"

    def analyze(self, context: PluginContext) -> VisualArtifactResult:
        started = time.perf_counter()
        face_analysis = context.face_analysis
        if face_analysis is None or not face_analysis.samples:
            return self._unavailable("没有足够的人脸样本，无法计算视觉伪影。")

        features = [_crop_features(sample.crop) for sample in face_analysis.samples]
        raw_metrics = {
            key: float(np.median([feature[key] for feature in features])) for key in features[0]
        }
        score = _heuristic_score(raw_metrics)
        beauty_score, beauty_raw = _beauty_filter_score(face_analysis.samples)
        representative = max(
            face_analysis.samples,
            key=lambda sample: sample.bbox[2] * sample.bbox[3],
        )
        heatmap = _make_heatmap(representative)
        latency_ms = int((time.perf_counter() - started) * 1000)
        signal = SignalResult(
            signal_id=self.signal_id,
            display_name=self.display_name,
            score=score,
            available=True,
            direction="指向生成" if score >= 0.5 else "中性",
            raw_value={
                **{key: round(value, 6) for key, value in raw_metrics.items()},
                "sample_count": len(features),
                "calibration": "bootstrap heuristic; replace with trained ONNX detector",
            },
            explanation=(
                "使用高频残差、块效应、边界颜色差和噪声一致性做弱监督启发式估计。"
                "该分数只能作为信号之一，不能单独形成判定。"
            ),
            model_name=self.model_name,
            model_version=self.model_version,
            license=self.license,
            latency_ms=latency_ms,
        )
        return VisualArtifactResult(
            signal=signal,
            heatmap_overlay=heatmap,
            representative_frame=representative.frame,
            beauty_filter_score=beauty_score,
            beauty_filter_raw=beauty_raw,
        )

    def _unavailable(self, reason: str) -> VisualArtifactResult:
        return VisualArtifactResult(
            signal=SignalResult(
                signal_id=self.signal_id,
                display_name=self.display_name,
                score=None,
                available=False,
                direction="中性",
                explanation=reason,
                model_name=self.model_name,
                model_version=self.model_version,
                license=self.license,
            ),
            heatmap_overlay=None,
            representative_frame=None,
            beauty_filter_score=0.0,
            beauty_filter_raw={"available": False},
        )


def _crop_features(crop: np.ndarray) -> dict[str, float]:
    resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    high_frequency_ratio = float(np.std(laplacian) / (np.mean(np.abs(gray)) + 1e-6))

    horizontal = np.abs(np.diff(gray, axis=1))
    vertical = np.abs(np.diff(gray, axis=0))
    boundary_columns = horizontal[:, 7::8]
    inner_columns = np.delete(horizontal, np.s_[7::8], axis=1)
    boundary_rows = vertical[7::8, :]
    inner_rows = np.delete(vertical, np.s_[7::8], axis=0)
    boundary_mean = float(
        (np.mean(boundary_columns) + np.mean(boundary_rows)) / 2
        if boundary_columns.size and boundary_rows.size
        else 0.0
    )
    inner_mean = float(
        (np.mean(inner_columns) + np.mean(inner_rows)) / 2
        if inner_columns.size and inner_rows.size
        else 0.0
    )
    blockiness = boundary_mean / (inner_mean + 1e-6)

    grid_size = 6
    local_stds: list[float] = []
    for row in range(grid_size):
        for column in range(grid_size):
            patch = gray[
                row * 224 // grid_size : (row + 1) * 224 // grid_size,
                column * 224 // grid_size : (column + 1) * 224 // grid_size,
            ]
            local_stds.append(float(np.std(patch)))
    local_stds_array = np.asarray(local_stds)
    noise_cv = float(np.std(local_stds_array) / (np.mean(local_stds_array) + 1e-6))

    color = resized.astype(np.float32)
    channels = cv2.split(color)
    channel_correlations = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        if float(np.std(channels[left])) < 1e-6 or float(np.std(channels[right])) < 1e-6:
            channel_correlations.append(0.0)
            continue
        value = np.corrcoef(channels[left].ravel(), channels[right].ravel())[0, 1]
        channel_correlations.append(float(value) if np.isfinite(value) else 0.0)

    edges = cv2.Canny(gray.astype(np.uint8), 60, 140)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    return {
        "high_frequency_ratio": high_frequency_ratio,
        "blockiness": blockiness,
        "noise_cv": noise_cv,
        "channel_correlation_mean": float(np.mean(channel_correlations)),
        "edge_density": edge_density,
    }


def _heuristic_score(metrics: dict[str, float]) -> float:
    high_frequency_penalty = np.clip(
        (0.026 - metrics["high_frequency_ratio"]) / 0.026,
        0,
        1,
    )
    blockiness_penalty = np.clip((metrics["blockiness"] - 1.25) / 0.9, 0, 1)
    noise_penalty = np.clip((metrics["noise_cv"] - 0.58) / 0.65, 0, 1)
    correlation_penalty = np.clip((metrics["channel_correlation_mean"] - 0.965) / 0.035, 0, 1)
    edge_penalty = np.clip((0.08 - metrics["edge_density"]) / 0.08, 0, 1)
    logit = (
        -1.1
        + 1.35 * high_frequency_penalty
        + 0.85 * blockiness_penalty
        + 1.05 * noise_penalty
        + 0.55 * correlation_penalty
        + 0.75 * edge_penalty
    )
    return float(1 / (1 + math.exp(-logit)))


def _beauty_filter_score(samples: list[FaceSample]) -> tuple[float, dict]:
    """估计磨皮/美颜/滤镜干扰，并按背景复杂度归一化。

    背景纹理越丰富，人脸纹理相对背景偏低属于正常现象；固定阈值会把
    高复杂度背景误判为美颜。这里先用背景复杂度估计期望的人脸/背景
    纹理比，再看实际比值偏离期望的程度。
    """
    face_scores: list[float] = []
    background_scores: list[float] = []
    for sample in samples:
        height, width = sample.frame.shape[:2]
        x, y, box_width, box_height = sample.bbox
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + box_width)
        y2 = min(height, y + box_height)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        face_gray = cv2.cvtColor(sample.frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        face_scores.append(float(cv2.Laplacian(face_gray, cv2.CV_64F).var()))

        gray = cv2.cvtColor(sample.frame, cv2.COLOR_BGR2GRAY)
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        mask = np.ones((height, width), dtype=bool)
        margin_x = int(box_width * 0.12)
        margin_y = int(box_height * 0.12)
        mask[
            max(0, y1 - margin_y) : min(height, y2 + margin_y),
            max(0, x1 - margin_x) : min(width, x2 + margin_x),
        ] = False
        if mask.sum() > 200:
            background_scores.append(float(np.var(laplacian[mask])))
    face_median = float(np.median(face_scores)) if face_scores else 0.0
    background_median = float(np.median(background_scores)) if background_scores else face_median
    ratio = face_median / (background_median + 1e-6)
    background_complexity = _background_complexity(background_median)
    expected_ratio = float(np.clip(0.95 - 0.45 * background_complexity, 0.45, 0.95))
    deviation = expected_ratio - ratio
    reliability = float(np.clip(face_median / 12.0, 0.0, 1.0))
    beauty_score = float(np.clip(deviation / max(0.30, 0.75 * expected_ratio), 0, 1) * reliability)
    return beauty_score, {
        "face_texture_variance": face_median,
        "background_texture_variance": background_median,
        "face_to_background_texture_ratio": ratio,
        "background_complexity": background_complexity,
        "expected_texture_ratio": expected_ratio,
        "texture_reliability": reliability,
        "region": "strict_face_bbox",
    }


def _background_complexity(background_texture_variance: float) -> float:
    """把背景拉普拉斯方差映射到 [0, 1] 的复杂度（对数刻度）。"""
    low = float(np.log1p(15.0))
    high = float(np.log1p(1500.0))
    value = float(np.log1p(max(0.0, background_texture_variance)))
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _make_heatmap(sample: FaceSample) -> np.ndarray:
    crop = cv2.resize(sample.crop, (224, 224), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    patch_size = 16
    patches = []
    for row in range(0, 224, patch_size):
        for column in range(0, 224, patch_size):
            patches.append(
                float(np.mean(residual[row : row + patch_size, column : column + patch_size]))
            )
    patch_values = np.asarray(patches).reshape(14, 14)
    median = float(np.median(patch_values))
    mad = float(np.median(np.abs(patch_values - median))) + 1e-6
    anomaly = np.clip((patch_values - median) / (mad * 3.5), 0, 1)
    anomaly = cv2.resize(anomaly, (sample.frame.shape[1], sample.frame.shape[0]))
    anomaly = cv2.GaussianBlur(anomaly, (0, 0), sigmaX=12)
    color = cv2.applyColorMap((anomaly * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(sample.frame, 0.62, color, 0.38, 0)
    return overlay
