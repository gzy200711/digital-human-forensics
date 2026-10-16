from __future__ import annotations

import math
import time

import cv2
import numpy as np

from apps.api.detector_plugins.base import PluginContext, SignalResult, SignalWindow


class TemporalConsistencyPlugin:
    signal_id = "temporal_consistency"
    display_name = "时序一致性"
    model_name = "track-statistics-heuristic"
    model_version = "bootstrap-1"
    license = "internal"

    def analyze(self, context: PluginContext) -> SignalResult:
        started = time.perf_counter()
        face_analysis = context.face_analysis
        if face_analysis is None or len(face_analysis.samples) < 4:
            return self._unavailable("人脸样本少于 4 帧，无法计算时序一致性。")

        samples = face_analysis.samples
        centers = np.asarray(
            [
                (
                    sample.bbox[0] + sample.bbox[2] / 2,
                    sample.bbox[1] + sample.bbox[3] / 2,
                )
                for sample in samples
            ],
            dtype=float,
        )
        areas = np.asarray([sample.bbox[2] * sample.bbox[3] for sample in samples], dtype=float)
        times = np.asarray([sample.timestamp for sample in samples], dtype=float)
        delta_times = np.diff(times)
        delta_times[delta_times <= 0] = 1 / 2

        # 只看人脸框内的运动：用框宽归一化，避免全帧或全局运镜主导分数。
        scales = np.sqrt(areas[1:, None]) + 1e-6
        normalized_velocity = np.diff(centers, axis=0) / delta_times[:, None] / scales
        velocity = np.linalg.norm(normalized_velocity, axis=1)
        acceleration = np.diff(normalized_velocity, axis=0) / np.maximum(
            delta_times[1:, None], 1e-6
        )
        jitter_raw = (
            float(np.median(np.linalg.norm(acceleration, axis=1))) if len(acceleration) else 0
        )
        area_cv = float(np.std(areas) / (np.mean(areas) + 1e-6))

        histograms = []
        brightness = []
        for sample in samples:
            region = sample.face_crop
            if region.size == 0:
                region = sample.crop
            resized = cv2.resize(region, (64, 64), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            histogram = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            histogram = cv2.normalize(histogram, histogram).flatten()
            histograms.append(histogram)
            brightness.append(float(np.mean(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))))
        histogram_drift = [
            float(
                cv2.compareHist(histograms[index - 1], histograms[index], cv2.HISTCMP_BHATTACHARYYA)
            )
            for index in range(1, len(histograms))
        ]
        brightness_flicker = float(
            np.median(np.abs(np.diff(np.asarray(brightness, dtype=float)))) / 255
        )

        jitter_component = np.clip(jitter_raw / 6.0, 0, 1)
        area_component = np.clip((area_cv - 0.035) / 0.22, 0, 1)
        drift_component = np.clip((float(np.median(histogram_drift)) - 0.22) / 0.5, 0, 1)
        flicker_component = np.clip((brightness_flicker - 0.018) / 0.09, 0, 1)
        logit = (
            -1.75
            + 2.0 * jitter_component
            + 1.7 * area_component
            + 1.75 * drift_component
            + 1.0 * flicker_component
        )
        score = float(1 / (1 + math.exp(-logit)))

        windows: list[SignalWindow] = []
        for index, sample in enumerate(samples[1:], start=1):
            local_drift = histogram_drift[index - 1] if index - 1 < len(histogram_drift) else 0.0
            local_velocity = float(velocity[index - 1]) if index - 1 < velocity.size else 0.0
            local_score = float(
                np.clip(
                    0.5 * np.clip(local_drift / 0.5, 0, 1)
                    + 0.5 * np.clip(local_velocity / 1.2, 0, 1),
                    0,
                    1,
                )
            )
            if local_score >= 0.68:
                windows.append(
                    SignalWindow(
                        start=max(0.0, sample.timestamp - 0.5),
                        end=sample.timestamp + 0.5,
                        score=local_score,
                        reason="该片段存在较强的人脸纹理或运动突变。",
                    )
                )

        return SignalResult(
            signal_id=self.signal_id,
            display_name=self.display_name,
            score=score,
            available=True,
            direction="指向生成" if score >= 0.5 else "中性",
            raw_value={
                "jitter": jitter_raw,
                "velocity_median": float(np.median(velocity)) if velocity.size else 0.0,
                "area_coefficient_of_variation": area_cv,
                "histogram_drift_median": float(np.median(histogram_drift)),
                "brightness_flicker_median": brightness_flicker,
                "sample_count": len(samples),
                "region": "face_bbox",
            },
            explanation=(
                "仅统计人脸框内的像素：框内颜色直方图变化、亮度闪烁，"
                "以及用人脸尺寸归一化后的头部运动加速度。"
                "广告剪辑、快速运镜和强压缩仍会提高该分数。"
            ),
            time_windows=_merge_windows(windows),
            model_name=self.model_name,
            model_version=self.model_version,
            license=self.license,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def _unavailable(self, reason: str) -> SignalResult:
        return SignalResult(
            signal_id=self.signal_id,
            display_name=self.display_name,
            score=None,
            available=False,
            direction="中性",
            explanation=reason,
            model_name=self.model_name,
            model_version=self.model_version,
            license=self.license,
        )


def _merge_windows(windows: list[SignalWindow]) -> list[SignalWindow]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda item: item.start)
    merged = [ordered[0]]
    for window in ordered[1:]:
        previous = merged[-1]
        if window.start <= previous.end + 0.25:
            merged[-1] = SignalWindow(
                start=previous.start,
                end=max(previous.end, window.end),
                score=max(previous.score, window.score),
                reason=previous.reason,
            )
        else:
            merged.append(window)
    return merged
