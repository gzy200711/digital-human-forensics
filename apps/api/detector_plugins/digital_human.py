from __future__ import annotations

import math
import time

import cv2
import numpy as np

from apps.api.detector_plugins.base import PluginContext, SignalResult, SignalWindow
from apps.api.detector_plugins.face_quality import FaceSample

EYE_BAND_TOP = 0.22
EYE_BAND_BOTTOM = 0.58
MIN_UNFROZEN_EYE_MOTION = 0.012
MIN_OBSERVED_SECONDS = 6.0
HAND_AREA_RANGE = (0.004, 0.10)


class DigitalHumanPlugin:
    """数字人专属弱信号：眨眼/睁眼动态、眼周冻结、头部微动频谱和手部缺失。"""

    signal_id = "digital_human"
    display_name = "数字人专属特征"
    model_name = "digital-human-cv-heuristic"
    model_version = "2026-09"
    license = "internal"
    min_samples = 8

    def __init__(self, *, sample_fps: float | None = None) -> None:
        self.sample_fps = sample_fps
        self.eye_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )
        self.eye_detector_alt = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def analyze(self, context: PluginContext) -> SignalResult:
        started = time.perf_counter()
        face_analysis = context.face_analysis
        if face_analysis is None or len(face_analysis.samples) < self.min_samples:
            return self._unavailable("人脸样本少于 8 帧，无法计算数字人专属特征。")
        samples = list(face_analysis.samples)
        # 长视频受 max_frames 限制后，实际采样帧率可能低于配置值；
        # 频谱分析必须使用时间戳估计出的有效帧率，否则会把低频运动误判为高频。
        fps = _estimate_fps(samples) or self.sample_fps or 2.0
        observed_seconds = float(samples[-1].timestamp - samples[0].timestamp)
        if observed_seconds < MIN_OBSERVED_SECONDS:
            return self._unavailable(
                f"有效人脸时长不足 {MIN_OBSERVED_SECONDS:g} 秒"
                f"（当前 {observed_seconds:.2f} 秒），眨眼与微动统计不可靠。"
            )

        blink = _blink_features(samples, self.eye_detector, self.eye_detector_alt)
        motion = _head_micro_motion_features(samples, fps)
        hands = _hand_presence_features(samples)
        components = _feature_components(blink, motion, hands)
        logit = (
            -1.15 + 1.60 * components["eye"] + 1.35 * components["head"] + 0.80 * components["hand"]
        )
        score = float(1 / (1 + math.exp(-logit)))

        windows: list[SignalWindow] = []
        for start, end in blink["no_blink_windows"]:
            windows.append(
                SignalWindow(
                    start=float(start),
                    end=float(end),
                    score=0.62,
                    reason="该时间段持续未检测到眨眼，眼周动态可能被冻结。",
                )
            )

        return SignalResult(
            signal_id=self.signal_id,
            display_name=self.display_name,
            score=score,
            available=True,
            direction="指向生成" if score >= 0.5 else "中性",
            raw_value={
                "eye_visibility_ratio": blink["eye_visibility_ratio"],
                "blink_count": blink["blink_count"],
                "blink_rate_per_minute": blink["blink_rate_per_minute"],
                "eye_motion_median": blink["eye_motion_median"],
                "head_micro_motion_ratio": motion["high_band_ratio"],
                "head_frozen_ratio": motion["frozen_ratio"],
                "hand_presence_ratio": hands["presence_ratio"],
                "components": components,
                "observed_seconds": round(observed_seconds, 3),
                "sample_count": len(samples),
                "sample_fps": round(float(fps), 3),
                "confidence_note": (
                    "采样帧率低于眨眼持续时间时会漏检眨眼；侧脸、遮挡、眼镜和低分辨率同样会降低可靠性。"
                ),
            },
            explanation=(
                "综合眨眼/眼周动态、头部微动频谱和手部可见性给出数字人专属弱信号。"
                "纯口播真人视频同样可能长期不出现手部，因此该分数必须与其他层证据联合使用。"
            ),
            time_windows=windows,
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


def _estimate_fps(samples: list[FaceSample]) -> float | None:
    times = np.asarray([sample.timestamp for sample in samples], dtype=float)
    if times.size < 2:
        return None
    deltas = np.diff(times)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return None
    return float(1.0 / np.median(deltas))


def _blink_features(
    samples: list[FaceSample],
    primary,
    alternate,
) -> dict:
    visibility: list[bool | None] = []
    eye_bands: list[np.ndarray] = []
    for sample in samples:
        band = _eye_band(sample)
        if band is None:
            visibility.append(None)
            continue
        visibility.append(_eyes_visible(sample, primary, alternate))
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        eye_bands.append(
            (gray.astype(np.float32) - float(np.mean(gray))) / (float(np.std(gray)) + 1e-6)
        )

    judged = [value for value in visibility if value is not None]
    eye_visibility_ratio = float(np.mean(np.asarray(judged, dtype=float))) if judged else 0.0
    # 级联检测器完全找不到眼睛时，不能把“检测失败”当成“没有眨眼”。
    eye_detection_available = len(judged) >= 4 and eye_visibility_ratio >= 0.05

    durations = np.asarray([sample.timestamp for sample in samples], dtype=float)
    duration = float(durations[-1] - durations[0]) if durations.size > 1 else 0.0
    blink_events, no_blink_windows = _blink_events(samples, visibility)
    blink_rate = float(len(blink_events) / duration * 60.0) if duration >= 4.0 else 0.0
    eye_motion = _eye_motion(eye_bands)
    return {
        "eye_visibility_ratio": round(eye_visibility_ratio, 4),
        "blink_count": len(blink_events),
        "blink_rate_per_minute": round(blink_rate, 3),
        "eye_motion_median": round(eye_motion, 5),
        "no_blink_windows": no_blink_windows,
        "observed_seconds": round(duration, 3),
        "judged_frames": len(judged),
        "eye_detection_available": eye_detection_available,
    }


def _blink_events(
    samples: list[FaceSample],
    visibility: list[bool | None],
) -> tuple[list[float], list[tuple[float, float]]]:
    events: list[float] = []
    windows: list[tuple[float, float]] = []
    last_visible: float | None = None
    closed_since: float | None = None
    for sample, state in zip(samples, visibility, strict=True):
        if state is None:
            continue
        if state:
            if closed_since is not None and last_visible is not None:
                events.append(closed_since)
            closed_since = None
            last_visible = sample.timestamp
        elif closed_since is None:
            closed_since = sample.timestamp
    if last_visible is not None and closed_since is None:
        end = float(samples[-1].timestamp)
        if end - last_visible >= 6.0:
            windows.append((round(last_visible, 3), round(end, 3)))
    return events, windows


def _eye_band(sample: FaceSample) -> np.ndarray | None:
    face = sample.face_crop
    if face.size == 0 or face.shape[0] < 24 or face.shape[1] < 24:
        return None
    height, width = face.shape[:2]
    top = int(height * EYE_BAND_TOP)
    bottom = max(top + 1, int(height * EYE_BAND_BOTTOM))
    left = int(width * 0.08)
    right = max(left + 1, int(width * 0.92))
    band = face[top:bottom, left:right]
    return band if band.size else None


def _eyes_visible(sample: FaceSample, primary, alternate) -> bool:
    face = sample.face_crop
    if face.size == 0:
        return False
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    upper = gray[0 : max(1, int(height * 0.65)), :]
    min_size = max(8, int(min(upper.shape[:2]) * 0.12))
    for detector in (primary, alternate):
        if detector is None or detector.empty():
            continue
        eyes = detector.detectMultiScale(
            upper,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_size, min_size),
        )
        if len(eyes) >= 1:
            return True
    return False


def _eye_motion(eye_bands: list[np.ndarray]) -> float:
    if len(eye_bands) < 2:
        return 0.0
    values: list[float] = []
    for previous, current in zip(eye_bands, eye_bands[1:], strict=False):
        resized_previous = cv2.resize(previous, (24, 12), interpolation=cv2.INTER_AREA)
        resized_current = cv2.resize(current, (24, 12), interpolation=cv2.INTER_AREA)
        values.append(float(np.mean(np.abs(resized_current - resized_previous))))
    return float(np.median(values)) if values else 0.0


def _head_micro_motion_features(samples: list[FaceSample], fps: float) -> dict:
    centers = np.asarray(
        [
            (sample.bbox[0] + sample.bbox[2] / 2, sample.bbox[1] + sample.bbox[3] / 2)
            for sample in samples
        ],
        dtype=float,
    )
    scales = np.asarray([max(1.0, float(sample.bbox[2])) for sample in samples], dtype=float)
    normalized_x = centers[:, 0] / scales
    normalized_y = centers[:, 1] / scales
    log_area = np.log(
        np.asarray([sample.bbox[2] * sample.bbox[3] for sample in samples], dtype=float) + 1.0
    )

    times = np.asarray([sample.timestamp for sample in samples], dtype=float)
    seconds = float(times[-1] - times[0]) if times.size > 1 else 0.0
    deltas = np.diff(np.stack([normalized_x, normalized_y, log_area], axis=1), axis=0)
    if deltas.size == 0:
        return {
            "high_band_ratio": 0.0,
            "frozen_ratio": 1.0,
            "sample_count": len(samples),
            "observed_seconds": round(seconds, 3),
        }
    frozen_ratio = float(np.mean(np.linalg.norm(deltas[:, :2], axis=1) < 1.5e-3))
    high_band_ratio = _spectral_ratio(normalized_x, normalized_y, fps)
    return {
        "high_band_ratio": round(high_band_ratio, 4),
        "frozen_ratio": round(frozen_ratio, 4),
        "sample_count": len(samples),
        "observed_seconds": round(seconds, 3),
    }


def _spectral_ratio(x: np.ndarray, y: np.ndarray, fps: float) -> float:
    if x.size < 8 or fps <= 0:
        return 0.0
    signal = np.stack([_detrend(x), _detrend(y)], axis=1).reshape(-1)
    if float(np.std(signal)) < 1e-6:
        return 0.0
    window = np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(signal * window)) ** 2
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / fps)
    total = float(np.sum(spectrum)) + 1e-12
    band = (frequencies >= 0.15) & (frequencies <= max(0.2, fps / 2))
    if not np.any(band):
        return 0.0
    return float(np.sum(spectrum[band]) / total)


def _detrend(values: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return values - float(np.mean(values))
    index = np.arange(values.size, dtype=float)
    slope, intercept = np.polyfit(index, values, 1)
    return values - (slope * index + intercept)


def _hand_presence_features(samples: list[FaceSample]) -> dict:
    checks: list[bool] = []
    for sample in samples[:: max(1, len(samples) // 24)]:
        checks.append(_hand_like_blob_present(sample.frame, sample.bbox))
    presence = float(np.mean(np.asarray(checks, dtype=float))) if checks else 0.0
    return {"presence_ratio": round(presence, 4), "frames_checked": len(checks)}


def _hand_like_blob_present(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    mask = _skin_mask(frame)
    skin_ratio = float(np.count_nonzero(mask) / mask.size)
    if skin_ratio > 0.6:
        # 大面积肤色（例如近景或肤色背景）无法区分手部，跳过该帧。
        return False
    kernel = np.ones((5, 5), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(frame.shape[0] * frame.shape[1])
    x, y, width, height = bbox
    face_box = (
        x - int(width * 0.2),
        y - int(height * 0.2),
        x + width + int(width * 0.2),
        y + height + int(height * 0.2),
    )
    for contour in contours:
        area_ratio = cv2.contourArea(contour) / max(1.0, frame_area)
        if not HAND_AREA_RANGE[0] <= area_ratio <= HAND_AREA_RANGE[1]:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bh == 0:
            continue
        aspect = bw / bh
        if not 0.3 <= aspect <= 3.5:
            continue
        center_x = bx + bw / 2
        center_y = by + bh / 2
        if face_box[0] <= center_x <= face_box[2] and face_box[1] <= center_y <= face_box[3]:
            continue
        return True
    return False


def _skin_mask(frame: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(
        ycrcb,
        np.asarray([0, 133, 77], dtype=np.uint8),
        np.asarray([255, 173, 127], dtype=np.uint8),
    )
    return mask


def _feature_components(blink: dict, motion: dict, hands: dict) -> dict[str, float]:
    visibility = float(blink["eye_visibility_ratio"])
    blink_rate = float(blink["blink_rate_per_minute"])
    eye_motion = float(blink["eye_motion_median"])
    observed = float(blink["observed_seconds"])
    eye_detection_available = bool(blink.get("eye_detection_available", False))
    time_reliability = float(np.clip(observed / 12.0, 0.0, 1.0))

    low_visibility = 0.0
    if eye_detection_available:
        low_visibility = float(np.clip((0.35 - visibility) / 0.35, 0.0, 1.0))
    frozen_eyes = float(
        np.clip((MIN_UNFROZEN_EYE_MOTION - eye_motion) / MIN_UNFROZEN_EYE_MOTION, 0.0, 1.0)
    )
    missing_blink = 0.0
    if eye_detection_available and observed >= 8.0 and visibility >= 0.5:
        missing_blink = float(np.clip((6.0 - blink_rate) / 6.0, 0.0, 1.0))
    eye_component = max(low_visibility * 0.7, frozen_eyes * 0.85, missing_blink * 0.8) * (
        0.4 + 0.6 * time_reliability
    )

    head_reliability = float(
        np.clip(
            min(
                float(motion.get("observed_seconds", 0.0)) / 10.0,
                float(motion.get("sample_count", 0)) / 16.0,
            ),
            0.0,
            1.0,
        )
    )
    spectrum_component = float(np.clip((0.25 - float(motion["high_band_ratio"])) / 0.25, 0.0, 1.0))
    frozen_component = float(np.clip((float(motion["frozen_ratio"]) - 0.55) / 0.45, 0.0, 1.0))
    head_component = (0.5 * spectrum_component + 0.5 * frozen_component) * (
        0.3 + 0.7 * head_reliability
    )

    hand_reliability = float(np.clip(float(hands.get("frames_checked", 0)) / 12.0, 0.0, 1.0))
    hand_component = float(np.clip((0.15 - float(hands["presence_ratio"])) / 0.15, 0.0, 1.0)) * (
        0.4 + 0.6 * hand_reliability
    )
    return {
        "eye": round(eye_component, 4),
        "head": round(head_component, 4),
        "hand": round(hand_component, 4),
        "reliability": round(time_reliability, 4),
    }
