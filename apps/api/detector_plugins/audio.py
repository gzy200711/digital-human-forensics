from __future__ import annotations

import math
import time

import numpy as np

from apps.api.detector_plugins.base import PluginContext, SignalResult, SignalWindow
from apps.api.services.media import decode_audio_mono


class AudioConsistencyPlugin:
    signal_id = "lip_sync"
    display_name = "唇音同步"
    model_name = "envelope-correlation-baseline"
    model_version = "bootstrap-1"
    license = "internal"

    def analyze(self, context: PluginContext) -> SignalResult:
        started = time.perf_counter()
        face_analysis = context.face_analysis
        if not context.probe.has_audio:
            return self._unavailable("输入不含音轨，未执行唇音同步。")
        if face_analysis is None or len(face_analysis.samples) < 6:
            return self._unavailable("可用人脸帧不足，未执行唇音同步。")

        try:
            audio = decode_audio_mono(
                context.probe.path,
                sample_rate=16000,
                max_duration_seconds=context.settings.max_media_duration_seconds,
            )
        except Exception as exc:
            return self._unavailable(f"音轨解码失败: {exc}")
        if audio.size < 1600:
            return self._unavailable("有效音轨过短，无法比较唇部运动。")

        audio_times, envelope = _audio_envelope(audio, sample_rate=16000)
        face_times = np.asarray([sample.timestamp for sample in face_analysis.samples])
        mouth_motion = _mouth_motion(face_analysis.samples)
        if mouth_motion.size < 4 or np.std(mouth_motion) < 1e-6:
            return self._unavailable("嘴部区域运动变化不足，无法稳定对齐音轨。")

        interpolated_envelope = np.interp(face_times, audio_times, envelope)
        correlation = _best_correlation(
            interpolated_envelope,
            mouth_motion,
            max_lag_seconds=0.8,
            frame_rate=float(1 / np.median(np.diff(face_times))) if len(face_times) > 1 else 2.0,
        )
        score = float(np.clip((0.42 - correlation) / 0.72, 0, 1))
        windows: list[SignalWindow] = []
        if score >= 0.58:
            windows.append(
                SignalWindow(
                    start=float(face_times[0]),
                    end=float(face_times[-1]),
                    score=score,
                    reason="整段唇部运动与音频包络相关度偏低。",
                )
            )
        return SignalResult(
            signal_id=self.signal_id,
            display_name=self.display_name,
            score=score,
            available=True,
            direction="指向生成" if score >= 0.5 else "中性",
            raw_value={
                "envelope_correlation": correlation,
                "audio_seconds": round(audio.size / 16000, 3),
                "face_sample_count": len(face_analysis.samples),
                "method": "mouth motion vs audio envelope cross-correlation",
            },
            explanation=(
                "比较嘴部像素运动与音频响度包络的时域相关性。"
                "该基线不能替代 SyncNet/AuViRe，侧脸、遮挡和音乐场景会失效。"
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


class AudioSpoofPlugin:
    signal_id = "audio_spoof"
    display_name = "音频反欺骗"
    model_name = "spectral-stability-heuristic"
    model_version = "bootstrap-1"
    license = "internal"

    def analyze(self, context: PluginContext) -> SignalResult:
        started = time.perf_counter()
        if not context.probe.has_audio:
            return SignalResult(
                signal_id=self.signal_id,
                display_name=self.display_name,
                score=None,
                available=False,
                direction="中性",
                explanation="输入不含音轨，未执行音频反欺骗。",
                model_name=self.model_name,
                model_version=self.model_version,
                license=self.license,
            )
        try:
            audio = decode_audio_mono(context.probe.path, sample_rate=16000)
        except Exception as exc:
            return SignalResult(
                signal_id=self.signal_id,
                display_name=self.display_name,
                score=None,
                available=False,
                direction="中性",
                explanation=f"音轨解码失败: {exc}",
                model_name=self.model_name,
                model_version=self.model_version,
                license=self.license,
            )
        if audio.size < 6400:
            return SignalResult(
                signal_id=self.signal_id,
                display_name=self.display_name,
                score=None,
                available=False,
                direction="中性",
                explanation="有效音轨不足 0.4 秒。",
                model_name=self.model_name,
                model_version=self.model_version,
                license=self.license,
            )

        _, envelope = _audio_envelope(audio, sample_rate=16000)
        envelope_cv = float(np.std(envelope) / (np.mean(envelope) + 1e-6))
        flatness, centroid = _spectral_features(audio)
        zero_crossing_std = _zero_crossing_variation(audio)
        flatness_component = np.clip((0.34 - flatness) / 0.34, 0, 1)
        envelope_component = np.clip((0.45 - envelope_cv) / 0.45, 0, 1)
        zcr_component = np.clip((0.065 - zero_crossing_std) / 0.065, 0, 1)
        logit = -0.85 + 1.25 * flatness_component + 1.2 * envelope_component + 0.7 * zcr_component
        score = float(1 / (1 + math.exp(-logit)))
        return SignalResult(
            signal_id=self.signal_id,
            display_name=self.display_name,
            score=score,
            available=True,
            direction="指向生成" if score >= 0.5 else "中性",
            raw_value={
                "spectral_flatness": flatness,
                "spectral_centroid_hz": centroid,
                "envelope_coefficient_of_variation": envelope_cv,
                "zero_crossing_rate_std": zero_crossing_std,
                "analyzed_seconds": round(audio.size / 16000, 3),
            },
            explanation=(
                "基于频谱平坦度、响度动态和过零率稳定性的弱启发式信号。"
                "音乐、降噪、电话编码和后期母带都可能造成误判。"
            ),
            model_name=self.model_name,
            model_version=self.model_version,
            license=self.license,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def _audio_envelope(
    audio: np.ndarray,
    *,
    sample_rate: int,
    frame_size: int = 512,
    hop_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    if audio.size < frame_size:
        return np.array([0.0]), np.array([float(np.sqrt(np.mean(audio**2)))])
    frame_count = 1 + (audio.size - frame_size) // hop_size
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(frame_count, frame_size),
        strides=(audio.strides[0] * hop_size, audio.strides[0]),
        writeable=False,
    )
    envelope = np.sqrt(np.mean(frames**2, axis=1))
    envelope = envelope / (np.max(envelope) + 1e-6)
    times = (np.arange(frame_count) * hop_size + frame_size / 2) / sample_rate
    return times, envelope.astype(float)


def _spectral_features(audio: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(20260918)
    chunk_size = 1024
    max_chunks = min(200, max(1, audio.size // chunk_size))
    starts = rng.integers(0, max(1, audio.size - chunk_size), size=max_chunks)
    flatness_values: list[float] = []
    centroid_values: list[float] = []
    window = np.hanning(chunk_size)
    for start in starts:
        chunk = audio[start : start + chunk_size]
        if chunk.size < chunk_size:
            continue
        spectrum = np.abs(np.fft.rfft(chunk * window)) + 1e-10
        power = spectrum**2
        flatness = float(np.exp(np.mean(np.log(power))) / (np.mean(power) + 1e-10))
        frequencies = np.fft.rfftfreq(chunk_size, d=1 / 16000)
        centroid = float(np.sum(frequencies * spectrum) / (np.sum(spectrum) + 1e-10))
        flatness_values.append(flatness)
        centroid_values.append(centroid)
    return (
        float(np.median(flatness_values)) if flatness_values else 0.5,
        float(np.median(centroid_values)) if centroid_values else 0.0,
    )


def _zero_crossing_variation(audio: np.ndarray) -> float:
    chunk_size = 1024
    rates = []
    for start in range(0, max(0, audio.size - chunk_size + 1), chunk_size * 2):
        chunk = audio[start : start + chunk_size]
        rates.append(float(np.mean(np.abs(np.diff(np.signbit(chunk).astype(np.int8))))))
    return float(np.std(rates)) if rates else 0.0


def _mouth_motion(samples: list) -> np.ndarray:
    values: list[float] = []
    previous: np.ndarray | None = None
    for sample in samples:
        x, y, width, height = sample.bbox
        mouth = sample.frame[
            y + int(height * 0.55) : y + height,
            x + int(width * 0.18) : x + int(width * 0.82),
        ]
        if mouth.size == 0:
            continue
        gray = np.mean(mouth, axis=2).astype(np.float32)
        gray = (gray - np.mean(gray)) / (np.std(gray) + 1e-6)
        if previous is None:
            values.append(0.0)
        else:
            resized_previous = np.resize(previous, gray.shape)
            difference = float(np.mean(np.abs(gray - resized_previous)))
            values.append(difference)
        previous = gray
    return np.asarray(values, dtype=float)


def _best_correlation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_lag_seconds: float,
    frame_rate: float,
) -> float:
    length = min(left.size, right.size)
    left = left[:length]
    right = right[:length]
    left = (left - np.mean(left)) / (np.std(left) + 1e-6)
    right = (right - np.mean(right)) / (np.std(right) + 1e-6)
    max_lag = max(0, int(max_lag_seconds * max(frame_rate, 0.1)))
    correlations: list[float] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = left[-lag:]
            b = right[: length + lag]
        elif lag > 0:
            a = left[: length - lag]
            b = right[lag:]
        else:
            a = left
            b = right
        if a.size < 4:
            continue
        correlations.append(float(np.mean(a * b)))
    if not correlations:
        return 0.0
    return float(np.clip(np.max(correlations), -1, 1))
