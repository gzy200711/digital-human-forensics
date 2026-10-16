from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from apps.api.detector_plugins.base import PluginContext, SignalResult, SignalWindow
from apps.api.services.media import decode_audio_mono

# CoreML 在部分导出模型上可能推理失败；记录后该进程内改用 CPU 会话。
_COREML_DISABLED_MODELS: set[str] = set()


class OnnxAudioSpoofPlugin:
    """基于 ONNX 的音频反欺骗分支（AASIST / WavLM 等）。"""

    signal_id = "audio_spoof"
    display_name = "音频反欺骗"
    model_name = "onnx-audio-antispoof"
    license = "check-upstream"

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_mode: str = "waveform",
        sample_rate: int = 16000,
        segment_seconds: float = 4.04,
        max_segments: int = 8,
        aggregation: str = "max",
        source: str = "AASIST",
        license_name: str = "MIT",
    ) -> None:
        self.model_path = Path(model_path)
        self.input_mode = input_mode.strip().lower()
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.max_segments = max_segments
        self.aggregation = aggregation.strip().lower()
        self.source = source
        self.license = license_name
        self.model_name = source or self.model_name
        self.model_version = (
            f"size-{self.model_path.stat().st_size}" if self.model_path.exists() else "missing"
        )
        self._session = None
        self._cpu_session = None

    def analyze(self, context: PluginContext) -> SignalResult:
        started = time.perf_counter()
        if not context.probe.has_audio:
            return self._unavailable("输入不含音轨，未执行音频反欺骗。")
        if not self.model_path.exists():
            return self._unavailable(
                f"配置的音频反欺骗模型不存在: {self.model_path}；未回退到启发式分支。"
            )
        if self.input_mode not in {"waveform", "logmel"}:
            return self._unavailable(f"不支持的音频输入模式: {self.input_mode}")
        try:
            audio = decode_audio_mono(
                context.probe.path,
                sample_rate=self.sample_rate,
                max_duration_seconds=context.settings.max_media_duration_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            return self._unavailable(f"音轨解码失败: {exc}")
        if audio.size < int(self.sample_rate * 0.4):
            return self._unavailable("有效音轨不足 0.4 秒。")

        try:
            session = self._get_session()
            batch, segment_starts = self._prepare(audio)
            probabilities = self._predict(session, batch)
        except Exception as exc:  # noqa: BLE001
            return self._unavailable(f"ONNX 音频模型加载或推理失败: {exc}")

        score = _aggregate(probabilities, self.aggregation)
        windows = [
            SignalWindow(
                start=max(0.0, start),
                end=max(0.0, start) + self.segment_seconds,
                score=float(probability),
                reason="该音频片段被反欺骗模型判为合成/克隆音色。",
            )
            for start, probability in zip(segment_starts, probabilities, strict=True)
            if probability >= 0.7
        ]
        return SignalResult(
            signal_id=self.signal_id,
            display_name=self.display_name,
            score=score,
            available=True,
            direction="指向生成" if score >= 0.5 else "中性",
            raw_value={
                "aggregation": self.aggregation,
                "segment_count": int(probabilities.size),
                "segment_probabilities": [round(float(value), 4) for value in probabilities],
                "mean_probability": float(np.mean(probabilities)),
                "max_probability": float(np.max(probabilities)),
                "input_mode": self.input_mode,
                "sample_rate": self.sample_rate,
                "segment_seconds": self.segment_seconds,
                "analyzed_seconds": round(
                    float(probabilities.size * self.segment_seconds), 3
                ),
                "source": self.source,
                "model_path": str(self.model_path),
                "note": "分数为模型输出的合成语音概率；短音频与音乐场景需要单独评测。",
            },
            explanation=(
                "对音轨分段运行 ONNX 音频反欺骗模型，聚合各段合成语音概率。"
                "TTS 与克隆音色是该分支的主要目标，音乐、强降噪和电话编码会增加误差。"
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
            raw_value={
                "model_path": str(self.model_path),
                "input_mode": self.input_mode,
                "source": self.source,
            },
            explanation=reason,
            model_name=self.model_name,
            model_version=self.model_version,
            license=self.license,
        )

    def _prepare(self, audio: np.ndarray) -> tuple[np.ndarray, list[float]]:
        if self.input_mode == "waveform":
            segments, starts = segment_waveform(
                audio,
                sample_rate=self.sample_rate,
                segment_seconds=self.segment_seconds,
                max_segments=self.max_segments,
            )
            return segments.astype(np.float32), starts
        features, starts = logmel_segments(
            audio,
            sample_rate=self.sample_rate,
            segment_seconds=self.segment_seconds,
            max_segments=self.max_segments,
        )
        return features.astype(np.float32), starts

    def _get_session(self):
        if self._session is None:
            import onnxruntime as ort

            model_key = str(self.model_path)
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            if model_key in _COREML_DISABLED_MODELS:
                providers = ["CPUExecutionProvider"]
            available = set(ort.get_available_providers())
            selected = [provider for provider in providers if provider in available]
            try:
                self._session = ort.InferenceSession(
                    str(self.model_path),
                    providers=selected or ["CPUExecutionProvider"],
                )
            except Exception:
                _COREML_DISABLED_MODELS.add(model_key)
                self._session = self._get_cpu_session()
        return self._session

    def _get_cpu_session(self):
        if self._cpu_session is None:
            import onnxruntime as ort

            self._cpu_session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
        return self._cpu_session

    def _predict(self, session, batch: np.ndarray) -> np.ndarray:
        input_name = session.get_inputs()[0].name
        try:
            outputs = session.run(None, {input_name: batch})
            return self._logits_to_probabilities(outputs)
        except Exception:
            pass
        # 导出的 AASIST 模型内部 Reshape 固定了批量维度，批量推理会失败：
        # 改用 CPU 会话逐段（batch=1）推理。该分支禁用 CoreML 避免重复失败。
        _COREML_DISABLED_MODELS.add(str(self.model_path))
        cpu_session = self._get_cpu_session()
        cpu_input = cpu_session.get_inputs()[0].name
        probabilities: list[float] = []
        for row in batch:
            sample = np.asarray(row, dtype=np.float32)[None, :]
            outputs = cpu_session.run(None, {cpu_input: sample})
            probabilities.append(float(self._logits_to_probabilities(outputs)[0]))
        return np.asarray(probabilities, dtype=float)

    @staticmethod
    def _logits_to_probabilities(outputs) -> np.ndarray:
        raw = np.asarray(outputs[0])
        if raw.ndim == 1 and raw.size == 2:
            values = raw.reshape(1, -1)
        elif raw.ndim == 1:
            values = raw
        else:
            values = raw.reshape(raw.shape[0], -1)
        if values.ndim == 2 and values.shape[1] >= 2:
            shifted = values[:, :2] - np.max(values[:, :2], axis=1, keepdims=True)
            exponentials = np.exp(shifted)
            return (exponentials / np.sum(exponentials, axis=1, keepdims=True))[:, 1]
        if values.ndim == 2:
            values = values[:, 0]
        if float(np.min(values)) < 0.0 or float(np.max(values)) > 1.0:
            values = 1.0 / (1.0 + np.exp(-values))
        return np.clip(values.astype(float), 0.0, 1.0)


def segment_waveform(
    audio: np.ndarray,
    *,
    sample_rate: int = 16000,
    segment_seconds: float = 4.04,
    max_segments: int = 8,
) -> tuple[np.ndarray, list[float]]:
    """按固定长度切分波形；最后一段补零，返回 (batch, samples) 与起始时间。"""
    length = max(1, int(round(segment_seconds * sample_rate)))
    if audio.size == 0:
        return np.zeros((1, length), dtype=np.float32), [0.0]
    if audio.size <= length:
        padded = np.pad(audio.astype(np.float32), (0, length - audio.size))
        return padded[None, :], [0.0]

    segments: list[np.ndarray] = []
    starts: list[float] = []
    for start in range(0, audio.size, length):
        if len(segments) >= max_segments:
            break
        chunk = audio[start : start + length]
        if chunk.size < length:
            chunk = np.pad(chunk, (0, length - chunk.size))
        segments.append(chunk.astype(np.float32))
        starts.append(start / sample_rate)
    return np.stack(segments, axis=0), starts


def logmel_segments(
    audio: np.ndarray,
    *,
    sample_rate: int = 16000,
    segment_seconds: float = 4.04,
    max_segments: int = 8,
    n_mels: int = 80,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> tuple[np.ndarray, list[float]]:
    """切分为片段并计算 log-mel 特征，返回 ((batch, frames, mels)) 与起始时间。"""
    waveforms, starts = segment_waveform(
        audio,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        max_segments=max_segments,
    )
    features = [
        _logmel(waveform, sample_rate=sample_rate, n_mels=n_mels, frame_ms=frame_ms, hop_ms=hop_ms)
        for waveform in waveforms
    ]
    width = min(feature.shape[0] for feature in features)
    stacked = np.stack([feature[:width] for feature in features], axis=0)
    return stacked.astype(np.float32), starts


def _logmel(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    n_mels: int,
    frame_ms: float,
    hop_ms: float,
) -> np.ndarray:
    frame_length = max(1, int(round(sample_rate * frame_ms / 1000)))
    hop_length = max(1, int(round(sample_rate * hop_ms / 1000)))
    if waveform.size < frame_length:
        waveform = np.pad(waveform, (0, frame_length - waveform.size))
    frame_count = 1 + (waveform.size - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(frame_count, frame_length),
        strides=(waveform.strides[0] * hop_length, waveform.strides[0]),
        writeable=False,
    )
    window = np.hanning(frame_length)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    filters = _mel_filterbank(sample_rate, frame_length, n_mels)
    mel = spectrum @ filters.T
    log_mel = np.log(np.maximum(mel, 1e-10))
    return ((log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)).astype(np.float32)


def _mel_filterbank(sample_rate: int, frame_length: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(frequency: float) -> float:
        return 2595.0 * math.log10(1.0 + frequency / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    fft_frequencies = np.fft.rfftfreq(frame_length, d=1.0 / sample_rate)
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(sample_rate / 2), n_mels + 2)
    hz_points = np.asarray([mel_to_hz(point) for point in mel_points])
    filters = np.zeros((n_mels, fft_frequencies.size), dtype=np.float32)
    for index in range(n_mels):
        lower, center, upper = hz_points[index : index + 3]
        if center <= lower or upper <= center:
            continue
        left = (fft_frequencies - lower) / (center - lower)
        right = (upper - fft_frequencies) / (upper - center)
        filters[index] = np.maximum(0.0, np.minimum(left, right))
    return filters


def _aggregate(probabilities: np.ndarray, mode: str) -> float:
    values = np.asarray(probabilities, dtype=float)
    if values.size == 0:
        return 0.0
    if mode == "mean":
        return float(np.mean(values))
    if mode == "trimmed":
        ordered = np.sort(values)[len(values) // 2 :]
        return float(np.mean(ordered)) if ordered.size else float(np.max(values))
    return float(np.max(values))
