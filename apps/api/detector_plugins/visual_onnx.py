from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from apps.api.detector_plugins.base import PluginContext, SignalResult, SignalWindow
from apps.api.detector_plugins.visual_artifact import VisualArtifactPlugin

# CoreML 在这些导出的 ViT 模型上可能区分节点失败；记录后该进程内改用 CPU 会话。
_COREML_DISABLED_MODELS: set[str] = set()


class OnnxVisualArtifactPlugin:
    signal_id = "visual_artifact"
    display_name = "帧级视觉伪影"
    model_name = "onnx-visual-adapter"
    model_version = "external"
    license = "Apache-2.0"

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: int,
        source: str,
        license_name: str,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.source = source
        self.license = license_name
        self.model_name = source
        self.model_version = (
            f"size-{self.model_path.stat().st_size}" if self.model_path.exists() else "missing"
        )
        self._session = None
        self._cpu_session = None

    def analyze(self, context: PluginContext):
        started = time.perf_counter()
        base_result = VisualArtifactPlugin().analyze(context)
        if not base_result.signal.available or context.face_analysis is None:
            return base_result
        if not self.model_path.exists():
            signal = SignalResult(
                signal_id=self.signal_id,
                display_name=self.display_name,
                score=None,
                available=False,
                direction="中性",
                raw_value={"model_path": str(self.model_path)},
                explanation=(
                    f"配置的 ONNX 视觉模型不存在: {self.model_path}；"
                    "未回退到启发式分支。"
                ),
                model_name=self.model_name,
                model_version=self.model_version,
                license=self.license,
            )
            return base_result.__class__(
                signal=signal,
                heatmap_overlay=base_result.heatmap_overlay,
                representative_frame=base_result.representative_frame,
                beauty_filter_score=base_result.beauty_filter_score,
                beauty_filter_raw=base_result.beauty_filter_raw,
            )
        try:
            session = self._get_session()
            crops = [sample.crop for sample in context.face_analysis.samples]
            probabilities = self._predict(session, crops)
        except Exception as exc:  # noqa: BLE001
            signal = SignalResult(
                signal_id=self.signal_id,
                display_name=self.display_name,
                score=None,
                available=False,
                direction="中性",
                raw_value={"model_path": str(self.model_path), "error": str(exc)},
                explanation=f"ONNX 视觉模型加载或推理失败，已停用该分支: {exc}",
                model_name=self.model_name,
                model_version=self.model_version,
                license=self.license,
            )
            return base_result.__class__(
                signal=signal,
                heatmap_overlay=base_result.heatmap_overlay,
                representative_frame=base_result.representative_frame,
                beauty_filter_score=base_result.beauty_filter_score,
                beauty_filter_raw=base_result.beauty_filter_raw,
            )

        score = float(
            np.clip(
                0.65 * np.median(probabilities) + 0.35 * np.quantile(probabilities, 0.75),
                0,
                1,
            )
        )
        windows: list[SignalWindow] = []
        for sample, probability in zip(context.face_analysis.samples, probabilities, strict=False):
            if probability >= 0.68:
                windows.append(
                    SignalWindow(
                        start=max(0.0, sample.timestamp - 0.5),
                        end=sample.timestamp + 0.5,
                        score=float(probability),
                        reason="ONNX 帧级模型在该时间点给出较高生成概率。",
                    )
                )
        base_result.signal = SignalResult(
            signal_id=self.signal_id,
            display_name=self.display_name,
            score=score,
            available=True,
            direction="指向生成" if score >= 0.5 else "中性",
            raw_value={
                "frame_probability_median": float(np.median(probabilities)),
                "frame_probability_p75": float(np.quantile(probabilities, 0.75)),
                "frame_probability_max": float(np.max(probabilities)),
                "frame_count": len(probabilities),
                "source": self.source,
                "note": "热力图为高频残差代理，不是该 ONNX 模型的 Grad-CAM。",
            },
            explanation=(
                "对质量通过的人脸帧批量运行 ONNX 图像分类模型，聚合帧级生成概率。"
                "该模型并非专门在中文数字人视频上训练，跨生成器误差需要内部评测确认。"
            ),
            time_windows=windows,
            model_name=self.model_name,
            model_version=self.model_version,
            license=self.license,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return base_result

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
                # CoreML 初始化失败时直接退回 CPU，保证检测分支可用。
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

    def _predict(self, session, crops: list[np.ndarray]) -> np.ndarray:
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        probabilities: list[float] = []
        batch_size = 16
        for start in range(0, len(crops), batch_size):
            batch_crops = crops[start : start + batch_size]
            batch = np.stack([self._preprocess(crop) for crop in batch_crops], axis=0)
            try:
                outputs = session.run(None, {input_name: batch})
            except Exception:
                # CoreML 在部分节点或批量尺寸上失败：记住并回退到 CPU 会话重试。
                _COREML_DISABLED_MODELS.add(str(self.model_path))
                session = self._get_cpu_session()
                outputs = session.run(None, {input_name: batch})
            logits = np.asarray(outputs[0])
            if logits.ndim == 1 and logits.size == 2:
                logits = logits[None, :]
            elif logits.ndim == 1:
                logits = logits[:, None]
            if logits.ndim != 2 or logits.shape[-1] < 2:
                raise ValueError("ONNX 输出至少需要两个类别")
            logits = logits[:, :2]
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_values = np.exp(shifted)
            softmax = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            probabilities.extend(softmax[:, 1].astype(float).tolist())
        return np.asarray(probabilities, dtype=float)

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA,
        )
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        return np.transpose(normalized, (2, 0, 1))
