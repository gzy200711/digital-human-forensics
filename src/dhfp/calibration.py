from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field


class ThresholdConfig(BaseModel):
    trusted_max: float = Field(default=0.30, ge=0, le=1)
    suspect_min: float = Field(default=0.70, ge=0, le=1)
    method: Literal["quantile", "manual"] = "quantile"
    trusted_quantile: float = Field(default=0.25, gt=0, lt=1)
    suspect_quantile: float = Field(default=0.75, gt=0, lt=1)
    calibration_count: int | None = Field(default=None, ge=0)
    real_negative_count: int | None = Field(default=None, ge=0)
    notes: str = ""


def quantile_thresholds(
    scores: np.ndarray,
    *,
    trusted_quantile: float = 0.25,
    suspect_quantile: float = 0.75,
    notes: str = "",
) -> ThresholdConfig:
    """用分数分位数确定灰区上下界，替代固定阈值。

    分位数在内部评测集上计算，数量不足或分布变化时必须重新拟合。
    """
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("scores 不能为空")
    if not 0 < trusted_quantile < suspect_quantile < 1:
        raise ValueError("分位数必须满足 0 < trusted_quantile < suspect_quantile < 1")
    trusted = float(np.clip(np.quantile(values, trusted_quantile), 0.0, 1.0))
    suspect = float(np.clip(np.quantile(values, suspect_quantile), 0.0, 1.0))
    if trusted >= suspect:
        suspect = float(np.clip(trusted + 1e-6, 0.0, 1.0))
    return ThresholdConfig(
        trusted_max=trusted,
        suspect_min=suspect,
        method="quantile",
        trusted_quantile=trusted_quantile,
        suspect_quantile=suspect_quantile,
        calibration_count=int(values.size),
        notes=notes or f"由 {values.size} 条校准分数按分位数生成。",
    )


class CalibrationProfile(BaseModel):
    version: str = "bootstrap-1"
    method: str = "temperature"
    temperature: float = Field(default=1.35, gt=0)
    bias: float = 0.0
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path) -> CalibrationProfile:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(payload)

    def calibrate(self, score: float) -> float:
        clipped = float(np.clip(score, 1e-6, 1 - 1e-6))
        logit = math.log(clipped / (1 - clipped))
        calibrated = 1 / (1 + math.exp(-((logit / self.temperature) + self.bias)))
        return float(np.clip(calibrated, 0, 1))


def fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bias: bool = False,
) -> CalibrationProfile:
    """用网格搜索拟合温度缩放。

    默认关闭 bias：当前评测集规模不足以稳定估计偏移项，
    负 bias 会把整体分数系统性地拉向“可信”，不适合作为默认行为。
    """
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    labels = np.asarray(labels, dtype=float)
    if probabilities.shape != labels.shape or probabilities.size == 0:
        raise ValueError("probabilities 与 labels 必须是非空且等长的数组")

    logits = np.log(probabilities / (1 - probabilities))
    best = (float("inf"), 1.0, 0.0)
    temperatures = np.concatenate(
        [
            np.linspace(0.25, 0.95, 15),
            np.linspace(1.0, 5.0, 41),
        ]
    )
    biases = np.linspace(-0.75, 0.75, 31) if bias else np.array([0.0])

    for temperature in temperatures:
        scaled = logits / temperature
        for offset in biases:
            calibrated = 1 / (1 + np.exp(-(scaled + offset)))
            loss = -np.mean(
                labels * np.log(np.clip(calibrated, 1e-8, 1))
                + (1 - labels) * np.log(np.clip(1 - calibrated, 1e-8, 1))
            )
            if loss < best[0]:
                best = (float(loss), float(temperature), float(offset))

    return CalibrationProfile(
        version="fitted",
        method="temperature",
        temperature=best[1],
        bias=best[2],
        notes=f"由带标签评分网格搜索得到；偏移项 {'启用' if bias else '禁用'}。",
    )
