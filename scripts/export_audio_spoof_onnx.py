from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


class LogitsOnly(torch.nn.Module):
    """只导出分类 logits，避免 ONNX 输出里混入 AASIST 的中间隐状态。"""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:  # noqa: D102
        _, logits = self.model(waveform)
        return logits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="导出 AASIST 音频反欺骗模型为 ONNX；该步骤不在 FastAPI 主进程运行。"
    )
    parser.add_argument("--repo", default="third_party/AASIST")
    parser.add_argument("--config")
    parser.add_argument("--weights")
    parser.add_argument("--output", default="data/models/audio_aasist.onnx")
    parser.add_argument("--metadata", default="data/models/audio_aasist.json")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    if not repo.exists():
        raise SystemExit(f"AASIST 仓库不存在: {repo}")
    config_path = Path(args.config).expanduser() if args.config else repo / "config" / "AASIST.conf"
    weight_path = (
        Path(args.weights).expanduser()
        if args.weights
        else repo / "models" / "weights" / "AASIST.pth"
    )
    if not config_path.exists() or not weight_path.exists():
        raise SystemExit(f"缺少配置或权重: {config_path}, {weight_path}")

    sys.path.insert(0, str(repo))
    import models.AASIST as aasist_module  # type: ignore[import-not-found]

    # 官方仓库演变：旧版模型类名为 AASIST，当前 clovaai/aasist 为 Model。
    model_class = getattr(aasist_module, "AASIST", None) or getattr(
        aasist_module, "Model", None
    )
    if model_class is None:
        raise SystemExit("AASIST 仓库中未找到模型类（AASIST 或 Model）。")

    model_config = json.loads(config_path.read_text(encoding="utf-8"))["model_config"]
    model = model_class(model_config)
    checkpoint = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    sample_count = int(model_config.get("nb_samp", 64600))
    wrapper = LogitsOnly(model).eval()
    dummy = torch.randn(1, sample_count)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        output,
        input_names=["waveform"],
        output_names=["logits"],
        dynamic_axes={"waveform": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    metadata = {
        "source": "AASIST",
        "license": "MIT",
        "input_mode": "waveform",
        "sample_rate": 16000,
        "sample_count": sample_count,
        "segment_seconds": round(sample_count / 16000, 4),
        "labels": {"0": "bonafide", "1": "spoof"},
        "config": str(config_path),
        "weights": str(weight_path),
        "warning": (
            "AASIST 在英文 ASVspoof 数据上训练，中文 TTS/克隆音色必须重新跑内部评测与校准。"
        ),
    }
    Path(args.metadata).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"ONNX: {output}")
    print(f"metadata: {args.metadata}")


if __name__ == "__main__":
    main()
