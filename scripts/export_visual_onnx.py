from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageClassification


def main() -> None:
    parser = argparse.ArgumentParser(
        description="导出 HuggingFace ViT 深伪图像模型为 ONNX；该步骤不在 FastAPI 主进程运行。"
    )
    parser.add_argument(
        "--model-id",
        default="dima806/deepfake_vs_real_image_detection",
    )
    parser.add_argument("--output", default="data/models/visual_deepfake.onnx")
    parser.add_argument("--metadata", default="data/models/visual_deepfake.json")
    args = parser.parse_args()

    model = AutoModelForImageClassification.from_pretrained(args.model_id)
    model.eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    metadata = {
        "source": args.model_id,
        "license": "Apache-2.0",
        "architecture": model.config.architectures[0] if model.config.architectures else "unknown",
        "input_size": 224,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "labels": model.config.id2label,
        "warning": "图像级模型不能替代视频时序模型，必须重新跑内部评测与校准。",
    }
    Path(args.metadata).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"ONNX: {output}")
    print(f"metadata: {args.metadata}")


if __name__ == "__main__":
    main()
