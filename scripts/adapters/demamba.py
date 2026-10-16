from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

from scripts.adapters._common import emit, fail, media_parser, resolve_repo


def main() -> None:
    args = media_parser("DeMamba 视频生成检测适配器")
    repo = resolve_repo("DEMAMBA_REPO", "third_party/DeMamba")
    config_path = Path(
        os.environ.get("DEMAMBA_CONFIG", str(repo / "configs" / "XCLIP_DeMamba.yaml"))
    ).resolve()
    weight_dir = Path(os.environ.get("DEMAMBA_WEIGHTS", "data/models/demamba")).resolve()
    weights = sorted(weight_dir.glob("*.pth")) if weight_dir.exists() else []
    if not config_path.exists() or not weights:
        fail(f"缺少 DeMamba 配置或权重: {config_path}, {weight_dir}/*.pth")

    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "models"))
    import torch
    import yaml
    from util import build_model

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = build_model(config["model"])
    checkpoint = torch.load(weights[0], map_location="cpu")
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    model.load_state_dict(state_dict, strict=False)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)
    model.eval()

    frames = _sample_frames(args.media, 8)
    tensor = torch.from_numpy(np.stack(frames)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor).reshape(-1)
        score = float(torch.sigmoid(logits[0]).cpu())
    emit(
        score,
        {
            "weights": str(weights[0]),
            "config": str(config_path),
            "device": device,
            "frame_count": len(frames),
        },
    )


def _sample_frames(path: str, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        fail(f"无法打开视频: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(0, total - 1), count, dtype=int)
    frames: list[np.ndarray] = []
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = capture.read()
        if not success:
            continue
        rgb = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - mean) / std
        frames.append(np.transpose(normalized, (2, 0, 1)))
    capture.release()
    if len(frames) < count:
        if not frames:
            fail(f"无法从视频读取有效帧: {path}")
        frames.extend([frames[-1].copy()] * (count - len(frames)))
    return frames


if __name__ == "__main__":
    main()
