from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from scripts.adapters._common import decode_audio_mono, emit, fail, media_parser, resolve_repo


def main() -> None:
    args = media_parser("AASIST 音频反欺骗适配器")
    repo = resolve_repo("AASIST_REPO", "third_party/AASIST")
    config_path = Path(
        os.environ.get("AASIST_CONFIG", str(repo / "config" / "AASIST.conf"))
    ).resolve()
    weight_path = Path(
        os.environ.get(
            "AASIST_WEIGHTS",
            str(repo / "models" / "weights" / "AASIST.pth"),
        )
    ).resolve()
    if not config_path.exists() or not weight_path.exists():
        fail(f"缺少 AASIST 配置或权重: {config_path}, {weight_path}")

    sys.path.insert(0, str(repo))
    import torch
    from models.AASIST import AASIST

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = config["model_config"]
    model = AASIST(model_config)
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    audio = np.asarray(decode_audio_mono(args.media, 16000), dtype=np.float32)
    segment_length = int(model_config.get("nb_samp", 64600))
    if audio.size < segment_length:
        audio = np.pad(audio, (0, segment_length - audio.size))
    segments = [
        audio[start : start + segment_length]
        for start in range(0, max(segment_length, audio.size), segment_length)
        if audio[start : start + segment_length].size == segment_length
    ][:8]
    if not segments:
        fail("有效音频长度不足")
    batch = torch.from_numpy(np.stack(segments)).float()
    with torch.no_grad():
        _, logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    emit(
        float(probabilities.max()),
        {
            "segments": len(segments),
            "mean_spoof_probability": float(probabilities.mean()),
            "max_spoof_probability": float(probabilities.max()),
            "weights": str(weight_path),
        },
    )


if __name__ == "__main__":
    main()
