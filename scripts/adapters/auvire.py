from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from scripts.adapters._common import emit, fail, media_parser, resolve_repo


def main() -> None:
    args = media_parser("AuViRe 音视频定位适配器")
    repo = resolve_repo("AUVIRE_REPO", "third_party/AuViRe")
    config_path = Path(
        os.environ.get(
            "AUVIRE_CONFIG",
            str(
                repo
                / "ckpt"
                / "lavdf_b_avhubert_t_cnn_cnn_h_8_d_128_l_r2d2_w_15_o_subtraction_rl_r2d3u3s2_rm_av_aa_vv_f_True_conv_lr-_c_focal_diou_rec.json"
            ),
        )
    ).resolve()
    checkpoint_path = Path(
        os.environ.get(
            "AUVIRE_WEIGHTS",
            str(config_path).replace(".json", ".pth"),
        )
    ).resolve()
    if not config_path.exists() or not checkpoint_path.exists():
        fail(f"缺少 AuViRe 配置或权重: {config_path}, {checkpoint_path}")

    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from src.robustness import extract_visual_feature
    from src.training import Experiment

    configuration = json.loads(config_path.read_text(encoding="utf-8"))["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment = Experiment(cfg=configuration)
    model = experiment.get_model()
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    audio_features, video_features = extract_visual_feature(
        video_path=args.media,
        audio_path=args.media,
    )
    max_length = int(configuration["dataset"]["params"]["max_length"])
    length = min(video_features.shape[0], audio_features.shape[0], max_length)
    video_tensor = _pad_features(video_features[:length], max_length, device)
    audio_tensor = _pad_features(audio_features[:length], max_length, device)
    with torch.no_grad():
        outputs = model([video_tensor, audio_tensor])
    logit = max(output[0, :, 0].max(dim=-1)[0].cpu().item() for output in outputs[0])
    score = float(1 / (1 + np.exp(-logit)))
    emit(
        score,
        {
            "weights": str(checkpoint_path),
            "device": device,
            "feature_frames": length,
        },
    )


def _pad_features(features, max_length: int, device: str):
    import torch

    padding = torch.zeros(
        (max_length - features.shape[0], features.shape[1]),
        dtype=features.dtype,
    )
    return torch.cat([features.cpu(), padding]).unsqueeze(0).to(device)


if __name__ == "__main__":
    main()
