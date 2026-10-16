# 模型与许可证

本表记录代码接入状态和上游许可证。许可证会随上游版本变化，正式使用前必须锁定 commit / revision 并再次核验。

| 组件 | 用途 | 上游许可证 | 当前状态 |
|---|---|---|---|
| FFmpeg / ffprobe | 解码、媒体探针、音轨提取 | LGPL-2.1+ / GPL 组件取决于构建 | 已作为系统依赖接入 |
| c2patool | C2PA / Content Credentials | Apache-2.0 / MIT（按上游组件） | 可选，未安装时明确标记不可用 |
| OpenCV | 人脸检测、质量与图像特征 | Apache-2.0 | 已接入 |
| dima806/deepfake_vs_real_image_detection | 图像级真假分类 | Apache-2.0 | 可导出 ONNX；不是视频模型 |
| DeMamba | AI 生成视频检测 | Apache-2.0 | 外部命令适配器；未捆绑权重 |
| GenConViT (`erprogs/GenConViT`) | 换脸视频检测 | MIT | 外部命令适配器；未捆绑权重 |
| AuViRe | 音视频重构残差与时间定位 | 上游仓库许可证需锁定核验 | 外部命令适配器；需要 AVHubert 特征管线 |
| AASIST | 音频反欺骗 | MIT | 外部命令适配器与 ONNX 主进程分支（`scripts/export_audio_spoof_onnx.py`）；权重由上游仓库提供 |
| 规则证据摘要 | 灰区证据复核默认提供器 | 项目内实现 | 已接入；只复述已有证据 |
| 外部 L6 MLLM | 灰区工具调用式复核 | 取决于具体模型 | OpenAI-compatible HTTP 适配器已实现；未捆绑权重 |
| InsightFace / SCRFD | 人脸检测/关键点（生产建议） | 按具体模型，常为非商业限制 | 当前 MVP 未捆绑 |
| ByteTrack | 多目标跟踪（生产建议） | MIT | 当前 MVP 未捆绑 |

## 选择原则

1. 不在主推理进程加载 PyTorch。
2. 不把上游“可运行代码”误写成“已下载权重”。
3. 不把研究许可证模型用于商业服务，除非许可证明确允许。
4. 不把图像级模型包装成视频时序模型。
5. 每个模型记录来源、revision、权重哈希、预处理和后处理版本。

## ONNX 图像分支

导出命令：

```bash
uv sync --extra export
uv run python scripts/export_visual_onnx.py \
  --model-id dima806/deepfake_vs_real_image_detection \
  --output data/models/visual_deepfake.onnx
```

该模型训练在图像真假分类任务上，不能覆盖：

- 唇形驱动数字人。
- 音视频不同步。
- 实时通话注入。
- 音频克隆。
- 新生成器跨域泛化。

## ONNX 音频分支

导出命令：

```bash
uv sync --extra export
uv run python scripts/export_audio_spoof_onnx.py \
  --repo third_party/AASIST \
  --weights third_party/AASIST/models/weights/AASIST.pth \
  --output data/models/audio_aasist.onnx
```

通过 `AUDIO_SPOOF_ONNX_PATH` 启用。启用后主进程只使用 ONNX Runtime，不加载 PyTorch；
模型缺失或推理失败会在报告中标记“不可用”，不会静默回退到启发式分支。

公开 AASIST 权重主要在英文 ASVspoof 数据上训练，不能直接覆盖中文 TTS、克隆音色、
音乐和强降噪场景，正式使用前必须用内部授权数据重新评测与校准。

## 外部基线

外部基线通过 `configs/baselines.json` 注册。缺少仓库或权重时，评测 runner 会把该检测器标记为不可用并写入原因，不会用默认常数替代模型输出。
