# 内部评测集数据卡

状态：待录入授权数据。

当前仓库不包含真人视频、生成视频或人脸数据集。`data/indexes/` 只保存索引和切分，不复制原始媒体。

## 必需字段

- `sample_id`
- `media_path`
- `label`
- `generator`
- `forgery_type`
- `source_person_id`
- `authorization_status`
- `duration_seconds`
- `width`
- `height`
- `frame_rate`
- `has_audio`
- `compression`
- `beauty_filter`
- `temporal_annotations`（可选）
- `spatial_annotations`（可选）

时间标注格式：

```json
{"start": 1.2, "end": 2.8}
```

空间标注格式：

```json
{"timestamp": 1.4, "x1": 120, "y1": 80, "x2": 360, "y2": 420}
```

`timestamp` 可省略，用于单帧或图像级样本；视频数据应优先提供时间戳，避免跨帧错误匹配。

## 目标覆盖

| 类别 | 目标数量 |
|---|---:|
| 真实视频 | 800–1500 |
| 生成视频 | 2000–4000 |
| 硬负样本 | 300–500 |

生成器至少覆盖：

- 全合成视频
- 数字人播报
- 换脸
- 口型驱动
- 音频克隆
- 组合伪造

硬负样本至少覆盖：

- 美颜滤镜
- 动漫滤镜
- 老片修复
- AI 超分
- 游戏 CG
- 视频通话录屏
- 虚拟背景
- 抠像合成

## ChatCut 专属生成器族

ChatCut 量产输出必须登记为独立生成器族（`generator=chatcut`），并保留变体维度：

- 形象 / avatar
- 音色 / voice
- 画面比例 / ratio
- 有无背景 / background

变体信息写入 `subset` 与 `prompt_or_driver`，切分时必须整族保留，禁止跨训练/测试拆分。
构建方式见 `scripts/build_chatcut_dataset.py`；与真人对照样本必须施加完全相同的重编码强度
（`dhfp augment-reencode`），否则模型会把编码强度当作类别捷径。

## 切分约束

- 测试生成器不参与训练。
- 同一来源人不能跨训练/验证/测试。
- 固定随机种子。
- 所有授权状态为 `pending` 或 `missing` 的真人样本自动标记为不可用。

## 已知偏差

在数据尚未录入前，不能声明任何性能水平。必须重点关注：

- 中文主播场景覆盖。
- 安卓前置摄像头和微信压缩链路。
- 暗光、侧脸、遮挡和多人场景。
- 最新生成器的时间漂移。
- 美颜与生成伪影的混淆。

## 不可用于

- 司法鉴定。
- 对个人身份或犯罪事实的裁决。
- 未授权人脸数据的训练。
- 对外传播生成样本。
