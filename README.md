# 数字人鉴别平台

证据优先的数字人 / 生成合成内容鉴别平台。项目不做二元真假判决，输出三级结论：

- `可信`
- `无法确定`
- `疑似生成合成`

元数据只作为阳性证据；质量不足、信号冲突或校准灰区都会降级为“无法确定”。当前仓库已实现阶段 0 的评测地基、阶段 1 的批量检测 MVP 闭环，以及阶段 2 的复核、定位、红队和持续学习代码接口。

## 已实现

后端：

- `POST /v1/jobs`：上传文件或提交媒体 URL。
- `GET /v1/jobs`：分页查询检测历史，支持文件名 / 任务 ID、状态和综合判定筛选。
- `GET /v1/jobs/{job_id}`：查询状态、进度和错误码。
- `GET /v1/jobs/{job_id}/report`：获取结构化报告。
- `POST /v1/jobs/{job_id}/appeal`：提交申诉并写入审计日志。
- `GET /v1/assets/{key}`：读取原始媒体、关键帧、热力图和报告产物，支持 Range 请求。
- `GET /healthz`：检查数据库、队列和存储配置。

检测层：

- L0：`ffprobe`、容器/流元数据、`c2patool`（C2PA / Content Credentials），
  以及可选的 SynthID / 联盟水印核验服务。
- L0 弱阳性：渲染链路指纹。容器/流标签命中 `comment` 中的 Remotion，或 `Lavf` muxer 与 Remotion 版本模式同时出现时，作为弱阳性参与融合，不触发硬规则（指纹可被重编码剥离或伪造）。
- L1：人脸检测、尺寸、清晰度、姿态、遮挡和压缩质量门控。
- L2：视觉伪影启发式基线；可切换到独立 ONNX Runtime。
- L2.5：数字人专属弱信号——眨眼/眼周动态、头部微动频谱、手部缺失。
- L3：人脸轨迹抖动、尺度漂移、直方图变化和亮度闪烁。
- L5：唇部运动/音频包络相关性和音频频谱稳定性基线；配置权重后切换为 ONNX 音频反欺骗（AASIST / WavLM 等）。
- L7：分层融合（层内加权平均 → 层间加权）、温度缩放（偏移项固定为 0）、分位数阈值、冲突降级和三级判定。
- 美颜/滤镜干扰作为独立分支：按背景复杂度归一化后，把视觉伪影分数拉向 0.5（中性），而不是拉向“真实”。
- 灰区证据复核：校准分数位于 `[gray_zone_min, gray_zone_max]`（默认 `[0.30, 0.70]`）时触发；默认规则提供器只复述现有证据，外部 L6 provider 的输出必须逐条引用有效证据 ID。
- 音频强证据升级：音频反欺骗综合分 ≥ `0.95` 且 ≥60% 分段不低于 `0.80` 时，按 `configs/fusion.json` 的 `audio_escalation` 规则升级综合判定为“疑似生成合成”（单模态规则，可在配置中关闭；升级原因写入报告摘要与 `processing.escalation_rule`）。
- 外部 L6：支持 OpenAI-compatible Chat Completions 风格的 HTTP provider；只发送证据元数据，不发送原始媒体。

前端：

- Next.js 提交页。
- 历史检测页：按提交时间查看任务状态、综合判定、判定强度与报告入口，并支持筛选和分页。
- 异步任务进度。
- 三级判定与带范围限定的结论说明。
- 模态级结论卡片：画面 / 音频 / 跨模态 / 溯源分别成像。
- 覆盖率条与人脸、音频、跨模态的实际分析比例。
- “无法确定”的拒判原因列表。
- 视频时间轴疑似区间。
- 人脸证据热力图叠加。
- 证据卡片（证据 ID、来源、时间范围、区域、模型版本、不确定度）、质量门控、溯源和局限声明。
- 灰区证据复核与证据引用展示。
- 阈值策略、模型快照与审计追踪号。
- 申诉入口。

报告契约（schema 1.1）：

- `modality_verdicts`：`visual` / `audio` / `cross_modal` / `provenance` 四类模态级结论，
  各自带 `score`、`coverage`、`status`、`signal_ids`、`summary` 与 `limitations`。
- `coverage`：`full_frame` / `face` / `audio` / `cross_modal`，无法估计时保留 `null`，不用 0 冒充阴性。
- `abstention_reasons` + `abstention_codes`：`gray_zone`、`evidence_conflict`、
  `insufficient_corroboration`、`quality_gate_failed` 等拒判原因。
- `calibrated_probability`：带 `is_calibrated`、`calibration_group`、`distribution`、
  `confidence_interval` 与语义说明；未拟合校准参数时明确标记为未校准。
- `gray_zone`、`threshold_policy`、`model_snapshot`、`audit_trace_id`：阈值来源、FPR 样本量
  证据状态、模型登记快照与审计追踪。
- `evidence[].evidence_id`（字段名 `id`）/ `source` / `unit` / `time_range` / `region` /
  `model` / `uncertainty` / `limitations`。

判定规则要点：

- 跨模态方向不同（例如音频疑似合成、画面未见异常）不判为信号冲突，只在各自模态内保留结论。
- 同一模态内、强度超过锁定阈值的分歧才标记 `evidence_conflict` 并降为“无法确定”。
- “疑似生成合成”要求至少两个独立信号越过阈值；单一信号越界时输出模态级提示并降为“无法确定”。

评测：

- 严格的数据索引 schema。
- generator-disjoint + identity-disjoint 切分。
- AUC、EER、Accuracy、TPR@FPR=1%、TPR@FPR=0.1%、ECE 和可靠性曲线。
- 时间片段与空间框的 IoU、AP@IoU 和片段级校准误差。
- 红队扰动：JPEG 压缩、降分辨率、录屏模拟、字幕烧录、调色和轻量噪声代理攻击。
- 持续学习：固定随机种子的经验回放清单，以及相对旧模型的 TPR、ECE、AUC 和定位 AP 回归门禁。
- 按生成器、伪造类型、分辨率和压缩条件分组。
- JSON、Markdown、PNG 和失败案例关键帧。
- DeMamba、GenConViT、AuViRe、AASIST 外部适配器注册表。

## 重要边界

当前默认视觉、时序、数字人特征和音频分数是用于打通系统的启发式基线，不能作为生产能力或司法鉴定结论。`configs/calibration.json` 也是 bootstrap 校准参数：偏移项固定为 0，灰区阈值按分数分位数确定，必须使用内部授权评测集重新拟合。

渲染链路指纹（Remotion + Lavf）和数字人专属特征都属于弱阳性证据：转码、剪辑软件、录制链路都可能改变或伪造容器标签，纯口播真人也会长期不出现手部。它们只参与融合，不能单独形成判定。

即使换成 ONNX 音频反欺骗模型，也必须在中文 TTS / 克隆音色上重新评测：AASIST 等公开权重主要在英文 ASVspoof 数据上训练，跨语种、跨编解码的性能需要内部数据确认。

项目中的 ONNX 适配器可用 Apache-2.0 的 `dima806/deepfake_vs_real_image_detection` 导出，但它只是图像级模型，不覆盖视频时序、换脸边界或中文数字人全部分布。正式上线前必须完成内部评测、校准和发布门禁。

阶段 2 的代码接口已经落地，但没有捆绑外部 MLLM、授权评测数据或梯度对抗攻击模型。启用外部复核时需要配置独立服务、提示词版本、超时与回退策略，并继续禁止模型生成证据链之外的事实。

## 本地启动

要求：

- Python 3.12
- Node.js 20+
- FFmpeg / ffprobe
- `c2patool`（可选，用于 C2PA 清单核验：`brew install c2patool`）
- `uv`（推荐）

安装：

```bash
make install
```

启动 API：

```bash
make run-api
```

启动前端：

```bash
make run-web
```

访问：

- 前端：http://localhost:3000
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/healthz

默认使用 SQLite、本地对象存储和进程内后台任务，适合开发。生产配置使用 PostgreSQL、Redis、Celery 和 MinIO。

## macOS 常驻运行

如果希望登录后自动启动、进程异常自动重启，并允许同一局域网内的手机或电脑访问，
可以安装本机常驻服务：

```bash
./scripts/platform-service.sh install
```

安装脚本会：

- 在缺少 `.env` 时生成随机 `SECRET_KEY` 和单机运行配置；
- 以生产模式构建 Next.js 前端；
- 安装 `com.dhfp.api`、`com.dhfp.web` 两个 `launchd` 服务；
- 默认安装 `com.dhfp.keepawake`，在接通电源时阻止 Mac 进入系统休眠。

访问地址：

- 本机：http://localhost:3000
- Windows / 局域网：先运行 `./scripts/platform-service.sh url` 获取当前地址，
  格式为 `http://<Mac 当前局域网 IP>:3000`
- Bonjour：`http://<Mac LocalHostName>.local:3000`，IP 变化时通常仍可使用
- 公网：运行 `./scripts/platform-service.sh public-url`，格式为
  `https://<随机名称>.trycloudflare.com`
- API 文档仅监听本机：http://127.0.0.1:8000/docs

Windows 上的 `localhost` 指向 Windows 自己，不能打开 Mac 服务。必须使用脚本输出的
“Windows / LAN”地址或 Bonjour 地址；若 Windows 开启了系统代理，请把该地址加入代理例外。

安装脚本会同时安装 `com.dhfp.tunnel` Cloudflare Tunnel 服务。公网地址自动启用 HTTP
Basic Auth，用户名和密码保存在权限为 `600` 的 `access-credentials.txt`。没有密码时，
公网请求返回 `401`；局域网和本机地址仍按原有方式访问。

Cloudflare Quick Tunnel 不保证永久域名，Mac 重启或隧道进程重建后网址可能变化，但普通
`platform-service.sh restart` 不会改变当前网址。需要永久固定域名时，应登录一次 Cloudflare
或 Tailscale 账号并使用 named tunnel / Tailscale Funnel。

服务管理：

```bash
./scripts/platform-service.sh status
./scripts/platform-service.sh restart
./scripts/platform-service.sh public-url
./scripts/platform-service.sh logs api
./scripts/platform-service.sh logs web
./scripts/platform-service.sh logs tunnel
./scripts/platform-service.sh keep-awake off
./scripts/platform-service.sh uninstall
```

`uninstall` 只移除自启动服务，不删除 `.env`、数据库、上传文件和报告。

说明：

- `launchd` 服务在用户登录后启动，日志写入 `logs/`。
- 断电、关机或合盖休眠期间服务无法访问；接通电源并保持系统唤醒可减少中断。
- 局域网内的其他设备可以打开页面，但后端端口不会单独暴露。当前平台没有用户登录，
  请不要将 3000 端口直接映射到公网。需要真正的公网随时访问时，应使用 Tailscale、
  Cloudflare Tunnel 或部署到云服务器，并先补上访问控制。

## Docker Compose

```bash
docker compose up --build
```

服务：

- `web`：Next.js
- `api`：FastAPI
- `worker`：Celery worker
- `postgres`：任务、报告、申诉和审计日志
- `redis`：Celery broker/result backend
- `minio`：原始媒体和派生产物

## 调用示例

提交任务：

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -F "file=@sample.mp4" \
  -F "retention_days=7" \
  -F "consent_ack=true"
```

查询状态：

```bash
curl http://localhost:8000/v1/jobs/<job_id>
```

获取报告：

```bash
curl http://localhost:8000/v1/jobs/<job_id>/report
```

提交申诉：

```bash
curl -X POST http://localhost:8000/v1/jobs/<job_id>/appeal \
  -H "Content-Type: application/json" \
  -d '{"reason":"素材来自本人授权拍摄，请复核时序证据。"}'
```

## 评测集索引

索引为 JSONL。每条记录至少包含：

```json
{
  "sample_id": "real-iphone-0001",
  "media_path": "/authorized/data/real-iphone-0001.mp4",
  "label": "real",
  "generator": "iPhone 15 Pro",
  "forgery_type": "none",
  "source_person_id": "person-001",
  "authorization_status": "verified",
  "duration_seconds": 12.4,
  "width": 1080,
  "height": 1920,
  "frame_rate": 30,
  "has_audio": true,
  "compression": "original",
  "beauty_filter": false,
  "subset": "real_mobile"
}
```

运行：

```bash
make dataset-check
make split
make eval
```

评测报告写入 `data/reports/`。如果没有真实索引，命令会明确报错，不会生成伪造的性能数字。

## ONNX 视觉模型

导出：

```bash
uv sync --extra export
uv run python scripts/export_visual_onnx.py \
  --model-id dima806/deepfake_vs_real_image_detection \
  --output data/models/visual_deepfake.onnx
```

启用：

```bash
export VISUAL_ONNX_PATH=data/models/visual_deepfake.onnx
```

FastAPI 主进程只使用 ONNX Runtime，不加载 PyTorch。

## 音频反欺骗（ONNX）

音频反欺骗是当前性价比最高的补充信号：TTS 或克隆音色会留下频谱与相位层面的痕迹，
而画面侧的启发式特征往往被压缩掩盖。AASIST 属于外部依赖，不随本仓库分发；
先按上游说明克隆仓库并获取权重，再导出 ONNX：

```bash
git clone https://github.com/clovaai/aasist.git third_party/AASIST
# 权重放到 third_party/AASIST/models/weights/AASIST.pth（见上游 README）
uv sync --extra export
uv run python scripts/export_audio_spoof_onnx.py \
  --repo third_party/AASIST \
  --output data/models/audio_aasist.onnx
```

启用：

```bash
export AUDIO_SPOOF_ONNX_PATH=data/models/audio_aasist.onnx
export AUDIO_SPOOF_INPUT_MODE=waveform   # 或 logmel
export AUDIO_SPOOF_AGGREGATION=max       # max / mean / trimmed
```

只要显式配置了模型路径，运行时就不会静默回退到启发式分支：模型缺失或推理失败会
返回“不可用”并写入报告，避免以弱启发式冒充实测模型结果。

## 数字人专属特征与分层融合

`digital_human` 分支用采样人脸帧估计：

- 眨眼次数与眼周动态（长时间无眨眼、眼周纹理冻结）；
- 头部微动频谱（0.15 Hz 以上的运动能量占比、冻结比例）；
- 手部可见性（脸框外肤色块的存在比例）。

这些特征都是弱信号：纯口播真人同样可能整段不出现手部，低采样帧率也会漏检眨眼，
因此它们只与其他层的证据联合使用。融合器先把同层信号加权平均，再按层权重合并，
避免同一层的相关信号重复计权；`processing.layer_scores` 输出各层分数。

时序分支只使用严格人脸框内的像素（`region=face_bbox`），并用框尺寸归一化头部运动，
背景变化和全局运镜不会再直接抬高时序分数。

## ChatCut 评测集与同强度重编码增广

把量产输出整理成独立生成器族的索引：

```bash
uv run python scripts/build_chatcut_dataset.py \
  --input data/raw/chatcut \
  --output data/indexes/chatcut.jsonl \
  --generator-version chatcut-v1 \
  --real-dir data/authorized/real
```

目录名可用 `avatar=a1,voice=v2/ratio=9x16/bg=on/` 形式组织，变体信息会写入
`subset` 与 `prompt_or_driver`，也支持用 `--manifest` 覆盖。切分时 ChatCut 必须作为
独立生成器族整族保留在测试集，否则会高估泛化能力。

训练前对全部标签施加完全相同的重编码强度，避免编码强度本身成为类别捷径：

```bash
uv run python -m dhfp.evaluation.cli augment-reencode \
  --index data/indexes/dataset.jsonl \
  --output-dir data/augmented \
  --output-index data/indexes/dataset_augmented.jsonl \
  --strengths crf23,crf28,crf32
```

命令会拒绝在不同标签间使用不同强度，并把 `reencode:<强度>` 写入 `compression` 便于分组评测。

## 溯源核验（C2PA / 隐式水印）

`c2patool` 负责本地 C2PA 清单核验，首次使用先拉取官方信任列表：

```bash
brew install c2patool
c2patool init trust
```

隐式水印（SynthID Detector / 联盟核验服务）属于外部服务，需要配置端点与凭据：

```dotenv
WATERMARK_PROVIDER=auto        # off / auto / synthid / alliance
WATERMARK_ENDPOINT=
WATERMARK_API_KEY=             # SynthID
WATERMARK_APP_ID=              # 联盟服务
WATERMARK_SECRET=              # 联盟服务
WATERMARK_ALLOW_MEDIA_UPLOAD=false
```

默认不把原始媒体上传给任何外部服务；开启 `WATERMARK_ALLOW_MEDIA_UPLOAD=true` 之前
需要确认素材授权与数据出境要求。服务未配置、报错或返回不可解析结果时，报告标注
“不可用”，不会被当成阴性。

核验单个文件：

```bash
uv run python scripts/verify_provenance.py sample.mp4
make provenance-check FILE=sample.mp4
```

完整参数、响应契约与排错见 `docs/provenance.md`。

## 外部基线

`configs/baselines.json` 注册了：

- DeMamba
- GenConViT
- AuViRe
- AASIST

这些项目的依赖、权重和预处理管线差异很大，因此采用独立命令适配器，不放进主进程。仓库和权重缺失时，评测报告会标注“不可用”，不会伪造分数。具体权重、许可证和接入状态见 `docs/model-licenses.md`。

## 测试

```bash
make test
make lint
```

## 目录

```text
apps/api       FastAPI、队列、检测插件、报告
apps/web       Next.js 提交页和报告页
src/dhfp       数据契约、切分、评测、基线插件
configs        融合、校准、基线注册表
schemas        JSON Schema
docs           架构、API、模型许可和数据卡
tests          单元与端到端测试
```

## 合规

- 不使用未授权人脸数据。
- 原始媒体默认 7 天删除。
- 报告统一使用“疑似生成合成内容”，不作“此人一定是 AI”的断言。
- 查看、提交和申诉均写审计日志。
- 元数据阴性不进入真实性证据。
- 不宣称司法鉴定结论；司法用途需与有资质机构合作。
