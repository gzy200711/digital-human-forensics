# 运维与安全

## 原始文件留存

`RAW_RETENTION_DAYS` 控制默认留存期，任务也允许每条 1–365 天覆盖。

API 进程会按 `RETENTION_SWEEP_INTERVAL_SECONDS`（默认 3600 秒）执行留存清理。
过期任务的原始媒体、关键帧和热力图会被删除；结构化报告、静态报告和审计记录保留，
报告中的失效媒体链接会同步清空。排队或处理中的任务不会被删除，避免与分析进程竞争。

多实例部署时各实例可安全执行同一清理扫描；删除操作幂等，留存记录使用唯一主键去重。

## URL 下载

默认禁止访问私网、回环、链路本地和保留地址，并限制：

- 只允许 `http/https`
- 最多 6 次重定向
- 下载超时
- 最大文件大小
- 响应内容类型

URL 下载默认忽略进程的 `HTTP_PROXY` / `HTTPS_PROXY` 环境变量，并在建立连接后复核
实际对端地址，避免 DNS 重绑定或本地代理绕过私网限制。

如确需内网源站，显式设置 `ALLOW_PRIVATE_URLS=true`，并配合出口代理和域名白名单。

## 公网访问

macOS 常驻安装会启动 `com.dhfp.tunnel` Cloudflare Quick Tunnel。当前公网地址：

```bash
./scripts/platform-service.sh public-url
```

公网请求必须通过 HTTP Basic Auth；凭据写入 `access-credentials.txt`（权限 `600`）。
局域网和 loopback 地址不触发公网认证，保持原有使用方式。

Quick Tunnel 不提供永久 SLA，Mac 重启或隧道进程重建后域名会变化。需要固定域名时，
应改用 Cloudflare named tunnel 或 Tailscale Funnel，并继续保留公网认证与最小暴露原则。

## 密钥

生产环境必须替换：

- `SECRET_KEY`
- PostgreSQL 密码
- Redis 密码
- MinIO/S3 密钥

不要使用 compose 中的开发默认值。

## 灰区 L6 Provider

默认 `REVIEW_PROVIDER_MODE=rules`，不会发起外部请求。接入 OpenAI-compatible
Chat Completions 服务时设置：

```bash
export REVIEW_PROVIDER_MODE=http
export REVIEW_PROVIDER_ENDPOINT=https://provider.example/v1/chat/completions
export REVIEW_PROVIDER_API_KEY=...
export REVIEW_PROVIDER_MODEL=...
export REVIEW_PROMPT_VERSION=evidence-review-v1
```

Provider 只接收证据 ID、名称、方向、强度、分数和可鉴定性状态，不接收原始媒体、
人脸图像或报告中的联系方式。请求失败、超时或输出不合规时，报告标记为
`unavailable`，不会用自由文本补足缺失证据。

## 数据保护

- 当前本地模式使用文件系统明文保存原始媒体和派生产物；生产环境必须启用
  S3 / PostgreSQL / 备份介质的静态加密，并收紧文件权限与密钥轮换。
- 人脸区域属于敏感个人信息，只在必要时间内保留；留存清理会删除原媒体、关键帧和热力图。
- 当前版本没有用户登录和短期签名下载；只能部署在可信内网，不能把 3000 端口直接暴露到公网。
- 审计日志当前写入业务数据库，部署时应同步到独立、不可篡改的归档，并保留至少 6 个月。
- 默认不把媒体发送给外部模型；启用水印核验或外部复核前必须完成授权、数据出境和供应商评估。

## 发布门禁

建议在 CI/CD 强制：

- 真实人像集 FPR ≤ 2%
- 未见生成器 TPR@FPR=1% ≥ 0.65
- ECE ≤ 0.06
- generator 与 identity 双重隔离审计通过
- 平凡元数据基线未被模型显著超过时禁止发布

当前 bootstrap 版本不满足生产发布门禁。

## 红队与持续学习

生成红队媒体：

```bash
uv run dhfp redteam \
  --input sample.mp4 \
  --output data/redteam/sample
```

构建经验回放清单：

```bash
uv run dhfp replay \
  --index data/indexes/dataset.jsonl \
  --splits data/indexes/splits.json \
  --output data/reports/replay.json
```

比较候选模型和已发布模型：

```bash
uv run dhfp regression-gate \
  --previous data/reports/released.json \
  --candidate data/reports/candidate.json \
  --detector fusion-pipeline \
  --output data/reports/gate.json
```

门禁失败时命令返回非零状态，适合直接接入 CI/CD。
