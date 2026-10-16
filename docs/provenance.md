# 溯源核验：C2PA 与隐式水印

平台的溯源层包含三部分：容器 / 流元数据、C2PA（Content Credentials）清单、
隐式水印核验服务。前两项在本机完成，第三项需要外部服务凭据。

## C2PA：c2patool

本机安装（macOS）：

```bash
brew install c2patool
c2patool --version
```

首次使用需要拉取官方信任列表，否则签名证书会被标记为 `signingCredential.untrusted`：

```bash
c2patool init trust
# 写入 ~/.config/c2pa/c2pa-trust-list.pem，下次运行自动加载
```

手动核验单个文件：

```bash
c2patool outputs/final_720x1280_9000k.mp4 --detailed
```

平台侧的检查项名称是“C2PA / Content Credentials”：

- 找到清单：`阳性`，`details.summary` 带 `validation_state`、签发者、声明生成器和
  `digital_source_type`；命中 `trainedAlgorithmicMedia` 等生成类来源类型时，
  作为显式生成标识进入融合判定。
- 退出码非 0：`阴性`，表示未发现可解析清单，不证明内容真实。
- 未安装 `c2patool`：`不可用`，不是阴性。

Docker 部署时镜像会在 `amd64` 上自动安装 c2patool；官方没有 Linux `arm64`
构建，这类构建请把宿主机信任列表挂进容器：

```bash
docker run --rm -v ~/.config/c2pa:/root/.config/c2pa:ro ...
```

## 隐式水印：SynthID / 联盟核验服务

平台不内置 SynthID 检测算法。Google 的 SynthID Detector 只对已获权限的账号开放，
视频与音频没有可离线部署的公开检测权重；联盟（AIIA / 信通院等）水印核验同样需要
服务方账号。因此这一层实现为可配置的 HTTP 适配器：把媒体提交给已授权的核验端点，
再把返回结果归一化为平台证据。

### 配置项

| 变量 | 说明 |
| --- | --- |
| `WATERMARK_PROVIDER` | `off`（关闭）/ `auto`（按凭据自动选择）/ `synthid` / `alliance` |
| `WATERMARK_ENDPOINT` | 核验服务的完整 URL |
| `WATERMARK_API_KEY` | 认证密钥，默认以 `Authorization: Bearer <key>` 发送 |
| `WATERMARK_APP_ID` | 联盟服务分配的应用 ID |
| `WATERMARK_SECRET` | 联盟服务分配的应用密钥，用于请求签名 |
| `WATERMARK_MEDIA_FIELD` | 媒体字段名，默认 `file` |
| `WATERMARK_EXTRA_FIELDS` | 追加的表单字段，JSON 对象，例如 `{"business_id":"dhfp"}` |
| `WATERMARK_AUTH_HEADER` / `WATERMARK_AUTH_SCHEME` | 认证头与方案，默认 `Authorization` / `Bearer` |
| `WATERMARK_SIGN_REQUESTS` | 联盟服务是否按内置规则签名，默认 `true` |
| `WATERMARK_VERIFY_TLS` | 是否校验 TLS 证书，默认 `true` |
| `WATERMARK_TIMEOUT_SECONDS` | 单次核验超时，默认 30 秒 |
| `WATERMARK_ALLOW_MEDIA_UPLOAD` | 是否允许把原始媒体上传给外部服务，默认 `false` |

SynthID Detector 示例：

```dotenv
WATERMARK_PROVIDER=synthid
WATERMARK_ENDPOINT=https://<synthid-detector-endpoint>/v1/detect
WATERMARK_API_KEY=<api-key>
WATERMARK_MEDIA_FIELD=file
WATERMARK_ALLOW_MEDIA_UPLOAD=true
```

联盟水印核验示例：

```dotenv
WATERMARK_PROVIDER=alliance
WATERMARK_ENDPOINT=https://<vendor>/api/watermark/verify
WATERMARK_APP_ID=<app-id>
WATERMARK_SECRET=<app-secret>
WATERMARK_SIGN_REQUESTS=true
WATERMARK_ALLOW_MEDIA_UPLOAD=true
```

联盟适配器默认按常见约定提交 `app_id`、`timestamp`（毫秒）、`nonce`、
`media_sha256` 和 `sign`，其中
`sign = HMAC-SHA256(secret, app_id + timestamp + nonce + sha256(media))`。
服务方规则不同时，用 `WATERMARK_EXTRA_FIELDS` 追加字段；如果签名算法不一致，
需要改动 `apps/api/detector_plugins/watermark.py` 中的签名与字段映射。

### 响应契约

推荐服务方直接返回：

```json
{
  "detected": true,
  "confidence": 0.97,
  "watermark_id": "synthid-v1-8f2c",
  "details": {"model": "synthid-detector-2026"}
}
```

适配器同时兼容 `{"code":0,"data":{...}}`、`{"result":{...}}` 包装，
以及 `watermark_detected` / `is_watermarked` / `score` / `watermarkId` 等常见字段名。
以下情况一律输出 `不可用`，不会被当成阴性：

- 未配置端点或凭据；
- HTTP 4xx / 5xx、超时、TLS 失败；
- 响应不是 JSON、`success=false`、`code` 非成功值；
- 响应里找不到任何可识别的检出结论字段。

### 状态语义

| 状态 | 含义 | 对判定的影响 |
| --- | --- | --- |
| `阳性` | 核验服务检出隐式水印 | 作为显式生成合成标识，进入硬规则 |
| `阴性` | 服务正常返回，未检出 | 仅说明本次未检出，不作为真实性证据 |
| `未检查` | 未启用服务，或未开启媒体上传同意 | 不参与融合 |
| `不可用` | 服务未配置、报错或返回不可解析结果 | 不参与融合 |

## 合规边界

- 默认 `WATERMARK_ALLOW_MEDIA_UPLOAD=false`，平台不会把原始媒体送出本机；
  开启前必须确认素材授权、服务方数据处理条款与数据出境要求。
- 报告中只记录服务端点主机名、媒体 `sha256`、水印标识与分数，不记录套餐凭据。
- 未检出、服务不可用、清单被剥离都不能作为“内容真实”的证据。
- 阳性只说明存在生成合成标识，署名与责任归属仍需结合签发者信任链和操作记录核验。

## 本地核验命令

```bash
# 文本输出
uv run python scripts/verify_provenance.py sample.mp4

# 等价写法
make provenance-check FILE=sample.mp4

# 查看完整 JSON（含响应原文，便于排查字段映射）
uv run python scripts/verify_provenance.py sample.mp4 --json
```
