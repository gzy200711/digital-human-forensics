# API

Base URL：`http://localhost:8000`

完整 OpenAPI：`/docs`

## 提交任务

`POST /v1/jobs`

`multipart/form-data`：

| 字段 | 必填 | 说明 |
|---|---:|---|
| `file` | 二选一 | 视频或音频文件 |
| `source_url` | 二选一 | `http/https` 媒体 URL |
| `retention_days` | 否 | 1–365，默认 7 |
| `consent_ack` | 是 | 必须为 true |

返回 `202`：

```json
{
  "job_id": "uuid",
  "status": "queued",
  "status_url": "/v1/jobs/uuid",
  "report_url": "/v1/jobs/uuid/report"
}
```

## 查询状态

`GET /v1/jobs/{job_id}`

状态：

- `queued`
- `processing`
- `completed`
- `failed`

失败任务返回稳定的 `error_code` 和可读 `error_message`。

## 获取报告

`GET /v1/jobs/{job_id}/report`

报告尚未生成时返回 `409 REPORT_NOT_READY`。

报告顶层字段（schema `1.1`，仍兼容 `1.0` 旧报告）：

- `verdict`：`可信` / `无法确定` / `疑似生成合成`，结论带范围限定。
- `modality_verdicts`：`visual` / `audio` / `cross_modal` / `provenance` 四类模态级结论。
- `calibrated_probability`：`is_calibrated`、`calibration_group`、`distribution`、
  `confidence_interval`、`semantics`；未拟合校准时 `value` 不可解释为校准概率。
- `coverage`：`full_frame` / `face` / `audio` / `cross_modal`，无法估计时为 `null`。
- `gray_zone`、`threshold_policy`：灰区范围、阈值来源、目标 FPR 与真实阴性样本量证据状态。
- `abstention_reasons`、`abstention_codes`：`gray_zone`、`evidence_conflict`、
  `insufficient_corroboration`、`quality_marginal`、`quality_gate_failed`、`no_usable_modality`。
- `confidence`、`calibrated_score`：判定强度与未校准前的融合分数。
- `time_ranges`、`evidence`、`review`、`assessability`。
- `provenance`：与取证结果分开，含 `status`（阳性 / 无声明 / 未检查）与 `limitation`。
  `checks` 覆盖容器与流元数据、渲染链路指纹、C2PA / Content Credentials 与隐式水印；
  每项状态为 `阳性` / `阴性` / `未检查` / `不可用`，配置方式见 `docs/provenance.md`。
- `limitations`、`recommended_action`、`artifacts`。
- `model_versions`、`model_snapshot`、`audit_trace_id`：模型登记快照与审计追踪号。

Schema：`schemas/report.schema.json`

`evidence` 中每条记录带 `id`（证据 ID）、`layer` 字段（溯源层、帧内视觉层、数字人特征层、
时序层、音视频一致性层、音频层、干扰层、质量门控层）、`source`、`unit`、`time_range`、
`region`、`model`（名称 / 版本 / 许可证）、`uncertainty` 与 `limitations`。
`processing.layer_scores` 给出各层分数，`processing.modality_scores` 给出模态分数，
`processing.modality_divergence` 记录跨模态方向差异（不构成信号冲突）。

`review` 表示灰区证据复核。只有校准分数位于 `[gray_zone_min, gray_zone_max]`（默认 `[0.30, 0.70]`，由分位数标定生成）时才会触发。每条陈述包含 `claim`、`kind` 和 `evidence_ids`，`evidence_ids` 必须能在 `evidence` 中找到。

## 申诉

`POST /v1/jobs/{job_id}/appeal`

```json
{
  "reason": "不少于 10 个字符的申诉理由",
  "contact": "可选联系方式"
}
```

## 资源

`GET /v1/assets/{key}`

支持 `Range`，用于视频播放和定位。派生产物包括：

- 原始媒体
- `heatmap.png`
- `keyframe.jpg`
- `report.json`
- `report.html`

## 健康检查

`GET /healthz`

返回数据库、队列、存储和环境状态。
