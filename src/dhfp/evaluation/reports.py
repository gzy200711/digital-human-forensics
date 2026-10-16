from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_reliability_curve(
    curve: dict[str, list[float]],
    *,
    detector_name: str,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.2, 5.2), dpi=160)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#98a2b3", linewidth=1.5, label="完美校准")
    axis.plot(
        curve["confidence"],
        curve["observed_positive_rate"],
        marker="o",
        markersize=4,
        linewidth=1.8,
        color="#175cd3",
        label="观测正例率",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("平均预测风险")
    axis.set_ylabel("实际正例率")
    axis.set_title(f"{detector_name} 可靠性曲线")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def render_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# 数字人鉴别平台阶段 0 评测报告",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 数据集记录数：{payload['dataset_summary']['total']}",
        f"- 可用记录数：{payload['dataset_summary']['usable']}",
        f"- 测试集规模：{payload['split_counts'].get('test', 0)}",
        "- 目标操作点：TPR@FPR=1%",
        "",
        "> 本报告只描述当前内部索引上的实测结果。索引缺失、样本不足或生成器未覆盖时，不得把结果外推为生产能力。",
        "",
        "## 核心结论",
        "",
        "| 检测器 | 许可证 | 状态 | AUC | EER | ACC | TPR@FPR=1% | TPR@FPR=0.1% | ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _detector_id, record in payload["detectors"].items():
        metrics = record.get("metrics")
        if not metrics:
            lines.append(
                f"| {record['display_name']} | {record['license']} | 不可用 | - | - | - | - | - | - |"
            )
            continue
        lines.append(
            "| {name} | {license} | 已评估 | {auc:.3f} | {eer:.3f} | {acc:.3f} | "
            "{tpr1:.3f} | {tpr01:.3f} | {ece:.3f} |".format(
                name=record["display_name"],
                license=record["license"],
                auc=metrics["auc"],
                eer=metrics["eer"],
                acc=metrics["accuracy"],
                tpr1=metrics["tpr_at_fpr_1pct"],
                tpr01=metrics["tpr_at_fpr_0.1pct"],
                ece=metrics["ece"],
            )
        )

    lines.extend(["", "## 审计控制", ""])
    for item in payload["audits"]:
        marker = "通过" if item["passed"] else "未通过"
        lines.append(f"- **{item['control']} / {marker}**：{item['details']}")

    lines.extend(["", "## 分组性能", ""])
    for _detector_id, record in payload["detectors"].items():
        if not record.get("groups"):
            continue
        lines.extend([f"### {record['display_name']}", ""])
        lines.append("| 分组维度 | 分组 | 样本数 | TPR@FPR=1% | AUC |")
        lines.append("|---|---|---:|---:|---:|")
        for dimension, groups in record["groups"].items():
            for group, metrics in groups.items():
                if metrics.get("insufficient_classes"):
                    lines.append(f"| {dimension} | {group} | {metrics['count']} | 类别不足 | - |")
                else:
                    lines.append(
                        f"| {dimension} | {group} | {metrics['count']} | "
                        f"{metrics['tpr_at_fpr_1pct']:.3f} | {metrics['auc']:.3f} |"
                    )
        lines.append("")

    lines.extend(["## 失败案例抽样", ""])
    for _detector_id, record in payload["detectors"].items():
        failures = record.get("failure_cases", [])
        if not failures:
            continue
        lines.append(f"### {record['display_name']}")
        for failure in failures:
            lines.append(
                f"- `{failure['sample_id']}` / {failure['generator']} / "
                f"{failure['failure_type']} / score={failure['score']:.3f} / "
                f"局部图：`{failure.get('thumbnail', '未生成')}`"
            )
        lines.append("")

    lines.extend(["## 定位评测", ""])
    has_localization = False
    for _detector_id, record in payload["detectors"].items():
        localization = record.get("localization")
        if not localization:
            continue
        has_localization = True
        lines.append(f"### {record['display_name']}")
        for dimension, metrics in localization.items():
            label = "时间定位" if dimension == "temporal" else "空间定位"
            lines.append(
                f"- **{label}**：AP={metrics['average_precision']:.3f}，"
                f"Precision={metrics['precision']:.3f}，Recall={metrics['recall']:.3f}，"
                f"匹配 {metrics['matched_count']}/{metrics['target_count']}，"
                f"ECE={_format_optional(metrics.get('calibration_error'))}"
            )
        lines.append("")
    if not has_localization:
        lines.append("当前测试集没有可用的时间或空间标注，未生成定位指标。")
        lines.append("")

    lines.extend(
        [
            "## 使用限制",
            "",
            "- 该报告不能替代司法鉴定。",
            "- 元数据阴性不进入真实性证据。",
            "- 结果不覆盖未出现在索引中的生成器、编码链和人脸采集条件。",
            "- 正式发布前必须复核数据授权状态、生成器隔离和身份隔离。",
            "",
        ]
    )
    return "\n".join(lines)


def _format_optional(value: float | int | None) -> str:
    return "-" if value is None else f"{float(value):.3f}"
