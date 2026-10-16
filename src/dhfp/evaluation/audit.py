from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from dhfp.evaluation.metrics import BinaryMetrics
from dhfp.evaluation.schema import DatasetLabel, DatasetRecord


@dataclass(frozen=True)
class AuditFinding:
    control: str
    passed: bool
    details: str
    values: dict

    def as_dict(self) -> dict:
        return {
            "control": self.control,
            "passed": self.passed,
            "details": self.details,
            "values": self.values,
        }


def audit_split(
    records: list[DatasetRecord],
    splits: dict[str, list[str]],
) -> list[AuditFinding]:
    # 不可用记录不会进入 split；它们不应被误报为“漏分配”。
    lookup = {record.sample_id: record for record in records if record.usable}
    train = [lookup[sample_id] for sample_id in splits["train"] if sample_id in lookup]
    validation = [
        lookup[sample_id] for sample_id in splits["validation"] if sample_id in lookup
    ]
    test = [lookup[sample_id] for sample_id in splits["test"] if sample_id in lookup]

    train_generators = {
        record.generator for record in train if record.label == DatasetLabel.GENERATED
    }
    test_generators = {
        record.generator for record in test if record.label == DatasetLabel.GENERATED
    }
    generator_leaks = train_generators & test_generators

    def identities(records_for_split: list[DatasetRecord]) -> set[str]:
        return {
            record.source_person_id
            for record in records_for_split
            if record.source_person_id is not None
        }

    train_identities = identities(train)
    validation_identities = identities(validation)
    test_identities = identities(test)
    identity_leaks = (
        (train_identities & validation_identities)
        | (train_identities & test_identities)
        | (validation_identities & test_identities)
    )

    all_sample_ids = [sample_id for values in splits.values() for sample_id in values]
    duplicates = [sample_id for sample_id, count in Counter(all_sample_ids).items() if count > 1]
    assigned = set(all_sample_ids)
    missing = sorted(set(lookup) - assigned)
    unknown = sorted(assigned - set(lookup))

    return [
        AuditFinding(
            control="generator_disjoint",
            passed=not generator_leaks,
            details=(
                "训练与测试生成器无交集。"
                if not generator_leaks
                else f"发现生成器泄漏：{sorted(generator_leaks)}"
            ),
            values={
                "train_generators": sorted(train_generators),
                "test_generators": sorted(test_generators),
                "leaks": sorted(generator_leaks),
            },
        ),
        AuditFinding(
            control="identity_disjoint",
            passed=not identity_leaks,
            details=(
                "来源身份未跨 split 泄漏。"
                if not identity_leaks
                else f"发现身份泄漏：{len(identity_leaks)} 个"
            ),
            values={
                "train_identity_count": len(train_identities),
                "validation_identity_count": len(validation_identities),
                "test_identity_count": len(test_identities),
                "leaks": sorted(identity_leaks),
            },
        ),
        AuditFinding(
            control="partition_integrity",
            passed=not duplicates and not missing and not unknown,
            details=(
                "样本无重复且全部被分配。"
                if not duplicates and not missing and not unknown
                else (
                    f"重复样本 {len(duplicates)} 条，未分配样本 {len(missing)} 条，"
                    f"未知样本 {len(unknown)} 条。"
                )
            ),
            values={"duplicates": duplicates, "missing": missing, "unknown": unknown},
        ),
    ]


def audit_model(
    metrics: BinaryMetrics,
    trivial_metrics: BinaryMetrics,
    *,
    max_fpr: float = 0.02,
    min_unseen_tpr: float = 0.65,
    max_ece: float = 0.06,
) -> list[AuditFinding]:
    above_floor = metrics.tpr_at_fpr_1pct > trivial_metrics.tpr_at_fpr_1pct + 0.05
    return [
        AuditFinding(
            control="above_floor_margin",
            passed=above_floor,
            details=(
                "低 FPR 检出不高于平凡基线。"
                if not above_floor
                else "低 FPR 检出高于平凡基线至少 5 个百分点。"
            ),
            values={
                "model_tpr_at_fpr_1pct": metrics.tpr_at_fpr_1pct,
                "trivial_tpr_at_fpr_1pct": trivial_metrics.tpr_at_fpr_1pct,
            },
        ),
        AuditFinding(
            control="operating_point",
            passed=metrics.tpr_at_fpr_1pct >= min_unseen_tpr,
            details=f"TPR@FPR=1% 为 {metrics.tpr_at_fpr_1pct:.3f}。",
            values={"minimum": min_unseen_tpr, "actual": metrics.tpr_at_fpr_1pct},
        ),
        AuditFinding(
            control="calibration",
            passed=metrics.ece <= max_ece,
            details=f"ECE 为 {metrics.ece:.4f}。",
            values={"maximum": max_ece, "actual": metrics.ece},
        ),
        AuditFinding(
            control="false_positive_budget",
            passed=metrics.operational_fpr <= max_fpr,
            details=f"评测集操作点 FPR 为 {metrics.operational_fpr:.4f}。",
            values={"maximum": max_fpr, "actual": metrics.operational_fpr},
        ),
    ]
