from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dhfp.calibration import ThresholdConfig
from dhfp.evaluation.audit import audit_split
from dhfp.evaluation.runner import EvaluationRunner
from dhfp.evaluation.schema import index_summary, load_index, write_index
from dhfp.evaluation.split import build_splits, load_splits, save_splits
from dhfp.learning.continual import (
    ContinualGateThresholds,
    evaluate_regression_gate,
    select_replay_samples,
)
from dhfp.plugins.registry import load_detectors
from dhfp.redteam.transforms import run_redteam_suite, write_redteam_manifest
from dhfp.training.augment import (
    apply_plan,
    augmented_records,
    build_plan,
    parse_specs,
    plan_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dhfp", description="数字人鉴别平台评测工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("dataset-check", help="检查并汇总数据索引")
    check.add_argument("--index", required=True)
    check.add_argument("--output")

    split = subparsers.add_parser("split", help="构建 generator/identity 双重隔离切分")
    split.add_argument("--index", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=20260918)
    split.add_argument("--test-generators", type=int, default=3)
    split.add_argument("--validation-ratio", type=float, default=0.15)

    evaluate = subparsers.add_parser("eval", help="运行统一评测并输出 Markdown/JSON/PNG")
    evaluate.add_argument("--index", required=True)
    evaluate.add_argument("--splits")
    evaluate.add_argument("--output", default="data/reports")
    evaluate.add_argument("--baselines", default="configs/baselines.json")
    evaluate.add_argument("--seed", type=int, default=20260918)
    evaluate.add_argument("--test-generators", type=int, default=3)

    calibrate = subparsers.add_parser("calibrate", help="根据评测结果生成校准配置")
    calibrate.add_argument("--report", required=True)
    calibrate.add_argument("--detector", default="fusion-pipeline")
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument(
        "--trusted-quantile",
        type=float,
        default=0.25,
        help="灰区下界取 validation 校准分数的分位数，默认 0.25。",
    )
    calibrate.add_argument(
        "--suspect-quantile",
        type=float,
        default=0.75,
        help="灰区上界取 validation 校准分数的分位数，默认 0.75。",
    )

    redteam = subparsers.add_parser("redteam", help="生成红队扰动媒体和审计清单")
    redteam.add_argument("--input", required=True)
    redteam.add_argument("--output", required=True)
    redteam.add_argument("--families")
    redteam.add_argument("--timeout", type=int, default=300)

    replay = subparsers.add_parser("replay", help="构建持续学习经验回放清单")
    replay.add_argument("--index", required=True)
    replay.add_argument("--splits")
    replay.add_argument("--output", required=True)
    replay.add_argument("--max-samples", type=int, default=200)
    replay.add_argument("--seed", type=int, default=20260918)
    replay.add_argument("--per-generator-limit", type=int)

    augment = subparsers.add_parser(
        "augment-reencode",
        help="生成同强度重编码增广：所有标签使用完全相同的编码参数",
    )
    augment.add_argument("--index", required=True)
    augment.add_argument("--output-dir", required=True)
    augment.add_argument("--output-index")
    augment.add_argument(
        "--strengths",
        default="crf28",
        help="逗号分隔的强度名，可选: crf23,crf28,crf32,crf28-720p",
    )
    augment.add_argument("--only-augmented", action="store_true")
    augment.add_argument("--dry-run", action="store_true")
    augment.add_argument("--timeout", type=int, default=900)

    gate = subparsers.add_parser("regression-gate", help="执行持续学习发布回归门禁")
    gate.add_argument("--previous", required=True)
    gate.add_argument("--candidate", required=True)
    gate.add_argument("--detector", required=True)
    gate.add_argument("--output", required=True)
    gate.add_argument("--min-tpr", type=float, default=0.65)
    gate.add_argument("--max-ece", type=float, default=0.06)
    gate.add_argument("--max-tpr-drop", type=float, default=0.03)
    gate.add_argument("--max-ece-increase", type=float, default=0.01)
    gate.add_argument("--max-auc-drop", type=float, default=0.02)
    gate.add_argument("--max-localization-ap-drop", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "dataset-check":
        return _dataset_check(args)
    if args.command == "split":
        return _split(args)
    if args.command == "eval":
        return _eval(args)
    if args.command == "calibrate":
        return _calibrate(args)
    if args.command == "redteam":
        return _redteam(args)
    if args.command == "replay":
        return _replay(args)
    if args.command == "augment-reencode":
        return _augment_reencode(args)
    if args.command == "regression-gate":
        return _regression_gate(args)
    return 2


def _dataset_check(args: argparse.Namespace) -> int:
    records = load_index(args.index)
    summary = index_summary(records)
    output = json.dumps(summary, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    return 0


def _split(args: argparse.Namespace) -> int:
    records = load_index(args.index)
    payload = build_splits(
        records,
        seed=args.seed,
        generator_test_count=args.test_generators,
        validation_ratio=args.validation_ratio,
    )
    save_splits(payload, args.output)
    audits = audit_split(records, payload["splits"])
    print(json.dumps([item.as_dict() for item in audits], ensure_ascii=False, indent=2))
    return 0 if all(item.passed for item in audits) else 1


def _eval(args: argparse.Namespace) -> int:
    records = load_index(args.index)
    splits = (
        load_splits(args.splits)
        if args.splits
        else build_splits(
            records,
            seed=args.seed,
            generator_test_count=args.test_generators,
        )
    )
    detectors = load_detectors(args.baselines)
    runner = EvaluationRunner(
        records=records,
        splits=splits,
        detectors=detectors,
        output_dir=args.output,
    )
    payload = runner.run()
    available = sum(1 for item in payload["detectors"].values() if item["available"])
    print(f"报告已写入: {Path(args.output).resolve()}")
    print(f"成功评测检测器: {available}/{len(detectors)}")
    return 0 if available else 1


def _calibrate(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
    detector = payload["detectors"].get(args.detector)
    if not detector or not detector.get("available"):
        print(f"检测器不可用: {args.detector}", file=sys.stderr)
        return 1
    if not 0 < args.trusted_quantile < args.suspect_quantile < 1:
        print("分位数必须满足 0 < trusted-quantile < suspect-quantile < 1", file=sys.stderr)
        return 1
    quantiles = detector.get("validation_score_quantiles") or {}
    if not quantiles:
        print(
            "评测报告缺少 validation_score_quantiles；"
            "为避免用测试集调阈值，请用当前版本重新运行 eval 后再校准。",
            file=sys.stderr,
        )
        return 1
    trusted = _lookup_quantile(quantiles, args.trusted_quantile)
    suspect = _lookup_quantile(quantiles, args.suspect_quantile)
    if trusted >= suspect:
        print("分位数映射后的阈值顺序异常，请检查评测分数分布。", file=sys.stderr)
        return 1
    profile = detector["calibration"]
    profile["bias"] = 0.0
    profile["thresholds"] = ThresholdConfig(
        trusted_max=trusted,
        suspect_min=suspect,
        method="quantile",
        trusted_quantile=args.trusted_quantile,
        suspect_quantile=args.suspect_quantile,
        calibration_count=detector.get("validation_count"),
        real_negative_count=detector.get("validation_negative_count"),
        notes=(
            f"由 {args.detector} validation 校准分数分位数生成；"
            "偏移项固定为 0，评测集变更后必须重新拟合。"
        ),
    ).model_dump()
    profile["notes"] = "由评测报告重新生成：禁用偏移项，阈值按 validation 分位数确定。"
    Path(args.output).write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"校准配置已写入: {Path(args.output).resolve()}")
    print(f"灰区阈值: trusted<={trusted:.4f} suspect>={suspect:.4f}")
    return 0


def _lookup_quantile(quantiles: dict[str, float], target: float) -> float:
    key = f"{target:g}"
    if key in quantiles:
        return float(quantiles[key])
    available = sorted((float(item), float(value)) for item, value in quantiles.items())
    nearest = min(available, key=lambda item: abs(item[0] - target))
    print(
        f"警告: 报告未包含分位数 {key}，回退到最接近的 {nearest[0]:g}。",
        file=sys.stderr,
    )
    return nearest[1]


def _redteam(args: argparse.Namespace) -> int:
    families = (
        [item.strip() for item in args.families.split(",") if item.strip()]
        if args.families
        else None
    )
    results = run_redteam_suite(
        args.input,
        args.output,
        families=families,
        timeout_seconds=args.timeout,
    )
    manifest = write_redteam_manifest(results, Path(args.output) / "manifest.json")
    failures = [result for result in results if not result.success]
    print(f"红队清单已写入: {manifest.resolve()}")
    print(f"成功: {len(results) - len(failures)}/{len(results)}")
    for result in failures:
        print(f"失败: {result.family} - {result.error}", file=sys.stderr)
    return 0 if not failures else 1


def _replay(args: argparse.Namespace) -> int:
    records = load_index(args.index)
    excluded: set[str] = set()
    if args.splits:
        splits = load_splits(args.splits)
        excluded = set(splits.get("metadata", {}).get("test_generators", []))
    selection = select_replay_samples(
        records,
        max_samples=args.max_samples,
        seed=args.seed,
        excluded_generators=excluded,
        per_generator_limit=args.per_generator_limit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(selection.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"经验回放清单已写入: {output.resolve()}")
    print(f"样本数: {len(selection.sample_ids)}")
    return 0


def _augment_reencode(args: argparse.Namespace) -> int:
    records = load_index(args.index)
    specs = parse_specs(args.strengths)
    jobs = build_plan(records, args.output_dir, specs)
    summary = plan_summary(jobs)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["same_strength_across_labels"]:
        print("不同标签的重编码强度不一致，拒绝执行。", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0
    results = apply_plan(jobs, timeout_seconds=args.timeout)
    failures = [result for result in results if not result.success]
    augmented = [
        record
        for record in augmented_records(records, specs, args.output_dir)
        if any(result.success and result.output_path == record.media_path for result in results)
    ]
    if args.output_index:
        payload = augmented if args.only_augmented else [*records, *augmented]
        write_index(payload, args.output_index)
        print(f"增广索引已写入: {Path(args.output_index).resolve()}")
    print(f"成功: {len(results) - len(failures)}/{len(results)}")
    for failure in failures[:20]:
        print(f"失败: {failure.case_id} - {failure.error}", file=sys.stderr)
    return 0 if not failures else 1


def _regression_gate(args: argparse.Namespace) -> int:
    previous = json.loads(Path(args.previous).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    thresholds = ContinualGateThresholds(
        min_tpr_at_fpr_1pct=args.min_tpr,
        max_ece=args.max_ece,
        max_tpr_drop=args.max_tpr_drop,
        max_ece_increase=args.max_ece_increase,
        max_auc_drop=args.max_auc_drop,
        max_localization_ap_drop=args.max_localization_ap_drop,
    )
    gate = evaluate_regression_gate(
        previous,
        candidate,
        detector_id=args.detector,
        thresholds=thresholds,
    )

    payload = gate.as_dict()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"门禁结果已写入: {output.resolve()}")
    print(f"结果: {'通过' if gate.passed else '未通过'}")
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
