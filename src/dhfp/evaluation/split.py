from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from dhfp.evaluation.schema import DatasetLabel, DatasetRecord


def _identity_key(record: DatasetRecord) -> str:
    if record.source_person_id:
        return f"person:{record.source_person_id}"
    return f"sample:{record.sample_id}"


def build_splits(
    records: list[DatasetRecord],
    *,
    seed: int = 20260918,
    generator_test_count: int = 3,
    validation_ratio: float = 0.15,
) -> dict[str, list[str]]:
    usable = [record for record in records if record.usable]
    generated_generators = sorted(
        {
            record.generator
            for record in usable
            if record.label == DatasetLabel.GENERATED and record.generator
        }
    )
    if not generated_generators:
        raise ValueError("数据集没有可用的生成器，无法构建 generator-disjoint 切分")
    if generator_test_count >= len(generated_generators):
        raise ValueError("测试生成器数量必须小于可用生成器总数")

    rng = random.Random(seed)
    shuffled_generators = generated_generators[:]
    rng.shuffle(shuffled_generators)
    test_generators = set(shuffled_generators[:generator_test_count])
    train_pool_generators = set(shuffled_generators[generator_test_count:])

    by_identity: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in usable:
        by_identity[_identity_key(record)].append(record)

    test_identities: set[str] = set()
    for identity, identity_records in by_identity.items():
        if any(
            record.label == DatasetLabel.GENERATED and record.generator in test_generators
            for record in identity_records
        ):
            test_identities.add(identity)

    remaining_identities = sorted(set(by_identity) - test_identities)
    rng.shuffle(remaining_identities)
    validation_count = round(len(remaining_identities) * validation_ratio)
    validation_identities = set(remaining_identities[:validation_count])

    splits = {"train": [], "validation": [], "test": []}
    for identity in sorted(by_identity):
        records_for_identity = by_identity[identity]
        if identity in test_identities:
            bucket = "test"
        elif identity in validation_identities:
            bucket = "validation"
        else:
            bucket = "train"
        splits[bucket].extend(record.sample_id for record in records_for_identity)

    metadata = {
        "seed": seed,
        "strategy": ["generator_disjoint", "identity_disjoint"],
        "test_generators": sorted(test_generators),
        "train_generators": sorted(train_pool_generators),
        "counts": {key: len(value) for key, value in splits.items()},
    }
    return {"splits": splits, "metadata": metadata}


def save_splits(payload: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_splits(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_map(payload: dict) -> dict[str, str]:
    return {
        sample_id: split_name
        for split_name, sample_ids in payload["splits"].items()
        for sample_id in sample_ids
    }
