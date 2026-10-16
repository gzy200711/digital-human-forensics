from __future__ import annotations

import json
from pathlib import Path

from dhfp.plugins.base import EvaluationDetector
from dhfp.plugins.builtin import (
    FusionPipelineDetector,
    MetadataRuleBaseline,
    TrivialMetadataBaseline,
)
from dhfp.plugins.external import ExternalCommandDetector

BUILTINS = {
    "trivial_metadata": TrivialMetadataBaseline,
    "metadata_rule": MetadataRuleBaseline,
    "fusion_pipeline": FusionPipelineDetector,
}


def load_detectors(path: str | Path) -> list[EvaluationDetector]:
    manifests = json.loads(Path(path).read_text(encoding="utf-8"))
    detectors: list[EvaluationDetector] = []
    for manifest in manifests:
        detector_type = manifest["type"]
        if detector_type in BUILTINS:
            detectors.append(BUILTINS[detector_type]())
        elif detector_type == "external_command":
            detectors.append(ExternalCommandDetector(manifest))
        else:
            raise ValueError(f"未知评测插件类型: {detector_type}")
    return detectors
