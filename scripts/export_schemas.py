from __future__ import annotations

import json
from pathlib import Path

from dhfp.domain import ForensicReport
from dhfp.evaluation.schema import DatasetRecord


def export_schema(model, path: Path, title: str) -> None:
    schema = model.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = title
    path.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "schemas"
    output.mkdir(parents=True, exist_ok=True)
    export_schema(ForensicReport, output / "report.schema.json", "数字人鉴别报告")
    export_schema(DatasetRecord, output / "dataset.schema.json", "内部评测集记录")
    print(f"schemas written: {output}")


if __name__ == "__main__":
    main()
