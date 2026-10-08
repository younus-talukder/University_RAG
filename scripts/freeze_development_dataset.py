"""Write or verify an auditable, immutable development-dataset freeze manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.manifest import atomic_json
from src.evaluation.schema import LANGUAGES, expand, load_and_validate


def freeze(dataset: Path, output: Path, *, annotation_version: str,
           q007_adjudication: str) -> dict:
    rows, validation = load_and_validate(dataset)
    if not validation["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(validation["errors"][:10]))
    variants = list(expand(rows))
    counts = Counter(row["language"] for row in variants)
    q007 = next((row for row in rows if row["_base_question_id"] == "Q007"), None)
    if q007 is None:
        raise ValueError("Q007 is missing from the development dataset")
    manifest = {
        "development_dataset_version": "v1-final",
        "dataset_filename": validation["dataset_filename"],
        "dataset_sha256": validation["dataset_sha256"],
        "row_count": validation["row_count"],
        "language_variant_count": validation["language_variant_count"],
        "language_variants": {language: counts[language] for language in LANGUAGES},
        "column_schema": validation["column_schema"],
        "annotation_version": annotation_version,
        "q007_adjudication_version": q007_adjudication,
        "ground_truth_status_distribution": dict(sorted(Counter(row["_annotation_status"] for row in rows).items())),
        "q007_evaluation_annotation": {
            "english_reference": q007.get("ground_truth_answer_en", ""),
            "bangla_reference": q007.get("ground_truth_answer_bn", ""),
            "banglish_reference": q007.get("ground_truth_answer_banglish", ""),
            "expected_source": q007.get("expected_source", ""),
            "expected_page": q007.get("expected_page", ""),
            "notes": q007.get("notes", ""),
        },
        "evaluation_only": True,
        "inference_rows_executed": 0,
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("freeze manifest already exists and differs; use a new annotation version/path")
    else:
        atomic_json(output, manifest)
    return manifest


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--dataset", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--annotation-version", required=True)
    command.add_argument("--q007-adjudication", required=True)
    args = command.parse_args()
    result = freeze(args.dataset, args.output,
                    annotation_version=args.annotation_version,
                    q007_adjudication=args.q007_adjudication)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
