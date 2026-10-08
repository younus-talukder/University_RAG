"""Compare aligned runs without rerunning historical experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.manifest import atomic_json


def _rows(path: Path) -> dict[str, dict[str, str]]:
    with (path / "results.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {row["evaluation_id"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate evaluation IDs in {path}")
    return output


def compare(left: Path, right: Path, *, aligned_subset: bool = False) -> dict:
    one = json.loads((left / "run_manifest.json").read_text(encoding="utf-8"))
    two = json.loads((right / "run_manifest.json").read_text(encoding="utf-8"))
    if one["dataset"]["dataset_sha256"] != two["dataset"]["dataset_sha256"] and not aligned_subset:
        raise ValueError("dataset hashes differ; use --aligned-subset only with explicitly aligned evaluation IDs")
    left_rows, right_rows = _rows(left), _rows(right)
    if set(left_rows) != set(right_rows) and not aligned_subset:
        raise ValueError("evaluation IDs differ; use --aligned-subset for the common-ID subset")
    aligned = sorted(set(left_rows) & set(right_rows))
    if not aligned:
        raise ValueError("runs have no aligned evaluation IDs")
    def supported(row):
        failures = row.get("answer_safety_failures") or "[]"
        try:
            failures = json.loads(failures)
        except json.JSONDecodeError:
            failures = ["UNPARSEABLE_SAFETY_AUDIT"]
        return (row.get("answerability_status") == "SUPPORTED" and
                row.get("answer_returned") == "True" and
                row.get("grounding_status") == "True" and
                row.get("language_status") == "True" and
                row.get("semantic_status") in {"True", "", None} and not failures)
    left_coverage = sum(supported(left_rows[key]) for key in aligned) / len(aligned)
    right_coverage = sum(supported(right_rows[key]) for key in aligned) / len(aligned)
    return {
        "left_run_id": one["run_id"], "right_run_id": two["run_id"],
        "dataset_hashes_match": one["dataset"]["dataset_sha256"] == two["dataset"]["dataset_sha256"],
        "explicit_aligned_subset": aligned_subset, "aligned_rows": len(aligned),
        "left_safe_answer_coverage": left_coverage,
        "right_safe_answer_coverage": right_coverage,
        "delta": right_coverage - left_coverage,
        "warning": "Automatic paired support comparison; not human correctness or thesis accuracy",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--aligned-subset", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.left, args.right, aligned_subset=args.aligned_subset)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
