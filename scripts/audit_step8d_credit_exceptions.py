"""Bounded follow-up for two source-level course-credit discrepancies."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_step8d import fingerprint, measure
from src.evaluator import load_dataset


def run() -> None:
    output = ROOT / "results" / "step8d_credit_exception_checkpoint.jsonl"
    selected = [row for row in load_dataset() if row["question_id"] in {"Q034", "Q065"}]
    if len(selected) != 6:
        raise RuntimeError("Expected two three-language credit groups")
    current = fingerprint()
    existing = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()] if output.exists() else []
    if any(row["run_fingerprint"] != current for row in existing):
        raise RuntimeError("Credit exception checkpoint fingerprint mismatch")
    completed = {(row["question_id"], row["language"]): row for row in existing}
    for row in selected:
        key = row["question_id"], row["expected_language"]
        if key in completed:
            continue
        result = measure(row, current)
        with output.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        completed[key] = result
        print(f"{len(completed)}/6 {key[1]} {key[0]} {result['answerability_status']} "
              f"values={result['conflicting_values']}", flush=True)


if __name__ == "__main__":
    run()
