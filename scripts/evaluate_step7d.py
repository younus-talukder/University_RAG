from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluate_step7c import (
    evaluate, load_dataset, manual_sample, read_checkpoint, rows_to_resume,
    select_smoke, summarize, write_csv,
)


RESULTS = ROOT_DIR / "results"
BEFORE_CSV = RESULTS / "step7c_300_results.csv"
BEFORE_SMOKE_CSV = RESULTS / "step7c_smoke_results.csv"
BEFORE_REVIEW_CSV = RESULTS / "step7c_manual_review_sample.csv"
SMOKE_CSV = RESULTS / "step7d_smoke_results.csv"
REVIEW_RUN_CSV = RESULTS / "step7d_review_run.csv"
FULL_CSV = RESULTS / "step7d_300_results.csv"
SMOKE_COMPARISON_JSON = RESULTS / "step7d_smoke_comparison.json"
SUMMARY_JSON = RESULTS / "step7d_summary.json"
MANUAL_CSV = RESULTS / "step7d_manual_review_sample.csv"

HUMAN_FIELDS = (
    "correctness", "relevance", "groundedness", "completeness", "naturalness",
    "semantic_consistency", "error_category", "notes",
)


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("question_id")), str(row.get("language") or row.get("expected_language"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _review_dataset(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    review_questions = {str(row["question"]) for row in _load_csv(BEFORE_REVIEW_CSV)}
    selected = [row for row in dataset if str(row.get("question")) in review_questions]
    if len(selected) != 30:
        raise RuntimeError(f"Expected 30 existing review questions, found {len(selected)}.")
    return selected


def _comparison(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        generated = [row for row in rows if str(row.get("generation_used")).casefold() == "true" or row.get("generation_used") is True]
        retries = [row for row in generated if str(row.get("retry_used")).casefold() == "true" or row.get("retry_used") is True]
        return {
            "rows": len(rows),
            "strategies": dict(Counter(str(row.get("answer_strategy")) for row in rows)),
            "generation_attempted": len(generated),
            "accepted": sum(row.get("answer_strategy") == "gguf_generation" for row in generated),
            "rejected": sum(row.get("answer_strategy") != "gguf_generation" for row in generated),
            "retries": len(retries),
            "successful_retries": sum(row.get("answer_strategy") == "gguf_generation" for row in retries),
            "failed_retries": sum(row.get("answer_strategy") != "gguf_generation" for row in retries),
            "semantic_rejections": dict(Counter(str(row.get("generation_rejection_reason") or "") for row in generated if row.get("generation_rejection_reason"))),
        }
    before_by_key = {_key(row): row for row in before}
    paired = []
    for row in after:
        previous = before_by_key.get(_key(row), {})
        paired.append({
            "question_id": row.get("question_id"),
            "language": row.get("language"),
            "question": row.get("question"),
            "before_answer": previous.get("final_answer", ""),
            "before_strategy": previous.get("answer_strategy", ""),
            "after_answer": row.get("final_answer", ""),
            "after_strategy": row.get("answer_strategy", ""),
            "after_language_reason": row.get("language_validation_reason", ""),
            "after_grounding_reason": row.get("grounding_validation_reason", ""),
            "after_rejection_reason": row.get("generation_rejection_reason", ""),
        })
    return {"before": metrics(before), "after": metrics(after), "paired_rows": paired}


def _write_manual(after: list[dict[str, Any]]) -> None:
    before = {_key(row): row for row in read_checkpoint(BEFORE_CSV)}
    rows: list[dict[str, Any]] = []
    for row in after:
        old = before.get(_key(row), {})
        item = {
            "question_id": row.get("question_id"),
            "language": row.get("language"),
            "question": row.get("question"),
            "reference": row.get("reference"),
            "before_step7d_answer": old.get("final_answer", ""),
            "after_step7d_answer": row.get("final_answer", ""),
            "strategy": row.get("answer_strategy"),
            "source": row.get("source"),
            "page": row.get("page"),
            "supporting_evidence": row.get("supporting_excerpt"),
            "language_validation_reason": row.get("language_validation_reason"),
            "grounding_validation_reason": row.get("grounding_validation_reason"),
            "semantic_rejection_reason": row.get("generation_rejection_reason"),
        }
        item.update({field: "" for field in HUMAN_FIELDS})
        rows.append(item)
    write_csv(MANUAL_CSV, rows)


def _run(selected: list[dict[str, Any]], path: Path, resume: bool) -> list[dict[str, Any]]:
    checkpoint = read_checkpoint(path) if resume else []
    return evaluate(rows_to_resume(selected, checkpoint), path, initial_rows=checkpoint)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("smoke", "review", "full", "finalize"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    dataset = load_dataset()

    if args.action == "smoke":
        selected = select_smoke(dataset)
        after = _run(selected, SMOKE_CSV, args.resume)
        before = read_checkpoint(BEFORE_SMOKE_CSV)
        SMOKE_COMPARISON_JSON.write_text(json.dumps(_comparison(before, after), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summarize(after), ensure_ascii=False, indent=2))
    elif args.action == "review":
        selected = _review_dataset(dataset)
        after = _run(selected, REVIEW_RUN_CSV, args.resume)
        _write_manual(after)
        print(json.dumps({"rows": len(after), "manual_file": str(MANUAL_CSV)}, ensure_ascii=False, indent=2))
    elif args.action == "full":
        after = _run(dataset, FULL_CSV, args.resume)
        SUMMARY_JSON.write_text(json.dumps({"summary": summarize(after), "comparison": _comparison(read_checkpoint(BEFORE_CSV), after)}, ensure_ascii=False, indent=2), encoding="utf-8")
        if len(after) == 300:
            selected_keys = {_key(row) for row in _review_dataset(dataset)}
            review_rows = [row for row in after if _key(row) in selected_keys]
            write_csv(REVIEW_RUN_CSV, review_rows)
            _write_manual(review_rows)
        print(json.dumps(summarize(after), ensure_ascii=False, indent=2))
    else:
        after = read_checkpoint(FULL_CSV)
        SUMMARY_JSON.write_text(json.dumps({"summary": summarize(after), "comparison": _comparison(read_checkpoint(BEFORE_CSV), after)}, ensure_ascii=False, indent=2), encoding="utf-8")
        if len(after) == 300:
            selected_keys = {_key(row) for row in _review_dataset(dataset)}
            review_rows = [row for row in after if _key(row) in selected_keys]
            write_csv(REVIEW_RUN_CSV, review_rows)
            _write_manual(review_rows)


if __name__ == "__main__":
    main()
