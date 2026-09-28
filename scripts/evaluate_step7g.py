"""Targeted Step 7G check using frozen Step 7F failures as offline inputs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import answer_question

RESULTS = ROOT / "results"
RECOVERY = RESULTS / "step7f_failure_recovery.csv"
OUTPUT = RESULTS / "step7g_targeted_results.csv"
REVIEW = RESULTS / "step7g_targeted_review.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    frozen = read_rows(RECOVERY)
    selected = [row for row in frozen if
                (row["language"] == "bangla" and row["merged_support_status"] == "supported")
                or row["language"] == "banglish"]
    results = []
    for number, row in enumerate(selected, 1):
        response = answer_question(row["original_query"], use_answer_bank=False)
        relation = response.get("relation_extracted") or {}
        results.append({
            "question_id": row["question_id"], "language": row["language"],
            "question": row["original_query"], "previous_status": row["final_answer_status"],
            "initial_support_status": response.get("initial_support_status"),
            "fallback_triggered": response.get("fallback_triggered"),
            "merged_support_status": response.get("merged_support_status"),
            "support_status": response.get("support_status"),
            "final_status": response.get("final_status"),
            "answer_strategy": response.get("answer_strategy"),
            "relation_type": response.get("relation_type"),
            "relation_values": " | ".join(relation.get("values", [])),
            "final_answer": response.get("answer"),
            "source": response.get("source"), "page": response.get("page"),
            "chunk_id": response.get("chunk_id"),
            "supporting_excerpt": response.get("supporting_excerpt"),
            "continuation_chunk_ids": " | ".join(relation.get("continuation_chunk_ids", [])),
            "language_validation_passed": response.get("language_validation_passed"),
            "grounding_validation_passed": response.get("grounding_validation_passed"),
            "grounding_validation_reason": response.get("grounding_validation_reason"),
            "generation_used": response.get("generation_used"),
            "generation_attempts": response.get("generation_attempts"),
            "total_seconds": (response.get("latency_seconds") or {}).get("total"),
        })
        write_rows(OUTPUT, results, list(results[0]))
        print(f"{number}/{len(selected)} {row['language']} {row['question_id']} "
              f"{response.get('final_status')} {response.get('answer_strategy')}", flush=True)

    review = [{key: row[key] for key in (
        "question_id", "language", "question", "previous_status", "support_status",
        "final_status", "answer_strategy", "relation_type", "final_answer", "source",
        "page", "chunk_id", "supporting_excerpt", "language_validation_passed",
        "grounding_validation_passed")} | {
            "human_correctness_score": "", "human_evidence_score": "",
            "human_language_score": "", "human_notes": ""} for row in results]
    write_rows(REVIEW, review, list(review[0]))


if __name__ == "__main__":
    main()
