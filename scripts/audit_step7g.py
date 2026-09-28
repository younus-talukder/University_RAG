"""Offline Step 7G Banglish and English regression audit; no production mappings."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import answer_question

RESULTS = ROOT / "results"
TARGETS = {"Q003", "Q005", "Q007", "Q010", "Q011", "Q019", "Q020", "Q036", "Q085"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save(path: Path, data: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def run(kind: str) -> None:
    if kind == "banglish":
        source = [row for row in rows(RESULTS / "step7f_failure_funnel.csv")
                  if row["language"] == "banglish" and row["primary_stage"] == "CANONICAL_GENERATION_FAILURE"]
        output = RESULTS / "step7g_banglish_audit.csv"
    elif kind == "english":
        source = [row for row in rows(RESULTS / "step7e_300_results.csv")
                  if row["language"] == "english" and row["question_id"] in TARGETS]
        output = RESULTS / "step7g_english_regression.csv"
    else:
        raise SystemExit("usage: audit_step7g.py banglish|english")
    data = []
    for number, old in enumerate(source, 1):
        question = old["original_query"] if kind == "banglish" else old["question"]
        response = answer_question(question, use_answer_bank=False)
        data.append({
            "question_id": old["question_id"], "language": kind,
            "question": question, "previous_status": old["final_status"],
            "previous_stage": old.get("primary_stage", ""),
            "support_status": response.get("support_status"),
            "final_status": response.get("final_status"),
            "answer_strategy": response.get("answer_strategy"),
            "relation_type": response.get("relation_type"),
            "final_answer": response.get("answer"), "page": response.get("page"),
            "chunk_id": response.get("chunk_id"),
            "language_validation_passed": response.get("language_validation_passed"),
            "grounding_validation_passed": response.get("grounding_validation_passed"),
            "generation_used": response.get("generation_used"),
            "total_seconds": (response.get("latency_seconds") or {}).get("total"),
        })
        save(output, data)
        print(f"{number}/{len(source)} {kind} {old['question_id']} "
              f"{response.get('final_status')} {response.get('answer_strategy')}", flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "")
