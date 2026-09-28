"""Step 7H full measurement using a separate resumable checkpoint.

The Q007 decision is an evaluation annotation only. No benchmark reference is
passed into retrieval, evidence assessment, or answer generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_step7g_full as base

base.RESULT_CSV = ROOT / "results/step7h_full_300_results.csv"
base.SUMMARY_JSON = ROOT / "results/step7h_full_300_summary.json"
base.REVIEW_CSV = ROOT / "results/step7h_full_manual_review.csv"

_old_fingerprint = base.run_fingerprint
_old_evaluate = base.evaluate
_old_select_review = base.select_review


def fingerprint() -> str:
    digest = hashlib.sha256(_old_fingerprint().encode())
    for relative in ("src/answer_policy.py", "src/course_rows.py", "src/grounding_validator.py",
                     "src/semantic_contract.py", "scripts/evaluate_step7h_full.py"):
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()[:20]


def evaluate(row: dict, expected_page: str, run_fingerprint: str) -> dict:
    result = _old_evaluate(row, expected_page, run_fingerprint)
    # User-approved B: score against the page-4/reference Engineering wording.
    # This column never feeds the runtime pipeline.
    if row["question_id"] == "Q007":
        result["q007_evaluation_reference"] = "page-4 Engineering"
        result["q007_reference_agreement"] = "AGREES" if "Engineering" in result["final_answer"] else "MISMATCH"
    else:
        result["q007_evaluation_reference"] = ""
        result["q007_reference_agreement"] = ""
    return result


def select_review(rows: list[dict]) -> list[dict]:
    selected = _old_select_review(rows)
    indexed = {(row["question_id"], row["language"]): row for row in rows}
    for language, mandatory in {
        "english": ("Q030", "Q096", "Q007", "Q009"),
        "bangla": ("Q007", "Q009", "Q023"),
        "banglish": ("Q006", "Q007", "Q009", "Q023"),
    }.items():
        current = [row for row in selected if row["language"] == language]
        present = {row["question_id"] for row in current}
        for question_id in mandatory:
            if question_id in present:
                continue
            source = indexed[(question_id, language)]
            replacement = dict(current[-1])
            for key in ("question_id", "language", "question", "reference", "final_answer", "final_status",
                        "answer_strategy", "relation_type", "source", "page", "chunk_id"):
                replacement[key] = source[key]
            replacement["supporting_evidence"] = source["supporting_excerpt"]
            replacement["crosslingual_recovered"] = base.truth(source["fallback_triggered"]) and source["support_status"] == "supported"
            replacement["language_validation_passed"] = source["language_validation_passed"]
            replacement["grounding_validation_passed"] = source["grounding_validation_passed"]
            replacement["previous_failure_category"] = "human_reject" if (question_id, language) in {
                ("Q030", "english"), ("Q006", "banglish"), ("Q023", "banglish")
            } else "human_partial_or_source_review"
            victim = next(index for index in range(len(current) - 1, -1, -1)
                          if current[index]["question_id"] not in mandatory)
            current[victim] = replacement
            present.add(question_id)
            current.sort(key=lambda item: item["question_id"])
        selected = [row for row in selected if row["language"] != language] + current
    return sorted(selected, key=lambda row: (base.LANGUAGES.index(row["language"]), row["question_id"]))


base.run_fingerprint = fingerprint
base.evaluate = evaluate
base.select_review = select_review


def finalize() -> None:
    base.finalize()
    summary = json.loads(base.SUMMARY_JSON.read_text(encoding="utf-8"))
    summary["notice"] = "STEP 7H DEVELOPMENT EVALUATION; NOT FINAL THESIS ACCURACY"
    summary["q007_adjudication"] = {
        "decision": "B", "evaluation_reference": "page-4 Engineering",
        "production_answer_unchanged_by_adjudication": True,
        "reference_agreement": {row["language"]: row["q007_reference_agreement"]
                                for row in base.read_csv(base.RESULT_CSV) if row["question_id"] == "Q007"},
    }
    returned = [row for row in base.read_csv(base.RESULT_CSV) if row["final_status"] == "ANSWER_RETURNED"]
    summary["returned_answer_validation_failures"] = [
        {"question_id": row["question_id"], "language": row["language"],
         "grounding_passed": row["grounding_validation_passed"],
         "language_passed": row["language_validation_passed"]}
        for row in returned if not base.truth(row["grounding_validation_passed"])
        or not base.truth(row["language_validation_passed"])
    ]
    base.SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "finalize"))
    action = parser.parse_args().action
    base.run() if action == "run" else finalize()
