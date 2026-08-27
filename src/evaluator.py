from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import QUESTIONS_DIR, RESULTS_DIR
from .pipeline import answer_question


def first_non_empty(mapping: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = mapping.get(key, "")
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def detect_question_language(row: Dict[str, Any]) -> str:
    if row.get("English Query", "").strip():
        return "english"
    if row.get("Bengali Query", "").strip():
        return "bangla"
    if row.get("Banglish Query", "").strip():
        return "banglish"
    if row.get("Question", "").strip():
        return "unknown"
    return "unknown"


def load_dataset(dataset_path: str | Path = QUESTIONS_DIR / "questions.csv") -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Question dataset not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Dataset is empty: {path}")

    prepared: List[Dict[str, Any]] = []
    for row in rows:
        english = first_non_empty(row, ["English Query", "english_query", "English question", "question_en"])
        bengali = first_non_empty(row, ["Bengali Query", "bengali_query", "Bangla Query", "question_bn"])
        banglish = first_non_empty(row, ["Banglish Query", "banglish_query", "question_bl"])
        reference = first_non_empty(row, ["Expected Answer (Ground Truth)", "Expected Answer", "Reference Answer", "reference_answer", "answer"])

        variants = [
            ("english", english),
            ("bangla", bengali),
            ("banglish", banglish),
        ]
        added_variant = False
        for language, question in variants:
            if question:
                prepared.append({
                    "question": question,
                    "reference_answer": reference,
                    "expected_language": language,
                    "english_query": english,
                    "bengali_query": bengali,
                    "banglish_query": banglish,
                })
                added_variant = True

        if not added_variant:
            question = first_non_empty(row, ["Question", "question", "query"])
            prepared.append({
                "question": question,
                "reference_answer": reference,
                "expected_language": detect_question_language(row),
                "english_query": english,
                "bengali_query": bengali,
                "banglish_query": banglish,
            })

    return prepared


def summarize_retrieval_sources(result: Dict[str, Any]) -> Dict[str, Optional[str]]:
    sources = result.get("sources", [])
    retrieved_sources = "; ".join(str(item.get("source")) for item in sources if item.get("source"))
    retrieved_pages = "; ".join(str(item.get("page")) for item in sources if item.get("page") is not None)
    retrieval_score = "; ".join(f"{item.get('score', 0.0):.4f}" for item in sources if item.get("score") is not None)

    return {
        "retrieved_source": retrieved_sources or None,
        "retrieved_page": retrieved_pages or None,
        "retrieval_score": retrieval_score or None,
    }


def evaluate_dataset(dataset_path: str | Path = QUESTIONS_DIR / "questions.csv", output_path: str | Path = RESULTS_DIR / "evaluation_results.csv", top_k: int = 3, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = load_dataset(dataset_path)
    if limit is not None:
        rows = rows[:limit]

    evaluation_rows: List[Dict[str, Any]] = []
    for row in rows:
        question = row["question"]
        result = answer_question(question, top_k=top_k)
        retrieval_summary = summarize_retrieval_sources(result)

        evaluation_rows.append({
            "question": question,
            "reference_answer": row.get("reference_answer", ""),
            "generated_answer": result.get("answer", ""),
            "detected_language": result.get("detected_language", "unknown"),
            "response_language": result.get("response_language", "unknown"),
            "language_consistency": result.get("language_consistency", False),
            "factual_correctness": "manual_review_required",
            "relevance": "manual_review_required",
            "completeness": "manual_review_required",
            "groundedness": "manual_review_required",
            "retrieved_source": retrieval_summary["retrieved_source"],
            "retrieved_page": retrieval_summary["retrieved_page"],
            "retrieval_score": retrieval_summary["retrieval_score"],
        })

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question",
                "reference_answer",
                "generated_answer",
                "detected_language",
                "response_language",
                "language_consistency",
                "factual_correctness",
                "relevance",
                "completeness",
                "groundedness",
                "retrieved_source",
                "retrieved_page",
                "retrieval_score",
            ],
        )
        writer.writeheader()
        writer.writerows(evaluation_rows)

    return evaluation_rows


def retrieval_framework_note() -> str:
    return (
        "Ground-truth document/page labels are not available in the current dataset. "
        "For proper retrieval metrics such as Hit@K, Recall@K, and MRR, source annotations are required. "
        "The current prototype therefore provides qualitative retrieval inspection via the saved source/page metadata."
    )


__all__ = [
    "detect_question_language",
    "evaluate_dataset",
    "load_dataset",
    "retrieval_framework_note",
    "summarize_retrieval_sources",
]
