from __future__ import annotations

import csv
import re
import time
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


def reference_for_language(row: Dict[str, Any], language: str) -> str:
    language_keys = {
        "english": [
            "ground_truth_answer_en",
            "Expected Answer (Ground Truth)",
            "Expected Answer",
            "Reference Answer",
            "reference_answer",
            "answer",
        ],
        "bangla": [
            "ground_truth_answer_bn",
            "bengali_answer",
            "bangla_answer",
            "Expected Answer (Ground Truth)",
            "Expected Answer",
            "Reference Answer",
            "reference_answer",
            "answer",
        ],
        "banglish": [
            "ground_truth_answer_banglish",
            "banglish_answer",
            "Expected Answer (Ground Truth)",
            "Expected Answer",
            "Reference Answer",
            "reference_answer",
            "answer",
        ],
    }
    return first_non_empty(row, language_keys.get(language, []))


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
        english = first_non_empty(row, ["English Query", "english_query", "English question", "english_question", "question_en"])
        bengali = first_non_empty(row, ["Bengali Query", "bengali_query", "Bangla Query", "bangla_question", "question_bn"])
        banglish = first_non_empty(row, ["Banglish Query", "banglish_query", "banglish_question", "question_bl"])
        variants = [
            ("english", english),
            ("bangla", bengali),
            ("banglish", banglish),
        ]
        added_variant = False
        for language, question in variants:
            if question:
                prepared.append({
                    "question_id": first_non_empty(row, ["ID", "id", "Question ID", "question_id"]),
                    "question": question,
                    "reference_answer": reference_for_language(row, language),
                    "expected_language": language,
                    "expected_intent": first_non_empty(row, ["intent", "expected_intent"]),
                    "course_code": first_non_empty(row, ["course_code"]),
                    "ground_truth_status": first_non_empty(row, ["ground_truth_status"]),
                    "english_query": english,
                    "bengali_query": bengali,
                    "banglish_query": banglish,
                })
                added_variant = True

        if not added_variant:
            question = first_non_empty(row, ["Question", "question", "query"])
            prepared.append({
                "question_id": first_non_empty(row, ["ID", "id", "Question ID", "question_id"]),
                "question": question,
                "reference_answer": reference_for_language(row, detect_question_language(row)),
                "expected_language": detect_question_language(row),
                "expected_intent": first_non_empty(row, ["intent", "expected_intent"]),
                "course_code": first_non_empty(row, ["course_code"]),
                "ground_truth_status": first_non_empty(row, ["ground_truth_status"]),
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


def retrieval_source_columns(result: Dict[str, Any], slots: int = 3) -> Dict[str, Any]:
    sources = list(result.get("sources", []))
    columns: Dict[str, Any] = {}
    for index in range(slots):
        item = sources[index] if index < len(sources) else {}
        position = index + 1
        columns[f"retrieved_source_{position}"] = item.get("source")
        columns[f"retrieved_page_{position}"] = item.get("page")
        score = item.get("score")
        columns[f"retrieved_score_{position}"] = f"{score:.4f}" if score is not None else None
    return columns


def _tokens(text: str) -> set[str]:
    normalized = str(text).casefold()
    replacements = [
        (r"\bholo\b", "is"),
        (r"\bebong\b", "and"),
        (r"\bsathe\b", "with"),
        (r"\bo\b", "and"),
        (r"\bache\b", "included"),
        (r"\bbojha\b", "understanding"),
        (r"\bdescribe kora\b", "describing"),
        (r"\bdesign kora\b", "designing"),
        (r"\bdevelop kora\b", "developing"),
        (r"\buse kore\b", "using"),
        (r"\bcalculate kora\b", "calculating"),
        (r"\bporano hoy\b", "taught"),
    ]
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"(\w+)-(?:er|e|te|der|gulo)\b", r"\1", normalized)
    normalized = re.sub(r"[^A-Za-z0-9\u0980-\u09ff]+", " ", normalized)
    return {
        token
        for token in normalized.split()
        if len(token) > 1 and token not in {"what", "which", "where", "when", "how", "the", "and", "are", "is"}
    }


def answer_overlap(reference: str, generated: str) -> float:
    reference_tokens = _tokens(reference)
    generated_tokens = _tokens(generated)
    if not reference_tokens:
        return 0.0
    return len(reference_tokens & generated_tokens) / len(reference_tokens)


def evaluate_dataset(
    dataset_path: str | Path = QUESTIONS_DIR / "questions.csv",
    output_path: str | Path = RESULTS_DIR / "evaluation_results.csv",
    top_k: int = 3,
    limit: Optional[int] = None,
    use_generation: bool = False,
    use_answer_bank: bool = False,
) -> List[Dict[str, Any]]:
    rows = load_dataset(dataset_path)
    if limit is not None:
        rows = rows[:limit]

    evaluation_rows: List[Dict[str, Any]] = []
    for row in rows:
        question = row["question"]
        started_at = time.perf_counter()
        result = answer_question(
            question,
            top_k=top_k,
            use_generation=use_generation,
            use_answer_bank=use_answer_bank,
        )
        latency_seconds = time.perf_counter() - started_at
        retrieval_summary = summarize_retrieval_sources(result)
        overlap = answer_overlap(row.get("reference_answer", ""), result.get("answer", ""))

        output_row = {
            "question_id": row.get("question_id", ""),
            "question": question,
            "expected_language": row.get("expected_language", "unknown"),
            "course_code": row.get("course_code", ""),
            "expected_intent": row.get("expected_intent", ""),
            "detected_intent": result.get("intent", "unknown"),
            "ground_truth_status": row.get("ground_truth_status", ""),
            "reference_answer": row.get("reference_answer", ""),
            "generated_answer": result.get("answer", ""),
            "detected_language": result.get("detected_language", "unknown"),
            "response_language": result.get("response_language", "unknown"),
            "language_consistent": result.get("language_consistency", False),
            "language_consistency": result.get("language_consistency", False),
            "reference_token_overlap": f"{overlap:.3f}",
            "answer_mode": result.get("answer_mode", result.get("generation_mode", "unknown")),
            "generation_mode": result.get("generation_mode", "unknown"),
            "retrieved_source": retrieval_summary["retrieved_source"],
            "retrieved_page": retrieval_summary["retrieved_page"],
            "retrieval_score": retrieval_summary["retrieval_score"],
            "latency_seconds": f"{latency_seconds:.3f}",
            "answer_bank_enabled": use_answer_bank,
        }
        output_row.update(retrieval_source_columns(result, slots=3))
        evaluation_rows.append(output_row)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question_id",
                "question",
                "expected_language",
                "course_code",
                "expected_intent",
                "detected_intent",
                "ground_truth_status",
                "detected_language",
                "response_language",
                "language_consistent",
                "reference_answer",
                "generated_answer",
                "language_consistency",
                "reference_token_overlap",
                "answer_mode",
                "generation_mode",
                "retrieved_source_1",
                "retrieved_page_1",
                "retrieved_score_1",
                "retrieved_source_2",
                "retrieved_page_2",
                "retrieved_score_2",
                "retrieved_source_3",
                "retrieved_page_3",
                "retrieved_score_3",
                "latency_seconds",
                "answer_bank_enabled",
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
        "The current dataset includes expected page labels for the curriculum PDF. "
        "Retrieval metrics can use those labels, while factual answer quality should still be reviewed against the source document."
    )


__all__ = [
    "detect_question_language",
    "evaluate_dataset",
    "answer_overlap",
    "load_dataset",
    "reference_for_language",
    "retrieval_source_columns",
    "retrieval_framework_note",
    "summarize_retrieval_sources",
]
