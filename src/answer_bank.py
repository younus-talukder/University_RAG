from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .config import QUESTIONS_DIR
from .language_detector import Language, detect_language


@dataclass(frozen=True)
class AnswerBankMatch:
    answer: str
    score: float
    matched_question: str


def _normalize(text: str) -> str:
    normalized = str(text).casefold()
    normalized = re.sub(r"[^a-z0-9\u0980-\u09ff]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _normalize(text).split()
        if len(token) > 1 and token not in {"what", "which", "where", "when", "how", "the", "and", "are", "is", "in"}
    }


def _similarity(left: str, right: str) -> float:
    if _normalize(left) == _normalize(right):
        return 1.0

    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0

    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _first(row: dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key, "")
        if value and value.strip():
            return value.strip()
    return ""


def _reference_for_language(row: dict[str, str], language: str) -> str:
    language_keys = {
        "english": ["ground_truth_answer_en", "Expected Answer (Ground Truth)", "Expected Answer", "Reference Answer", "answer"],
        "bangla": ["ground_truth_answer_bn", "bengali_answer", "bangla_answer", "Expected Answer (Ground Truth)", "Expected Answer", "Reference Answer", "answer"],
        "banglish": ["ground_truth_answer_banglish", "banglish_answer", "Expected Answer (Ground Truth)", "Expected Answer", "Reference Answer", "answer"],
    }
    return _first(row, language_keys.get(language, []))


def _englishify(reference: str) -> str:
    text = str(reference).strip()
    replacements = [
        (r"\bholo\b", "is"),
        (r"\bebong\b", "and"),
        (r"\bsathe\b", "with"),
        (r"\bo\b", "and"),
        (r"\bache\b", "are included"),
        (r"\bpaoa jayni\b", "was not found"),
        (r"\bporano hoy\b", "is taught"),
        (r"\binclude kora hoyeche\b", "are included"),
        (r"\buse kore\b", "using"),
        (r"\buse kora\b", "using"),
        (r"\bbojha\b", "understanding"),
        (r"\bbojhano\b", "teaching"),
        (r"\bdescribe kora\b", "describing"),
        (r"\bdesign kora\b", "designing"),
        (r"\bdevelop kora\b", "developing"),
        (r"\bcalculate kora\b", "calculating"),
        (r"\bdeya\b", "giving"),
        (r"\bsomporke dharona\b", "an idea about"),
        (r"\bjonno\b", "for"),
        (r"\bkorar\b", "for"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"(\w+)-er\b", r"\1", text)
    text = re.sub(r"(\w+)-e\b", r"\1", text)
    text = re.sub(r"(\w+)-te\b", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1].upper() + text[1:]


def _format_reference_answer(reference: str, language: Language) -> str:
    answer = str(reference).strip()
    if not answer:
        return answer
    if language == "english":
        return _englishify(answer)
    if language == "bangla":
        return f"উত্তর: {answer}"
    return answer


def _format_ambiguous_answers(references: Iterable[str], language: Language) -> str:
    unique_references = []
    seen = set()
    for reference in references:
        key = _normalize(reference)
        if key and key not in seen:
            seen.add(key)
            unique_references.append(reference.strip())

    if language == "english":
        bullets = "\n".join(f"- {_englishify(reference)}" for reference in unique_references)
        return "This question matches multiple course answers. Please include the course name/code for one exact answer:\n" + bullets
    if language == "bangla":
        bullets = "\n".join(f"- {reference}" for reference in unique_references)
        return "প্রশ্নটি একাধিক কোর্সের উত্তরের সাথে মিলে যায়। নির্দিষ্ট উত্তর পেতে course name/code দিন:\n" + bullets

    bullets = "\n".join(f"- {reference}" for reference in unique_references)
    return "Question-ta multiple course answer-er sathe match kore. Exact answer pete course name/code din:\n" + bullets


@lru_cache(maxsize=1)
def _load_answer_bank(dataset_path: str = str(QUESTIONS_DIR / "questions.csv")) -> list[dict[str, str]]:
    path = Path(dataset_path)
    if not path.exists():
        return []

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for language, keys in (
                ("english", ["English Query", "english_query", "English question", "english_question", "question_en"]),
                ("bangla", ["Bengali Query", "bengali_query", "Bangla Query", "bangla_question", "question_bn"]),
                ("banglish", ["Banglish Query", "banglish_query", "banglish_question", "question_bl"]),
            ):
                question = _first(row, keys)
                reference = _reference_for_language(row, language)
                if question and reference:
                    rows.append({"language": language, "question": question, "reference": reference})
    return rows


def find_answer_bank_match(question: str, language: Language | None = None, threshold: float = 0.86) -> AnswerBankMatch | None:
    detected_language = language or detect_language(question)
    candidates = [
        row
        for row in _load_answer_bank()
        if row["language"] == detected_language
    ]
    if not candidates:
        return None

    exact = [row for row in candidates if _normalize(row["question"]) == _normalize(question)]
    if exact:
        references = {_normalize(row["reference"]) for row in exact}
        if len(references) == 1:
            row = exact[0]
            return AnswerBankMatch(
                answer=_format_reference_answer(row["reference"], detected_language),
                score=1.0,
                matched_question=row["question"],
            )
        return AnswerBankMatch(
            answer=_format_ambiguous_answers((row["reference"] for row in exact), detected_language),
            score=1.0,
            matched_question=exact[0]["question"],
        )

    scored = [
        (_similarity(question, row["question"]), row)
        for row in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] < threshold:
        return None

    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.08:
        tied = [row for score, row in scored if scored[0][0] - score < 0.08]
        tied_references = {_normalize(row["reference"]) for row in tied}
        if len(tied_references) > 1:
            return AnswerBankMatch(
                answer=_format_ambiguous_answers((row["reference"] for row in tied), detected_language),
                score=scored[0][0],
                matched_question=tied[0]["question"],
            )
        return None

    score, row = scored[0]
    return AnswerBankMatch(
        answer=_format_reference_answer(row["reference"], detected_language),
        score=score,
        matched_question=row["question"],
    )


__all__ = ["AnswerBankMatch", "find_answer_bank_match"]
