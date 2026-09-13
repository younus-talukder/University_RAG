from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.fast_answer import DEFAULT_COURSE_SOURCE, detect_runtime_intent

OUTPUT_COLUMNS = [
    "question_id",
    "course_code",
    "intent",
    "difficulty",
    "english_question",
    "bangla_question",
    "banglish_question",
    "ground_truth_answer_en",
    "ground_truth_answer_bn",
    "ground_truth_answer_banglish",
    "expected_source",
    "expected_page",
    "ground_truth_status",
    "notes",
]


def _clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _course_code(question: str) -> str:
    normalized = re.sub(r"\s+", " ", question.upper())
    eng_match = re.search(r"\bENG\s*\(?\s*CSE\s*\)?\s*101\b", normalized)
    if eng_match:
        return "ENG (CSE) 101"
    match = re.search(r"\b([A-Z]{2,5})\s*(\d{3})\b", normalized)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return "GENERAL"


def _intent(question: str) -> str:
    lowered = question.casefold()
    if "prospectus" in lowered and "title" in lowered:
        return "prospectus_metadata"
    if "published" in lowered or "publisher" in lowered:
        return "publisher"
    if "edition" in lowered:
        return "edition"
    if "prerequisite" in lowered:
        return "prerequisite"
    if "semester" in lowered:
        return "semester_plan"
    detected = detect_runtime_intent(question)
    return "general" if detected == "unknown" else detected


def import_workbook(source_path: Path, output_path: Path, expected_source: str) -> int:
    df = pd.read_excel(source_path)
    rows = []
    for _, row in df.iterrows():
        english_question = _clean(row.get("english_question"))
        rows.append({
            "question_id": _clean(row.get("question_id")),
            "course_code": _course_code(english_question),
            "intent": _intent(english_question),
            "difficulty": "medium",
            "english_question": english_question,
            "bangla_question": _clean(row.get("bangla_question")),
            "banglish_question": _clean(row.get("banglish_question")),
            "ground_truth_answer_en": _clean(row.get("ground_truth_answer_en")),
            "ground_truth_answer_bn": _clean(row.get("ground_truth_answer_bn")),
            "ground_truth_answer_banglish": _clean(row.get("ground_truth_answer_banglish")),
            "expected_source": expected_source,
            "expected_page": _clean(row.get("expected_page_of_answer") or row.get("expected_page")),
            "ground_truth_status": "verified",
            "notes": f"Imported from {source_path.name}.",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(output_path, index=False, encoding="utf-8-sig")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import the UAP multilingual RAG question workbook into the canonical CSV.")
    parser.add_argument("source_path", type=Path, help="Source .xlsx workbook.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "data" / "questions" / "questions.csv",
        help="Canonical CSV output path.",
    )
    parser.add_argument(
        "--expected-source",
        default=DEFAULT_COURSE_SOURCE,
        help="PDF filename used as the expected retrieval source.",
    )
    args = parser.parse_args()

    count = import_workbook(args.source_path, args.output, args.expected_source)
    print(f"Imported {count} rows to {args.output}")


if __name__ == "__main__":
    main()
