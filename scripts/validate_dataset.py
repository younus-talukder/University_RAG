from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "data" / "questions" / "questions.csv"
DOCUMENT_DIR = ROOT_DIR / "data" / "documents"

REQUIRED_COLUMNS = [
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

VALID_INTENTS = {
    "course_metadata",
    "course_title",
    "course_code",
    "course_type",
    "course_credit",
    "clo",
    "course_objective",
    "topics",
    "weekly_content",
    "midterm_marks",
    "final_exam_marks",
    "assessment",
    "mark_distribution",
    "prerequisite",
    "prospectus_metadata",
    "publisher",
    "edition",
    "admission",
    "degree_requirement",
    "semester_plan",
    "contact_info",
    "general",
}
VALID_DIFFICULTY = {"easy", "medium", "hard"}
VALID_STATUS = {"verified", "needs_review", "unsupported"}
PAGE_RE = re.compile(r"^\d+(;\d+)*$")
COURSE_RE = re.compile(r"^(GENERAL|[A-Z]{2,5}(?: \([A-Z]{2,5}\))? \d{3})$")
MOJIBAKE_RE = re.compile(r"(�|Ã|Â|â€™|â€œ|â€|ðŸ)")


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_rows() -> tuple[list[str], list[dict[str, str]]]:
    with DATASET_PATH.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def main() -> int:
    configure_stdout()
    errors: list[str] = []
    warnings: list[str] = []

    if not DATASET_PATH.exists():
        print(f"ERROR: dataset not found: {DATASET_PATH}")
        return 1

    header, rows = read_rows()
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in header]
    extra_columns = [column for column in header if column not in REQUIRED_COLUMNS]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    if extra_columns:
        warnings.append(f"Extra columns found: {', '.join(extra_columns)}")

    ids = [row.get("question_id", "").strip() for row in rows]
    id_counts = Counter(ids)
    for question_id, count in sorted(id_counts.items()):
        if not question_id:
            errors.append("Blank question_id found")
        elif count > 1:
            errors.append(f"Duplicate question_id {question_id}: {count} rows")
        elif not re.fullmatch(r"Q\d{3}", question_id):
            warnings.append(f"{question_id}: question_id should match Q001 format")

    question_columns = ["english_question", "bangla_question", "banglish_question"]
    for index, row in enumerate(rows, start=2):
        label = row.get("question_id") or f"row {index}"
        for column in question_columns:
            if not row.get(column, "").strip():
                errors.append(f"{label}: empty {column}")

        intent = row.get("intent", "").strip()
        if intent not in VALID_INTENTS:
            warnings.append(f"{label}: unrecognized intent '{intent}'")

        course = row.get("course_code", "").strip()
        if not COURSE_RE.fullmatch(course):
            errors.append(f"{label}: invalid course_code '{course}'")

        difficulty = row.get("difficulty", "").strip()
        if difficulty not in VALID_DIFFICULTY:
            errors.append(f"{label}: invalid difficulty '{difficulty}'")

        status = row.get("ground_truth_status", "").strip()
        if status not in VALID_STATUS:
            errors.append(f"{label}: invalid ground_truth_status '{status}'")

        source = row.get("expected_source", "").strip()
        if source and not (DOCUMENT_DIR / source).exists():
            errors.append(f"{label}: expected_source does not exist in data/documents/: {source}")

        expected_page = row.get("expected_page", "").strip()
        if expected_page and not PAGE_RE.fullmatch(expected_page):
            errors.append(f"{label}: invalid expected_page format '{expected_page}'")

        combined_text = " ".join(str(value) for value in row.values())
        if MOJIBAKE_RE.search(combined_text):
            warnings.append(f"{label}: possible mojibake or replacement character")

    question_tuples = [
        (
            row.get("english_question", "").strip().casefold(),
            row.get("bangla_question", "").strip(),
            row.get("banglish_question", "").strip().casefold(),
        )
        for row in rows
    ]
    for question_tuple, count in Counter(question_tuples).items():
        if count > 1:
            warnings.append(f"Duplicate multilingual question tuple appears {count} times: {question_tuple[0]}")

    print(f"Dataset: {DATASET_PATH}")
    print(f"Rows: {len(rows)}")
    print(f"Columns: {len(header)}")
    print(f"Errors: {len(errors)}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Warnings: {len(warnings)}")
    for warning in warnings:
        print(f"WARNING: {warning}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
