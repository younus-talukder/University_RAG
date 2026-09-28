"""Field values owned by one verified course row, never nearby table totals."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Sequence

from .query_normalization import extract_course_entities


_CODE = r"[A-Z]{2,5}\s*\d{3}(?:\s*\([A-Z]\))?"
_PREREQUISITE = rf"(?:Nil|None|N/A|{_CODE})(?:\s*,\s*{_CODE})*"
_LABEL = r"pre\s*[- ]?\s*requisites?"


def requested_course_code(question: str) -> str | None:
    entities = extract_course_entities(question)
    return entities[0].compact if len(entities) == 1 else None


def _compact_code(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value).upper()


def _row_values(text: str, code: str, field: str) -> list[str]:
    values: list[str] = []
    clean = " ".join(text.split())
    # Labeled syllabus records are bounded by the next Course Code label.
    for match in re.finditer(rf"\bCourse\s+Code\s*:\s*({_CODE})\b", clean, re.I):
        if _compact_code(match.group(1)) != code:
            continue
        next_row = re.search(r"\bCourse\s+Code\s*:", clean[match.end():], re.I)
        end = match.end() + next_row.start() if next_row else len(clean)
        row = clean[match.end():end]
        pattern = (r"\bCredit(?:s|\s+Value)?\s*:\s*(\d+(?:\.\d+)?)\b" if field == "credits"
                   else rf"\b{_LABEL}\s*[:=-]?\s*({_PREREQUISITE})\b")
        value = re.search(pattern, row, re.I)
        if value:
            values.append(" ".join(value.group(1).split()))
    # Compact curriculum rows: CODE TITLE CREDIT PREREQUISITE.
    for match in re.finditer(
        rf"(?<![A-Za-z0-9])(?P<code>{_CODE})\s+"
        rf"(?P<title>[A-Za-z][A-Za-z: &./-]{{2,110}}?)\s+"
        rf"(?P<credits>\d{{1,2}}(?:\.\d{{1,2}})?)\s+"
        rf"(?P<prerequisite>{_PREREQUISITE})(?![A-Za-z0-9])",
        clean, re.I,
    ):
        if _compact_code(match.group("code")) == code:
            values.append(" ".join(match.group(field).split()))
    # A direct course-specific sentence is an equally valid owned relation.
    if field == "prerequisite":
        for match in re.finditer(rf"\b(?:Course\s+)?({_CODE})\s+has\s+no\s+prerequisite\b", clean, re.I):
            if _compact_code(match.group(1)) == code:
                values.append("Nil")
    return values


def course_field_value(question: str, field: str, excerpts: Sequence[str]) -> str | None:
    """Return a value only when a single requested course owns it in a verified row."""
    code = requested_course_code(question)
    if not code or field not in {"credits", "prerequisite"}:
        return None
    def normalize(value: str) -> tuple[str, str]:
        if field == "credits":
            return str(Decimal(value)), value
        if value.casefold() in {"nil", "none", "n/a"}:
            return "none", "Nil"
        codes = re.findall(_CODE, value, re.I)
        if codes and re.sub(r"\s*,\s*", "", value).replace(" ", "") == "".join(code.replace(" ", "") for code in codes):
            canonical = ", ".join(re.sub(r"\s+", " ", code).upper().replace(" ", "", 1) for code in codes)
            # Restore a single separator between the alpha prefix and digits.
            canonical = re.sub(r"\b([A-Z]{2,5})(\d{3})", r"\1 \2", canonical)
            return canonical.casefold(), canonical
        return value.casefold(), value

    values = {key: display for excerpt in excerpts for value in _row_values(excerpt, code, field)
              for key, display in (normalize(value),)}
    return next(iter(values.values())) if len(values) == 1 else None


__all__ = ["course_field_value", "requested_course_code"]
