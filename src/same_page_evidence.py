"""Bind a passage to a uniquely labeled course header on the same PDF page.

The header supplies provenance only. It contributes no answer value, and no
course or document name is encoded in this rule.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Sequence

from .evidence import EvidenceAssessment, SupportStatus, analyze_query, assess_evidence


_COURSE_LABEL = re.compile(
    r"\bCourse\s+(?:No\.?\s*/\s*)?(?:Course\s+)?Code\s*:\s*"
    r"([A-Za-z]{2,8}\s*\d{2,4}(?:\s*\([A-Za-z]\))?)\b",
    re.I,
)
_STOP = {"what", "which", "when", "where", "who", "how", "does", "the",
         "this", "that", "with", "from", "into", "have", "course", "named",
         "listed", "about", "described", "related", "area", "task"}


def _compact(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value).upper()


def _terms(question: str, code: str) -> set[str]:
    tokens = {token.casefold() for token in re.findall(r"[A-Za-z]{3,}", question)}
    tokens.difference_update(_STOP)
    tokens.difference_update(token.casefold() for token in re.findall(r"[A-Za-z]{2,}", code))
    return tokens


def assess_unique_same_page_course_passage(
    question: str,
    original_pool: Sequence[dict[str, Any]],
    metadata: Sequence[dict[str, Any]],
    *,
    limit: int = 15,
) -> tuple[EvidenceAssessment, dict[str, Any]] | None:
    """Return one verified passage or abstain when page ownership is unclear."""
    request = analyze_query(question)
    if (request.ambiguous or request.requested_field != "general"
            or len(request.entities) != 1 or request.entities[0].kind != "course_code"):
        return None
    return _unique_same_page_join(question, request, original_pool, metadata, limit)


# R5a (SAME_PAGE_LABELED_FIELD_JOIN): the same ownership rule for a labeled field
# (objective, CLO, weekly plan, ...). The joined passage must still pass the
# unchanged entity + field gate in assess_evidence.
LABELED_FIELD_EXCLUDED = {"general", "course_title", "document_metadata"}


def assess_unique_same_page_labeled_field_passage(
    question: str,
    original_pool: Sequence[dict[str, Any]],
    metadata: Sequence[dict[str, Any]],
    *,
    limit: int = 15,
) -> tuple[EvidenceAssessment, dict[str, Any]] | None:
    request = analyze_query(question)
    if (request.ambiguous or request.requested_field in LABELED_FIELD_EXCLUDED
            or len(request.entities) != 1 or request.entities[0].kind != "course_code"):
        return None
    return _unique_same_page_join(question, request, original_pool, metadata, limit, strict=True)


# Strict mode (R5a only): function words never count as distinctive terms, every
# identifier family the question names (PLO, CLO, CT, ...) must occur in the
# candidate itself, and only candidates that already pass the unchanged gate
# compete for the single-winner rule.
_FUNCTION_WORDS = {"are", "and", "for", "its", "was", "were", "has", "had", "can", "per", "any", "all", "not", "but"}
_ACRONYM = re.compile(r"(?<![A-Za-z])([A-Z]{2,6})(?:s|-?\d+)?(?![A-Za-z])")


def _acronyms(question: str, code_value: str) -> set[str]:
    code_letters = {token.upper() for token in re.findall(r"[A-Za-z]{2,}", code_value)}
    return {token for token in _ACRONYM.findall(question)} - code_letters


_NUMBERED_FAMILIES = {"CLO", "PLO", "PO", "CO", "CT"}
_NUMBERED_ID = re.compile(r"(?<![A-Za-z])([A-Za-z]{2,8})\s*-?\s*0*(\d{1,2})(?!\d)")


def distinctive_match(question: str, code_value: str, text: str) -> bool:
    """Strict relevance test for passages that carry no course code of their own.

    At least two distinctive question terms (and at least half of them), every
    identifier family (PLO, CLO, ...) and every numbered identifier ("CLO2",
    "Week 7") named by the question must occur in the passage; a family asked
    for without a number must occur with one.
    """
    terms = _terms(question, code_value) - _FUNCTION_WORDS
    # "CLO" is present in "CLO2": a word may be followed by its number.
    present = {term for term in terms if re.search(rf"\b{re.escape(term)}(?=\b|\d)", text, re.I)}
    if len(present) < 2 or len(present) * 2 < len(terms):
        return False
    if any(not re.search(rf"(?<![A-Za-z]){acronym}(?![a-z])", text) for acronym in _acronyms(question, code_value)):
        return False
    code_letters = {token.casefold() for token in re.findall(r"[A-Za-z]{2,}", code_value)}
    # An outcome/assessment identifier family named without a number ("Which
    # PLO ...") is the value being asked for, so the passage must contain a
    # numbered member of it ("PLO3").
    numbered = {prefix.upper() for prefix, _ in _NUMBERED_ID.findall(question)}
    for acronym in (_acronyms(question, code_value) & _NUMBERED_FAMILIES) - numbered:
        if not re.search(rf"(?<![A-Za-z]){acronym}\s*-?\s*\d", text):
            return False
    for prefix, number in _NUMBERED_ID.findall(question):
        if prefix.casefold() in code_letters:
            continue
        if not re.search(rf"(?<![A-Za-z]){re.escape(prefix)}\s*-?\s*0*{int(number)}(?!\d)", text, re.I):
            return False
    return True


def _unique_same_page_join(question, request, original_pool, metadata, limit, strict=False):
    code = request.entities[0].normalized
    terms = _terms(question, request.entities[0].value)
    if strict:
        terms -= _FUNCTION_WORDS
    if len(terms) < 2:
        return None

    pages = {(item.get("document_id"), item.get("page")) for item in original_pool[:limit]}
    headers: dict[tuple[Any, Any], list[tuple[str, str, str]]] = {}
    for item in metadata:
        key = (item.get("document_id"), item.get("page"))
        if key not in pages:
            continue
        for match in _COURSE_LABEL.finditer(str(item.get("text") or "")):
            headers.setdefault(key, []).append(
                (_compact(match.group(1)), str(item.get("chunk_id") or ""), match.group(0)))

    candidates: list[tuple[int, int, dict[str, Any], str, str]] = []
    for rank, item in enumerate(original_pool[:limit]):
        key = (item.get("document_id"), item.get("page"))
        page_headers = headers.get(key, [])
        if {entry[0] for entry in page_headers} != {code}:
            continue
        text = str(item.get("text") or "")
        if not text or _COURSE_LABEL.search(text):
            continue
        present = {term for term in terms if re.search(rf"\b{re.escape(term)}\b", text, re.I)}
        if len(present) < 2 or len(present) * 2 < len(terms):
            continue
        if strict and not distinctive_match(question, request.entities[0].value, text):
            continue
        header = next(entry for entry in page_headers if entry[1] != str(item.get("chunk_id")))
        if strict and assess_evidence(question, [{**item, "text": f"{header[2]} {text}"}]).status is not SupportStatus.SUPPORTED:
            continue
        candidates.append((len(present), rank, item, header[1], header[2]))
    if not candidates:
        return None
    candidates.sort(key=lambda row: (-row[0], row[1]))
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    _, _, item, header_id, label = candidates[0]
    joined = {**item, "text": f"{label} {item['text']}",
              "same_page_owner_chunk_id": header_id}
    assessment = assess_evidence(question, [joined])
    if assessment.status is not SupportStatus.SUPPORTED or len(assessment.evidence) != 1:
        return None
    evidence = replace(assessment.evidence[0], continuation_chunk_ids=(header_id,))
    return replace(assessment, evidence=(evidence,)), joined


__all__ = ["assess_unique_same_page_course_passage", "assess_unique_same_page_labeled_field_passage", "distinctive_match"]
