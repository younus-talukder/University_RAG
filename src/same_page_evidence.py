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
    code = request.entities[0].normalized
    terms = _terms(question, request.entities[0].value)
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
        header = next(entry for entry in page_headers if entry[1] != str(item.get("chunk_id")))
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


__all__ = ["assess_unique_same_page_course_passage"]
