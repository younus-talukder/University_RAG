"""Bind a labeled course title to a labeled course code on one PDF page.

This is a structural lookup over indexed PDF chunks. It never reads questions,
answers, filenames, or labels from an evaluation dataset.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from typing import Any, Sequence

from .evidence import Evidence, EvidenceAssessment, QueryRequest, SupportStatus, analyze_query


_CODE = r"[A-Za-z]{2,12}(?:\s*\([A-Za-z]{2,12}\))?\s*\d{2,4}(?:\s*\([A-Za-z0-9]+\))?"
_CODE_LABEL = re.compile(
    rf"\bCourse\s+(?:No\.?\s*/\s*Course\s+)?Code\s*:\s*(?P<code>{_CODE})\b", re.I
)
_TITLE_LABEL = re.compile(r"\bCourse\s+Title\s*:\s*", re.I)
_NEXT_FIELD = re.compile(
    r"\s+(?:\d+(?:\.\d+)*[.)]\s+[A-Za-z]|"
    r"(?:Course\s+(?:Code|Title|Type|Credit|Instructor|Objective)|"
    r"Credit\s+(?:Value|Hour)|Semester|Level|Name\s*&\s*Designation)\s*:)" , re.I
)


def _compact(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value).upper()


def _title_labels(text: str) -> list[tuple[str, str]]:
    found = []
    for match in _TITLE_LABEL.finditer(text):
        tail = text[match.end():]
        boundary = _NEXT_FIELD.search(tail)
        value = " ".join((tail[:boundary.start()] if boundary else tail).split()).strip(" .;:-–—")
        if 2 <= len(value) <= 120 and re.search(r"[A-Za-z\u0980-\u09ff]", value):
            found.append((value, text[match.start():match.end()] + value))
    return found


def _page_key(item: dict[str, Any]) -> tuple[str, str, int] | None:
    document_id = str(item.get("document_id") or "")
    source = str(item.get("source") or "")
    page = item.get("page")
    if not document_id or not source or not isinstance(page, int) or page < 1:
        return None
    return document_id, source, page


def assess_course_title(
    question: str,
    retrieved: Sequence[dict[str, Any]],
    metadata: Sequence[dict[str, Any]],
) -> EvidenceAssessment:
    """Use only uniquely bound code/title labels from the same document page.

    A retrieved chunk must explicitly name the requested course code. The
    rest of that page may supply the title, but another labeled course code or
    a different labeled title on the page makes the pairing ambiguous.
    """
    request: QueryRequest = analyze_query(question)
    if request.requested_field != "course_title" or len(request.entities) != 1 or request.entities[0].kind != "course_code":
        return EvidenceAssessment(request, SupportStatus.INSUFFICIENT, (),
                                  "A course-title lookup needs one explicit course code.")
    wanted = request.entities[0].normalized
    pages: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in metadata:
        key = _page_key(item)
        if key is not None:
            pages[key].append(item)
    seeded: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in retrieved:
        key = _page_key(item)
        if key in pages and any(_compact(m.group("code")) == wanted for m in _CODE_LABEL.finditer(str(item.get("text") or ""))):
            seeded.setdefault(key, item)

    supported: list[tuple[str, Evidence]] = []
    for key, seed in seeded.items():
        chunks = pages[key]
        codes = {_compact(m.group("code")) for item in chunks
                 for m in _CODE_LABEL.finditer(str(item.get("text") or ""))}
        if codes != {wanted}:
            continue
        title_spans = [(value, span, item) for item in chunks
                       for value, span in _title_labels(str(item.get("text") or ""))]
        titles = {" ".join(value.casefold().split()) for value, _, _ in title_spans}
        if len(titles) != 1:
            continue
        value, title_span, title_item = title_spans[0]
        code_match = next(m for m in _CODE_LABEL.finditer(str(seed.get("text") or ""))
                          if _compact(m.group("code")) == wanted)
        excerpt = f"{code_match.group(0)} {title_span}"
        title_chunk = str(title_item.get("chunk_id") or "")
        code_chunk = str(seed.get("chunk_id") or "")
        evidence = Evidence(
            document_id=key[0], source=key[1],
            relative_path=str(seed.get("relative_path") or title_item.get("relative_path") or key[1]),
            page=key[2], chunk_id=seed.get("chunk_id"), text=excerpt, excerpt=excerpt,
            retrieval_score=float(seed.get("score") or 0.0),
            detected_entity=request.primary_entity, matched_entity=request.primary_entity,
            requested_field="course_title", matched_field="course_title",
            support_status=SupportStatus.SUPPORTED.value,
            continuation_chunk_ids=(title_chunk,) if title_chunk and title_chunk != code_chunk else (),
        )
        supported.append((value, evidence))
    if not supported:
        return EvidenceAssessment(request, SupportStatus.INSUFFICIENT, (),
                                  "No uniquely paired course code and title labels on a retrieved document page.")
    distinct = {" ".join(value.casefold().split()) for value, _ in supported}
    if len(distinct) > 1:
        conflicting = tuple(replace(evidence, support_status=SupportStatus.CONFLICTING.value)
                            for _, evidence in supported)
        return EvidenceAssessment(request, SupportStatus.CONFLICTING, conflicting,
                                  "Labeled titles disagree for the requested course.",
                                  tuple(sorted({value for value, _ in supported})))
    return EvidenceAssessment(request, SupportStatus.SUPPORTED,
                              tuple(evidence for _, evidence in supported),
                              "The requested course code and one title are explicitly labeled on the same PDF page.")


def title_value_from_excerpts(excerpts: Sequence[str]) -> str | None:
    values = {value for excerpt in excerpts for value, _ in _title_labels(excerpt)}
    return next(iter(values)) if len({value.casefold() for value in values}) == 1 else None


def bound_course_title_value(assessment: EvidenceAssessment) -> str | None:
    if assessment.status is not SupportStatus.SUPPORTED or assessment.request.requested_field != "course_title":
        return None
    return title_value_from_excerpts([evidence.excerpt for evidence in assessment.evidence])


__all__ = ["assess_course_title", "bound_course_title_value", "title_value_from_excerpts"]
