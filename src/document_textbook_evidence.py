"""Find a uniquely labeled textbook within a uniquely owned course document.

The retrieved cover must identify the requested course. The indexed PDF, rather
than its filename or an evaluation reference, then supplies the labeled entry.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Sequence

from .evidence import Evidence, EvidenceAssessment, SupportStatus, analyze_query


_CODE = re.compile(
    r"\bCourse\s+(?:No\.?\s*/\s*Course\s+)?Code\s*:\s*"
    r"([A-Za-z]{2,12}(?:\s*\([A-Za-z]{2,12}\))?\s*\d{2,4}(?:\s*\([A-Za-z0-9]+\))?)\b",
    re.I,
)
_TEXTBOOK = re.compile(r"\bText\s*Book\s*\b", re.I)
_NEXT_SECTION = re.compile(r"\bReference\s+Books?\s*&\s*Materials\b", re.I)


def _compact(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value).upper()


def assess_document_textbook(
    question: str,
    retrieved: Sequence[dict[str, Any]],
    metadata: Sequence[dict[str, Any]],
) -> tuple[EvidenceAssessment, dict[str, Any] | None, str | None]:
    """Return one textbook entry only after a retrieved, unique course owner.

    A reference-book heading, second textbook entry, conflicting owner, or
    absent labeled boundary makes the lookup insufficient.
    """
    request = analyze_query(question)
    insufficient = EvidenceAssessment(request, SupportStatus.INSUFFICIENT, (),
                                      "No unique labeled textbook for the retrieved course document.")
    if (request.requested_field != "textbook" or len(request.entities) != 1
            or request.entities[0].kind != "course_code"):
        return insufficient, None, None
    wanted = request.entities[0].normalized
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in metadata:
        if item.get("document_id"):
            grouped[str(item["document_id"])].append(item)
    matches: list[tuple[Evidence, dict[str, Any], str]] = []
    for document_id, chunks in grouped.items():
        owners = {_compact(match.group(1)) for item in chunks
                  for match in _CODE.finditer(str(item.get("text") or ""))}
        if owners != {wanted}:
            continue
        seeds = [item for item in retrieved if str(item.get("document_id")) == document_id
                 and any(_compact(match.group(1)) == wanted
                         for match in _CODE.finditer(str(item.get("text") or "")))]
        if not seeds:
            continue
        pages: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in chunks:
            if isinstance(item.get("page"), int):
                pages[item["page"]].append(item)
        for page, page_chunks in pages.items():
            page_chunks.sort(key=lambda item: (int(item.get("word_start") or 0), str(item.get("chunk_id") or "")))
            page_text = " ".join(" ".join(str(item.get("text") or "").split()) for item in page_chunks)
            heading = _TEXTBOOK.search(page_text)
            if not heading:
                continue
            boundary = _NEXT_SECTION.search(page_text, heading.end())
            if not boundary:
                continue
            entry = page_text[heading.end():boundary.start()].strip()
            entry = re.sub(r"^(?:\d+\.\s*)+", "", entry)
            entry = re.sub(r"\s+\d+\.\s*$", "", entry).strip(" -–:.;")
            if (not 8 <= len(entry) <= 300 or re.search(r"(?:^|\s)\d+\.\s+\S", entry)):
                continue
            value_chunks = [item for item in page_chunks if entry[:20].casefold() in
                            str(item.get("text") or "").casefold()]
            if not value_chunks:
                continue
            primary = value_chunks[0]
            excerpt = page_text[heading.start():boundary.start()].strip()
            evidence = Evidence(
                document_id=document_id, source=str(primary.get("source") or "unknown"),
                relative_path=str(primary.get("relative_path") or primary.get("source") or ""),
                page=page, chunk_id=primary.get("chunk_id"), text=excerpt,
                excerpt=excerpt, retrieval_score=0.0,
                detected_entity=request.primary_entity, matched_entity=request.primary_entity,
                requested_field="textbook", matched_field="textbook",
                support_status=SupportStatus.SUPPORTED.value,
                continuation_chunk_ids=tuple(str(item.get("chunk_id")) for item in page_chunks
                                             if item.get("chunk_id") != primary.get("chunk_id")
                                             and (_TEXTBOOK.search(str(item.get("text") or ""))
                                                  or item in value_chunks)),
            )
            candidate = {**primary, "text": excerpt, "score": 0.0,
                         "field_types": ["textbook"],
                         "document_owner_chunk_id": seeds[0].get("chunk_id"),
                         "document_owner_page": seeds[0].get("page"),
                         "selection_method": "unique_course_document_labeled_textbook"}
            matches.append((evidence, candidate, entry))
    distinct = {value.casefold() for _, _, value in matches}
    if len(matches) != 1 or len(distinct) != 1:
        return insufficient, None, None
    evidence, candidate, value = matches[0]
    return (EvidenceAssessment(request, SupportStatus.SUPPORTED, (evidence,),
                               "Unique course owner and one labeled textbook entry in its PDF."),
            candidate, value)


__all__ = ["assess_document_textbook"]
