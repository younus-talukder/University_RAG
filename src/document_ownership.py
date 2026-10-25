"""R5b / R1: detect documents that describe exactly one course.

A course-outline PDF labels its course once ("Course No. / Course Code: CSE 212")
and every later page belongs to that course. A curriculum, prospectus or any
document with several labeled course blocks or a course table never qualifies.
The classification uses only the index's own text and metadata.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from .query_normalization import extract_course_entities
from .same_page_evidence import distinctive_match

COURSE_HEADER = re.compile(
    r"\bCourse\s+(?:No\.?\s*/\s*)?(?:Course\s+)?Code\s*:\s*([A-Za-z]{2,8}\s*\d{2,4}(?:\s*\([A-Za-z]\))?)\b", re.I)
COURSE_TABLE_HEADER = re.compile(r"course\s+code\s+course\s+title|course\s+no\.?\s+course\s+title", re.I)


def _compact(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()


@dataclass
class DocumentOwnership:
    document_id: str
    status: str                      # SINGLE_COURSE | MULTI_COURSE | UNCERTAIN
    course_code: str | None
    labeled_codes: dict[str, int] = field(default_factory=dict)
    course_table_rows: int = 0
    reason: str = ""


def classify_documents(chunks: Sequence[dict[str, Any]]) -> dict[str, DocumentOwnership]:
    labeled: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tables: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        document = str(chunk.get("document_id") or chunk.get("source") or "")
        text = str(chunk.get("text") or "")
        for match in COURSE_HEADER.finditer(text):
            labeled[document][_compact(match.group(1))] += 1
        if chunk.get("entity_type") == "course" and chunk.get("entity_id"):
            labeled[document][_compact(chunk["entity_id"])] += 1
        if COURSE_TABLE_HEADER.search(text):
            tables[document] += 1
        labeled[document]  # register the document even without labels
    result = {}
    for document, codes in labeled.items():
        codes = dict(codes)
        if tables[document]:
            status, code, reason = "MULTI_COURSE", None, "course table header present"
        elif len(codes) > 1:
            status, code, reason = "MULTI_COURSE", None, "several labeled course codes"
        elif len(codes) == 1:
            status, code, reason = "SINGLE_COURSE", next(iter(codes)), "exactly one labeled course code"
        else:
            status, code, reason = "UNCERTAIN", None, "no labeled course code"
        result[document] = DocumentOwnership(document, status, code, codes, tables[document], reason)
    return result


def single_course_owner(code: str, ownership: dict[str, DocumentOwnership]) -> str | None:
    """The one single-course document labeled ``code``, else None."""
    owners = [doc for doc, item in ownership.items() if item.status == "SINGLE_COURSE" and item.course_code == _compact(code)]
    return owners[0] if len(owners) == 1 else None


def annotate_document_ownership(pool: Sequence[dict[str, Any]], ownership: dict[str, DocumentOwnership],
                                question: str | None = None) -> list[dict[str, Any]]:
    """Mark chunks of single-course documents with their owner course code.

    A chunk that itself names a different course code is never marked. With a
    question, a chunk is marked only if it is distinctively relevant to it
    (same_page_evidence.distinctive_match), because inherited ownership would
    otherwise let the broad field patterns accept any passage of the document.
    """
    annotated = []
    for item in pool:
        owner = ownership.get(str(item.get("document_id") or item.get("source") or ""))
        if owner is None or owner.status != "SINGLE_COURSE":
            annotated.append(item)
            continue
        mentioned = {entity.compact for entity in extract_course_entities(str(item.get("text") or ""))}
        if mentioned - {owner.course_code}:
            annotated.append(item)
            continue
        if question is not None and not distinctive_match(question, owner.course_code, str(item.get("text") or "")):
            annotated.append(item)
            continue
        annotated.append({**item, "document_owner_course": owner.course_code})
    return annotated


__all__ = ["DocumentOwnership", "annotate_document_ownership", "classify_documents", "single_course_owner"]
