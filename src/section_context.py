"""R3: label a chunk with the numbered section / table header that governs it.

PDF tables and lists are chunked row by row, so a row chunk ("Problem-Based
Learning Project Scheduling, Feasibility Study") loses its column header
("18. Teaching-Learning Strategies: Strategies Topics"), which may sit in an
earlier chunk or on an earlier page. This module walks the index's own
``previous_chunk_id`` chain back to the nearest numbered heading of the same
document and returns it as a label. The label is provenance only: it never
supplies a course code (labels containing a course label or code are refused),
and every gate still judges the passage.

The label is gate-side context only. ``strip_section_labels`` removes it after
the evidence decision, so extractors, validators and the generator see the
original passage text (a live check showed a labeled table row persuading the
generator to read column headers as the answer).
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Sequence

from .query_normalization import extract_course_entities

NUMBERED_HEADING = re.compile(r"(?:^|\s)(\d{1,2})\.\s+([A-Z][^:.\n]{2,90}?)\s*:\s*")
COURSE_LABEL = re.compile(r"\bCourse\s+(?:No\.?\s*/\s*)?(?:Course\s+)?Code\b", re.I)
MAX_STEPS = 15
MAX_PAGES_BACK = 2
MAX_TAIL_WORDS = 20


def _heading_label(text: str) -> str | None:
    """Return the last numbered heading in ``text`` plus a short column tail."""
    matches = list(NUMBERED_HEADING.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    title = f"{match.group(1)}. {match.group(2).strip()}:"
    if COURSE_LABEL.search(title) or extract_course_entities(title):
        return None
    tail = " ".join(text[match.end():].split())
    words = tail.split()
    # A short, unpunctuated tail is a column-label row; a sentence is content.
    if words and len(words) <= MAX_TAIL_WORDS and not re.search(r"[.!?]", tail) and not extract_course_entities(tail):
        title = f"{title} {tail}"
    return title


def governing_section(item: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> tuple[str, str] | None:
    """(label, header_chunk_id) for the heading governing ``item``, or None."""
    own = by_id.get(str(item.get("chunk_id")), item)
    if NUMBERED_HEADING.search(str(own.get("text") or "")):
        return None  # the chunk carries its own heading
    document, page = own.get("document_id"), own.get("page")
    current = own
    for _ in range(MAX_STEPS):
        previous = by_id.get(str(current.get("previous_chunk_id") or ""))
        if not previous or previous.get("document_id") != document:
            return None
        if isinstance(page, int) and isinstance(previous.get("page"), int) and page - previous["page"] > MAX_PAGES_BACK:
            return None
        if NUMBERED_HEADING.search(str(previous.get("text") or "")):
            label = _heading_label(str(previous.get("text") or ""))
            return (label, str(previous.get("chunk_id"))) if label else None
        current = previous
    return None


def annotate_section_context(pool: Sequence[dict[str, Any]], metadata: Sequence[dict[str, Any]], *,
                             label_text: bool = True) -> list[dict[str, Any]]:
    """Copy pool items with their governing section label and header chunk id.

    With ``label_text`` the label is prefixed to the text the gates judge;
    callers must not do that for broad (general-field) questions, where a
    column label such as "Teaching Strategy(s)" would count as matched query
    terms on every row of every such table.
    """
    by_id = {str(item.get("chunk_id")): item for item in metadata if item.get("chunk_id")}
    annotated = []
    for item in pool:
        found = governing_section(item, by_id)
        if found is None:
            annotated.append(item)
            continue
        label, header_id = found
        text = f"[{label}] {item.get('text', '')}" if label_text else item.get("text", "")
        annotated.append({**item, "text": text, "section_label": label, "section_header_chunk_id": header_id})
    return annotated


def _strip(text: str, label: str | None) -> str:
    prefix = f"[{label}] " if label else ""
    return text[len(prefix):] if prefix and text.startswith(prefix) else text


def strip_section_labels(assessment, retrieved: Sequence[dict[str, Any]]):
    """Remove gate-side labels from verified evidence and retrieved items."""
    labels = {str(item.get("chunk_id")): item.get("section_label") for item in retrieved if item.get("section_label")}
    if not labels:
        return assessment, list(retrieved)
    evidence = tuple(replace(e, text=_strip(e.text, labels.get(str(e.chunk_id))), excerpt=_strip(e.excerpt, labels.get(str(e.chunk_id))))
                     for e in assessment.evidence)
    items = [{**item, "text": _strip(str(item.get("text") or ""), item.get("section_label"))} if item.get("section_label") else item
             for item in retrieved]
    return replace(assessment, evidence=evidence), items


__all__ = ["annotate_section_context", "governing_section", "strip_section_labels"]
