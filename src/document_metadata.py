"""Conservative label-based front-matter metadata, independent of course rows."""

from __future__ import annotations

import re


def metadata_fields_for_query(question: str) -> tuple[str, ...]:
    text = " ".join(question.casefold().split())
    document_context = bool(re.search(r"prospectus|প্রসপেক্টাস|document|নথি", text))
    edition = bool(re.search(r"\bedition\b|সংস্করণ", text))
    date = bool(re.search(r"\bdate\b|তারিখ", text))
    publisher = bool(re.search(r"\bpublish(?:ed|er|ing)?\b|প্রকাশক|প্রকাশ", text))
    disclaimer = bool(re.search(r"\bdisclaimer\b|ডিসক্লেইমার", text))
    requested: list[str] = []
    if edition:
        requested.append("edition")
    if date and (edition or document_context):
        requested.append("publication_date")
    if publisher and document_context:
        requested.append("published_by")
    if disclaimer:
        requested.append("publication_disclaimer")
    return tuple(requested)


def extract_document_metadata(text: str) -> dict[str, str]:
    clean = " ".join(str(text).split())
    found: dict[str, str] = {}
    edition = re.search(
        r"\bEdition\s*[:\-]?\s*(\d{1,3}(?:st|nd|rd|th)\s+Edition)\s*,?\s*"
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})",
        clean, re.I,
    )
    if edition:
        found["edition"] = " ".join(edition.group(1).split())
        found["publication_date"] = " ".join(edition.group(2).split())

    published = re.search(
        r"^Published\s+by\s+(.+?)(?=\bEdition\b|\bAddress\s+of\s+Correspondence\b|\bDisclaimer\b|$)",
        clean, re.I,
    )
    if published:
        value = published.group(1).strip(" ,.;:-")
        address = re.search(r"\b\d+[A-Za-z]?(?:/[A-Za-z0-9]+)?\s*,", value)
        if address:
            value = value[:address.start()].strip(" ,.;:-")
        if len(value.split()) >= 2:
            found["published_by"] = value

    disclaimer = re.search(r"\bDisclaimer\s*[:\-]?\s+(.+?)(?=\bPublished\s+by\b|\bEdition\b|$)", clean, re.I)
    if disclaimer:
        value = disclaimer.group(1).strip()
        if len(value.split()) >= 6 and re.search(r"[.!?]$", value):
            found["publication_disclaimer"] = value
    return found


__all__ = ["extract_document_metadata", "metadata_fields_for_query"]
