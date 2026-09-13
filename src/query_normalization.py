from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# Deliberately small: these are high-confidence Latin-script Bangla function-word
# variants, not a translation dictionary or a dataset-specific query map.
BANGLISH_VARIANTS = {
    "ase": "ache",
    "hbe": "hobe",
    "jnno": "jonno",
    "kii": "ki",
    "koy": "koto",
    "krte": "korte",
    "lage": "lagbe",
}

NON_COURSE_PREFIXES = {
    "ABOVE", "AND", "AT", "BELOW", "CLO", "FOR", "FROM", "GRADE",
    "IN", "IS", "LEVEL", "MANY", "OF", "ON", "OR", "PAGE", "SEMESTER",
    "THE", "TO", "WEEK", "WITH", "YEAR",
}

COMPOSITE_COURSE_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{2,12})\s*\(\s*([A-Za-z]{2,12})\s*\)\s*[- ]?\s*"
    r"(\d{2,4})(?:(?:\s*\(\s*([A-Za-z0-9]{1,4})\s*\))|([A-Za-z]))?(?![A-Za-z0-9])"
)
MULTI_PREFIX_COURSE_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Z]{2,12})\s+([A-Z]{2,12})\s+(\d{2,4})"
    r"(?:(?:\s*\(\s*([A-Za-z0-9]{1,4})\s*\))|([A-Za-z]))?(?![A-Za-z0-9])"
)
SIMPLE_COURSE_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{2,12})\s*[- ]?\s*(\d{2,4})"
    r"(?:(?:\s*\(\s*([A-Za-z0-9]{1,4})\s*\))|([A-Za-z]))?(?![A-Za-z0-9])"
)


@dataclass(frozen=True)
class CourseEntity:
    raw: str
    canonical: str
    compact: str
    span: tuple[int, int]


@dataclass(frozen=True)
class QueryRepresentations:
    original_query: str
    normalized_query: str
    retrieval_query: str


def normalize_unicode(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).translate(BENGALI_DIGITS)
    value = re.sub(r"[\u2010-\u2015\u2212]", "-", value)
    value = re.sub(r"[\u00a0\u2007\u202f]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def extract_course_entities(text: str) -> tuple[CourseEntity, ...]:
    value = normalize_unicode(text)
    matches: list[CourseEntity] = []
    occupied: list[tuple[int, int]] = []

    def available(span: tuple[int, int]) -> bool:
        return not any(span[0] < end and span[1] > start for start, end in occupied)

    for match in COMPOSITE_COURSE_RE.finditer(value):
        if not available(match.span()):
            continue
        outer, inner, number, parenthesized_suffix, direct_suffix = match.groups()
        suffix = parenthesized_suffix or direct_suffix
        canonical = f"{outer.upper()} ({inner.upper()}) {number}"
        if suffix:
            canonical += f"({suffix.upper()})"
        matches.append(CourseEntity(match.group(0), canonical, re.sub(r"[^A-Za-z0-9]", "", canonical).upper(), match.span()))
        occupied.append(match.span())

    for match in MULTI_PREFIX_COURSE_RE.finditer(value):
        if not available(match.span()):
            continue
        outer, inner, number, parenthesized_suffix, direct_suffix = match.groups()
        suffix = parenthesized_suffix or direct_suffix
        canonical = f"{outer.upper()} ({inner.upper()}) {number}"
        if suffix:
            canonical += f"({suffix.upper()})"
        matches.append(CourseEntity(match.group(0), canonical, re.sub(r"[^A-Za-z0-9]", "", canonical).upper(), match.span()))
        occupied.append(match.span())

    for match in SIMPLE_COURSE_RE.finditer(value):
        if not available(match.span()) or match.group(1).upper() in NON_COURSE_PREFIXES:
            continue
        prefix, number, parenthesized_suffix, direct_suffix = match.groups()
        suffix = parenthesized_suffix or direct_suffix
        canonical = f"{prefix.upper()} {number}"
        if suffix:
            canonical += f"({suffix.upper()})"
        matches.append(CourseEntity(match.group(0), canonical, re.sub(r"[^A-Za-z0-9]", "", canonical).upper(), match.span()))
        occupied.append(match.span())
    return tuple(sorted(matches, key=lambda item: item.span))


def _canonicalize_course_codes(text: str) -> str:
    entities = extract_course_entities(text)
    if not entities:
        return text
    pieces: list[str] = []
    cursor = 0
    for entity in entities:
        pieces.append(text[cursor:entity.span[0]])
        pieces.append(entity.canonical.casefold())
        cursor = entity.span[1]
    pieces.append(text[cursor:])
    return "".join(pieces)


def normalize_retrieval_text(text: str, *, normalize_banglish: bool = True) -> str:
    value = _canonicalize_course_codes(normalize_unicode(text)).casefold()
    value = re.sub(r"\bpre\s*[- ]?\s*req(?:uisites?)?\b", "prerequisite", value)
    value = re.sub(r"প্রি[- ]?রিকুইজিট", "prerequisite", value)
    value = re.sub(r"(\d{1,2})\s*ম\s*সপ্তাহে?", r"সপ্তাহ \1", value)
    value = re.sub(r"সপ্তাহে\s*(\d{1,2})", r"সপ্তাহ \1", value)
    value = re.sub(r"\b([a-z]+)-(er|e|gulo)\b", r"\1 \2", value)
    if normalize_banglish:
        for variant, canonical in BANGLISH_VARIANTS.items():
            value = re.sub(rf"\b{re.escape(variant)}\b", canonical, value)
        value = re.sub(r"\bunder\s+e\b", "under", value)
        value = re.sub(r"\bboraddo\b", "allocated", value)
        value = re.sub(r"\bporano\s+hoy\b", "porano", value)
        value = re.sub(r"\bfinal\s+exam\s+e\b", "final exam", value)
        value = re.sub(r"\bweek\s+(\d+)\s+e\b", r"week \1", value)
    # Preserve characters that carry lexical value in emails, decimals,
    # percentages, dates, Bengali words, and technical identifiers.
    value = re.sub(r"[^\w\u0980-\u09ff@.%+()/\-]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip(" -/")


def build_query_representations(query: str) -> QueryRepresentations:
    original = str(query)
    normalized = normalize_retrieval_text(original)
    original_clean = original.strip()
    # Bounded sparse expansion: preserve every original lexical signal and add
    # at most one conservative normalized representation.
    retrieval = original_clean
    if normalized and normalized.casefold() != original_clean.casefold():
        retrieval = f"{original_clean} {normalized}"
    return QueryRepresentations(original, normalized, retrieval)


def dense_query_for_strategy(query: str, strategy: str) -> str:
    representations = build_query_representations(query)
    selected = strategy.strip().casefold()
    if selected == "original":
        return representations.original_query
    if selected == "normalized":
        return representations.normalized_query
    if selected == "original_plus_normalized":
        if representations.normalized_query.casefold() == representations.original_query.strip().casefold():
            return representations.original_query
        return f"{representations.original_query.strip()} {representations.normalized_query}"
    raise ValueError("Dense query strategy must be original, normalized, or original_plus_normalized.")


def compact_entity(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_unicode(value).casefold())


__all__ = [
    "BANGLISH_VARIANTS",
    "BENGALI_DIGITS",
    "CourseEntity",
    "QueryRepresentations",
    "build_query_representations",
    "compact_entity",
    "dense_query_for_strategy",
    "extract_course_entities",
    "normalize_retrieval_text",
    "normalize_unicode",
]
