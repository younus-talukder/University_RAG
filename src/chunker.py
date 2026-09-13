from __future__ import annotations

import hashlib
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .config import (
    CHUNKER_SCHEMA_VERSION,
    STRUCTURED_CHUNK_MAX_WORDS,
    STRUCTURED_CHUNK_MIN_WORDS,
    STRUCTURED_CHUNK_OVERLAP,
)


COURSE_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{2,12}(?:\s*\(\s*[A-Za-z]{2,12}\s*\))?\s*[- ]?\s*\d{2,4}(?:\s*\([A-Za-z0-9]{1,4}\))?)(?![A-Za-z0-9])"
)
COURSE_HEADER_RE = re.compile(r"^\s*Course\s+(?:Code|No\.?\s*/\s*Course\s+Code)\s*:\s*(.+?)\s*$", re.I)
COURSE_FIELD_RE = re.compile(r"^\s*(Course\s+Title|Credits?|Credit\s+Value|Pre\s*[- ]?\s*Requisites?)\s*:\s*(.+)$", re.I)
SEMESTER_RE = re.compile(r"\b((?:First|Second|Third|Fourth|Final)\s+Year\s+(?:First|Second|Third)\s+Semester)\b", re.I)

FIELD_PATTERNS: dict[str, str] = {
    "course_code": r"\bcourse\s+(?:code|no\.?)\b|" + COURSE_CODE_RE.pattern,
    "course_title": r"\bcourse\s+title\b",
    "credits": r"\bcredits?\b|\bcredit\s+value\b",
    "prerequisite": r"\bpre\s*[- ]?\s*requisites?\b",
    "attendance": r"\battendance\b",
    "percentage": r"\d+(?:\.\d+)?\s*%|\bpercentage\b",
    "email": r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
    "date": r"\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4})\b",
    "requirement": r"\brequired\b|\brequirements?\b|\bmust\b",
    "policy": r"\bpolic(?:y|ies)\b|\brules?\b|\bregulations?\b",
    "topics": r"\bcourse\s+content\b|\btopics?\b",
}


@dataclass
class Block:
    document_id: str
    source: str
    relative_path: str
    page: int
    block_index: int
    block_type: str
    text: str
    heading: str | None = None
    subsection: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    entity_label: str | None = None
    course_code: str | None = None
    course_title: str | None = None
    semester: str | None = None
    year: str | None = None
    field_types: list[str] = field(default_factory=list)

    @property
    def block_id(self) -> str:
        return f"{self.document_id}-p{self.page:06d}-b{self.block_index:04d}"


def _validate_chunk_settings(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")


def split_text_into_chunks(
    text: str,
    chunk_size: int = STRUCTURED_CHUNK_MAX_WORDS,
    overlap: int = STRUCTURED_CHUNK_OVERLAP,
) -> List[str]:
    _validate_chunk_settings(chunk_size, overlap)
    words = str(text).split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]
    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def normalize_course_code(value: str) -> str:
    match = COURSE_CODE_RE.search(str(value))
    if not match:
        return ""
    raw = re.sub(r"\s+", " ", match.group(1)).strip()
    raw = re.sub(r"([A-Za-z])(?=\d)", r"\1 ", raw)
    raw = re.sub(r"\s*\(\s*", " (", raw)
    raw = re.sub(r"\s*\)\s*", ") ", raw)
    return re.sub(r"\s+", " ", raw).strip().upper()


def course_codes(text: str) -> list[str]:
    codes = []
    seen = set()
    ignored = {"WEEK", "PAGE", "GRADE", "LEVEL", "SEMESTER", "YEAR"}
    for match in COURSE_CODE_RE.finditer(str(text)):
        code = normalize_course_code(match.group(1))
        prefix = re.match(r"[A-Z]+", code)
        if code and prefix and prefix.group(0) not in ignored and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def detect_field_types(text: str) -> list[str]:
    return [name for name, pattern in FIELD_PATTERNS.items() if re.search(pattern, text, re.I)]


def _is_low_information(text: str) -> bool:
    words = text.split()
    alphanumeric = re.sub(r"[^A-Za-z0-9\u0980-\u09ff]", "", text)
    return len(words) <= 4 and (len(alphanumeric) < 18 or all(word.isdigit() for word in words))


def _is_heading(line: str, next_line: str = "") -> bool:
    value = line.strip().strip("•")
    words = value.split()
    if not value or len(words) > 10 or len(value) > 110:
        return False
    if COURSE_HEADER_RE.match(value) or COURSE_FIELD_RE.match(value):
        return False
    if re.match(r"^\d+(?:\.\d+)*[.)]?\s+\D", value):
        return True
    if SEMESTER_RE.search(value):
        return True
    if value.endswith((".", ",", ";", "?", "!")):
        return False
    letters = [word for word in words if re.search(r"[A-Za-z]", word)]
    if not letters:
        return False
    uppercase = value.upper() == value and len(letters) >= 1
    title_ratio = sum(word[:1].isupper() for word in letters) / len(letters)
    title_like = title_ratio >= 0.85 and len(words) <= 8
    structural_terms = bool(re.search(
        r"\b(?:section|chapter|part|policy|rules?|regulations?|attendance|evaluation|examination|requirements?|discipline|objectives?|contents?|references?|books?|semester)\b",
        value,
        re.I,
    ))
    next_is_body = not next_line or len(next_line.split()) >= 5 or next_line[:1].islower()
    return next_is_body and (uppercase or title_like or (structural_terms and title_ratio >= 0.5))


def _metadata_from_text(text: str, primary_code: str | None = None) -> dict[str, Any]:
    code = primary_code or None
    title_match = re.search(r"Course\s+Title\s*:\s*(.+?)(?=\n|\s+Credits?\s*:|$)", text, re.I)
    semester_match = SEMESTER_RE.search(text)
    year_match = re.search(r"\b(First|Second|Third|Fourth|Final)\s+Year\b", text, re.I)
    return {
        "entity_type": "course" if code else None,
        "entity_id": code,
        "entity_label": title_match.group(1).strip() if title_match else None,
        "course_code": code,
        "course_title": title_match.group(1).strip() if title_match else None,
        "semester": semester_match.group(1).strip() if semester_match else None,
        "year": year_match.group(0).strip() if year_match else None,
        "field_types": detect_field_types(text),
    }


def _new_block(page: dict[str, Any], index: int, block_type: str, text: str, heading: str | None = None, primary_code: str | None = None) -> Block:
    metadata = _metadata_from_text(text, primary_code=primary_code)
    return Block(
        document_id=str(page.get("document_id") or _legacy_document_id(page)),
        source=str(page.get("source") or "unknown"),
        relative_path=str(page.get("relative_path") or page.get("source") or "unknown"),
        page=int(page.get("page") or 0),
        block_index=index,
        block_type=block_type,
        text=text.strip(),
        heading=heading,
        **metadata,
    )


def _legacy_document_id(page: dict[str, Any]) -> str:
    relative = str(page.get("relative_path") or page.get("source") or "unknown")
    return "doc-" + hashlib.sha256(relative.casefold().encode("utf-8")).hexdigest()[:24]


def _course_header_blocks(page: dict[str, Any], lines: list[str]) -> list[Block]:
    starts = [index for index, line in enumerate(lines) if COURSE_HEADER_RE.match(line)]
    if not starts:
        return []
    blocks: list[Block] = []
    ordinal = 1
    if starts[0] > 0:
        prefix = "\n".join(lines[:starts[0]]).strip()
        if prefix and not _is_low_information(prefix):
            blocks.extend(_heading_blocks(page, lines[:starts[0]], start_index=ordinal))
            ordinal = len(blocks) + 1
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        text = "\n".join(lines[start:end]).strip()
        header = COURSE_HEADER_RE.match(lines[start])
        code = normalize_course_code(header.group(1)) if header else ""
        blocks.append(_new_block(page, ordinal, "course_record", text, heading=lines[start], primary_code=code or None))
        ordinal += 1
    return blocks


def _table_row_blocks(page: dict[str, Any], lines: list[str]) -> list[Block]:
    flattened = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if not re.search(r"Course\s+Code\s+Course\s+Title\s+Credits?", flattened, re.I):
        return []
    matches = list(COURSE_CODE_RE.finditer(flattened))
    row_starts: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(flattened)
        between = flattened[match.end():next_start]
        if re.search(r"\b\d+(?:\.\d+)\b", between):
            code = normalize_course_code(match.group(1))
            if code:
                row_starts.append((match.start(), code))
    if len(row_starts) < 2:
        return []

    blocks: list[Block] = []
    first = row_starts[0][0]
    prefix = flattened[:first].strip()
    if prefix and not _is_low_information(prefix):
        blocks.append(_new_block(page, 1, "table_like", prefix, heading=None))
    ordinal = len(blocks) + 1
    for index, (start, code) in enumerate(row_starts):
        end = row_starts[index + 1][0] if index + 1 < len(row_starts) else len(flattened)
        row_text = flattened[start:end].strip()
        if not row_text:
            continue
        heading_candidates = list(SEMESTER_RE.finditer(flattened[:start]))
        heading = heading_candidates[-1].group(1) if heading_candidates else None
        block = _new_block(page, ordinal, "table_row", row_text, heading=heading, primary_code=code)
        decimal = re.search(r"\b\d+(?:\.\d+)\b", row_text)
        code_match = COURSE_CODE_RE.match(row_text)
        if decimal and code_match and decimal.start() > code_match.end():
            title = row_text[code_match.end():decimal.start()].strip(" -:;")
            if title:
                block.course_title = title
                block.entity_label = title
            block.field_types = sorted(set(block.field_types) | {"course_code", "course_title", "credits", "prerequisite"})
        blocks.append(block)
        ordinal += 1
    return blocks


def _heading_blocks(page: dict[str, Any], lines: list[str], start_index: int = 1) -> list[Block]:
    blocks: list[Block] = []
    current: list[str] = []
    current_heading: str | None = None
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if _is_heading(line, next_line):
            if current and len(" ".join(current).split()) >= STRUCTURED_CHUNK_MIN_WORDS:
                text = "\n".join(current).strip()
                block_type = "policy_section" if any(field in detect_field_types(text) for field in ("policy", "attendance", "requirement")) else "section"
                blocks.append(_new_block(page, start_index + len(blocks), block_type, text, heading=current_heading))
                current = []
            elif current:
                current.append(line)
                current_heading = current_heading or line
                continue
            current_heading = line
        current.append(line)
    if current:
        text = "\n".join(current).strip()
        if not _is_low_information(text):
            block_type = "policy_section" if any(field in detect_field_types(text) for field in ("policy", "attendance", "requirement")) else ("section" if current_heading else "fallback")
            blocks.append(_new_block(page, start_index + len(blocks), block_type, text, heading=current_heading))
    return blocks


def segment_page(page: dict[str, Any]) -> list[Block]:
    text = str(page.get("clean_text") or page.get("text") or "").strip()
    if not text:
        return []
    lines = [re.sub(r"\s+", " ", line).strip() for line in (page.get("structured_lines") or text.splitlines()) if str(line).strip()]
    while lines and re.fullmatch(r"\d{1,4}", lines[0]):
        lines.pop(0)
    if not lines:
        return []
    course_blocks = _course_header_blocks(page, lines)
    if course_blocks:
        return course_blocks
    table_blocks = _table_row_blocks(page, lines)
    if table_blocks:
        return table_blocks
    blocks = _heading_blocks(page, lines)
    if blocks:
        return blocks
    fallback_text = "\n".join(lines)
    if _is_low_information(fallback_text):
        return []
    return [_new_block(page, 1, "fallback", fallback_text)]


def build_blocks(pages: Sequence[Dict[str, Any]]) -> list[Block]:
    blocks: list[Block] = []
    for page in sorted(
        pages,
        key=lambda item: (
            str(item.get("document_id") or _legacy_document_id(item)),
            int(item.get("page") or 0),
        ),
    ):
        blocks.extend(segment_page(page))
    return blocks


def _block_chunks(block: Block, chunk_size: int, overlap: int) -> list[dict[str, Any]]:
    words = block.text.split()
    if not words:
        return []
    header_prefix = ""
    if block.course_code and len(words) > chunk_size:
        header_lines = []
        for line in block.text.splitlines()[:5]:
            if COURSE_HEADER_RE.match(line) or COURSE_FIELD_RE.match(line):
                header_lines.append(line)
        header_prefix = "\n".join(header_lines).strip()
    step = chunk_size - overlap
    chunks: list[dict[str, Any]] = []
    for child_index, start in enumerate(range(0, len(words), step), start=1):
        window = words[start:start + chunk_size]
        if not window:
            break
        text = " ".join(window)
        if header_prefix and start > 0:
            text = header_prefix + "\n" + text
        chunk_id = f"{block.block_id}-c{child_index:04d}"
        chunks.append(
            {
                "text": text,
                "document_id": block.document_id,
                "source": block.source,
                "relative_path": block.relative_path,
                "page": block.page,
                "page_end": block.page,
                "chunk_id": chunk_id,
                "block_id": block.block_id,
                "parent_id": block.block_id,
                "parent_section_id": block.block_id,
                "block_type": block.block_type,
                "heading": block.heading,
                "section": block.heading,
                "subsection": block.subsection,
                "entity_type": block.entity_type,
                "entity_id": block.entity_id,
                "entity_label": block.entity_label,
                "course_code": block.course_code,
                "course_title": block.course_title,
                "semester": block.semester,
                "year": block.year,
                "field_types": list(block.field_types),
                "word_start": start,
                "word_end": start + len(window),
                "chunker_schema_version": CHUNKER_SCHEMA_VERSION,
                "previous_chunk_id": None,
                "next_chunk_id": None,
                "continuation_of": None,
            }
        )
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = STRUCTURED_CHUNK_MAX_WORDS,
    overlap: int = STRUCTURED_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    _validate_chunk_settings(chunk_size, overlap)
    chunks: list[dict[str, Any]] = []
    for block in build_blocks(pages):
        chunks.extend(_block_chunks(block, chunk_size, overlap))

    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_document[chunk["document_id"]].append(chunk)
    for document_chunks in by_document.values():
        document_chunks.sort(key=lambda item: (item["page"], item["chunk_id"]))
        for index, chunk in enumerate(document_chunks):
            previous = document_chunks[index - 1] if index > 0 else None
            following = document_chunks[index + 1] if index + 1 < len(document_chunks) else None
            chunk["previous_chunk_id"] = previous["chunk_id"] if previous else None
            chunk["next_chunk_id"] = following["chunk_id"] if following else None
            if (
                previous
                and chunk["page"] == previous["page"] + 1
                and chunk["block_type"] == "fallback"
                and not chunk.get("heading")
                and not re.search(r"[.!?।]\s*$", str(previous.get("text", "")))
            ):
                chunk["continuation_of"] = previous["chunk_id"]
                chunk["parent_section_id"] = previous.get("parent_section_id")
    return chunks


def validate_chunk_quality(
    chunks: Sequence[Dict[str, Any]],
    min_words: int = STRUCTURED_CHUNK_MIN_WORDS,
    max_words: int = STRUCTURED_CHUNK_MAX_WORDS + 80,
    mixed_entity_threshold: int = 3,
) -> dict[str, Any]:
    required = {"document_id", "source", "relative_path", "page", "chunk_id", "text", "word_start", "word_end"}
    chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
    texts = [re.sub(r"\s+", " ", str(chunk.get("text", ""))).casefold().strip() for chunk in chunks]
    id_counts = Counter(chunk_ids)
    text_counts = Counter(texts)
    def mixed_entity(chunk: Dict[str, Any]) -> bool:
        codes = course_codes(str(chunk.get("text", "")))
        if chunk.get("entity_id") and chunk.get("block_type") in {"course_record", "table_row"}:
            # A course block may legitimately name prerequisites/references. Flag
            # only unusually broad records rather than treating relationships as
            # unrelated entities.
            return len(codes) > mixed_entity_threshold + 2
        return len(codes) > mixed_entity_threshold

    result = {
        "chunk_count": len(chunks),
        "empty_chunks": [chunk_ids[i] for i, text in enumerate(texts) if not text],
        "very_short_chunks": [chunk_ids[i] for i, text in enumerate(texts) if text and len(text.split()) < min_words],
        "very_long_chunks": [chunk_ids[i] for i, text in enumerate(texts) if len(text.split()) > max_words],
        "mixed_entity_chunks": [chunk_ids[i] for i, chunk in enumerate(chunks) if mixed_entity(chunk)],
        "missing_provenance_chunks": [chunk_ids[i] for i, chunk in enumerate(chunks) if not required.issubset(chunk)],
        "duplicate_chunk_ids": sorted(chunk_id for chunk_id, count in id_counts.items() if chunk_id and count > 1),
        "duplicate_text_chunks": sorted(text for text, count in text_counts.items() if text and count > 1),
    }
    result["ok"] = not any(result[key] for key in ("empty_chunks", "very_long_chunks", "missing_provenance_chunks", "duplicate_chunk_ids"))
    return result


def chunk_statistics(chunks: Sequence[Dict[str, Any]]) -> dict[str, Any]:
    word_counts = [len(str(chunk.get("text", "")).split()) for chunk in chunks]
    type_counts = Counter(str(chunk.get("block_type") or "unknown") for chunk in chunks)
    return {
        "chunk_count": len(chunks),
        "average_words": sum(word_counts) / len(word_counts) if word_counts else 0.0,
        "median_words": statistics.median(word_counts) if word_counts else 0.0,
        "min_words": min(word_counts) if word_counts else 0,
        "max_words": max(word_counts) if word_counts else 0,
        "block_type_counts": dict(sorted(type_counts.items())),
        "entity_chunks": sum(bool(chunk.get("entity_id")) for chunk in chunks),
    }


__all__ = [
    "Block", "build_blocks", "chunk_pages", "chunk_statistics", "course_codes",
    "detect_field_types", "normalize_course_code", "segment_page", "split_text_into_chunks",
    "validate_chunk_quality",
]
