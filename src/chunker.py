from __future__ import annotations

from typing import Any, Dict, List

from .config import CHUNK_OVERLAP, CHUNK_SIZE


def _validate_chunk_settings(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    _validate_chunk_settings(chunk_size, overlap)

    if not text or not text.strip():
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = max(1, chunk_size - overlap)
    chunks: List[str] = []

    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        if not window:
            continue
        chunks.append(" ".join(window))

        if start + chunk_size >= len(words):
            break

    return chunks


def chunk_pages(pages: List[Dict[str, Any]], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    _validate_chunk_settings(chunk_size, overlap)

    chunks: List[Dict[str, Any]] = []

    for page in pages:
        page_text = page.get("text", "")
        if not page_text.strip():
            continue

        words = page_text.split()
        step = chunk_size - overlap

        for index, start in enumerate(range(0, len(words), step), start=1):
            window = words[start:start + chunk_size]
            if not window:
                continue

            chunk_text = " ".join(window)
            chunk_id = f"{page.get('source', 'unknown')}-p{page.get('page', 'unknown')}-c{index}"
            chunks.append({
                "text": chunk_text,
                "source": page.get("source", "unknown"),
                "page": page.get("page", None),
                "chunk_id": chunk_id,
                "word_start": start,
                "word_end": start + len(window),
            })

            if start + chunk_size >= len(words):
                break

    return chunks


__all__ = ["split_text_into_chunks", "chunk_pages"]
