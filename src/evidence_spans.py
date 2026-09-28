"""Expand verified excerpts only within their own structural provenance."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Sequence

from .evidence import EvidenceAssessment


def excerpt_is_truncated(excerpt: str, full_text: str) -> bool:
    short = " ".join(str(excerpt).split()).strip()
    full = " ".join(str(full_text).split()).strip()
    return bool(short and full and ("…" in short or len(short) < len(full)))


def expand_assessment_evidence(
    assessment: EvidenceAssessment,
    metadata: Sequence[dict[str, Any]],
    *,
    max_chars: int = 3600,
) -> EvidenceAssessment:
    """Prefer same chunk, then same parent block, then one explicit continuation.

    Never join merely because chunks are on neighboring pages or nearby ranks.
    """
    by_id = {str(item.get("chunk_id")): item for item in metadata if item.get("chunk_id")}
    expanded = []
    for evidence in assessment.evidence:
        full = " ".join(evidence.text.split()).strip()
        if not excerpt_is_truncated(evidence.excerpt, full):
            expanded.append(evidence)
            continue
        source = by_id.get(str(evidence.chunk_id), {})
        text = full
        linked: list[str] = []
        if len(text) <= max_chars and not re.search(r"[.!?।]\s*$", text):
            parent = source.get("parent_id")
            siblings = [item for item in metadata
                        if parent and item.get("parent_id") == parent
                        and item.get("document_id") == source.get("document_id")
                        and item.get("page") == source.get("page")]
            siblings.sort(key=lambda item: int(item.get("word_start") or 0))
            if len(siblings) > 1 and sum(len(str(item.get("text", ""))) for item in siblings) <= max_chars:
                text = " ".join(" ".join(str(item.get("text", "")).split()) for item in siblings)
                linked = [str(item.get("chunk_id")) for item in siblings if str(item.get("chunk_id")) != str(evidence.chunk_id)]
            elif source.get("next_chunk_id"):
                following = by_id.get(str(source["next_chunk_id"]))
                if (following and following.get("continuation_of") == evidence.chunk_id
                        and following.get("document_id") == source.get("document_id")
                        and following.get("page") == source.get("page")):
                    candidate = text + " " + " ".join(str(following.get("text", "")).split())
                    if len(candidate) <= max_chars:
                        text = candidate
                        linked = [str(following["chunk_id"])]
        if len(text) <= max_chars:
            expanded.append(replace(evidence, excerpt=text, continuation_chunk_ids=tuple(linked)))
        else:
            expanded.append(evidence)
    return replace(assessment, evidence=tuple(expanded))


__all__ = ["excerpt_is_truncated", "expand_assessment_evidence"]
