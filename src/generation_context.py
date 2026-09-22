from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from .evidence import EvidenceAssessment
from .language_detector import Language


@dataclass(frozen=True)
class VerifiedEvidenceItem:
    source: str
    page: int | None
    heading: str | None
    entity: str | None
    excerpt: str
    chunk_id: str | None
    retrieval_score: float


@dataclass(frozen=True)
class VerifiedEvidencePackage:
    question: str
    original_language: Language
    requested_entity: str | None
    requested_field: str
    support_status: str
    evidence: tuple[VerifiedEvidenceItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "evidence": [asdict(item) for item in self.evidence]}


def build_verified_evidence_package(
    question: str,
    language: Language,
    assessment: EvidenceAssessment,
    retrieved: Sequence[dict[str, Any]] = (),
) -> VerifiedEvidencePackage:
    metadata_by_chunk = {str(item.get("chunk_id")): item for item in retrieved if item.get("chunk_id")}
    seen: set[str] = set()
    items: list[VerifiedEvidenceItem] = []
    # EvidenceAssessment preserves the final retrieval order.
    for evidence in assessment.evidence:
        excerpt = " ".join(evidence.excerpt.split()).strip()
        duplicate_key = excerpt.casefold()
        if not excerpt or duplicate_key in seen:
            continue
        seen.add(duplicate_key)
        source_metadata = metadata_by_chunk.get(str(evidence.chunk_id), {})
        items.append(VerifiedEvidenceItem(
            source=evidence.relative_path or evidence.source,
            page=evidence.page,
            heading=source_metadata.get("heading") or source_metadata.get("section"),
            entity=evidence.matched_entity or evidence.detected_entity,
            excerpt=excerpt,
            chunk_id=evidence.chunk_id,
            retrieval_score=evidence.retrieval_score,
        ))
    return VerifiedEvidencePackage(
        question=question,
        original_language=language,
        requested_entity=assessment.request.primary_entity,
        requested_field=assessment.request.requested_field,
        support_status=assessment.status.value,
        evidence=tuple(items),
    )


def format_evidence_block(item: VerifiedEvidenceItem, position: int) -> str:
    page = str(item.page) if item.page is not None else "not specified"
    heading = item.heading or "not specified"
    entity = item.entity or "not specified"
    return (
        f"[EVIDENCE {position}]\n"
        f"Source: {item.source}\n"
        f"Page: {page}\n"
        f"Heading: {heading}\n"
        f"Entity: {entity}\n"
        f"Text: {item.excerpt}"
    )


def select_evidence_blocks(
    package: VerifiedEvidencePackage,
    token_counter: Callable[[str], int],
    token_budget: int,
) -> tuple[list[str], int]:
    if token_budget <= 0:
        raise ValueError("Generation evidence token budget must be positive.")
    selected: list[str] = []
    used = 0
    for position, item in enumerate(package.evidence, start=1):
        block = format_evidence_block(item, position)
        cost = token_counter(block)
        if cost <= 0:
            continue
        if used + cost > token_budget:
            continue
        selected.append(block)
        used += cost
    if package.evidence and not selected:
        raise ValueError("The highest-ranked verified evidence does not fit the configured context budget.")
    return selected, used


def evidence_excerpts(package: VerifiedEvidencePackage) -> Sequence[str]:
    return [item.excerpt for item in package.evidence]


__all__ = [
    "VerifiedEvidenceItem", "VerifiedEvidencePackage", "build_verified_evidence_package",
    "evidence_excerpts", "format_evidence_block", "select_evidence_blocks",
]
