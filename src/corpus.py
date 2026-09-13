from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import (
    CHUNKER_SCHEMA_VERSION,
    DOCUMENT_DIR,
    EMBEDDING_MODEL,
    STRUCTURED_CHUNK_MAX_WORDS,
    STRUCTURED_CHUNK_OVERLAP,
    SUPPORTED_DOCUMENT_EXTENSIONS,
)


class DocumentStatus(str, Enum):
    DISCOVERED = "discovered"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    DUPLICATE = "duplicate"


@dataclass
class DocumentDescriptor:
    document_id: str
    source: str
    relative_path: str
    path: Path = field(repr=False)
    sha256: str
    size_bytes: int
    modified_timestamp: float | None = None
    page_count: int = 0
    extractable_page_count: int = 0
    empty_page_count: int = 0
    chunk_count: int = 0
    ingestion_status: str = DocumentStatus.DISCOVERED.value
    ingestion_errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duplicate_of: str | None = None

    def manifest_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("path", None)
        return value

    def identity_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass
class IngestionResult:
    documents: list[DocumentDescriptor]
    pages: list[dict[str, Any]]
    issues: list[dict[str, Any]]

    @property
    def document_count(self) -> int:
        return len(self.documents)

    def status_count(self, status: DocumentStatus | str) -> int:
        value = status.value if isinstance(status, DocumentStatus) else status
        return sum(document.ingestion_status == value for document in self.documents)

    @property
    def pages_total(self) -> int:
        return sum(document.page_count for document in self.documents if document.ingestion_status != DocumentStatus.DUPLICATE.value)

    @property
    def pages_with_text(self) -> int:
        return len(self.pages)

    @property
    def pages_empty(self) -> int:
        return sum(document.empty_page_count for document in self.documents)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def make_document_id(relative_path: str, content_sha256: str) -> str:
    material = f"{relative_path.casefold()}\0{content_sha256}".encode("utf-8")
    return "doc-" + hashlib.sha256(material).hexdigest()[:24]


def descriptor_for_path(path: str | Path, document_root: str | Path) -> DocumentDescriptor:
    resolved = Path(path).resolve()
    root = Path(document_root).resolve()
    relative = _relative_path(resolved, root)
    digest = sha256_file(resolved)
    stat = resolved.stat()
    return DocumentDescriptor(
        document_id=make_document_id(relative, digest),
        source=resolved.name,
        relative_path=relative,
        path=resolved,
        sha256=digest,
        size_bytes=stat.st_size,
        modified_timestamp=stat.st_mtime,
    )


def discover_documents(
    document_dir: str | Path = DOCUMENT_DIR,
    recursive: bool = True,
    require_documents: bool = True,
) -> list[DocumentDescriptor]:
    root = Path(document_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Document directory not found: {root}")
    iterator = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(
        (
            path.resolve()
            for path in iterator
            if path.is_file() and path.suffix.casefold() in SUPPORTED_DOCUMENT_EXTENSIONS
        ),
        key=lambda path: _relative_path(path, root).casefold(),
    )
    if require_documents and not paths:
        raise FileNotFoundError(f"No supported documents found in: {root}")
    return [descriptor_for_path(path, root) for path in paths]


def mark_exact_duplicates(documents: Sequence[DocumentDescriptor]) -> list[DocumentDescriptor]:
    canonical_by_hash: dict[str, DocumentDescriptor] = {}
    for document in sorted(documents, key=lambda item: item.relative_path.casefold()):
        canonical = canonical_by_hash.get(document.sha256)
        if canonical is None:
            canonical_by_hash[document.sha256] = document
            continue
        document.ingestion_status = DocumentStatus.DUPLICATE.value
        document.duplicate_of = canonical.document_id
        document.warnings.append(f"Exact duplicate of {canonical.relative_path}; skipped.")
    return list(documents)


def identity_records(documents: Sequence[DocumentDescriptor]) -> list[dict[str, Any]]:
    return sorted(
        (document.identity_dict() for document in documents),
        key=lambda item: item["relative_path"].casefold(),
    )


def chunking_configuration_id(
    chunk_size: int = STRUCTURED_CHUNK_MAX_WORDS,
    chunk_overlap: int = STRUCTURED_CHUNK_OVERLAP,
) -> str:
    material = json.dumps(
        {"strategy": CHUNKER_SCHEMA_VERSION, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def corpus_fingerprint(
    documents: Sequence[DocumentDescriptor],
    chunk_size: int = STRUCTURED_CHUNK_MAX_WORDS,
    chunk_overlap: int = STRUCTURED_CHUNK_OVERLAP,
    embedding_model: str = EMBEDDING_MODEL,
) -> str:
    material = {
        "documents": identity_records(documents),
        "chunking_configuration_id": chunking_configuration_id(chunk_size, chunk_overlap),
        "embedding_model": embedding_model,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ingestion_report(
    result: IngestionResult,
    chunks: Sequence[dict[str, Any]],
    fingerprint: str | None = None,
) -> dict[str, Any]:
    block_types: dict[str, int] = {}
    for chunk in chunks:
        name = str(chunk.get("block_type") or "unknown")
        block_types[name] = block_types.get(name, 0) + 1
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_fingerprint": fingerprint or corpus_fingerprint(result.documents),
        "document_count": result.document_count,
        "successful_document_count": result.status_count(DocumentStatus.SUCCESS),
        "partial_document_count": result.status_count(DocumentStatus.PARTIAL),
        "failed_document_count": result.status_count(DocumentStatus.FAILED),
        "duplicate_document_count": result.status_count(DocumentStatus.DUPLICATE),
        "pages_total": result.pages_total,
        "pages_with_text": result.pages_with_text,
        "pages_empty": result.pages_empty,
        "chunk_count": len(chunks),
        "block_count": len({str(chunk.get("block_id")) for chunk in chunks if chunk.get("block_id")}),
        "chunk_type_counts": dict(sorted(block_types.items())),
        "documents": [document.manifest_dict() for document in result.documents],
        "issues": list(result.issues),
    }


def write_ingestion_report(report: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "DocumentDescriptor", "DocumentStatus", "IngestionResult", "chunking_configuration_id",
    "corpus_fingerprint", "descriptor_for_path", "discover_documents", "identity_records",
    "ingestion_report", "make_document_id", "mark_exact_duplicates", "sha256_file",
    "write_ingestion_report",
]
