from __future__ import annotations

import json
import os
import pickle
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    faiss = None
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None

from .config import (
    CHUNKER_SCHEMA_VERSION,
    EMBEDDING_MODEL,
    STRUCTURED_CHUNK_MAX_WORDS,
    STRUCTURED_CHUNK_OVERLAP,
    VECTOR_DB_DIR,
)
from .corpus import (
    DocumentDescriptor,
    DocumentStatus,
    chunking_configuration_id,
    corpus_fingerprint,
    descriptor_for_path,
    identity_records,
    sha256_file,
)

MANIFEST_PATH = VECTOR_DB_DIR / "index_manifest.json"


def create_index(vectors: np.ndarray) -> Any:
    if faiss is None:
        raise ImportError("faiss-cpu is required for the vector database. Install project requirements first.") from _FAISS_IMPORT_ERROR
    if len(vectors) == 0:
        raise ValueError("Cannot build an index from an empty vector list.")
    if vectors.ndim != 2 or vectors.shape[1] <= 0:
        raise ValueError("Embedding vectors must be a non-empty two-dimensional array.")
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(vectors.astype(np.float32))
    return index


def build_index(
    vectors: np.ndarray,
    metadata: Sequence[Dict[str, Any]],
    index_path: str | Path = VECTOR_DB_DIR / "index.faiss",
) -> Any:
    """Backward-compatible immediate build; production builds use atomic publication."""
    index = create_index(vectors)
    save_index(index, metadata, index_path=index_path)
    return index


def save_index(
    index: Any,
    metadata: Sequence[Dict[str, Any]],
    index_path: str | Path = VECTOR_DB_DIR / "index.faiss",
    metadata_path: str | Path = VECTOR_DB_DIR / "metadata.pkl",
) -> None:
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with metadata_path.open("wb") as handle:
        pickle.dump(list(metadata), handle)


def load_index(
    index_path: str | Path = VECTOR_DB_DIR / "index.faiss",
    metadata_path: str | Path = VECTOR_DB_DIR / "metadata.pkl",
) -> Tuple[Any, List[Dict[str, Any]]]:
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if faiss is None:
        raise ImportError("faiss-cpu is required for the vector database. Install project requirements first.") from _FAISS_IMPORT_ERROR
    index = faiss.read_index(str(index_path))
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    return index, list(metadata)


def search(index: Any, query_vector: np.ndarray, metadata: Sequence[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    similarity_scores, indices = index.search(query_vector.astype(np.float32), top_k)
    results: List[Dict[str, Any]] = []
    for score, idx in zip(similarity_scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = dict(metadata[int(idx)])
        item["score"] = float(score)
        results.append(item)
    return results


def _coerce_documents(
    documents_or_paths: Sequence[DocumentDescriptor | str | Path],
    document_root: str | Path | None = None,
) -> list[DocumentDescriptor]:
    if not documents_or_paths:
        return []
    if all(isinstance(item, DocumentDescriptor) for item in documents_or_paths):
        return list(documents_or_paths)  # type: ignore[arg-type]
    paths = [Path(item).resolve() for item in documents_or_paths]
    if document_root is None:
        common = Path(os.path.commonpath([str(path.parent) for path in paths]))
    else:
        common = Path(document_root).resolve()
    return [descriptor_for_path(path, common) for path in paths]


def document_fingerprint(
    documents_or_paths: Sequence[DocumentDescriptor | str | Path],
    document_root: str | Path | None = None,
) -> List[Dict[str, Any]]:
    return identity_records(_coerce_documents(documents_or_paths, document_root=document_root))


def build_manifest(
    pdf_paths: Sequence[DocumentDescriptor | str | Path],
    chunk_count: int,
    embedding_dimension: int,
    index_type: str = "IndexFlatIP",
    document_root: str | Path | None = None,
) -> Dict[str, Any]:
    documents = _coerce_documents(pdf_paths, document_root=document_root)
    status = [document.ingestion_status for document in documents]
    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_fingerprint": corpus_fingerprint(documents),
        "documents": [document.manifest_dict() for document in sorted(documents, key=lambda item: item.relative_path.casefold())],
        "document_count": len(documents),
        "successful_document_count": status.count(DocumentStatus.SUCCESS.value),
        "partial_document_count": status.count(DocumentStatus.PARTIAL.value),
        "failed_document_count": status.count(DocumentStatus.FAILED.value),
        "duplicate_document_count": status.count(DocumentStatus.DUPLICATE.value),
        "chunk_count": int(chunk_count),
        "chunking_configuration_id": chunking_configuration_id(),
        "chunking_strategy": "semantic_blocks_then_bounded_windows",
        "chunker_schema_version": CHUNKER_SCHEMA_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": int(embedding_dimension),
        "embedding_normalized": True,
        "faiss_index_type": index_type,
        "chunk_size": STRUCTURED_CHUNK_MAX_WORDS,
        "chunk_overlap": STRUCTURED_CHUNK_OVERLAP,
    }


def save_manifest(manifest: Dict[str, Any], manifest_path: str | Path = MANIFEST_PATH) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_manifest(manifest_path: str | Path = MANIFEST_PATH) -> Dict[str, Any] | None:
    path = Path(manifest_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def index_is_fresh(
    pdf_paths: Sequence[DocumentDescriptor | str | Path],
    manifest_path: str | Path = MANIFEST_PATH,
    expected_embedding_model: str = EMBEDDING_MODEL,
    expected_chunk_size: int = STRUCTURED_CHUNK_MAX_WORDS,
    expected_chunk_overlap: int = STRUCTURED_CHUNK_OVERLAP,
    document_root: str | Path | None = None,
) -> bool:
    manifest = load_manifest(manifest_path)
    if manifest is None or manifest.get("schema_version") != 2:
        return False
    documents = _coerce_documents(pdf_paths, document_root=document_root)
    return (
        manifest.get("corpus_fingerprint")
        == corpus_fingerprint(
            documents,
            chunk_size=expected_chunk_size,
            chunk_overlap=expected_chunk_overlap,
            embedding_model=expected_embedding_model,
        )
        and manifest.get("embedding_model") == expected_embedding_model
        and manifest.get("chunk_size") == expected_chunk_size
        and manifest.get("chunk_overlap") == expected_chunk_overlap
    )


def stale_metadata_sources(
    metadata: Sequence[Dict[str, Any]],
    current_pdf_paths: Sequence[DocumentDescriptor | str | Path],
    document_root: str | Path | None = None,
) -> List[str]:
    documents = _coerce_documents(current_pdf_paths, document_root=document_root)
    known_ids = {document.document_id for document in documents}
    known_paths = {document.relative_path for document in documents}
    stale = {
        str(item.get("relative_path") or item.get("source") or "")
        for item in metadata
        if (
            item.get("document_id") not in known_ids
            or str(item.get("relative_path") or item.get("source") or "") not in known_paths
        )
    }
    return sorted(value for value in stale if value)


def validate_index_metadata_consistency(
    index: Any,
    metadata: Sequence[Dict[str, Any]],
    current_pdf_paths: Sequence[DocumentDescriptor | str | Path],
    manifest: Dict[str, Any] | None = None,
    document_root: str | Path | None = None,
) -> Dict[str, Any]:
    required_keys = {"text", "document_id", "source", "relative_path", "page", "chunk_id", "word_start", "word_end"}
    missing_key_rows = [i for i, item in enumerate(metadata) if not required_keys.issubset(item)]
    empty_chunk_rows = [i for i, item in enumerate(metadata) if not str(item.get("text", "")).strip()]
    chunk_ids = [str(item.get("chunk_id", "")) for item in metadata]
    chunk_id_counts = Counter(chunk_ids)
    duplicate_chunk_ids = sorted(chunk_id for chunk_id, count in chunk_id_counts.items() if chunk_id and count > 1)
    stale_sources = stale_metadata_sources(metadata, current_pdf_paths, document_root=document_root)
    vector_count = int(getattr(index, "ntotal", -1))
    metadata_count = len(metadata)
    index_dimension = int(getattr(index, "d", -1))
    expected_dimension = int(manifest.get("embedding_dimension", -1)) if manifest else index_dimension
    manifest_chunk_count = int(manifest.get("chunk_count", -1)) if manifest else metadata_count
    documents = _coerce_documents(current_pdf_paths, document_root=document_root)
    known_provenance = {
        (document.document_id, document.relative_path, document.source)
        for document in documents
    }
    invalid_document_rows = [
        i
        for i, item in enumerate(metadata)
        if (
            str(item.get("document_id", "")),
            str(item.get("relative_path", "")),
            str(item.get("source", "")),
        ) not in known_provenance
    ]
    document_id_counts = Counter(document.document_id for document in documents)
    duplicate_document_ids = sorted(
        document_id for document_id, count in document_id_counts.items() if count > 1
    )
    manifest_matches_corpus = (
        manifest is None or manifest.get("corpus_fingerprint") == corpus_fingerprint(documents)
    )
    result = {
        "vector_count": vector_count,
        "metadata_count": metadata_count,
        "counts_match": vector_count == metadata_count == manifest_chunk_count,
        "index_dimension": index_dimension,
        "expected_dimension": expected_dimension,
        "dimension_matches": index_dimension == expected_dimension,
        "manifest_matches_corpus": manifest_matches_corpus,
        "stale_sources": stale_sources,
        "missing_key_rows": missing_key_rows,
        "empty_chunk_rows": empty_chunk_rows,
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "duplicate_document_ids": duplicate_document_ids,
        "invalid_document_rows": invalid_document_rows,
    }
    result["ok"] = all(
        (
            result["counts_match"],
            result["dimension_matches"],
            result["manifest_matches_corpus"],
            not stale_sources,
            not missing_key_rows,
            not empty_chunk_rows,
            not duplicate_chunk_ids,
            not duplicate_document_ids,
            not invalid_document_rows,
        )
    )
    return result


def atomic_publish_index(
    index: Any,
    metadata: Sequence[Dict[str, Any]],
    manifest: Dict[str, Any],
    documents: Sequence[DocumentDescriptor],
    target_dir: str | Path = VECTOR_DB_DIR,
) -> Dict[str, Any]:
    """Stage, validate, then replace the index artifact set with rollback."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    names = ("index.faiss", "metadata.pkl", "index_manifest.json")
    with tempfile.TemporaryDirectory(prefix="index-stage-", dir=str(target.parent)) as temporary:
        stage = Path(temporary)
        staged_index = stage / names[0]
        staged_metadata = stage / names[1]
        staged_manifest = stage / names[2]
        save_index(index, metadata, staged_index, staged_metadata)
        save_manifest(manifest, staged_manifest)

        check_index, check_metadata = load_index(staged_index, staged_metadata)
        check_manifest = load_manifest(staged_manifest)
        consistency = validate_index_metadata_consistency(
            check_index,
            check_metadata,
            documents,
            manifest=check_manifest,
        )
        if not consistency["ok"]:
            raise RuntimeError(f"Staged index failed consistency checks: {consistency}")

        backup = stage / "backup"
        backup.mkdir()
        existed: dict[str, bool] = {}
        for name in names:
            destination = target / name
            existed[name] = destination.exists()
            if destination.exists():
                shutil.copy2(destination, backup / name)

        try:
            for name in names:
                os.replace(stage / name, target / name)
        except Exception:
            for name in names:
                destination = target / name
                saved = backup / name
                if existed[name] and saved.exists():
                    shutil.copy2(saved, destination)
                elif not existed[name] and destination.exists():
                    destination.unlink()
            raise
    return consistency


__all__ = [
    "MANIFEST_PATH", "atomic_publish_index", "build_index", "build_manifest", "create_index",
    "document_fingerprint", "index_is_fresh", "load_index", "load_manifest", "save_index",
    "save_manifest", "search", "sha256_file", "stale_metadata_sources",
    "validate_index_metadata_consistency",
]
