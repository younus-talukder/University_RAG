from __future__ import annotations

import json
import hashlib
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
    EMBEDDING_DIMENSION,
    EMBEDDING_DTYPE,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
    EMBEDDING_REVISION,
    EMBEDDING_USE_SAFETENSORS,
    STRUCTURED_CHUNK_MAX_WORDS,
    STRUCTURED_CHUNK_OVERLAP,
    VECTOR_DB_DIR,
)
from .embeddings import DEFAULT_EMBEDDING_CONFIG, EmbeddingConfiguration, dependency_versions
from .sparse_index import (
    SparseIndex,
    load_sparse_index,
    save_sparse_index,
    sparse_configuration_fingerprint,
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
FAISS_INDEX_TYPE = "IndexFlatIP"


class IndexCompatibilityError(RuntimeError):
    """Raised when an index cannot be searched with the runtime embedding configuration."""


def order_metadata_for_embedding(metadata: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Return deterministic metadata rows in the exact order used for vector insertion."""
    ordered = sorted(
        (dict(item) for item in metadata),
        key=lambda item: (
            str(item.get("relative_path") or item.get("source") or "").casefold(),
            int(item.get("page") or 0),
            str(item.get("block_id") or ""),
            str(item.get("chunk_id") or ""),
        ),
    )
    for row, item in enumerate(ordered):
        item["vector_row"] = row
        item["embedding_content_sha256"] = hashlib.sha256(
            str(item.get("text") or "").encode("utf-8")
        ).hexdigest()
    return ordered


def index_configuration_fingerprint(values: Dict[str, Any]) -> str:
    material = {
        "corpus_fingerprint": values.get("corpus_fingerprint"),
        "chunker_schema_version": values.get("chunker_schema_version"),
        "chunking_configuration_id": values.get("chunking_configuration_id"),
        "embedding_model": values.get("embedding_model"),
        "embedding_revision": values.get("embedding_revision"),
        "embedding_dimension": values.get("embedding_dimension"),
        "embedding_normalized": values.get("embedding_normalized"),
        "embedding_dtype": values.get("embedding_dtype"),
        "embedding_use_safetensors": values.get("embedding_use_safetensors"),
        "faiss_index_type": values.get("faiss_index_type"),
    }
    if int(values.get("schema_version", 0)) >= 4:
        material.update({
            "sparse_retrieval_enabled": values.get("sparse_retrieval_enabled"),
            "sparse_configuration_fingerprint": values.get("sparse_configuration_fingerprint"),
            "sparse_tokenizer_schema": values.get("sparse_tokenizer_schema"),
        })
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def create_index(
    vectors: np.ndarray,
    normalized: bool = EMBEDDING_NORMALIZE,
    norm_tolerance: float = DEFAULT_EMBEDDING_CONFIG.norm_tolerance,
) -> Any:
    if faiss is None:
        raise ImportError("faiss-cpu is required for the vector database. Install project requirements first.") from _FAISS_IMPORT_ERROR
    if len(vectors) == 0:
        raise ValueError("Cannot build an index from an empty vector list.")
    if vectors.ndim != 2 or vectors.shape[1] <= 0:
        raise ValueError("Embedding vectors must be a non-empty two-dimensional array.")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding vectors contain NaN or infinite values.")
    if normalized:
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=norm_tolerance, rtol=0.0):
            raise ValueError("IndexFlatIP requires the configured normalized embedding vectors.")
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


def validate_runtime_index_compatibility(
    index: Any,
    manifest: Dict[str, Any] | None,
    configuration: EmbeddingConfiguration = DEFAULT_EMBEDDING_CONFIG,
) -> None:
    if manifest is None:
        raise IndexCompatibilityError("Vector index manifest is missing; rebuild the index before semantic search.")
    expected = configuration.compatibility_dict()
    mismatches = [
        f"{key}: index={manifest.get(key)!r}, runtime={value!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    actual_type = type(index).__name__
    if manifest.get("faiss_index_type") != FAISS_INDEX_TYPE or actual_type != FAISS_INDEX_TYPE:
        mismatches.append(
            f"faiss_index_type: manifest={manifest.get('faiss_index_type')!r}, actual={actual_type!r}, expected={FAISS_INDEX_TYPE!r}"
        )
    if int(getattr(index, "d", -1)) != configuration.dimension:
        mismatches.append(
            f"FAISS dimension: index={getattr(index, 'd', None)!r}, runtime={configuration.dimension!r}"
        )
    if manifest.get("index_configuration_fingerprint") != index_configuration_fingerprint(manifest):
        mismatches.append("index configuration fingerprint is invalid")
    if mismatches:
        raise IndexCompatibilityError(
            "Vector index was built with a different embedding configuration and must be rebuilt. "
            + "; ".join(mismatches)
        )


def load_compatible_index(
    index_path: str | Path = VECTOR_DB_DIR / "index.faiss",
    metadata_path: str | Path = VECTOR_DB_DIR / "metadata.pkl",
    manifest_path: str | Path | None = None,
    configuration: EmbeddingConfiguration = DEFAULT_EMBEDDING_CONFIG,
) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)
    selected_manifest_path = Path(manifest_path) if manifest_path else index_path.parent / "index_manifest.json"
    index, metadata = load_index(index_path=index_path, metadata_path=metadata_path)
    manifest = load_manifest(selected_manifest_path)
    validate_runtime_index_compatibility(index, manifest, configuration)
    if int(getattr(index, "ntotal", -1)) != len(metadata):
        raise IndexCompatibilityError("FAISS vector count does not match metadata row count; rebuild the index.")
    invalid_rows = [row for row, item in enumerate(metadata) if item.get("vector_row") != row]
    if invalid_rows:
        raise IndexCompatibilityError(
            f"Index/metadata row binding is invalid at row {invalid_rows[0]}; rebuild the index."
        )
    return index, metadata, manifest


def load_hybrid_artifacts(
    index_path: str | Path = VECTOR_DB_DIR / "index.faiss",
    metadata_path: str | Path = VECTOR_DB_DIR / "metadata.pkl",
    manifest_path: str | Path | None = None,
    sparse_path: str | Path | None = None,
    configuration: EmbeddingConfiguration = DEFAULT_EMBEDDING_CONFIG,
) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any], SparseIndex]:
    index_path = Path(index_path)
    index, metadata, manifest = load_compatible_index(
        index_path=index_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        configuration=configuration,
    )
    if not manifest.get("sparse_retrieval_enabled"):
        raise IndexCompatibilityError("Hybrid sparse index is not enabled in the manifest; rebuild the index.")
    selected_sparse_path = Path(sparse_path) if sparse_path else index_path.parent / "sparse_index.pkl"
    sparse = load_sparse_index(selected_sparse_path, metadata)
    if (
        manifest.get("sparse_configuration_fingerprint") != sparse.configuration_fingerprint
        or manifest.get("sparse_metadata_binding_hash") != sparse.binding_hash
        or manifest.get("sparse_chunk_count") != sparse.chunk_count
    ):
        raise IndexCompatibilityError("Sparse index does not match the published manifest and must be rebuilt.")
    return index, metadata, manifest, sparse


def search(index: Any, query_vector: np.ndarray, metadata: Sequence[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    if query_vector.ndim != 2 or query_vector.shape[0] != 1:
        raise ValueError("Query embedding must contain exactly one vector.")
    if query_vector.shape[1] != int(getattr(index, "d", -1)):
        raise ValueError(
            f"Query embedding dimension {query_vector.shape[1]} does not match FAISS dimension {getattr(index, 'd', None)}."
        )
    if not np.isfinite(query_vector).all():
        raise ValueError("Query embedding contains NaN or infinite values.")
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
    index_type: str = FAISS_INDEX_TYPE,
    document_root: str | Path | None = None,
    embedding_provenance: Dict[str, Any] | None = None,
    build_metrics: Dict[str, Any] | None = None,
    sparse_index: SparseIndex | None = None,
) -> Dict[str, Any]:
    documents = _coerce_documents(pdf_paths, document_root=document_root)
    status = [document.ingestion_status for document in documents]
    provenance = {
        **DEFAULT_EMBEDDING_CONFIG.compatibility_dict(),
        "embedding_dimension": int(embedding_dimension),
        "embedding_device_class": DEFAULT_EMBEDDING_CONFIG.device.split(":", 1)[0].casefold(),
        "embedding_batch_size": DEFAULT_EMBEDDING_CONFIG.batch_size,
        "embedding_local_only": DEFAULT_EMBEDDING_CONFIG.local_only,
        "embedding_libraries": dependency_versions(),
    }
    if embedding_provenance:
        provenance.update(embedding_provenance)
    manifest = {
        "schema_version": 4,
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
        "faiss_index_type": index_type,
        "chunk_size": STRUCTURED_CHUNK_MAX_WORDS,
        "chunk_overlap": STRUCTURED_CHUNK_OVERLAP,
        **provenance,
        "sparse_retrieval_enabled": sparse_index is not None,
        "sparse_implementation": sparse_index.implementation if sparse_index else None,
        "sparse_tokenizer_schema": sparse_index.tokenizer_schema if sparse_index else None,
        "sparse_configuration_fingerprint": (
            sparse_index.configuration_fingerprint if sparse_index else None
        ),
        "sparse_chunk_count": sparse_index.chunk_count if sparse_index else 0,
        "sparse_vocabulary_size": sparse_index.vocabulary_size if sparse_index else 0,
        "sparse_metadata_binding_hash": sparse_index.binding_hash if sparse_index else None,
    }
    manifest["index_configuration_fingerprint"] = index_configuration_fingerprint(manifest)
    if build_metrics:
        manifest["build_metrics"] = dict(build_metrics)
    return manifest


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
    expected_embedding_revision: str = EMBEDDING_REVISION,
    expected_embedding_dimension: int = EMBEDDING_DIMENSION,
    expected_embedding_dtype: str = EMBEDDING_DTYPE,
    expected_embedding_normalized: bool = EMBEDDING_NORMALIZE,
    expected_use_safetensors: bool = EMBEDDING_USE_SAFETENSORS,
    expected_index_type: str = FAISS_INDEX_TYPE,
    expected_sparse_enabled: bool | None = None,
    expected_chunk_size: int = STRUCTURED_CHUNK_MAX_WORDS,
    expected_chunk_overlap: int = STRUCTURED_CHUNK_OVERLAP,
    document_root: str | Path | None = None,
) -> bool:
    manifest = load_manifest(manifest_path)
    if manifest is None or manifest.get("schema_version") != 4:
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
        and manifest.get("embedding_revision") == expected_embedding_revision
        and manifest.get("embedding_dimension") == expected_embedding_dimension
        and manifest.get("embedding_dtype") == expected_embedding_dtype
        and manifest.get("embedding_normalized") is expected_embedding_normalized
        and manifest.get("embedding_use_safetensors") is expected_use_safetensors
        and manifest.get("faiss_index_type") == expected_index_type
        and manifest.get("chunk_size") == expected_chunk_size
        and manifest.get("chunk_overlap") == expected_chunk_overlap
        and manifest.get("chunker_schema_version") == CHUNKER_SCHEMA_VERSION
        and manifest.get("index_configuration_fingerprint")
        == index_configuration_fingerprint(manifest)
        and (
            expected_sparse_enabled is None
            or manifest.get("sparse_retrieval_enabled") is expected_sparse_enabled
        )
        and (
            expected_sparse_enabled is not True
            or manifest.get("sparse_configuration_fingerprint") == sparse_configuration_fingerprint()
        )
    )


def dense_index_is_fresh(
    pdf_paths: Sequence[DocumentDescriptor | str | Path],
    manifest_path: str | Path = MANIFEST_PATH,
    document_root: str | Path | None = None,
) -> bool:
    """Check the Step-4 dense contract independently of optional sparse artifacts."""
    manifest = load_manifest(manifest_path)
    if manifest is None or int(manifest.get("schema_version", 0)) not in {3, 4}:
        return False
    documents = _coerce_documents(pdf_paths, document_root=document_root)
    expected = DEFAULT_EMBEDDING_CONFIG.compatibility_dict()
    return (
        manifest.get("corpus_fingerprint") == corpus_fingerprint(documents)
        and manifest.get("chunker_schema_version") == CHUNKER_SCHEMA_VERSION
        and manifest.get("chunking_configuration_id") == chunking_configuration_id()
        and manifest.get("faiss_index_type") == FAISS_INDEX_TYPE
        and all(manifest.get(key) == value for key, value in expected.items())
        and manifest.get("index_configuration_fingerprint") == index_configuration_fingerprint(manifest)
    )


def manifest_with_sparse_index(
    manifest: Dict[str, Any],
    sparse_index: SparseIndex,
    sparse_build_seconds: float,
) -> Dict[str, Any]:
    updated = dict(manifest)
    updated.update({
        "schema_version": 4,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sparse_retrieval_enabled": True,
        "sparse_implementation": sparse_index.implementation,
        "sparse_tokenizer_schema": sparse_index.tokenizer_schema,
        "sparse_configuration_fingerprint": sparse_index.configuration_fingerprint,
        "sparse_chunk_count": sparse_index.chunk_count,
        "sparse_vocabulary_size": sparse_index.vocabulary_size,
        "sparse_metadata_binding_hash": sparse_index.binding_hash,
    })
    metrics = dict(updated.get("build_metrics") or {})
    metrics["sparse_index_seconds"] = round(sparse_build_seconds, 6)
    updated["build_metrics"] = metrics
    updated["index_configuration_fingerprint"] = index_configuration_fingerprint(updated)
    return updated


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
    requires_row_binding = bool(manifest and int(manifest.get("schema_version", 0)) >= 3)
    invalid_vector_rows = [
        row for row, item in enumerate(metadata)
        if requires_row_binding and item.get("vector_row") != row
    ]
    manifest_configuration_fingerprint_valid = (
        manifest is None
        or manifest.get("index_configuration_fingerprint") == index_configuration_fingerprint(manifest)
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
        "invalid_vector_rows": invalid_vector_rows,
        "manifest_configuration_fingerprint_valid": manifest_configuration_fingerprint_valid,
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
            not invalid_vector_rows,
            manifest_configuration_fingerprint_valid,
        )
    )
    return result


def atomic_publish_index(
    index: Any,
    metadata: Sequence[Dict[str, Any]],
    manifest: Dict[str, Any],
    documents: Sequence[DocumentDescriptor],
    target_dir: str | Path = VECTOR_DB_DIR,
    sparse_index: SparseIndex | None = None,
) -> Dict[str, Any]:
    """Stage, validate, then replace the index artifact set with rollback."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    names = ["index.faiss", "metadata.pkl", "index_manifest.json"]
    if sparse_index is not None:
        names.append("sparse_index.pkl")
    with tempfile.TemporaryDirectory(prefix="index-stage-", dir=str(target.parent)) as temporary:
        stage = Path(temporary)
        staged_index = stage / names[0]
        staged_metadata = stage / names[1]
        staged_manifest = stage / names[2]
        save_index(index, metadata, staged_index, staged_metadata)
        save_manifest(manifest, staged_manifest)
        if sparse_index is not None:
            save_sparse_index(sparse_index, stage / "sparse_index.pkl")

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
        if sparse_index is not None:
            check_sparse = load_sparse_index(stage / "sparse_index.pkl", check_metadata)
            if (
                not check_manifest
                or not check_manifest.get("sparse_retrieval_enabled")
                or check_manifest.get("sparse_chunk_count") != check_sparse.chunk_count
                or check_manifest.get("sparse_metadata_binding_hash") != check_sparse.binding_hash
            ):
                raise RuntimeError("Staged sparse index failed manifest compatibility checks.")

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
    "FAISS_INDEX_TYPE", "MANIFEST_PATH", "IndexCompatibilityError", "atomic_publish_index",
    "build_index", "build_manifest", "create_index", "dense_index_is_fresh", "document_fingerprint",
    "index_configuration_fingerprint", "index_is_fresh", "load_compatible_index", "load_hybrid_artifacts", "load_index",
    "load_manifest", "manifest_with_sparse_index", "order_metadata_for_embedding", "save_index", "save_manifest", "search",
    "sha256_file", "stale_metadata_sources", "validate_index_metadata_consistency",
    "validate_runtime_index_compatibility",
]
