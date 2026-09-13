from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chunker import chunk_pages, validate_chunk_quality
from src.config import (
    DOCUMENT_DIR,
    INGESTION_REPORT_PATH,
    RESULTS_DIR,
    VECTOR_DB_DIR,
    ensure_directories,
)
from src.corpus import corpus_fingerprint, discover_documents, ingestion_report, write_ingestion_report
from src.embeddings import DEFAULT_EMBEDDING_CONFIG, get_embedding_model
from src.pdf_loader import ingest_documents
from src.sparse_index import SparseIndex
from src.vector_store import (
    atomic_publish_index,
    build_manifest,
    create_index,
    dense_index_is_fresh,
    index_is_fresh,
    load_compatible_index,
    load_hybrid_artifacts,
    manifest_with_sparse_index,
    order_metadata_for_embedding,
    validate_index_metadata_consistency,
)


BUILD_REPORT_PATH = RESULTS_DIR / "index_build_report.json"


def _write_build_report(report: dict) -> None:
    BUILD_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = BUILD_REPORT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, BUILD_REPORT_PATH)


def _build_vector_index(force_rebuild: bool = False) -> dict:
    build_started = time.perf_counter()
    ensure_directories()
    index_path = VECTOR_DB_DIR / "index.faiss"
    metadata_path = VECTOR_DB_DIR / "metadata.pkl"
    manifest_path = VECTOR_DB_DIR / "index_manifest.json"
    sparse_path = VECTOR_DB_DIR / "sparse_index.pkl"
    documents = discover_documents(DOCUMENT_DIR, recursive=True)

    dense_artifacts = all(path.exists() for path in (index_path, metadata_path, manifest_path))
    complete_artifacts = dense_artifacts and sparse_path.exists()
    if not force_rebuild and complete_artifacts and index_is_fresh(
        documents,
        expected_sparse_enabled=True,
    ):
        print(f"Fresh FAISS index found at {index_path}. Reusing existing index.")
        index, metadata, manifest, sparse = load_hybrid_artifacts(
            index_path=index_path,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            configuration=DEFAULT_EMBEDDING_CONFIG,
        )
        consistency = validate_index_metadata_consistency(index, metadata, documents, manifest=manifest)
        if not consistency["ok"]:
            raise RuntimeError(f"Existing index failed consistency checks: {consistency}")
        return {
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
            "manifest_path": str(manifest_path),
            "chunk_count": len(metadata),
            "document_count": len(documents),
            "corpus_fingerprint": manifest["corpus_fingerprint"] if manifest else None,
            "reused": True,
            "consistency": consistency,
            "build_metrics": manifest.get("build_metrics", {}),
            "sparse_vocabulary_size": sparse.vocabulary_size,
            "elapsed_seconds": time.perf_counter() - build_started,
        }

    if not force_rebuild and dense_artifacts and dense_index_is_fresh(documents, manifest_path=manifest_path):
        print("Fresh dense index found. Building and atomically attaching the sparse index.")
        index, metadata, previous_manifest = load_compatible_index(
            index_path=index_path,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            configuration=DEFAULT_EMBEDDING_CONFIG,
        )
        sparse_started = time.perf_counter()
        sparse = SparseIndex.build(metadata)
        sparse_seconds = time.perf_counter() - sparse_started
        manifest = manifest_with_sparse_index(previous_manifest, sparse, sparse_seconds)
        consistency = atomic_publish_index(
            index,
            metadata,
            manifest,
            documents,
            target_dir=VECTOR_DB_DIR,
            sparse_index=sparse,
        )
        print(f"Sparse index attached: {sparse.vocabulary_size} terms for {sparse.chunk_count} chunks.")
        return {
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
            "manifest_path": str(manifest_path),
            "sparse_path": str(sparse_path),
            "chunk_count": len(metadata),
            "document_count": len(documents),
            "corpus_fingerprint": manifest["corpus_fingerprint"],
            "reused": False,
            "dense_reused": True,
            "consistency": consistency,
            "sparse_vocabulary_size": sparse.vocabulary_size,
            "build_metrics": manifest.get("build_metrics", {}),
            "elapsed_seconds": time.perf_counter() - build_started,
        }

    if force_rebuild:
        print("Force rebuild requested. Building staged artifacts.")
    elif complete_artifacts:
        print("Corpus or ingestion configuration changed. Building a safe full replacement.")
    else:
        print("No complete compatible artifact set found. Building a staged index.")

    ingestion_started = time.perf_counter()
    result = ingest_documents(DOCUMENT_DIR, recursive=True, documents=documents)
    chunks = chunk_pages(result.pages)
    ingestion_seconds = time.perf_counter() - ingestion_started
    quality = validate_chunk_quality(chunks)
    if not quality["ok"]:
        raise RuntimeError(f"Structured chunks failed quality validation; previous index was not changed: {quality}")
    counts = Counter(chunk["document_id"] for chunk in chunks)
    for document in result.documents:
        document.chunk_count = counts.get(document.document_id, 0)

    fingerprint = corpus_fingerprint(result.documents)
    report = ingestion_report(result, chunks, fingerprint=fingerprint)
    write_ingestion_report(report, INGESTION_REPORT_PATH)
    if not chunks:
        raise ValueError("No safe non-empty chunks were produced; the previous index was not changed.")

    if result.issues:
        print("Ingestion warnings/errors:")
        for item in result.issues[:20]:
            page_info = f"page {item['page']}" if item.get("page") is not None else "document"
            print(f"- {item['relative_path']} | {page_info} | {item['issue_type']}: {item['error']}")

    chunks = order_metadata_for_embedding(chunks)
    sparse_started = time.perf_counter()
    sparse = SparseIndex.build(chunks)
    sparse_seconds = time.perf_counter() - sparse_started
    model_started = time.perf_counter()
    model = get_embedding_model()
    model_load_seconds = time.perf_counter() - model_started

    def progress(completed: int, total: int) -> None:
        if completed == total or completed % 256 == 0:
            print(f"Embedded {completed} / {total} chunks")

    embedding_started = time.perf_counter()
    vectors = model.embed_many([chunk["text"] for chunk in chunks], progress_callback=progress)
    embedding_seconds = time.perf_counter() - embedding_started
    index_started = time.perf_counter()
    index = create_index(
        vectors,
        normalized=model.configuration.normalize_embeddings,
        norm_tolerance=model.configuration.norm_tolerance,
    )
    index_seconds = time.perf_counter() - index_started
    build_metrics = {
        "ingestion_and_chunking_seconds": round(ingestion_seconds, 6),
        "model_load_seconds": round(model_load_seconds, 6),
        "embedding_seconds": round(embedding_seconds, 6),
        "index_creation_seconds": round(index_seconds, 6),
        "sparse_index_seconds": round(sparse_seconds, 6),
    }
    manifest = build_manifest(
        pdf_paths=result.documents,
        chunk_count=len(chunks),
        embedding_dimension=vectors.shape[1],
        index_type=type(index).__name__,
        embedding_provenance=model.provenance(),
        build_metrics=build_metrics,
        sparse_index=sparse,
    )
    consistency = atomic_publish_index(
        index,
        chunks,
        manifest,
        result.documents,
        target_dir=VECTOR_DB_DIR,
        sparse_index=sparse,
    )

    print(f"Documents discovered: {result.document_count}")
    print(f"Documents successful: {report['successful_document_count']}")
    print(f"Documents partial: {report['partial_document_count']}")
    print(f"Documents failed: {report['failed_document_count']}")
    print(f"Duplicate documents skipped: {report['duplicate_document_count']}")
    print(f"Pages total: {report['pages_total']}")
    print(f"Pages with text: {report['pages_with_text']}")
    print(f"Pages empty: {report['pages_empty']}")
    print(f"Chunks total: {len(chunks)}")
    print(f"Corpus fingerprint: {fingerprint}")
    print(f"Published artifact directory: {VECTOR_DB_DIR}")

    return {
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
        "sparse_path": str(sparse_path),
        "ingestion_report_path": str(INGESTION_REPORT_PATH),
        "chunk_count": len(chunks),
        "document_count": result.document_count,
        "corpus_fingerprint": fingerprint,
        "reused": False,
        "consistency": consistency,
        "report": report,
        "embedding_provenance": model.provenance(),
        "build_metrics": build_metrics,
        "sparse_vocabulary_size": sparse.vocabulary_size,
        "elapsed_seconds": time.perf_counter() - build_started,
    }


def build_vector_index(force_rebuild: bool = False) -> dict:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    base_report = {
        "started_at": started_at.isoformat(),
        "corpus_fingerprint": None,
        "chunk_count": None,
        **DEFAULT_EMBEDDING_CONFIG.compatibility_dict(),
        "embedding_local_only": DEFAULT_EMBEDDING_CONFIG.local_only,
        "embedding_batch_size": DEFAULT_EMBEDDING_CONFIG.batch_size,
        "faiss_index_type": "IndexFlatIP",
        "build_success": False,
        "reused": False,
    }
    try:
        result = _build_vector_index(force_rebuild=force_rebuild)
    except Exception as exc:
        _write_build_report({
            **base_report,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise
    build_report = {
        **base_report,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "corpus_fingerprint": result.get("corpus_fingerprint"),
        "chunk_count": result.get("chunk_count"),
        "document_count": result.get("document_count"),
        "build_success": True,
        "reused": bool(result.get("reused")),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "build_metrics": result.get("build_metrics", {}),
        "sparse_retrieval_enabled": True,
        "sparse_vocabulary_size": result.get("sparse_vocabulary_size"),
        "dense_reused": bool(result.get("dense_reused", result.get("reused"))),
    }
    if result.get("reused") and result.get("build_metrics"):
        build_report["last_full_build_measured_seconds"] = round(
            sum(float(value) for value in result["build_metrics"].values()),
            6,
        )
    _write_build_report(build_report)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or safely refresh the multi-document FAISS index.")
    parser.add_argument("--force", action="store_true", help="Rebuild even when the corpus fingerprint is unchanged.")
    args = parser.parse_args()
    build_vector_index(force_rebuild=args.force)


if __name__ == "__main__":
    main()
