from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chunker import chunk_pages, validate_chunk_quality
from src.config import (
    DOCUMENT_DIR,
    INGESTION_REPORT_PATH,
    VECTOR_DB_DIR,
    ensure_directories,
)
from src.corpus import corpus_fingerprint, discover_documents, ingestion_report, write_ingestion_report
from src.embeddings import EmbeddingModel
from src.pdf_loader import ingest_documents
from src.vector_store import (
    atomic_publish_index,
    build_manifest,
    create_index,
    index_is_fresh,
    load_index,
    load_manifest,
    validate_index_metadata_consistency,
)


def build_vector_index(force_rebuild: bool = False) -> dict:
    ensure_directories()
    index_path = VECTOR_DB_DIR / "index.faiss"
    metadata_path = VECTOR_DB_DIR / "metadata.pkl"
    manifest_path = VECTOR_DB_DIR / "index_manifest.json"
    documents = discover_documents(DOCUMENT_DIR, recursive=True)

    complete_artifacts = all(path.exists() for path in (index_path, metadata_path, manifest_path))
    if not force_rebuild and complete_artifacts and index_is_fresh(documents):
        print(f"Fresh FAISS index found at {index_path}. Reusing existing index.")
        index, metadata = load_index(index_path=index_path, metadata_path=metadata_path)
        manifest = load_manifest(manifest_path)
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
        }

    if force_rebuild:
        print("Force rebuild requested. Building staged artifacts.")
    elif complete_artifacts:
        print("Corpus or ingestion configuration changed. Building a safe full replacement.")
    else:
        print("No complete compatible artifact set found. Building a staged index.")

    result = ingest_documents(DOCUMENT_DIR, recursive=True, documents=documents)
    chunks = chunk_pages(result.pages)
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

    model = EmbeddingModel()
    vectors = model.embed_many([chunk["text"] for chunk in chunks])
    index = create_index(vectors)
    manifest = build_manifest(
        pdf_paths=result.documents,
        chunk_count=len(chunks),
        embedding_dimension=vectors.shape[1],
        index_type=type(index).__name__,
    )
    consistency = atomic_publish_index(index, chunks, manifest, result.documents, target_dir=VECTOR_DB_DIR)

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
        "ingestion_report_path": str(INGESTION_REPORT_PATH),
        "chunk_count": len(chunks),
        "document_count": result.document_count,
        "corpus_fingerprint": fingerprint,
        "reused": False,
        "consistency": consistency,
        "report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or safely refresh the multi-document FAISS index.")
    parser.add_argument("--force", action="store_true", help="Rebuild even when the corpus fingerprint is unchanged.")
    args = parser.parse_args()
    build_vector_index(force_rebuild=args.force)


if __name__ == "__main__":
    main()
