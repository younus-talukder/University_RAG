from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chunker import chunk_pages, chunk_statistics, validate_chunk_quality
from src.config import DOCUMENT_DIR, INGESTION_REPORT_PATH
from src.corpus import DocumentStatus, corpus_fingerprint, ingestion_report, write_ingestion_report
from src.pdf_loader import ingest_documents


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _configure_stdout()
    result = ingest_documents(DOCUMENT_DIR, recursive=True)
    chunks = chunk_pages(result.pages)
    counts = Counter(chunk["document_id"] for chunk in chunks)
    for document in result.documents:
        document.chunk_count = counts.get(document.document_id, 0)

    fingerprint = corpus_fingerprint(result.documents)
    report = ingestion_report(result, chunks, fingerprint=fingerprint)
    write_ingestion_report(report, INGESTION_REPORT_PATH)

    print(f"Documents discovered: {report['document_count']}")
    print(f"Documents successful: {report['successful_document_count']}")
    print(f"Documents partial: {report['partial_document_count']}")
    print(f"Documents failed: {report['failed_document_count']}")
    print(f"Duplicate documents skipped: {report['duplicate_document_count']}")
    print(f"Pages total: {report['pages_total']}")
    print(f"Pages with text: {report['pages_with_text']}")
    print(f"Pages empty: {report['pages_empty']}")
    print(f"Chunks total: {report['chunk_count']}")
    stats = chunk_statistics(chunks)
    quality = validate_chunk_quality(chunks)
    print(f"Chunk words: mean={stats['average_words']:.1f}, median={stats['median_words']}, min={stats['min_words']}, max={stats['max_words']}")
    print(f"Mixed-entity chunks flagged: {len(quality['mixed_entity_chunks'])}")
    print(f"Corpus fingerprint: {fingerprint}")
    print(f"Machine-readable report: {INGESTION_REPORT_PATH}")

    problem_documents = [
        document
        for document in result.documents
        if document.ingestion_status in {
            DocumentStatus.PARTIAL.value,
            DocumentStatus.FAILED.value,
            DocumentStatus.DUPLICATE.value,
        }
    ]
    if problem_documents:
        print("\nPartial, failed, or duplicate documents:")
        for document in problem_documents:
            reasons = "; ".join(issue["error"] for issue in document.ingestion_errors) or "; ".join(document.warnings)
            print(f"- {document.relative_path} | {document.ingestion_status}: {reasons}")


if __name__ == "__main__":
    main()
