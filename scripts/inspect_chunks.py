from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chunker import chunk_pages
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENT_DIR
from src.pdf_loader import get_pdf_files, load_pdf_documents


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _configure_stdout()

    pdf_files = get_pdf_files(DOCUMENT_DIR)
    print("PDF files:")
    for path in pdf_files:
        print(f"- {path.name}")

    pages, extraction_issues = load_pdf_documents(DOCUMENT_DIR)
    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    pages_by_source = Counter(page["source"] for page in pages)
    chunks_by_source = Counter(chunk["source"] for chunk in chunks)

    print("\nExtraction summary:")
    print(f"- Extracted pages: {len(pages)}")
    print(f"- Extraction issues: {len(extraction_issues)}")
    for source in sorted(pages_by_source):
        print(f"- {source}: {pages_by_source[source]} text pages, {chunks_by_source[source]} chunks")

    if extraction_issues:
        print("\nExtraction issues:")
        for issue in extraction_issues:
            page = issue["page"] if issue["page"] is not None else "document"
            print(f"- {issue['source']} | page={page} | {issue['issue_type']}: {issue['error']}")

    print("\nChunk settings:")
    print(f"- CHUNK_SIZE={CHUNK_SIZE} words")
    print(f"- CHUNK_OVERLAP={CHUNK_OVERLAP} words")

    print("\nSample chunks:")
    for chunk in chunks[:3]:
        preview = " ".join(chunk["text"].split()[:45])
        print(f"- {chunk['chunk_id']} | source={chunk['source']} | page={chunk['page']}")
        print(f"  {preview}")


if __name__ == "__main__":
    main()
