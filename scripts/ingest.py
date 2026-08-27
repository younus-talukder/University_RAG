from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chunker import chunk_pages
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENT_DIR
from src.pdf_loader import load_pdf_documents


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _configure_stdout()
    pages, extraction_issues = load_pdf_documents(DOCUMENT_DIR)
    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    print(f"Extracted pages: {len(pages)}")
    print(f"Chunks produced: {len(chunks)}")

    by_source = Counter(chunk["source"] for chunk in chunks)
    for source, count in sorted(by_source.items()):
        print(f"- {source}: {count} chunks")

    if extraction_issues:
        print("\nExtraction warnings:")
        for issue in extraction_issues:
            page = issue["page"] if issue["page"] is not None else "document"
            print(f"- {issue['source']} | page={page} | {issue['issue_type']}: {issue['error']}")


if __name__ == "__main__":
    main()
