from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chunker import chunk_pages, chunk_statistics, validate_chunk_quality
from src.config import DOCUMENT_DIR, STRUCTURED_CHUNK_MAX_WORDS, STRUCTURED_CHUNK_OVERLAP
from src.pdf_loader import ingest_documents


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _filtered(chunks: list[dict], source: str | None, page: int | None, entity: str | None, chunk_id: str | None) -> list[dict]:
    selected = chunks
    if source:
        selected = [item for item in selected if source.casefold() in str(item.get("relative_path") or item.get("source", "")).casefold()]
    if page is not None:
        selected = [item for item in selected if item.get("page") == page]
    if entity:
        selected = [item for item in selected if entity.casefold() in str(item.get("entity_id") or item.get("text", "")).casefold()]
    if chunk_id:
        selected = [item for item in selected if chunk_id.casefold() in str(item.get("chunk_id", "")).casefold()]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect structure-aware chunks without dumping the full corpus.")
    parser.add_argument("--source", help="Filter by source filename or relative path.")
    parser.add_argument("--page", type=int, help="Filter by one-based physical PDF page.")
    parser.add_argument("--entity", help="Filter by detected entity/course or text.")
    parser.add_argument("--chunk-id", help="Filter by full or partial chunk ID.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum chunk previews to print (default: 20).")
    args = parser.parse_args()
    _configure_stdout()

    result = ingest_documents(DOCUMENT_DIR, recursive=True)
    chunks = chunk_pages(result.pages)
    stats = chunk_statistics(chunks)
    quality = validate_chunk_quality(chunks)
    by_document = Counter(chunk["relative_path"] for chunk in chunks)
    by_page = Counter((chunk["relative_path"], chunk["page"]) for chunk in chunks)

    print("Chunk summary:")
    print(f"- Total chunks: {stats['chunk_count']}")
    print(f"- Words: average={stats['average_words']:.1f}, median={stats['median_words']}, min={stats['min_words']}, max={stats['max_words']}")
    print(f"- Blocks: {len({chunk['block_id'] for chunk in chunks})}")
    print(f"- Entity/course chunks: {stats['entity_chunks']}")
    for block_type, count in stats["block_type_counts"].items():
        print(f"- {block_type}: {count}")
    print(f"- Mixed-entity flagged: {len(quality['mixed_entity_chunks'])}")
    print(f"- Very small flagged: {len(quality['very_short_chunks'])}")
    print(f"- Very large flagged: {len(quality['very_long_chunks'])}")
    print(f"- Duplicate text flagged: {len(quality['duplicate_text_chunks'])}")
    print(f"- STRUCTURED_CHUNK_MAX_WORDS={STRUCTURED_CHUNK_MAX_WORDS}")
    print(f"- STRUCTURED_CHUNK_OVERLAP={STRUCTURED_CHUNK_OVERLAP}")

    print("\nChunks by document:")
    for source, count in sorted(by_document.items()):
        print(f"- {source}: {count}")
    print(f"Pages represented: {len(by_page)}")

    selected = _filtered(chunks, args.source, args.page, args.entity, args.chunk_id)
    print(f"\nMatching chunks: {len(selected)}; showing at most {max(0, args.limit)}")
    for chunk in selected[:max(0, args.limit)]:
        preview = " ".join(chunk["text"].split()[:55])
        print(
            f"- {chunk['chunk_id']} | {chunk['relative_path']} | page={chunk['page']} | "
            f"type={chunk['block_type']} | entity={chunk.get('entity_id')} | words={len(chunk['text'].split())}"
        )
        print(f"  {preview}")


if __name__ == "__main__":
    main()
