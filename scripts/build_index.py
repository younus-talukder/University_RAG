from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chunker import chunk_pages
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENT_DIR, VECTOR_DB_DIR, ensure_directories
from src.embeddings import EmbeddingModel
from src.pdf_loader import load_pdf_documents
from src.vector_store import build_index


def build_vector_index(force_rebuild: bool = False) -> dict:
    ensure_directories()
    index_path = VECTOR_DB_DIR / "index.faiss"
    metadata_path = VECTOR_DB_DIR / "metadata.pkl"

    if not force_rebuild and index_path.exists() and metadata_path.exists():
        print(f"Existing FAISS index found at {index_path}. Skipping rebuild.")
        return {
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
            "reused": True,
        }

    pages, extraction_errors = load_pdf_documents(DOCUMENT_DIR)
    if not pages:
        raise ValueError("No extractable text found in the university PDFs. Check the document files and extraction quality.")

    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if not chunks:
        raise ValueError("No chunks were produced from the extracted PDF text.")

    if extraction_errors:
        print("Extraction warnings:")
        for item in extraction_errors[:10]:
            page_info = f"Page {item['page']}" if item.get('page') is not None else "Document"
            print(f"- {item['source']} | {page_info} | {item['error']}")

    model = EmbeddingModel()
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.embed_many(texts)

    build_index(vectors=vectors, metadata=chunks, index_path=index_path)

    print(f"Built FAISS index with {len(chunks)} chunks.")
    print(f"Index path: {index_path}")
    print(f"Metadata path: {metadata_path}")

    return {
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "chunk_count": len(chunks),
        "reused": False,
    }


if __name__ == "__main__":
    build_vector_index(force_rebuild=False)
