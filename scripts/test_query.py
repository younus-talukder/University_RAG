from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import TOP_K, VECTOR_DB_DIR
from src.embeddings import EmbeddingModel
from src.vector_store import load_index, search


def run_test_query(question: str, top_k: int = TOP_K) -> None:
    print(f"Question: {question}\n")
    model = EmbeddingModel()
    query_vector = model.embed(question)

    index, metadata = load_index(index_path=VECTOR_DB_DIR / "index.faiss", metadata_path=VECTOR_DB_DIR / "metadata.pkl")
    results = search(index=index, query_vector=query_vector, metadata=metadata, top_k=top_k)

    print("RETRIEVED CHUNKS:")
    for i, item in enumerate(results, start=1):
        score = item.get("score", 0.0)
        source = item.get("source", "unknown")
        page = item.get("page", "unknown")
        text = item.get("text", "")
        print(f"\n[{i}] score={score:.4f} | source={source} | page={page}")
        print(text[:500])

    print("\nEND")


if __name__ == "__main__":
    q = input("Question: ")
    run_test_query(q, top_k=TOP_K)
