from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .config import TOP_K, VECTOR_DB_DIR
from .embeddings import EmbeddingModel
from .generator import generate_answer
from .retriever import Retriever
from .vector_store import load_index


def answer_question(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    index_path = str(VECTOR_DB_DIR / "index.faiss")
    metadata_path = str(VECTOR_DB_DIR / "metadata.pkl")

    try:
        load_index(index_path=index_path, metadata_path=metadata_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "FAISS index is missing. Build the index first using the ingestion/build scripts."
        ) from exc

    retriever = Retriever(embedding_model=EmbeddingModel(), top_k=top_k)
    retrieved = retriever.retrieve(question, index_path=index_path, metadata_path=metadata_path)

    context = [item["text"] for item in retrieved if item.get("text")]
    if not context:
        answer = "I could not find the requested information in the available university documents."
    else:
        answer = generate_answer(question=question, context=context)

    sources = [
        {
            "source": item.get("source"),
            "page": item.get("page"),
            "score": item.get("score"),
        }
        for item in retrieved
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_context": retrieved,
    }


__all__ = ["answer_question"]
