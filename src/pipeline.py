from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict

from .config import TOP_K, VECTOR_DB_DIR
from .embeddings import EmbeddingModel
from .fast_answer import build_extractive_answer, retrieve_lexical
from .generator import generate_answer, regenerate_answer_for_language
from .language_detector import detect_language
from .language_validator import unsupported_answer, validate_language
from .retriever import Retriever
from .vector_store import load_index


StatusCallback = Callable[[str], None]


@lru_cache(maxsize=1)
def _get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()


def answer_question(
    question: str,
    top_k: int = TOP_K,
    status_callback: StatusCallback | None = None,
    use_generation: bool = True,
) -> Dict[str, Any]:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    def report(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    report("Detecting question language...")
    detected_language = detect_language(question)
    index_path = str(VECTOR_DB_DIR / "index.faiss")
    metadata_path = str(VECTOR_DB_DIR / "metadata.pkl")

    try:
        report("Checking FAISS vector index...")
        index, metadata = load_index(index_path=index_path, metadata_path=metadata_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "FAISS index is missing. Build the index first using the ingestion/build scripts."
        ) from exc

    if use_generation:
        report("Loading embedding model...")
        embedding_model = _get_embedding_model()
        report("Retrieving relevant context...")
        retriever = Retriever(embedding_model=embedding_model, top_k=top_k)
        retrieved = retriever.retrieve(question, index_path=index_path, metadata_path=metadata_path)
    else:
        report("Retrieving relevant context with fast lexical search...")
        retrieved = retrieve_lexical(question, metadata=metadata, top_k=top_k)

    context = [item["text"] for item in retrieved if item.get("text")]
    if not context:
        answer = unsupported_answer(detected_language)
    elif not use_generation:
        report("Building fast answer from retrieved document text...")
        answer = build_extractive_answer(question, retrieved=retrieved, language=detected_language)
    else:
        report("Loading/generating with the local Qwen model...")
        answer = generate_answer(question=question, context=context, language=detected_language)

    report("Validating response language...")
    validation = validate_language(answer, detected_language)
    if use_generation and not validation["language_consistency"] and context:
        report("Regenerating answer in the detected language...")
        answer = regenerate_answer_for_language(
            question=question,
            context=context,
            language=detected_language,
            previous_answer=answer,
        )
        validation = validate_language(answer, detected_language)
    if use_generation and not validation["language_consistency"] and not context:
        answer = unsupported_answer(detected_language)
        validation = validate_language(answer, detected_language)

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
        "detected_language": detected_language,
        "answer": answer,
        "generation_mode": "local_llm" if use_generation else "fast_extractive",
        "response_language": validation["response_language"],
        "language_consistency": validation["language_consistency"],
        "sources": sources,
        "retrieved_context": retrieved,
    }


__all__ = ["answer_question"]
