from __future__ import annotations

import re
from typing import Any, Dict, List

from .config import TOP_K
from .embeddings import EmbeddingModel
from .vector_store import load_index, search


def _extract_course_hint(question: str) -> str:
    text = question.strip().lower()
    patterns = [
        r"cse\s*101",
        r"mth\s*101",
        r"eng\s*\(?cse\)?\s*101",
        r"english\s*course",
        r"computer\s+fundamentals\s+and\s+programming",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return ""


def _extract_question_keywords(question: str) -> List[str]:
    text = re.sub(r"[^\w\s]", " ", question.lower())
    tokens = [tok for tok in text.split() if len(tok) >= 2]
    stopwords = {
        "the", "what", "which", "where", "when", "why", "how", "is", "are", "of", "in",
        "on", "for", "to", "a", "an", "and", "or", "be", "this", "that", "it", "its",
        "ko", "kono", "ki", "kib", "kobe", "kothay", "kivabe", "ebong", "er", "the", "course",
    }
    return [tok for tok in tokens if tok not in stopwords]


class Retriever:
    def __init__(self, embedding_model: EmbeddingModel | None = None, top_k: int = TOP_K):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.top_k = top_k

    def retrieve(self, question: str, index_path: str | None = None, metadata_path: str | None = None) -> List[Dict[str, Any]]:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if index_path is None or metadata_path is None:
            from .config import VECTOR_DB_DIR
            index_path = str(VECTOR_DB_DIR / "index.faiss")
            metadata_path = str(VECTOR_DB_DIR / "metadata.pkl")

        index, metadata = load_index(index_path=index_path, metadata_path=metadata_path)
        query_vector = self.embedding_model.embed(question)
        raw_results = search(index=index, query_vector=query_vector, metadata=metadata, top_k=max(self.top_k, 15))

        question_keywords = _extract_question_keywords(question)
        course_hint = _extract_course_hint(question)

        ranked = []
        for item in raw_results:
            text = str(item.get("text", ""))
            source = str(item.get("source", ""))
            combined = f"{text} {source}".lower()

            score = float(item.get("score", 0.0))
            if course_hint:
                if course_hint in combined:
                    score += 1.5
                else:
                    score -= 0.75

            keyword_hits = sum(1 for kw in question_keywords if kw and kw in combined)
            score += 0.15 * keyword_hits

            if course_hint and course_hint not in combined and keyword_hits == 0:
                continue

            ranked.append({**item, "score": score})

        if not ranked:
            ranked = raw_results

        ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

        normalized = []
        for item in ranked[: self.top_k]:
            normalized.append({
                "text": item.get("text", ""),
                "score": float(item.get("score", 0.0)),
                "source": item.get("source", "unknown"),
                "page": item.get("page", None),
                "chunk_id": item.get("chunk_id", None),
            })

        return normalized


__all__ = ["Retriever"]
