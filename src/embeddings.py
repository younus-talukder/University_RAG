from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .config import EMBEDDING_MODEL

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover
    SentenceTransformer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL, normalize_embeddings: bool = True):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for BGE-M3 embeddings. "
                f"Install the project requirements first. Original import error: {_IMPORT_ERROR}"
            ) from _IMPORT_ERROR

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(
            model_name,
            device="cpu",
            model_kwargs={"use_safetensors": True},
        )

    def embed(self, text: str) -> np.ndarray:
        vector = self.model.encode([text], normalize_embeddings=self.normalize_embeddings)
        return np.asarray(vector[0], dtype=np.float32)

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self.model.encode(list(texts), normalize_embeddings=self.normalize_embeddings)
        return np.asarray(vectors, dtype=np.float32)


def make_embedding_model(model_name: str = EMBEDDING_MODEL, normalize_embeddings: bool = True) -> EmbeddingModel:
    return EmbeddingModel(model_name=model_name, normalize_embeddings=normalize_embeddings)


__all__ = ["EmbeddingModel", "make_embedding_model"]
