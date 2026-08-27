from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    faiss = None
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None

from .config import VECTOR_DB_DIR


def build_index(vectors: np.ndarray, metadata: Sequence[Dict[str, Any]], index_path: str | Path = VECTOR_DB_DIR / "index.faiss") -> Any:
    if faiss is None:
        raise ImportError("faiss-cpu is required for the vector database. Install project requirements first.") from _FAISS_IMPORT_ERROR

    if len(vectors) == 0:
        raise ValueError("Cannot build an index from an empty vector list.")

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors.astype(np.float32))

    save_index(index, metadata, index_path=index_path)
    return index


def save_index(index: Any, metadata: Sequence[Dict[str, Any]], index_path: str | Path = VECTOR_DB_DIR / "index.faiss", metadata_path: str | Path = VECTOR_DB_DIR / "metadata.pkl") -> None:
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    with metadata_path.open("wb") as handle:
        pickle.dump(list(metadata), handle)


def load_index(index_path: str | Path = VECTOR_DB_DIR / "index.faiss", metadata_path: str | Path = VECTOR_DB_DIR / "metadata.pkl") -> Tuple[Any, List[Dict[str, Any]]]:
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if faiss is None:
        raise ImportError("faiss-cpu is required for the vector database. Install project requirements first.") from _FAISS_IMPORT_ERROR

    index = faiss.read_index(str(index_path))
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    return index, list(metadata)


def search(index: Any, query_vector: np.ndarray, metadata: Sequence[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    similarity_scores, indices = index.search(query_vector.astype(np.float32), top_k)
    results: List[Dict[str, Any]] = []

    for score, idx in zip(similarity_scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = dict(metadata[int(idx)])
        item["score"] = float(score)
        results.append(item)

    return results


__all__ = ["build_index", "save_index", "load_index", "search"]
