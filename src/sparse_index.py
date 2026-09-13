from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import (
    BM25_B,
    BM25_IMPLEMENTATION,
    BM25_K1,
    SPARSE_TOKENIZER_SCHEMA,
    VECTOR_DB_DIR,
)


SPARSE_INDEX_PATH = VECTOR_DB_DIR / "sparse_index.pkl"
COURSE_PATTERN = re.compile(
    r"(?<!\w)([A-Za-z]{2,12})\s*[-]?\s*(\d{2,4})(?:\s*\(\s*([A-Za-z0-9]+)\s*\))?",
    flags=re.UNICODE,
)
TOKEN_PATTERN = re.compile(
    r"[\w.+-]+@[\w.-]+\.[^\W\d_]{2,}|\d{1,4}(?:[-/]\d{1,2}){1,2}|\d+(?:\.\d+)?%?|[\u0980-\u09ff]+|[^\W\d_]+",
    flags=re.UNICODE,
)


class SparseIndexError(RuntimeError):
    """Raised when a sparse artifact is missing, stale, or corrupt."""


def normalize_entity(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", str(value)).casefold())


def sparse_tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    tokens = [match.group(0) for match in TOKEN_PATTERN.finditer(normalized)]
    for match in COURSE_PATTERN.finditer(normalized):
        prefix, number, suffix = match.groups()
        compact = f"{prefix}{number}"
        aliases = (prefix, number, compact)
        if suffix:
            aliases += (suffix.casefold(), f"{compact}{suffix.casefold()}")
        tokens.extend(aliases)
    return [token for token in tokens if token]


def sparse_configuration_fingerprint(k1: float = BM25_K1, b: float = BM25_B) -> str:
    material = json.dumps(
        {
            "implementation": BM25_IMPLEMENTATION,
            "tokenizer_schema": SPARSE_TOKENIZER_SCHEMA,
            "k1": k1,
            "b": b,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def metadata_binding_hash(metadata: Sequence[dict[str, Any]]) -> str:
    rows = [
        {
            "vector_row": item.get("vector_row"),
            "chunk_id": item.get("chunk_id"),
            "content_sha256": hashlib.sha256(
                str(item.get("text") or "").encode("utf-8")
            ).hexdigest(),
        }
        for item in metadata
    ]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class SparseIndex:
    chunk_count: int
    document_lengths: list[int]
    average_document_length: float
    postings: dict[str, list[tuple[int, int]]]
    entity_rows: dict[str, list[int]]
    field_rows: dict[str, list[int]]
    binding_hash: str
    k1: float = BM25_K1
    b: float = BM25_B
    implementation: str = BM25_IMPLEMENTATION
    tokenizer_schema: str = SPARSE_TOKENIZER_SCHEMA

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    @property
    def configuration_fingerprint(self) -> str:
        return sparse_configuration_fingerprint(self.k1, self.b)

    @classmethod
    def build(
        cls,
        metadata: Sequence[dict[str, Any]],
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> "SparseIndex":
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_lengths: list[int] = []
        entity_rows: dict[str, list[int]] = defaultdict(list)
        field_rows: dict[str, list[int]] = defaultdict(list)
        for row, item in enumerate(metadata):
            tokens = sparse_tokenize(str(item.get("text") or ""))
            document_lengths.append(len(tokens))
            for token, frequency in Counter(tokens).items():
                postings[token].append((row, frequency))
            entities = {item.get("entity_id"), item.get("course_code")}
            for entity in entities:
                normalized = normalize_entity(str(entity or ""))
                if normalized and row not in entity_rows[normalized]:
                    entity_rows[normalized].append(row)
            for field in item.get("field_types") or []:
                field_rows[str(field).casefold()].append(row)
        count = len(metadata)
        return cls(
            chunk_count=count,
            document_lengths=document_lengths,
            average_document_length=(sum(document_lengths) / count if count else 0.0),
            postings=dict(postings),
            entity_rows=dict(entity_rows),
            field_rows=dict(field_rows),
            binding_hash=metadata_binding_hash(metadata),
            k1=k1,
            b=b,
        )

    def validate(self, metadata: Sequence[dict[str, Any]]) -> None:
        if self.implementation != BM25_IMPLEMENTATION or self.tokenizer_schema != SPARSE_TOKENIZER_SCHEMA:
            raise SparseIndexError("Sparse index configuration is incompatible and must be rebuilt.")
        if self.configuration_fingerprint != sparse_configuration_fingerprint():
            raise SparseIndexError("Sparse index configuration fingerprint is stale.")
        if self.chunk_count != len(metadata) or len(self.document_lengths) != len(metadata):
            raise SparseIndexError("Sparse index row count does not match metadata.")
        if self.binding_hash != metadata_binding_hash(metadata):
            raise SparseIndexError("Sparse index is bound to different chunk text or metadata ordering.")
        for token, entries in self.postings.items():
            if not token or any(row < 0 or row >= self.chunk_count or frequency <= 0 for row, frequency in entries):
                raise SparseIndexError("Sparse index contains an invalid posting.")

    def search(
        self,
        query: str,
        metadata: Sequence[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or not query.strip() or not self.chunk_count:
            return []
        scores: dict[int, float] = defaultdict(float)
        for token in set(sparse_tokenize(query)):
            entries = self.postings.get(token, [])
            if not entries:
                continue
            document_frequency = len(entries)
            inverse_document_frequency = math.log(
                1.0 + (self.chunk_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for row, frequency in entries:
                length = self.document_lengths[row]
                denominator = frequency + self.k1 * (
                    1.0 - self.b
                    + self.b * length / max(self.average_document_length, 1.0)
                )
                scores[row] += inverse_document_frequency * frequency * (self.k1 + 1.0) / denominator
        ranked = sorted(
            ((row, score) for row, score in scores.items() if score > 0.0),
            key=lambda pair: (-pair[1], str(metadata[pair[0]].get("chunk_id") or "")),
        )[:top_k]
        return [
            {**dict(metadata[row]), "sparse_score": float(score), "sparse_rank": rank}
            for rank, (row, score) in enumerate(ranked, start=1)
        ]

    def metadata_candidates(
        self,
        entity: str,
        requested_field: str | None,
        metadata: Sequence[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        normalized = normalize_entity(entity)
        rows = list(self.entity_rows.get(normalized, [])) if normalized else []
        requested = str(requested_field or "").casefold()
        ranked = sorted(
            rows,
            key=lambda row: (
                0 if requested and requested in {str(value).casefold() for value in metadata[row].get("field_types") or []} else 1,
                int(metadata[row].get("page") or 0),
                str(metadata[row].get("chunk_id") or ""),
            ),
        )[:top_k]
        return [
            {**dict(metadata[row]), "metadata_rank": rank}
            for rank, row in enumerate(ranked, start=1)
        ]


def save_sparse_index(index: SparseIndex, path: str | Path = SPARSE_INDEX_PATH) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_sparse_index(
    path: str | Path = SPARSE_INDEX_PATH,
    metadata: Sequence[dict[str, Any]] | None = None,
) -> SparseIndex:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Sparse index not found: {target}")
    try:
        with target.open("rb") as handle:
            index = pickle.load(handle)
    except Exception as exc:
        raise SparseIndexError(f"Sparse index could not be loaded: {target.name}") from exc
    if not isinstance(index, SparseIndex):
        raise SparseIndexError("Sparse index artifact has an unexpected type.")
    if metadata is not None:
        index.validate(metadata)
    return index


__all__ = [
    "SPARSE_INDEX_PATH",
    "SparseIndex",
    "SparseIndexError",
    "load_sparse_index",
    "metadata_binding_hash",
    "normalize_entity",
    "save_sparse_index",
    "sparse_configuration_fingerprint",
    "sparse_tokenize",
]
