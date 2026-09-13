from __future__ import annotations

import hashlib
import json
import platform
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_DIMENSION,
    EMBEDDING_DTYPE,
    EMBEDDING_LOCAL_ONLY,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
    EMBEDDING_NORM_TOLERANCE,
    EMBEDDING_REVISION,
    EMBEDDING_USE_SAFETENSORS,
)

try:
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - exercised only with missing dependencies
    SentenceTransformer = None
    snapshot_download = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class EmbeddingError(RuntimeError):
    """Base class for understandable embedding failures."""


class EmbeddingSnapshotError(EmbeddingError):
    """Raised when the configured model snapshot is missing or incomplete."""


class EmbeddingValidationError(EmbeddingError):
    """Raised when model output violates the vector contract."""


@dataclass(frozen=True)
class EmbeddingConfiguration:
    model_name: str = EMBEDDING_MODEL
    revision: str = EMBEDDING_REVISION
    device: str = EMBEDDING_DEVICE
    batch_size: int = EMBEDDING_BATCH_SIZE
    dimension: int = EMBEDDING_DIMENSION
    normalize_embeddings: bool = EMBEDDING_NORMALIZE
    local_only: bool = EMBEDDING_LOCAL_ONLY
    use_safetensors: bool = EMBEDDING_USE_SAFETENSORS
    dtype: str = EMBEDDING_DTYPE
    norm_tolerance: float = EMBEDDING_NORM_TOLERANCE

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("Embedding model name cannot be empty.")
        if not self.revision.strip():
            raise ValueError("Embedding model revision must be an immutable revision or local snapshot identity.")
        if self.batch_size <= 0 or self.dimension <= 0:
            raise ValueError("Embedding batch size and dimension must be positive.")
        if self.dtype != "float32":
            raise ValueError("The current FAISS contract requires float32 embeddings.")
        if self.norm_tolerance <= 0:
            raise ValueError("Embedding norm tolerance must be positive.")

    def compatibility_dict(self) -> dict[str, object]:
        return {
            "embedding_model": self.model_name,
            "embedding_revision": self.revision,
            "embedding_dimension": self.dimension,
            "embedding_normalized": self.normalize_embeddings,
            "embedding_dtype": self.dtype,
            "embedding_use_safetensors": self.use_safetensors,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.compatibility_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


DEFAULT_EMBEDDING_CONFIG = EmbeddingConfiguration()


def _version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"


def dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "sentence_transformers": _version("sentence-transformers"),
        "transformers": _version("transformers"),
        "torch": _version("torch"),
        "faiss": _version("faiss-cpu"),
        "huggingface_hub": _version("huggingface-hub"),
        "numpy": np.__version__,
    }


def _required_snapshot_groups(config: EmbeddingConfiguration) -> dict[str, tuple[str, ...]]:
    weight = ("model.safetensors",) if config.use_safetensors else ("pytorch_model.bin",)
    return {
        "transformer configuration": ("config.json",),
        "sentence-transformer modules": ("modules.json",),
        "sentence-transformer runtime configuration": ("sentence_bert_config.json",),
        "sentence-transformer version configuration": ("config_sentence_transformers.json",),
        "pooling configuration": ("1_Pooling/config.json",),
        "tokenizer": ("tokenizer.json", "sentencepiece.bpe.model"),
        "tokenizer configuration": ("tokenizer_config.json",),
        "special-token configuration": ("special_tokens_map.json",),
        "model weights": weight,
    }


def validate_model_snapshot(
    path: str | Path,
    config: EmbeddingConfiguration = DEFAULT_EMBEDDING_CONFIG,
) -> dict[str, object]:
    snapshot = Path(path)
    if not snapshot.is_dir():
        raise EmbeddingSnapshotError(f"Embedding model snapshot missing: {snapshot}")

    missing: list[str] = []
    inventory: dict[str, int] = {}
    for label, alternatives in _required_snapshot_groups(config).items():
        existing = [snapshot / relative for relative in alternatives if (snapshot / relative).is_file()]
        if not existing:
            missing.append(f"{label} ({' or '.join(alternatives)})")
            continue
        for item in existing:
            if item.stat().st_size <= 0:
                missing.append(f"{label} ({item.relative_to(snapshot).as_posix()} is empty)")
            inventory[item.relative_to(snapshot).as_posix()] = item.stat().st_size

    if missing:
        raise EmbeddingSnapshotError(
            "Embedding model snapshot incomplete. Missing required files: " + "; ".join(missing)
        )

    inventory_material = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    return {
        "snapshot_complete": True,
        "snapshot_revision": config.revision,
        "weight_format": "safetensors" if config.use_safetensors else "pytorch_bin",
        "snapshot_inventory_hash": hashlib.sha256(inventory_material.encode("utf-8")).hexdigest(),
        "validated_files": sorted(inventory),
    }


def resolve_model_snapshot(
    config: EmbeddingConfiguration = DEFAULT_EMBEDDING_CONFIG,
) -> tuple[Path, dict[str, object]]:
    local_path = Path(config.model_name)
    if local_path.is_dir():
        snapshot = local_path.resolve()
    else:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", config.revision):
            raise EmbeddingSnapshotError(
                "Remote embedding revision must be an immutable 40-character commit hash; "
                f"received {config.revision!r}."
            )
        if snapshot_download is None:
            raise EmbeddingSnapshotError(
                "huggingface-hub is required to resolve the configured embedding snapshot."
            ) from _IMPORT_ERROR
        try:
            allowed_files = sorted({
                relative
                for alternatives in _required_snapshot_groups(config).values()
                for relative in alternatives
            })
            resolved = snapshot_download(
                repo_id=config.model_name,
                revision=config.revision,
                local_files_only=config.local_only,
                allow_patterns=allowed_files,
            )
        except Exception as exc:
            policy = "LOCAL_ONLY" if config.local_only else "ONLINE_ALLOWED"
            raise EmbeddingSnapshotError(
                f"Embedding model snapshot missing for {config.model_name}@{config.revision} under {policy}. "
                "Provide the complete approved snapshot or explicitly allow online resolution."
            ) from exc
        snapshot = Path(resolved).resolve()
    return snapshot, validate_model_snapshot(snapshot, config)


def validate_embeddings(
    vectors: np.ndarray,
    expected_rows: int,
    config: EmbeddingConfiguration = DEFAULT_EMBEDDING_CONFIG,
) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise EmbeddingValidationError(f"Embedding output must be two-dimensional; received shape {array.shape}.")
    if array.shape[0] != expected_rows:
        raise EmbeddingValidationError(
            f"Embedding row count mismatch: expected {expected_rows}, received {array.shape[0]}."
        )
    if array.shape[1] != config.dimension:
        raise EmbeddingValidationError(
            f"Embedding dimension mismatch: expected {config.dimension}, received {array.shape[1]}."
        )
    if not np.isfinite(array).all():
        raise EmbeddingValidationError("Embedding output contains NaN or infinite values.")
    if config.normalize_embeddings and expected_rows:
        norms = np.linalg.norm(array, axis=1)
        if not np.allclose(norms, 1.0, atol=config.norm_tolerance, rtol=0.0):
            worst = float(np.max(np.abs(norms - 1.0)))
            raise EmbeddingValidationError(
                f"Normalized embedding contract failed; maximum norm deviation is {worst:.6g}."
            )
    return np.ascontiguousarray(array, dtype=np.float32)


class EmbeddingModel:
    def __init__(
        self,
        model_name: str | None = None,
        normalize_embeddings: bool | None = None,
        configuration: EmbeddingConfiguration | None = None,
    ):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for BGE-M3 embeddings. "
                f"Install the project requirements first. Original import error: {_IMPORT_ERROR}"
            ) from _IMPORT_ERROR

        config = configuration or DEFAULT_EMBEDDING_CONFIG
        if model_name is not None:
            config = replace(config, model_name=model_name)
        if normalize_embeddings is not None:
            config = replace(config, normalize_embeddings=normalize_embeddings)
        self.configuration = config
        self.model_name = config.model_name
        self.normalize_embeddings = config.normalize_embeddings
        self.snapshot_path, self.snapshot_validation = resolve_model_snapshot(config)
        try:
            self.model = SentenceTransformer(
                str(self.snapshot_path),
                device=config.device,
                trust_remote_code=False,
                local_files_only=True,
                model_kwargs={
                    "use_safetensors": config.use_safetensors,
                    "local_files_only": True,
                },
            )
        except Exception as exc:
            raise EmbeddingSnapshotError(
                f"Embedding model snapshot could not be loaded: {config.model_name}@{config.revision}."
            ) from exc

    def provenance(self) -> dict[str, object]:
        device_class = self.configuration.device.split(":", 1)[0].casefold()
        return {
            **self.configuration.compatibility_dict(),
            **self.snapshot_validation,
            "embedding_device_class": device_class,
            "embedding_batch_size": self.configuration.batch_size,
            "embedding_local_only": self.configuration.local_only,
            "embedding_configuration_fingerprint": self.configuration.fingerprint(),
            "embedding_libraries": dependency_versions(),
        }

    def embed(self, text: str) -> np.ndarray:
        vectors = self.embed_many([text])
        return vectors[0]

    def embed_many(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        total = len(texts)
        if total == 0:
            return np.empty((0, self.configuration.dimension), dtype=np.float32)
        selected_batch_size = batch_size or self.configuration.batch_size
        if selected_batch_size <= 0:
            raise ValueError("Embedding batch size must be positive.")

        batches: list[np.ndarray] = []
        for start in range(0, total, selected_batch_size):
            values = list(texts[start:start + selected_batch_size])
            encoded = self.model.encode(
                values,
                batch_size=selected_batch_size,
                normalize_embeddings=self.configuration.normalize_embeddings,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            batch = validate_embeddings(encoded, len(values), self.configuration)
            batches.append(batch)
            completed = min(start + len(values), total)
            if progress_callback is not None:
                progress_callback(completed, total)
        return validate_embeddings(np.concatenate(batches, axis=0), total, self.configuration)


@lru_cache(maxsize=4)
def _cached_embedding_model(configuration: EmbeddingConfiguration) -> EmbeddingModel:
    return EmbeddingModel(configuration=configuration)


def get_embedding_model(
    configuration: EmbeddingConfiguration = DEFAULT_EMBEDDING_CONFIG,
) -> EmbeddingModel:
    return _cached_embedding_model(configuration)


def make_embedding_model(
    model_name: str = EMBEDDING_MODEL,
    normalize_embeddings: bool = EMBEDDING_NORMALIZE,
) -> EmbeddingModel:
    config = replace(
        DEFAULT_EMBEDDING_CONFIG,
        model_name=model_name,
        normalize_embeddings=normalize_embeddings,
    )
    return get_embedding_model(config)


__all__ = [
    "DEFAULT_EMBEDDING_CONFIG",
    "EmbeddingConfiguration",
    "EmbeddingError",
    "EmbeddingModel",
    "EmbeddingSnapshotError",
    "EmbeddingValidationError",
    "dependency_versions",
    "get_embedding_model",
    "make_embedding_model",
    "resolve_model_snapshot",
    "validate_embeddings",
    "validate_model_snapshot",
]
