from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, Sequence

from .config import (
    APPROVED_RERANKER_MODEL,
    APPROVED_RERANKER_REVISION,
    RERANKER_BATCH_SIZE,
    RERANKER_DEVICE,
    RERANKER_LOCAL_ONLY,
    RERANKER_MAX_LENGTH,
    RERANKER_MODEL,
    RERANKER_REVISION,
)


REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class RerankerUnavailableError(RuntimeError):
    """Raised when reranking is enabled without a validated local model."""


class PairScorer(Protocol):
    def score_pairs(self, query: str, passages: Sequence[str]) -> Sequence[float]: ...


@dataclass(frozen=True)
class LocalRerankerStatus:
    model: str
    available: bool
    revision: str | None
    snapshot_path: str | None
    reason: str


@dataclass(frozen=True)
class RerankerConfiguration:
    model_name: str = RERANKER_MODEL
    revision: str = RERANKER_REVISION
    local_only: bool = RERANKER_LOCAL_ONLY
    device: str = RERANKER_DEVICE
    batch_size: int = RERANKER_BATCH_SIZE
    max_length: int = RERANKER_MAX_LENGTH

    def __post_init__(self) -> None:
        if self.model_name != APPROVED_RERANKER_MODEL or self.revision != APPROVED_RERANKER_REVISION:
            raise ValueError(
                "Only the approved pinned reranker is permitted: "
                f"{APPROVED_RERANKER_MODEL}@{APPROVED_RERANKER_REVISION}."
            )
        if not self.model_name.strip() or self.batch_size <= 0 or self.max_length <= 0:
            raise ValueError("Reranker model name, batch size, and maximum length must be valid.")


DEFAULT_RERANKER_CONFIG = RerankerConfiguration()


def inspect_local_reranker(model: str, required_revision: str | None = None) -> LocalRerankerStatus:
    """Inspect the Hugging Face cache without resolving or downloading anything."""
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
    except Exception as exc:
        return LocalRerankerStatus(model, False, None, None, f"Local Hugging Face cache could not be inspected: {exc}")

    repository = next((repo for repo in cache.repos if repo.repo_id == model and repo.repo_type == "model"), None)
    if repository is None:
        return LocalRerankerStatus(model, False, None, None, "No local cache entry exists for this model.")

    for revision in sorted(repository.revisions, key=lambda item: item.commit_hash):
        if required_revision and revision.commit_hash != required_revision:
            continue
        snapshot = Path(revision.snapshot_path)
        if all((snapshot / name).is_file() and (snapshot / name).stat().st_size > 0 for name in REQUIRED_FILES):
            return LocalRerankerStatus(model, True, revision.commit_hash, str(snapshot), "Complete local snapshot found.")
    suffix = f" for required revision {required_revision}" if required_revision else ""
    return LocalRerankerStatus(model, False, None, None, f"No complete config/tokenizer/weight snapshot was found{suffix}.")


def validate_reranker_snapshot(path: str | Path, configuration: RerankerConfiguration) -> dict[str, Any]:
    snapshot = Path(path)
    if not snapshot.is_dir() or snapshot.name != configuration.revision:
        raise RerankerUnavailableError(
            f"Reranker snapshot does not resolve to the pinned revision {configuration.revision}."
        )
    missing = [name for name in REQUIRED_FILES if not (snapshot / name).is_file() or (snapshot / name).stat().st_size <= 0]
    if missing:
        raise RerankerUnavailableError("Reranker snapshot is incomplete; missing: " + ", ".join(missing))
    try:
        model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise RerankerUnavailableError("Reranker config.json is invalid.") from exc
    if model_config.get("architectures") != ["XLMRobertaForSequenceClassification"]:
        raise RerankerUnavailableError("Reranker architecture is not the approved sequence-classification model.")
    inventory = {name: (snapshot / name).stat().st_size for name in REQUIRED_FILES}
    material = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    return {
        "reranker_model": configuration.model_name,
        "reranker_revision": configuration.revision,
        "reranker_local_only": configuration.local_only,
        "reranker_device": configuration.device,
        "reranker_weight_format": "safetensors",
        "reranker_weight_bytes": inventory["model.safetensors"],
        "reranker_snapshot_bytes": sum(inventory.values()) + ((snapshot / "README.md").stat().st_size if (snapshot / "README.md").is_file() else 0),
        "reranker_inventory_hash": hashlib.sha256(material.encode("utf-8")).hexdigest(),
        "reranker_validated_files": sorted(inventory),
    }


def resolve_reranker_snapshot(configuration: RerankerConfiguration = DEFAULT_RERANKER_CONFIG) -> tuple[Path, dict[str, Any]]:
    try:
        from huggingface_hub import snapshot_download
        resolved = snapshot_download(
            repo_id=configuration.model_name,
            revision=configuration.revision,
            local_files_only=configuration.local_only,
            allow_patterns=list(REQUIRED_FILES) + ["README.md"],
        )
    except Exception as exc:
        policy = "local-only" if configuration.local_only else "online-allowed"
        raise RerankerUnavailableError(
            f"Approved reranker snapshot is unavailable under {policy}: "
            f"{configuration.model_name}@{configuration.revision}."
        ) from exc
    snapshot = Path(resolved).resolve()
    return snapshot, validate_reranker_snapshot(snapshot, configuration)


class RerankerModel:
    def __init__(self, configuration: RerankerConfiguration = DEFAULT_RERANKER_CONFIG):
        self.configuration = configuration
        self.snapshot_path, self.snapshot_validation = resolve_reranker_snapshot(configuration)
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.snapshot_path), local_files_only=True, trust_remote_code=False
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.snapshot_path),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            ).to(configuration.device)
            self.model.eval()
        except Exception as exc:
            raise RerankerUnavailableError(
                f"Pinned reranker could not initialize on {configuration.device}: "
                f"{configuration.model_name}@{configuration.revision}."
            ) from exc

    def provenance(self) -> dict[str, Any]:
        return {**self.snapshot_validation, "reranker_batch_size": self.configuration.batch_size, "reranker_max_length": self.configuration.max_length}

    def score_pairs(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        if not passages:
            return []
        scores: list[float] = []
        for start in range(0, len(passages), self.configuration.batch_size):
            batch = list(passages[start:start + self.configuration.batch_size])
            encoded = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.configuration.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.configuration.device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                logits = self.model(**encoded).logits
            if logits.ndim != 2 or logits.shape[0] != len(batch) or logits.shape[1] != 1:
                raise RuntimeError(f"Reranker returned invalid logits shape {tuple(logits.shape)}.")
            scores.extend(float(value) for value in logits[:, 0].detach().cpu().tolist())
        if len(scores) != len(passages) or not all(math.isfinite(score) for score in scores):
            raise RuntimeError("Reranker output count or finite-score validation failed.")
        return scores


@lru_cache(maxsize=2)
def get_reranker_model(configuration: RerankerConfiguration = DEFAULT_RERANKER_CONFIG) -> RerankerModel:
    return RerankerModel(configuration)


def rerank_candidate_pool(
    query: str,
    candidates: Sequence[dict[str, Any]],
    scorer: PairScorer,
    candidate_k: int,
) -> list[dict[str, Any]]:
    if candidate_k <= 0:
        raise ValueError("Reranker candidate count must be positive.")
    pool = [dict(item) for item in candidates[:candidate_k]]
    scores = list(scorer.score_pairs(query, [str(item.get("text") or "") for item in pool]))
    if len(scores) != len(pool):
        raise ValueError("Reranker returned a different number of scores than candidate pairs.")
    if not all(math.isfinite(float(score)) for score in scores):
        raise ValueError("Reranker returned NaN or infinite scores.")
    for item, score in zip(pool, scores):
        item["reranker_score"] = float(score)
        item["pre_rerank_rank"] = item.get("final_rank")
    pool.sort(key=lambda item: (-item["reranker_score"], int(item.get("pre_rerank_rank") or 10**9), str(item.get("chunk_id") or "")))
    for rank, item in enumerate(pool, start=1):
        item["final_rank"] = rank
    return pool


__all__ = [
    "LocalRerankerStatus",
    "DEFAULT_RERANKER_CONFIG",
    "PairScorer",
    "REQUIRED_FILES",
    "RerankerConfiguration",
    "RerankerModel",
    "RerankerUnavailableError",
    "get_reranker_model",
    "inspect_local_reranker",
    "rerank_candidate_pool",
    "resolve_reranker_snapshot",
    "validate_reranker_snapshot",
]
