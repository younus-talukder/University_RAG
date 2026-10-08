"""Reproducible, secret-free benchmark provenance and atomic file writes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from src import config

ROOT = Path(__file__).resolve().parents[2]


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with staged.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(10):
            try:
                os.replace(staged, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.2 * (attempt + 1))
    finally:
        if staged.exists():
            staged.unlink()


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _git(*args: str) -> str:
    try:
        process = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                 text=True, encoding="utf-8", timeout=10, check=False)
        return process.stdout.strip() if process.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def code_identity() -> dict[str, Any]:
    status = _git("status", "--porcelain", "--untracked-files=all")
    files = list((ROOT / "src").rglob("*.py"))
    files.extend((ROOT / "scripts").glob("evaluate_benchmark.py"))
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "git_commit": _git("rev-parse", "HEAD") or "NOT_AVAILABLE",
        "working_tree_dirty": bool(status),
        "modified_file_count": len(status.splitlines()) if status else 0,
        "code_sha256": digest.hexdigest(),
    }


def corpus_identity(index_manifest: Path | None = None) -> dict[str, Any]:
    path = index_manifest or config.VECTOR_DB_DIR / "index_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    fields = (
        "corpus_fingerprint", "index_configuration_fingerprint", "document_count",
        "chunk_count", "embedding_model", "embedding_revision", "chunker_schema_version",
        "faiss_index_type", "sparse_implementation", "sparse_tokenizer_schema",
        "sparse_configuration_fingerprint", "chunking_configuration_id",
    )
    return {"manifest_path": str(path.resolve()), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **{key: data.get(key, "NOT_AVAILABLE") for key in fields}}


@lru_cache(maxsize=8)
def _file_sha256(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_identity() -> dict[str, Any]:
    settings = config.load_generator_settings()
    model_path = Path(settings.model_path)
    stem = model_path.name
    prefix = stem.split("-00001-of-")[0] if "-00001-of-" in stem else model_path.stem
    shard_paths = sorted(model_path.parent.glob(f"{prefix}*.gguf"))
    shards = [item.name for item in shard_paths]
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for path in shard_paths:
        stat = path.stat()
        hashes[path.name] = _file_sha256(str(path), stat.st_size, stat.st_mtime_ns)
        sizes[path.name] = stat.st_size
    return {
        "generator_model_name": prefix, "gguf_shard_filenames": shards or [stem],
        "quantization": "Q4_K_M" if "q4_k_m" in stem.casefold() else "NOT_AVAILABLE",
        "model_hashes": hashes if hashes else "NOT_AVAILABLE",
        "model_shard_sizes_bytes": sizes if sizes else "NOT_AVAILABLE",
        "profile": settings.profile,
        "context_size": settings.context_size, "threads": settings.threads,
        "batch_size": settings.batch_size, "gpu_layers": settings.gpu_layers,
        "temperature": settings.temperature, "top_p": settings.top_p,
        "max_tokens": settings.max_tokens, "use_mmap": settings.use_mmap,
        "use_mlock": settings.use_mlock,
        "reranker_enabled": config.RERANKER_ENABLED, "answer_bank_enabled": False,
    }


def environment_identity() -> dict[str, Any]:
    packages = ("faiss-cpu", "numpy", "sentence-transformers", "torch", "llama-cpp-python",
                "rank-bm25", "psutil", "openpyxl", "transformers")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return {"python": platform.python_version(), "os": platform.platform(),
            "dependency_versions": versions}


def retrieval_identity() -> dict[str, Any]:
    fields = (
        "DENSE_CANDIDATE_K", "SPARSE_CANDIDATE_K", "METADATA_CANDIDATE_K",
        "RRF_K", "RRF_DENSE_WEIGHT", "RRF_SPARSE_WEIGHT", "RRF_METADATA_WEIGHT",
        "DENSE_QUERY_STRATEGY", "TOP_K", "RERANKER_ENABLED",
    )
    return {field.casefold(): getattr(config, field) for field in fields}


def new_run_id(name: str = "benchmark") -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "-" for character in name).strip("-")
    safe = safe[:40] or "benchmark"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{safe}-{timestamp}-{uuid.uuid4().hex[:6]}"


def compatibility_identity(dataset_sha256: str, effective_config: dict[str, Any],
                           corpus: dict[str, Any], model: dict[str, Any],
                           code: dict[str, Any]) -> str:
    value = {
        "dataset_sha256": dataset_sha256,
        "effective_config": effective_config,
        "corpus_fingerprint": corpus["corpus_fingerprint"],
        "index_configuration_fingerprint": corpus["index_configuration_fingerprint"],
        "index_manifest_sha256": corpus["manifest_sha256"],
        "model": model, "code_sha256": code["code_sha256"],
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
