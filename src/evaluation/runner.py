"""Sequential, resumable benchmark execution with a strict runtime boundary."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from src.answer_safety import answer_safety_failures
from src.evaluation.manifest import (atomic_json, code_identity, compatibility_identity,
                                     corpus_identity, environment_identity, model_identity,
                                     new_run_id, retrieval_identity)
from src.evaluation.schema import LANGUAGES, expand, load_and_validate


def _bool(value: Any) -> bool:
    return value is True or str(value).casefold() == "true"


def _telemetry() -> dict[str, Any]:
    try:
        import psutil
        process = psutil.Process()
        memory = process.memory_info()
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {"rss_bytes": memory.rss,
                "peak_process_bytes": getattr(memory, "peak_wset", memory.rss),
                "total_ram_bytes": virtual.total, "available_ram_bytes": virtual.available,
                "pagefile_used_bytes_system": swap.used,
                "pagefile_total_bytes_system": swap.total}
    except (ImportError, OSError):
        return {"rss_bytes": "NOT_AVAILABLE", "peak_process_bytes": "NOT_AVAILABLE",
                "total_ram_bytes": "NOT_AVAILABLE", "available_ram_bytes": "NOT_AVAILABLE",
                "pagefile_used_bytes_system": "NOT_AVAILABLE",
                "pagefile_total_bytes_system": "NOT_AVAILABLE"}


def _truncate_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def _candidate(item: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank, "source": item.get("source") or "",
        "relative_path": item.get("relative_path") or "",
        "document_id": item.get("document_id") or "",
        "page": item.get("page") or "", "chunk_id": item.get("chunk_id") or "",
        "score": item.get("score") if item.get("score") is not None else "",
        "dense_rank": item.get("dense_rank") or "",
        "sparse_rank": item.get("sparse_rank") or "",
        "metadata_rank": item.get("metadata_rank") or "",
        "text": str(item.get("text") or item.get("supporting_excerpt") or ""),
        "field_types": item.get("field_types") or [],
    }


def infer_one(descriptor: dict[str, Any], run_id: str, *, top_k: int = 3,
              use_generation: bool = True,
              infer: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Only question text enters runtime. Labels join the record after inference."""
    if infer is None:
        from src.pipeline import answer_question
        infer = answer_question
    started = time.perf_counter()
    question = descriptor["question"]
    response: dict[str, Any] = {}
    error = ""
    try:
        response = infer(question, top_k=top_k, use_generation=use_generation,
                         use_answer_bank=False)
        if not isinstance(response, dict):
            raise TypeError("runtime response must be a dict")
    except Exception as exc:
        error = _truncate_error(exc)
    elapsed = time.perf_counter() - started
    status = str(response.get("answerability_status") or "SYSTEM_ERROR")
    runtime_error = error or str(response.get("runtime_error") or "")
    if runtime_error:
        status = "SYSTEM_ERROR"
    answer_returned = status == "SUPPORTED" and _bool(response.get("final_answer_allowed", True))
    evidence = list(response.get("evidence") or [])
    excerpts = [str(item.get("excerpt") or "") for item in evidence]
    relation = response.get("relation_extracted") or {}
    safety = list(answer_safety_failures(
        question, str(response.get("answer") or ""), excerpts,
        extracted_relation=response.get("relation_type"),
        relation_values=relation.get("values") or (),
    )) if answer_returned else []
    latency = response.get("latency_seconds") or {}
    result = {
        "run_id": run_id, "evaluation_id": descriptor["evaluation_id"],
        "base_question_id": descriptor["base_question_id"], "language": descriptor["language"],
        "question": question, "reference_answer": descriptor["reference_answer"],
        "reference_available": bool(descriptor["reference_answer"]),
        "expected_sources": descriptor["expected_sources"],
        "expected_pages": descriptor["expected_pages"],
        "expected_evidence": descriptor["expected_evidence"],
        "annotation_status": descriptor["annotation_status"],
        "intent": descriptor["intent"], "difficulty": descriptor["difficulty"],
        "course_code": descriptor["course_code"],
        "detected_language": response.get("detected_language") or "",
        "requested_entity": response.get("requested_entity") or "",
        "requested_field": response.get("requested_field") or "",
        "requested_relation": response.get("requested_relation") or "",
        "answerability_status": status,
        "answerability_reason": response.get("answerability_reason") or "",
        "evidence_level": response.get("evidence_level") or "",
        "answer_strategy": response.get("answer_strategy") or "",
        "requested_answer_strategy": response.get("requested_answer_strategy") or "",
        "final_answer": response.get("answer") or "",
        "answer_returned": answer_returned,
        "source": response.get("source") or "",
        "relative_path": response.get("relative_path") or "",
        "document_id": response.get("document_id") or "",
        "page": response.get("page") or "",
        "chunk_id": response.get("chunk_id") or "",
        "supporting_excerpt": response.get("supporting_excerpt") or "",
        "supporting_evidence_count": response.get("supporting_evidence_count") or 0,
        "independent_support_count": response.get("independent_support_count") or 0,
        "retrieval_channels": response.get("retrieval_channels") or [],
        "best_support_rank": response.get("best_support_rank") or "",
        "cross_lingual_fallback_used": _bool(response.get("cross_lingual_fallback_used")),
        "generation_used": _bool(response.get("generation_used")),
        "generation_attempts": int(response.get("generation_attempts") or 0),
        "retry_used": _bool(response.get("retry_used")),
        "grounding_status": response.get("grounding_validation_passed"),
        "semantic_status": response.get("semantic_validation_passed"),
        "language_status": response.get("language_validation_passed"),
        "language_consistency": response.get("language_consistency"),
        "answer_safety_failures": safety,
        "extracted_relation": response.get("relation_type") or "",
        "extracted_values": relation.get("values") or [],
        "latency_seconds": elapsed,
        "retrieval_seconds": latency.get("retrieval", "NOT_AVAILABLE"),
        "generation_seconds": latency.get("generation", "NOT_AVAILABLE"),
        "answerability_seconds": response.get("answerability_seconds", "NOT_AVAILABLE"),
        "error_type": runtime_error.split(":", 1)[0] if runtime_error else "",
        "error": runtime_error[:300],
        **_telemetry(),
        "retrieval_top3": [_candidate(item, rank) for rank, item in
                           enumerate((response.get("retrieved_context") or [])[:3], 1)],
    }
    return result


def _batch_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "checkpoint").glob("batch-*.json"))


def checkpoint_records(run_dir: Path, compatibility: str) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    for expected_number, path in enumerate(_batch_paths(run_dir), 1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("batch_number") != expected_number or payload.get("compatibility_sha256") != compatibility:
            raise ValueError(f"unsafe checkpoint batch: {path.name}")
        for row in payload.get("records", []):
            key = row.get("evaluation_id")
            if not key or key in seen:
                raise ValueError(f"duplicate or missing checkpoint evaluation_id: {key}")
            seen.add(key)
            yield row


def _write_batch(run_dir: Path, compatibility: str, batch: list[dict[str, Any]]) -> None:
    number = len(_batch_paths(run_dir)) + 1
    atomic_json(run_dir / "checkpoint" / f"batch-{number:06d}.json",
                {"batch_number": number, "compatibility_sha256": compatibility,
                 "records": batch})


def _atomic_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with staged.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                                 for key, value in row.items()})
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


RESULT_FIELDS = [
    "run_id", "evaluation_id", "base_question_id", "language", "question",
    "reference_answer", "reference_available", "expected_sources", "expected_pages",
    "expected_evidence", "annotation_status", "intent", "difficulty", "course_code",
    "detected_language", "requested_entity", "requested_field", "requested_relation",
    "answerability_status", "answerability_reason", "evidence_level", "answer_strategy",
    "requested_answer_strategy",
    "final_answer", "answer_returned", "source", "relative_path", "document_id", "page",
    "chunk_id", "supporting_excerpt", "supporting_evidence_count", "independent_support_count",
    "retrieval_channels", "best_support_rank", "cross_lingual_fallback_used",
    "generation_used", "generation_attempts", "retry_used", "grounding_status", "semantic_status",
    "language_status", "language_consistency", "answer_safety_failures", "extracted_relation",
    "extracted_values", "latency_seconds", "retrieval_seconds", "generation_seconds",
    "answerability_seconds", "error_type", "error", "rss_bytes", "peak_process_bytes",
    "total_ram_bytes", "available_ram_bytes", "pagefile_used_bytes_system",
    "pagefile_total_bytes_system",
]
TRACE_FIELDS = ["run_id", "evaluation_id", "language", "rank", "source", "relative_path",
                "document_id", "page", "chunk_id", "score", "dense_rank", "sparse_rank",
                "metadata_rank", "text", "field_types"]


def export_records(run_dir: Path, compatibility: str) -> None:
    _atomic_csv(run_dir / "results.csv", RESULT_FIELDS, checkpoint_records(run_dir, compatibility))
    def traces() -> Iterator[dict[str, Any]]:
        for record in checkpoint_records(run_dir, compatibility):
            for candidate in record.get("retrieval_top3", []):
                yield {"run_id": record["run_id"], "evaluation_id": record["evaluation_id"],
                       "language": record["language"], **candidate}
    _atomic_csv(run_dir / "retrieval_trace.csv", TRACE_FIELDS, traces())


def prepare_run(dataset: Path, output_dir: Path, *, languages: tuple[str, ...] = LANGUAGES,
                limit: int | None = None, run_name: str = "benchmark",
                checkpoint_every: int = 10, use_generation: bool = True,
                run_classification: str = "DEVELOPMENT", strict: bool = True,
                require_references: bool = False, resume: Path | None = None,
                seed: int = 42, human_sample_size: int = 30,
                run_mode: str = "full") -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    if checkpoint_every < 1 or (limit is not None and limit < 1):
        raise ValueError("checkpoint interval and limit must be positive")
    if run_classification not in {"DEVELOPMENT", "PILOT", "FINAL_BENCHMARK"}:
        raise ValueError("invalid run classification")
    if run_mode not in {"full", "skip_generation", "retrieval_only"}:
        raise ValueError("invalid run mode")
    if use_generation != (run_mode == "full"):
        raise ValueError("generation setting and run mode disagree")
    if run_classification == "FINAL_BENCHMARK" and (not strict or not use_generation or limit is not None):
        raise ValueError("FINAL_BENCHMARK requires strict full-generation, unlimited inference")
    rows, validation = load_and_validate(dataset, languages=languages,
                                         require_references=require_references)
    if not validation["valid"]:
        raise ValueError("dataset validation failed: " + "; ".join(validation["errors"][:10]))
    descriptors = list(expand(rows, languages))
    if limit is not None:
        descriptors = descriptors[:limit]
    from src.config import RERANKER_ENABLED, GENERATOR_TEMPERATURE
    if strict and (RERANKER_ENABLED or GENERATOR_TEMPERATURE != 0):
        raise ValueError("strict benchmark requires reranker OFF and temperature 0")
    effective = {
        "dataset": str(Path(dataset).resolve()), "dataset_sha256": validation["dataset_sha256"],
        "languages": list(languages), "limit": limit, "top_k": 3,
        "checkpoint_every": checkpoint_every, "run_mode": run_mode,
        "use_generation": use_generation,
        "run_classification": run_classification, "strict": strict,
        "require_references": require_references, "seed": seed,
        "human_sample_size": human_sample_size,
        "metrics_enabled": ["retrieval", "answerability", "answer_proxies", "parity", "latency", "human_sample"],
        "retrieval_settings": retrieval_identity(), "model_settings": model_identity(),
        "environment": environment_identity(),
        "answer_bank_enabled": False,
    }
    corpus = corpus_identity()
    model = effective["model_settings"]
    code = code_identity()
    compatibility = compatibility_identity(validation["dataset_sha256"], effective, corpus, model, code)
    if resume is None:
        run_id = new_run_id(run_name)
        run_dir = Path(output_dir).resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "checkpoint").mkdir()
        manifest = {
            "run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_classification": run_classification, "dataset": validation,
            "corpus_index": corpus, "model_runtime": model, "code_version": code,
            "environment": effective["environment"], "resource_preflight": _telemetry(),
            "compatibility_sha256": compatibility,
            "expected_evaluation_rows": len(descriptors), "completed_rows": 0,
            "run_complete": False,
            "memory_safety_stop": False,
            "leakage_audit": {"passed": True, "runtime_input_fields": ["question"],
                              "reference_fields_passed_to_runtime": [], "answer_bank_enabled": False},
        }
        atomic_json(run_dir / "dataset_validation.json", validation)
        atomic_json(run_dir / "expected_evaluation_ids.json",
                    [row["evaluation_id"] for row in descriptors])
        atomic_json(run_dir / "run_config.json", effective)
        atomic_json(run_dir / "run_manifest.json", manifest)
    else:
        run_dir = Path(resume).resolve()
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        old_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        if old_config != effective or manifest.get("compatibility_sha256") != compatibility:
            raise ValueError("unsafe resume: dataset, configuration, corpus/index, model, or code differs")
        if manifest.get("expected_evaluation_rows") != len(descriptors):
            raise ValueError("unsafe resume: descriptor count differs")
        saved_ids = json.loads((run_dir / "expected_evaluation_ids.json").read_text(encoding="utf-8"))
        if saved_ids != [row["evaluation_id"] for row in descriptors]:
            raise ValueError("unsafe resume: evaluation IDs differ")
        run_id = manifest["run_id"]
    return run_dir, descriptors, manifest


def run_benchmark(dataset: Path, output_dir: Path, *, languages: tuple[str, ...] = LANGUAGES,
                  limit: int | None = None, run_name: str = "benchmark", checkpoint_every: int = 10,
                  use_generation: bool = True, run_classification: str = "DEVELOPMENT",
                  strict: bool = True, require_references: bool = False,
                  resume: Path | None = None, seed: int = 42, stop_after: int | None = None,
                  infer: Callable[..., dict[str, Any]] | None = None,
                  human_sample_size: int = 30, run_mode: str = "full") -> Path:
    if run_classification == "FINAL_BENCHMARK" and stop_after is not None:
        raise ValueError("FINAL_BENCHMARK does not allow stop_after")
    run_dir, descriptors, manifest = prepare_run(
        dataset, output_dir, languages=languages, limit=limit, run_name=run_name,
        checkpoint_every=checkpoint_every, use_generation=use_generation,
        run_classification=run_classification, strict=strict,
        require_references=require_references, resume=resume, seed=seed,
        human_sample_size=human_sample_size, run_mode=run_mode)
    compatibility = manifest["compatibility_sha256"]
    expected = {row["evaluation_id"] for row in descriptors}
    completed = {row["evaluation_id"] for row in checkpoint_records(run_dir, compatibility)}
    if not completed <= expected:
        raise ValueError("checkpoint contains an unexpected evaluation_id")
    batch: list[dict[str, Any]] = []
    started = time.perf_counter()
    new_count = 0
    recent_seconds: list[float] = []
    memory_safety_stop = False
    for descriptor in descriptors:
        if descriptor["evaluation_id"] in completed:
            continue
        record = infer_one(descriptor, manifest["run_id"], top_k=3,
                           use_generation=use_generation, infer=infer)
        recent_seconds.append(float(record["latency_seconds"]))
        recent_seconds = recent_seconds[-20:]
        batch.append(record)
        new_count += 1
        if len(batch) >= checkpoint_every:
            _write_batch(run_dir, compatibility, batch)
            completed.update(item["evaluation_id"] for item in batch)
            batch = []
        done = len(completed) + len(batch)
        elapsed = time.perf_counter() - started
        eta = (len(descriptors) - done) * (sum(recent_seconds) / len(recent_seconds)) if recent_seconds else 0
        print(f"{done}/{len(descriptors)} {descriptor['language']} {descriptor['evaluation_id']} "
              f"elapsed={elapsed:.1f}s ETA={'warming-up' if new_count < 5 else f'{eta:.1f}s'}", flush=True)
        error = str(record.get("error") or "").casefold()
        # Low free physical RAM alone is not a failure on the sequential mmap/pagefile
        # profile; the historical Step-8B run completed despite a very low minimum.
        if "memoryerror" in error or "out of memory" in error or "allocation failed" in error:
            memory_safety_stop = True
            break
        if stop_after is not None and new_count >= stop_after:
            break
    if batch:
        _write_batch(run_dir, compatibility, batch)
        completed.update(item["evaluation_id"] for item in batch)
    run_lock_changed = False
    if manifest["run_classification"] == "FINAL_BENCHMARK":
        effective = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        current_dataset_hash = hashlib.sha256(Path(dataset).read_bytes()).hexdigest()
        current_compatibility = compatibility_identity(
            current_dataset_hash, effective, corpus_identity(), model_identity(), code_identity())
        run_lock_changed = current_compatibility != compatibility
    manifest["completed_rows"] = len(completed)
    manifest["run_complete"] = completed == expected and not memory_safety_stop and not run_lock_changed
    manifest["memory_safety_stop"] = memory_safety_stop
    if memory_safety_stop:
        manifest["stop_reason"] = "MEMORY_SAFETY_STOP"
    if run_lock_changed:
        manifest["run_invalidated"] = True
        manifest["stop_reason"] = "FINAL_RUN_LOCK_CHANGED"
    manifest["last_updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(run_dir / "run_manifest.json", manifest)
    export_records(run_dir, compatibility)
    from src.evaluation.metrics import analyze_run
    analyze_run(run_dir, human_sample_size=human_sample_size, seed=seed)
    if run_lock_changed:
        raise RuntimeError("FINAL_BENCHMARK run lock changed; run invalidated and a new run ID is required")
    return run_dir
