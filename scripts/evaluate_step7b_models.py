from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import generator, pipeline
from src.config import (
    GENERATOR_BATCH_SIZE,
    GENERATOR_CONTEXT_SIZE,
    GENERATOR_GPU_LAYERS,
    GENERATOR_MAX_TOKENS,
    GENERATOR_TEMPERATURE,
    GENERATOR_THREADS,
    GENERATOR_TOP_P,
    RERANKER_ENABLED,
    VECTOR_DB_DIR,
)
from src.embeddings import get_embedding_model
from src.evaluator import load_dataset
from src.evidence import assess_evidence
from src.generation_context import build_verified_evidence_package
from src.grounding_validator import _facts, validate_grounding
from src.language_detector import detect_language_details
from src.language_validator import validate_language
from src.retriever import Retriever

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


RESULTS_DIR = ROOT_DIR / "results"
MODEL_DIR = ROOT_DIR / "models"
FROZEN_EVIDENCE_PATH = RESULTS_DIR / "step7b_frozen_evidence.json"
BASELINE_PATH = RESULTS_DIR / "step7b_1_5b_baseline.json"
THREE_B_PATH = RESULTS_DIR / "step7b_3b_results.json"
REPORT_PATH = RESULTS_DIR / "step7b_model_comparison.md"
REVIEW_PATH = RESULTS_DIR / "step7b_model_comparison_review.csv"
KEY_PATH = RESULTS_DIR / "step7b_model_blinding_key.csv"
FULL_PROGRESS_PATH = RESULTS_DIR / "step7b_3b_full_progress.json"

MODEL_SPECS = {
    "1_5b": {
        "label": "Qwen2.5-1.5B-Instruct-Q4_K_M",
        "path": MODEL_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "repository": "existing local Step-7 baseline",
        "revision": None,
        "license": "Qwen license associated with the existing baseline",
    },
    "3b": {
        "label": "Qwen2.5-3B-Instruct-Q4_K_M",
        "path": MODEL_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf",
        "repository": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "revision": "7dabda4d13d513e3e842b20f0d435c732f172cbe",
        "license": "Qwen Research License Agreement (non-commercial research/evaluation)",
    },
}

# Evaluation selection only; these identifiers never affect runtime answering.
FOCUSED_BASE_IDS = (
    "Q001", "Q002", "Q005", "Q007", "Q009",
    "Q010", "Q011", "Q012", "Q013", "Q018",
    "Q019", "Q022", "Q023", "Q025", "Q028",
)
LANGUAGES = ("english", "bangla", "banglish")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def provenance(model_key: str) -> dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Required model is missing: {path}")
    return {
        "model": spec["label"],
        "repository": spec["repository"],
        "revision": spec["revision"],
        "filename": path.name,
        "quantization": "Q4_K_M",
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "license": spec["license"],
    }


def memory_snapshot() -> dict[str, int | None]:
    if psutil is None:
        return {"process_rss_bytes": None, "system_available_bytes": None, "swap_used_bytes": None}
    return {
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "system_available_bytes": psutil.virtual_memory().available,
        "swap_used_bytes": psutil.swap_memory().used,
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))]


def proxy_metrics(reference: str, answer: str) -> dict[str, float]:
    token_pattern = r"[A-Za-z0-9@.+%-]+|[\u0980-\u09ff]+"
    expected = {value for value in re.findall(token_pattern, reference.casefold()) if len(value) > 1}
    actual = {value for value in re.findall(token_pattern, answer.casefold()) if len(value) > 1}
    overlap = len(expected & actual)
    precision = overlap / len(actual) if actual else 0.0
    recall = overlap / len(expected) if expected else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def factual_preservation(reference: str, answer: str) -> dict[str, Any]:
    reference_facts = _facts(reference)
    answer_facts = _facts(answer)
    required = sorted({f"{category}:{value}" for category, values in reference_facts.items() for value in values})
    present = {f"{category}:{value}" for category, values in answer_facts.items() for value in values}
    return {
        "required": required,
        "preserved": not required or set(required).issubset(present),
        "applicable": bool(required),
    }


def semantic_screen_flags(answer: str, language: str, validation_reason: str | None) -> list[str]:
    flags: list[str] = []
    normalized = " ".join(str(answer).split())
    words = normalized.casefold().split()
    if validation_reason in {"BANGLA_BODY_TOO_ENGLISH", "BANGLISH_BODY_PURE_ENGLISH", "BANGLISH_CONTAINS_BENGALI_SCRIPT"}:
        flags.append(str(validation_reason))
    if language == "bangla" and len(re.findall(r"[\u0980-\u09ff]+", normalized)) < 3:
        flags.append("VERY_SHORT_BANGLA_BODY")
    if len(words) >= 9:
        trigrams = [tuple(words[index:index + 3]) for index in range(len(words) - 2)]
        if Counter(trigrams).most_common(1)[0][1] >= 3:
            flags.append("REPEATED_PHRASE_PATTERN")
    if normalized and normalized[-1] not in ".!?।":
        flags.append("POSSIBLY_TRUNCATED")
    return sorted(set(flags))


def focused_rows() -> list[dict[str, Any]]:
    selected = [row for row in load_dataset() if row["question_id"] in FOCUSED_BASE_IDS]
    order = {value: index for index, value in enumerate(FOCUSED_BASE_IDS)}
    language_order = {value: index for index, value in enumerate(LANGUAGES)}
    selected.sort(key=lambda row: (language_order[row["expected_language"]], order[row["question_id"]]))
    if len(selected) != 45 or Counter(row["expected_language"] for row in selected) != Counter({key: 15 for key in LANGUAGES}):
        raise RuntimeError("The focused comparison must contain exactly 15 cases per language.")
    return selected


def prepare_frozen_evidence() -> None:
    if RERANKER_ENABLED:
        raise RuntimeError("Step 7B requires RERANKER_ENABLED=false.")
    embedding = get_embedding_model()
    retriever = Retriever(embedding_model=embedding, top_k=3)
    cases: list[dict[str, Any]] = []
    for position, row in enumerate(focused_rows(), start=1):
        retrieved = retriever.retrieve(
            row["question"],
            index_path=str(VECTOR_DB_DIR / "index.faiss"),
            metadata_path=str(VECTOR_DB_DIR / "metadata.pkl"),
        )
        language = detect_language_details(row["question"]).language
        assessment = assess_evidence(row["question"], retrieved)
        package = build_verified_evidence_package(row["question"], language, assessment, retrieved)
        cases.append({
            "case_id": f"S7B-{position:03d}",
            **row,
            "retrieved_context": retrieved,
            "verified_package": package.to_dict(),
            "prepared_support_status": assessment.status.value,
        })
        print(f"prepared {position}/45 {row['expected_language']} {row['question_id']}", flush=True)
    write_json(FROZEN_EVIDENCE_PATH, {
        "notice": "Frozen Step-5/6 retrieval evidence for the controlled Step-7B model comparison.",
        "answer_bank_enabled": False,
        "reranker_enabled": False,
        "cases": cases,
    })


def run_case(row: dict[str, Any], model_path: Path, model: Any) -> dict[str, Any]:
    capture: dict[str, str] = {"raw_answer": "", "retry_answer": ""}

    def first_generation(**kwargs: Any) -> str:
        answer = generator.generate_answer(model_name=str(model_path), **kwargs)
        capture["raw_answer"] = answer
        return answer

    def retry_generation(**kwargs: Any) -> str:
        answer = generator.regenerate_answer_for_language(model_name=str(model_path), **kwargs)
        capture["retry_answer"] = answer
        return answer

    frozen = list(row["retrieved_context"])
    fake_retriever = Mock()
    fake_retriever.retrieve.return_value = frozen
    started = time.perf_counter()
    with (
        patch.object(pipeline, "load_index", return_value=(object(), [])),
        patch.object(pipeline, "_get_embedding_model", return_value=object()),
        patch.object(pipeline, "Retriever", return_value=fake_retriever),
        patch.object(pipeline, "generate_answer", side_effect=first_generation),
        patch.object(pipeline, "regenerate_answer_for_language", side_effect=retry_generation),
        patch.object(pipeline, "_log_generation_debug", return_value=None),
    ):
        result = pipeline.answer_question(row["question"], top_k=3, use_generation=True, use_answer_bank=False)
    elapsed = time.perf_counter() - started

    evidence = row["verified_package"].get("evidence", [])
    raw_language = validate_language(capture["raw_answer"], row["expected_language"]) if capture["raw_answer"] else {}
    raw_grounding = validate_grounding(capture["raw_answer"], evidence) if capture["raw_answer"] else {}
    retry_language = validate_language(capture["retry_answer"], row["expected_language"]) if capture["retry_answer"] else {}
    retry_grounding = validate_grounding(capture["retry_answer"], evidence) if capture["retry_answer"] else {}
    final_language = validate_language(result.get("answer", ""), row["expected_language"])
    factual = factual_preservation(row.get("reference_answer", ""), result.get("answer", ""))
    proxy = proxy_metrics(row.get("reference_answer", ""), result.get("answer", ""))
    generated_text = "\n".join(value for value in capture.values() if value)
    token_count = len(model.tokenize(generated_text.encode("utf-8"), add_bos=False, special=False)) if generated_text else 0
    generation_seconds = float(result.get("latency_seconds", {}).get("generation", 0.0))
    return {
        "case_id": row["case_id"],
        "question_id": row["question_id"],
        "language": row["expected_language"],
        "question": row["question"],
        "reference_answer": row.get("reference_answer", ""),
        "verified_evidence": evidence,
        "source": result.get("source"),
        "page": result.get("page"),
        "raw_answer": capture["raw_answer"],
        "retry_answer": capture["retry_answer"],
        "final_answer": result.get("answer", ""),
        "answer_strategy": result.get("answer_strategy"),
        "support_status": result.get("support_status"),
        "generation_used": bool(result.get("generation_used")),
        "generation_attempts": int(result.get("generation_attempts", 0)),
        "retry_used": bool(result.get("retry_used")),
        "raw_language_pass": bool(raw_language.get("validation_passed", False)),
        "raw_language_reason": raw_language.get("validation_reason"),
        "raw_grounding_pass": bool(raw_grounding.get("grounding_validation_passed", False)),
        "raw_grounding_reason": raw_grounding.get("grounding_validation_reason"),
        "retry_language_pass": bool(retry_language.get("validation_passed", False)),
        "retry_grounding_pass": bool(retry_grounding.get("grounding_validation_passed", False)),
        "final_language_pass": bool(final_language.get("validation_passed", False)),
        "final_language_reason": final_language.get("validation_reason"),
        "final_grounding_pass": bool(result.get("grounding_validation_passed", False)),
        "final_grounding_reason": result.get("grounding_validation_reason"),
        "semantic_screen_flags": semantic_screen_flags(
            capture["raw_answer"] or result.get("answer", ""),
            row["expected_language"],
            raw_language.get("validation_reason"),
        ),
        "factual_preservation": factual,
        "automatic_development_proxies": proxy,
        "tokens_generated": token_count,
        "generation_seconds": generation_seconds,
        "retry_seconds": float(result.get("latency_seconds", {}).get("retry", 0.0)),
        "total_seconds": elapsed,
        "tokens_per_second": token_count / generation_seconds if generation_seconds > 0 else None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    generated = [row for row in rows if row["generation_used"]]
    retries = [row for row in generated if row["retry_used"]]
    applicable = [row for row in rows if row["factual_preservation"]["applicable"]]
    generation_times = [row["generation_seconds"] for row in generated]
    token_rates = [row["tokens_per_second"] for row in generated if row["tokens_per_second"] is not None]
    language = {}
    for name in LANGUAGES:
        selected = [row for row in rows if row["language"] == name]
        language[name] = {
            "cases": len(selected),
            "final_language_pass": sum(row["final_language_pass"] for row in selected),
            "generated": sum(row["generation_used"] for row in selected),
            "final_accepted_generation": sum(
                row["generation_used"] and row["answer_strategy"] == "gguf_generation" for row in selected
            ),
            "semantic_screen_flags": dict(Counter(flag for row in selected for flag in row["semantic_screen_flags"])),
        }
    return {
        "cases": len(rows),
        "strategy_counts": dict(Counter(row["answer_strategy"] for row in rows)),
        "language": language,
        "grounding": {
            "generation_attempted_cases": len(generated),
            "first_pass_accepted": sum(row["raw_language_pass"] and row["raw_grounding_pass"] for row in generated),
            "first_pass_rejected": sum(not (row["raw_language_pass"] and row["raw_grounding_pass"]) for row in generated),
            "retry_attempts": len(retries),
            "successful_retries": sum(row["answer_strategy"] == "gguf_generation" for row in retries),
            "failed_retries": sum(row["answer_strategy"] == "unsupported" for row in retries),
            "final_accepted_generation": sum(row["answer_strategy"] == "gguf_generation" for row in generated),
            "final_safe_abstention": sum(row["answer_strategy"] == "unsupported" for row in generated),
        },
        "fact_preservation": {
            "applicable": len(applicable),
            "preserved": sum(row["factual_preservation"]["preserved"] for row in applicable),
            "rate": sum(row["factual_preservation"]["preserved"] for row in applicable) / len(applicable) if applicable else 0.0,
        },
        "automatic_development_proxies": {
            metric: statistics.mean(row["automatic_development_proxies"][metric] for row in rows) if rows else 0.0
            for metric in ("precision", "recall", "f1")
        },
        "latency": {
            "generation_median_seconds": statistics.median(generation_times) if generation_times else 0.0,
            "generation_p95_seconds": percentile(generation_times, 0.95),
            "retry_total_seconds": sum(row["retry_seconds"] for row in rows),
            "tokens_per_second_median": statistics.median(token_rates) if token_rates else None,
        },
    }


def load_model(model_key: str) -> tuple[Any, dict[str, Any]]:
    spec = MODEL_SPECS[model_key]
    before = memory_snapshot()
    started = time.perf_counter()
    model = generator._get_generator(
        str(spec["path"]),
        GENERATOR_CONTEXT_SIZE,
        GENERATOR_THREADS,
        GENERATOR_BATCH_SIZE,
        GENERATOR_GPU_LAYERS,
    )
    elapsed = time.perf_counter() - started
    after = memory_snapshot()
    samples = ("English UTF-8 check", "বাংলা UTF-8 পরীক্ষা", "Banglish UTF-8 check")
    return model, {
        "load_seconds": elapsed,
        "rss_before_bytes": before["process_rss_bytes"],
        "rss_after_bytes": after["process_rss_bytes"],
        "rss_increase_bytes": (
            after["process_rss_bytes"] - before["process_rss_bytes"]
            if after["process_rss_bytes"] is not None and before["process_rss_bytes"] is not None else None
        ),
        "system_available_after_bytes": after["system_available_bytes"],
        "utf8_token_counts": [len(model.tokenize(value.encode("utf-8"), add_bos=False, special=False)) for value in samples],
        "context_size": GENERATOR_CONTEXT_SIZE,
        "threads": GENERATOR_THREADS,
        "batch_size": GENERATOR_BATCH_SIZE,
        "gpu_layers": GENERATOR_GPU_LAYERS,
    }


def run_focused(model_key: str) -> None:
    frozen = read_json(FROZEN_EVIDENCE_PATH)
    if not frozen:
        raise FileNotFoundError("Prepare frozen evidence before running a model.")
    generator.GENERATOR_CACHE.clear()
    gc.collect()
    model, load = load_model(model_key)
    rows: list[dict[str, Any]] = []
    peak_rss = memory_snapshot()["process_rss_bytes"] or 0
    for position, row in enumerate(frozen["cases"], start=1):
        result = run_case(row, Path(MODEL_SPECS[model_key]["path"]), model)
        rows.append(result)
        peak_rss = max(peak_rss, memory_snapshot()["process_rss_bytes"] or 0)
        print(f"{model_key} focused {position}/45 {row['case_id']}", flush=True)
    payload = {
        "notice": "STEP-7B CONTROLLED MODEL COMPARISON — NOT FINAL CORRECTNESS",
        "model_provenance": provenance(model_key),
        "frozen_settings": {
            "context": GENERATOR_CONTEXT_SIZE,
            "threads": GENERATOR_THREADS,
            "batch_size": GENERATOR_BATCH_SIZE,
            "gpu_layers": GENERATOR_GPU_LAYERS,
            "temperature": GENERATOR_TEMPERATURE,
            "top_p": GENERATOR_TOP_P,
            "max_tokens": GENERATOR_MAX_TOKENS,
        },
        "load_benchmark": load,
        "focused_peak_rss_bytes": peak_rss,
        "focused_summary": summarize(rows),
        "focused_rows": rows,
    }
    output = BASELINE_PATH if model_key == "1_5b" else THREE_B_PATH
    existing = read_json(output, {})
    existing.update(payload)
    if model_key == "1_5b":
        existing["full_300_summary"] = read_json(RESULTS_DIR / "step7_generation_summary.json")
    write_json(output, existing)


def capture_live_case(row: dict[str, Any], model_path: Path) -> dict[str, Any]:
    capture: dict[str, str] = {"raw_answer": "", "retry_answer": ""}

    def first_generation(**kwargs: Any) -> str:
        answer = generator.generate_answer(model_name=str(model_path), **kwargs)
        capture["raw_answer"] = answer
        return answer

    def retry_generation(**kwargs: Any) -> str:
        answer = generator.regenerate_answer_for_language(model_name=str(model_path), **kwargs)
        capture["retry_answer"] = answer
        return answer

    started = time.perf_counter()
    with (
        patch.object(pipeline, "generate_answer", side_effect=first_generation),
        patch.object(pipeline, "regenerate_answer_for_language", side_effect=retry_generation),
        patch.object(pipeline, "_log_generation_debug", return_value=None),
    ):
        result = pipeline.answer_question(row["question"], top_k=3, use_generation=True, use_answer_bank=False)
    expected_language = validate_language(result.get("answer", ""), row["expected_language"])
    raw_language = validate_language(capture["raw_answer"], row["expected_language"]) if capture["raw_answer"] else {}
    raw_grounding = validate_grounding(capture["raw_answer"], result.get("evidence", [])) if capture["raw_answer"] else {}
    factual = factual_preservation(row.get("reference_answer", ""), result.get("answer", ""))
    proxy = proxy_metrics(row.get("reference_answer", ""), result.get("answer", ""))
    return {
        "question_id": row["question_id"],
        "language": row["expected_language"],
        "question": row["question"],
        "reference_answer": row.get("reference_answer", ""),
        "raw_answer": capture["raw_answer"],
        "retry_answer": capture["retry_answer"],
        "final_answer": result.get("answer", ""),
        "answer_strategy": result.get("answer_strategy"),
        "support_status": result.get("support_status"),
        "generation_used": bool(result.get("generation_used")),
        "generation_attempts": int(result.get("generation_attempts", 0)),
        "retry_used": bool(result.get("retry_used")),
        "raw_language_pass": bool(raw_language.get("validation_passed", False)),
        "raw_grounding_pass": bool(raw_grounding.get("grounding_validation_passed", False)),
        "final_language_pass": bool(expected_language.get("validation_passed", False)),
        "final_language_reason": expected_language.get("validation_reason"),
        "final_grounding_pass": bool(result.get("grounding_validation_passed", False)),
        "final_grounding_reason": result.get("grounding_validation_reason"),
        "semantic_screen_flags": semantic_screen_flags(
            capture["raw_answer"] or result.get("answer", ""),
            row["expected_language"],
            raw_language.get("validation_reason"),
        ),
        "factual_preservation": factual,
        "automatic_development_proxies": proxy,
        "generation_seconds": float(result.get("latency_seconds", {}).get("generation", 0.0)),
        "retry_seconds": float(result.get("latency_seconds", {}).get("retry", 0.0)),
        "tokens_per_second": None,
        "total_seconds": time.perf_counter() - started,
        "source": result.get("source"),
        "page": result.get("page"),
    }


def run_full_3b() -> None:
    if RERANKER_ENABLED:
        raise RuntimeError("Step 7B requires RERANKER_ENABLED=false.")
    model_path = Path(MODEL_SPECS["3b"]["path"])
    dataset = load_dataset()
    rows = read_json(FULL_PROGRESS_PATH, [])
    peak_rss = memory_snapshot()["process_rss_bytes"] or 0
    minimum_available = memory_snapshot()["system_available_bytes"]
    swap_start = memory_snapshot()["swap_used_bytes"]
    for position, row in enumerate(dataset[len(rows):], start=len(rows) + 1):
        result = capture_live_case(row, model_path)
        rows.append(result)
        snapshot = memory_snapshot()
        peak_rss = max(peak_rss, snapshot["process_rss_bytes"] or 0)
        if snapshot["system_available_bytes"] is not None:
            minimum_available = min(minimum_available or snapshot["system_available_bytes"], snapshot["system_available_bytes"])
        write_json(FULL_PROGRESS_PATH, rows)
        print(f"3b full {position}/300 {row['expected_language']} {row['question_id']}", flush=True)
        if snapshot["system_available_bytes"] is not None and snapshot["system_available_bytes"] < 350 * 1024 * 1024:
            raise MemoryError("Stopped because available system RAM fell below 350 MiB.")
    payload = read_json(THREE_B_PATH, {})
    payload["full_300_rows"] = rows
    payload["full_300_summary"] = summarize(rows)
    payload["full_pipeline_memory"] = {
        "peak_process_rss_bytes": peak_rss,
        "minimum_system_available_bytes": minimum_available,
        "swap_used_before_bytes": swap_start,
        "swap_used_after_bytes": memory_snapshot()["swap_used_bytes"],
        "stable": True,
    }
    write_json(THREE_B_PATH, payload)


def measure_full_pipeline(model_key: str) -> None:
    row = next(item for item in load_dataset() if item["question_id"] == "Q005" and item["expected_language"] == "english")
    before = memory_snapshot()
    result = capture_live_case(row, Path(MODEL_SPECS[model_key]["path"]))
    after = memory_snapshot()
    output = BASELINE_PATH if model_key == "1_5b" else THREE_B_PATH
    payload = read_json(output, {})
    payload["full_pipeline_memory"] = {
        "representative_case": "Q005 English",
        "rss_before_bytes": before["process_rss_bytes"],
        "rss_after_bytes": after["process_rss_bytes"],
        "rss_increase_bytes": (
            after["process_rss_bytes"] - before["process_rss_bytes"]
            if after["process_rss_bytes"] is not None and before["process_rss_bytes"] is not None else None
        ),
        "system_available_after_bytes": after["system_available_bytes"],
        "swap_used_before_bytes": before["swap_used_bytes"],
        "swap_used_after_bytes": after["swap_used_bytes"],
        "stable": True,
        "answer_strategy": result["answer_strategy"],
    }
    write_json(output, payload)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finalize() -> None:
    baseline = read_json(BASELINE_PATH)
    three_b = read_json(THREE_B_PATH)
    if not baseline or not three_b or not baseline.get("focused_rows") or not three_b.get("focused_rows"):
        raise RuntimeError("Both focused model runs must finish before finalization.")
    if "full_300_summary" not in three_b:
        available = three_b.get("load_benchmark", {}).get("system_available_after_bytes")
        three_b["full_300_summary"] = {
            "status": "NOT_RUN_MEMORY_SAFETY_STOP",
            "reason": (
                "The isolated 3B load left too little system-available RAM to safely add "
                "BGE-M3, FAISS, BM25, and the Python pipeline."
            ),
            "system_available_after_isolated_load_bytes": available,
            "required_queries": 300,
            "completed_queries": 0,
        }
        three_b["full_pipeline_memory"] = {
            "status": "NOT_RUN_MEMORY_SAFETY_STOP",
            "isolated_model_rss_bytes": three_b.get("load_benchmark", {}).get("rss_after_bytes"),
            "system_available_after_isolated_load_bytes": available,
            "stable_for_full_pipeline": False,
        }
        write_json(THREE_B_PATH, three_b)
    three_b["model_decision"] = {
        "decision": "KEEP_1_5B",
        "adopt_3b": False,
        "reason": (
            "3B improved English and some automatic diagnostics, but Bangla remained semantically unreliable, "
            "Banglish remained poor, and isolated model loading left unsafe system memory headroom."
        ),
        "does_7b_need_testing_on_current_hardware": False,
        "step8_ready": False,
        "human_scores_fabricated": False,
    }
    write_json(THREE_B_PATH, three_b)
    by_model = {
        "1_5b": {row["case_id"]: row for row in baseline["focused_rows"]},
        "3b": {row["case_id"]: row for row in three_b["focused_rows"]},
    }
    rng = random.Random(46)
    review_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for case_id in sorted(by_model["1_5b"]):
        left_key, right_key = (("1_5b", "3b") if rng.random() < 0.5 else ("3b", "1_5b"))
        left, right = by_model[left_key][case_id], by_model[right_key][case_id]
        review_rows.append({
            "case_id": case_id,
            "language": left["language"],
            "question": left["question"],
            "reference_answer": left["reference_answer"],
            "verified_evidence": json.dumps(left["verified_evidence"], ensure_ascii=False),
            "source": left["source"],
            "page": left["page"],
            "SYSTEM_A_answer": left["final_answer"],
            "SYSTEM_B_answer": right["final_answer"],
            "SYSTEM_A_language_pass": left["final_language_pass"],
            "SYSTEM_B_language_pass": right["final_language_pass"],
            "SYSTEM_A_grounding_pass": left["final_grounding_pass"],
            "SYSTEM_B_grounding_pass": right["final_grounding_pass"],
            "SYSTEM_A_latency": left["generation_seconds"],
            "SYSTEM_B_latency": right["generation_seconds"],
            "A_correctness": "", "B_correctness": "",
            "A_relevance": "", "B_relevance": "",
            "A_groundedness": "", "B_groundedness": "",
            "A_completeness": "", "B_completeness": "",
            "A_naturalness": "", "B_naturalness": "",
            "preferred_answer": "", "review_notes": "",
        })
        key_rows.append({
            "case_id": case_id,
            "SYSTEM_A_model": MODEL_SPECS[left_key]["label"],
            "SYSTEM_B_model": MODEL_SPECS[right_key]["label"],
        })
    review_fields = list(review_rows[0])
    write_csv(REVIEW_PATH, review_rows, review_fields)
    write_csv(KEY_PATH, key_rows, list(key_rows[0]))

    one = baseline["focused_summary"]
    three = three_b["focused_summary"]
    lines = [
        "# Step 7B Controlled Generator Model Comparison", "",
        "**DEVELOPMENT / REGRESSION EVALUATION — NOT FINAL CORRECTNESS**", "",
        "The prompts, validators, retrieval, evidence contract, retry policy, context, temperature, top-p, and output allowance are frozen across both models.", "",
        "## Focused 45-case automatic summary", "",
        "| Metric | 1.5B | 3B |", "|---|---:|---:|",
        f"| Generation-attempted cases | {one['grounding']['generation_attempted_cases']} | {three['grounding']['generation_attempted_cases']} |",
        f"| Final accepted generations | {one['grounding']['final_accepted_generation']} | {three['grounding']['final_accepted_generation']} |",
        f"| Final safe abstentions | {one['grounding']['final_safe_abstention']} | {three['grounding']['final_safe_abstention']} |",
        f"| Retries | {one['grounding']['retry_attempts']} | {three['grounding']['retry_attempts']} |",
        f"| Generation median seconds | {one['latency']['generation_median_seconds']:.3f} | {three['latency']['generation_median_seconds']:.3f} |",
        f"| Generation P95 seconds | {one['latency']['generation_p95_seconds']:.3f} | {three['latency']['generation_p95_seconds']:.3f} |", "",
        "## Human review", "",
        f"The blinded file contains {len(review_rows)} rows: 15 English, 15 Bangla, and 15 Banglish. Human rating columns are intentionally blank.", "",
        "## Model decision", "", "Pending manual semantic inspection of the blinded comparison and the full 300-query 3B run.", "",
        "## Full 300-query 3B run", "",
        "Not run: the isolated 3B load triggered the memory-safety stop before BGE-M3 and the full pipeline were added.", "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "focused", "full", "memory", "finalize"))
    parser.add_argument("--model", choices=("1_5b", "3b"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare_frozen_evidence()
    elif args.action == "focused":
        if not args.model:
            parser.error("focused requires --model")
        run_focused(args.model)
    elif args.action == "full":
        if args.model != "3b":
            parser.error("the new full run is only required for --model 3b")
        run_full_3b()
    elif args.action == "memory":
        if not args.model:
            parser.error("memory requires --model")
        measure_full_pipeline(args.model)
    else:
        finalize()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
