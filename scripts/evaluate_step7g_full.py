"""Checkpointed, offline Step 7G 300-variant development measurement.

Question pairing, references, expected pages, and prior failure labels are
evaluation-only inputs; they never enter answer_question.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RERANKER_ENABLED
from src.evaluator import load_dataset
from src.evidence import FIELD_PATTERNS, analyze_query, assess_evidence
from src.generator import get_generation_telemetry, reset_generation_telemetry
from src.pipeline import answer_question

RESULTS = ROOT / "results"
RESULT_CSV = RESULTS / "step7g_full_300_results.csv"
SUMMARY_JSON = RESULTS / "step7g_full_300_summary.json"
REPORT_MD = RESULTS / "step7g_full_300_report.md"
REVIEW_CSV = RESULTS / "step7g_full_manual_review.csv"
LANGUAGES = ("english", "bangla", "banglish")
METRIC_NAMES = ("page_hit_1", "page_hit_3", "mrr_3", "entity_hit_1", "entity_hit_3",
                "field_hit_1", "field_hit_3", "supportable_1", "supportable_3")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(10):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))


def truth(value: object) -> bool:
    return value is True or str(value).casefold() == "true"


def run_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in ("data/questions/questions.csv", "vector_db/index_manifest.json",
                     "src/pipeline.py", "src/relations.py", "src/evidence.py",
                     "src/chunker.py", "src/config.py", "src/crosslingual.py"):
        digest.update(relative.encode())
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()[:20]


def measures(question: str, candidates: list[dict], expected_page: str) -> dict[str, object]:
    """Step 7F metric definitions, applied to current final candidate ordering."""
    first = candidates[:3]
    request = analyze_query(question)
    pages = [str(x.get("page")) == expected_page for x in first]
    entity = request.primary_entity
    if entity:
        compact = re.sub(r"[^a-z0-9]", "", entity.casefold())
        entity_hits = [compact in re.sub(r"[^a-z0-9]", "", str(x.get("text", "")).casefold())
                       for x in first]
    else:
        entity_hits = []
    patterns = FIELD_PATTERNS.get(request.requested_field, ()) if request.requested_field != "general" else ()
    field_hits = [any(re.search(pattern, str(x.get("text", "")), re.I) for pattern in patterns)
                  or request.requested_field in (x.get("field_types") or []) for x in first] if patterns else []
    return {
        "page_hit_1": bool(pages and pages[0]), "page_hit_3": any(pages),
        "mrr_3": next((1 / rank for rank, hit in enumerate(pages, 1) if hit), 0.0),
        "entity_applicable": bool(entity),
        "entity_hit_1": bool(entity_hits and entity_hits[0]), "entity_hit_3": any(entity_hits),
        "field_applicable": bool(patterns),
        "field_hit_1": bool(field_hits and field_hits[0]), "field_hit_3": any(field_hits),
        "supportable_1": assess_evidence(question, first[:1]).status.value == "supported",
        "supportable_3": assess_evidence(question, first).status.value == "supported",
    }


def evaluate(row: dict, expected_page: str, fingerprint: str) -> dict:
    started = time.perf_counter()
    reset_generation_telemetry()
    answer = {}
    error = ""
    try:
        answer = answer_question(row["question"], top_k=3, use_generation=True, use_answer_bank=False)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    process = psutil.Process().memory_info()
    telemetry = get_generation_telemetry()
    initial = list(answer.get("initial_top3") or [])
    final = list(answer.get("retrieved_context") or [])
    initial_metrics = measures(row["question"], initial, expected_page)
    current_metrics = measures(row["question"], final, expected_page)
    relation = answer.get("relation_extracted") or {}
    latency = answer.get("latency_seconds") or {}
    return {
        "run_fingerprint": fingerprint,
        "question_id": row["question_id"], "language": row["expected_language"],
        "question": row["question"], "reference": row["reference_answer"],
        "expected_page_offline": expected_page,
        "final_answer": answer.get("answer", ""), "final_status": answer.get("final_status", "RUNTIME_ERROR"),
        "answer_strategy": answer.get("answer_strategy", ""), "requested_answer_strategy": answer.get("requested_answer_strategy", ""),
        "relation_type": answer.get("relation_type") or "", "relation_values": json.dumps(relation.get("values", []), ensure_ascii=False),
        "support_status": answer.get("support_status", ""),
        "initial_support_status": answer.get("initial_support_status", ""),
        "merged_support_status": answer.get("merged_support_status", ""),
        "fallback_triggered": bool(answer.get("fallback_triggered")),
        "metadata_evidence_widened": bool(answer.get("metadata_evidence_widened")),
        "source": answer.get("source", ""), "page": answer.get("page", ""),
        "chunk_id": answer.get("chunk_id", ""), "supporting_excerpt": answer.get("supporting_excerpt", ""),
        "language_validation_passed": bool(answer.get("language_validation_passed")),
        "language_validation_reason": answer.get("language_validation_reason", ""),
        "grounding_validation_passed": bool(answer.get("grounding_validation_passed")),
        "grounding_validation_reason": answer.get("grounding_validation_reason", ""),
        "generation_used": bool(answer.get("generation_used")),
        "generation_attempts": int(answer.get("generation_attempts", 0)),
        "retry_used": bool(answer.get("retry_used")),
        "canonical_validation_passed": bool(answer.get("canonical_validation_passed")),
        "canonical_grounding_passed": bool(answer.get("canonical_grounding_passed")),
        "realization_validation_passed": bool(answer.get("realization_validation_passed")),
        "generation_rejection_reason": answer.get("generation_rejection_reason", ""),
        "generation_error": answer.get("generation_error", ""),
        "prompt_tokens": sum(int(x.get("prompt_tokens", 0)) for x in telemetry),
        "completion_tokens": sum(int(x.get("completion_tokens", 0)) for x in telemetry),
        "retrieval_seconds": latency.get("retrieval", 0), "generation_seconds": latency.get("generation", 0),
        "retry_seconds": latency.get("retry", 0), "total_seconds": elapsed,
        "rss_bytes": process.rss, "peak_process_bytes": getattr(process, "peak_wset", process.rss),
        "available_ram_bytes": psutil.virtual_memory().available,
        "pagefile_used_bytes_system": psutil.swap_memory().used,
        "runtime_error": error, "answer_bank_enabled": False, "reranker_enabled": False,
        **{f"initial_{name}": value for name, value in initial_metrics.items()},
        **current_metrics,
    }


def run() -> None:
    if RERANKER_ENABLED:
        raise RuntimeError("Measurement requires RERANKER_ENABLED=false")
    dataset = load_dataset()
    keys = [(x["question_id"], x["expected_language"]) for x in dataset]
    if len(dataset) != 300 or len(set(keys)) != 300 or Counter(x["expected_language"] for x in dataset) != Counter({x: 100 for x in LANGUAGES}):
        raise RuntimeError("Expected exactly 100 unique questions per language")
    expected = {x["question_id"]: x["expected_page"] for x in read_csv(ROOT / "data/questions/questions.csv")}
    fingerprint = run_fingerprint()
    existing = read_csv(RESULT_CSV)
    if existing and any(x["run_fingerprint"] != fingerprint for x in existing):
        raise RuntimeError("Checkpoint fingerprint differs from current dataset/index/code; preserve or remove it explicitly before a new run")
    completed = {(x["question_id"], x["language"]): x for x in existing}
    for index, row in enumerate(dataset, 1):
        key = (row["question_id"], row["expected_language"])
        if key in completed:
            continue
        completed[key] = evaluate(row, expected.get(row["question_id"], ""), fingerprint)
        ordered = [completed[(x["question_id"], x["expected_language"])] for x in dataset
                   if (x["question_id"], x["expected_language"]) in completed]
        write_csv(RESULT_CSV, ordered)
        result = completed[key]
        print(f"{len(ordered)}/300 {key[1]} {key[0]} {result['final_status']} {result['answer_strategy']} {result['total_seconds']:.1f}s", flush=True)
    print(f"COMPLETE {len(completed)}/300 fingerprint={fingerprint}", flush=True)


def aggregate_metrics(rows: list[dict]) -> dict:
    result = {}
    for language in (*LANGUAGES, "overall"):
        subset = rows if language == "overall" else [x for x in rows if x["language"] == language]
        result[language] = {}
        for name in METRIC_NAMES:
            applicable = ([x for x in subset if truth(x["entity_applicable"])] if name.startswith("entity_") else
                          [x for x in subset if truth(x["field_applicable"])] if name.startswith("field_") else subset)
            total = sum(float(x[name]) if name == "mrr_3" else truth(x[name]) for x in applicable)
            result[language][name] = {"value": total / len(applicable) if applicable else None,
                                      "numerator": total, "denominator": len(applicable)}
    return result


def parity(rows: list[dict]) -> dict:
    by_id: dict[str, dict[str, bool]] = {}
    for row in rows:
        by_id.setdefault(row["question_id"], {})[row["language"]] = row["support_status"] == "supported"
    categories = Counter()
    for values in by_id.values():
        en, ba, bl = (values.get(lang, False) for lang in LANGUAGES)
        if en and ba and bl:
            label = "ALL_THREE_SUPPORTED"
        elif en and not ba and not bl:
            label = "ENGLISH_ONLY"
        elif en and ba and not bl:
            label = "ENGLISH_AND_BANGLA"
        elif en and bl and not ba:
            label = "ENGLISH_AND_BANGLISH"
        elif not any((en, ba, bl)):
            label = "NONE_SUPPORTED"
        else:
            label = "OTHER_MISMATCH"
        categories[label] += 1
    return dict(categories)


def select_review(rows: list[dict]) -> list[dict]:
    old_failures = {(x["question_id"], x["language"]): x.get("primary_stage", "")
                    for x in read_csv(RESULTS / "step7f_failure_funnel.csv")}
    selected = []
    for language in LANGUAGES:
        candidates = [x for x in rows if x["language"] == language]
        def score(x: dict) -> tuple[int, str]:
            value = (5 if x["answer_strategy"] == "semi_structured_relation" else
                     4 if x["answer_strategy"] == "gguf_generation" else 0)
            value += 3 if old_failures.get((x["question_id"], language)) else 0
            value += 3 if truth(x["fallback_triggered"]) and x["support_status"] == "supported" else 0
            value += 2 if x["final_status"] == "GENERATION_REJECTED" else 0
            return (-value, x["question_id"])
        chosen: list[dict] = []
        used: set[str] = set()

        def add(pool: list[dict], limit: int) -> None:
            for candidate in sorted(pool, key=score):
                if candidate["question_id"] not in used:
                    chosen.append(candidate)
                    used.add(candidate["question_id"])
                if len(chosen) >= limit:
                    break

        add([x for x in candidates if x["answer_strategy"] == "gguf_generation"], 2)
        add([x for x in candidates if x["answer_strategy"] == "semi_structured_relation"], 5)
        add([x for x in candidates if x["final_status"] == "ANSWER_RETURNED"
             and not truth(x["grounding_validation_passed"])], 7)
        add([x for x in candidates if old_failures.get((x["question_id"], language))
             or (truth(x["fallback_triggered"]) and x["support_status"] == "supported")], 9)
        add(candidates, 10)
        for row in chosen:
            selected.append({
                "question_id": row["question_id"], "language": language,
                "question": row["question"], "reference": row["reference"],
                "final_answer": row["final_answer"], "final_status": row["final_status"],
                "answer_strategy": row["answer_strategy"], "relation_type": row["relation_type"],
                "previous_failure_category": old_failures.get((row["question_id"], language), ""),
                "crosslingual_recovered": truth(row["fallback_triggered"]) and row["support_status"] == "supported",
                "source": row["source"], "page": row["page"], "chunk_id": row["chunk_id"],
                "supporting_evidence": row["supporting_excerpt"],
                "language_validation_passed": row["language_validation_passed"],
                "grounding_validation_passed": row["grounding_validation_passed"],
                "correctness": "", "relevance": "", "groundedness": "", "completeness": "",
                "naturalness": "", "semantic_consistency": "", "overall_acceptability": "", "review_notes": "",
            })
    return selected


def finalize() -> None:
    rows = read_csv(RESULT_CSV)
    fingerprint = run_fingerprint()
    if len(rows) != 300 or len({(x["question_id"], x["language"]) for x in rows}) != 300:
        raise RuntimeError(f"Full run incomplete: {len(rows)}/300 rows")
    if any(x["run_fingerprint"] != fingerprint for x in rows):
        raise RuntimeError("Result fingerprint differs from current code/index/dataset")
    per_language = {}
    for language in LANGUAGES:
        subset = [x for x in rows if x["language"] == language]
        generated = [x for x in subset if truth(x["generation_used"])]
        retries = [x for x in subset if truth(x["retry_used"])]
        returned = [x for x in subset if x["final_status"] == "ANSWER_RETURNED"]
        per_language[language] = {
            "rows": len(subset), "answers_returned": len(returned),
            "strategies": dict(Counter(x["answer_strategy"] for x in subset)),
            "statuses": dict(Counter(x["final_status"] for x in subset)),
            "language_consistency": {"passed": sum(truth(x["language_validation_passed"]) for x in subset), "denominator": len(subset)},
            "grounding_validation": {"passed": sum(truth(x["grounding_validation_passed"]) for x in returned), "denominator": len(returned)},
            "generation": {"rows": len(generated), "attempts": sum(int(x["generation_attempts"]) for x in generated),
                           "accepted": sum(x["final_status"] == "ANSWER_RETURNED" for x in generated),
                           "rejected": sum(x["final_status"] == "GENERATION_REJECTED" for x in generated),
                           "retries": len(retries), "successful_retries": sum(x["final_status"] == "ANSWER_RETURNED" for x in retries),
                           "failed_retries": sum(x["final_status"] != "ANSWER_RETURNED" for x in retries)},
            "runtime_errors": sum(bool(x["runtime_error"]) for x in subset),
        }
    metrics = aggregate_metrics(rows)
    old_metrics = read_csv(RESULTS / "step7f_retrieval_metrics.csv")
    comparisons = {}
    for language in (*LANGUAGES, "overall"):
        subset = old_metrics if language == "overall" else [x for x in old_metrics if x["language"] == language]
        comparisons[language] = {}
        for name in METRIC_NAMES:
            applicable = ([x for x in subset if truth(x["after_entity_applicable"])] if name.startswith("entity_") else
                          [x for x in subset if truth(x["after_field_applicable"])] if name.startswith("field_") else subset)
            comparisons[language][name] = (sum(float(x["after_" + name]) if name == "mrr_3" else truth(x["after_" + name])
                                               for x in applicable) / len(applicable) if applicable else None)
    times = [float(x["total_seconds"]) for x in rows]
    summary = {
        "notice": "STEP 7G DEVELOPMENT EVALUATION; NOT FINAL THESIS ACCURACY",
        "completion": {"rows": len(rows), "expected": 300, "complete": True},
        "run_fingerprint": fingerprint, "index_chunks": 491, "answer_bank_enabled": False, "reranker_enabled": False,
        "per_language": per_language, "retrieval_metrics": metrics,
        "frozen_step7f_retrieval_comparison": comparisons,
        "multilingual_parity": parity(rows),
        "latency_seconds": {"mean": statistics.mean(times), "median": statistics.median(times),
                            "p95": sorted(times)[int(0.95 * (len(times) - 1))], "maximum": max(times), "total": sum(times)},
        "memory": {"peak_process_rss_bytes": max(int(x["rss_bytes"]) for x in rows),
                   "peak_process_working_set_bytes": max(int(x["peak_process_bytes"]) for x in rows),
                   "minimum_available_ram_bytes": min(int(x["available_ram_bytes"]) for x in rows),
                   "maximum_system_pagefile_used_bytes": max(int(x["pagefile_used_bytes_system"]) for x in rows)},
        "runtime_errors": [x for x in rows if x["runtime_error"]],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    review = select_review(rows)
    if Counter(x["language"] for x in review) != Counter({x: 10 for x in LANGUAGES}):
        raise RuntimeError("Manual review sample must contain ten cases per language")
    write_csv(REVIEW_CSV, review)
    print(json.dumps({"rows": len(rows), "review_rows": len(review),
                      "statuses": {x: per_language[x]["statuses"] for x in LANGUAGES}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "finalize"))
    action = parser.parse_args().action
    run() if action == "run" else finalize()
