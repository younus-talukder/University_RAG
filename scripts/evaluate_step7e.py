from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import psutil

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluate_step7c import factual_tokens, load_dataset, proxy_metrics
from evaluate_step7d import _review_dataset
from src.config import RERANKER_ENABLED
from src.generator import get_generation_telemetry, reset_generation_telemetry
from src.language_validator import validate_language
from src.pipeline import answer_question


RESULTS = ROOT_DIR / "results"
FAILURE_TRACE_CSV = RESULTS / "step7e_failure_trace.csv"
BEFORE_REVIEW_CSV = RESULTS / "step7d_review_run.csv"
BEFORE_AFTER_CSV = RESULTS / "step7e_before_after_review.csv"
MANUAL_CSV = RESULTS / "step7e_manual_review_sample.csv"
FULL_CSV = RESULTS / "step7e_300_results.csv"
SUMMARY_JSON = RESULTS / "step7e_summary.json"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(10):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.25 * (attempt + 1))


def _compact_chunks(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": position,
            "chunk_id": item.get("chunk_id"),
            "source": item.get("relative_path") or item.get("source"),
            "page": item.get("page"),
            "score": item.get("score"),
            "text": item.get("text") or item.get("supporting_excerpt"),
        }
        for position, item in enumerate(items[:3], start=1)
    ]


def _failure_classification(result: dict[str, Any]) -> str:
    final_strategy = str(result.get("answer_strategy") or "")
    if final_strategy not in {"unsupported", "ambiguous", "conflicting"}:
        return ""
    pre_status = str(result.get("pre_generation_support_status") or result.get("support_status") or "")
    reason = str(result.get("generation_rejection_reason") or "")
    support_reason = str(result.get("pre_generation_support_reason") or "").casefold()
    if pre_status != "supported":
        if not result.get("retrieved_context"):
            return "RETRIEVAL_FAILED"
        requested_field = str(result.get("requested_field") or "general")
        if pre_status == "ambiguous":
            return "ENTITY_VALIDATION_FAILED"
        if requested_field != "general" or "field" in support_reason:
            return "FIELD_VALIDATION_FAILED"
        return "NO_EVIDENCE"
    if result.get("retry_used"):
        return "RETRY_FAILED"
    if reason == "WRONG_POLARITY":
        return "GENERATION_POLARITY_FAILED"
    if reason in {
        "BANGLA_BODY_TOO_ENGLISH", "BANGLISH_BODY_PURE_ENGLISH",
        "BANGLISH_CONTAINS_BENGALI_SCRIPT", "ENGLISH_CONTAINS_BENGALI_SENTENCE",
        "ENGLISH_CONTAINS_BANGLISH_FRAMING", "UNEXPECTED_FOREIGN_SCRIPT", "MIXED_SCRIPT_TOKEN",
    }:
        return "GENERATION_LANGUAGE_FAILED"
    if reason in {"WRONG_RELATION_VALUE", "MISSING_REQUIRED_ENTITY", "SEMANTIC_CONTRACT_UNCERTAIN"}:
        return "GENERATION_SEMANTIC_FAILED"
    if reason:
        return "GENERATION_GROUNDING_FAILED"
    return "OTHER"


def reclassify_trace() -> list[dict[str, Any]]:
    rows = _read_csv(FAILURE_TRACE_CSV)
    for row in rows:
        language = str(row.get("language") or "")
        final_status = str(row.get("final_status") or "")
        if language not in {"bangla", "banglish"} or final_status not in {
            "insufficient", "ambiguous", "conflicting", "GENERATION_REJECTED"
        }:
            row["failure_classification"] = ""
            continue
        pre_status = str(row.get("support_status_before_generation") or "")
        if pre_status != "supported":
            if not row.get("retrieved_top_3_chunks") or row.get("retrieved_top_3_chunks") == "[]":
                label = "RETRIEVAL_FAILED"
            elif pre_status == "ambiguous":
                label = "ENTITY_VALIDATION_FAILED"
            elif str(row.get("requested_field") or "general") != "general":
                label = "FIELD_VALIDATION_FAILED"
            else:
                label = "NO_EVIDENCE"
        elif str(row.get("raw_retry_output") or ""):
            label = "RETRY_FAILED"
        else:
            reason = str(row.get("final_fallback_reason") or "")
            if reason == "WRONG_POLARITY":
                label = "GENERATION_POLARITY_FAILED"
            elif "LANGUAGE" in reason or reason.startswith(("BANGLA_", "BANGLISH_", "ENGLISH_")):
                label = "GENERATION_LANGUAGE_FAILED"
            elif reason in {"WRONG_RELATION_VALUE", "MISSING_REQUIRED_ENTITY", "SEMANTIC_CONTRACT_UNCERTAIN"}:
                label = "GENERATION_SEMANTIC_FAILED"
            elif reason:
                label = "GENERATION_GROUNDING_FAILED"
            else:
                label = "OTHER"
        row["failure_classification"] = label
    _write_csv(FAILURE_TRACE_CSV, rows)
    return rows


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("question_id")), str(row.get("language") or row.get("expected_language"))


def _truth(value: Any) -> bool:
    return value is True or str(value).casefold() == "true"


def _evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    reset_generation_telemetry()
    error = ""
    try:
        result = answer_question(row["question"], top_k=3, use_generation=True, use_answer_bank=False)
    except Exception as exc:
        result = {}
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    telemetry = get_generation_telemetry()
    prompt_tokens = sum(int(item.get("prompt_tokens", 0)) for item in telemetry)
    completion_tokens = sum(int(item.get("completion_tokens", 0)) for item in telemetry)
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    expected_validation = validate_language(result.get("answer", ""), row.get("expected_language", "english"))
    proxy = proxy_metrics(row.get("reference_answer", ""), result.get("answer", ""))
    required = factual_tokens(row.get("reference_answer", ""))
    actual = factual_tokens(result.get("answer", ""))
    latency = result.get("latency_seconds") or {}
    evidence_available = str(result.get("pre_generation_support_status") or result.get("support_status")) == "supported"
    canonical_successful = bool(result.get("canonical_validation_passed") and result.get("canonical_grounding_passed"))
    return {
        "question_id": row.get("question_id"), "language": row.get("expected_language"),
        "question": row.get("question"), "reference": row.get("reference_answer"),
        "final_answer": result.get("answer"), "final_status": result.get("final_status"),
        "answer_strategy": result.get("answer_strategy"), "generation_used": bool(result.get("generation_used")),
        "generation_attempts": int(result.get("generation_attempts", 0)), "retry_used": bool(result.get("retry_used")),
        "support_status": result.get("support_status"),
        "pre_generation_support_status": result.get("pre_generation_support_status") or result.get("support_status"),
        "evidence_available": evidence_available, "target_language": result.get("target_language"),
        "canonical_answer": result.get("canonical_answer") or "", "canonical_successful": canonical_successful,
        "canonical_validation_reason": result.get("canonical_validation_reason") or "",
        "target_language_realization": result.get("target_language_realization") or "",
        "realization_successful": bool(result.get("realization_validation_passed")),
        "realization_validation_reason": result.get("realization_validation_reason") or "",
        "language_validation_passed": bool(expected_validation.get("validation_passed")),
        "language_validation_reason": expected_validation.get("validation_reason"),
        "runtime_language_validation_passed": bool(result.get("language_validation_passed")),
        "grounding_validation_passed": bool(result.get("grounding_validation_passed")),
        "grounding_validation_reason": result.get("grounding_validation_reason"),
        "generation_rejection_reason": result.get("generation_rejection_reason") or "",
        "fact_tokens_required": sorted(required), "fact_tokens_preserved": bool(required and required <= actual),
        "fact_preservation_applicable": bool(required), "automatic_precision": proxy["precision"],
        "automatic_recall": proxy["recall"], "automatic_f1": proxy["f1"],
        "source": result.get("source"), "page": result.get("page"),
        "supporting_excerpt": result.get("supporting_excerpt"),
        "retrieval_seconds": latency.get("retrieval", 0.0), "answering_seconds": latency.get("answering", 0.0),
        "canonical_generation_seconds": latency.get("canonical_generation", 0.0),
        "language_realization_seconds": latency.get("language_realization", 0.0),
        "generation_seconds": latency.get("generation", 0.0), "retry_seconds": latency.get("retry", 0.0),
        "total_seconds": elapsed, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "rss_bytes": psutil.Process().memory_info().rss, "available_ram_bytes": memory.available,
        "swap_used_bytes": swap.used, "error": error or result.get("generation_error", "") or "",
        "answer_bank_enabled": False, "reranker_enabled": False,
    }


def run_evaluation(selected: list[dict[str, Any]], path: Path, resume: bool) -> list[dict[str, Any]]:
    if RERANKER_ENABLED:
        raise RuntimeError("Step 7E requires RERANKER_ENABLED=false.")
    existing = [row for row in _read_csv(path) if not row.get("error")] if resume else []
    completed = {_key(row) for row in existing}
    pending = [row for row in selected if _key(row) not in completed]
    output: list[dict[str, Any]] = list(existing)
    for position, row in enumerate(pending, start=1):
        output.append(_evaluate_row(row))
        print(f"answered {len(existing) + position}/{len(existing) + len(pending)} {row.get('expected_language')} {row.get('question_id')}", flush=True)
        if (len(existing) + position) % 5 == 0 or position == len(pending):
            _write_csv(path, output)
    return output


def _write_review_artifacts(after: list[dict[str, Any]]) -> None:
    before = {_key(row): row for row in _read_csv(BEFORE_REVIEW_CSV)}
    paired: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    for row in after:
        old = before.get(_key(row), {})
        paired.append({
            "question_id": row.get("question_id"), "language": row.get("language"),
            "question": row.get("question"), "before_step7e_answer": old.get("final_answer", ""),
            "after_step7e_answer": row.get("final_answer", ""), "canonical_answer": row.get("canonical_answer", ""),
            "target_language_realization": row.get("target_language_realization", ""),
            "final_validation": _json({
                "language": _truth(row.get("language_validation_passed")),
                "grounding": _truth(row.get("grounding_validation_passed")),
                "canonical": _truth(row.get("canonical_successful")),
                "realization": _truth(row.get("realization_successful")),
            }),
            "final_status": row.get("final_status"), "support_status": row.get("support_status"),
            "generation_rejection_reason": row.get("generation_rejection_reason", ""),
        })
        manual.append({
            "language": row.get("language"), "question": row.get("question"), "reference": row.get("reference"),
            "supporting_evidence": row.get("supporting_excerpt"), "canonical_answer": row.get("canonical_answer", ""),
            "final_answer": row.get("final_answer"), "status": row.get("final_status"),
            "correctness": "", "relevance": "", "groundedness": "", "completeness": "",
            "naturalness": "", "semantic_consistency": "", "overall_acceptability": "", "review_notes": "",
        })
    _write_csv(BEFORE_AFTER_CSV, paired)
    _write_csv(MANUAL_CSV, manual)


def summarize_step7e(rows: list[dict[str, Any]]) -> dict[str, Any]:
    languages: dict[str, Any] = {}
    for language in ("english", "bangla", "banglish"):
        selected = [row for row in rows if row.get("language") == language]
        generated = [row for row in selected if _truth(row.get("generation_used"))]
        languages[language] = {
            "rows": len(selected), "structured_answers": sum(str(row.get("answer_strategy", "")).startswith("structured") for row in selected),
            "generation_required": len(generated), "evidence_available": sum(_truth(row.get("evidence_available")) for row in generated),
            "canonical_successful": sum(_truth(row.get("canonical_successful")) for row in generated),
            "realization_successful": sum(_truth(row.get("realization_successful")) for row in generated),
            "final_answer_returned": sum(row.get("final_status") == "ANSWER_RETURNED" for row in generated),
            "generation_rejected": sum(row.get("final_status") == "GENERATION_REJECTED" for row in selected),
            "insufficient_evidence": sum(row.get("final_status") == "INSUFFICIENT_EVIDENCE" for row in selected),
        }
    canonical = [float(row.get("canonical_generation_seconds") or 0) for row in rows if float(row.get("canonical_generation_seconds") or 0)]
    bangla = [float(row.get("language_realization_seconds") or 0) for row in rows if row.get("language") == "bangla" and float(row.get("language_realization_seconds") or 0)]
    banglish = [float(row.get("language_realization_seconds") or 0) for row in rows if row.get("language") == "banglish" and float(row.get("language_realization_seconds") or 0)]
    total = [float(row.get("generation_seconds") or 0) for row in rows if float(row.get("generation_seconds") or 0)]
    return {
        "notice": "STEP-7E DEVELOPMENT / REGRESSION EVALUATION — NOT FINAL THESIS ACCURACY",
        "queries": len(rows), "answer_bank_enabled": False, "reranker_enabled": False,
        "status_counts": dict(Counter(str(row.get("final_status")) for row in rows)), "languages": languages,
        "latency_seconds_median": {
            "canonical": statistics.median(canonical) if canonical else 0.0,
            "bangla_realization": statistics.median(bangla) if bangla else 0.0,
            "banglish_realization": statistics.median(banglish) if banglish else 0.0,
            "total_generation": statistics.median(total) if total else 0.0,
        },
        "memory": {
            "peak_rss_bytes": max((int(float(row.get("rss_bytes") or 0)) for row in rows), default=0),
            "minimum_available_ram_bytes": min((int(float(row.get("available_ram_bytes") or 0)) for row in rows), default=0),
            "peak_swap_used_bytes": max((int(float(row.get("swap_used_bytes") or 0)) for row in rows), default=0),
        },
        "errors": sum(bool(row.get("error")) for row in rows),
    }


def _trace_row(source: dict[str, Any], result: dict[str, Any], error: str, elapsed: float) -> dict[str, Any]:
    first_language = result.get("first_language_validation") or {}
    first_semantic = result.get("first_semantic_validation") or {}
    first_grounding = result.get("first_grounding_validation") or {}
    retry_language = result.get("retry_language_validation") or {}
    retry_semantic = result.get("retry_semantic_validation") or {}
    retry_grounding = result.get("retry_grounding_validation") or {}
    evidence = result.get("pre_generation_evidence") or result.get("evidence") or []
    return {
        "question_id": source.get("question_id"),
        "language": source.get("expected_language"),
        "question": source.get("question"),
        "requested_entity": result.get("requested_entity"),
        "requested_field": result.get("requested_field"),
        "retrieved_top_3_chunks": _json(_compact_chunks(list(result.get("retrieved_context") or []))),
        "supporting_evidence": _json(evidence),
        "support_status_before_generation": result.get("pre_generation_support_status") or result.get("support_status"),
        "support_reason_before_generation": result.get("pre_generation_support_reason") or result.get("support_reason"),
        "answer_strategy_requested": result.get("requested_answer_strategy") or result.get("answer_strategy"),
        "raw_first_generation": result.get("raw_first_generation") or "",
        "first_language_validation_passed": first_language.get("validation_passed", ""),
        "first_language_validation_reason": first_language.get("validation_reason", ""),
        "first_semantic_validation_passed": first_semantic.get("passed", ""),
        "first_semantic_validation_reason": first_semantic.get("reason", ""),
        "first_grounding_validation_passed": first_grounding.get("grounding_validation_passed", ""),
        "first_grounding_validation_reason": first_grounding.get("grounding_validation_reason", ""),
        "retry_reason": result.get("retry_reason") or "",
        "raw_retry_output": result.get("raw_retry_output") or "",
        "retry_language_validation_passed": retry_language.get("validation_passed", ""),
        "retry_language_validation_reason": retry_language.get("validation_reason", ""),
        "retry_semantic_validation_passed": retry_semantic.get("passed", ""),
        "retry_semantic_validation_reason": retry_semantic.get("reason", ""),
        "retry_grounding_validation_passed": retry_grounding.get("grounding_validation_passed", ""),
        "retry_grounding_validation_reason": retry_grounding.get("grounding_validation_reason", ""),
        "final_answer": result.get("answer") or "",
        "final_status": "GENERATION_REJECTED" if result.get("final_fallback_reason") == "GENERATION_REJECTED" else result.get("support_status"),
        "final_fallback_reason": result.get("generation_rejection_reason") or result.get("final_fallback_reason") or "",
        "failure_classification": _failure_classification(result),
        "elapsed_seconds": elapsed,
        "error": error or result.get("generation_error") or "",
    }


def run_trace(resume: bool = False) -> list[dict[str, Any]]:
    selected = _review_dataset(load_dataset())
    existing = _read_csv(FAILURE_TRACE_CSV) if resume else []
    completed = {(row.get("question_id"), row.get("language")) for row in existing}
    pending = [
        row for row in selected
        if (str(row.get("question_id")), str(row.get("expected_language"))) not in completed
    ]
    output: list[dict[str, Any]] = list(existing)
    for position, row in enumerate(pending, start=1):
        started = time.perf_counter()
        error = ""
        try:
            result = answer_question(row["question"], top_k=3, use_generation=True, use_answer_bank=False)
        except Exception as exc:
            result = {}
            error = f"{type(exc).__name__}: {exc}"
        output.append(_trace_row(row, result, error, time.perf_counter() - started))
        print(f"traced {len(existing) + position}/30 {row.get('expected_language')} {row.get('question_id')}", flush=True)
        if (len(existing) + position) % 5 == 0 or position == len(pending):
            _write_csv(FAILURE_TRACE_CSV, output)
    return output


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("trace", "classify", "review", "full", "finalize"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "trace":
        rows = run_trace(args.resume)
        print(json.dumps({"rows": len(rows), "output": str(FAILURE_TRACE_CSV)}, ensure_ascii=False, indent=2))
    elif args.action == "classify":
        rows = reclassify_trace()
        print(json.dumps({"rows": len(rows), "classifications": dict(Counter(row.get("failure_classification", "") for row in rows))}, ensure_ascii=False, indent=2))
    elif args.action == "review":
        rows = run_evaluation(_review_dataset(load_dataset()), BEFORE_AFTER_CSV.with_name("step7e_review_run.csv"), args.resume)
        _write_review_artifacts(rows)
        summary = summarize_step7e(rows)
        SUMMARY_JSON.write_text(json.dumps({"review": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.action == "full":
        rows = run_evaluation(load_dataset(), FULL_CSV, args.resume)
        summary = summarize_step7e(rows)
        SUMMARY_JSON.write_text(json.dumps({"full": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        rows = _read_csv(FULL_CSV)
        summary = summarize_step7e(rows)
        SUMMARY_JSON.write_text(json.dumps({"full": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
