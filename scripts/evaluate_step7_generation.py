from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import RERANKER_ENABLED
from src.evaluator import load_dataset
from src.generator import FIXED_N_CTX, MAX_NEW_TOKENS, TEMPERATURE, TOP_P
from src.language_validator import validate_language
from src.pipeline import answer_question


RESULT_CSV = ROOT_DIR / "results" / "step7_generation_results.csv"
SUMMARY_JSON = ROOT_DIR / "results" / "step7_generation_summary.json"
REPORT_MD = ROOT_DIR / "results" / "step7_generation_report.md"
MANUAL_CSV = ROOT_DIR / "results" / "step7_manual_review_sample.csv"
SMOKE_JSON = ROOT_DIR / "results" / "step7_real_model_smoke.json"


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))]


def tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[A-Za-z0-9@.+%-]+|[\u0980-\u09ff]+", str(text).casefold())
        if len(token) > 1
    }


def proxy_metrics(reference: str, answer: str) -> dict[str, float]:
    expected, actual = tokens(reference), tokens(answer)
    overlap = len(expected & actual)
    precision = overlap / len(actual) if actual else 0.0
    recall = overlap / len(expected) if expected else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def factual_tokens(text: str) -> set[str]:
    patterns = (
        r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        r"\b[A-Za-z]{2,12}\s*\d{2,4}(?:\s*\([A-Za-z0-9]+\))?\b",
        r"\b\d+(?:\.\d+)?%?\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4})\b",
    )
    return {re.sub(r"\s+", "", value).casefold() for pattern in patterns for value in re.findall(pattern, text, re.I)}


def evaluate(rows: list[dict[str, Any]], output_path: Path, initial_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if RERANKER_ENABLED:
        raise RuntimeError("Step-7 evaluation requires RERANKER_ENABLED=false.")
    output: list[dict[str, Any]] = list(initial_rows or [])
    completed = len(output)
    for position, row in enumerate(rows, start=1):
        started = time.perf_counter()
        result = answer_question(row["question"], top_k=3, use_generation=True, use_answer_bank=False)
        elapsed = time.perf_counter() - started
        expected_validation = validate_language(result.get("answer", ""), row.get("expected_language", "english"))
        proxy = proxy_metrics(row.get("reference_answer", ""), result.get("answer", ""))
        required = factual_tokens(row.get("reference_answer", ""))
        actual = factual_tokens(result.get("answer", ""))
        item = {
            "question_id": row.get("question_id"),
            "language": row.get("expected_language"),
            "question": row.get("question"),
            "reference": row.get("reference_answer"),
            "final_answer": result.get("answer"),
            "answer_strategy": result.get("answer_strategy"),
            "generation_used": bool(result.get("generation_used")),
            "generation_attempts": int(result.get("generation_attempts", 0)),
            "retry_used": bool(result.get("retry_used")),
            "support_status": result.get("support_status"),
            "target_language": result.get("target_language"),
            "language_validation_passed": bool(expected_validation.get("validation_passed")),
            "language_validation_reason": expected_validation.get("validation_reason"),
            "runtime_language_validation_passed": bool(result.get("language_validation_passed")),
            "grounding_validation_passed": bool(result.get("grounding_validation_passed")),
            "grounding_validation_reason": result.get("grounding_validation_reason"),
            "fact_tokens_required": sorted(required),
            "fact_tokens_preserved": bool(required and required <= actual),
            "fact_preservation_applicable": bool(required),
            "automatic_precision": proxy["precision"],
            "automatic_recall": proxy["recall"],
            "automatic_f1": proxy["f1"],
            "source": result.get("source"),
            "page": result.get("page"),
            "supporting_excerpt": result.get("supporting_excerpt"),
            "retrieval_seconds": result.get("latency_seconds", {}).get("retrieval", 0.0),
            "answering_seconds": result.get("latency_seconds", {}).get("answering", 0.0),
            "generation_seconds": result.get("latency_seconds", {}).get("generation", 0.0),
            "retry_seconds": result.get("latency_seconds", {}).get("retry", 0.0),
            "total_seconds": elapsed,
            "answer_bank_enabled": False,
            "reranker_enabled": False,
        }
        output.append(item)
        total_position = completed + position
        total_expected = completed + len(rows)
        if total_position % 10 == 0 or position == len(rows):
            print(f"answered {total_position}/{total_expected}", flush=True)
            # Atomic checkpoints avoid exposing a partially written CSV.
            write_csv(output_path, output)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    serializable = []
    for row in rows:
        serializable.append({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value for key, value in row.items()})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable[0]))
        writer.writeheader()
        writer.writerows(serializable)
    temporary.replace(path)


def read_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    boolean_fields = {
        "generation_used", "retry_used", "language_validation_passed", "runtime_language_validation_passed",
        "grounding_validation_passed", "fact_tokens_preserved", "fact_preservation_applicable",
        "answer_bank_enabled", "reranker_enabled",
    }
    integer_fields = {"generation_attempts"}
    float_fields = {
        "automatic_precision", "automatic_recall", "automatic_f1", "retrieval_seconds", "answering_seconds",
        "generation_seconds", "retry_seconds", "total_seconds",
    }
    for row in rows:
        for field in boolean_fields:
            row[field] = str(row.get(field, "")).casefold() == "true"
        for field in integer_fields:
            row[field] = int(row.get(field) or 0)
        for field in float_fields:
            row[field] = float(row.get(field) or 0.0)
        row["page"] = int(row["page"]) if str(row.get("page", "")).isdigit() else None
        try:
            row["fact_tokens_required"] = json.loads(row.get("fact_tokens_required") or "[]")
        except json.JSONDecodeError:
            row["fact_tokens_required"] = []
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strategies = Counter(str(row["answer_strategy"]) for row in rows)
    languages = {}
    for language in ("english", "bangla", "banglish"):
        selected = [row for row in rows if row["language"] == language]
        languages[language] = {
            "passed": sum(row["language_validation_passed"] for row in selected),
            "total": len(selected),
            "rate": sum(row["language_validation_passed"] for row in selected) / len(selected) if selected else 0.0,
            "reasons": dict(Counter(str(row["language_validation_reason"]) for row in selected)),
        }
    generated = [row for row in rows if row["generation_used"]]
    accepted = [row for row in generated if row["answer_strategy"] == "gguf_generation" and row["grounding_validation_passed"]]
    rejected = [row for row in generated if row["answer_strategy"] == "unsupported"]
    retries = [row for row in generated if row["retry_used"]]
    applicable = [row for row in rows if row["fact_preservation_applicable"]]
    structured_latencies = [float(row["answering_seconds"]) for row in rows if str(row["answer_strategy"]).startswith("structured")]
    generation_latencies = [float(row["generation_seconds"]) for row in generated]
    total_latencies = [float(row["total_seconds"]) for row in rows]
    return {
        "notice": "STEP-7 DEVELOPMENT / REGRESSION EVALUATION — NOT FINAL THESIS ACCURACY",
        "queries": len(rows),
        "answer_bank_enabled": False,
        "reranker_enabled": False,
        "strategy_counts": dict(strategies),
        "language_consistency": {**languages, "overall": {
            "passed": sum(row["language_validation_passed"] for row in rows), "total": len(rows),
            "rate": sum(row["language_validation_passed"] for row in rows) / len(rows) if rows else 0.0,
        }},
        "grounding": {
            "generated_answers": len(generated), "accepted_generation": len(accepted), "rejected_generation": len(rejected),
            "retries": len(retries),
            "successful_retries": sum(row["answer_strategy"] == "gguf_generation" for row in retries),
            "failed_retries": sum(row["answer_strategy"] == "unsupported" for row in retries),
        },
        "fact_preservation": {
            "applicable": len(applicable), "preserved": sum(row["fact_tokens_preserved"] for row in applicable),
            "rate": sum(row["fact_tokens_preserved"] for row in applicable) / len(applicable) if applicable else 0.0,
        },
        "automatic_diagnostic_proxies": {
            metric: statistics.mean(float(row[f"automatic_{metric}"]) for row in rows) if rows else 0.0
            for metric in ("precision", "recall", "f1")
        },
        "latency_seconds": {
            "structured_median": statistics.median(structured_latencies) if structured_latencies else 0.0,
            "structured_p95": percentile(structured_latencies, 0.95),
            "gguf_median": statistics.median(generation_latencies) if generation_latencies else 0.0,
            "gguf_p95": percentile(generation_latencies, 0.95),
            "retry_total": sum(float(row["retry_seconds"]) for row in rows),
            "total_median": statistics.median(total_latencies) if total_latencies else 0.0,
            "total_p95": percentile(total_latencies, 0.95),
        },
        "qwen": {"model": "Qwen2.5-1.5B-Instruct", "quantization": "Q4_K_M", "context": FIXED_N_CTX, "temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_NEW_TOKENS},
    }


def manual_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sample: list[dict[str, Any]] = []
    for language in ("english", "bangla", "banglish"):
        candidates = [row for row in rows if row["language"] == language]
        by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            by_strategy[str(row["answer_strategy"])].append(row)
        chosen: list[dict[str, Any]] = []
        while len(chosen) < 10 and any(by_strategy.values()):
            for strategy in sorted(by_strategy):
                if by_strategy[strategy] and len(chosen) < 10:
                    chosen.append(by_strategy[strategy].pop(0))
        sample.extend({key: row[key] for key in ("language", "question", "reference", "final_answer", "answer_strategy", "source", "page", "supporting_excerpt")} for row in chosen)
    return sample


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Step 7 Multilingual Answering Development Report", "",
        "**DEVELOPMENT / REGRESSION EVALUATION — NOT FINAL THESIS ACCURACY**", "",
        f"Queries: {summary['queries']}; answer bank: off; reranker: off.", "",
        "## Answer strategies", "", json.dumps(summary["strategy_counts"], ensure_ascii=False), "",
        "## Language consistency", "",
    ]
    for language in ("english", "bangla", "banglish", "overall"):
        value = summary["language_consistency"][language]
        lines.append(f"- {language.title()}: {value['passed']}/{value['total']} ({100 * value['rate']:.2f}%)")
    lines.extend([
        "", "## Grounding", "", json.dumps(summary["grounding"], ensure_ascii=False), "",
        "## Fact-preservation diagnostic", "", json.dumps(summary["fact_preservation"], ensure_ascii=False), "",
        "## Automatic diagnostic proxies (not human correctness)", "", json.dumps(summary["automatic_diagnostic_proxies"], ensure_ascii=False), "",
        "## Latency seconds", "", json.dumps(summary["latency_seconds"], ensure_ascii=False), "",
    ])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def select_smoke(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for language in ("english", "bangla", "banglish"):
        candidates = [row for row in rows if row["expected_language"] == language]
        indices = (0, 1, 2, 6, 9)
        selected.extend(candidates[index] for index in indices)
    return selected


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    dataset = load_dataset()
    if args.smoke_only:
        selected = select_smoke(dataset)
        rows = evaluate(selected, ROOT_DIR / "results" / "step7_smoke_progress.csv")
        SMOKE_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        selected = dataset[:args.limit] if args.limit else dataset
        checkpoint = read_checkpoint(RESULT_CSV) if args.resume else []
        if len(checkpoint) > len(selected):
            raise RuntimeError("Checkpoint contains more rows than the selected evaluation dataset.")
        rows = evaluate(selected[len(checkpoint):], RESULT_CSV, initial_rows=checkpoint)
        summary = summarize(rows)
        SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(summary)
        write_csv(MANUAL_CSV, manual_sample(rows))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
