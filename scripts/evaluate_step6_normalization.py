from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_step5_hybrid import grouped_summary, score_row, variants
from src.config import RERANKER_CANDIDATE_K, RERANKER_MODEL, RERANKER_REVISION
from src.embeddings import get_embedding_model
from src.evidence import SupportStatus, analyze_query, assess_evidence, identify_entities
from src.language_detector import detect_language
from src.query_normalization import build_query_representations, dense_query_for_strategy, normalize_retrieval_text
from src.reranker import inspect_local_reranker
from src.retriever import DEFAULT_HYBRID_CONFIG, reciprocal_rank_fusion
from src.vector_store import load_hybrid_artifacts, search


STEP5_RESULT = ROOT_DIR / "results" / "step5_hybrid_results.json"
BASELINE_OUTPUT = ROOT_DIR / "results" / "step6_hybrid_baseline.json"
NORMALIZED_OUTPUT = ROOT_DIR / "results" / "step6_normalized_results.json"
RERANKED_OUTPUT = ROOT_DIR / "results" / "step6_reranked_results.json"
REPORT_OUTPUT = ROOT_DIR / "results" / "step6_report.md"

STRATEGIES = ("original", "normalized", "original_plus_normalized")
FAILURE_CATEGORIES = (
    "QUERY_NORMALIZATION_ERROR",
    "LANGUAGE_DETECTION_ERROR",
    "DENSE_CANDIDATE_MISS",
    "SPARSE_CANDIDATE_MISS",
    "FUSION_ERROR",
    "RERANKER_ERROR",
    "ENTITY_RESOLUTION_ERROR",
    "FIELD_RESOLUTION_ERROR",
    "GROUND_TRUTH_LABEL_AMBIGUITY",
)


def expected_pages(value: str) -> set[int]:
    return {int(part.strip()) for group in str(value).split(";") for part in group.split(",") if part.strip().isdigit()}


def supportable(question: str, item: dict[str, Any]) -> bool:
    return assess_evidence(question, [item]).status in {SupportStatus.SUPPORTED, SupportStatus.CONFLICTING}


def choose_strategy(summaries: dict[str, dict[str, Any]], baseline: dict[str, Any]) -> str:
    protected = {}
    for name, report in summaries.items():
        current = report["overall"]
        if (
            current["page_hit_at_3"] >= baseline["page_hit_at_3"] - 0.01
            and current["entity_hit_at_3"] >= baseline["entity_hit_at_3"] - 0.01
            and current["field_hit_at_3"] >= baseline["field_hit_at_3"] - 0.01
            and current["supportable_at_3"] >= baseline["supportable_at_3"] - 0.01
        ):
            protected[name] = report
    eligible = protected or summaries
    return max(eligible, key=lambda name: (
        eligible[name]["overall"]["supportable_at_1"],
        eligible[name]["overall"]["field_hit_at_1"],
        eligible[name]["overall"]["entity_hit_at_1"],
        eligible[name]["overall"]["page_hit_at_1"],
        eligible[name]["overall"]["mrr_at_3"],
        min(item["supportable_at_1"] for item in eligible[name]["languages"].values()),
        min(item["page_hit_at_1"] for item in eligible[name]["languages"].values()),
        name,
    ))


def classify_failure(source: dict[str, str], record: dict[str, Any]) -> str | None:
    pages = expected_pages(source["expected_page"])
    correct = lambda item: str(item.get("source", "")) == source["expected_source"] and item.get("page") in pages
    if any(correct(item) for item in record["final"][:3]):
        return None
    if detect_language(source["question"]) != source["language"]:
        return "LANGUAGE_DETECTION_ERROR"
    if len(pages) > 1:
        return "GROUND_TRUTH_LABEL_AMBIGUITY"
    original_entities = identify_entities(source["question"])
    normalized_entities = identify_entities(record["normalized_query"])
    if original_entities and not normalized_entities:
        return "QUERY_NORMALIZATION_ERROR"
    if not any(correct(item) for item in record["dense"]):
        return "DENSE_CANDIDATE_MISS"
    if not any(correct(item) for item in record["sparse"]):
        return "SPARSE_CANDIDATE_MISS"
    if original_entities and not record["metadata"]:
        return "ENTITY_RESOLUTION_ERROR"
    request = analyze_query(record["normalized_query"])
    if request.requested_field != "general" and not any(
        supportable(source["question"], item)
        for item in record["dense"] + record["sparse"] + record["metadata"]
    ):
        return "FIELD_RESOLUTION_ERROR"
    return "FUSION_ERROR"


def language_detection_results(dataset: list[dict[str, str]]) -> dict[str, Any]:
    by_language: dict[str, dict[str, Any]] = {}
    categories = Counter()
    for language in ("english", "bangla", "banglish"):
        rows = [row for row in dataset if row["language"] == language]
        predictions = [detect_language(row["question"]) for row in rows]
        correct = sum(prediction == language for prediction in predictions)
        by_language[language] = {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}
        for prediction in predictions:
            if prediction != language:
                categories[f"{language.upper()}_AS_{prediction.upper()}"] += 1
    return {"languages": by_language, "failure_categories": dict(sorted(categories.items()))}


def spelling_variation_results() -> dict[str, Any]:
    cases = (
        ("CSE205 prereq", ("cse 205", "prerequisite"), "english"),
        ("CSE-205 credits", ("cse 205", "credits"), "english"),
        ("ABC ১২৩ এর ক্রেডিট কত", ("abc 123", "ক্রেডিট"), "bangla"),
        ("৫ম সপ্তাহে প্রিরিকুইজিট কী", ("সপ্তাহ 5", "prerequisite"), "bangla"),
        ("XYZ201A er credit koy", ("xyz 201(a)", "credit koto"), "banglish"),
        ("kii requirement hbe", ("ki requirement hobe",), "banglish"),
        ("info@example.edu er jnno ki krte hbe", ("info@example.edu", "jonno", "korte", "hobe"), "banglish"),
    )
    rows = []
    for query, expected_terms, language in cases:
        actual = normalize_retrieval_text(query)
        detected = detect_language(query)
        rows.append({
            "query": query,
            "normalized": actual,
            "expected_terms": expected_terms,
            "expected_language": language,
            "detected_language": detected,
            "passed": all(term in actual for term in expected_terms) and detected == language,
        })
    return {"passed": sum(row["passed"] for row in rows), "total": len(rows), "rows": rows}


def run() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen_step5 = json.loads(STEP5_RESULT.read_text(encoding="utf-8"))
    baseline = {
        "notice": "FROZEN STEP-5 DEVELOPMENT BASELINE - NOT FINAL THESIS ACCURACY",
        "source": str(STEP5_RESULT.relative_to(ROOT_DIR)),
        "selected_configuration": frozen_step5["selected_configuration"],
        "overall": frozen_step5["selected_summary"]["overall"],
        "languages": frozen_step5["selected_summary"]["languages"],
    }
    dataset = variants()
    model = get_embedding_model()
    index, metadata, manifest, sparse_index = load_hybrid_artifacts(configuration=model.configuration)

    vectors: dict[str, Any] = {}
    embedding_seconds: dict[str, float] = {}
    for strategy in STRATEGIES:
        started = time.perf_counter()
        vectors[strategy] = model.embed_many([dense_query_for_strategy(row["question"], strategy) for row in dataset])
        embedding_seconds[strategy] = time.perf_counter() - started

    rows_by_strategy: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATEGIES}
    records_by_strategy: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATEGIES}
    for strategy in STRATEGIES:
        for source, vector in zip(dataset, vectors[strategy]):
            started = time.perf_counter()
            dense = search(index, vector, metadata, top_k=min(len(metadata), DEFAULT_HYBRID_CONFIG.dense_candidate_k))
            for rank, item in enumerate(dense, start=1):
                item["dense_rank"] = rank
                item["dense_score"] = float(item.pop("score", 0.0))
            representations = build_query_representations(source["question"])
            sparse = sparse_index.search(representations.retrieval_query, metadata, DEFAULT_HYBRID_CONFIG.sparse_candidate_k)
            request = analyze_query(representations.normalized_query)
            metadata_ranked = sparse_index.metadata_candidates(
                request.primary_entity or "", request.requested_field, metadata, DEFAULT_HYBRID_CONFIG.metadata_candidate_k
            )
            final = reciprocal_rank_fusion(dense, sparse, metadata_ranked, DEFAULT_HYBRID_CONFIG)[:3]
            latency = time.perf_counter() - started
            rows_by_strategy[strategy].append(score_row(source, final, latency))
            records_by_strategy[strategy].append({
                "normalized_query": representations.normalized_query,
                "dense": dense,
                "sparse": sparse,
                "metadata": metadata_ranked,
                "final": final,
            })

    summaries = {name: grouped_summary(rows) for name, rows in rows_by_strategy.items()}
    selected = choose_strategy(summaries, baseline["overall"])
    failures = Counter(
        category for source, record in zip(dataset, records_by_strategy[selected])
        if (category := classify_failure(source, record))
    )
    normalized = {
        "notice": "STEP-6 NORMALIZATION DEVELOPMENT EVALUATION - NOT FINAL THESIS ACCURACY",
        "query_count": len(dataset),
        "strategies": STRATEGIES,
        "embedding_seconds": embedding_seconds,
        "strategy_summaries": summaries,
        "selected_strategy": selected,
        "selected_summary": summaries[selected],
        "failure_analysis": {name: failures.get(name, 0) for name in FAILURE_CATEGORIES},
        "language_detection": language_detection_results(dataset),
        "spelling_variations": spelling_variation_results(),
        "manifest_index_configuration_fingerprint": manifest.get("index_configuration_fingerprint"),
        "rows": rows_by_strategy[selected],
    }
    local_status = inspect_local_reranker(RERANKER_MODEL, RERANKER_REVISION)
    reranked = {
        "notice": "NOT RUN - USER PERMISSION REQUIRED BEFORE DOWNLOADING A RERANKER",
        "model": RERANKER_MODEL,
        "revision": RERANKER_REVISION,
        "available_locally": local_status.available,
        "download_required": not local_status.available,
        "candidate_k": RERANKER_CANDIDATE_K,
        "reason": local_status.reason,
        "expected_repository_storage": "approximately 2.29 GB (2.27 GB weights plus tokenizer/config files)",
        "estimated_cpu_ram": "approximately 3-5 GB for weights and inference overhead; longer batches may require more",
        "cpu_suitability": "CPU inference is supported, but top-15 latency must be benchmarked and may be several seconds per query",
        "license": "Apache-2.0 as declared by the model repository",
        "metrics": None,
        "win_loss_analysis": None,
    }
    return baseline, normalized, reranked


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def write_report(baseline: dict[str, Any], normalized: dict[str, Any], reranked: dict[str, Any]) -> None:
    lines = [
        "# Step 6 Multilingual Normalization Development Report",
        "",
        "**DEVELOPMENT / REGRESSION EVALUATION — NOT FINAL THESIS ACCURACY**",
        "",
        f"Selected dense representation: `{normalized['selected_strategy']}`",
        "",
        "| Language | Stage | Page H@1 | Page H@3 | MRR@3 | Entity H@1/H@3 | Field H@1/H@3 | Supportable@1/@3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for language in ("english", "bangla", "banglish", "overall"):
        before = baseline["overall"] if language == "overall" else baseline["languages"][language]
        after = normalized["selected_summary"]["overall"] if language == "overall" else normalized["selected_summary"]["languages"][language]
        for stage, values in (("Step 5", before), ("Normalized", after)):
            lines.append(
                f"| {language.title()} | {stage} | {pct(values['page_hit_at_1'])} | {pct(values['page_hit_at_3'])} | "
                f"{values['mrr_at_3']:.4f} | {pct(values['entity_hit_at_1'])}/{pct(values['entity_hit_at_3'])} | "
                f"{pct(values['field_hit_at_1'])}/{pct(values['field_hit_at_3'])} | "
                f"{pct(values['supportable_at_1'])}/{pct(values['supportable_at_3'])} |"
            )
    lines.extend(["", "## Dense representation experiments", ""])
    for name, report in normalized["strategy_summaries"].items():
        values = report["overall"]
        lines.append(
            f"- `{name}`: supportable@1/@3={pct(values['supportable_at_1'])}/{pct(values['supportable_at_3'])}; "
            f"field H@1/H@3={pct(values['field_hit_at_1'])}/{pct(values['field_hit_at_3'])}; "
            f"page H@1/H@3={pct(values['page_hit_at_1'])}/{pct(values['page_hit_at_3'])}; MRR@3={values['mrr_at_3']:.4f}."
        )
    lines.extend(["", "## Language detection", ""])
    for language, values in normalized["language_detection"]["languages"].items():
        lines.append(f"- {language.title()}: {values['correct']}/{values['total']} ({pct(values['accuracy'])})")
    lines.extend(["", "## Failure analysis", ""])
    for name, count in normalized["failure_analysis"].items():
        lines.append(f"- {name}: {count}")
    lines.extend([
        "",
        "## Reranker status",
        "",
        f"- Model: `{reranked['model']}`",
        f"- Proposed pinned revision: `{reranked['revision']}`",
        f"- Available locally: {reranked['available_locally']}",
        f"- Download required: {reranked['download_required']}",
        f"- Repository storage: {reranked['expected_repository_storage']}",
        f"- Estimated CPU RAM: {reranked['estimated_cpu_ram']}",
        f"- CPU suitability: {reranked['cpu_suitability']}",
        f"- License: {reranked['license']}",
        f"- Result: {reranked['notice']}",
    ])
    REPORT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    baseline, normalized, reranked = run()
    BASELINE_OUTPUT.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    NORMALIZED_OUTPUT.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    RERANKED_OUTPUT.write_text(json.dumps(reranked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(baseline, normalized, reranked)
    print(json.dumps({
        "selected_strategy": normalized["selected_strategy"],
        "selected_summary": normalized["selected_summary"],
        "language_detection": normalized["language_detection"],
        "spelling_variations": normalized["spelling_variations"],
        "failure_analysis": normalized["failure_analysis"],
        "reranker": reranked,
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
