from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_step5_hybrid import expected_pages, grouped_summary, percentile, score_row, supportable, variants
from src.config import RERANKER_MODEL, RERANKER_REVISION
from src.embeddings import get_embedding_model
from src.evidence import analyze_query, identify_entities
from src.language_detector import detect_language
from src.query_normalization import build_query_representations
from src.reranker import DEFAULT_RERANKER_CONFIG, get_reranker_model, rerank_candidate_pool
from src.retriever import DEFAULT_HYBRID_CONFIG, reciprocal_rank_fusion
from src.vector_store import load_hybrid_artifacts, search


KS = (10, 15, 20)
BASELINE_PATH = ROOT_DIR / "results" / "step6_normalized_results.json"
OUTPUTS = {k: ROOT_DIR / "results" / f"step6_reranker_k{k}.json" for k in KS}
SELECTED_OUTPUT = ROOT_DIR / "results" / "step6_reranked_results.json"
REPORT_OUTPUT = ROOT_DIR / "results" / "step6_reranker_report.md"
FAILURE_CATEGORIES = (
    "FIRST_STAGE_CANDIDATE_MISS",
    "RERANKER_ORDERING_ERROR",
    "ENTITY_ERROR",
    "FIELD_ERROR",
    "LANGUAGE_NORMALIZATION_ERROR",
    "GROUND_TRUTH_LABEL_AMBIGUITY",
)


def correct_page(source: dict[str, Any], item: dict[str, Any]) -> bool:
    return str(item.get("source", "")) == source["expected_source"] and item.get("page") in expected_pages(source["expected_page"])


def pool_oracle(source: dict[str, Any], pool: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "correct_page_available": any(correct_page(source, item) for item in pool),
        "supportable_available": any(supportable(source["question"], item) for item in pool),
    }


def classify_change(before: dict[str, Any], after: dict[str, Any]) -> str:
    old_rank, new_rank = int(before["page_rank"]), int(after["page_rank"])
    if new_rank and (not old_rank or new_rank < old_rank):
        return "PROMOTED_CORRECT_EVIDENCE"
    if old_rank and (not new_rank or new_rank > old_rank):
        return "DEMOTED_CORRECT_EVIDENCE"
    if old_rank and new_rank:
        return "UNCHANGED_CORRECT"
    return "UNCHANGED_INCORRECT"


def classify_failure(source: dict[str, Any], row: dict[str, Any], oracle: dict[str, bool]) -> str | None:
    # Diagnose final top-three correct-page failures. Supportability is reported independently.
    if 0 < int(row["page_rank"]) <= 3:
        return None
    if detect_language(source["question"]) != source["language"]:
        return "LANGUAGE_NORMALIZATION_ERROR"
    if len(expected_pages(source["expected_page"])) > 1:
        return "GROUND_TRUTH_LABEL_AMBIGUITY"
    if not oracle["correct_page_available"]:
        return "FIRST_STAGE_CANDIDATE_MISS"
    if row["has_entity"] and not row["entity_hit_3"]:
        return "ENTITY_ERROR"
    if row["has_field"] and not row["field_hit_3"]:
        return "FIELD_ERROR"
    return "RERANKER_ORDERING_ERROR"


def summarize_counts(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    overall: Counter[str] = Counter()
    for row in rows:
        value = row[field]
        if value:
            grouped[row["language"]][value] += 1
            overall[value] += 1
    names = FAILURE_CATEGORIES if field == "failure_category" else (
        "PROMOTED_CORRECT_EVIDENCE", "DEMOTED_CORRECT_EVIDENCE", "UNCHANGED_CORRECT", "UNCHANGED_INCORRECT"
    )
    return {
        "overall": {name: overall.get(name, 0) for name in names},
        "languages": {
            language: {name: grouped[language].get(name, 0) for name in names}
            for language in ("english", "bangla", "banglish")
        },
    }


def metric_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    fields = (
        "page_hit_at_1", "page_hit_at_3", "mrr_at_3", "entity_hit_at_1", "entity_hit_at_3",
        "field_hit_at_1", "field_hit_at_3", "supportable_at_1", "supportable_at_3",
    )
    return {field: float(current[field]) - float(baseline[field]) for field in fields}


def choose_k(reports: dict[int, dict[str, Any]], baseline: dict[str, Any]) -> int:
    # Protect every language and overall top-3 support/field/entity within one percentage point.
    eligible: list[int] = []
    for k, report in reports.items():
        groups = [(report["summary"]["overall"], baseline["overall"])] + [
            (report["summary"]["languages"][language], baseline["languages"][language])
            for language in ("english", "bangla", "banglish")
        ]
        if all(
            current[metric] >= old[metric] - 0.01
            for current, old in groups
            for metric in ("supportable_at_3", "field_hit_at_3", "entity_hit_at_3", "page_hit_at_3")
        ):
            eligible.append(k)
    choices = eligible or list(reports)
    return max(choices, key=lambda k: (
        reports[k]["summary"]["overall"]["supportable_at_1"],
        reports[k]["summary"]["overall"]["field_hit_at_1"],
        reports[k]["summary"]["overall"]["entity_hit_at_1"],
        reports[k]["summary"]["overall"]["page_hit_at_1"],
        reports[k]["summary"]["overall"]["mrr_at_3"],
        min(reports[k]["summary"]["languages"][language]["supportable_at_1"] for language in ("english", "bangla", "banglish")),
        -reports[k]["latency_seconds"]["reranker_median"],
    ))


def compact_item(item: dict[str, Any]) -> dict[str, Any]:
    return {name: item.get(name) for name in (
        "chunk_id", "source", "page", "dense_rank", "sparse_rank", "metadata_rank", "fusion_score",
        "pre_rerank_rank", "reranker_score", "final_rank",
    )}


def run() -> dict[str, Any]:
    baseline_document = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baseline = baseline_document["selected_summary"]
    baseline_rows = baseline_document["rows"]
    dataset = variants()
    if len(dataset) != 300 or [sum(row["language"] == language for row in dataset) for language in ("english", "bangla", "banglish")] != [100, 100, 100]:
        raise RuntimeError("Step-6 reranker evaluation requires exactly 100 English, 100 Bangla, and 100 Banglish queries.")

    embedding = get_embedding_model()
    index, metadata, manifest, sparse_index = load_hybrid_artifacts(configuration=embedding.configuration)
    embedding_started = time.perf_counter()
    vectors = embedding.embed_many([row["question"] for row in dataset])
    embedding_seconds = time.perf_counter() - embedding_started
    embedding_per_query = embedding_seconds / len(dataset)

    pools: list[list[dict[str, Any]]] = []
    first_stage_seconds: list[float] = []
    for source, vector in zip(dataset, vectors):
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
        pools.append(reciprocal_rank_fusion(dense, sparse, metadata_ranked, DEFAULT_HYBRID_CONFIG)[: max(KS)])
        first_stage_seconds.append(time.perf_counter() - started + embedding_per_query)

    try:
        import psutil
        process = psutil.Process()
        rss_before = process.memory_info().rss
    except Exception:
        process = None
        rss_before = None
    load_started = time.perf_counter()
    reranker = get_reranker_model(DEFAULT_RERANKER_CONFIG)
    load_seconds = time.perf_counter() - load_started
    rss_after = process.memory_info().rss if process else None

    rows_by_k: dict[int, list[dict[str, Any]]] = {k: [] for k in KS}
    reranker_latency_by_k: dict[int, list[float]] = {k: [] for k in KS}
    full_latency_by_k: dict[int, list[float]] = {k: [] for k in KS}
    for position, (source, pool) in enumerate(zip(dataset, pools), start=1):
        segment_scores: list[float] = []
        cumulative_seconds = 0.0
        scores_by_k: dict[int, list[float]] = {}
        seconds_by_k: dict[int, float] = {}
        previous = 0
        for k in KS:
            segment = pool[previous:k]
            started = time.perf_counter()
            segment_scores.extend(reranker.score_pairs(source["question"], [str(item.get("text") or "") for item in segment]))
            cumulative_seconds += time.perf_counter() - started
            scores_by_k[k] = list(segment_scores)
            seconds_by_k[k] = cumulative_seconds
            previous = k
        for k in KS:
            class FixedScorer:
                def score_pairs(self, query: str, passages: list[str], values: list[float] = scores_by_k[k]) -> list[float]:
                    return values
            ranked = rerank_candidate_pool(source["question"], pool, FixedScorer(), k)
            reranker_latency_by_k[k].append(seconds_by_k[k])
            full_seconds = first_stage_seconds[position - 1] + seconds_by_k[k]
            full_latency_by_k[k].append(full_seconds)
            row = score_row(source, ranked[:3], full_seconds)
            oracle = pool_oracle(source, pool[:k])
            row["candidate_pool"] = oracle
            row["change_class"] = classify_change(baseline_rows[position - 1], row)
            row["failure_category"] = classify_failure(source, row, oracle)
            row["retrieved"] = [compact_item(item) for item in ranked[:3]]
            rows_by_k[k].append(row)
        if position % 10 == 0:
            print(f"reranked {position}/{len(dataset)}", flush=True)

    reports: dict[int, dict[str, Any]] = {}
    for k in KS:
        summary = grouped_summary(rows_by_k[k])
        oracle = {
            scope: {
                "queries": len(selected),
                "candidate_pool_correct_page_at_k": sum(row["candidate_pool"]["correct_page_available"] for row in selected) / len(selected),
                "candidate_pool_supportable_at_k": sum(row["candidate_pool"]["supportable_available"] for row in selected) / len(selected),
            }
            for scope, selected in [("overall", rows_by_k[k])] + [
                (language, [row for row in rows_by_k[k] if row["language"] == language])
                for language in ("english", "bangla", "banglish")
            ]
        }
        reports[k] = {
            "notice": "STEP-6 RERANKER DEVELOPMENT / REGRESSION EVALUATION - NOT FINAL THESIS ACCURACY",
            "query_count": len(dataset),
            "answer_bank_enabled": False,
            "generation_enabled": False,
            "candidate_k": k,
            "query_representation": "original",
            "passage_representation": "candidate evidence text",
            "summary": summary,
            "delta_vs_selected_normalized_baseline": {
                "overall": metric_delta(summary["overall"], baseline["overall"]),
                "languages": {language: metric_delta(summary["languages"][language], baseline["languages"][language]) for language in ("english", "bangla", "banglish")},
            },
            "candidate_pool_oracle": oracle,
            "win_loss_analysis": summarize_counts(rows_by_k[k], "change_class"),
            "failure_analysis": summarize_counts(rows_by_k[k], "failure_category"),
            "latency_seconds": {
                "reranker_median": statistics.median(reranker_latency_by_k[k]),
                "reranker_p95": percentile(reranker_latency_by_k[k], 0.95),
                "full_retrieval_median": statistics.median(full_latency_by_k[k]),
                "full_retrieval_p95": percentile(full_latency_by_k[k], 0.95),
                "embedding_mode": "batched evaluation time amortized per query",
            },
            "rows": rows_by_k[k],
        }
    selected_k = choose_k(reports, baseline)
    return {
        "reports": reports,
        "selected_k": selected_k,
        "selected": reports[selected_k],
        "baseline": baseline,
        "model": {
            **reranker.provenance(),
            "in_process_load_seconds_after_embedding_model": load_seconds,
            "process_rss_before_bytes": rss_before,
            "process_rss_after_bytes": rss_after,
            "process_rss_delta_bytes": rss_after - rss_before if rss_before is not None and rss_after is not None else None,
        },
        "first_stage_latency_seconds": {
            "median": statistics.median(first_stage_seconds),
            "p95": percentile(first_stage_seconds, 0.95),
            "batched_embedding_total": embedding_seconds,
        },
        "manifest_index_configuration_fingerprint": manifest.get("index_configuration_fingerprint"),
    }


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def write_outputs(result: dict[str, Any]) -> None:
    for k in KS:
        OUTPUTS[k].write_text(json.dumps(result["reports"][k], ensure_ascii=False, indent=2), encoding="utf-8")
    selected = {
        "notice": "SELECTED STEP-6 RERANKER DEVELOPMENT RESULT - NOT FINAL THESIS ACCURACY",
        "selected_k": result["selected_k"],
        "default_enabled": False,
        "model": result["model"],
        "first_stage_latency_seconds": result["first_stage_latency_seconds"],
        **result["selected"],
    }
    SELECTED_OUTPUT.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Step 6 Neural Reranker Development Report", "",
        "**DEVELOPMENT / REGRESSION EVALUATION — NOT FINAL THESIS ACCURACY**", "",
        f"Pinned model: `{RERANKER_MODEL}@{RERANKER_REVISION}` (offline/local-only)", "",
        "| K | Language | Page H@1/H@3 | MRR@3 | Entity H@1/H@3 | Field H@1/H@3 | Supportable@1/@3 | Reranker median/P95 |", "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in KS:
        report = result["reports"][k]
        for language in ("english", "bangla", "banglish", "overall"):
            values = report["summary"]["overall"] if language == "overall" else report["summary"]["languages"][language]
            latency = report["latency_seconds"]
            lines.append(
                f"| {k} | {language.title()} | {pct(values['page_hit_at_1'])}/{pct(values['page_hit_at_3'])} | {values['mrr_at_3']:.4f} | "
                f"{pct(values['entity_hit_at_1'])}/{pct(values['entity_hit_at_3'])} | {pct(values['field_hit_at_1'])}/{pct(values['field_hit_at_3'])} | "
                f"{pct(values['supportable_at_1'])}/{pct(values['supportable_at_3'])} | {latency['reranker_median']:.3f}s/{latency['reranker_p95']:.3f}s |"
            )
    lines.extend(["", f"Selected candidate K: **{result['selected_k']}**", "", "RRF creates the bounded pool; raw neural logits only determine its final order. No raw-score addition is used."])
    REPORT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    output = run()
    write_outputs(output)
    print(json.dumps({"selected_k": output["selected_k"], "selected_summary": output["selected"]["summary"], "model": output["model"]}, ensure_ascii=False, indent=2))
