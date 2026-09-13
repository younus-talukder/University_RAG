from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import QUESTIONS_DIR, VECTOR_DB_DIR
from src.embeddings import get_embedding_model
from src.evidence import SupportStatus, assess_evidence, detect_requested_field, identify_entities
from src.retriever import (
    HybridRetrievalConfiguration,
    reciprocal_rank_fusion,
    rerank_candidates,
)
from src.sparse_index import load_sparse_index
from src.vector_store import load_hybrid_artifacts, search


DENSE_OUTPUT = ROOT_DIR / "results" / "step5_dense_baseline.json"
HYBRID_OUTPUT = ROOT_DIR / "results" / "step5_hybrid_results.json"
REPORT_OUTPUT = ROOT_DIR / "results" / "step5_hybrid_report.md"

CONFIGURATIONS = {
    "A_equal": HybridRetrievalConfiguration(dense_weight=1.0, sparse_weight=1.0, metadata_weight=0.8),
    "B_dense_1_2": HybridRetrievalConfiguration(dense_weight=1.2, sparse_weight=1.0, metadata_weight=0.8),
    "C_sparse_0_8": HybridRetrievalConfiguration(dense_weight=1.0, sparse_weight=0.8, metadata_weight=0.8),
}

FAILURE_CATEGORIES = (
    "DENSE_MISSED",
    "SPARSE_MISSED",
    "BOTH_MISSED",
    "FUSION_RANK_ERROR",
    "ENTITY_METADATA_MISSED",
    "FIELD_METADATA_MISSED",
    "LABEL_AMBIGUITY",
)


def variants() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with (QUESTIONS_DIR / "questions.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            for language, column in (
                ("english", "english_question"),
                ("bangla", "bangla_question"),
                ("banglish", "banglish_question"),
            ):
                question = source.get(column, "").strip()
                if question:
                    rows.append({
                        "question_id": source.get("question_id", ""),
                        "language": language,
                        "question": question,
                        "expected_source": source.get("expected_source", ""),
                        "expected_page": source.get("expected_page", ""),
                    })
    return rows


def expected_pages(value: str) -> set[int]:
    return {
        int(part.strip())
        for group in str(value).split(";")
        for part in group.split(",")
        if part.strip().isdigit()
    }


def entity_match(question: str, chunk: dict[str, Any]) -> bool:
    entities = [entity for entity in identify_entities(question) if entity.kind == "course_code"]
    if not entities:
        return False
    compact = re.sub(r"[^A-Za-z0-9]", "", str(chunk.get("text", ""))).upper()
    return all(entity.normalized in compact for entity in entities)


def supportable(question: str, item: dict[str, Any]) -> bool:
    status = assess_evidence(question, [{**item, "score": item.get("score", item.get("fusion_score", 0.0))}]).status
    return status in {SupportStatus.SUPPORTED, SupportStatus.CONFLICTING}


def legacy_dense_select(ranked: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Frozen Step-4 behavior used only to produce the before baseline."""
    if not ranked or top_k <= 0:
        return []
    selected = [ranked[0]]
    seen = {ranked[0].get("chunk_id")}
    for item in sorted(
        ranked[1:],
        key=lambda value: (
            float(value.get("_semantic_score", value.get("score", 0.0))),
            float(value.get("score", 0.0)),
        ),
        reverse=True,
    ):
        if item.get("chunk_id") not in seen:
            selected.append(item)
            seen.add(item.get("chunk_id"))
        if len(selected) >= top_k:
            break
    return selected


def score_row(source: dict[str, str], retrieved: list[dict[str, Any]], latency: float) -> dict[str, Any]:
    pages = expected_pages(source["expected_page"])
    page_rank = next((
        rank for rank, item in enumerate(retrieved, start=1)
        if str(item.get("source", "")) == source["expected_source"] and item.get("page") in pages
    ), 0)
    entities = [entity for entity in identify_entities(source["question"]) if entity.kind == "course_code"]
    requested_field = detect_requested_field(source["question"])
    supports = [supportable(source["question"], item) for item in retrieved]
    return {
        **source,
        "page_rank": page_rank,
        "has_entity": bool(entities),
        "entity_hit_1": bool(retrieved and entity_match(source["question"], retrieved[0])),
        "entity_hit_3": any(entity_match(source["question"], item) for item in retrieved[:3]),
        "has_field": requested_field != "general",
        "field_hit_1": bool(supports and supports[0]),
        "field_hit_3": any(supports[:3]),
        "supportable_at_1": bool(supports and supports[0]),
        "supportable_at_3": any(supports[:3]),
        "latency_seconds_excluding_embedding": latency,
        "retrieved": [{
            "chunk_id": item.get("chunk_id"),
            "source": item.get("source"),
            "page": item.get("page"),
            "dense_rank": item.get("dense_rank"),
            "dense_score": item.get("dense_score", item.get("_semantic_score")),
            "sparse_rank": item.get("sparse_rank"),
            "sparse_score": item.get("sparse_score"),
            "metadata_rank": item.get("metadata_rank"),
            "fusion_score": item.get("fusion_score"),
            "final_rank": item.get("final_rank"),
        } for item in retrieved[:3]],
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))]


def approximate_size_bytes(value: Any) -> int:
    """Estimate the live Python object graph without counting shared objects twice."""
    seen: set[int] = set()

    def visit(item: Any) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        size = sys.getsizeof(item)
        if isinstance(item, dict):
            size += sum(visit(key) + visit(child) for key, child in item.items())
        elif isinstance(item, (list, tuple, set, frozenset)):
            size += sum(visit(child) for child in item)
        elif hasattr(item, "__dict__"):
            size += visit(vars(item))
        return size

    return visit(value)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entity_rows = [row for row in rows if row["has_entity"]]
    field_rows = [row for row in rows if row["has_field"]]
    latencies = [float(row["latency_seconds_excluding_embedding"]) for row in rows]
    count = len(rows)
    return {
        "queries": count,
        "page_hit_at_1": sum(row["page_rank"] == 1 for row in rows) / count if count else 0.0,
        "page_hit_at_3": sum(0 < row["page_rank"] <= 3 for row in rows) / count if count else 0.0,
        "mrr_at_3": sum(1 / row["page_rank"] if 0 < row["page_rank"] <= 3 else 0 for row in rows) / count if count else 0.0,
        "entity_queries": len(entity_rows),
        "entity_hit_at_1": sum(row["entity_hit_1"] for row in entity_rows) / len(entity_rows) if entity_rows else 0.0,
        "entity_hit_at_3": sum(row["entity_hit_3"] for row in entity_rows) / len(entity_rows) if entity_rows else 0.0,
        "field_queries": len(field_rows),
        "field_hit_at_1": sum(row["field_hit_1"] for row in field_rows) / len(field_rows) if field_rows else 0.0,
        "field_hit_at_3": sum(row["field_hit_3"] for row in field_rows) / len(field_rows) if field_rows else 0.0,
        "supportable_at_1": sum(row["supportable_at_1"] for row in rows) / count if count else 0.0,
        "supportable_at_3": sum(row["supportable_at_3"] for row in rows) / count if count else 0.0,
        "median_search_seconds_excluding_embedding": statistics.median(latencies) if latencies else 0.0,
        "p95_search_seconds_excluding_embedding": percentile(latencies, 0.95),
    }


def grouped_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["language"]].append(row)
    return {
        "overall": summarize(rows),
        "languages": {language: summarize(values) for language, values in sorted(grouped.items())},
    }


def select_configuration(reports: dict[str, dict[str, Any]], baseline: dict[str, Any]) -> str:
    baseline_entity = baseline["overall"]["entity_hit_at_3"]
    baseline_field = baseline["overall"]["field_hit_at_3"]
    eligible = {
        name: report for name, report in reports.items()
        if report["overall"]["entity_hit_at_3"] >= baseline_entity - 0.01
        and report["overall"]["field_hit_at_3"] >= baseline_field - 0.01
    } or reports
    return max(
        eligible,
        key=lambda name: (
            min(values["field_hit_at_3"] for values in eligible[name]["languages"].values()),
            min(values["entity_hit_at_3"] for values in eligible[name]["languages"].values()),
            min(values["page_hit_at_3"] for values in eligible[name]["languages"].values()),
            eligible[name]["overall"]["supportable_at_3"],
            eligible[name]["overall"]["mrr_at_3"],
            name,
        ),
    )


def failure_class(
    source: dict[str, str],
    dense: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    final: list[dict[str, Any]],
) -> str | None:
    pages = expected_pages(source["expected_page"])
    correct = lambda item: str(item.get("source", "")) == source["expected_source"] and item.get("page") in pages
    if any(correct(item) for item in final[:3]):
        return None
    dense_hit = any(correct(item) for item in dense)
    sparse_hit = any(correct(item) for item in sparse)
    if len(pages) > 1:
        return "LABEL_AMBIGUITY"
    if not dense_hit and not sparse_hit:
        return "BOTH_MISSED"
    if not dense_hit:
        return "DENSE_MISSED"
    if not sparse_hit:
        return "SPARSE_MISSED"
    has_entity = any(entity.kind == "course_code" for entity in identify_entities(source["question"]))
    if has_entity and not metadata:
        return "ENTITY_METADATA_MISSED"
    if detect_requested_field(source["question"]) != "general" and not any(
        supportable(source["question"], item) for item in dense + sparse + metadata
    ):
        return "FIELD_METADATA_MISSED"
    return "FUSION_RANK_ERROR"


def run() -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = variants()
    model = get_embedding_model()
    index, metadata, manifest, sparse_index = load_hybrid_artifacts(configuration=model.configuration)
    embedding_started = time.perf_counter()
    vectors = model.embed_many([row["question"] for row in dataset])
    embedding_seconds = time.perf_counter() - embedding_started

    baseline_rows: list[dict[str, Any]] = []
    configuration_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in CONFIGURATIONS}
    channel_records: list[tuple[dict, list, list, list, dict[str, list]]] = []
    dense_channel_seconds: list[float] = []
    sparse_metadata_seconds: list[float] = []
    fusion_channel_seconds: list[float] = []

    for source, vector in zip(dataset, vectors):
        dense_started = time.perf_counter()
        dense = search(index, vector, metadata, top_k=min(len(metadata), 30))
        for rank, item in enumerate(dense, start=1):
            item["dense_rank"] = rank
            item["dense_score"] = float(item.get("score", 0.0))
        dense_seconds = time.perf_counter() - dense_started
        dense_channel_seconds.append(dense_seconds)
        baseline_postprocess_started = time.perf_counter()
        baseline_candidates = rerank_candidates(source["question"], [dict(item) for item in dense[:15]])
        baseline_retrieved = legacy_dense_select(baseline_candidates, 3)
        baseline_latency = dense_seconds + (time.perf_counter() - baseline_postprocess_started)
        baseline_rows.append(score_row(source, baseline_retrieved, baseline_latency))

        sparse_started = time.perf_counter()
        sparse = sparse_index.search(source["question"], metadata, top_k=min(len(metadata), 30))
        request_entities = identify_entities(source["question"])
        primary_entity = request_entities[0].value if request_entities else ""
        metadata_ranked = sparse_index.metadata_candidates(
            primary_entity,
            detect_requested_field(source["question"]),
            metadata,
            top_k=min(len(metadata), 20),
        )
        channel_seconds = time.perf_counter() - sparse_started
        sparse_metadata_seconds.append(channel_seconds)
        fused_by_configuration: dict[str, list] = {}
        for name, configuration in CONFIGURATIONS.items():
            fusion_started = time.perf_counter()
            fused = reciprocal_rank_fusion(dense, sparse, metadata_ranked, configuration)[:3]
            fusion_seconds = time.perf_counter() - fusion_started
            if name == "A_equal":
                fusion_channel_seconds.append(fusion_seconds)
            fused_by_configuration[name] = fused
            configuration_rows[name].append(score_row(
                source,
                fused,
                dense_seconds + channel_seconds + fusion_seconds,
            ))
        channel_records.append((source, dense, sparse, metadata_ranked, fused_by_configuration))

    baseline_summary = grouped_summary(baseline_rows)
    configuration_summaries = {
        name: grouped_summary(rows) for name, rows in configuration_rows.items()
    }
    selected = select_configuration(configuration_summaries, baseline_summary)
    observed_failures = Counter(
        classification
        for source, dense, sparse, metadata_ranked, fused in channel_records
        if (classification := failure_class(source, dense, sparse, metadata_ranked, fused[selected]))
    )
    failures = {name: observed_failures.get(name, 0) for name in FAILURE_CATEGORIES}
    sparse_load_started = time.perf_counter()
    loaded_sparse = load_sparse_index(VECTOR_DB_DIR / "sparse_index.pkl", metadata)
    sparse_load_seconds = time.perf_counter() - sparse_load_started
    artifact_statistics = {
        "faiss_artifact_bytes": (VECTOR_DB_DIR / "index.faiss").stat().st_size,
        "metadata_artifact_bytes": (VECTOR_DB_DIR / "metadata.pkl").stat().st_size,
        "sparse_artifact_bytes": (VECTOR_DB_DIR / "sparse_index.pkl").stat().st_size,
        "approximate_loaded_sparse_python_bytes": approximate_size_bytes(sparse_index),
        "sparse_load_seconds": sparse_load_seconds,
        "sparse_chunk_count": loaded_sparse.chunk_count,
        "sparse_vocabulary_size": loaded_sparse.vocabulary_size,
    }
    shared = {
        "notice": "DEVELOPMENT / REGRESSION EVALUATION - NOT FINAL THESIS ACCURACY",
        "query_count": len(dataset),
        "batched_query_embedding_seconds": embedding_seconds,
        "average_batched_embedding_seconds_per_query": embedding_seconds / len(dataset),
        "manifest_index_configuration_fingerprint": manifest.get("index_configuration_fingerprint"),
        "channel_latency_seconds": {
            "dense_faiss_median": statistics.median(dense_channel_seconds),
            "dense_faiss_p95": percentile(dense_channel_seconds, 0.95),
            "bm25_plus_metadata_median": statistics.median(sparse_metadata_seconds),
            "bm25_plus_metadata_p95": percentile(sparse_metadata_seconds, 0.95),
            "rrf_median": statistics.median(fusion_channel_seconds),
            "rrf_p95": percentile(fusion_channel_seconds, 0.95),
        },
    }
    dense_output = {
        **shared,
        "mode": "frozen_step4_dense_baseline",
        **baseline_summary,
        "rows": baseline_rows,
    }
    hybrid_output = {
        **shared,
        "mode": "bm25_rrf_hybrid",
        "configurations": {name: asdict(value) for name, value in CONFIGURATIONS.items()},
        "configuration_summaries": configuration_summaries,
        "selected_configuration": selected,
        "selected_summary": configuration_summaries[selected],
        "failure_analysis": failures,
        "artifact_statistics": artifact_statistics,
        "rows": configuration_rows[selected],
    }
    return dense_output, hybrid_output


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def mib(value: int) -> str:
    return f"{value / (1024 * 1024):.2f} MiB"


def write_report(dense: dict[str, Any], hybrid: dict[str, Any]) -> None:
    selected = hybrid["selected_configuration"]
    lines = [
        "# Step 5 Hybrid Retrieval Development Report",
        "",
        "**DEVELOPMENT / REGRESSION EVALUATION — NOT FINAL THESIS ACCURACY**",
        "",
        f"Selected configuration: `{selected}`",
        "",
        "| Language | Mode | Page H@1 | Page H@3 | MRR@3 | Entity H@1/H@3 | Field H@1/H@3 | Supportable@1/@3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for language in ("english", "bangla", "banglish", "overall"):
        for mode, report in (("Dense", dense), ("Hybrid", hybrid["selected_summary"])):
            values = report["overall"] if language == "overall" else report["languages"][language]
            lines.append(
                f"| {language.title()} | {mode} | {pct(values['page_hit_at_1'])} | {pct(values['page_hit_at_3'])} | "
                f"{values['mrr_at_3']:.4f} | {pct(values['entity_hit_at_1'])}/{pct(values['entity_hit_at_3'])} | "
                f"{pct(values['field_hit_at_1'])}/{pct(values['field_hit_at_3'])} | "
                f"{pct(values['supportable_at_1'])}/{pct(values['supportable_at_3'])} |"
            )
    lines.extend([
        "",
        "## Dense to hybrid delta",
        "",
        "| Language | Page H@1 | Page H@3 | MRR@3 | Entity H@3 | Field H@3 | Supportable@3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for language in ("english", "bangla", "banglish", "overall"):
        before = dense["overall"] if language == "overall" else dense["languages"][language]
        after = hybrid["selected_summary"]["overall"] if language == "overall" else hybrid["selected_summary"]["languages"][language]
        lines.append(
            f"| {language.title()} | {pct(after['page_hit_at_1'] - before['page_hit_at_1'])} | "
            f"{pct(after['page_hit_at_3'] - before['page_hit_at_3'])} | "
            f"{after['mrr_at_3'] - before['mrr_at_3']:+.4f} | "
            f"{pct(after['entity_hit_at_3'] - before['entity_hit_at_3'])} | "
            f"{pct(after['field_hit_at_3'] - before['field_hit_at_3'])} | "
            f"{pct(after['supportable_at_3'] - before['supportable_at_3'])} |"
        )
    lines.extend([
        "",
        "## RRF configurations",
        "",
    ])
    for name, configuration in hybrid["configurations"].items():
        summary = hybrid["configuration_summaries"][name]["overall"]
        lines.append(
            f"- `{name}`: dense={configuration['dense_weight']}, sparse={configuration['sparse_weight']}, "
            f"metadata={configuration['metadata_weight']}, k={configuration['rrf_k']}; "
            f"page H@1/H@3={pct(summary['page_hit_at_1'])}/{pct(summary['page_hit_at_3'])}, "
            f"entity H@3={pct(summary['entity_hit_at_3'])}, field H@3={pct(summary['field_hit_at_3'])}."
        )
    lines.extend([
        "",
        "## Failure analysis",
        "",
    ])
    for name, count in hybrid["failure_analysis"].items():
        lines.append(f"- {name}: {count}")
    dense_latency = dense["overall"]
    hybrid_latency = hybrid["selected_summary"]["overall"]
    channels = hybrid["channel_latency_seconds"]
    artifacts = hybrid["artifact_statistics"]
    lines.extend([
        "",
        "## Retrieval latency",
        "",
        f"- Dense baseline search/post-processing (embedding excluded): median {dense_latency['median_search_seconds_excluding_embedding']:.6f}s; P95 {dense_latency['p95_search_seconds_excluding_embedding']:.6f}s.",
        f"- Hybrid search/fusion (embedding excluded): median {hybrid_latency['median_search_seconds_excluding_embedding']:.6f}s; P95 {hybrid_latency['p95_search_seconds_excluding_embedding']:.6f}s.",
        f"- Shared batched BGE-M3 embedding: {hybrid['batched_query_embedding_seconds']:.3f}s for {hybrid['query_count']} queries ({hybrid['average_batched_embedding_seconds_per_query']:.6f}s/query average).",
        f"- Channel medians: FAISS {channels['dense_faiss_median']:.6f}s; BM25 + metadata {channels['bm25_plus_metadata_median']:.6f}s; RRF {channels['rrf_median']:.6f}s.",
        "",
        "## Artifact and sparse-memory cost",
        "",
        f"- FAISS: {mib(artifacts['faiss_artifact_bytes'])}",
        f"- Metadata: {mib(artifacts['metadata_artifact_bytes'])}",
        f"- Sparse index: {mib(artifacts['sparse_artifact_bytes'])}",
        f"- Approximate loaded sparse Python object graph: {mib(artifacts['approximate_loaded_sparse_python_bytes'])}",
        f"- Sparse load + validation: {artifacts['sparse_load_seconds']:.6f}s",
        f"- Sparse corpus: {artifacts['sparse_chunk_count']} chunks; {artifacts['sparse_vocabulary_size']} terms.",
    ])
    REPORT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    dense, hybrid = run()
    DENSE_OUTPUT.write_text(json.dumps(dense, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    HYBRID_OUTPUT.write_text(json.dumps(hybrid, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(dense, hybrid)
    print(json.dumps({
        "dense": {"overall": dense["overall"], "languages": dense["languages"]},
        "selected_configuration": hybrid["selected_configuration"],
        "hybrid": hybrid["selected_summary"],
        "failure_analysis": hybrid["failure_analysis"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
