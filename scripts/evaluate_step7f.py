from __future__ import annotations

"""Read-only Step 7F failure funnel and parallel-language parity diagnostic.

Parallel dataset questions are used here only; production code never imports this file.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import get_embedding_model
from src.evidence import analyze_query, assess_evidence
from src.query_normalization import build_query_representations
from src.retriever import Retriever

RESULTS = ROOT / "results"
BASELINE = RESULTS / "step7e_300_results.csv"
TRACE = RESULTS / "step7e_failure_trace.csv"
DATASET = ROOT / "data" / "questions" / "questions.csv"
FUNNEL = RESULTS / "step7f_failure_funnel.csv"
PARITY = RESULTS / "step7f_language_parity_before.csv"
RECOVERY = RESULTS / "step7f_failure_recovery.csv"
PARITY_AFTER = RESULTS / "step7f_language_parity_after.csv"
METRICS = RESULTS / "step7f_retrieval_metrics.csv"
REVIEW = RESULTS / "step7f_multilingual_review.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, data: list[dict[str, object]]) -> None:
    if not data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def packed(candidates: list[dict], limit: int = 30) -> str:
    return json.dumps([
        {"rank": i, "chunk_id": item.get("chunk_id"), "page": item.get("page"),
         "source": item.get("source"), "score": item.get("score"),
         "dense_rank": item.get("dense_rank"), "sparse_rank": item.get("sparse_rank"),
         "metadata_rank": item.get("metadata_rank"), "text": str(item.get("text", ""))[:240]}
        for i, item in enumerate(candidates[:limit], 1)
    ], ensure_ascii=False)


def channel(candidates: list[dict], key: str) -> str:
    return packed(sorted((x for x in candidates if x.get(key)), key=lambda x: x[key]), 30)


def support(question: str, candidates: list[dict], count: int) -> bool:
    return assess_evidence(question, candidates[:count]).status.value == "supported"


def parity_label(values: dict[str, bool], after: bool = False) -> str:
    en, ba, bl = (values.get(key, False) for key in ("english", "bangla", "banglish"))
    if en and ba and bl:
        return "ALL_THREE_SUPPORTED"
    if en and not ba and not bl:
        return "ENGLISH_ONLY"
    if en and ba and not bl:
        return "ENGLISH_AND_BANGLA"
    if en and bl and not ba:
        return "ENGLISH_AND_BANGLISH"
    if not en and not ba and not bl:
        return "NONE_SUPPORTED"
    if after:
        return "OTHER_MISMATCH"
    if ba and not en and not bl:
        return "BANGLA_ONLY"
    if bl and not en and not ba:
        return "BANGLISH_ONLY"
    return "MULTILINGUAL_MISMATCH"


def primary_stage(base: dict[str, str], trace: dict[str, str], request, candidates: list[dict],
                  expected_page: str, english_ids: set[str]) -> str:
    status = base["final_status"]
    if status == "AMBIGUOUS":
        return "AMBIGUOUS_QUERY"
    if status in {"CONFLICTING_EVIDENCE", "CONFLICTING"}:
        return "OTHER"
    if status == "GENERATION_REJECTED":
        if not base.get("canonical_successful", "").lower() == "true":
            return "CANONICAL_GENERATION_FAILURE"
        if not base.get("realization_successful", "").lower() == "true":
            return "LANGUAGE_REALIZATION_FAILURE"
        reason = (base.get("generation_rejection_reason", "") + " " + base.get("realization_validation_reason", "")).lower()
        return "GROUNDING_REJECTION" if "ground" in reason else "SEMANTIC_VALIDATION_REJECTION"
    if request.requested_field == "general" and not request.entities:
        return "FIELD_DETECTION_FAILURE"
    if expected_page and any(str(x.get("page")) == expected_page for x in candidates[:3]):
        return "CORRECT_PAGE_WRONG_CHUNK"
    if english_ids and any(str(x.get("chunk_id")) in english_ids for x in candidates[:30]):
        return "EVIDENCE_VALIDATION_FAILURE" if any(str(x.get("chunk_id")) in english_ids for x in candidates[:3]) else "HYBRID_RANKING_MISS"
    if support(base["question"], candidates, 30):
        return "HYBRID_RANKING_MISS"
    if expected_page and any(str(x.get("page")) == expected_page for x in candidates[:30]):
        return "HYBRID_RANKING_MISS"
    if not candidates:
        return "DENSE_RETRIEVAL_MISS"
    if not any(x.get("dense_rank") for x in candidates):
        return "DENSE_RETRIEVAL_MISS"
    if not any(x.get("sparse_rank") for x in candidates):
        return "SPARSE_RETRIEVAL_MISS"
    return "TRUE_INFORMATION_ABSENT" if not english_ids else "HYBRID_RANKING_MISS"


def diagnose() -> None:
    baseline = rows(BASELINE)
    trace_lookup = {(x["question_id"], x["language"]): x for x in rows(TRACE)}
    dataset = {x["question_id"]: x for x in rows(DATASET)}
    by_id: dict[str, dict[str, dict[str, str]]] = {}
    for base in baseline:
        by_id.setdefault(base["question_id"], {})[base["language"]] = base
    parity = []
    for qid, trio in by_id.items():
        values = {language: row.get("support_status") == "supported" for language, row in trio.items()}
        parity.append({"question_id": qid, "english_status": trio["english"].get("support_status"),
                       "bangla_status": trio["bangla"].get("support_status"),
                       "banglish_status": trio["banglish"].get("support_status"),
                       "category": parity_label(values)})
    write_rows(PARITY, parity)

    failures = [x for x in baseline if x["language"] in {"bangla", "banglish"}
                and x["final_status"] != "ANSWER_RETURNED"]
    existing = {(x["question_id"], x["language"]): x for x in rows(FUNNEL)} if FUNNEL.exists() else {}
    retriever = Retriever(get_embedding_model(), top_k=80, reranker_enabled=False)
    english_cache: dict[str, set[str]] = {}
    output = []
    for index, base in enumerate(failures, 1):
        key = (base["question_id"], base["language"])
        if key in existing:
            if base["final_status"] == "CONFLICTING":
                existing[key]["primary_stage"] = "OTHER"
            output.append(existing[key])
            continue
        qid, language, question = base["question_id"], base["language"], base["question"]
        if qid not in english_cache:
            english = by_id[qid]["english"]
            if english.get("support_status") == "supported":
                eq = dataset[qid]["english_question"]
                english_results = retriever.retrieve(eq, debug=True)
                english_assessment = assess_evidence(eq, english_results[:3])
                english_cache[qid] = {str(x.chunk_id) for x in english_assessment.evidence}
            else:
                english_cache[qid] = set()
        candidates = retriever.retrieve(question, debug=True)
        assessment = assess_evidence(question, candidates[:3])
        request = analyze_query(question)
        representation = build_query_representations(question)
        trace = trace_lookup.get(key, {})
        expected_page = dataset[qid].get("expected_page", "")
        english_ids = english_cache[qid]
        stage = primary_stage(base, trace, request, candidates, expected_page, english_ids)
        output.append({
            "question_id": qid, "language": language, "original_query": question,
            "normalized_query": representation.normalized_query,
            "detected_entity": request.primary_entity or "", "requested_field": request.requested_field,
            "dense_top_candidates": channel(candidates, "dense_rank"),
            "bm25_top_candidates": channel(candidates, "sparse_rank"),
            "metadata_candidates": channel(candidates, "metadata_rank"),
            "rrf_top_candidates": packed(candidates), "top_3_final_chunks": packed(candidates, 3),
            "supportable_top_1": support(question, candidates, 1),
            "supportable_top_3": support(question, candidates, 3),
            "supportable_top_10": support(question, candidates, 10),
            "supportable_top_30": support(question, candidates, 30),
            "english_supporting_chunk_ids_offline": json.dumps(sorted(english_ids)),
            "english_support_chunk_hit_top_1": bool(english_ids & {str(x.get('chunk_id')) for x in candidates[:1]}),
            "english_support_chunk_hit_top_3": bool(english_ids & {str(x.get('chunk_id')) for x in candidates[:3]}),
            "english_support_chunk_hit_top_10": bool(english_ids & {str(x.get('chunk_id')) for x in candidates[:10]}),
            "english_support_chunk_hit_top_30": bool(english_ids & {str(x.get('chunk_id')) for x in candidates[:30]}),
            "entity_supported": bool(assessment.evidence and (not request.entities or assessment.evidence[0].matched_entity)),
            "field_supported": bool(assessment.evidence), "support_status": assessment.status.value,
            "canonical_generation_attempted": trace.get("answer_strategy_requested", "") not in {"", "ambiguous", "insufficient", "conflicting"},
            "canonical_generation_success": base.get("canonical_successful", ""),
            "realization_attempted": bool(base.get("target_language_realization")),
            "realization_success": base.get("realization_successful", ""),
            "semantic_validation": trace.get("first_semantic_validation_reason", ""),
            "language_validation": base.get("language_validation_reason", ""),
            "grounding_validation": base.get("grounding_validation_reason", ""),
            "final_status": base["final_status"],
            "final_reason": base.get("generation_rejection_reason") or trace.get("final_fallback_reason", ""),
            "expected_page_offline": expected_page, "primary_stage": stage,
        })
        if index % 5 == 0:
            write_rows(FUNNEL, output)
            print(f"diagnosed {index}/{len(failures)}", flush=True)
    write_rows(FUNNEL, output)
    print("failure stages:", dict(Counter((x["language"], x["primary_stage"]) for x in output)))
    print("parity before:", dict(Counter(x["category"] for x in parity)))


def failure_subset(rerun_ids: set[str] | None = None) -> None:
    """Run only the currently eligible multilingual retrieval failures."""
    from src.pipeline import answer_question

    baseline = rows(BASELINE)
    eligible = [x for x in baseline if x["language"] in {"bangla", "banglish"}
                and x["final_status"] == "INSUFFICIENT_EVIDENCE"]
    completed = {(x["question_id"], x["language"]): x for x in rows(RECOVERY)} if RECOVERY.exists() else {}
    output = []
    for index, base in enumerate(eligible, 1):
        key = (base["question_id"], base["language"])
        if key in completed and base["question_id"] not in (rerun_ids or set()):
            output.append(completed[key])
            continue
        try:
            result = answer_question(base["question"], use_answer_bank=False)
            try:
                import psutil
                process_memory = psutil.Process().memory_info()
                rss_bytes = process_memory.rss
                peak_rss_bytes = getattr(process_memory, "peak_wset", rss_bytes)
                swap_used_bytes = psutil.swap_memory().used
            except ImportError:
                rss_bytes = peak_rss_bytes = swap_used_bytes = ""
            original = result.get("initial_top3", [])
            fallback = result.get("fallback_top3", [])
            output.append({
                "question_id": base["question_id"], "language": base["language"],
                "original_query": base["question"], "initial_status": result.get("initial_support_status"),
                "initial_top1": packed(original, 1), "initial_top3": packed(original, 3),
                "detected_entity": result.get("requested_entity"), "requested_field": result.get("requested_field"),
                "fallback_triggered": result.get("fallback_triggered"),
                "rewrite_method": result.get("rewrite_method"),
                "canonical_retrieval_query": result.get("canonical_retrieval_query"),
                "rewrite_validation": result.get("rewrite_validation"),
                "fallback_top1": packed(fallback, 1), "fallback_top3": packed(fallback, 3),
                "merged_support_status": result.get("merged_support_status"),
                "crosslingual_gate_reason": result.get("crosslingual_gate_reason"),
                "final_answer_status": result.get("final_status"), "final_answer": result.get("answer"),
                "source": result.get("source"), "page": result.get("page"),
                "supporting_evidence": result.get("supporting_excerpt"),
                "response_language": result.get("response_language"),
                "retrieval_seconds": result.get("latency_seconds", {}).get("retrieval"),
                "initial_retrieval_seconds": result.get("initial_retrieval_seconds"),
                "rewrite_seconds": result.get("rewrite_seconds"),
                "canonical_retrieval_seconds": result.get("canonical_retrieval_seconds"),
                "multi_query_fusion_seconds": result.get("multi_query_fusion_seconds"),
                "fallback_seconds": result.get("fallback_seconds"),
                "total_seconds": result.get("latency_seconds", {}).get("total"),
                "rss_bytes": rss_bytes, "peak_rss_bytes": peak_rss_bytes,
                "swap_used_bytes": swap_used_bytes,
                "error": result.get("generation_error", ""),
            })
        except Exception as exc:
            output.append({"question_id": base["question_id"], "language": base["language"],
                           "original_query": base["question"], "initial_status": "ERROR",
                           "final_answer_status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
        snapshot = dict(completed)
        snapshot.update({(x["question_id"], x["language"]): x for x in output})
        write_rows(RECOVERY, [snapshot[(x["question_id"], x["language"])]
                              for x in eligible if (x["question_id"], x["language"]) in snapshot])
        print(f"subset {index}/{len(eligible)} {key} -> {output[-1].get('final_answer_status')}", flush=True)

    # Refresh the complete ordered checkpoint after a targeted rerun; a mid-run
    # write must not leave previously completed rows beyond the rerun point out.
    write_rows(RECOVERY, output)

    recovery = {(x["question_id"], x["language"]): x for x in output}
    by_id: dict[str, dict[str, dict[str, str]]] = {}
    for row in baseline:
        by_id.setdefault(row["question_id"], {})[row["language"]] = row
    parity = []
    for qid, trio in by_id.items():
        before = {lang: row.get("support_status") == "supported" for lang, row in trio.items()}
        after = dict(before)
        for lang in ("bangla", "banglish"):
            changed = recovery.get((qid, lang))
            if changed:
                after[lang] = changed.get("merged_support_status") == "supported"
        parity.append({"question_id": qid, "before_category": parity_label(before),
                       "english_status": trio["english"].get("support_status"),
                       "bangla_status": "supported" if after["bangla"] else trio["bangla"].get("support_status"),
                       "banglish_status": "supported" if after["banglish"] else trio["banglish"].get("support_status"),
                       "after_category": parity_label(after, after=True)})
    write_rows(PARITY_AFTER, parity)
    print("subset statuses:", dict(Counter((x["language"], x.get("final_answer_status")) for x in output)))
    print("parity after:", dict(Counter(x["after_category"] for x in parity)))


def retrieval_metrics(refresh: bool = False) -> None:
    """Offline paired retrieval metrics; expected pages never enter runtime."""
    import re
    from src.crosslingual import Rewrite, crosslingual_assessment, fuse_queries
    from src.evidence import FIELD_PATTERNS

    baseline = rows(BASELINE)
    dataset = {x["question_id"]: x for x in rows(DATASET)}
    recovery = {(x["question_id"], x["language"]): x for x in rows(RECOVERY)}
    completed = {(x["question_id"], x["language"]): x for x in rows(METRICS)} if METRICS.exists() and not refresh else {}
    retriever = Retriever(get_embedding_model(), top_k=30, reranker_enabled=False)
    output = []

    def measures(question: str, candidates: list[dict], expected_page: str) -> dict[str, object]:
        request = analyze_query(question)
        pages = [str(x.get("page")) == expected_page for x in candidates[:3]]
        entity = request.primary_entity
        if entity:
            entity_compact = re.sub(r"[^a-z0-9]", "", entity.casefold())
            entity_hits = [entity_compact in re.sub(r"[^a-z0-9]", "", str(x.get("text", "")).casefold())
                           for x in candidates[:3]]
        else:
            entity_hits = []
        field = request.requested_field
        patterns = FIELD_PATTERNS.get(field, ()) if field != "general" else ()
        field_hits = [any(re.search(p, str(x.get("text", "")), re.I) for p in patterns)
                      or field in (x.get("field_types") or []) for x in candidates[:3]] if patterns else []
        return {
            "page_hit_1": bool(pages[:1] and pages[0]), "page_hit_3": any(pages),
            "mrr_3": next((1 / rank for rank, hit in enumerate(pages, 1) if hit), 0),
            "entity_applicable": bool(entity),
            "entity_hit_1": bool(entity_hits[:1] and entity_hits[0]),
            "entity_hit_3": any(entity_hits),
            "field_applicable": bool(patterns),
            "field_hit_1": bool(field_hits[:1] and field_hits[0]),
            "field_hit_3": any(field_hits),
            "supportable_1": support(question, candidates, 1),
            "supportable_3": support(question, candidates, 3),
        }

    for index, base in enumerate(baseline, 1):
        key = (base["question_id"], base["language"])
        if key in completed:
            output.append(completed[key])
            continue
        question = base["question"]
        original = retriever.retrieve(question)
        after = original[:3]
        accepted_rewrite = None
        recovered = recovery.get(key)
        if recovered and recovered.get("canonical_retrieval_query"):
            canonical = retriever.retrieve(recovered["canonical_retrieval_query"])
            merged = fuse_queries(original, canonical, 3)
            rewrite = Rewrite(recovered["canonical_retrieval_query"],
                              recovered.get("rewrite_method", "NONE"), "VALIDATED")
            assessed, _ = crosslingual_assessment(question, rewrite, merged)
            if assessed.status.value == "supported":
                after = merged
                accepted_rewrite = rewrite
        expected = dataset[base["question_id"]].get("expected_page", "")
        before_metrics = measures(question, original[:3], expected)
        after_metrics = measures(question, after, expected)
        if accepted_rewrite:
            after_metrics["supportable_1"] = crosslingual_assessment(question, accepted_rewrite, after[:1])[0].status.value == "supported"
            after_metrics["supportable_3"] = crosslingual_assessment(question, accepted_rewrite, after[:3])[0].status.value == "supported"
        output.append({"question_id": base["question_id"], "language": base["language"],
                       "expected_page_offline": expected,
                       **{f"before_{name}": value for name, value in before_metrics.items()},
                       **{f"after_{name}": value for name, value in after_metrics.items()}})
        if index % 20 == 0:
            write_rows(METRICS, output)
            print(f"retrieval metrics {index}/{len(baseline)}", flush=True)
    write_rows(METRICS, output)
    for language in ("english", "bangla", "banglish"):
        subset = [x for x in output if x["language"] == language]
        print(language, "page_hit_3 before/after",
              sum(str(x["before_page_hit_3"]).lower() == "true" for x in subset),
              sum(str(x["after_page_hit_3"]).lower() == "true" for x in subset))


def human_review() -> None:
    baseline = {(x["question_id"], x["language"]): x for x in rows(BASELINE)}
    output = []
    for item in rows(RECOVERY):
        if item.get("merged_support_status") != "supported":
            continue
        before = baseline[(item["question_id"], item["language"])]
        output.append({
            "question_id": item["question_id"], "question": item["original_query"],
            "language": item["language"], "before_status": before["final_status"],
            "after_status": item["final_answer_status"],
            "canonical_retrieval_query": item["canonical_retrieval_query"],
            "supporting_evidence": item["supporting_evidence"],
            "final_answer": item["final_answer"],
            "correctness": "", "relevance": "", "groundedness": "",
            "naturalness": "", "semantic_consistency": "",
            "overall_acceptability": "", "review_notes": "",
        })
    write_rows(REVIEW, output)
    print(f"human review rows: {len(output)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["diagnose", "subset", "metrics", "review"])
    parser.add_argument("--rerun", action="append", default=[], help="Question ID to refresh in subset")
    parser.add_argument("--refresh", action="store_true", help="Recompute retrieval metrics")
    arguments = parser.parse_args()
    if arguments.command == "diagnose":
        diagnose()
    elif arguments.command == "subset":
        refresh_ids = ({x["question_id"] for x in rows(BASELINE)} if arguments.refresh
                       else set(arguments.rerun))
        failure_subset(refresh_ids)
    elif arguments.command == "metrics":
        retrieval_metrics(arguments.refresh)
    elif arguments.command == "review":
        human_review()
