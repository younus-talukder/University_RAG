"""Offline benchmark measurements; all answer scores are labeled proxies."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.evaluation.manifest import atomic_json

ERROR_TAXONOMY = (
    "RETRIEVAL_MISS", "RANKING_MISS", "ENTITY_DETECTION", "FIELD_DETECTION",
    "RELATION_DETECTION", "EVIDENCE_VALIDATION", "CONFLICT_HANDLING",
    "AMBIGUITY_HANDLING", "CANONICAL_GENERATION", "LANGUAGE_REALIZATION",
    "GROUNDING_REJECTION", "SEMANTIC_REJECTION", "LANGUAGE_REJECTION",
    "UNSAFE_RETURN", "ANNOTATION_ERROR", "SOURCE_CONFLICT", "SYSTEM_ERROR", "UNKNOWN",
)
ABSTENTION_STATUSES = (
    "INSUFFICIENT_EVIDENCE", "RETRIEVAL_UNCERTAIN", "AMBIGUOUS_QUERY",
    "CONFLICTING_EVIDENCE", "GENERATION_REJECTED", "OUT_OF_DOMAIN", "SYSTEM_ERROR",
)


def truth(value: Any) -> bool:
    return value is True or str(value).casefold() == "true"


def applicable_truth(value: Any) -> bool:
    return value is None or value == "" or truth(value)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, Any]:
    if total < 1:
        return {"value": "NOT_AVAILABLE", "n": 0, "lower": "NOT_AVAILABLE", "upper": "NOT_AVAILABLE"}
    rate = successes / total
    z2 = z * z
    center = (rate + z2 / (2 * total)) / (1 + z2 / total)
    margin = z * math.sqrt(rate * (1 - rate) / total + z2 / (4 * total * total)) / (1 + z2 / total)
    return {"value": rate, "n": total, "lower": max(0, center - margin),
            "upper": min(1, center + margin), "method": "Wilson 95%"}


def _source_match(candidate: dict[str, Any], expected: list[str]) -> bool:
    if not expected:
        return False
    values = {str(candidate.get("source") or "").replace("\\", "/").casefold(),
              str(candidate.get("relative_path") or "").replace("\\", "/").casefold()}
    values |= {Path(value).name.casefold() for value in values}
    for label in expected:
        clean = str(label).replace("\\", "/").casefold()
        if clean in values or Path(clean).name in values:
            return True
    return False


def _page_match(candidate: dict[str, Any], expected: list[str]) -> bool:
    return bool(expected) and str(candidate.get("page") or "") in expected


def retrieval_scores(record: dict[str, Any]) -> dict[str, Any]:
    candidates = record.get("retrieval_top3") or []
    sources = record.get("expected_sources") or []
    pages = record.get("expected_pages") or []
    pairs = record.get("expected_evidence") or []
    labeled = bool(sources or pages or pairs)
    def hit(item: dict[str, Any]) -> bool:
        if pairs:
            return any(_source_match(item, [pair["source"]]) and _page_match(item, [pair["page"]])
                       for pair in pairs)
        return (not sources or _source_match(item, sources)) and (not pages or _page_match(item, pages))
    matches = [hit(candidate) for candidate in candidates] if labeled else []
    output: dict[str, Any] = {
        "hit_1": bool(matches[:1] and matches[0]) if labeled else "NOT_AVAILABLE",
        "hit_3": any(matches[:3]) if labeled else "NOT_AVAILABLE",
        "mrr_3": next((1 / rank for rank, matched in enumerate(matches[:3], 1) if matched), 0.0)
                 if labeled else "NOT_AVAILABLE",
        "source_accuracy": _source_match(candidates[0], sources) if sources and candidates else
                           (False if sources else "NOT_AVAILABLE"),
        "page_accuracy": _page_match(candidates[0], pages) if pages and candidates else
                         (False if pages else "NOT_AVAILABLE"),
    }
    try:
        from src.evidence import FIELD_PATTERNS, analyze_query, assess_evidence
        request = analyze_query(record["question"])
        entity = request.primary_entity
        if entity:
            compact = re.sub(r"[^a-z0-9]", "", entity.casefold())
            entity_matches = [compact in re.sub(r"[^a-z0-9]", "", str(item.get("text") or "").casefold())
                              for item in candidates[:3]]
            output["entity_hit_1"] = bool(entity_matches[:1] and entity_matches[0])
            output["entity_hit_3"] = any(entity_matches)
        else:
            output["entity_hit_1"] = output["entity_hit_3"] = "NOT_AVAILABLE"
        patterns = FIELD_PATTERNS.get(request.requested_field, ()) if request.requested_field != "general" else ()
        if patterns:
            field_matches = [any(re.search(pattern, str(item.get("text") or ""), re.I) for pattern in patterns)
                             or request.requested_field in (item.get("field_types") or [])
                             for item in candidates[:3]]
            output["field_hit_1"] = bool(field_matches[:1] and field_matches[0])
            output["field_hit_3"] = any(field_matches)
        else:
            output["field_hit_1"] = output["field_hit_3"] = "NOT_AVAILABLE"
        output["supportable_1"] = assess_evidence(record["question"], candidates[:1]).status.value == "supported"
        output["supportable_3"] = assess_evidence(record["question"], candidates[:3]).status.value == "supported"
    except (ImportError, AttributeError, ValueError, TypeError):
        for key in ("entity_hit_1", "entity_hit_3", "field_hit_1", "field_hit_3",
                    "supportable_1", "supportable_3"):
            output[key] = "NOT_AVAILABLE"
    return output


_WORD = re.compile(r"[A-Za-z]+\d*|\d+|[\u0980-\u09ff]+", re.UNICODE)
_IDENTIFIER = re.compile(r"\b[A-Z]{2,6}\s*\(?[A-Z]{0,5}\)?\s*\d{3,4}\b")


def answer_proxies(record: dict[str, Any]) -> dict[str, Any]:
    """Reference overlap, not factual correctness or a hallucination detector."""
    if (not record.get("answer_returned") or not record.get("reference_answer") or
            record.get("annotation_status") not in {"verified", "unlabeled"}):
        return {key: "NOT_AVAILABLE" for key in (
            "normalized_exact_match", "token_precision", "token_recall", "token_f1",
            "numeric_preservation", "entity_preservation", "reference_token_overlap")}
    reference = str(record["reference_answer"])
    answer = str(record.get("final_answer") or "")
    ref_tokens = [token.casefold() for token in _WORD.findall(reference)]
    ans_tokens = [token.casefold() for token in _WORD.findall(answer)]
    ref_set, ans_set = set(ref_tokens), set(ans_tokens)
    overlap = len(ref_set & ans_set)
    precision = overlap / len(ans_set) if ans_set else 0.0
    recall = overlap / len(ref_set) if ref_set else 0.0
    ref_numbers = set(re.findall(r"\d+(?:\.\d+)?", reference))
    ans_numbers = set(re.findall(r"\d+(?:\.\d+)?", answer))
    identifiers = {re.sub(r"\s+", "", value).casefold() for value in _IDENTIFIER.findall(reference)}
    answer_compact = re.sub(r"\s+", "", answer).casefold()
    return {
        "normalized_exact_match": ref_tokens == ans_tokens,
        "token_precision": precision, "token_recall": recall,
        "token_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "numeric_preservation": ref_numbers <= ans_numbers if ref_numbers else "NOT_AVAILABLE",
        "entity_preservation": all(value in answer_compact for value in identifiers)
                               if identifiers else "NOT_AVAILABLE",
        "reference_token_overlap": recall,
    }


def safe_return(record: dict[str, Any]) -> bool:
    if not record.get("answer_returned"):
        return False
    return (truth(record.get("grounding_status")) and
            applicable_truth(record.get("semantic_status")) and
            truth(record.get("language_status")) and
            not record.get("answer_safety_failures"))


def classify_error(record: dict[str, Any], retrieval: dict[str, Any]) -> str:
    """Conservative post-inference taxonomy; UNKNOWN means no diagnosis proven."""
    status = record.get("answerability_status")
    annotation = record.get("annotation_status")
    if record.get("error") or status == "SYSTEM_ERROR":
        return "SYSTEM_ERROR"
    if record.get("answer_returned") and not safe_return(record):
        return "UNSAFE_RETURN"
    if annotation in {"needs_review", "ambiguous_reference"}:
        return "ANNOTATION_ERROR"
    if annotation == "source_conflict" or status == "CONFLICTING_EVIDENCE":
        return "SOURCE_CONFLICT"
    if status == "AMBIGUOUS_QUERY":
        return "AMBIGUITY_HANDLING"
    if status == "GENERATION_REJECTED":
        if int(record.get("generation_attempts") or 0) == 0:
            # Generation-disabled framework checks have no generator failure to diagnose.
            return "UNKNOWN"
        if record.get("language_status") is False:
            return "LANGUAGE_REJECTION"
        if record.get("grounding_status") is False:
            return "GROUNDING_REJECTION"
        if record.get("semantic_status") is False:
            return "SEMANTIC_REJECTION"
        return "CANONICAL_GENERATION"
    if status in {"INSUFFICIENT_EVIDENCE", "RETRIEVAL_UNCERTAIN"}:
        if retrieval.get("hit_3") is False:
            return "RETRIEVAL_MISS"
        if retrieval.get("hit_1") is False and retrieval.get("hit_3") is True:
            return "RANKING_MISS"
        return "UNKNOWN"
    return "UNKNOWN"


def _numeric(values: Iterable[Any]) -> list[float]:
    output = []
    for value in values:
        try:
            output.append(float(value))
        except (TypeError, ValueError):
            pass
    return output


def latency_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = sorted(_numeric(row.get(field) for row in rows))
    if not values:
        return {"n": 0, "mean": "NOT_AVAILABLE", "median": "NOT_AVAILABLE", "p95": "NOT_AVAILABLE"}
    index = math.ceil(0.95 * len(values)) - 1
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "p95": values[index]}


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    safe = sum(bool(row["safe_return"]) for row in rows)
    abstentions = Counter(row["answerability_status"] for row in rows if not row.get("answer_returned"))
    def mean_available(key: str) -> Any:
        values = _numeric(row.get(key) for row in rows if row.get(key) != "NOT_AVAILABLE")
        return statistics.mean(values) if values else "NOT_AVAILABLE"
    retrieval_proportions = {}
    for key in ("hit_1", "hit_3", "source_accuracy", "page_accuracy",
                "entity_hit_1", "entity_hit_3", "field_hit_1", "field_hit_3",
                "supportable_1", "supportable_3"):
        values = [row[key] for row in rows if row.get(key) != "NOT_AVAILABLE"]
        retrieval_proportions[key] = wilson_interval(sum(truth(value) for value in values), len(values))
    return {
        "n": n, "safe_answer_coverage": wilson_interval(safe, n),
        "abstention_rate": wilson_interval(sum(abstentions.values()), n),
        "unsafe_returned_answer_count": sum(row.get("answer_returned") and not row["safe_return"] for row in rows),
        "answerability_status_distribution": dict(Counter(row["answerability_status"] for row in rows)),
        "abstention_status_distribution": {key: abstentions[key] for key in ABSTENTION_STATUSES},
        "language_consistency": wilson_interval(
            sum(truth(row.get("language_consistency")) for row in rows), n),
        "trust_gate_pass_rates": {
            key: wilson_interval(sum(truth(row.get(key)) for row in rows if row.get(key) is not None),
                                 sum(row.get(key) is not None for row in rows))
            for key in ("grounding_status", "semantic_status", "language_status")},
        "retrieval": {key: mean_available(key) for key in (
            "hit_1", "hit_3", "mrr_3", "source_accuracy", "page_accuracy",
            "entity_hit_1", "entity_hit_3", "field_hit_1", "field_hit_3",
            "supportable_1", "supportable_3")},
        "retrieval_proportion_intervals": retrieval_proportions,
        "automatic_answer_proxies": {key: mean_available(key) for key in (
            "normalized_exact_match", "token_precision", "token_recall", "token_f1",
            "numeric_preservation", "entity_preservation", "reference_token_overlap")},
        "latency_seconds": {key: latency_summary(rows, key) for key in (
            "latency_seconds", "retrieval_seconds", "generation_seconds", "answerability_seconds")},
        "generation_attempts": sum(int(row.get("generation_attempts") or 0) for row in rows),
        "generation_required_cases": sum(row.get("requested_answer_strategy") == "gguf_generation"
                                         for row in rows),
        "generation_questions": sum(truth(row.get("generation_used")) for row in rows),
        "generation_accepted": sum(truth(row.get("generation_used")) and row.get("safe_return") for row in rows),
        "generation_rejected": sum(int(row.get("generation_attempts") or 0) > 0 and
                                   row.get("answerability_status") == "GENERATION_REJECTED" for row in rows),
        "retries": sum(truth(row.get("retry_used")) for row in rows),
        "successful_retries": sum(truth(row.get("retry_used")) and row.get("safe_return") for row in rows),
        "failed_retries": sum(truth(row.get("retry_used")) and not row.get("safe_return") for row in rows),
    }


def _parity(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_base: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_base[row["base_question_id"]][row["language"]] = row
    parity = []
    counts: Counter[str] = Counter()
    categories = {
        frozenset(("english", "bangla", "banglish")): "ALL_THREE_SUPPORTED",
        frozenset(("english",)): "ENGLISH_ONLY",
        frozenset(("bangla",)): "BANGLA_ONLY",
        frozenset(("banglish",)): "BANGLISH_ONLY",
        frozenset(("english", "bangla")): "ENGLISH_AND_BANGLA",
        frozenset(("english", "banglish")): "ENGLISH_AND_BANGLISH",
        frozenset(("bangla", "banglish")): "BANGLA_AND_BANGLISH",
        frozenset(): "NONE_SUPPORTED",
    }
    for base_id, languages in sorted(by_base.items()):
        supported = frozenset(lang for lang, row in languages.items()
                              if row["answerability_status"] == "SUPPORTED")
        category = categories.get(supported, "OTHER_MISMATCH") if len(languages) == 3 else "OTHER_MISMATCH"
        counts[category] += 1
        evidence = {(str(row.get("document_id")), str(row.get("page"))) for row in languages.values()
                    if row["safe_return"]}
        values = {tuple(sorted(re.findall(r"\d+(?:\.\d+)?", str(row.get("final_answer") or ""))))
                  for row in languages.values() if row["safe_return"]}
        parity.append({
            "base_question_id": base_id, "category": category,
            **{f"{lang}_status": languages.get(lang, {}).get("answerability_status", "NOT_AVAILABLE")
               for lang in ("english", "bangla", "banglish")},
            "equivalent_primary_evidence": len(evidence) == 1 if len(supported) == 3 else "NOT_AVAILABLE",
            "equivalent_numeric_values": len(values) == 1 if len(supported) == 3 and
                                         any(values) else "NOT_AVAILABLE",
            "equivalent_extracted_values": (
                len({tuple(sorted(str(value).casefold() for value in row.get("extracted_values") or []))
                     for row in languages.values()}) == 1
                if len(supported) == 3 and all(row.get("extracted_values") for row in languages.values())
                else "NOT_AVAILABLE"),
        })
    return parity, dict(counts)


def _flat_group(group: str, value: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        group: value, "n": summary["n"],
        "safe_answer_coverage": summary["safe_answer_coverage"]["value"],
        "abstention_rate": summary["abstention_rate"]["value"],
        "unsafe_returned_answer_count": summary["unsafe_returned_answer_count"],
        **summary["retrieval"],
        "latency_mean_seconds": summary["latency_seconds"]["latency_seconds"]["mean"],
        "latency_p95_seconds": summary["latency_seconds"]["latency_seconds"]["p95"],
    }


def analyze_run(run_dir: str | Path, *, human_sample_size: int = 30, seed: int = 42) -> dict[str, Any]:
    from src.evaluation.runner import _atomic_csv, checkpoint_records
    from src.evaluation.human_review import create_sample

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    error_rows = []
    proxy_rows = []
    for record in checkpoint_records(run_dir, manifest["compatibility_sha256"]):
        retrieval = retrieval_scores(record)
        proxies = answer_proxies(record)
        light = {key: value for key, value in record.items() if key != "retrieval_top3"}
        light.update(retrieval)
        light.update(proxies)
        light["safe_return"] = safe_return(record)
        light["error_category"] = classify_error(record, retrieval)
        rows.append(light)
        proxy_rows.append({"evaluation_id": light["evaluation_id"], "language": light["language"],
                           "answer_returned": light["answer_returned"],
                           "reference_available": light["reference_available"], **proxies})
        if light["error_category"] != "UNKNOWN" or light["answerability_status"] != "SUPPORTED":
            error_rows.append({key: light.get(key, "") for key in (
                "evaluation_id", "language", "answerability_status", "answerability_reason",
                "error_category", "error")})
    expected_ids = set(json.loads((run_dir / "expected_evaluation_ids.json").read_text(encoding="utf-8")))
    expected = manifest["expected_evaluation_rows"]
    ids = [row["evaluation_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate evaluation IDs in checkpoint")
    if not set(ids) <= expected_ids:
        raise ValueError("checkpoint contains unexpected evaluation IDs")
    run_complete = set(ids) == expected_ids and len(ids) == expected and manifest.get("run_complete") is True
    overall = group_summary(rows)
    groups = {}
    for key in ("language", "intent", "difficulty", "course_code", "answer_strategy"):
        bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get(key):
                bucket[str(row[key])].append(row)
        groups[key] = {label: group_summary(items) for label, items in sorted(bucket.items())
                       if key != "course_code" or len(items) >= 5}
    parity, parity_counts = _parity(rows)
    elapsed = sum(_numeric(row.get("latency_seconds") for row in rows))
    generated = [row for row in rows if truth(row.get("generation_used"))]
    retrieval_only = [row for row in rows if not truth(row.get("generation_used"))]
    summary = {
        "run_id": manifest["run_id"], "run_classification": manifest["run_classification"],
        "run_mode": run_config["run_mode"],
        "run_complete": run_complete, "completed_rows": len(rows), "expected_rows": expected,
        "overall": overall, "by_group": groups,
        "multilingual_parity": parity_counts, "error_taxonomy": list(ERROR_TAXONOMY),
        "error_category_distribution": dict(Counter(row["error_category"] for row in rows)),
        "throughput_questions_per_hour": len(rows) / elapsed * 3600 if elapsed else "NOT_AVAILABLE",
        "generation_questions_per_hour": len(generated) /
            sum(_numeric(row.get("latency_seconds") for row in generated)) * 3600
            if generated and sum(_numeric(row.get("latency_seconds") for row in generated)) else "NOT_AVAILABLE",
        "retrieval_only_questions_per_hour": len(retrieval_only) /
            sum(_numeric(row.get("latency_seconds") for row in retrieval_only)) * 3600
            if retrieval_only and sum(_numeric(row.get("latency_seconds") for row in retrieval_only)) else "NOT_AVAILABLE",
        "estimated_6000_same_mode_hours": 6000 / (len(rows) / elapsed * 3600)
            if rows and elapsed else "NOT_AVAILABLE",
        "estimated_6000_full_generation_hours": 6000 / (len(rows) / elapsed * 3600)
            if rows and elapsed and run_config["run_mode"] == "full" else "NOT_AVAILABLE",
        "estimate_warning": "ESTIMATE from warm per-question time in this run mode; excludes startup, corpus growth, and hardware differences. Skip-generation time cannot estimate Qwen inference.",
        "resource_observations": {
            key: max(_numeric(row.get(key) for row in rows), default="NOT_AVAILABLE")
            for key in ("rss_bytes", "peak_process_bytes", "pagefile_used_bytes_system")},
        "minimum_available_ram_bytes": min(_numeric(row.get("available_ram_bytes") for row in rows),
                                            default="NOT_AVAILABLE"),
        "resource_preflight": manifest.get("resource_preflight", {}),
        "memory_safety_stop": manifest.get("memory_safety_stop", False),
        "metric_language": "AUTOMATIC DEVELOPMENT / BENCHMARK PROXIES; not human correctness or thesis accuracy",
    }
    atomic_json(run_dir / "metrics_overall.json", summary)
    for key, filename in (("language", "metrics_by_language.csv"),
                          ("intent", "metrics_by_intent.csv"),
                          ("difficulty", "metrics_by_difficulty.csv"),
                          ("course_code", "metrics_by_entity.csv"),
                          ("answer_strategy", "metrics_by_strategy.csv")):
        flat = [_flat_group(key, name, value) for name, value in groups[key].items()]
        fields = list(flat[0]) if flat else [key, "n", "safe_answer_coverage", "abstention_rate"]
        _atomic_csv(run_dir / filename, fields, flat)
    _atomic_csv(run_dir / "multilingual_parity.csv", list(parity[0]) if parity else ["base_question_id", "category"], parity)
    _atomic_csv(run_dir / "error_analysis.csv",
                list(error_rows[0]) if error_rows else ["evaluation_id", "error_category"], error_rows)
    _atomic_csv(run_dir / "answer_proxy_metrics.csv", list(proxy_rows[0]) if proxy_rows else ["evaluation_id"], proxy_rows)
    _atomic_csv(run_dir / "retrieval_metrics_table.csv",
                ["language", "n", *overall["retrieval"].keys()],
                [{"language": name, "n": value["n"], **value["retrieval"]}
                 for name, value in groups["language"].items()])
    _atomic_csv(run_dir / "answerability_metrics_table.csv",
                ["language", "n", "safe_answer_coverage", "abstention_rate", "unsafe_returned_answer_count"],
                [{key: row[key] for key in ("language", "n", "safe_answer_coverage", "abstention_rate",
                                           "unsafe_returned_answer_count")}
                 for row in (_flat_group("language", name, value)
                             for name, value in groups["language"].items())])
    _atomic_csv(run_dir / "language_comparison_table.csv",
                ["language", "n", "safe_answer_coverage", "hit_1", "hit_3", "mrr_3"],
                [{key: row[key] for key in ("language", "n", "safe_answer_coverage", "hit_1", "hit_3", "mrr_3")}
                 for row in (_flat_group("language", name, value)
                             for name, value in groups["language"].items())])
    _atomic_csv(run_dir / "latency_table.csv", ["group", "n", "mean", "median", "p95"],
                [{"group": key, **value} for key, value in overall["latency_seconds"].items()])
    _atomic_csv(run_dir / "human_metrics_table.csv", ["status", "note"],
                [{"status": "NOT_AVAILABLE", "note": "No imported human review scores"}])
    create_sample(rows, run_dir / "human_review_sample.csv", sample_size=human_sample_size, seed=seed)
    def display_metric(value: Any) -> str:
        return f"{value:.3f}" if isinstance(value, (int, float)) else str(value)

    report = [
        f"# Benchmark run {manifest['run_id']}", "",
        f"Run classification: **{manifest['run_classification']}**", "",
        f"Run mode: **{run_config['run_mode']}**", "",
        f"Completion: {len(rows)}/{expected}; run_complete={str(run_complete).lower()}", "",
        "This run measures retrieval, safe answer coverage, abstentions, latency, and automatic answer proxies. "
        "It does not establish human correctness or final thesis accuracy.", "",
        *( ["Generation was disabled. `GENERATION_REJECTED` rows in this mode are abstentions, not failed Qwen attempts.", ""]
           if run_config["run_mode"] != "full" else [] ),
        f"Safe answer coverage: {display_metric(overall['safe_answer_coverage']['value'])} "
        f"(Wilson 95% CI {display_metric(overall['safe_answer_coverage']['lower'])}–"
        f"{display_metric(overall['safe_answer_coverage']['upper'])}; n={len(rows)})", "",
        f"Abstention rate: {overall['abstention_rate']['value']}", "",
        f"Known unsafe returned answers: {overall['unsafe_returned_answer_count']}", "",
        "## By language", "",
        "| Language | Rows | Safe answer coverage | Hit@1 | Hit@3 | MRR@3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *[f"| {language} | {value['n']} | {display_metric(value['safe_answer_coverage']['value'])} | "
          f"{display_metric(value['retrieval']['hit_1'])} | "
          f"{display_metric(value['retrieval']['hit_3'])} | "
          f"{display_metric(value['retrieval']['mrr_3'])} |"
          for language, value in groups["language"].items()], "",
        "## Retrieval", "",
        *[f"- {key}: {value}" for key, value in overall["retrieval"].items()], "",
        "## Status distribution", "",
        *[f"- {key}: {value}" for key, value in overall["answerability_status_distribution"].items()], "",
        "## Multilingual parity", "",
        *[f"- {key}: {value}" for key, value in parity_counts.items()], "",
        "## Latency and generation", "",
        f"Mean / median / P95 total latency: "
        f"{display_metric(overall['latency_seconds']['latency_seconds']['mean'])} / "
        f"{display_metric(overall['latency_seconds']['latency_seconds']['median'])} / "
        f"{display_metric(overall['latency_seconds']['latency_seconds']['p95'])} seconds.", "",
        f"Generation attempts: {overall['generation_attempts']}; retries: {overall['retries']}.", "",
        "## Post-inference error categories", "",
        *[f"- {key}: {value}" for key, value in summary["error_category_distribution"].items()], "",
        "## Reproducibility", "",
        f"Dataset SHA-256: {manifest['dataset']['dataset_sha256']}", "",
        f"Corpus fingerprint: {manifest['corpus_index']['corpus_fingerprint']}", "",
        f"Git commit: {manifest['code_version']['git_commit']}; "
        f"working_tree_dirty={str(manifest['code_version']['working_tree_dirty']).lower()}", "",
        "Human sample rows have blank rating fields. No human scores are inferred from automatic metrics.", "",
    ]
    from src.evaluation.manifest import atomic_bytes
    atomic_bytes(run_dir / "benchmark_report.md", "\n".join(report).encode("utf-8"))
    return summary
