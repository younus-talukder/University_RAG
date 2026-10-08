"""Deterministic, stratified review sampling and genuine-score aggregation."""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HUMAN_FIELDS = ("correctness", "relevance", "groundedness", "completeness",
                "naturalness", "semantic_consistency", "overall_acceptability", "review_notes")
SCORE_FIELDS = HUMAN_FIELDS[:5]
SAMPLE_FIELDS = ("review_id", "evaluation_id", "base_question_id", "language", "question",
                 "reference_answer", "final_answer", "answer_strategy", "answerability_status",
                 "intent", "difficulty", "cross_lingual_fallback_used", "source", "page",
                 "supporting_excerpt", "reviewer_id",
                 *HUMAN_FIELDS)


def _write_csv(path: Path, fields: tuple[str, ...] | list[str], rows: list[dict[str, Any]]) -> None:
    from src.evaluation.runner import _atomic_csv
    _atomic_csv(path, list(fields), rows)


def create_sample(rows: list[dict[str, Any]], output: Path, *, sample_size: int = 30,
                  seed: int = 42, blinded: bool = False,
                  blinding_key: Path | None = None) -> list[dict[str, Any]]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if blinded and blinding_key is None:
        raise ValueError("blinded sampling requires a separate blinding key path")
    if blinded and blinding_key.resolve() == output.resolve():
        raise ValueError("blinding key must not overwrite the review file")
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    sample_size = min(sample_size, len(pool))
    selected: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    language_counts = Counter(row.get("language", "") for row in pool)
    targets = {lang: sample_size // len(language_counts) for lang in language_counts}
    for lang in list(targets)[:sample_size % max(1, len(language_counts))]:
        targets[lang] += 1
    while pool and len(selected) < sample_size:
        best_index = 0
        best_score = float("-inf")
        for index, row in enumerate(pool):
            language = str(row.get("language") or "")
            score = 10 if counts[("language", language)] < targets.get(language, 0) else -10
            for field in ("answer_strategy", "answerability_status", "intent", "difficulty"):
                value = str(row.get(field) or "MISSING")
                score += 1 / (1 + counts[(field, value)])
            if row.get("cross_lingual_fallback_used"):
                score += 2 / (1 + counts[("cross_lingual_fallback_used", "true")])
            if score > best_score:
                best_index, best_score = index, score
        row = pool.pop(best_index)
        selected.append(row)
        for field in ("language", "answer_strategy", "answerability_status", "intent", "difficulty"):
            counts[(field, str(row.get(field) or "MISSING"))] += 1
        if row.get("cross_lingual_fallback_used"):
            counts[("cross_lingual_fallback_used", "true")] += 1
    if blinded:
        rng.shuffle(selected)
    output_rows = []
    key_rows = []
    for index, row in enumerate(selected, 1):
        review_id = f"R{index:05d}"
        item = {field: row.get(field, "") for field in SAMPLE_FIELDS}
        item["review_id"] = review_id
        item["reviewer_id"] = ""
        for field in HUMAN_FIELDS:
            item[field] = ""
        if blinded:
            system_id = str(row.get("system_id") or "")
            key_rows.append({"review_id": review_id, "evaluation_id": item["evaluation_id"],
                             "review_target_id": f"{system_id}::{item['evaluation_id']}" if system_id else item["evaluation_id"],
                             "system_id": system_id,
                             "answer_strategy": item["answer_strategy"],
                             "answerability_status": item["answerability_status"],
                             "language": item["language"]})
            for field in ("evaluation_id", "answer_strategy", "answerability_status"):
                item.pop(field, None)
        output_rows.append(item)
    fields = [field for field in SAMPLE_FIELDS if not blinded or field not in
              {"evaluation_id", "answer_strategy", "answerability_status"}]
    _write_csv(output, fields, output_rows)
    if blinded:
        assert blinding_key is not None
        _write_csv(blinding_key, ["review_id", "evaluation_id", "review_target_id", "system_id",
                                  "answer_strategy", "answerability_status", "language"], key_rows)
    return output_rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_scores(review_rows: list[dict[str, str]], valid_ids: set[str],
                    *, blinding_key: dict[str, str] | None = None) -> list[dict[str, str]]:
    output = []
    seen: set[tuple[str, str]] = set()
    for number, source in enumerate(review_rows, 2):
        row = dict(source)
        if blinding_key is not None:
            review_id = row.get("review_id", "")
            if review_id not in blinding_key:
                raise ValueError(f"row {number}: unknown review_id {review_id!r}")
            row["evaluation_id"] = blinding_key[review_id]
        evaluation_id = row.get("evaluation_id", "")
        if evaluation_id not in valid_ids:
            raise ValueError(f"row {number}: unexpected evaluation_id {evaluation_id!r}")
        reviewer_id = row.get("reviewer_id", "").strip() or "reviewer-1"
        key = evaluation_id, reviewer_id
        if key in seen:
            raise ValueError(f"row {number}: duplicate review by {reviewer_id!r} for {evaluation_id!r}")
        seen.add(key)
        row["reviewer_id"] = reviewer_id
        for field in SCORE_FIELDS:
            value = row.get(field, "").strip()
            if value and value not in {"1", "2", "3", "4", "5"}:
                raise ValueError(f"row {number}: {field} must be 1–5")
        semantic = row.get("semantic_consistency", "").strip().upper()
        if semantic and semantic not in {"PASS", "FAIL"}:
            raise ValueError(f"row {number}: semantic_consistency must be PASS or FAIL")
        row["semantic_consistency"] = semantic
        overall = row.get("overall_acceptability", "").strip().upper()
        if overall and overall not in {"ACCEPT", "PARTIAL", "REJECT"}:
            raise ValueError(f"row {number}: overall_acceptability must be ACCEPT, PARTIAL, or REJECT")
        row["overall_acceptability"] = overall
        output.append(row)
    return output


def _kappa(pairs: list[tuple[str, str]], *, ordinal: bool = False) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "percent_agreement": "NOT_AVAILABLE", "cohens_kappa": "NOT_AVAILABLE"}
    agreement = sum(a == b for a, b in pairs) / len(pairs)
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    labels = sorted(set(left) | set(right))
    n = len(pairs)
    if ordinal:
        weights = {(a, b): ((int(a) - int(b)) / 4) ** 2 for a in labels for b in labels}
        observed = sum(weights[a, b] for a, b in pairs) / n
        expected = sum(weights[a, b] * left[a] * right[b] / (n * n) for a in labels for b in labels)
        kappa = 1 - observed / expected if expected else (1.0 if observed == 0 else "NOT_AVAILABLE")
    else:
        expected = sum(left[label] * right[label] for label in labels) / (n * n)
        kappa = (agreement - expected) / (1 - expected) if expected < 1 else (1.0 if agreement == 1 else "NOT_AVAILABLE")
    return {"n": n, "percent_agreement": agreement, "cohens_kappa": kappa,
            "method": "quadratic-weighted" if ordinal else "nominal"}


def agreement_statistics(rows: list[dict[str, str]]) -> dict[str, Any]:
    reviewers = {row["reviewer_id"] for row in rows}
    if len(reviewers) < 2:
        return {"available": False, "reason": "fewer than two reviewers scored the same rows"}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["evaluation_id"]].append(row)
    shared = [items for items in grouped.values() if len(items) >= 2]
    if not shared:
        return {"available": False, "reason": "no rows scored by at least two reviewers"}
    output = {"available": True, "shared_evaluation_rows": len(shared), "fields": {}}
    for field in (*SCORE_FIELDS, "semantic_consistency", "overall_acceptability"):
        pairs = []
        for items in shared:
            values = [item.get(field, "") for item in sorted(items, key=lambda x: x["reviewer_id"])
                      if item.get(field, "")]
            for left in range(len(values)):
                for right in range(left + 1, len(values)):
                    pairs.append((values[left], values[right]))
        output["fields"][field] = _kappa(pairs, ordinal=field in SCORE_FIELDS)
    return output


def human_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "status": "NOT_AVAILABLE"}
    result: dict[str, Any] = {"n": len(rows), "status": "HUMAN-REVIEW SAMPLE RESULT"}
    for field in SCORE_FIELDS:
        scores = [int(row[field]) for row in rows if row.get(field)]
        result[f"mean_{field}"] = statistics.mean(scores) if scores else "NOT_AVAILABLE"
        result[f"{field}_n"] = len(scores)
    semantic = [row["semantic_consistency"] for row in rows if row.get("semantic_consistency")]
    result["semantic_consistency_pass_rate"] = semantic.count("PASS") / len(semantic) if semantic else "NOT_AVAILABLE"
    result["semantic_consistency_n"] = len(semantic)
    accept = [row["overall_acceptability"] for row in rows if row.get("overall_acceptability")]
    from src.evaluation.metrics import wilson_interval
    for label in ("ACCEPT", "PARTIAL", "REJECT"):
        result[f"{label.casefold()}_rate"] = accept.count(label) / len(accept) if accept else "NOT_AVAILABLE"
        result[f"{label.casefold()}_rate_interval"] = wilson_interval(accept.count(label), len(accept))
    result["acceptability_n"] = len(accept)
    return result


def import_scores(results_path: Path | list[Path], reviews_path: Path, output_dir: Path,
                  *, blinding_key_path: Path | None = None) -> dict[str, Any]:
    paths = [results_path] if isinstance(results_path, Path) else list(results_path)
    result_rows = []
    by_id = {}
    for path in paths:
        for row in _read_csv(path):
            target = f"{path.stem}::{row['evaluation_id']}" if len(paths) > 1 else row["evaluation_id"]
            if target in by_id:
                raise ValueError("results file contains duplicate evaluation IDs")
            by_id[target] = row
            result_rows.append(row)
    key = None
    if blinding_key_path:
        key_rows = _read_csv(blinding_key_path)
        key = {row["review_id"]: (row.get("review_target_id") or row["evaluation_id"])
               for row in key_rows}
        if len(key) != len(key_rows):
            raise ValueError("blinding key contains duplicate review IDs")
    scores = validate_scores(_read_csv(reviews_path), set(by_id), blinding_key=key)
    from src.evaluation.runner import _atomic_csv
    fields = list(dict.fromkeys([*SAMPLE_FIELDS, *[name for row in scores for name in row]]))
    _atomic_csv(output_dir / "human_review_scores.csv", fields, scores)
    groups = {}
    for field in ("language", "answer_strategy", "intent", "difficulty"):
        buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
        for score in scores:
            label = by_id[score["evaluation_id"]].get(field, "")
            if label:
                buckets[label].append(score)
        groups[field] = {label: human_metrics(items) for label, items in buckets.items() if len(items) >= 3}
    report = {
        "label": "HUMAN-REVIEW SAMPLE RESULT; not overall benchmark accuracy",
        "overall": human_metrics(scores), "by_group": groups,
        "inter_rater_agreement": agreement_statistics(scores),
    }
    from src.evaluation.manifest import atomic_json
    atomic_json(output_dir / "human_metrics.json", report)
    _atomic_csv(output_dir / "human_metrics_table.csv", ["group", "n", "mean_correctness",
                "mean_relevance", "mean_groundedness", "mean_completeness", "mean_naturalness",
                "semantic_consistency_pass_rate", "accept_rate", "partial_rate", "reject_rate"],
                [{"group": "overall", **report["overall"]}])
    return report
