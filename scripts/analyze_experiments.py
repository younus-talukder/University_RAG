from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "data" / "questions" / "questions.csv"
RESULT_FILES = {
    "Fast": ROOT_DIR / "results" / "experiment_a_fast_answer_bank_off.csv",
    "GGUF": ROOT_DIR / "results" / "experiment_b_gguf_answer_bank_off.csv",
}
SUMMARY_PATH = ROOT_DIR / "results" / "experiment_metrics_summary.md"
OVERALL_PATH = ROOT_DIR / "results" / "experiment_metrics_overall.csv"
BREAKDOWN_PATH = ROOT_DIR / "results" / "experiment_metrics_breakdown.csv"
PER_QUERY_PATH = ROOT_DIR / "results" / "experiment_metrics_per_query.csv"


LANGUAGE_REFERENCE_COLUMN = {
    "english": "ground_truth_answer_en",
    "bangla": "ground_truth_answer_bn",
    "banglish": "ground_truth_answer_banglish",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def truthy(value: Any) -> bool:
    return str(value).strip().casefold() == "true"


def parse_expected_pages(value: str) -> set[str]:
    pages = set()
    for part in re.split(r"[;,]", str(value)):
        cleaned = part.strip()
        if cleaned:
            pages.add(cleaned)
    return pages


def retrieved_slots(row: dict[str, str], k: int = 3) -> list[tuple[str, str]]:
    slots = []
    for index in range(1, k + 1):
        source = row.get(f"retrieved_source_{index}", "").strip()
        page = row.get(f"retrieved_page_{index}", "").strip()
        if source or page:
            slots.append((source, page))
    return slots


def normalize_tokens(text: str) -> set[str]:
    normalized = str(text).casefold()
    replacements = [
        (r"\bholo\b", "is"),
        (r"\bebong\b", "and"),
        (r"\bsathe\b", "with"),
        (r"\bo\b", "and"),
        (r"\bache\b", "included"),
        (r"\bbojha\b", "understanding"),
        (r"\bdescribe kora\b", "describing"),
        (r"\bdesign kora\b", "designing"),
        (r"\bdevelop kora\b", "developing"),
        (r"\buse kore\b", "using"),
        (r"\bcalculate kora\b", "calculating"),
        (r"\bporano hoy\b", "taught"),
    ]
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"(\w+)-(?:er|e|te|der|gulo)\b", r"\1", normalized)
    normalized = re.sub(r"[^A-Za-z0-9\u0980-\u09ff]+", " ", normalized)
    stopwords = {
        "what", "which", "where", "when", "how", "the", "and", "are", "is", "was",
        "for", "with", "this", "that", "course", "কী", "কি", "এর", "এবং",
    }
    return {token for token in normalized.split() if len(token) > 1 and token not in stopwords}


def token_recall(reference: str, generated: str) -> float:
    reference_tokens = normalize_tokens(reference)
    generated_tokens = normalize_tokens(generated)
    if not reference_tokens:
        return 0.0
    return len(reference_tokens & generated_tokens) / len(reference_tokens)


def token_precision(reference: str, generated: str) -> float:
    reference_tokens = normalize_tokens(reference)
    generated_tokens = normalize_tokens(generated)
    if not generated_tokens:
        return 0.0
    return len(reference_tokens & generated_tokens) / len(generated_tokens)


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil(0.95 * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def avg_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row[key]) / len(rows)


def avg_float(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return mean(values) if values else 0.0


def build_joined_rows() -> list[dict[str, Any]]:
    dataset = {row["question_id"]: row for row in read_csv(DATASET_PATH)}
    joined: list[dict[str, Any]] = []

    for experiment, path in RESULT_FILES.items():
        for row in read_csv(path):
            dataset_row = dataset[row["question_id"]]
            expected_pages = parse_expected_pages(dataset_row["expected_page"])
            expected_source = dataset_row["expected_source"].strip()
            slots = retrieved_slots(row, k=3)
            source_rank = next((i for i, (source, _) in enumerate(slots, start=1) if source == expected_source), None)
            page_rank = next(
                (
                    i
                    for i, (source, page) in enumerate(slots, start=1)
                    if source == expected_source and page in expected_pages
                ),
                None,
            )
            expected_language = row["expected_language"]
            reference = dataset_row.get(LANGUAGE_REFERENCE_COLUMN[expected_language], row.get("reference_answer", ""))
            generated = row.get("generated_answer", "")
            recall = token_recall(reference, generated)
            precision = token_precision(reference, generated)
            relevance = f1(precision, recall)
            unsupported = bool(re.search(r"could not be found|paoa jayni|পাওয়া যায়নি", generated, flags=re.IGNORECASE))

            joined.append({
                "experiment": experiment,
                "question_id": row["question_id"],
                "question": row["question"],
                "course_code": dataset_row["course_code"],
                "intent": dataset_row["intent"],
                "difficulty": dataset_row["difficulty"],
                "ground_truth_status": dataset_row["ground_truth_status"],
                "expected_language": expected_language,
                "detected_language": row["detected_language"],
                "language_detection_correct": row["detected_language"] == expected_language,
                "response_language_consistent": truthy(row["language_consistent"]),
                "expected_source": expected_source,
                "expected_page": dataset_row["expected_page"],
                "retrieved_source_1": row.get("retrieved_source_1", ""),
                "retrieved_page_1": row.get("retrieved_page_1", ""),
                "hit_at_1": page_rank == 1,
                "hit_at_3": page_rank is not None and page_rank <= 3,
                "mrr": 1.0 / page_rank if page_rank else 0.0,
                "source_accuracy": source_rank == 1,
                "page_accuracy": page_rank == 1,
                "reference_token_recall": recall,
                "reference_token_precision": precision,
                "reference_token_f1": relevance,
                "correctness_proxy": recall >= 0.70 and not unsupported,
                "relevance_proxy": relevance >= 0.45 and not unsupported,
                "groundedness_proxy": page_rank is not None and not unsupported,
                "completeness_proxy": recall >= 0.85 and not unsupported,
                "latency_seconds": float(row["latency_seconds"]),
                "answer_mode": row["answer_mode"],
                "answer_bank_enabled": row["answer_bank_enabled"],
            })
    return joined


def summarize_group(rows: list[dict[str, Any]], experiment: str, group_name: str, group_value: str) -> dict[str, Any]:
    latencies = [row["latency_seconds"] for row in rows]
    return {
        "experiment": experiment,
        "group": group_name,
        "value": group_value,
        "n": len(rows),
        "hit_at_1_pct": pct(avg_bool(rows, "hit_at_1")),
        "hit_at_3_pct": pct(avg_bool(rows, "hit_at_3")),
        "mrr": f"{avg_float(rows, 'mrr'):.3f}",
        "source_accuracy_pct": pct(avg_bool(rows, "source_accuracy")),
        "page_accuracy_pct": pct(avg_bool(rows, "page_accuracy")),
        "language_detection_accuracy_pct": pct(avg_bool(rows, "language_detection_correct")),
        "response_language_consistency_pct": pct(avg_bool(rows, "response_language_consistent")),
        "correctness_proxy_pct": pct(avg_bool(rows, "correctness_proxy")),
        "relevance_proxy_pct": pct(avg_bool(rows, "relevance_proxy")),
        "groundedness_proxy_pct": pct(avg_bool(rows, "groundedness_proxy")),
        "completeness_proxy_pct": pct(avg_bool(rows, "completeness_proxy")),
        "avg_reference_token_recall": f"{avg_float(rows, 'reference_token_recall'):.3f}",
        "avg_reference_token_f1": f"{avg_float(rows, 'reference_token_f1'):.3f}",
        "avg_latency_seconds": f"{mean(latencies):.3f}" if latencies else "0.000",
        "median_latency_seconds": f"{median(latencies):.3f}" if latencies else "0.000",
        "p95_latency_seconds": f"{p95(latencies):.3f}" if latencies else "0.000",
    }


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return dict(sorted(groups.items()))


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def write_summary(overall_rows: list[dict[str, Any]], breakdown_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Experiment Metrics Summary",
        "",
        "Answer-quality values are automatic proxy metrics based on language-specific reference token overlap and retrieval grounding. Use them for screening; use manual adjudication for final thesis claims about correctness, relevance, groundedness, and completeness.",
        "",
        "## Overall",
        "",
        markdown_table(overall_rows, [
            "experiment",
            "n",
            "hit_at_1_pct",
            "hit_at_3_pct",
            "mrr",
            "source_accuracy_pct",
            "page_accuracy_pct",
            "language_detection_accuracy_pct",
            "response_language_consistency_pct",
            "correctness_proxy_pct",
            "relevance_proxy_pct",
            "groundedness_proxy_pct",
            "completeness_proxy_pct",
            "avg_latency_seconds",
            "median_latency_seconds",
            "p95_latency_seconds",
        ]),
        "",
        "## Breakdown Tables",
        "",
        f"Full breakdown written to `{BREAKDOWN_PATH.relative_to(ROOT_DIR)}`.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    joined = build_joined_rows()
    per_query_fields = list(joined[0].keys())
    write_csv(PER_QUERY_PATH, joined, per_query_fields)

    overall_rows = []
    breakdown_rows = []
    for experiment in RESULT_FILES:
        experiment_rows = [row for row in joined if row["experiment"] == experiment]
        overall_rows.append(summarize_group(experiment_rows, experiment, "overall", "overall"))
        for key in ("expected_language", "intent", "course_code", "difficulty"):
            for value, rows in grouped(experiment_rows, key).items():
                breakdown_rows.append(summarize_group(rows, experiment, key, value))

    metric_fields = list(overall_rows[0].keys())
    write_csv(OVERALL_PATH, overall_rows, metric_fields)
    write_csv(BREAKDOWN_PATH, breakdown_rows, metric_fields)
    write_summary(overall_rows, breakdown_rows)

    for row in overall_rows:
        print(row)
    print(f"Wrote {OVERALL_PATH.relative_to(ROOT_DIR)}")
    print(f"Wrote {BREAKDOWN_PATH.relative_to(ROOT_DIR)}")
    print(f"Wrote {PER_QUERY_PATH.relative_to(ROOT_DIR)}")
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
