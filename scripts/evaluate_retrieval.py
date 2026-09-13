from __future__ import annotations

import csv
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import QUESTIONS_DIR, VECTOR_DB_DIR
from src.embeddings import get_embedding_model
from src.fast_answer import DEFAULT_COURSE_SOURCE
from src.retriever import RETRIEVAL_RERANK_WEIGHTS, Retriever


DATASET_PATH = QUESTIONS_DIR / "questions.csv"
BASELINE_PATH = ROOT_DIR / "results" / "experiment_b_gguf_answer_bank_off.csv"
PER_QUERY_PATH = ROOT_DIR / "results" / "retrieval_optimized_per_query.csv"
OVERALL_PATH = ROOT_DIR / "results" / "retrieval_optimized_overall.csv"
BREAKDOWN_PATH = ROOT_DIR / "results" / "retrieval_optimized_breakdown.csv"
FAILURE_PATH = ROOT_DIR / "results" / "retrieval_baseline_failure_analysis.csv"
SUMMARY_PATH = ROOT_DIR / "results" / "retrieval_optimization_summary.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def expected_pages(value: str) -> set[str]:
    pages = set()
    for part in str(value).split(";"):
        for page in part.split(","):
            page = page.strip()
            if page:
                pages.add(page)
    return pages


def dataset_variants() -> list[dict[str, str]]:
    rows = []
    for row in read_csv(DATASET_PATH):
        for language, column in (
            ("english", "english_question"),
            ("bangla", "bangla_question"),
            ("banglish", "banglish_question"),
        ):
            question = row.get(column, "").strip()
            if question:
                rows.append({
                    "question_id": row["question_id"],
                    "question": question,
                    "expected_language": language,
                    "intent": row.get("intent", "general"),
                    "course_code": row.get("course_code", "GENERAL"),
                    "difficulty": row.get("difficulty", "medium"),
                    "expected_source": row.get("expected_source", DEFAULT_COURSE_SOURCE),
                    "expected_page": row.get("expected_page", row.get("expected_page_of_answer", "")),
                    "ground_truth_status": row.get("ground_truth_status", "verified"),
                })
    return rows


def slots_from_result(result: dict[str, Any]) -> list[tuple[str, str, float]]:
    slots = []
    for source in result.get("sources", []):
        slots.append((str(source.get("source", "")), str(source.get("page", "")), float(source.get("score", 0.0))))
    return slots


def slots_from_csv(row: dict[str, str]) -> list[tuple[str, str, float]]:
    slots = []
    for index in range(1, 4):
        source = row.get(f"retrieved_source_{index}", "").strip()
        page = row.get(f"retrieved_page_{index}", "").strip()
        score = row.get(f"retrieved_score_{index}", "0").strip()
        if source or page:
            slots.append((source, page, float(score or 0.0)))
    return slots


def score_retrieval(row: dict[str, str], slots: list[tuple[str, str, float]]) -> dict[str, Any]:
    source = row["expected_source"]
    pages = expected_pages(row["expected_page"])
    source_rank = next((i for i, (got_source, _, _) in enumerate(slots, start=1) if got_source == source), None)
    page_rank = next(
        (
            i
            for i, (got_source, got_page, _) in enumerate(slots, start=1)
            if got_source == source and got_page in pages
        ),
        None,
    )
    ambiguous_pages = len(pages) > 1
    if page_rank == 1:
        failure_class = "A_SOURCE_AND_PAGE_RANK1"
    elif source_rank is not None and source_rank <= 3 and page_rank is not None:
        failure_class = "B_SOURCE_TOP3_PAGE_NOT_RANK1"
    elif source_rank is not None and source_rank <= 3:
        failure_class = "C_SOURCE_TOP3_PAGE_ABSENT"
    else:
        failure_class = "D_SOURCE_ABSENT_TOP3"

    return {
        "hit_at_1": page_rank == 1,
        "hit_at_3": page_rank is not None and page_rank <= 3,
        "mrr": 1.0 / page_rank if page_rank else 0.0,
        "source_accuracy": source_rank == 1,
        "page_accuracy": page_rank == 1,
        "source_rank": source_rank or "",
        "page_rank": page_rank or "",
        "ambiguous_pages": ambiguous_pages,
        "failure_class": failure_class,
    }


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))]


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def avg_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for row in rows if row[key]) / len(rows) if rows else 0.0


def avg_float(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row[key]) for row in rows) if rows else 0.0


def summarize(rows: list[dict[str, Any]], experiment: str, group: str, value: str) -> dict[str, Any]:
    latencies = [float(row.get("latency_seconds", 0.0)) for row in rows]
    return {
        "experiment": experiment,
        "group": group,
        "value": value,
        "n": len(rows),
        "hit_at_1_pct": pct(avg_bool(rows, "hit_at_1")),
        "hit_at_3_pct": pct(avg_bool(rows, "hit_at_3")),
        "mrr": f"{avg_float(rows, 'mrr'):.3f}",
        "source_accuracy_pct": pct(avg_bool(rows, "source_accuracy")),
        "page_accuracy_pct": pct(avg_bool(rows, "page_accuracy")),
        "avg_latency_seconds": f"{mean(latencies):.3f}" if latencies else "0.000",
        "median_latency_seconds": f"{median(latencies):.3f}" if latencies else "0.000",
        "p95_latency_seconds": f"{p95(latencies):.3f}" if latencies else "0.000",
    }


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return dict(sorted(groups.items()))


def baseline_rows() -> list[dict[str, Any]]:
    dataset = {
        (row["question_id"], row["expected_language"]): row
        for row in dataset_variants()
    }
    rows = []
    for row in read_csv(BASELINE_PATH):
        key = (row["question_id"], row["expected_language"])
        expected = dataset[key]
        metrics = score_retrieval(expected, slots_from_csv(row))
        rows.append({
            **expected,
            **metrics,
            "experiment": "Baseline GGUF retrieval",
            "latency_seconds": float(row.get("latency_seconds", 0.0) or 0.0),
            "retrieved_source_1": row.get("retrieved_source_1", ""),
            "retrieved_page_1": row.get("retrieved_page_1", ""),
            "retrieved_source_2": row.get("retrieved_source_2", ""),
            "retrieved_page_2": row.get("retrieved_page_2", ""),
            "retrieved_source_3": row.get("retrieved_source_3", ""),
            "retrieved_page_3": row.get("retrieved_page_3", ""),
        })
    return rows


def optimized_rows() -> list[dict[str, Any]]:
    embedding_model = get_embedding_model()
    retriever = Retriever(embedding_model=embedding_model, top_k=3)
    rows = []
    for row in dataset_variants():
        started = time.perf_counter()
        retrieved = retriever.retrieve(
            row["question"],
            index_path=str(VECTOR_DB_DIR / "index.faiss"),
            metadata_path=str(VECTOR_DB_DIR / "metadata.pkl"),
            debug=True,
        )
        latency = time.perf_counter() - started
        slots = slots_from_result({"sources": retrieved})
        metrics = score_retrieval(row, slots)
        output = {
            **row,
            **metrics,
            "experiment": "Optimized retrieval",
            "latency_seconds": f"{latency:.3f}",
        }
        for index in range(1, 4):
            item = retrieved[index - 1] if index <= len(retrieved) else {}
            output[f"retrieved_source_{index}"] = item.get("source", "")
            output[f"retrieved_page_{index}"] = item.get("page", "")
            output[f"retrieved_score_{index}"] = f"{float(item.get('score', 0.0)):.4f}" if item else ""
            output[f"rerank_debug_{index}"] = item.get("rerank_debug", "")
        rows.append(output)
    return rows


def failure_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for group in ("expected_language", "intent", "course_code", "difficulty"):
        for value, grouped in group_rows(rows, group).items():
            counts = Counter(row["failure_class"] for row in grouped)
            output.append({
                "group": group,
                "value": value,
                "n": len(grouped),
                "A_source_and_page_rank1": counts["A_SOURCE_AND_PAGE_RANK1"],
                "B_source_top3_page_not_rank1": counts["B_SOURCE_TOP3_PAGE_NOT_RANK1"],
                "C_source_top3_page_absent": counts["C_SOURCE_TOP3_PAGE_ABSENT"],
                "D_source_absent_top3": counts["D_SOURCE_ABSENT_TOP3"],
                "E_multiple_valid_pages": sum(1 for row in grouped if row["ambiguous_pages"]),
            })
    return output


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_summary(
    baseline_overall: dict[str, Any],
    optimized_overall: dict[str, Any],
    breakdowns: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    delta_rows = []
    for metric in ("hit_at_1_pct", "hit_at_3_pct", "mrr", "source_accuracy_pct", "page_accuracy_pct"):
        base = float(baseline_overall[metric])
        opt = float(optimized_overall[metric])
        delta_rows.append({"metric": metric, "baseline": base, "optimized": opt, "delta": f"{opt - base:.3f}"})

    lines = [
        "# Retrieval Optimization Summary",
        "",
        "Baseline experiment files were read only. Optimized retrieval outputs are saved separately.",
        "",
        "## Overall",
        "",
        markdown_table([baseline_overall, optimized_overall], [
            "experiment",
            "n",
            "hit_at_1_pct",
            "hit_at_3_pct",
            "mrr",
            "source_accuracy_pct",
            "page_accuracy_pct",
            "avg_latency_seconds",
            "median_latency_seconds",
            "p95_latency_seconds",
        ]),
        "",
        "## Delta",
        "",
        markdown_table(delta_rows, ["metric", "baseline", "optimized", "delta"]),
        "",
        "## Reranking Weights",
        "",
        "```text",
        "\n".join(f"{key} = {value}" for key, value in RETRIEVAL_RERANK_WEIGHTS.items()),
        "```",
        "",
        f"Full optimized per-query output: `{PER_QUERY_PATH.relative_to(ROOT_DIR)}`",
        f"Full breakdown output: `{BREAKDOWN_PATH.relative_to(ROOT_DIR)}`",
        f"Baseline failure analysis: `{FAILURE_PATH.relative_to(ROOT_DIR)}`",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    baseline = baseline_rows()
    optimized = optimized_rows()
    all_rows = baseline + optimized

    baseline_overall = summarize(baseline, "Baseline GGUF retrieval", "overall", "overall")
    optimized_overall = summarize(optimized, "Optimized retrieval", "overall", "overall")

    breakdowns = []
    for experiment, rows in (("Baseline GGUF retrieval", baseline), ("Optimized retrieval", optimized)):
        for group in ("expected_language", "intent", "course_code", "difficulty"):
            for value, grouped in group_rows(rows, group).items():
                breakdowns.append(summarize(grouped, experiment, group, value))

    failures = failure_analysis(baseline)

    per_query_fields = list(optimized[0].keys())
    write_csv(PER_QUERY_PATH, optimized, per_query_fields)
    write_csv(OVERALL_PATH, [baseline_overall, optimized_overall], list(baseline_overall.keys()))
    write_csv(BREAKDOWN_PATH, breakdowns, list(breakdowns[0].keys()))
    write_csv(FAILURE_PATH, failures, list(failures[0].keys()))
    write_summary(baseline_overall, optimized_overall, breakdowns, failures)

    print(baseline_overall)
    print(optimized_overall)
    print(f"Wrote {PER_QUERY_PATH.relative_to(ROOT_DIR)}")
    print(f"Wrote {OVERALL_PATH.relative_to(ROOT_DIR)}")
    print(f"Wrote {BREAKDOWN_PATH.relative_to(ROOT_DIR)}")
    print(f"Wrote {FAILURE_PATH.relative_to(ROOT_DIR)}")
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
