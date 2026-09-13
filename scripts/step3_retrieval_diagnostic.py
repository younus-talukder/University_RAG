from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import QUESTIONS_DIR, VECTOR_DB_DIR
from src.embeddings import EmbeddingModel
from src.evidence import SupportStatus, assess_evidence, detect_requested_field, identify_entities
from src.retriever import RETRIEVAL_RERANK_WEIGHTS, rerank_candidates, select_retrieval_results
from src.vector_store import load_index, search


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
                    rows.append(
                        {
                            "language": language,
                            "question": question,
                            "expected_source": source.get("expected_source", ""),
                            "expected_page": source.get("expected_page", ""),
                        }
                    )
    return rows


def expected_pages(value: str) -> set[int]:
    return {
        int(part.strip())
        for group in str(value).split(";")
        for part in group.split(",")
        if part.strip().isdigit()
    }


def entity_match(question: str, chunk: dict) -> bool:
    entities = [entity for entity in identify_entities(question) if entity.kind == "course_code"]
    if not entities:
        return False
    compact = re.sub(r"[^A-Za-z0-9]", "", str(chunk.get("text", ""))).upper()
    return all(entity.normalized in compact for entity in entities)


def summarize(rows: list[dict]) -> dict:
    count = len(rows)
    entity_rows = [row for row in rows if row["has_entity"]]
    field_rows = [row for row in rows if row["has_field"]]
    return {
        "queries": count,
        "page_hit_at_1": sum(row["page_rank"] == 1 for row in rows) / count if count else 0.0,
        "page_hit_at_3": sum(0 < row["page_rank"] <= 3 for row in rows) / count if count else 0.0,
        "mrr_at_3": sum((1.0 / row["page_rank"]) if 0 < row["page_rank"] <= 3 else 0.0 for row in rows) / count if count else 0.0,
        "entity_queries": len(entity_rows),
        "entity_hit_at_1": sum(row["entity_hit_1"] for row in entity_rows) / len(entity_rows) if entity_rows else 0.0,
        "entity_hit_at_3": sum(row["entity_hit_3"] for row in entity_rows) / len(entity_rows) if entity_rows else 0.0,
        "field_queries": len(field_rows),
        "field_hit_at_1": sum(row["field_hit_1"] for row in field_rows) / len(field_rows) if field_rows else 0.0,
        "field_hit_at_3": sum(row["field_hit_3"] for row in field_rows) / len(field_rows) if field_rows else 0.0,
    }


def run(label: str) -> dict:
    started = time.perf_counter()
    dataset = variants()
    index, metadata = load_index()
    model = EmbeddingModel()
    vectors = model.embed_many([row["question"] for row in dataset])
    results: list[dict] = []
    candidate_pool = min(
        len(metadata),
        max(
            int(RETRIEVAL_RERANK_WEIGHTS["candidate_pool_min"]),
            3 * int(RETRIEVAL_RERANK_WEIGHTS["candidate_pool_multiplier"]),
        ),
    )
    for row, vector in zip(dataset, vectors):
        candidates = search(index, vector, metadata, top_k=candidate_pool)
        retrieved = select_retrieval_results(rerank_candidates(row["question"], candidates), 3)
        pages = expected_pages(row["expected_page"])
        page_rank = next(
            (
                rank
                for rank, item in enumerate(retrieved, start=1)
                if str(item.get("source", "")) == row["expected_source"] and item.get("page") in pages
            ),
            0,
        )
        entities = [entity for entity in identify_entities(row["question"]) if entity.kind == "course_code"]
        requested_field = detect_requested_field(row["question"])
        supports = [
            assess_evidence(row["question"], [{**item, "score": item.get("score", 0.0)}]).status
            in {SupportStatus.SUPPORTED, SupportStatus.CONFLICTING}
            for item in retrieved
        ]
        results.append(
            {
                **row,
                "page_rank": page_rank,
                "has_entity": bool(entities),
                "entity_hit_1": bool(retrieved and entity_match(row["question"], retrieved[0])),
                "entity_hit_3": any(entity_match(row["question"], item) for item in retrieved),
                "has_field": requested_field != "general",
                "field_hit_1": bool(supports and supports[0]),
                "field_hit_3": any(supports),
            }
        )
    by_language: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_language[row["language"]].append(row)
    return {
        "label": label,
        "notice": "DEVELOPMENT DIAGNOSTIC - NOT FINAL THESIS ACCURACY",
        "elapsed_seconds": time.perf_counter() - started,
        "overall": summarize(results),
        "languages": {language: summarize(rows) for language, rows in sorted(by_language.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Step 3 retrieval-only development diagnostic.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
