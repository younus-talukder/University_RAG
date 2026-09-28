"""Narrow, checkpointed Step 8C regression measurement; no 300-row rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_step7g_full import read_csv, truth, write_csv
from scripts.evaluate_step8 import evaluate
from scripts.evaluate_step8b import audit, targeted_gate
from src.config import RERANKER_ENABLED
from src.evaluator import load_dataset

RESULTS = ROOT / "results"
CHECKPOINT = RESULTS / "step8c_targeted_checkpoint.jsonl"
TARGETED_CSV = RESULTS / "step8c_targeted_results.csv"
TARGET_IDS = {"Q014", "Q016", "Q018", "Q020", "Q031", "Q081"}
LANGUAGES = ("english", "bangla", "banglish")


def fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in (
        "data/questions/questions.csv", "vector_db/index_manifest.json",
        "src/query_normalization.py", "src/evidence.py", "src/fast_answer.py",
        "src/relations.py", "src/answer_safety.py", "src/pipeline.py",
        "scripts/evaluate_step8.py", "scripts/evaluate_step8c.py",
    ):
        digest.update(relative.encode())
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()[:20]


def q020_failures(rows: list[dict]) -> list[str]:
    issues: list[str] = []
    chosen = [row for row in rows if row["question_id"] == "Q020"]
    if len(chosen) != 3 or {row["language"] for row in chosen} != set(LANGUAGES):
        issues.append("Q020_EXPECTED_THREE_LANGUAGES")
    for row in chosen:
        key = "Q020:" + row["language"]
        for column, expected in (
            ("answerability_status", "SUPPORTED"), ("requested_field", "assessment"),
            ("requested_relation", "mark_distribution"), ("extracted_relation", "mark_distribution"),
            ("page", "18"),
        ):
            if str(row.get(column) or "") != expected:
                issues.append(f"{key}:{column}:{row.get(column)}")
        if not str(row.get("source") or "").endswith("curricula_BSc-Curriculum-New.pdf"):
            issues.append(key + ":SOURCE")
        for value in ("30%", "20%", "50%"):
            if value not in str(row.get("final_answer") or "") or value not in str(row.get("supporting_excerpt") or ""):
                issues.append(key + ":MISSING_" + value)
        for column in ("language_validation_passed", "grounding_validation_passed", "semantic_validation_passed"):
            if not truth(row.get(column)):
                issues.append(key + ":FAILED_" + column)
        issues.extend(key + ":" + item for item in audit(row))
    return issues


def targeted() -> None:
    if RERANKER_ENABLED:
        raise RuntimeError("Step 8C requires reranker OFF")
    selected = [row for row in load_dataset() if row["question_id"] in TARGET_IDS]
    if len(selected) != 18:
        raise RuntimeError("Expected six three-language target groups")
    current = fingerprint()
    existing = []
    if CHECKPOINT.exists():
        with CHECKPOINT.open(encoding="utf-8") as handle:
            existing = [json.loads(line) for line in handle if line.strip()]
    if any(row["run_fingerprint"] != current for row in existing):
        raise RuntimeError("Step 8C checkpoint fingerprint mismatch; preserve it before restarting")
    completed = {(row["question_id"], row["language"]): row for row in existing}
    if len(completed) != len(existing):
        raise RuntimeError("Duplicate Step 8C checkpoint key")
    for row in selected:
        key = row["question_id"], row["expected_language"]
        if key in completed:
            continue
        result = evaluate(row, current)
        with CHECKPOINT.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
        completed[key] = result
        print(f"{len(completed)}/18 {key[1]} {key[0]} {result['answerability_status']} "
              f"{result['answer_strategy']} {result['total_seconds']:.1f}s", flush=True)
    ordered = [completed[(row["question_id"], row["expected_language"])] for row in selected]
    write_csv(TARGETED_CSV, ordered)
    issues = q020_failures(ordered)
    issues += targeted_gate([row for row in ordered if row["question_id"] != "Q020"])
    # The reused 8B gate expects exactly 15 rows; no production rule contains IDs.
    (RESULTS / "step8c_targeted_gate.json").write_text(
        json.dumps({"passed": not issues, "issues": issues, "completion": len(ordered),
                    "fingerprint": current}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TARGETED_GATE {'PASS' if not issues else 'FAIL'} {issues}", flush=True)
    if issues:
        raise SystemExit(2)


def legacy() -> None:
    from scripts import evaluate_step7h_targeted as frozen
    frozen.OUTPUT = RESULTS / "step8c_step7h_targeted_results.csv"
    frozen.SUMMARY = RESULTS / "step8c_step7h_targeted_summary.json"
    frozen.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("targeted", "legacy"))
    phase = parser.parse_args().phase
    targeted() if phase == "targeted" else legacy()
