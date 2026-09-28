"""Sequential, checkpointed Step 8B safety regression and development rerun."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_step7g_full import LANGUAGES, read_csv, truth, write_csv
from scripts.evaluate_step8 import evaluate
from src.answer_safety import answer_safety_failures, requested_relation
from src.config import RERANKER_ENABLED
from src.evaluator import load_dataset

RESULTS = ROOT / "results"
TARGET_IDS = {"Q014", "Q016", "Q018", "Q031", "Q081"}
PREFIX = RESULTS / "step8b"


def fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in (
        "data/questions/questions.csv", "vector_db/index_manifest.json",
        "src/answer_safety.py", "src/answerability.py", "src/evidence.py",
        "src/pipeline.py", "src/relations.py", "src/answer_policy.py",
        "src/grounding_validator.py", "src/semantic_contract.py",
        "scripts/evaluate_step8.py", "scripts/evaluate_step8b.py",
    ):
        digest.update(relative.encode())
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()[:20]


def checkpoint_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    keys = [(row["question_id"], row["language"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate checkpoint keys in {path}")
    return rows


def audit(row: dict) -> tuple[str, ...]:
    if row.get("answerability_status") != "SUPPORTED":
        return ()
    failures = []
    if row.get("evidence_level") == "WEAK":
        failures.append("WEAK_RETURN")
    for key in ("language_validation_passed", "grounding_validation_passed", "semantic_validation_passed"):
        if not truth(row.get(key)):
            failures.append("FAILED_" + key.upper())
    failures.extend(json.loads(row.get("answer_safety_failures") or "[]"))
    wanted = requested_relation(row["question"])
    extracted = row.get("extracted_relation") or None
    if wanted and extracted and wanted != extracted:
        failures.append("RELATION_MISMATCH")
    if wanted == "regular_semester_duration" and not any(
        token in str(row.get("final_answer") or "").casefold()
        for token in ("week", "সপ্তাহ", "শপ্তাহ", "shoptaho", "soptaho")
    ):
        failures.append("RELATION_MISMATCH")
    excerpts = [str(row.get("supporting_excerpt") or "")]
    failures.extend(answer_safety_failures(row["question"], str(row.get("final_answer") or ""),
                                           excerpts, extracted_relation=extracted,
                                           relation_values=json.loads(row.get("extracted_relation_values") or "[]")))
    return tuple(dict.fromkeys(failures))


def targeted_gate(rows: list[dict]) -> list[str]:
    issues = []
    if len(rows) != 15:
        issues.append(f"EXPECTED_15_GOT_{len(rows)}")
    for row in rows:
        key = f"{row['question_id']}:{row['language']}"
        status = row.get("answerability_status")
        if status == "SYSTEM_ERROR":
            issues.append(key + ":SYSTEM_ERROR")
        if row["question_id"] in {"Q018", "Q031"} and status == "AMBIGUOUS_QUERY":
            issues.append(key + ":FALSE_AMBIGUITY")
        if row["question_id"] == "Q081" and status == "CONFLICTING_EVIDENCE":
            issues.append(key + ":FALSE_CONFLICT")
        if row["question_id"] == "Q016" and status == "SUPPORTED":
            if "week" not in str(row.get("final_answer") or "").casefold() and "সপ্তাহ" not in str(row.get("final_answer") or ""):
                issues.append(key + ":WRONG_RELATION")
        if row["question_id"] == "Q014" and status == "SUPPORTED":
            answer = str(row.get("final_answer") or "").casefold()
            if not all(value in answer for value in ("admission@uap-bd.edu", "registrar@uap-bd.edu")):
                issues.append(key + ":MISSING_EMAIL")
        issues.extend(key + ":" + reason for reason in audit(row))
    return issues


def run(phase: str) -> None:
    if RERANKER_ENABLED:
        raise RuntimeError("Step 8B measurement requires reranker OFF")
    dataset = load_dataset()
    if len(dataset) != 300 or Counter(row["expected_language"] for row in dataset) != Counter({lang: 100 for lang in LANGUAGES}):
        raise RuntimeError("Expected 300 development variants")
    selected = [row for row in dataset if phase == "full" or row["question_id"] in TARGET_IDS]
    checkpoint = RESULTS / f"step8b_{phase}_checkpoint.jsonl"
    output = RESULTS / f"step8b_{phase}_results.csv"
    current = fingerprint()
    existing = checkpoint_rows(checkpoint)
    if any(row["run_fingerprint"] != current for row in existing):
        raise RuntimeError("Step 8B checkpoint fingerprint changed; preserve the old checkpoint first")
    completed = {(row["question_id"], row["language"]): row for row in existing}
    for row in selected:
        key = (row["question_id"], row["expected_language"])
        if key in completed:
            continue
        result = evaluate(row, current)
        with checkpoint.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
        completed[key] = result
        print(f"{phase} {len(completed)}/{len(selected)} {key[1]} {key[0]} "
              f"{result['answerability_status']} {result['answer_strategy']} "
              f"{result['total_seconds']:.1f}s", flush=True)
    ordered = [completed[(row["question_id"], row["expected_language"])] for row in selected]
    write_csv(output, ordered)
    if phase == "targeted":
        issues = targeted_gate(ordered)
        (RESULTS / "step8b_targeted_gate.json").write_text(
            json.dumps({"passed": not issues, "issues": issues, "completion": len(ordered)}, indent=2), encoding="utf-8"
        )
        print(f"TARGETED_GATE {'PASS' if not issues else 'FAIL'} {issues}", flush=True)
        if issues:
            raise SystemExit(2)


def finalize() -> None:
    targeted = json.loads((RESULTS / "step8b_targeted_gate.json").read_text(encoding="utf-8"))
    if not targeted["passed"]:
        raise RuntimeError("Targeted gate did not pass")
    rows = read_csv(RESULTS / "step8b_full_results.csv")
    if len(rows) != 300 or len({(row["question_id"], row["language"]) for row in rows}) != 300:
        raise RuntimeError("Step 8B full result must contain 300 unique variants")
    unsafe = [{**row, "unsafe_reason": ",".join(audit(row))} for row in rows if audit(row)]
    unsafe_path = RESULTS / "step8b_unsafe_answer_review.csv"
    if unsafe:
        write_csv(unsafe_path, unsafe)
    else:
        with unsafe_path.open("w", encoding="utf-8-sig", newline="") as handle:
            csv.writer(handle).writerow(list(rows[0]) + ["unsafe_reason"])
    groups = {lang: [row for row in rows if row["language"] == lang] for lang in LANGUAGES}
    groups["overall"] = rows
    summary = {
        "notice": "Development automatic safety proxies; not final human or thesis accuracy",
        "completion": 300, "targeted_gate": targeted, "unsafe_returned_answers": len(unsafe),
        "per_language": {lang: {"answers_returned": sum(row["answerability_status"] == "SUPPORTED" for row in group),
                                "states": dict(Counter(row["answerability_status"] for row in group)),
                                "strategies": dict(Counter(row["answer_strategy"] for row in group))}
                         for lang, group in groups.items()},
        "generation_attempts": sum(int(row["generation_attempts"] or 0) for row in rows),
        "generation_used_cases": sum(truth(row["generation_used"]) for row in rows),
        "retry_used_cases": sum(truth(row["retry_used"]) for row in rows),
        "latency_seconds": {"median": statistics.median(float(row["total_seconds"]) for row in rows),
                            "p95": sorted(float(row["total_seconds"]) for row in rows)[284]},
        "memory": {"peak_rss_bytes": max(int(row["rss_bytes"]) for row in rows),
                   "peak_process_bytes": max(int(row["peak_process_bytes"]) for row in rows),
                   "minimum_available_ram_bytes": min(int(row["available_ram_bytes"]) for row in rows),
                   "peak_pagefile_used_bytes_system": max(int(row["pagefile_used_bytes_system"]) for row in rows)},
        "errors": [f"{row['question_id']}:{row['language']}:{row['runtime_error']}" for row in rows if row["runtime_error"]],
        "answer_bank_enabled": False, "reranker_enabled": False,
    }
    (RESULTS / "step8b_full_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"completion": 300, "unsafe": len(unsafe), "states": summary["per_language"]["overall"]["states"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("targeted", "full", "finalize"))
    args = parser.parse_args()
    finalize() if args.phase == "finalize" else run(args.phase)
