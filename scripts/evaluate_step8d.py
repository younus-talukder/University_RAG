"""Sequential, checkpointed Step 8D development-only conflict gate.

Only new Step 8D files are written. References and IDs remain evaluation data;
production conflict code receives only question text and retrieved evidence.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_step8 import evaluation_reference
from src.answer_safety import answer_safety_failures
from src.config import RERANKER_ENABLED
from src.evaluator import load_dataset
from src.pipeline import answer_question

RESULTS = ROOT / "results"
CHECKPOINT = RESULTS / "step8d_targeted_checkpoint_v3.jsonl"
TARGET_IDS = {"Q014", "Q016", "Q018", "Q020", "Q031", "Q081", "Q088", "Q095"}
LANGUAGES = {"english", "bangla", "banglish"}


def fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in (
        "data/questions/questions.csv", "vector_db/index_manifest.json",
        "src/evidence.py", "src/answerability.py", "src/course_rows.py",
        "src/answer_safety.py", "src/pipeline.py", "scripts/evaluate_step8d.py",
    ):
        digest.update(relative.encode())
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()[:20]


def measure(row: dict, current: str) -> dict:
    started = time.perf_counter()
    response = answer_question(row["question"], top_k=3, use_generation=True, use_answer_bank=False)
    memory = psutil.Process().memory_info()
    reference, expected_page, authority = evaluation_reference(row)
    evidence = list(response.get("evidence") or [])
    excerpts = [str(item.get("excerpt") or "") for item in evidence]
    status = str(response.get("answerability_status") or "")
    failures = (answer_safety_failures(
        row["question"], str(response.get("answer") or ""), excerpts,
        extracted_relation=response.get("relation_type"),
        relation_values=(response.get("relation_extracted") or {}).get("values") or (),
    ) if status == "SUPPORTED" else ())
    trace = None
    if row["question_id"] in {"Q018", "Q088", "Q095"}:
        trace = {
            "retrieval_candidates": response.get("retrieved_context") or [],
            "verified_evidence": evidence,
            "conflict_sources": response.get("conflict_sources") or [],
            "normalized_conflict_values": response.get("conflicting_values") or [],
            "initial_support_status": response.get("initial_support_status"),
            "merged_support_status": response.get("merged_support_status"),
        }
    return {
        "run_fingerprint": current,
        "question_id": row["question_id"], "language": row["expected_language"],
        "question": row["question"], "reference": reference,
        "expected_page_offline": expected_page, "evaluation_authority": authority,
        "answerability_status": status,
        "answerability_reason": response.get("answerability_reason"),
        "requested_entity": response.get("requested_entity"),
        "requested_field": response.get("requested_field"),
        "requested_relation": response.get("requested_relation"),
        "extracted_relation": response.get("relation_type"),
        "answer_strategy": response.get("answer_strategy"),
        "final_answer": response.get("answer"),
        "source": response.get("source"), "page": response.get("page"),
        "chunk_id": response.get("chunk_id"),
        "supporting_excerpt": response.get("supporting_excerpt"),
        "supporting_evidence_count": response.get("supporting_evidence_count"),
        "conflict_detected": response.get("conflict_detected"),
        "conflict_sources": response.get("conflict_sources") or [],
        "conflicting_values": response.get("conflicting_values") or [],
        "cross_lingual_fallback_used": response.get("cross_lingual_fallback_used"),
        "language_validation_passed": response.get("language_validation_passed"),
        "grounding_validation_passed": response.get("grounding_validation_passed"),
        "semantic_validation_passed": response.get("semantic_validation_passed"),
        "answer_safety_failures": list(failures),
        "generation_used": response.get("generation_used"),
        "generation_attempts": response.get("generation_attempts"),
        "runtime_error": response.get("runtime_error") or "",
        "total_seconds": time.perf_counter() - started,
        "rss_bytes": memory.rss,
        "available_ram_bytes": psutil.virtual_memory().available,
        "pagefile_used_bytes_system": psutil.swap_memory().used,
        "answer_bank_enabled": False, "reranker_enabled": False,
        "trace": trace,
    }


def gate(rows: list[dict]) -> list[str]:
    issues: list[str] = []
    by_key = {(row["question_id"], row["language"]): row for row in rows}
    if len(rows) != len(TARGET_IDS) * len(LANGUAGES) or len(by_key) != len(rows):
        issues.append("INCOMPLETE_OR_DUPLICATE_TARGETS")
    for language in LANGUAGES:
        def one(qid: str) -> dict:
            return by_key.get((qid, language), {})
        q018 = one("Q018")
        if q018.get("answerability_status") == "CONFLICTING_EVIDENCE":
            issues.append(f"Q018:{language}:FALSE_CONFLICT")
        if q018.get("answerability_status") == "SUPPORTED" and (
            str(q018.get("page")) != "17" or
            not q018.get("grounding_validation_passed")
        ):
            issues.append(f"Q018:{language}:WRONG_RULE_OR_PAGE")
        q088 = one("Q088")
        if q088.get("answerability_status") != "SUPPORTED":
            issues.append(f"Q088:{language}:UNRESOLVED_{q088.get('answerability_status')}")
        if one("Q095").get("answerability_status") != "SUPPORTED":
            issues.append(f"Q095:{language}:COLLATERAL_EQUIVALENCE")
        if one("Q081").get("answerability_status") != "SUPPORTED":
            issues.append(f"Q081:{language}:NONE_NIL_REGRESSION")
        if one("Q020").get("answerability_status") != "SUPPORTED":
            issues.append(f"Q020:{language}:MARK_DISTRIBUTION_REGRESSION")
        if one("Q014").get("answerability_status") != "SUPPORTED":
            issues.append(f"Q014:{language}:EMAIL_REGRESSION")
        if one("Q016").get("answerability_status") == "SUPPORTED" and "2 semester" in str(one("Q016").get("final_answer") or "").casefold():
            issues.append(f"Q016:{language}:WRONG_RELATION")
        if one("Q031").get("answerability_status") == "AMBIGUOUS_QUERY":
            issues.append(f"Q031:{language}:FALSE_AMBIGUITY")
    for row in rows:
        if row.get("runtime_error"):
            issues.append(f"{row['question_id']}:{row['language']}:RUNTIME_ERROR")
        if row.get("answerability_status") == "SUPPORTED" and row.get("answer_safety_failures"):
            issues.append(f"{row['question_id']}:{row['language']}:UNSAFE_RETURN")
    return issues


def run() -> None:
    if RERANKER_ENABLED:
        raise RuntimeError("Step 8D requires reranker OFF")
    selected = [row for row in load_dataset() if row["question_id"] in TARGET_IDS]
    if len(selected) != 24:
        raise RuntimeError("Expected 24 targeted language variants")
    current = fingerprint()
    existing = []
    if CHECKPOINT.exists():
        existing = [json.loads(line) for line in CHECKPOINT.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row["run_fingerprint"] != current for row in existing):
        raise RuntimeError("Step 8D checkpoint fingerprint mismatch; preserve it before restarting")
    completed = {(row["question_id"], row["language"]): row for row in existing}
    if len(completed) != len(existing):
        raise RuntimeError("Duplicate Step 8D checkpoint key")
    for row in selected:
        key = row["question_id"], row["expected_language"]
        if key in completed:
            continue
        record = measure(row, current)
        with CHECKPOINT.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        completed[key] = record
        print(f"{len(completed)}/24 {key[1]} {key[0]} {record['answerability_status']} "
              f"{record['total_seconds']:.1f}s", flush=True)
    ordered = [completed[(row["question_id"], row["expected_language"])] for row in selected]
    issues = gate(ordered)
    (RESULTS / "step8d_targeted_gate.json").write_text(json.dumps(
        {"passed": not issues, "issues": issues, "completed": len(ordered), "fingerprint": current},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"STEP8D_GATE {'PASS' if not issues else 'FAIL'} {issues}", flush=True)
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    run()
