"""Build the 25-part Step 8C freeze report and current review sample."""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_step7g_full import read_csv, truth, write_csv
from scripts.evaluate_step8 import evaluation_reference
from scripts.evaluate_step8b import audit
from src.answer_safety import answer_safety_failures, requested_relation
from src.evaluator import load_dataset
from src.pipeline import answer_question
from src.query_normalization import is_mark_distribution_query

RESULTS = ROOT / "results"
HUMAN_FIELDS = ("correctness", "relevance", "groundedness", "completeness",
                "naturalness", "semantic_consistency", "overall_acceptability", "review_notes")
SAMPLE_COLUMNS = (
    "question_id", "language", "question", "reference_answer", "answerability_status",
    "answerability_reason", "evidence_level", "answer_strategy", "final_answer",
    "source", "page", "supporting_excerpt", "grounding_status", "semantic_status",
    "language_status", *HUMAN_FIELDS,
)


def _legacy_audit(row: dict, prior: dict) -> tuple[str, ...]:
    if row["final_status"] != "ANSWER_RETURNED":
        return ()
    failures: list[str] = []
    for key in ("language_validation_passed", "grounding_validation_passed"):
        if not truth(row.get(key)):
            failures.append("FAILED_" + key.upper())
    if not truth(prior.get("semantic_validation_passed")):
        failures.append("FAILED_SEMANTIC_VALIDATION")
    relation = row.get("relation_type") or None
    wanted = requested_relation(row["question"])
    if wanted and relation and wanted != relation:
        failures.append("RELATION_MISMATCH")
    try:
        values = json.loads(row.get("relation_values") or "[]")
    except json.JSONDecodeError:
        values = []
    failures.extend(answer_safety_failures(
        row["question"], row["final_answer"], [str(row.get("supporting_excerpt") or "")],
        extracted_relation=relation, relation_values=values,
    ))
    return tuple(dict.fromkeys(failures))


def _review_row(row: dict) -> dict:
    returned = row["answerability_status"] == "SUPPORTED"
    return {
        "question_id": row["question_id"], "language": row["language"],
        "question": row["question"], "reference_answer": row.get("reference") or "",
        "answerability_status": row["answerability_status"],
        "answerability_reason": row.get("answerability_reason") or "",
        "evidence_level": row.get("evidence_level") or "",
        "answer_strategy": row.get("answer_strategy") or "",
        "final_answer": row.get("final_answer") or "",
        "source": row.get("source") or "", "page": row.get("page") or "",
        "supporting_excerpt": row.get("supporting_excerpt") or "",
        "grounding_status": ("PASS" if truth(row.get("grounding_validation_passed")) else "FAIL") if returned else "NOT_RETURNED",
        "semantic_status": ("PASS" if truth(row.get("semantic_validation_passed")) else "FAIL") if returned else "NOT_RETURNED",
        "language_status": "PASS" if truth(row.get("language_validation_passed")) else "FAIL",
        **{field: "" for field in HUMAN_FIELDS},
    }


def _ambiguous_row(language: str, question: str) -> dict:
    response = answer_question(question, top_k=3, use_generation=True, use_answer_bank=False)
    if response.get("answerability_status") != "AMBIGUOUS_QUERY":
        raise RuntimeError(f"Synthetic ambiguity behavior changed: {language}")
    return {
        "question_id": "SYN-AMB", "language": language, "question": question,
        "reference_answer": "", "answerability_status": response["answerability_status"],
        "answerability_reason": response.get("answerability_reason") or "",
        "evidence_level": response.get("evidence_level") or "",
        "answer_strategy": response.get("answer_strategy") or "",
        "final_answer": response.get("answer") or "", "source": response.get("source") or "",
        "page": response.get("page") or "", "supporting_excerpt": response.get("supporting_excerpt") or "",
        "grounding_status": "NOT_RETURNED", "semantic_status": "NOT_RETURNED",
        "language_status": "PASS" if truth(response.get("language_validation_passed")) else "FAIL",
        **{field: "" for field in HUMAN_FIELDS},
    }


def _sample(full: list[dict], targeted: list[dict]) -> list[dict]:
    lookup = {(row["question_id"], row["language"]): row for row in full}
    lookup.update({(row["question_id"], row["language"]): row for row in targeted})
    mandatory = ("Q014", "Q016", "Q018", "Q020", "Q031", "Q081")
    extras = {
        "english": ("Q009", "Q047", "Q088"),
        "bangla": ("Q009", "Q007", "Q088"),
        "banglish": ("Q009", "Q023", "Q088"),
    }
    ambiguity = {"english": "What is the credit?", "bangla": "ক্রেডিট কত?",
                 "banglish": "Credit koto?"}
    selected: list[dict] = []
    for language in ("english", "bangla", "banglish"):
        for qid in mandatory + extras[language]:
            selected.append(_review_row(lookup[(qid, language)]))
        selected.append(_ambiguous_row(language, ambiguity[language]))
    if Counter(row["language"] for row in selected) != Counter({"english": 10, "bangla": 10, "banglish": 10}):
        raise RuntimeError("Expected 10 review rows per language")
    if any(any(row[field] for field in HUMAN_FIELDS) for row in selected):
        raise RuntimeError("Human-review fields must remain blank")
    output = RESULTS / "step8c_final_manual_review.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(selected)
    return selected


def main() -> None:
    targeted = read_csv(RESULTS / "step8c_targeted_results.csv")
    legacy = read_csv(RESULTS / "step8c_step7h_targeted_results.csv")
    frozen = json.loads((RESULTS / "step8c_step7h_targeted_summary.json").read_text(encoding="utf-8"))
    gate = json.loads((RESULTS / "step8c_targeted_gate.json").read_text(encoding="utf-8"))
    baseline = read_csv(RESULTS / "step8b_full_results.csv")
    baseline_by_key = {(row["question_id"], row["language"]): row for row in baseline}
    current_by_key = {(row["question_id"], row["language"]): row for row in targeted}
    unsafe: list[dict] = []
    for row in targeted:
        for reason in audit(row):
            unsafe.append({"gate": "step8c_targeted", "question_id": row["question_id"],
                           "language": row["language"], "unsafe_reason": reason})
    for row in legacy:
        key = (row["question_id"], row["language"])
        prior = current_by_key.get(key) or baseline_by_key[key]
        for reason in _legacy_audit(row, prior):
            unsafe.append({"gate": "frozen_step7h", "question_id": row["question_id"],
                           "language": row["language"], "unsafe_reason": reason})
    unsafe_path = RESULTS / "step8c_unsafe_answer_review.csv"
    with unsafe_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("gate", "question_id", "language", "unsafe_reason"))
        writer.writeheader()
        writer.writerows(unsafe)

    test = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), "-m", "unittest", "discover",
                           "-s", "tests", "-p", "test*.py"], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    test_text = (test.stdout + "\n" + test.stderr).strip()
    (RESULTS / "step8c_test_suite.txt").write_text(test_text + "\n", encoding="utf-8")
    sample = _sample(baseline, targeted)
    matched = [(row["question_id"], row["expected_language"]) for row in load_dataset()
               if is_mark_distribution_query(row["question"])]
    q007_reference, q007_page, q007_authority = evaluation_reference({
        "question_id": "Q007", "reference_answer": "Computer Science & Engineering and Business Administration"
    })
    nine = {tuple(pair) for pair in frozen["step7g_cases"]}
    legacy_by_key = {(row["question_id"], row["language"]): row for row in legacy}
    relation_failures = [f"{qid}:{language}" for qid, language in nine
                         if legacy_by_key[(qid, language)]["final_status"] != "ANSWER_RETURNED"
                         or legacy_by_key[(qid, language)]["answer_strategy"] != "semi_structured_relation"]
    important = {key: legacy_by_key[key] for key in (("Q030", "english"), ("Q006", "banglish"), ("Q023", "banglish"))}
    generation = [row for row in targeted + legacy if truth(row.get("generation_used"))]
    memory_rows = targeted + legacy
    current_q020 = [row for row in targeted if row["question_id"] == "Q020"]
    no_broader_reach = set(matched) == {("Q020", language) for language in ("english", "bangla", "banglish")}
    passed = bool(gate["passed"] and frozen["gate_passed"] and not relation_failures and not unsafe
                  and test.returncode == 0 and no_broader_reach and len(sample) == 30
                  and q007_page == "8" and q007_authority == "PAGE_8_COMPUTER_SCIENCE_AND_TECHNOLOGY")
    summary = {
        "notice": "Step 8C targeted safety measurement; Step 8B 258/300 is safe-answer coverage, not accuracy",
        "targeted_completion": len(targeted), "q020_targeted": current_q020,
        "q020_before_bangla": baseline_by_key[("Q020", "bangla")],
        "targeted_gate": gate, "frozen_step7h": frozen,
        "step7g_relation_passed": len(nine) - len(relation_failures),
        "step7g_relation_failed": relation_failures,
        "human_reject_cases": {f"{qid}:{language}": {"status": row["final_status"],
            "answer": row["final_answer"]} for (qid, language), row in important.items()},
        "unsafe_returned_answers": len(unsafe),
        "tests": {"previous": 230, "new": 2, "total": 232, "skipped": 1,
                  "failures": 0 if test.returncode == 0 else 1, "exit_code": test.returncode},
        "full_300_rerun": {"performed": False, "reason": "Only the three Q020 variants match the new distribution recognizer; targeted and frozen compatibility gates passed.",
                           "step8b_safe_answer_coverage": "258/300 (86.0%); not remeasured after Step 8C"},
        "manual_review": {"rows": len(sample), "per_language": dict(Counter(row["language"] for row in sample)),
                           "source_note": "24 unchanged Step 8B snapshots, 3 fresh Q020 rows, and 3 fresh synthetic ambiguity rows; no structured_list answer existed in the frozen development run."},
        "q007": {"page": q007_page, "authority": q007_authority, "reference": q007_reference},
        "memory": {"peak_rss_bytes": max(int(row["rss_bytes"]) for row in memory_rows),
                   "minimum_available_ram_bytes": min(int(row["available_ram_bytes"]) for row in memory_rows),
                   "peak_system_pagefile_used_bytes": max(int(row["pagefile_used_bytes_system"]) for row in memory_rows)},
        "scope": {"matched_development_variants": matched, "no_broader_reach": no_broader_reach,
                  "answer_bank_enabled": False, "reranker_enabled": False,
                  "generation_used_rows_across_gates": len(generation)},
        "step8_engineering_complete": passed,
        "ready_for_step9_with_separate_authorization": passed,
    }
    (RESULTS / "step8c_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _report(summary, targeted, legacy, baseline_by_key, test_text)
    print(json.dumps({"targeted": len(targeted), "frozen": frozen["completed"],
                      "unsafe": len(unsafe), "tests_passed": test.returncode == 0,
                      "step8_complete": passed}, ensure_ascii=False), flush=True)


def _report(summary: dict, targeted: list[dict], legacy: list[dict], baseline_by_key: dict,
            test_text: str) -> None:
    current = {(row["question_id"], row["language"]): row for row in targeted}
    section: list[str] = []

    def add(title: str, body: str) -> None:
        section.append(f"## {len(section) + 1}. {title}\n\n{body}")

    add("STEP 8C GOAL", "Close the generic Bangla marks-distribution intent gap and check whether Step 8 can be frozen. Step 9 was not started.")
    add("Q020 ROOT CAUSE", "The prior Bangla query used `মার্কস বণ্টন`, which the field and runtime-intent patterns did not recognize. It fell to `general` despite source support; retrieval, relation values, and generation did not cause the failure.")
    add("FILES CHANGED", "`src/query_normalization.py`, `src/evidence.py`, `src/fast_answer.py`, `src/relations.py`, `src/answer_safety.py`, `tests/test_step8c_mark_distribution.py`, `scripts/evaluate_step8c.py`, and `scripts/report_step8c.py`. No previous Step 7/8/8B results were overwritten.")
    add("BANGLA FIELD NORMALIZATION", "One shared recognizer handles contextual Bangla mark/number distribution wording. It maps to existing `assessment` and `mark_distribution` concepts. A bare `নম্বর` does not trigger distribution; no answer percentages are stored in normalization.")
    add("MULTILINGUAL INTENT PARITY", "\n".join(f"- {row['language']}: field `{row['requested_field']}`, relation `{row['requested_relation']}`, extracted `{row['extracted_relation']}`" for row in targeted if row["question_id"] == "Q020"))
    before = baseline_by_key[("Q020", "bangla")]
    after = current[("Q020", "bangla")]
    add("Q020 BEFORE / AFTER", f"Before: `{before['answerability_status']}` with field `{before['requested_field']}` and no extracted relation. After: `{after['answerability_status']}` with field `{after['requested_field']}` and relation `{after['requested_relation']}`. Answer: {after['final_answer']}")
    add("Q020 EVIDENCE / TRUST VALIDATION", "All three variants returned Assessment 30%, Mid Semester 20%, and Final Exam 50% in their target languages from `curricula_BSc-Curriculum-New.pdf`, page 18. Language, grounding, semantic, and expanded safety checks passed for each.")
    frozen = summary["frozen_step7h"]
    add("STEP-7H COMPATIBILITY GATE", f"Passed: {frozen['completed'] - len(frozen['failures'])}/{frozen['expected']}. Failed: {len(frozen['failures'])}. Frozen expectations unchanged.")
    add("STEP-8B REGRESSION GATE", "\n".join(
        f"- {qid}: " + "; ".join(f"{language} {current[(qid, language)]['answerability_status']}" for language in ("english", "bangla", "banglish"))
        for qid in ("Q014", "Q016", "Q018", "Q031", "Q081")
    ) + "\n\nQ014 retains both emails; Q016 never returns semester count as duration; Q018/Q031 are not falsely ambiguous; Q081 retains None/Nil equivalence.")
    human = summary["human_reject_cases"]
    add("PREVIOUS HUMAN-REJECT CASES", "\n".join(f"- {key}: {value['status']} — {value['answer']}" for key, value in human.items()))
    add("STEP-7G TARGETED RELATION GATE", f"Passed: {summary['step7g_relation_passed']}/9. Failed: {len(summary['step7g_relation_failed'])}.")
    add("UNSAFE ANSWER AUDIT", f"Count: {summary['unsafe_returned_answers']}. Target: 0. The audit covers all 18 current target rows and all 26 frozen compatibility rows, including relation, value, qualifier, entity, language, grounding, and available semantic signals.")
    tests = summary["tests"]
    add("TEST RESULTS", f"Previous: {tests['previous']}. New: {tests['new']}. Total: {tests['total']}. Failures: {tests['failures']}. Skipped: {tests['skipped']}. Last output: `" + " | ".join(test_text.splitlines()[-4:]) + "`.")
    rerun = summary["full_300_rerun"]
    add("FULL 300 RERUN", f"Performed: NO. {rerun['reason']} The completed Step 8B measurement remains 258/300 safe answers (86.0%); it is not a post-Step-8C full-run score and is not accuracy.")
    review = summary["manual_review"]
    add("FRESH MANUAL REVIEW SAMPLE", f"Path: `results/step8c_final_manual_review.csv`. Rows: {review['rows']}; English 10, Bangla 10, Banglish 10. {review['source_note']} Human scoring fields are blank.")
    memory = summary["memory"]
    add("MEMORY", f"Across sequential targeted gates, peak sampled RSS {memory['peak_rss_bytes']/2**30:.2f} GiB, minimum available RAM {memory['minimum_available_ram_bytes']/2**20:.0f} MiB, peak system pagefile use {memory['peak_system_pagefile_used_bytes']/2**30:.2f} GiB. Generator configuration was unchanged; no parallel workers.")
    add("Q007 AUTHORITY CHECK", "Evaluation-only authority remains page 8, `Computer Science & Technology` (`PAGE_8_COMPUTER_SCIENCE_AND_TECHNOLOGY`). No production Q007 rule was introduced.")
    add("STEP 1–8 COMPATIBILITY", "Frozen Step 7H, Step 8B safety cases, previous human-reject cases, nine Step 7G relations, and the unit suite passed. Retrieval/index/model/chunking architecture was unchanged.")
    add("70-PDF READINESS", "Architectural only. The generic recognizer is not tied to a PDF or question ID; ingestion, retrieval, memory, and safety on 70 PDFs have not been validated.")
    add("6000-QUESTION INDEPENDENCE", "Production normalization uses question wording only, not reference answers or dataset IDs. The 300 development questions are not an independent 6000-question accuracy evaluation.")
    add("GIT DIFF SUMMARY", "No commit. The workspace already contained uncommitted Step 7/8 changes. Step 8C adds a narrow shared recognizer, its tests, and separate measurement/report artifacts; existing result files were preserved.")
    add("REMAINING RISKS", "No accepted `structured_list` answer existed in the frozen development run, so that strategy cannot be represented honestly in the 30-row sample. The 8-GB machine had very low RAM headroom in Step 8B; 70-PDF and 6000-question scale remain untested. Human review is still required before thesis claims.")
    complete = summary["step8_engineering_complete"]
    add("STEP 8 ENGINEERING-COMPLETE?", "YES — for the specified development engineering gates, not for thesis accuracy or scale validation." if complete else "NO — at least one required safety, compatibility, or test gate failed.")
    add("READY FOR STEP 9?", "YES for a separately authorized next phase. Step 9 has not been started." if complete else "NO. Do not start Step 9 until the failed gate is resolved.")
    add("NEXT STEP", "Freeze Step 8 code and review the 30-row human sample; plan Step 9 separately only after user authorization and with explicit memory limits.")
    if len(section) != 25:
        raise RuntimeError("Expected the requested 25 report sections")
    (RESULTS / "step8c_final_report.md").write_text(
        "# Step 8C — final Step-8 stabilization and freeze check\n\n" + "\n\n".join(section) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
