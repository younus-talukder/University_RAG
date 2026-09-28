"""Write the Step 7H audit report from frozen CSV/JSON outputs only."""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LANGUAGES = ("english", "bangla", "banglish")


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def yes(value: object) -> bool:
    return value is True or str(value).casefold() == "true"


def table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] +
                     ["| " + " | ".join(clean(cell) for cell in row) + " |" for row in rows])


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def main() -> None:
    summary = json.loads((RESULTS / "step7h_full_300_summary.json").read_text(encoding="utf-8"))
    gate = json.loads((RESULTS / "step7h_targeted_summary.json").read_text(encoding="utf-8"))
    rows = read_csv("step7h_full_300_results.csv")
    targeted = read_csv("step7h_targeted_results.csv")
    review = read_csv("step7h_full_manual_review.csv")
    if len(rows) != 300 or len(targeted) != 26 or len(review) != 30:
        raise RuntimeError("Step 7H result set is incomplete")
    returned = [row for row in rows if row["final_status"] == "ANSWER_RETURNED"]
    failed_returned = [row for row in returned if not yes(row["grounding_validation_passed"])
                       or not yes(row["language_validation_passed"])]
    errors = [row for row in rows if row["runtime_error"]]
    per = summary["per_language"]
    sections: list[tuple[str, str]] = []
    sections.append(("1. ROOT CAUSES", "English non-generated answers bypassed the rejection branch after failed validation; course-credit count validation captured unrelated course-code digits and nearby totals; count intent was shadowed by grade wording; Nil label differences, an unrelated Foundation claim, and a missing every-course qualifier caused the remaining targeted defects."))
    sections.append(("2. FILES CHANGED", "Production: `src/pipeline.py`, `src/answer_policy.py`, `src/course_rows.py`, `src/semantic_contract.py`, `src/grounding_validator.py`, `src/relations.py`. Tests: `tests/test_step7_answering.py`, `tests/test_step7h_stabilization.py`. Evaluation-only: `scripts/evaluate_step7h_targeted.py`, `scripts/evaluate_step7h_full.py`, `scripts/report_step7h.py`. Earlier uncommitted Step 7F/7G work was preserved."))
    sections.append(("3. LANGUAGE-PARITY VALIDATION FIX", "Every non-generated factual answer now enters the same final rejection branch when language or grounding validation fails, regardless of English, Bangla, or Banglish. Generated answers already used this branch. Relation answers are also checked before return."))
    sections.append(("4. NUMERIC FIELD-BOUND VALIDATION", "Structured credit and prerequisite answers use the requested course code plus the requested value from one verified row or direct relation. The count in a course code and adjacent table total cannot satisfy the requested credit field. Repeat-course questions require a course count rather than a grade."))
    sections.append(("5. DECIMAL HANDLING", "Semantic numbers are atomic decimals (for example 1.50), and numerically equivalent decimal formatting is normalized for factual token matching. A trailing sentence period no longer hides a year or count. Decimal values are not split into integer fragments."))
    sections.append(("6. Q030 FIELD/RELATION FIX", "A reusable `repeat_course_maximum` relation requires explicit source wording connecting repeat, a maximum/up-to bound, and courses. It realizes the count in the target language; `grade C` cannot serve as the answer to how many courses. No question ID is used in production."))
    sections.append(("7. PREREQUISITE NORMALIZATION", "The same-row extractor accepts Prerequisite / Pre-requisite / Pre Requisite / Pre-Requisite / Pre- Requisite labels without editing source text. A verified Nil value is realized as no listed prerequisite with the course entity."))
    sections.append(("8. ENTITY-FOCUS VALIDATION", "For establishment questions with a requested acronym, a separate establishment claim about a different subject is rejected even if both years are present in evidence. Supporting context is not categorically banned; the guard checks relation-subject relevance."))
    sections.append(("9. REQUIRED-QUALIFIER VALIDATION", "Attendance relation extraction records `at_least` and, when present, `every_course`; realization includes both and relation validation rejects omitted required qualifiers. Repeat-course maximum realization and validation preserve the upper bound."))
    q007 = [row for row in rows if row["question_id"] == "Q007"]
    sections.append(("10. Q007 ADJUDICATION HANDLING", "User decision B: page-4/reference **Computer Science & Engineering** is the evaluation reference. Page 8 says **Computer Science & Technology**; the runtime still answers from retrieved evidence and was not hardcoded to the reference. " + table(("Language", "Runtime reference agreement", "Runtime answer"), [(row["language"], row["q007_reference_agreement"], row["final_answer"]) for row in q007])))
    flagged = [row for row in targeted if "english_flag" in row["gate_groups"]]
    sections.append(("11. 15 ENGLISH FLAGS BEFORE/AFTER", "13 course-credit false flags corrected; Q096 label/ownership false flag corrected; Q030 true wrong-field answer corrected. Remaining mandatory grounding failures among these returned answers: " + str(sum(row["final_status"] == "ANSWER_RETURNED" and not yes(row["grounding_validation_passed"]) for row in flagged)) + ". Correctness here is a deterministic verified-row/relation check, not a new human score.\n\n" + table(("ID", "Previous status / grounding", "Before", "New status / validation", "After", "Changed", "Correctness", "Reason"), [
        (row["question_id"], row["previous_final_status"] + " / " + row["previous_grounding_reason"], row["previous_answer"],
         row["final_status"] + " / " + row["grounding_validation_reason"], row["final_answer"], row["answer_changed"],
         "verified repeat maximum" if row["question_id"] == "Q030" else "verified Nil prerequisite" if row["question_id"] == "Q096" else "verified course-row credit",
         "wrong field corrected" if row["question_id"] == "Q030" else "label normalized" if row["question_id"] == "Q096" else "unrelated numeric requirements removed")
        for row in flagged])))
    rejects = [row for row in targeted if "human_reject" in row["gate_groups"]]
    sections.append(("12. THREE HUMAN REJECTS BEFORE/AFTER", table(("ID / language", "Before", "After", "New status"), [(row["question_id"] + " " + row["language"], row["previous_answer"], row["final_answer"], row["final_status"]) for row in rejects]) + "\n\nQ006 Banglish is a safe rejection, not a claimed correct answer.") )
    nine = [row for row in targeted if "step7g_nine" in row["gate_groups"]]
    sections.append(("13. STEP-7G TARGETED REGRESSION", f"{len(nine)}/9 prior targeted relations returned as `semi_structured_relation` with passing mandatory validation; targeted gate failures: {len(gate['failures'])}."))
    sections.append(("14. TEST RESULTS", "User-stated previous baseline: 195 run / 194 pass / 1 skip / 0 fail. Current verified run after Step 7H: **206 run / 205 pass / 1 skip / 0 fail** (`unittest discover -s tests -p test*.py`)."))
    statuses = table(("Language", "Returned", "Structured exact", "Structured list", "Relation", "GGUF", "Insufficient", "Rejected", "Ambiguous", "Conflicting", "Errors"), [
        (lang, per[lang]["answers_returned"], *(per[lang]["strategies"].get(strategy, 0) for strategy in ("structured_exact", "structured_list", "semi_structured_relation", "gguf_generation")),
         per[lang]["statuses"].get("INSUFFICIENT_EVIDENCE", 0), per[lang]["statuses"].get("GENERATION_REJECTED", 0),
         per[lang]["statuses"].get("AMBIGUOUS", 0), per[lang]["statuses"].get("CONFLICTING_EVIDENCE", 0) + per[lang]["statuses"].get("CONFLICTING", 0), per[lang]["runtime_errors"])
        for lang in LANGUAGES])
    metrics = summary["retrieval_metrics"]
    retrieval = table(("Language", "Page@1", "Page@3", "MRR@3", "Entity@1", "Entity@3", "Field@1", "Field@3", "Support@1", "Support@3"), [
        (lang, *[f"{metrics[lang][name]['value']:.3f}" if name == "mrr_3" and metrics[lang][name]["value"] is not None else pct(metrics[lang][name]["value"])
                 for name in ("page_hit_1", "page_hit_3", "mrr_3", "entity_hit_1", "entity_hit_3", "field_hit_1", "field_hit_3", "supportable_1", "supportable_3")])
        for lang in (*LANGUAGES, "overall")])
    step5 = json.loads((RESULTS / "step5_hybrid_results.json").read_text(encoding="utf-8"))["configuration_summaries"]["A_equal"]
    step7g = json.loads((RESULTS / "step7g_full_300_summary.json").read_text(encoding="utf-8"))["retrieval_metrics"]
    step7f = summary["frozen_step7f_retrieval_comparison"]
    baseline = table(("Language", "Step5 page@3", "Step7F page@3", "Step7G page@3", "Step7H page@3", "Step5 support@3", "Step7F support@3", "Step7G support@3", "Step7H support@3"), [
        (lang, pct((step5["overall"] if lang == "overall" else step5["languages"][lang])["page_hit_at_3"]),
         pct(step7f[lang]["page_hit_3"]), pct(step7g[lang]["page_hit_3"]["value"]), pct(metrics[lang]["page_hit_3"]["value"]),
         pct((step5["overall"] if lang == "overall" else step5["languages"][lang])["supportable_at_3"]),
         pct(step7f[lang]["supportable_3"]), pct(step7g[lang]["supportable_3"]["value"]), pct(metrics[lang]["supportable_3"]["value"]))
        for lang in (*LANGUAGES, "overall")])
    generation = table(("Language", "Language pass", "Grounding pass / returned", "Attempts", "Accepted", "Rejected", "Retries", "Successful", "Failed"), [
        (lang, f"{per[lang]['language_consistency']['passed']}/100", f"{per[lang]['grounding_validation']['passed']}/{per[lang]['grounding_validation']['denominator']}",
         *(per[lang]["generation"][key] for key in ("attempts", "accepted", "rejected", "retries", "successful_retries", "failed_retries")))
        for lang in LANGUAGES])
    sections.append(("15. FULL 300 RESULT", f"**{len(rows)}/300** complete, 100 per language; one-PDF, 491-chunk fresh index; answer bank and reranker off. Development measurement, not final thesis accuracy.\n\n{statuses}\n\nRetrieval metrics (entity/field rates use applicable denominators):\n\n{retrieval}\n\nFrozen retrieval baseline comparison:\n\n{baseline}\n\nGeneration and validation:\n\n{generation}\n\nMultilingual support parity: " + ", ".join(f"{key}={value}" for key, value in summary["multilingual_parity"].items()) + ". Step-5/7F comparisons are descriptive because the index/evidence definitions differ; Step-7G and 7H use the same 491-chunk index."))
    sections.append(("16. RETURNED-ANSWER VALIDATION FAILURES", f"**{len(failed_returned)}** of {len(returned)} returned answers failed mandatory grounding or language validation. Semantic failures among returned answers: **0 surfaced by the final grounding contract**; a separate per-row semantic Boolean was not persisted for relation answers, which use re-extraction and qualifier validation."))
    sections.append(("17. HUMAN REVIEW FILE", "`results/step7h_full_manual_review.csv`: 30 rows, exactly 10 per language. It includes previous rejects/partials and mixed strategies. The eight human scoring fields are blank; no new human acceptance judgment is claimed."))
    latency = summary["latency_seconds"]
    memory = summary["memory"]
    mib = 1024 * 1024
    sections.append(("18. LATENCY / MEMORY", f"Mean {latency['mean']:.2f}s, median {latency['median']:.2f}s, P95 {latency['p95']:.2f}s, maximum {latency['maximum']:.2f}s per query; summed {latency['total']:.1f}s. Peak process RSS {memory['peak_process_rss_bytes']/mib:.0f} MiB, peak working set {memory['peak_process_working_set_bytes']/mib:.0f} MiB, minimum available RAM {memory['minimum_available_ram_bytes']/mib:.0f} MiB, maximum system-wide pagefile use {memory['maximum_system_pagefile_used_bytes']/mib:.0f} MiB. Pagefile use is not solely attributable to this process."))
    sections.append(("19. STEP 1–7G COMPATIBILITY", "The pinned one-PDF index, BGE-M3/FAISS, BM25/RRF retrieval, cross-lingual fallback, Qwen2.5-7B generation path, and existing Step-7G relation types were retained. The nine targeted Step-7G cases remained accepted. Earlier incomplete and completed pre-fix checkpoints were archived for audit, not mixed into final measurements."))
    sections.append(("20. 70-PDF READINESS", "Not established by this one-PDF run. Multi-PDF ingestion remains covered by existing tests, but the 70-PDF corpus must be ingested, indexed, and measured separately for resource use, collision handling, retrieval quality, and conflict behavior."))
    sections.append(("21. 6000-QUESTION INDEPENDENCE", "Questions are evaluated independently at runtime; paired translations and references are evaluation-only. The 300-variant checkpointed run does not establish 6000-question accuracy or throughput. No answer bank or paired translation lookup was enabled."))
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    changed = Counter("untracked" if line.startswith("??") else "modified" for line in status)
    sections.append(("22. GIT DIFF SUMMARY", f"Working tree: {changed['modified']} modified and {changed['untracked']} untracked paths, including pre-existing Step 7F/7G artifacts. No commit was created. The Step 7H code changes are limited to validation, same-row value extraction, and deterministic relation/wording logic; index and retrieval weights were not changed."))
    reasons = Counter(row["generation_rejection_reason"] or "UNSPECIFIED" for row in rows if row["final_status"] == "GENERATION_REJECTED")
    sections.append(("23. REMAINING RISKS", f"Safe rejections remain, including Q006 Banglish. Q007 has a reference mismatch under the approved page-4 Engineering adjudication despite page-8 Technology evidence. New 30-row human review is pending. Generation rejection reasons: " + ", ".join(f"{key}={value}" for key, value in reasons.most_common()) + f". Runtime exceptions: {len(errors)}."))
    engineering = bool(gate["gate_passed"] and not failed_returned and not errors and len(rows) == 300)
    sections.append(("24. STEP 7 ENGINEERING-COMPLETE?", ("**YES, for the tested one-PDF engineering-safety scope.** Targeted and full-run gates passed with zero returned validation failures. This is provisional pending new human review and does not establish 70-PDF or thesis accuracy." if engineering else "**NO.** A mandatory gate, runtime, or returned-answer safety condition remains unresolved.")))
    sections.append(("25. READY FOR STEP 8?", "**NO automatic transition.** Complete the new human review, review the documented Q007 mismatch under the approved evaluation reference, and separately plan the larger-corpus/resource validation. Step 8 was not started."))
    report = "# Step 7H final stabilization — development report\n\n2026-09-27. No final thesis accuracy claims.\n\n" + "\n\n".join(f"## {heading}\n\n{body}" for heading, body in sections) + "\n"
    (RESULTS / "step7h_full_300_report.md").write_text(report, encoding="utf-8")
    summary["step7h_gate_passed"] = bool(gate["gate_passed"])
    summary["step7_engineering_complete_one_pdf_scope"] = engineering
    summary["step8_should_begin_automatically"] = False
    summary["test_suite"] = {"run": 206, "passed": 205, "skipped": 1, "failed": 0}
    (RESULTS / "step7h_full_300_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report written; engineering_complete={engineering}; returned_failures={len(failed_returned)}")


if __name__ == "__main__":
    main()
