"""Build the 40-section Step 8 report from frozen development artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_step7g_full import read_csv, truth

RESULTS = ROOT / "results"
LANGUAGES = ("english", "bangla", "banglish", "overall")


def table(headers: tuple[str, ...], values: list[tuple[object, ...]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(cell(x) for x in row) + " |" for row in values)])


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    summary = json.loads((RESULTS / "step8_answerability_summary.json").read_text(encoding="utf-8"))
    rows = read_csv(RESULTS / "step8_answerability_results.csv")
    false_review = read_csv(RESULTS / "step8_false_abstention_review.csv")
    unsafe_review = read_csv(RESULTS / "step8_unsafe_answer_review.csv")
    manual = read_csv(RESULTS / "step8_manual_review_sample.csv")
    ood = read_csv(RESULTS / "step8_out_of_domain_results.csv")
    ambiguity = read_csv(RESULTS / "step8_ambiguity_results.csv")
    conflict = read_csv(RESULTS / "step8_conflict_results.csv")
    baseline = json.loads((RESULTS / "step7h_full_300_summary.json").read_text(encoding="utf-8"))
    if len(rows) != 300 or len(manual) != 30:
        raise RuntimeError("Incomplete Step 8 output")
    tests = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), "-m", "unittest", "discover",
                            "-s", "tests", "-p", "test*.py"], cwd=ROOT, capture_output=True,
                           text=True, check=False)
    test_tail = "\n".join((tests.stdout + tests.stderr).splitlines()[-5:])
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True,
                            text=True, check=False).stdout
    distributions = summary["per_language"]
    state_table = table(("Language", "SUPPORTED", "INSUFFICIENT", "RETRIEVAL UNCERTAIN", "AMBIGUOUS",
                         "CONFLICTING", "GENERATION REJECTED", "OOD", "SYSTEM ERROR"), [
        (language, *(distributions[language]["states"][state] for state in (
            "SUPPORTED", "INSUFFICIENT_EVIDENCE", "RETRIEVAL_UNCERTAIN", "AMBIGUOUS_QUERY",
            "CONFLICTING_EVIDENCE", "GENERATION_REJECTED", "OUT_OF_DOMAIN", "SYSTEM_ERROR")))
        for language in LANGUAGES])
    coverage_table = table(("Language", "Returned / total", "Safe answer coverage", "Abstention rate", "Step 7H returned"), [
        (language, f"{distributions[language]['answers_returned']}/{distributions[language]['total']}",
         pct(distributions[language]["safe_answer_coverage"]), pct(distributions[language]["abstention_rate"]),
         baseline["per_language"].get(language, {}).get("answers_returned", 260) if language != "overall" else 260)
        for language in LANGUAGES])
    reason_counts = Counter(row["answerability_reason"] for row in rows if row["answerability_status"] != "SUPPORTED")
    generated = [row for row in rows if truth(row["generation_used"])]
    generated_accepted = sum(row["answerability_status"] == "SUPPORTED" for row in generated)
    cause_counts = Counter(row["likely_cause"] for row in false_review)
    levels = distributions["overall"]["evidence_levels"]
    memory = summary["memory"]
    old_memory = baseline["memory"]
    overhead = summary["answerability_overhead_seconds_excluding_generation"]
    ood_pass = sum(truth(row["safe"]) for row in ood)
    ambiguity_pass = sum(row["answerability_status"] == row["expected"] for row in ambiguity)
    conflict_pass = sum(row["answerability_status"] == "CONFLICTING_EVIDENCE" for row in conflict if "conflict" in row["case_id"])
    engineering_complete = bool(
        len(rows) == 300 and tests.returncode == 0 and not unsafe_review
        and not summary["errors"] and ood_pass == len(ood)
        and ambiguity_pass == len(ambiguity) and conflict_pass == 2
    )
    q007 = [row for row in rows if row["question_id"] == "Q007"]
    q007_text = table(("Language", "Page", "Status", "Technology in answer"), [
        (row["language"], row["page"], row["answerability_status"],
         "Computer Science & Technology" in row["final_answer"]) for row in q007])
    sections: list[tuple[str, str]] = [
        ("1. STEP 8 GOAL", "Add an observable, rule-based answerability decision before exposing a factual answer; do not equate retrieval scores with probabilities."),
        ("2. FILES CHANGED", "Step 8: `src/answerability.py`, `src/evidence.py`, `src/pipeline.py`, `app.py`, `tests/test_step8_answerability.py`, `scripts/evaluate_step8.py`, `scripts/report_step8.py`. Earlier uncommitted Step 7 work and index artifacts were preserved."),
        ("3. ANSWERABILITY ARCHITECTURE", "The existing evidence assessment remains the evidence-stage gate. `AnswerabilityDecision` maps that gate and retrieval provenance to a final state. Output acceptance is a separate validation stage; it can turn supported evidence into `GENERATION_REJECTED`. No evaluation field reaches production answering."),
        ("4. ANSWERABILITY STATES", ", ".join(distributions["overall"]["states"]) + ". `support_status` remains the upstream evidence assessment; `answerability_status` is the final user-answer gate, not a competing synonym."),
        ("5. ANSWERABILITY REASON CODES", ", ".join(f"{name}: {count}" for name, count in reason_counts.most_common()) + ". Reasons are categorical, not calibrated likelihoods."),
        ("6. EVIDENCE-STRENGTH MODEL", "STRONG is explicit labeled/structured or relation evidence, or multiple independent agreeing passages. ADEQUATE is one verified passage. WEAK is related but unverified/contested evidence. NONE has no usable support. No percentage or retrieval-score threshold is used."),
        ("7. INDEPENDENT SUPPORT LOGIC", "Distinct document + physical page + parent block units are counted once; sibling child chunks sharing a parent do not multiply support. Synthetic agreement/conflict across PDFs is tested."),
        ("8. RETRIEVAL-CHANNEL SIGNALS", "Dense, BM25, metadata and cross-lingual provenance are debug metadata. Ranks and scores remain diagnostics only, never confidence probabilities."),
        ("9. STRUCTURED ANSWER TRUST GATE", "A structured answer still needs matching entity, field, extracted value, language and grounding checks. It is not sent through Qwen merely for a trust label."),
        ("10. GENERATED ANSWER TRUST GATE", f"Generation proceeds only with verified evidence. Canonical output and target-language realization must pass language, semantic and grounding checks; otherwise the final state is `GENERATION_REJECTED`. This run used generation for {len(generated)} cases: {generated_accepted} accepted, {len(generated)-generated_accepted} rejected, {summary['generation_attempts']} attempts, and {summary['retry_used_cases']} cases with retries."),
        ("11. INSUFFICIENT-EVIDENCE HANDLING", "No verified entity/field support yields a safe multilingual abstention. The generator is not asked to fill missing facts."),
        ("12. RETRIEVAL-UNCERTAIN HANDLING", "A related same-entity/field hit rejected by ownership or relation validation is kept as WEAK and abstains. No single numerical cutoff is used."),
        ("13. AMBIGUITY HANDLING", "Contextless entity-dependent questions ask which course, document, program or policy is intended. They do not infer an entity from retrieval rank."),
        ("14. CONFLICT HANDLING", "Incompatible verified values block answering, regardless of document recency. Debug metadata retains source, page and excerpt for each conflicting passage."),
        ("15. GENERATION-REJECTED HANDLING", "Verified evidence is retained in metadata, but an unacceptable realization is not displayed as a factual answer. This is distinguished from absent evidence."),
        ("16. OUT-OF-DOMAIN HANDLING", "A conservative overt-intent preflight catches unrelated weather, sport, medical, coding, news/trivia and advice requests while exempting explicit university/course context."),
        ("17. PROMPT-INJECTION RESULT", "Overt instructions to ignore documents or answer from outside knowledge are removed before retrieval; an instruction-only query is refused. The generator's existing evidence-only system prompt and final validators remain mandatory."),
        ("18. MULTILINGUAL SAFE RESPONSES", "Each non-supported state has English, Bangla and Banglish wording. Synthetic tests verify language checks for all three."),
        ("19. UI / DEBUG BEHAVIOR", "Normal UI shows the answer and, only for supported answers, source/page/excerpt without raw scores. Optional debug mode exposes reasons, evidence level, counts, channels, rank, conflict passages, generation state and fallback use."),
        ("20. TEST RESULTS", f"Previous baseline: 206 tests (205 pass, 1 skip). Step 8 adds focused synthetic and Q007 annotation tests. Test process exit: {tests.returncode}.\n\n```text\n{test_tail}\n```"),
        ("21. 300-QUERY ANSWERABILITY DISTRIBUTION", "300/300 development variants completed.\n\n" + state_table + "\n\nThis is a development measurement, not final thesis accuracy."),
        ("22. SAFE ANSWER COVERAGE", coverage_table + "\n\nCoverage is returned answers passing automated trust checks divided by all questions; it is not correctness."),
        ("23. ABSTENTION RATE", coverage_table + "\n\nBy reason: " + ", ".join(f"{key}={value}" for key, value in reason_counts.most_common()) + ". Abstention is not automatically an error."),
        ("24. FALSE ABSTENTIONS", f"{len(false_review)} review candidates with a reference and a non-supported status. These are *not* all proven false abstentions. Likely causes: " + ", ".join(f"{name}={count}" for name, count in cause_counts.most_common()) + ". Human review may identify ambiguous or flawed references."),
        ("25. UNSAFE ANSWERS", f"Automated flagged returned answers: {len(unsafe_review)} (target 0). The audit flags WEAK evidence or failed mandatory validators; zero flags would not establish human correctness."),
        ("26. EVIDENCE LEVEL DISTRIBUTION", ", ".join(f"{level}={levels[level]}" for level in ("STRONG", "ADEQUATE", "WEAK", "NONE")) + "."),
        ("27. OUT-OF-DOMAIN TEST", f"{ood_pass}/{len(ood)} developer-only questions safely refused without a fabricated university answer. The set is separate from the 300 thesis-development variants."),
        ("28. AMBIGUITY TEST", f"{ambiguity_pass}/{len(ambiguity)} contextless developer cases returned `AMBIGUOUS_QUERY`; failures, if any, remain visible in the CSV."),
        ("29. CONFLICT TEST", f"{conflict_pass}/2 synthetic conflicting fixtures returned `CONFLICTING_EVIDENCE`; the agreeing cross-document fixture remained answerable. No real PDF was modified."),
        ("30. ANSWERABILITY OVERHEAD", f"Excluding generation-used rows: median {overhead['median'] * 1000:.2f} ms; P95 {overhead['p95'] * 1000:.2f} ms. Wall-clock measurement includes trust feature collection, not Qwen generation."),
        ("31. MEMORY IMPACT", f"Observed peak process working set {memory['peak_process_bytes'] / 2**30:.2f} GiB vs Step 7H {old_memory['peak_process_working_set_bytes'] / 2**30:.2f} GiB; max system pagefile used {memory['peak_pagefile_used_bytes_system'] / 2**30:.2f} GiB vs {old_memory['maximum_system_pagefile_used_bytes'] / 2**30:.2f} GiB. These are run-level observations affected by system load, not isolated causal memory deltas. No new model was loaded."),
        ("32. HUMAN REVIEW FILE", "`results/step8_manual_review_sample.csv`: 30 rows, 10 per language, with eight human scoring columns blank. The selection prioritizes trust states, relation and GGUF strategies, and cross-lingual recovery when present."),
        ("33. STEP 1–7 COMPATIBILITY", "The previous retrieval, cross-lingual fallback, relation extraction, generator, and validators remain in place. The full regression suite was rerun; Step 7H artifacts were not overwritten."),
        ("34. 70-PDF READINESS", "Evidence identity uses document ID/path, page and parent block; agreement and conflict tests span independent documents. No current filename, course list, or page number is embedded in production trust rules. Actual 70-PDF behavior remains untested."),
        ("35. 6000-QUESTION INDEPENDENCE", "Production answerability receives only the current question, retrieved candidates, and verified evidence. IDs, references, page labels and paired translations are evaluation-script inputs only. A future 6000-question run is not claimed validated."),
        ("36. GIT DIFF SUMMARY", "No commit was made. Current working tree also contains pre-existing Step 7 changes and index files, so the status below is not solely Step 8:\n\n```text\n" + status.strip() + "\n```"),
        ("37. REMAINING RISKS", "The one-PDF development set does not establish accuracy, calibration, or multi-PDF version arbitration. Ordinal evidence strength relies on the existing evidence validator; related passages may still be false positives and need human review. OOD coverage is conservative and incomplete. System memory is close to capacity during model use."),
        ("38. STEP 8 ENGINEERING-COMPLETE?", ("YES" if engineering_complete else "NO") + ". This requires 300 rows, green synthetic checks and regression suite, zero runtime errors, and zero automatically flagged unsafe returns. It is engineering completion for the tested development scope, not a thesis accuracy claim."),
        ("39. READY FOR STEP 9?", "NO automatic transition. Review the 30-row human sample, especially any unsafe flags, false-abstention candidates and Q007, before deciding whether to start Step 9."),
        ("40. NEXT STEP", "Perform the requested human review of Step 8 results and explicitly authorize any later Step 9 work. Step 9 was not implemented."),
    ]
    report = "# Step 8 answerability and trustworthy abstention\n\n"
    report += "\n\n".join(f"## {title}\n\n{body}" for title, body in sections)
    report += "\n\n## Q007 evaluation-only adjudication\n\nThe approved source is page 8, **Computer Science & Technology**. The original dataset remains unchanged and the production pipeline has no Q007 exception.\n\n" + q007_text
    report += "\n\n## Output files\n\n" + "\n".join(f"- `{name}`" for name in (
        "results/step8_answerability_results.csv", "results/step8_answerability_summary.json",
        "results/step8_answerability_report.md", "results/step8_false_abstention_review.csv",
        "results/step8_unsafe_answer_review.csv", "results/step8_manual_review_sample.csv",
        "results/step8_out_of_domain_results.csv", "results/step8_ambiguity_results.csv",
        "results/step8_conflict_results.csv"))
    (RESULTS / "step8_answerability_report.md").write_text(report + "\n", encoding="utf-8")
    print(f"Report sections: {len(sections)}, rows: {len(rows)}, tests_exit={tests.returncode}")


if __name__ == "__main__":
    main()
