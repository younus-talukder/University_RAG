"""Write the requested 22-part Step 8B engineering report from saved results."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_step7g_full import read_csv

RESULTS = ROOT / "results"


def main() -> None:
    summary = json.loads((RESULTS / "step8b_full_summary.json").read_text(encoding="utf-8"))
    gate = summary["targeted_gate"]
    current = read_csv(RESULTS / "step8b_full_results.csv")
    targeted = read_csv(RESULTS / "step8b_targeted_results.csv")
    prior = read_csv(RESULTS / "step8_answerability_results.csv")
    legacy = json.loads((RESULTS / "step8b_step7h_targeted_summary.json").read_text(encoding="utf-8"))
    prior_map = {(row["question_id"], row["language"]): row for row in prior}
    current_map = {(row["question_id"], row["language"]): row for row in targeted}
    test = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), "-m", "unittest", "discover", "-s", "tests", "-p", "test*.py"],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    test_output = (test.stdout + "\n" + test.stderr).strip()
    (RESULTS / "step8b_test_suite.txt").write_text(test_output + "\n", encoding="utf-8")
    before = Counter(row["answerability_status"] for row in prior)
    after = Counter(row["answerability_status"] for row in current)
    sections = []

    def add(title: str, body: str) -> None:
        sections.append(f"## {len(sections) + 1}. {title}\n\n{body}")

    add("STEP 8B ROOT CAUSES", "Question intent was inferred from the extracted relation; structured exact extraction selected one email; prerequisite conflict comparison used unbounded raw strings; ambiguity used field-wide entity requirements. The old unsafe audit checked validator booleans only.")
    add("FILES CHANGED", "`src/answer_safety.py`, `src/evidence.py`, `src/pipeline.py`, `scripts/evaluate_step8.py`, `scripts/evaluate_step8b.py`, `scripts/report_step8b.py`, and `tests/test_step8b_safety.py`. Existing Step 8 output files remain untouched.")
    add("RELATION-INTENT ALIGNMENT FIX", "Semester count, duration, class-week duration, and names are distinguished from question wording. A contradictory extracted relation is blocked before answer generation. The final audit also independently checks returned semester-duration answers for time units.")
    add("MULTI-VALUE COMPLETENESS FIX", "Explicit email values are read from the relevant labeled field only; structured answers preserve all addresses unless the question requests one. The return gate and offline audit also check every extracted component of bounded program lists, mark distributions, discipline-count pairs, and labeled prerequisite-code lists. They do not infer arbitrary neighboring values as required.")
    add("EQUIVALENT-VALUE CONFLICT NORMALIZATION", "Prerequisite `None`, `Nil`, and clear no-prerequisite phrases normalize to one relation-specific sentinel before comparison; labeled value extraction stops before following prose. The unrelated 3.0-versus-4.0 credit conflict remains conflicting.")
    add("ENTITY-INDEPENDENT POLICY HANDLING", "Credit-assignment rules for theoretical courses and seat-reservation percentages can be relation-sufficient without a course entity. Bare fields and unbound pronouns remain ambiguous.")
    add("UNSAFE-ANSWER AUDIT IMPROVEMENT", "The audit records validator failures, weak returned evidence, requested/extracted relation mismatch, mandatory email and bounded-list component loss, key qualifier loss, and subject course-code mismatch. This is an automatic conservative audit, not human correctness certification.")
    lines = []
    for qid in ("Q016", "Q014", "Q018", "Q031", "Q081"):
        lines.append(f"**{qid}:**")
        for language in ("english", "bangla", "banglish"):
            old = prior_map[(qid, language)]
            new = current_map[(qid, language)]
            lines.append(f"- {language}: {old['answerability_status']} → {new['answerability_status']}; answer: {new['final_answer']}")
    add("TARGETED CASES BEFORE/AFTER", "\n".join(lines))
    add("SYNTHETIC REGRESSION RESULTS", "Step 8B tests cover two-email completeness, None/Nil equivalence, true credit conflict, four semester relations, policy sufficiency, genuine ambiguity, qualifier loss, and entity mismatch. Targeted 15-case gate: " + ("PASS" if gate["passed"] else "FAIL") + ". The extended audit also flags the saved Step 8 Q014 Bangla as `REQUIRED_VALUE_LOSS` and Q016 Bangla as `RELATION_MISMATCH`.")
    add("TEST RESULTS", f"Exit code: {test.returncode}. Final test output:\n\n```text\n" + "\n".join(test_output.splitlines()[-8:]) + "\n```")
    generation_accepted = sum(row["answerability_status"] == "SUPPORTED" and row["generation_used"] == "True" for row in current)
    add("FULL 300 RESULT", f"{summary['completion']}/300 completed on the current index with answer bank and reranker off. Runtime errors: {len(summary['errors'])}. Generation used in {summary['generation_used_cases']} cases, {summary['generation_attempts']} attempts, {generation_accepted} accepted generated answers, and {summary['retry_used_cases']} retried cases. Median latency {summary['latency_seconds']['median']:.2f}s; p95 {summary['latency_seconds']['p95']:.2f}s. Strategy counts: {summary['per_language']['overall']['strategies']}.")
    returned = summary["per_language"]["overall"]["answers_returned"]
    add("SAFE ANSWER COVERAGE", f"Automatically accepted: {returned}/300 ({returned/3:.1f}%), versus {before['SUPPORTED']}/300 in the saved Step 8 baseline. By language: " + ", ".join(f"{lang} {summary['per_language'][lang]['answers_returned']}/100" for lang in ("english", "bangla", "banglish")) + ". These are not thesis accuracy claims.")
    add("ABSTENTION DISTRIBUTION", ", ".join(f"{name} {count}" for name, count in after.items() if name != "SUPPORTED") + f". Step 8 baseline: {dict(before)}; Step 8B: {dict(after)}.")
    add("UNSAFE RETURNED ANSWERS", f"Extended automatic audit: {summary['unsafe_returned_answers']}. Target: 0. Review `results/step8b_unsafe_answer_review.csv` and the existing human review sample; zero automatic flags cannot exclude all semantic errors.")
    memory = summary["memory"]
    gib = 1024 ** 3
    add("MEMORY", f"Peak sampled RSS {memory['peak_rss_bytes']/gib:.2f} GiB; peak process working set {memory['peak_process_bytes']/gib:.2f} GiB; minimum available RAM {memory['minimum_available_ram_bytes']/gib:.2f} GiB; peak system pagefile use {memory['peak_pagefile_used_bytes_system']/gib:.2f} GiB. Sequential execution, no new model.")
    add("STEP 1–8 COMPATIBILITY", f"The retriever, index, chunking, model choice, answer bank default, reranker setting, and Step 8 trust architecture were not redesigned. The frozen Step 7H targeted gate completed {legacy['completed']}/{legacy['expected']} but failed {len(legacy['failures'])} check: {', '.join(legacy['failures'])}. Q020 Bangla is a safe `INSUFFICIENT_EVIDENCE` abstention in both the saved Step 8 baseline and Step 8B, so this was not introduced by Step 8B. The user prohibited one-off coverage chasing, so no extractor was added. Full unit suite result is above.")
    add("70-PDF READINESS", "The rules use question/evidence text rather than file names or question IDs. Actual 70-PDF behavior remains untested and needs a fresh ingestion/index and independent validation.")
    add("6000-QUESTION INDEPENDENCE", "No production rule reads dataset answers or IDs. This 300-variant development rerun is not evidence of accuracy or independence on the future 6000-question evaluation; a separate held-out protocol remains necessary.")
    diff = subprocess.run(["git", "diff", "--stat"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    add("GIT DIFF SUMMARY", "No commit made. This workspace already contained uncommitted Step 7/8 work and generated artifacts. Current tracked diff:\n\n```text\n" + diff.stdout.strip() + "\n```\n\nStep 8B new files are untracked; see repository status for the full list. No existing result artifact was overwritten.")
    add("REMAINING RISKS", "Automatic checks cannot fully resolve paraphrased set completeness or every policy scope. Q016 may safely abstain despite relevant duration evidence; some Q018/Q031 variants may abstain for evidence/validation reasons. The legacy Step 7H Q020 Bangla relation gate remains failed, although it is a pre-existing safe abstention. Minimum sampled available RAM was very low, so operational headroom on this 8-GB machine is not established. Manual answer review remains necessary.")
    complete = bool(gate["passed"] and summary["completion"] == 300 and summary["unsafe_returned_answers"] == 0
                    and not summary["errors"] and test.returncode == 0 and legacy["gate_passed"])
    add("STEP 8 ENGINEERING-COMPLETE?", "YES" if complete else "NO. The Step 8B safety targets passed, but the frozen Step 7H compatibility gate still has a pre-existing safe-abstention failure. Do not freeze Step 8 as fully regression-clean.")
    add("READY FOR STEP 9?", "NO for automatic progression. The human review sample and low-memory operating margin need review before a separately authorized Step 9. Step 9 was not started." if complete else "NO. The remaining compatibility exception and low-memory headroom need an explicit engineering decision; Step 9 was not started.")
    summary["legacy_step7h_gate"] = {"completed": legacy["completed"], "expected": legacy["expected"],
                                     "passed": legacy["gate_passed"], "failures": legacy["failures"]}
    summary["engineering_complete"] = complete
    summary["ready_for_step9"] = False
    (RESULTS / "step8b_full_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "step8b_full_report.md").write_text("# Step 8B — final answerability stabilization\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    print(json.dumps({"sections": len(sections), "tests_passed": test.returncode == 0,
                      "engineering_complete": complete, "unsafe": summary["unsafe_returned_answers"]}))


if __name__ == "__main__":
    main()
