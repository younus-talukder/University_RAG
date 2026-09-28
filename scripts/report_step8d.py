"""Compile Step 8D's evidence audit, status audit, test gate, and freeze report.

CSV matrices are handed to the artifact-tool builder as JSON; earlier Step 8
measurement files are only read and are never modified.
"""

from __future__ import annotations

import csv
import json
import pickle
import re
import subprocess
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_step8d import CHECKPOINT, LANGUAGES, TARGET_IDS, gate
from src.course_rows import course_field_value, requested_course_code
from src.evaluator import load_dataset
from src.evidence import detect_requested_field
from src.evidence import _fact_values

RESULTS = ROOT / "results"
HUMAN = ("correctness", "relevance", "groundedness", "completeness", "naturalness",
         "semantic_consistency", "overall_acceptability", "review_notes")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def targeted() -> list[dict]:
    rows = [json.loads(line) for line in CHECKPOINT.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 24 or gate(rows):
        raise RuntimeError(f"Targeted Step 8D gate not green: {gate(rows)}")
    return rows


def credit_exceptions() -> list[dict]:
    path = RESULTS / "step8d_credit_exception_checkpoint.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 6 or {row["answerability_status"] for row in rows} != {"CONFLICTING_EVIDENCE"}:
        raise RuntimeError("Six credit exception variants did not remain genuine conflicts")
    return rows


def corpus_scope_audit() -> dict:
    with (ROOT / "vector_db" / "metadata.pkl").open("rb") as handle:
        chunks = pickle.load(handle)
    output: dict = {"indexed_chunks": len(chunks), "fields": {}}
    for field in ("credits", "prerequisite"):
        questions = {requested_course_code(row["question"]): row["question"] for row in load_dataset()
                     if detect_requested_field(row["question"]) == field and requested_course_code(row["question"])}
        claims: dict[str, list[dict]] = {code: [] for code in questions}
        for item in chunks:
            code = re.sub(r"[^A-Za-z0-9]", "", str(item.get("entity_id") or "")).upper()
            if code not in claims:
                continue
            value = course_field_value(questions[code], field, [str(item.get("text") or "")])
            if value:
                claims[code].append({"value": value, "page": item.get("page"),
                                     "chunk_id": item.get("chunk_id"), "source": item.get("source")})
        divergent = {code: entries for code, entries in claims.items()
                     if len({str(Decimal(entry["value"]).normalize()) if field == "credits"
                             else entry["value"].casefold() for entry in entries}) > 1}
        output["fields"][field] = {"course_entities_checked": len(claims),
                                    "unparsed_entities": sum(not entries for entries in claims.values()),
                                    "divergent_entities": divergent}
    if len(output["fields"]["credits"]["divergent_entities"]) != 2 or output["fields"]["prerequisite"]["divergent_entities"]:
        raise RuntimeError("Corpus scope audit changed; reconsider full 300 rerun")
    return output


def q088_audit(rows: list[dict]) -> list[dict]:
    observed: dict[str, dict] = {}
    for row in rows:
        if row["question_id"] != "Q088":
            continue
        trace = row.get("trace") or {}
        candidates = {item.get("chunk_id"): item for item in trace.get("retrieval_candidates") or []}
        for evidence in trace.get("verified_evidence") or []:
            chunk_id = str(evidence.get("chunk_id") or "")
            item = candidates.get(chunk_id, {})
            if not chunk_id:
                continue
            current = observed.setdefault(chunk_id, {
                "document": evidence.get("source") or "",
                "page": evidence.get("page") or "",
                "chunk_id": chunk_id,
                "parent_block": str(item.get("parent_id") or re.sub(r"-c\d+$", "", chunk_id)),
                "entity": item.get("entity_id") or evidence.get("matched_entity") or "",
                "requested_field": "prerequisite",
                "requested_relation": "course_prerequisite",
                "raw_evidence": evidence.get("text") or "",
                "extracted_value": "",
                "normalized_value": "",
                "relation_scope": "course_prerequisite:MTH 101",
                "qualifiers": "",
                "curricular_context": item.get("heading") or "",
                "source_structure": "",
                "document_version": item.get("edition") or item.get("effective_date") or item.get("publication_year") or "",
                "languages_contributing": set(),
            })
            current["languages_contributing"].add(row["language"])
    for item in observed.values():
        source = " ".join(item["raw_evidence"].split())
        labeled = re.search(r"pre\s*[- ]?\s*requisites?\s*[:=-]\s*(Nil|None|N/A|No\s+prerequisite)\b", source, re.I)
        row_value = re.search(r"\b\d+(?:\.\d+)?\s+(Nil|None|N/A)\b", source, re.I)
        raw = (labeled or row_value).group(1) if (labeled or row_value) else ""
        normalized = _fact_values("prerequisite", source)
        item["extracted_value"] = raw
        item["normalized_value"] = next(iter(normalized)) if len(normalized) == 1 else ""
        item["source_structure"] = ("Explicit prerequisite label in MTH 101 course block" if labeled else
                                    "MTH 101 curriculum row; Pre-Requisite column verified on PDF page")
        item["qualifiers"] = "first_year_first_semester" if "First Year First Semester" in str(item["curricular_context"]) else ""
        item["languages_contributing"] = ", ".join(sorted(item["languages_contributing"]))
    audited = sorted(observed.values(), key=lambda item: int(item["page"]))
    if len(audited) != 2 or {item["normalized_value"] for item in audited} != {"NO_PREREQUISITE"}:
        raise RuntimeError("Q088 evidence did not resolve to the two verified prerequisite forms")
    if {item["extracted_value"].casefold() for item in audited} != {"nil", "n/a"}:
        raise RuntimeError("Q088 evidence forms changed; human adjudication required")
    return audited


def status_audit(rows: list[dict]) -> list[dict]:
    old = {(item["question_id"], item["language"]): item
           for item in read_csv(RESULTS / "step8b_full_results.csv")}
    audit: list[dict] = []
    for row in rows:
        qid, language = row["question_id"], row["language"]
        before = old[(qid, language)]["answerability_status"]
        after = row["answerability_status"]
        issue = "NONE"
        determination = ""
        if before == "CONFLICTING_EVIDENCE" and after != before:
            issue = "FALSE_CONFLICT_RESOLVED"
            determination = "Same-relation audit removed an unrelated or equivalent-value comparison."
        elif qid in {"Q034", "Q065"} and after == "CONFLICTING_EVIDENCE":
            issue = "GENUINE_CONFLICT_REVEALED"
            determination = "Same course and credit relation have two visibly verified PDF values; do not select one."
        elif after == "CONFLICTING_EVIDENCE":
            issue = "CONFLICT_REQUIRES_REVIEW"
            determination = "Inspect contributing same-relation claims."
        elif after == "AMBIGUOUS_QUERY" and qid in {"Q018", "Q031"}:
            issue = "FALSE_AMBIGUITY"
            determination = "Question states its policy or theoretical-course scope."
        elif qid in {"Q016", "Q031"} and after != "SUPPORTED":
            issue = "SAFE_COVERAGE_GAP"
            determination = "Documented answer exists; no incorrect answer was returned. Out of Step 8D scope."
        elif after == "INSUFFICIENT_EVIDENCE" and row.get("reference"):
            issue = "POSSIBLE_FALSE_INSUFFICIENT_EVIDENCE"
            determination = "Reference exists, but source support was not adjudicated in this targeted gate."
        audit.append({
            "question_id": qid, "language": language, "question": row["question"],
            "prior_status": before, "current_status": after,
            "issue": issue, "determination": determination,
            "requested_field": row.get("requested_field") or "",
            "requested_relation": row.get("requested_relation") or "",
            "source": row.get("source") or "", "page": row.get("page") or "",
            "supporting_excerpt": row.get("supporting_excerpt") or "",
        })
    return audit


def manual_review(rows: list[dict]) -> list[dict]:
    review: list[dict] = []
    for row in rows:
        review.append({
            "question_id": row["question_id"], "language": row["language"],
            "question": row["question"],
            "reference_answer": row.get("reference") or "",
            "answerability_status": row.get("answerability_status") or "",
            "answerability_reason": row.get("answerability_reason") or "",
            "answer_strategy": row.get("answer_strategy") or "",
            "final_answer": row.get("final_answer") or "",
            "source": row.get("source") or "", "page": row.get("page") or "",
            "supporting_excerpt": row.get("supporting_excerpt") or "",
            "conflict_sources": json.dumps(row.get("conflict_sources") or [], ensure_ascii=False),
            "grounding_status": str(row.get("grounding_validation_passed") or ""),
            "semantic_status": str(row.get("semantic_validation_passed") or ""),
            "language_status": str(row.get("language_validation_passed") or ""),
            "measurement_origin": "step8d_fresh",
            **{name: "" for name in HUMAN},
        })
    counts = Counter(row["language"] for row in review)
    if len(review) != 30 or counts != {language: 10 for language in LANGUAGES}:
        raise RuntimeError(f"Manual review sample count mismatch: {counts}")
    return review


def test_suite() -> dict:
    run = subprocess.run([sys.executable, "-X", "utf8", "-m", "unittest", "discover",
                          "-s", "tests", "-p", "test*.py"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = run.stdout + run.stderr
    (RESULTS / "step8d_test_suite.txt").write_text(output, encoding="utf-8")
    match = re.search(r"Ran\s+(\d+)\s+tests?", output)
    skip = re.search(r"skipped=(\d+)", output)
    result = {"previous": 232, "new": int(match.group(1)) - 232 if match else None,
              "total": int(match.group(1)) if match else None,
              "skipped": int(skip.group(1)) if skip else 0,
              "failures": 0 if run.returncode == 0 else 1,
              "exit_code": run.returncode}
    if run.returncode or not match:
        raise RuntimeError("Full unit suite failed; inspect results/step8d_test_suite.txt")
    return result


def report(summary: dict, rows: list[dict], q088: list[dict], audit: list[dict]) -> str:
    by_key = {(row["question_id"], row["language"]): row for row in rows}
    before = {(row["question_id"], row["language"]): row
              for row in read_csv(RESULTS / "step8c_targeted_results.csv")}
    sections = ["# Step 8D — conflict scope and equivalence stabilization", ""]
    def section(number: int, title: str, body: str) -> None:
        sections.extend([f"## {number}. {title}", "", body, ""])
    section(1, "STEP 8D GOAL", "Stop false document-conflict claims while preserving genuine same-relation contradictions. Step 9 was not started.")
    section(2, "FILES CHANGED", "`src/evidence.py`, `tests/test_step8d_conflict_scope.py`, `scripts/evaluate_step8d.py`, `scripts/audit_step8d_credit_exceptions.py`, `scripts/report_step8d.py`, and the new Step 8D output builder. Prior Step-8 artifacts were preserved; no commit was made.")
    trace_lines = []
    for language in ("english", "bangla", "banglish"):
        item = by_key[("Q018", language)]
        trace = item.get("trace") or {}
        candidates = ", ".join(f"p{c.get('page')} {c.get('chunk_id')}" for c in trace.get("retrieval_candidates") or [])
        verified = ", ".join(f"p{e.get('page')} {e.get('chunk_id')}" for e in trace.get("verified_evidence") or [])
        trace_lines.append(f"- {language}: field `{item['requested_field']}`; public relation label `{item.get('requested_relation') or 'none'}`; conflict identity `theoretical_course_credit_rule`; retrieved {candidates}; verified {verified}; normalized conflict values {trace.get('normalized_conflict_values') or []}.")
    section(3, "Q018 ROOT CAUSE", "The former field-only comparator treated course/lab credit values as possible contradictions of the theoretical-course credit-assignment rule. Frozen Bangla conflict candidates: page 17 theory rule (no numeric field value), page 81 lab value 1.50, page 63 course value 3.00. Frozen Banglish candidates: page 17 theory, page 63 lab/course value 1.5, page 81 course value 1.50. The numeric values belong to different relations. Current complete top-three retrieval and verified-evidence trace:\n\n" + "\n".join(trace_lines) + "\n\nThe full passage text and metadata are retained in `results/step8d_targeted_checkpoint_v3.jsonl`.")
    section(4, "CONFLICT-SCOPE MODEL", "A reusable conflict identity carries entity, field, relation, scope, material qualifiers, and document context. Values are compared only under a matching semantic key. Document context is retained for review, not used for automatic recency arbitration.")
    section(5, "RELATION / QUALIFIER COMPATIBILITY", "Theoretical credit assignment, lab assignment, course credit value, and semester total are distinct. Attendance, seat reservation, marks, and admission percentages are distinct domains. Explicit category, program level, every-course, and per-semester qualifiers remain in the comparison identity.")
    q018_lines = []
    for language in ("english", "bangla", "banglish"):
        old = before[("Q018", language)]["answerability_status"]
        new = by_key[("Q018", language)]
        pages = [str(item.get("page")) for item in (new.get("trace") or {}).get("verified_evidence") or []]
        q018_lines.append(f"- {language}: {old} → {new['answerability_status']}; verified page(s) {', '.join(pages) or 'none'}; answer: {new['final_answer']}")
    section(6, "Q018 BEFORE / AFTER", "\n".join(q018_lines))
    evidence_lines = []
    for index, item in enumerate(q088, 1):
        excerpt = " ".join(item["raw_evidence"].split())[:280]
        evidence_lines.append(f"- Evidence {chr(64+index)}: `{item['document']}`, PDF page {item['page']}, `{item['chunk_id']}`; `{item['extracted_value']}` → `{item['normalized_value']}`; {excerpt}…")
    section(7, "Q088 COMPLETE EVIDENCE AUDIT", "\n".join(evidence_lines) + "\n\nBoth contributing passages are in `results/step8d_q088_evidence_audit.csv` with complete raw text, parent block, source structure, scope, and context. A retrieved MTH 201 passage was excluded because MTH 101 is mentioned only as its prerequisite, not as the subject course.")
    section(8, "Q088 ADJUDICATION", "Equivalent: YES, for the verified MTH 101 prerequisite relation in the same BSc curriculum PDF. Genuine conflict: NO. Ambiguous: NO. Page 51 has `Nil` in the MTH 101 Pre-Requisite table column; page 63 explicitly has `Prerequisite: N/A` under MTH 101. No distinct edition or supersession metadata was present in these chunks.")
    section(9, "NO-PREREQUISITE NORMALIZATION", "`Nil` and `None` retain existing equivalence. `N/A` maps to `NO_PREREQUISITE` only after it is extracted from a verified prerequisite label/row; an unrelated N/A is not globally rewritten.")
    section(10, "TRUE CONFLICT REGRESSION", "Same-course credit 3.00 versus 4.00: CONFLICTING. Same-policy/same-category attendance requirement 70% versus 75%: CONFLICTING. Different document editions are retained for review; newer is not assumed to win.")
    section(11, "FALSE-CONFLICT REGRESSION", "Theory versus lab rule: no conflict. Course credit versus semester total: no conflict. Nil/None/N/A in a verified prerequisite field: equivalent. Explicitly different policy categories: no conflict.")
    targeted_lines = []
    for qid in ("Q014", "Q016", "Q018", "Q020", "Q031", "Q081", "Q088"):
        states = ", ".join(f"{language}={by_key[(qid, language)]['answerability_status']}" for language in ("english", "bangla", "banglish"))
        targeted_lines.append(f"- {qid}: {states}")
    section(12, "TARGETED CASE RESULTS", "\n".join(targeted_lines) + "\n- Q095 collateral equivalence: all three languages supported.\n- Q034 HSS 101: all three safely report a genuine page-51 3.00 versus page-61 1.50 credit conflict.\n- Q065 CSE 304: all three safely report a genuine page-53 0.75 versus page-78 3.00 credit conflict. Neither source value is silently selected.")
    issue_counts = Counter(row["issue"] for row in audit)
    section(13, "MISLEADING-STATUS AUDIT", f"Targeted false conflict: {summary['misleading_status']['current_false_conflict']}. Targeted false ambiguity: {summary['misleading_status']['current_false_ambiguity']}. Other: {issue_counts.get('GENUINE_CONFLICT_REVEALED', 0)} genuine newly surfaced conflict variants (Q034/Q065), {issue_counts.get('SAFE_COVERAGE_GAP', 0)} safe coverage-gap variants (Q016/Q031), and {issue_counts.get('POSSIBLE_FALSE_INSUFFICIENT_EVIDENCE', 0)} unadjudicated possible insufficient-evidence variants. The CSV retains prior/current status per case.")
    section(14, "UNSAFE RETURNED ANSWERS", f"Targeted count: {summary['unsafe_returned_answers']}. Target: 0. This is not a new full-300 audit.")
    tests = summary["tests"]
    section(15, "TEST RESULTS", f"Previous: {tests['previous']}; new: {tests['new']}; total: {tests['total']}; failures: {tests['failures']}; skipped: {tests['skipped']}.")
    scope = summary["corpus_scope_audit"]
    section(16, "FULL 300 RERUN", f"Performed: NO. The prior full run had eight conflict-status rows, all in Q018, Q088, or Q095; all were targeted. A read-only scan of all {scope['indexed_chunks']} indexed chunks covered {scope['credit_course_entities_checked']} credit-question course entities and {scope['prerequisite_course_entities_checked']} prerequisite-question entities. It found exactly two further contradictory credit entities (HSS 101, CSE 304) and no divergent prerequisites; all six affected language variants were then run live. This bounds the detected status changes without a memory-heavy 300-answer run. The prior 258/300 safe-answer coverage is historical, not a current Step-8D score; no new coverage or accuracy score is claimed.")
    memory = summary["memory"]
    section(17, "MEMORY", f"Sequential target peak RSS {memory['peak_rss_gib']:.2f} GiB, minimum sampled available RAM {memory['min_available_mib']:.0f} MiB, peak sampled system pagefile use {memory['peak_pagefile_gib']:.2f} GiB. No generator settings or parallel workers changed.")
    section(18, "FINAL MANUAL REVIEW FILE", "`results/step8d_final_manual_review.csv`: 30 fresh current-pipeline rows, 10 per language, including Q018, Q088, Q081, and both newly surfaced credit conflicts. Human review fields are blank.")
    section(19, "STEP 1–8C COMPATIBILITY", "Q014 emails, Q016 safe duration handling, Q020 three-language distribution, Q031 non-ambiguity, Q081 None/Nil, Q018 non-conflict, Q088/Q095 contextual equivalence, synthetic true conflicts, and the complete unit suite passed. Frozen Step-7H was not rerun in this narrow phase.")
    section(20, "70-PDF READINESS", "Architectural only. Cross-document identity and version metadata are designed conservatively; 70-PDF ingestion and memory/quality at that scale were not measured.")
    section(21, "6000-QUESTION INDEPENDENCE", "Production rules use question and evidence text, not dataset IDs or reference answers. This targeted development gate is not an independent 6000-question accuracy evaluation.")
    section(22, "GIT DIFF SUMMARY", "No commit. Existing dirty Step-7/8 work was preserved. This phase changes only conflict handling, synthetic tests, and new Step-8D evaluation/report outputs.")
    section(23, "REMAINING RISKS", "Q016 and Q031 remain safe answer-coverage gaps; Q018 Bangla also safely rejects generation after the false conflict is removed. HSS 101 and CSE 304 have genuine conflicting values inside the source PDF; the system now abstains instead of choosing one. Some Bangla/Banglish wording remains awkward. The 8-GB machine has little RAM headroom. Source edition metadata is incomplete; genuine cross-version discrepancies require explicit review, never automatic newer-wins arbitration.")
    section(24, "STEP 8 ENGINEERING-COMPLETE?", "YES for the specified development trust and compatibility gates, subject to human review of the new 30-row sample. This is not a thesis accuracy or scale claim.")
    section(25, "READY FOR STEP 9?", "YES for a separately authorized phase. Step 9 has not begun.")
    section(26, "HUMAN INPUT NEEDED?", "No Q088 adjudication is needed: both complete source contexts establish the same MTH 101 prerequisite field. HSS 101 and CSE 304 source values genuinely disagree; an authoritative correction would require a document owner, but the system safely surfaces the conflicts without that decision. Human review of the new 30-row sample remains a separate quality step.")
    section(27, "NEXT STEP", "Review the new sample and freeze Step 8 only after the reviewer accepts the conflict/status changes. Plan Step 9 separately with explicit authorization and memory constraints.")
    return "\n".join(sections)


def run() -> None:
    primary = targeted()
    extras = credit_exceptions()
    if {row["run_fingerprint"] for row in primary + extras} != {primary[0]["run_fingerprint"]}:
        raise RuntimeError("Step 8D target and credit-exception fingerprints differ")
    rows = primary + extras
    scope = corpus_scope_audit()
    q088 = q088_audit(primary)
    audit = status_audit(rows)
    review = manual_review(rows)
    tests = test_suite()
    unsafe = [row for row in rows if row["answerability_status"] == "SUPPORTED" and (
        row.get("answer_safety_failures") or row.get("language_validation_passed") is False or
        row.get("grounding_validation_passed") is False or row.get("semantic_validation_passed") is False)]
    current_false_conflict = sum(1 for row in audit if row["issue"] == "CONFLICT_REQUIRES_REVIEW")
    current_false_ambiguity = sum(1 for row in audit if row["issue"] == "FALSE_AMBIGUITY")
    summary = {
        "targeted_completion": len(rows), "primary_targeted_completion": len(primary),
        "credit_exception_completion": len(extras), "targeted_gate_passed": not gate(primary),
        "q018": {language: next(row["answerability_status"] for row in rows if row["question_id"] == "Q018" and row["language"] == language) for language in LANGUAGES},
        "q088": {language: next(row["answerability_status"] for row in rows if row["question_id"] == "Q088" and row["language"] == language) for language in LANGUAGES},
        "q088_contributing_passages": len(q088),
        "q088_adjudication": "EQUIVALENT_NO_PREREQUISITE",
        "misleading_status": {"current_false_conflict": current_false_conflict,
                              "current_false_ambiguity": current_false_ambiguity,
                              "prior_false_conflicts_resolved": sum(row["issue"] == "FALSE_CONFLICT_RESOLVED" for row in audit)},
        "unsafe_returned_answers": len(unsafe), "unsafe_keys": [(row["question_id"], row["language"]) for row in unsafe],
        "tests": tests,
        "full_300_rerun": {"performed": False, "prior_step8b_safe_answer_coverage": "258/300 (86.0%), not remeasured"},
        "corpus_scope_audit": {"indexed_chunks": scope["indexed_chunks"],
                               "credit_course_entities_checked": scope["fields"]["credits"]["course_entities_checked"],
                               "prerequisite_course_entities_checked": scope["fields"]["prerequisite"]["course_entities_checked"],
                               "divergent_credit_entities": sorted(scope["fields"]["credits"]["divergent_entities"]),
                               "divergent_prerequisite_entities": sorted(scope["fields"]["prerequisite"]["divergent_entities"])},
        "manual_review": {"rows": len(review), "per_language": dict(Counter(row["language"] for row in review)),
                          "human_fields_blank": all(not row[name] for row in review for name in HUMAN)},
        "memory": {"peak_rss_gib": max(row["rss_bytes"] for row in rows) / 2**30,
                   "min_available_mib": min(row["available_ram_bytes"] for row in rows) / 2**20,
                   "peak_pagefile_gib": max(row["pagefile_used_bytes_system"] for row in rows) / 2**30},
        "step8_engineering_complete": not unsafe and not current_false_conflict and not current_false_ambiguity and tests["failures"] == 0,
        "step9_started": False,
    }
    if not summary["step8_engineering_complete"]:
        raise RuntimeError("Step 8D freeze criteria not met; inspect summary inputs")
    (RESULTS / "step8d_corpus_scope_audit.json").write_text(
        json.dumps(scope, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {"targeted_results": rows, "q088_evidence_audit": q088,
               "status_audit": audit, "final_manual_review": review}
    (RESULTS / "step8d_artifact_payload.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "step8d_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "step8d_final_report.md").write_text(
        report(summary, rows, q088, audit), encoding="utf-8")
    print(json.dumps({"targeted": len(rows), "q088_passages": len(q088),
                      "unsafe": len(unsafe), "tests": tests["total"],
                      "review": len(review)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
