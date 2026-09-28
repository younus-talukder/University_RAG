# Step 8 answerability and trustworthy abstention

## 1. STEP 8 GOAL

Add an observable, rule-based answerability decision before exposing a factual answer; do not equate retrieval scores with probabilities.

## 2. FILES CHANGED

Step 8: `src/answerability.py`, `src/evidence.py`, `src/pipeline.py`, `app.py`, `tests/test_step8_answerability.py`, `scripts/evaluate_step8.py`, `scripts/report_step8.py`. Earlier uncommitted Step 7 work and index artifacts were preserved.

## 3. ANSWERABILITY ARCHITECTURE

The existing evidence assessment remains the evidence-stage gate. `AnswerabilityDecision` maps that gate and retrieval provenance to a final state. Output acceptance is a separate validation stage; it can turn supported evidence into `GENERATION_REJECTED`. No evaluation field reaches production answering.

## 4. ANSWERABILITY STATES

SUPPORTED, INSUFFICIENT_EVIDENCE, RETRIEVAL_UNCERTAIN, AMBIGUOUS_QUERY, CONFLICTING_EVIDENCE, GENERATION_REJECTED, OUT_OF_DOMAIN, SYSTEM_ERROR. `support_status` remains the upstream evidence assessment; `answerability_status` is the final user-answer gate, not a competing synonym.

## 5. ANSWERABILITY REASON CODES

GENERATION_GROUNDING_FAILURE: 18, FIELD_NOT_SUPPORTED: 8, AMBIGUOUS_ENTITY: 6, GENERATION_SEMANTIC_FAILURE: 5, CONFLICTING_VALUES: 3, ENTITY_OWNERSHIP_UNCERTAIN: 1. Reasons are categorical, not calibrated likelihoods.

## 6. EVIDENCE-STRENGTH MODEL

STRONG is explicit labeled/structured or relation evidence, or multiple independent agreeing passages. ADEQUATE is one verified passage. WEAK is related but unverified/contested evidence. NONE has no usable support. No percentage or retrieval-score threshold is used.

## 7. INDEPENDENT SUPPORT LOGIC

Distinct document + physical page + parent block units are counted once; sibling child chunks sharing a parent do not multiply support. Synthetic agreement/conflict across PDFs is tested.

## 8. RETRIEVAL-CHANNEL SIGNALS

Dense, BM25, metadata and cross-lingual provenance are debug metadata. Ranks and scores remain diagnostics only, never confidence probabilities.

## 9. STRUCTURED ANSWER TRUST GATE

A structured answer still needs matching entity, field, extracted value, language and grounding checks. It is not sent through Qwen merely for a trust label.

## 10. GENERATED ANSWER TRUST GATE

Generation proceeds only with verified evidence. Canonical output and target-language realization must pass language, semantic and grounding checks; otherwise the final state is `GENERATION_REJECTED`. This run used generation for 18 cases: 10 accepted, 8 rejected, 25 attempts, and 4 cases with retries.

## 11. INSUFFICIENT-EVIDENCE HANDLING

No verified entity/field support yields a safe multilingual abstention. The generator is not asked to fill missing facts.

## 12. RETRIEVAL-UNCERTAIN HANDLING

A related same-entity/field hit rejected by ownership or relation validation is kept as WEAK and abstains. No single numerical cutoff is used.

## 13. AMBIGUITY HANDLING

Contextless entity-dependent questions ask which course, document, program or policy is intended. They do not infer an entity from retrieval rank.

## 14. CONFLICT HANDLING

Incompatible verified values block answering, regardless of document recency. Debug metadata retains source, page and excerpt for each conflicting passage.

## 15. GENERATION-REJECTED HANDLING

Verified evidence is retained in metadata, but an unacceptable realization is not displayed as a factual answer. This is distinguished from absent evidence.

## 16. OUT-OF-DOMAIN HANDLING

A conservative overt-intent preflight catches unrelated weather, sport, medical, coding, news/trivia and advice requests while exempting explicit university/course context.

## 17. PROMPT-INJECTION RESULT

Overt instructions to ignore documents or answer from outside knowledge are removed before retrieval; an instruction-only query is refused. The generator's existing evidence-only system prompt and final validators remain mandatory.

## 18. MULTILINGUAL SAFE RESPONSES

Each non-supported state has English, Bangla and Banglish wording. Synthetic tests verify language checks for all three.

## 19. UI / DEBUG BEHAVIOR

Normal UI shows the answer and, only for supported answers, source/page/excerpt without raw scores. Optional debug mode exposes reasons, evidence level, counts, channels, rank, conflict passages, generation state and fallback use.

## 20. TEST RESULTS

Previous baseline: 206 tests (205 pass, 1 skip). Step 8 adds focused synthetic and Q007 annotation tests. Test process exit: 0.

```text
............................................................s..................................................................................................................................................................
----------------------------------------------------------------------
Ran 223 tests in 0.736s

OK (skipped=1)
```

## 21. 300-QUERY ANSWERABILITY DISTRIBUTION

300/300 development variants completed.

| Language | SUPPORTED | INSUFFICIENT | RETRIEVAL UNCERTAIN | AMBIGUOUS | CONFLICTING | GENERATION REJECTED | OOD | SYSTEM ERROR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| english | 87 | 2 | 1 | 2 | 1 | 7 | 0 | 0 |
| bangla | 85 | 6 | 0 | 2 | 1 | 6 | 0 | 0 |
| banglish | 87 | 0 | 0 | 2 | 1 | 10 | 0 | 0 |
| overall | 259 | 8 | 1 | 6 | 3 | 23 | 0 | 0 |

This is a development measurement, not final thesis accuracy.

## 22. SAFE ANSWER COVERAGE

| Language | Returned / total | Safe answer coverage | Abstention rate | Step 7H returned |
| --- | --- | --- | --- | --- |
| english | 87/100 | 87.0% | 13.0% | 87 |
| bangla | 85/100 | 85.0% | 15.0% | 86 |
| banglish | 87/100 | 87.0% | 13.0% | 87 |
| overall | 259/300 | 86.3% | 13.7% | 260 |

Coverage is returned answers passing automated trust checks divided by all questions; it is not correctness.

## 23. ABSTENTION RATE

| Language | Returned / total | Safe answer coverage | Abstention rate | Step 7H returned |
| --- | --- | --- | --- | --- |
| english | 87/100 | 87.0% | 13.0% | 87 |
| bangla | 85/100 | 85.0% | 15.0% | 86 |
| banglish | 87/100 | 87.0% | 13.0% | 87 |
| overall | 259/300 | 86.3% | 13.7% | 260 |

By reason: GENERATION_GROUNDING_FAILURE=18, FIELD_NOT_SUPPORTED=8, AMBIGUOUS_ENTITY=6, GENERATION_SEMANTIC_FAILURE=5, CONFLICTING_VALUES=3, ENTITY_OWNERSHIP_UNCERTAIN=1. Abstention is not automatically an error.

## 24. FALSE ABSTENTIONS

41 review candidates with a reference and a non-supported status. These are *not* all proven false abstentions. Likely causes: GROUNDING_VALIDATION=18, FIELD_VALIDATION=8, UNKNOWN=6, SEMANTIC_VALIDATION=5, CONFLICT=3, ENTITY_VALIDATION=1. Human review may identify ambiguous or flawed references.

## 25. UNSAFE ANSWERS

Automated flagged returned answers: 0 (target 0). The audit flags WEAK evidence or failed mandatory validators; zero flags would not establish human correctness.

## 26. EVIDENCE LEVEL DISTRIBUTION

STRONG=280, ADEQUATE=2, WEAK=4, NONE=14.

## 27. OUT-OF-DOMAIN TEST

20/20 developer-only questions safely refused without a fabricated university answer. The set is separate from the 300 thesis-development variants.

## 28. AMBIGUITY TEST

8/8 contextless developer cases returned `AMBIGUOUS_QUERY`; failures, if any, remain visible in the CSV.

## 29. CONFLICT TEST

2/2 synthetic conflicting fixtures returned `CONFLICTING_EVIDENCE`; the agreeing cross-document fixture remained answerable. No real PDF was modified.

## 30. ANSWERABILITY OVERHEAD

Excluding generation-used rows: median 0.27 ms; P95 0.58 ms. Wall-clock measurement includes trust feature collection, not Qwen generation.

## 31. MEMORY IMPACT

Observed peak process working set 5.37 GiB vs Step 7H 5.35 GiB; max system pagefile used 2.73 GiB vs 2.97 GiB. These are run-level observations affected by system load, not isolated causal memory deltas. No new model was loaded.

## 32. HUMAN REVIEW FILE

`results/step8_manual_review_sample.csv`: 30 rows, 10 per language, with eight human scoring columns blank. The selection prioritizes trust states, relation and GGUF strategies, and cross-lingual recovery when present.

## 33. STEP 1–7 COMPATIBILITY

The previous retrieval, cross-lingual fallback, relation extraction, generator, and validators remain in place. The full regression suite was rerun; Step 7H artifacts were not overwritten.

## 34. 70-PDF READINESS

Evidence identity uses document ID/path, page and parent block; agreement and conflict tests span independent documents. No current filename, course list, or page number is embedded in production trust rules. Actual 70-PDF behavior remains untested.

## 35. 6000-QUESTION INDEPENDENCE

Production answerability receives only the current question, retrieved candidates, and verified evidence. IDs, references, page labels and paired translations are evaluation-script inputs only. A future 6000-question run is not claimed validated.

## 36. GIT DIFF SUMMARY

No commit was made. Current working tree also contains pre-existing Step 7 changes and index files, so the status below is not solely Step 8:

```text
M app.py
 M requirements.txt
 M results/index_build_report.json
 M results/ingestion_report.json
 M src/answer_policy.py
 M src/chunker.py
 M src/config.py
 M src/evidence.py
 M src/generator.py
 M src/grounding_validator.py
 M src/pipeline.py
 M src/query_normalization.py
 M src/semantic_contract.py
 M tests/test_step7_answering.py
 M vector_db/index.faiss
 M vector_db/index_manifest.json
 M vector_db/metadata.pkl
 M vector_db/sparse_index.pkl
?? results/step7f_failure_funnel.csv
?? results/step7f_failure_recovery.csv
?? results/step7f_final_report.md
?? results/step7f_language_parity_after.csv
?? results/step7f_language_parity_before.csv
?? results/step7f_multilingual_review.csv
?? results/step7f_retrieval_metrics.csv
?? results/step7g_banglish_audit.csv
?? results/step7g_english_regression.csv
?? results/step7g_final_report.md
?? results/step7g_full_300_report.md
?? results/step7g_full_300_results.csv
?? results/step7g_full_300_summary.json
?? results/step7g_full_manual_review.csv
?? results/step7g_targeted_results.csv
?? results/step7g_targeted_review.csv
?? results/step7h_full_300_aborted_numeric_punctuation.csv
?? results/step7h_full_300_report.md
?? results/step7h_full_300_report_pre_course_equivalence.md
?? results/step7h_full_300_results.csv
?? results/step7h_full_300_results_pre_bound_parity.csv
?? results/step7h_full_300_results_pre_course_equivalence.csv
?? results/step7h_full_300_summary.json
?? results/step7h_full_300_summary_pre_course_equivalence.json
?? results/step7h_full_manual_review.csv
?? results/step7h_full_manual_review_pre_course_equivalence.csv
?? results/step7h_targeted_pre_numeric_punctuation.csv
?? results/step7h_targeted_results.csv
?? results/step7h_targeted_results_pre_bound_parity.csv
?? results/step7h_targeted_results_pre_course_equivalence.csv
?? results/step7h_targeted_summary.json
?? results/step7h_targeted_summary_pre_bound_parity.json
?? results/step7h_targeted_summary_pre_course_equivalence.json
?? results/step8_ambiguity_results.csv
?? results/step8_answerability_checkpoint.jsonl
?? results/step8_answerability_checkpoint_pre_ood_fix.jsonl
?? results/step8_answerability_report.md
?? results/step8_answerability_results.csv
?? results/step8_answerability_summary.json
?? results/step8_conflict_results.csv
?? results/step8_false_abstention_review.csv
?? results/step8_manual_review_sample.csv
?? results/step8_out_of_domain_results.csv
?? results/step8_unsafe_answer_review.csv
?? scripts/audit_step7g.py
?? scripts/evaluate_step7f.py
?? scripts/evaluate_step7g.py
?? scripts/evaluate_step7g_full.py
?? scripts/evaluate_step7h_full.py
?? scripts/evaluate_step7h_targeted.py
?? scripts/evaluate_step8.py
?? scripts/report_step7g_full.py
?? scripts/report_step7h.py
?? scripts/report_step8.py
?? src/answerability.py
?? src/course_rows.py
?? src/crosslingual.py
?? src/document_metadata.py
?? src/evidence_spans.py
?? src/relations.py
?? tests/test_step7f_crosslingual.py
?? tests/test_step7g_relations.py
?? tests/test_step7h_stabilization.py
?? tests/test_step8_answerability.py
```

## 37. REMAINING RISKS

The one-PDF development set does not establish accuracy, calibration, or multi-PDF version arbitration. Ordinal evidence strength relies on the existing evidence validator; related passages may still be false positives and need human review. OOD coverage is conservative and incomplete. System memory is close to capacity during model use.

## 38. STEP 8 ENGINEERING-COMPLETE?

YES. This requires 300 rows, green synthetic checks and regression suite, zero runtime errors, and zero automatically flagged unsafe returns. It is engineering completion for the tested development scope, not a thesis accuracy claim.

## 39. READY FOR STEP 9?

NO automatic transition. Review the 30-row human sample, especially any unsafe flags, false-abstention candidates and Q007, before deciding whether to start Step 9.

## 40. NEXT STEP

Perform the requested human review of Step 8 results and explicitly authorize any later Step 9 work. Step 9 was not implemented.

## Q007 evaluation-only adjudication

The approved source is page 8, **Computer Science & Technology**. The original dataset remains unchanged and the production pipeline has no Q007 exception.

| Language | Page | Status | Technology in answer |
| --- | --- | --- | --- |
| english | 8 | SUPPORTED | True |
| bangla | 8 | SUPPORTED | True |
| banglish | 8 | SUPPORTED | True |

## Output files

- `results/step8_answerability_results.csv`
- `results/step8_answerability_summary.json`
- `results/step8_answerability_report.md`
- `results/step8_false_abstention_review.csv`
- `results/step8_unsafe_answer_review.csv`
- `results/step8_manual_review_sample.csv`
- `results/step8_out_of_domain_results.csv`
- `results/step8_ambiguity_results.csv`
- `results/step8_conflict_results.csv`
