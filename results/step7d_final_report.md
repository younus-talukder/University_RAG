# Step 7D Multilingual Semantic Quality Report

**DEVELOPMENT / REGRESSION EVALUATION — NOT FINAL THESIS ACCURACY**

## 1. STEP 7C REMAINING PROBLEM

Step 7C proved that the local Qwen2.5-7B-Instruct Q4_K_M runtime could complete the workload, but script and token checks still accepted multilingual answers that reversed polarity, changed a relation, omitted a required condition, used malformed Bangla/Banglish, or leaked pure English into Banglish. Step 7D therefore prioritizes correct meaning over fluency and coverage.

## 2. FILES CHANGED

Core Step 7D logic:

- `src/semantic_contract.py`: evidence-derived fact plans, polarity, relation/value, exact-count, and multi-condition validation.
- `src/semantic_errors.py`: generic multilingual error taxonomy.
- `src/grounding_validator.py`: semantic-contract validation integrated with factual-token grounding.
- `src/generator.py`: fact plan and semantic contract added to the bounded prompt.
- `src/answer_policy.py`: concise Bangla/Banglish evidence-expression instructions.
- `src/language_detector.py`: additional generic Banglish grammar/count markers.
- `src/language_validator.py`: stronger pure-English, mixed-script, and foreign-script checks.
- `src/pipeline.py`: evidence-only contracts, conservative failure routing, targeted one-retry policy, and rejection metadata.

Evaluation and tests:

- `scripts/evaluate_step7c.py`, `scripts/evaluate_step7d.py`
- `tests/test_step7_answering.py`, `tests/test_step7d_semantic_quality.py`
- `results/step7d_error_taxonomy.md`
- `results/step7d_smoke_results.csv`, `results/step7d_smoke_comparison.json`
- `results/step7d_review_run.csv`, `results/step7d_manual_review_sample.csv`
- `results/step7d_300_results.csv`, `results/step7d_summary.json`
- `README.md`

## 3. SEMANTIC CONTRACT

For every generation-required answer, the pipeline derives a contract from the question, requested entity/field, and verified evidence only. It records required entities, relations, values, polarity, evidence-supported anchors, forbidden incidental values for strict relations, and exact cardinalities when the question requests a count or conditions. Evaluation references are never passed to this logic.

## 4. POLARITY / NEGATION VALIDATION

English, Bangla, and Banglish negation markers are detected and compared with evidence polarity. An affirmative fact cannot become a denial, and negative evidence such as “has no prerequisite” must remain negative. `WRONG_POLARITY` is repairable once with a targeted instruction; a second failure abstains.

## 5. RELATION VALIDATION

Reusable subject/relation/value patterns cover relations such as publication, accreditation, operation under an act, location, establishment, offerings, eligibility, and prerequisites. High-value anchors are checked against the parsed relation rather than merely against the whole passage. This rejects wrong organizations, supported-but-misattached dates, omitted exact counts, and missing numeric conditions.

## 6. BANGLA IMPROVEMENTS

The prompt now asks for one or two short natural Bengali sentences that only express verified facts. It explicitly preserves polarity, names, technical identifiers, and numbers. Mixed-script corruption, unexpected foreign scripts, excessive English, and semantic uncertainty lead to rejection rather than a fluent guess.

## 7. BANGLISH IMPROVEMENTS

Banglish uses Latin-script Bangla grammar with technical English where appropriate. Pure-English prose and Bengali-script leakage are rejected. The detector recognizes generic function/inflection patterns including suffix attachments and common forms such as `onujayi`, `kotojon`, and `peyechilo`. A missed Banglish count question found during full-result inspection was corrected and regression-tested.

## 8. FACT-PLAN DESIGN

The fact plan contains compact evidence-derived `subject`, `relation`, `value`, and `polarity` items. Parsed subject/value anchors exclude incidental facts elsewhere in the same sentence. If extraction is limited for a high-value multilingual relation, the answer abstains instead of treating uncertainty as correctness.

## 9. GENERIC SEMI-STRUCTURED EXPANSION

Before: 31 generation-required cases.

After: 31 generation-required cases.

No relation category was moved into deterministic answering. Inspection found that the current retrieved prose often combines implicit relations and incidental facts; converting it would not yet be reliably generic. Step 7D instead added lightweight semantic contracts around the existing generation boundary. The existing 205 structured-exact and 2 structured-list routes are unchanged.

## 10. RETRY CHANGES

Maximum retry count remains one. Retries are allowed only for repairable language/style, polarity, missing required entity/value/count, or formatting failures. Wrong relation values, unsupported facts, semantic uncertainty, and semantic nonsense abstain without an open-ended retry.

## 11. TEST RESULTS

- Previous baseline: 139 tests, 0 failures.
- New Step 7D tests: 20.
- Total: 159 tests.
- Failures: 0.
- Skipped: 1 optional real reranker smoke test requiring the separately enabled 2+ GB model.

The exact command was `python -m unittest discover -s tests -p "test*.py"` using the project virtual environment.

## 12. 15-QUESTION BEFORE VS AFTER

All 15 cases attempted generation in both runs.

- English: before 5 accepted / 0 rejected; after 4 accepted / 1 rejected.
- Bangla: before 3 accepted / 2 rejected; after 0 accepted / 5 rejected.
- Banglish: before 0 accepted / 5 rejected; after 0 accepted / 5 rejected.

The four final accepted smoke answers were manually inspected and were evidence-bound. Previously accepted wrong Bangla negation/malformed meaning is now rejected. Pure-English Banglish, Bengali-script/mixed-script leakage, missing exact values, and uncertain relations abstain. This is a safety/precision improvement, not a coverage improvement.

## 13. 300-QUERY DEVELOPMENT RESULT

The smoke gate passed, so the full development set ran with answer bank OFF and reranker OFF. The first process reached 245/300 before a transient Windows atomic-file replacement lock; the durable 240-row checkpoint was verified and resume completed the remaining rows without duplicates.

- Rows: 300/300 unique (`100 English + 100 Bangla + 100 Banglish`).
- Runtime errors: 0.
- Strategies: structured exact 205; structured list 2; GGUF generation 7; unsupported 78; ambiguous 5; conflicting 3.
- Final output language consistency: 300/300.

Two unsafe accepted outputs exposed during final inspection—an incomplete Banglish repeat-examination condition and a Banglish count answer with no count—were converted into generic regression rules, re-evaluated individually, and merged through checkpoint resume.

## 14. GENERATION ACCEPTANCE

Acceptance is a routing result, not proof of correctness.

- Before Step 7D: attempted 31; accepted 17; rejected 14.
- After Step 7D: attempted 31; accepted 7; rejected 24.
- Retries: 14; successful retries: 0; failed retries: 14.

Final semantic rejections include uncertainty, unsupported factual tokens, wrong relation values, wrong polarity, missing required values/counts, pure-English Banglish, excessive-English Bangla, and mixed-script corruption. The lower acceptance rate is intentional under the stated “meaning > fluency > coverage” priority.

## 15. LANGUAGE QUALITY

- English: 100/100 passed automatic language validation.
- Bangla: 100/100 passed.
- Banglish: 100/100 passed.
- Overall: 300/300 passed.

These are language-consistency checks, not human naturalness or correctness scores.

## 16. SEMANTIC FAILURE ANALYSIS

Generation rejections increased because the pipeline now detects errors that Step 7C could not: 7 contract-uncertain cases, 4 wrong relation values, 1 wrong polarity, at least 6 missing required entity/value/count cases after targeted corrections, plus language and unsupported-token failures. Accepted-generation coverage is presently English-heavy; Bangla/Banglish explanatory coverage remains the main quality tradeoff. Human scoring is still required to quantify correctness, completeness, and naturalness.

## 17. LATENCY

- Step 7C generation median: 29.10 s; P95: 81.76 s; retry total: 232.99 s.
- Step 7D generation median: 33.30 s; P95: 86.43 s; retry total: 296.64 s.
- Step 7D total median: 0.218 s; total P95: 34.64 s.
- Prompt tokens: median 694; P95 866; maximum 889 (Step 7C median 520; P95 663; maximum 680).

The semantic contract adds bounded prompt and validation overhead. Retry count rose from 9 to 14 because stricter repairable failures are now detected; the one-retry cap remains unchanged.

## 18. MEMORY

No neural model was added. Final peak RSS was 5,718,827,008 bytes versus 5,767,397,376 in Step 7C. Peak swap was 3,292,442,624 bytes versus 3,938,754,560. Minimum available RAM fell to about 5.5 MB during the run, so the 8 GB profile remains under severe memory pressure and depends on the pagefile. There is no obvious peak-memory regression, but the machine has little safety margin.

## 19. MANUAL REVIEW FILE

`results/step7d_manual_review_sample.csv` contains 30 balanced rows: 10 English, 10 Bangla, and 10 Banglish. It includes question, reference, before/after answers, strategy, source/page, supporting evidence, and validation reasons. All human fields—correctness, relevance, groundedness, completeness, naturalness, semantic consistency, error category, and notes—are present and blank. No human ratings were fabricated.

## 20. STEP 1–7C COMPATIBILITY

Step 7D does not change BGE-M3, FAISS, BM25, RRF weights, candidate counts, chunking, query normalization, reranker state, answer bank behavior, dataset ground truth, Qwen model, or quantization. Existing structured answering and evidence/citation boundaries remain in place. Historical uncommitted Step 7C files are preserved.

## 21. 70-PDF READINESS

The new rules contain no PDF filename or page-specific branch and add no corpus-sized model. They operate per verified evidence package, so the design can move to the planned 70-PDF environment without source-code redesign. Actual retrieval quality, memory, and latency at 70 PDFs remain unbenchmarked.

## 22. 6000-QUESTION INDEPENDENCE

No question IDs, current course lists, exact dataset sentences, reference answers, or fixed dataset length are used. Checkpoint keys remain `(question_id, language)`, and the evaluator is dataset-length independent. The 6,000-question workload was not run.

## 23. GIT DIFF SUMMARY

The worktree remains uncommitted as requested. Step 7D adds semantic-contract/taxonomy modules, strengthens generation/language/grounding integration, adds a checkpointed Step 7D evaluator and generic tests, and writes separate Step 7D artifacts. `git diff --check` reports no whitespace errors. Existing prior-step uncommitted changes were not discarded or overwritten.

## 24. REMAINING RISKS

- Conservative rejection substantially reduces Bangla/Banglish explanatory coverage.
- All 14 repair attempts failed to produce an accepted answer; retry value should be reviewed later, not increased blindly.
- Automatic overlap metrics are diagnostic proxies and do not replace the blank human review.
- Evidence relation extraction is intentionally shallow; complex tables, implicit cross-sentence relations, and OCR noise can still force abstention.
- The current machine reached extremely low available RAM and remains pagefile-dependent.
- The 70-PDF / 6,000-question target has not been performance-tested.

## 25. READY FOR STEP 8?

**YES**, from a Step 7D implementation and regression-safety perspective. The required semantic safeguards work, the final development run is complete with zero errors, unsafe inspected generations abstain, language consistency is 300/300, and all 159 tests pass. This does not mean thesis accuracy is established: the 30 human-review rows still require actual human scoring. Step 8 has not been started.

## 26. NEXT STEP

Complete the human review fields for the 30-row Step 7D sample, analyze correctness/completeness/naturalness by language and error category, and use that evidence to define Step 8 acceptance criteria. Do not expand to 70 PDFs or 6,000 questions until the next step is explicitly authorized.
