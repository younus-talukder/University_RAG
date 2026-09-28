# Step 8C — final Step-8 stabilization and freeze check

## 1. STEP 8C GOAL

Close the generic Bangla marks-distribution intent gap and check whether Step 8 can be frozen. Step 9 was not started.

## 2. Q020 ROOT CAUSE

The prior Bangla query used `মার্কস বণ্টন`, which the field and runtime-intent patterns did not recognize. It fell to `general` despite source support; retrieval, relation values, and generation did not cause the failure.

## 3. FILES CHANGED

`src/query_normalization.py`, `src/evidence.py`, `src/fast_answer.py`, `src/relations.py`, `src/answer_safety.py`, `tests/test_step8c_mark_distribution.py`, `scripts/evaluate_step8c.py`, and `scripts/report_step8c.py`. No previous Step 7/8/8B results were overwritten.

## 4. BANGLA FIELD NORMALIZATION

One shared recognizer handles contextual Bangla mark/number distribution wording. It maps to existing `assessment` and `mark_distribution` concepts. A bare `নম্বর` does not trigger distribution; no answer percentages are stored in normalization.

## 5. MULTILINGUAL INTENT PARITY

- english: field `assessment`, relation `mark_distribution`, extracted `mark_distribution`
- bangla: field `assessment`, relation `mark_distribution`, extracted `mark_distribution`
- banglish: field `assessment`, relation `mark_distribution`, extracted `mark_distribution`

## 6. Q020 BEFORE / AFTER

Before: `INSUFFICIENT_EVIDENCE` with field `general` and no extracted relation. After: `SUPPORTED` with field `assessment` and relation `mark_distribution`. Answer: নম্বর বণ্টন হলো: মূল্যায়ন 30%, মধ্য সেমিস্টার 20%, চূড়ান্ত পরীক্ষা 50%।

## 7. Q020 EVIDENCE / TRUST VALIDATION

All three variants returned Assessment 30%, Mid Semester 20%, and Final Exam 50% in their target languages from `curricula_BSc-Curriculum-New.pdf`, page 18. Language, grounding, semantic, and expanded safety checks passed for each.

## 8. STEP-7H COMPATIBILITY GATE

Passed: 26/26. Failed: 0. Frozen expectations unchanged.

## 9. STEP-8B REGRESSION GATE

- Q014: english SUPPORTED; bangla SUPPORTED; banglish SUPPORTED
- Q016: english GENERATION_REJECTED; bangla RETRIEVAL_UNCERTAIN; banglish GENERATION_REJECTED
- Q018: english SUPPORTED; bangla CONFLICTING_EVIDENCE; banglish CONFLICTING_EVIDENCE
- Q031: english GENERATION_REJECTED; bangla GENERATION_REJECTED; banglish GENERATION_REJECTED
- Q081: english SUPPORTED; bangla SUPPORTED; banglish SUPPORTED

Q014 retains both emails; Q016 never returns semester count as duration; Q018/Q031 are not falsely ambiguous; Q081 retains None/Nil equivalence.

## 10. PREVIOUS HUMAN-REJECT CASES

- Q030:english: ANSWER_RETURNED — A student may repeat at most 4 courses for grade improvement.
- Q006:banglish: GENERATION_REJECTED — Relevant tothyo paoa geche, kintu reliable uttor toiri kora jayni.
- Q023:banglish: ANSWER_RETURNED — Final exam-e boshte protiti course-e kompakhe 70% class-e attend korte hobe.

## 11. STEP-7G TARGETED RELATION GATE

Passed: 9/9. Failed: 0.

## 12. UNSAFE ANSWER AUDIT

Count: 0. Target: 0. The audit covers all 18 current target rows and all 26 frozen compatibility rows, including relation, value, qualifier, entity, language, grounding, and available semantic signals.

## 13. TEST RESULTS

Previous: 230. New: 2. Total: 232. Failures: 0. Skipped: 1. Last output: `---------------------------------------------------------------------- | Ran 232 tests in 0.822s |  | OK (skipped=1)`.

## 14. FULL 300 RERUN

Performed: NO. Only the three Q020 variants match the new distribution recognizer; targeted and frozen compatibility gates passed. The completed Step 8B measurement remains 258/300 safe answers (86.0%); it is not a post-Step-8C full-run score and is not accuracy.

## 15. FRESH MANUAL REVIEW SAMPLE

Path: `results/step8c_final_manual_review.csv`. Rows: 30; English 10, Bangla 10, Banglish 10. 24 unchanged Step 8B snapshots, 3 fresh Q020 rows, and 3 fresh synthetic ambiguity rows; no structured_list answer existed in the frozen development run. Human scoring fields are blank.

## 16. MEMORY

Across sequential targeted gates, peak sampled RSS 5.31 GiB, minimum available RAM 40 MiB, peak system pagefile use 2.23 GiB. Generator configuration was unchanged; no parallel workers.

## 17. Q007 AUTHORITY CHECK

Evaluation-only authority remains page 8, `Computer Science & Technology` (`PAGE_8_COMPUTER_SCIENCE_AND_TECHNOLOGY`). No production Q007 rule was introduced.

## 18. STEP 1–8 COMPATIBILITY

Frozen Step 7H, Step 8B safety cases, previous human-reject cases, nine Step 7G relations, and the unit suite passed. Retrieval/index/model/chunking architecture was unchanged.

## 19. 70-PDF READINESS

Architectural only. The generic recognizer is not tied to a PDF or question ID; ingestion, retrieval, memory, and safety on 70 PDFs have not been validated.

## 20. 6000-QUESTION INDEPENDENCE

Production normalization uses question wording only, not reference answers or dataset IDs. The 300 development questions are not an independent 6000-question accuracy evaluation.

## 21. GIT DIFF SUMMARY

No commit. The workspace already contained uncommitted Step 7/8 changes. Step 8C adds a narrow shared recognizer, its tests, and separate measurement/report artifacts; existing result files were preserved.

## 22. REMAINING RISKS

No accepted `structured_list` answer existed in the frozen development run, so that strategy cannot be represented honestly in the 30-row sample. The 8-GB machine had very low RAM headroom in Step 8B; 70-PDF and 6000-question scale remain untested. Human review is still required before thesis claims.

## 23. STEP 8 ENGINEERING-COMPLETE?

YES — for the specified development engineering gates, not for thesis accuracy or scale validation.

## 24. READY FOR STEP 9?

YES for a separately authorized next phase. Step 9 has not been started.

## 25. NEXT STEP

Freeze Step 8 code and review the 30-row human sample; plan Step 9 separately only after user authorization and with explicit memory limits.
