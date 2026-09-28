# Step 7H final stabilization — development report

2026-09-27. No final thesis accuracy claims.

## 1. ROOT CAUSES

English non-generated answers bypassed the rejection branch after failed validation; course-credit count validation captured unrelated course-code digits and nearby totals; count intent was shadowed by grade wording; Nil label differences, an unrelated Foundation claim, and a missing every-course qualifier caused the remaining targeted defects.

## 2. FILES CHANGED

Production: `src/pipeline.py`, `src/answer_policy.py`, `src/course_rows.py`, `src/semantic_contract.py`, `src/grounding_validator.py`, `src/relations.py`. Tests: `tests/test_step7_answering.py`, `tests/test_step7h_stabilization.py`. Evaluation-only: `scripts/evaluate_step7h_targeted.py`, `scripts/evaluate_step7h_full.py`, `scripts/report_step7h.py`. Earlier uncommitted Step 7F/7G work was preserved.

## 3. LANGUAGE-PARITY VALIDATION FIX

Every non-generated factual answer now enters the same final rejection branch when language or grounding validation fails, regardless of English, Bangla, or Banglish. Generated answers already used this branch. Relation answers are also checked before return.

## 4. NUMERIC FIELD-BOUND VALIDATION

Structured credit and prerequisite answers use the requested course code plus the requested value from one verified row or direct relation. The count in a course code and adjacent table total cannot satisfy the requested credit field. Repeat-course questions require a course count rather than a grade.

## 5. DECIMAL HANDLING

Semantic numbers are atomic decimals (for example 1.50), and numerically equivalent decimal formatting is normalized for factual token matching. A trailing sentence period no longer hides a year or count. Decimal values are not split into integer fragments.

## 6. Q030 FIELD/RELATION FIX

A reusable `repeat_course_maximum` relation requires explicit source wording connecting repeat, a maximum/up-to bound, and courses. It realizes the count in the target language; `grade C` cannot serve as the answer to how many courses. No question ID is used in production.

## 7. PREREQUISITE NORMALIZATION

The same-row extractor accepts Prerequisite / Pre-requisite / Pre Requisite / Pre-Requisite / Pre- Requisite labels without editing source text. A verified Nil value is realized as no listed prerequisite with the course entity.

## 8. ENTITY-FOCUS VALIDATION

For establishment questions with a requested acronym, a separate establishment claim about a different subject is rejected even if both years are present in evidence. Supporting context is not categorically banned; the guard checks relation-subject relevance.

## 9. REQUIRED-QUALIFIER VALIDATION

Attendance relation extraction records `at_least` and, when present, `every_course`; realization includes both and relation validation rejects omitted required qualifiers. Repeat-course maximum realization and validation preserve the upper bound.

## 10. Q007 ADJUDICATION HANDLING

User decision B: page-4/reference **Computer Science & Engineering** is the evaluation reference. Page 8 says **Computer Science & Technology**; the runtime still answers from retrieved evidence and was not hardcoded to the reference. | Language | Runtime reference agreement | Runtime answer |
|---|---|---|
| english | MISMATCH | UAP initially offered the bachelor degree programs Computer Science & Technology and Business Administration. |
| bangla | MISMATCH | UAP শুরুতে Computer Science & Technology এবং Business Administration—এই দুইটি ব্যাচেলর ডিগ্রি প্রোগ্রাম চালু করেছিল। |
| banglish | MISMATCH | UAP shurute Computer Science & Technology ar Business Administration—ei duita bachelor degree program offer korechilo. |

## 11. 15 ENGLISH FLAGS BEFORE/AFTER

13 course-credit false flags corrected; Q096 label/ownership false flag corrected; Q030 true wrong-field answer corrected. Remaining mandatory grounding failures among these returned answers: 0. Correctness here is a deterministic verified-row/relation check, not a new human score.

| ID | Previous status / grounding | Before | New status / validation | After | Changed | Correctness | Reason |
|---|---|---|---|---|---|---|---|
| Q030 | ANSWER_RETURNED / SEMANTIC_CONTRACT_UNCERTAIN | The grade is C. | ANSWER_RETURNED / RELATION_EVIDENCE_VERIFIED | A student may repeat at most 4 courses for grade improvement. | True | verified repeat maximum | wrong field corrected |
| Q047 | ANSWER_RETURNED / MISSING_REQUIRED_ENTITY | CHEM 112 carries 1.50 credits. | ANSWER_RETURNED / FACTUAL_TOKENS_SUPPORTED | CHEM 112 carries 1.50 credits. | False | verified course-row credit | unrelated numeric requirements removed |
| Q050 | ANSWER_RETURNED / MISSING_REQUIRED_ENTITY | CSE 205 carries 3.00 credits. | ANSWER_RETURNED / FACTUAL_TOKENS_SUPPORTED | CSE 205 carries 3.00 credits. | False | verified course-row credit | unrelated numeric requirements removed |
| Q052 | ANSWER_RETURNED / MISSING_REQUIRED_ENTITY | MTH 201 carries 3.00 credits. | ANSWER_RETURNED / FACTUAL_TOKENS_SUPPORTED | MTH 201 carries 3.00 credits. | False | verified course-row credit | unrelated numeric requirements removed |
| Q055 | ANSWER_RETURNED / MISSING_REQUIRED_ENTITY | MTH 203 carries 3.00 credits. | ANSWER_RETURNED / FACTUAL_TOKENS_SUPPORTED | MTH 203 carries 3.00 credits. | False | verified course-row credit | unrelated numeric requirements removed |
| Q056 | ANSWER_RETURNED / MISSING_REQUIRED_ENTITY | CSE 207 carries 3.00 credits. | ANSWER_RETURNED / FACTUAL_TOKENS_SUPPORTED | CSE 207 carries 3.00 credits. | False | verified course-row credit | unrelated numeric requirements removed |
| Q062 | ANSWER_RETURNED / MISSING_REQUIRED_ENTITY | MTH 205 carries 3.00 credits. | ANSWER_RETURNED / FACTUAL_TOKENS_SUPPORTED | MTH 205 carries 3.00 credits. | False | verified course-row credit | unrelated numeric requirements removed |
| Q063 | ANSWER_RETURNED / MISSING_REQUIRED_ENTITY | ECN 201 carries 2.00 credits. | ANSWER_RETURNED / FACTUAL_TOKENS_SUPPORTED | ECN 201 carries 2.00 credits. | False | verified course-row credit | unrelated numeric requirements removed |
| Q064 | ANSWER_RETURNED / MISSING_REQUIRED_ENTITY | CSE 303 carries 3.00 credits. | ANSWER_RETURNED / FACTUAL_TOKENS_SUPPORTED | CSE 303 carries 3.00 credits. | False | verified course-row credit | unrelated numeric requirements removed |
| Q069 | ANSWER_RETURNED / MISSING_REQUIRED_ENTITY | CSE 309 carries 3.00 credits. | ANSWER_RETURNED / FACTUAL_TOKENS_SUPPORTED | CSE 309 carries 3.00 credits. | False | verified course-row credit | unrelated numeric requirements removed |
| Q073 | ANSWER_RETURNED / MISSING_REQUIRED_ENTITY | HSS 301 carries 2.00 credits. | ANSWER_RETURNED / FACTUAL_TOKENS_SUPPORTED | HSS 301 carries 2.00 credits. | False | verified course-row credit | unrelated numeric requirements removed |
| Q074 | ANSWER_RETURNED / MISSING_REQUIRED_ENTITY | CSE 313 carries 3.00 credits. | ANSWER_RETURNED / FACTUAL_TOKENS_SUPPORTED | CSE 313 carries 3.00 credits. | False | verified course-row credit | unrelated numeric requirements removed |
| Q076 | ANSWER_RETURNED / MISSING_REQUIRED_ENTITY | CSE 315 carries 3.00 credits. | ANSWER_RETURNED / FACTUAL_TOKENS_SUPPORTED | CSE 315 carries 3.00 credits. | False | verified course-row credit | unrelated numeric requirements removed |
| Q077 | ANSWER_RETURNED / MISSING_REQUIRED_ENTITY | CSE 316 carries 1.50 credits. | ANSWER_RETURNED / FACTUAL_TOKENS_SUPPORTED | CSE 316 carries 1.50 credits. | False | verified course-row credit | unrelated numeric requirements removed |
| Q096 | ANSWER_RETURNED / UNSUPPORTED_FACTUAL_TOKENS | Prerequisite: Nil. | ANSWER_RETURNED / FACTUAL_TOKENS_SUPPORTED | CHEM 112 has no listed prerequisite. | True | verified Nil prerequisite | label normalized |

## 12. THREE HUMAN REJECTS BEFORE/AFTER

| ID / language | Before | After | New status |
|---|---|---|---|
| Q030 english | The grade is C. | A student may repeat at most 4 courses for grade improvement. | ANSWER_RETURNED |
| Q006 banglish | University of Asia Pacific (UAP) 1996. The University of Asia Pacific Foundation was established in 1995-e establish hoyechilo. | Relevant information university document-e paoa geche, kintu ei muhurte ami eta reliable Banglish-e present korte parini. | GENERATION_REJECTED |
| Q023 banglish | Final exam-e boshte kompakhe 70% class-e attend korte hobe. | Final exam-e boshte protiti course-e kompakhe 70% class-e attend korte hobe. | ANSWER_RETURNED |

Q006 Banglish is a safe rejection, not a claimed correct answer.

## 13. STEP-7G TARGETED REGRESSION

9/9 prior targeted relations returned as `semi_structured_relation` with passing mandatory validation; targeted gate failures: 0.

## 14. TEST RESULTS

User-stated previous baseline: 195 run / 194 pass / 1 skip / 0 fail. Current verified run after Step 7H: **206 run / 205 pass / 1 skip / 0 fail** (`unittest discover -s tests -p test*.py`).

## 15. FULL 300 RESULT

**300/300** complete, 100 per language; one-PDF, 491-chunk fresh index; answer bank and reranker off. Development measurement, not final thesis accuracy.

| Language | Returned | Structured exact | Structured list | Relation | GGUF | Insufficient | Rejected | Ambiguous | Conflicting | Errors |
|---|---|---|---|---|---|---|---|---|---|---|
| english | 87 | 21 | 0 | 63 | 3 | 3 | 8 | 1 | 1 | 0 |
| bangla | 86 | 21 | 0 | 62 | 3 | 3 | 7 | 3 | 1 | 0 |
| banglish | 87 | 23 | 0 | 60 | 4 | 0 | 11 | 1 | 1 | 0 |

Retrieval metrics (entity/field rates use applicable denominators):

| Language | Page@1 | Page@3 | MRR@3 | Entity@1 | Entity@3 | Field@1 | Field@3 | Support@1 | Support@3 |
|---|---|---|---|---|---|---|---|---|---|
| english | 67.0% | 99.0% | 0.828 | 100.0% | 100.0% | 93.5% | 97.8% | 91.0% | 94.0% |
| bangla | 53.0% | 97.0% | 0.740 | 100.0% | 100.0% | 94.4% | 97.8% | 85.0% | 88.0% |
| banglish | 53.0% | 95.0% | 0.735 | 100.0% | 100.0% | 94.4% | 100.0% | 93.0% | 97.0% |
| overall | 57.7% | 97.0% | 0.768 | 100.0% | 100.0% | 94.1% | 98.5% | 89.7% | 93.0% |

Frozen retrieval baseline comparison:

| Language | Step5 page@3 | Step7F page@3 | Step7G page@3 | Step7H page@3 | Step5 support@3 | Step7F support@3 | Step7G support@3 | Step7H support@3 |
|---|---|---|---|---|---|---|---|---|
| english | 99.0% | 99.0% | 99.0% | 99.0% | 94.0% | 93.0% | 94.0% | 94.0% |
| bangla | 94.0% | 97.0% | 97.0% | 97.0% | 86.0% | 91.0% | 88.0% | 88.0% |
| banglish | 95.0% | 95.0% | 95.0% | 95.0% | 96.0% | 95.0% | 97.0% | 97.0% |
| overall | 96.0% | 97.0% | 97.0% | 97.0% | 92.0% | 93.0% | 93.0% | 93.0% |

Generation and validation:

| Language | Language pass | Grounding pass / returned | Attempts | Accepted | Rejected | Retries | Successful | Failed |
|---|---|---|---|---|---|---|---|---|
| english | 100/100 | 87/87 | 6 | 3 | 2 | 1 | 0 | 1 |
| bangla | 100/100 | 86/86 | 7 | 3 | 2 | 1 | 0 | 1 |
| banglish | 100/100 | 87/87 | 12 | 4 | 4 | 2 | 0 | 2 |

Multilingual support parity: ENGLISH_AND_BANGLISH=4, ALL_THREE_SUPPORTED=91, OTHER_MISMATCH=3, NONE_SUPPORTED=2. Step-5/7F comparisons are descriptive because the index/evidence definitions differ; Step-7G and 7H use the same 491-chunk index.

## 16. RETURNED-ANSWER VALIDATION FAILURES

**0** of 260 returned answers failed mandatory grounding or language validation. Semantic failures among returned answers: **0 surfaced by the final grounding contract**; a separate per-row semantic Boolean was not persisted for relation answers, which use re-extraction and qualifier validation.

## 17. HUMAN REVIEW FILE

`results/step7h_full_manual_review.csv`: 30 rows, exactly 10 per language. It includes previous rejects/partials and mixed strategies. The eight human scoring fields are blank; no new human acceptance judgment is claimed.

## 18. LATENCY / MEMORY

Mean 3.37s, median 0.26s, P95 27.86s, maximum 186.88s per query; summed 1012.0s. Peak process RSS 5425 MiB, peak working set 5475 MiB, minimum available RAM 18 MiB, maximum system-wide pagefile use 3037 MiB. Pagefile use is not solely attributable to this process.

## 19. STEP 1–7G COMPATIBILITY

The pinned one-PDF index, BGE-M3/FAISS, BM25/RRF retrieval, cross-lingual fallback, Qwen2.5-7B generation path, and existing Step-7G relation types were retained. The nine targeted Step-7G cases remained accepted. Earlier incomplete and completed pre-fix checkpoints were archived for audit, not mixed into final measurements.

## 20. 70-PDF READINESS

Not established by this one-PDF run. Multi-PDF ingestion remains covered by existing tests, but the 70-PDF corpus must be ingested, indexed, and measured separately for resource use, collision handling, retrieval quality, and conflict behavior.

## 21. 6000-QUESTION INDEPENDENCE

Questions are evaluated independently at runtime; paired translations and references are evaluation-only. The 300-variant checkpointed run does not establish 6000-question accuracy or throughput. No answer bank or paired translation lookup was enabled.

## 22. GIT DIFF SUMMARY

Working tree: 17 modified and 49 untracked paths, including pre-existing Step 7F/7G artifacts. No commit was created. The Step 7H code changes are limited to validation, same-row value extraction, and deterministic relation/wording logic; index and retrieval weights were not changed.

## 23. REMAINING RISKS

Safe rejections remain, including Q006 Banglish. Q007 has a reference mismatch under the approved page-4 Engineering adjudication despite page-8 Technology evidence. New 30-row human review is pending. Generation rejection reasons: NO_SAFE_ANSWER_FROM_SUPPORTED_EVIDENCE=13, SEMANTIC_CONTRACT_UNCERTAIN=6, MISSING_REQUIRED_ENTITY=3, ENTITY_FOCUS_MISMATCH=1, UNSUPPORTED_REALIZATION_FACTS=1, WRONG_RELATION_VALUE=1, WRONG_FIELD_VALUE=1. Runtime exceptions: 0.

## 24. STEP 7 ENGINEERING-COMPLETE?

**YES, for the tested one-PDF engineering-safety scope.** Targeted and full-run gates passed with zero returned validation failures. This is provisional pending new human review and does not establish 70-PDF or thesis accuracy.

## 25. READY FOR STEP 8?

**NO automatic transition.** Complete the new human review, review the documented Q007 mismatch under the approved evaluation reference, and separately plan the larger-corpus/resource validation. Step 8 was not started.
