# Step 8B — final answerability stabilization

## 1. STEP 8B ROOT CAUSES

Question intent was inferred from the extracted relation; structured exact extraction selected one email; prerequisite conflict comparison used unbounded raw strings; ambiguity used field-wide entity requirements. The old unsafe audit checked validator booleans only.

## 2. FILES CHANGED

`src/answer_safety.py`, `src/evidence.py`, `src/pipeline.py`, `scripts/evaluate_step8.py`, `scripts/evaluate_step8b.py`, `scripts/report_step8b.py`, and `tests/test_step8b_safety.py`. Existing Step 8 output files remain untouched.

## 3. RELATION-INTENT ALIGNMENT FIX

Semester count, duration, class-week duration, and names are distinguished from question wording. A contradictory extracted relation is blocked before answer generation. The final audit also independently checks returned semester-duration answers for time units.

## 4. MULTI-VALUE COMPLETENESS FIX

Explicit email values are read from the relevant labeled field only; structured answers preserve all addresses unless the question requests one. The return gate and offline audit also check every extracted component of bounded program lists, mark distributions, discipline-count pairs, and labeled prerequisite-code lists. They do not infer arbitrary neighboring values as required.

## 5. EQUIVALENT-VALUE CONFLICT NORMALIZATION

Prerequisite `None`, `Nil`, and clear no-prerequisite phrases normalize to one relation-specific sentinel before comparison; labeled value extraction stops before following prose. The unrelated 3.0-versus-4.0 credit conflict remains conflicting.

## 6. ENTITY-INDEPENDENT POLICY HANDLING

Credit-assignment rules for theoretical courses and seat-reservation percentages can be relation-sufficient without a course entity. Bare fields and unbound pronouns remain ambiguous.

## 7. UNSAFE-ANSWER AUDIT IMPROVEMENT

The audit records validator failures, weak returned evidence, requested/extracted relation mismatch, mandatory email and bounded-list component loss, key qualifier loss, and subject course-code mismatch. This is an automatic conservative audit, not human correctness certification.

## 8. TARGETED CASES BEFORE/AFTER

**Q016:**
- english: GENERATION_REJECTED → GENERATION_REJECTED; answer: Relevant information was found, but I couldn't produce a reliable answer.
- bangla: SUPPORTED → RETRIEVAL_UNCERTAIN; answer: সম্পর্কিত তথ্য পাওয়া গেছে, কিন্তু তা এই নির্দিষ্ট প্রশ্নের উত্তর কি না নিশ্চিত করা যায়নি।
- banglish: GENERATION_REJECTED → GENERATION_REJECTED; answer: Relevant tothyo paoa geche, kintu reliable uttor toiri kora jayni.
**Q014:**
- english: GENERATION_REJECTED → SUPPORTED; answer: The email address is admission@uap-bd.edu, registrar@uap-bd.edu.
- bangla: SUPPORTED → SUPPORTED; answer: প্রাসঙ্গিক email address হলো admission@uap-bd.edu, registrar@uap-bd.edu।
- banglish: GENERATION_REJECTED → SUPPORTED; answer: Relevant email address holo admission@uap-bd.edu, registrar@uap-bd.edu.
**Q018:**
- english: AMBIGUOUS_QUERY → SUPPORTED; answer: One lecture per week per semester is equivalent to one credit for theoretical courses.
- bangla: AMBIGUOUS_QUERY → CONFLICTING_EVIDENCE; answer: বিশ্ববিদ্যালয়ের নথিতে এই বিষয়ে পরস্পরবিরোধী তথ্য রয়েছে।
- banglish: AMBIGUOUS_QUERY → CONFLICTING_EVIDENCE; answer: University document-gulote ei bishoye conflicting tothyo ache.
**Q031:**
- english: AMBIGUOUS_QUERY → GENERATION_REJECTED; answer: Relevant information was found, but I couldn't produce a reliable answer.
- bangla: AMBIGUOUS_QUERY → GENERATION_REJECTED; answer: প্রাসঙ্গিক তথ্য পাওয়া গেছে, কিন্তু নির্ভরযোগ্য উত্তর তৈরি করা যায়নি।
- banglish: AMBIGUOUS_QUERY → GENERATION_REJECTED; answer: Relevant tothyo paoa geche, kintu reliable uttor toiri kora jayni.
**Q081:**
- english: CONFLICTING_EVIDENCE → SUPPORTED; answer: The prerequisite for CSE 101 is none listed.
- bangla: CONFLICTING_EVIDENCE → SUPPORTED; answer: CSE 101 কোর্সের পূর্বশর্ত: none listed।
- banglish: CONFLICTING_EVIDENCE → SUPPORTED; answer: CSE 101 course-er kono prerequisite deya nei.

## 9. SYNTHETIC REGRESSION RESULTS

Step 8B tests cover two-email completeness, None/Nil equivalence, true credit conflict, four semester relations, policy sufficiency, genuine ambiguity, qualifier loss, and entity mismatch. Targeted 15-case gate: PASS. The extended audit also flags the saved Step 8 Q014 Bangla as `REQUIRED_VALUE_LOSS` and Q016 Bangla as `RELATION_MISMATCH`.

## 10. TEST RESULTS

Exit code: 0. Final test output:

```text
answered 9/10 english Q009 elapsed=0.0s eta=0.0s
answered 10/10 english Q010 elapsed=0.0s eta=0.0s

............................................................s.........................................................................................................................................................................
----------------------------------------------------------------------
Ran 230 tests in 0.795s

OK (skipped=1)
```

## 11. FULL 300 RESULT

300/300 completed on the current index with answer bank and reranker off. Runtime errors: 0. Generation used in 19 cases, 26 attempts, 11 accepted generated answers, and 4 retried cases. Median latency 0.27s; p95 31.21s. Strategy counts: {'generation_rejected': 24, 'unsupported': 9, 'semi_structured_relation': 183, 'structured_exact': 64, 'gguf_generation': 11, 'supported': 1, 'conflicting': 8}.

## 12. SAFE ANSWER COVERAGE

Automatically accepted: 258/300 (86.0%), versus 259/300 in the saved Step 8 baseline. By language: english 88/100, bangla 83/100, banglish 87/100. These are not thesis accuracy claims.

## 13. ABSTENTION DISTRIBUTION

GENERATION_REJECTED 24, INSUFFICIENT_EVIDENCE 8, RETRIEVAL_UNCERTAIN 2, CONFLICTING_EVIDENCE 8. Step 8 baseline: {'GENERATION_REJECTED': 23, 'INSUFFICIENT_EVIDENCE': 8, 'SUPPORTED': 259, 'AMBIGUOUS_QUERY': 6, 'RETRIEVAL_UNCERTAIN': 1, 'CONFLICTING_EVIDENCE': 3}; Step 8B: {'GENERATION_REJECTED': 24, 'INSUFFICIENT_EVIDENCE': 8, 'SUPPORTED': 258, 'RETRIEVAL_UNCERTAIN': 2, 'CONFLICTING_EVIDENCE': 8}.

## 14. UNSAFE RETURNED ANSWERS

Extended automatic audit: 0. Target: 0. Review `results/step8b_unsafe_answer_review.csv` and the existing human review sample; zero automatic flags cannot exclude all semantic errors.

## 15. MEMORY

Peak sampled RSS 5.12 GiB; peak process working set 5.23 GiB; minimum available RAM 0.01 GiB; peak system pagefile use 4.78 GiB. Sequential execution, no new model.

## 16. STEP 1–8 COMPATIBILITY

The retriever, index, chunking, model choice, answer bank default, reranker setting, and Step 8 trust architecture were not redesigned. The frozen Step 7H targeted gate completed 26/26 but failed 1 check: ('Q020', 'bangla'): Step-7G relation regression. Q020 Bangla is a safe `INSUFFICIENT_EVIDENCE` abstention in both the saved Step 8 baseline and Step 8B, so this was not introduced by Step 8B. The user prohibited one-off coverage chasing, so no extractor was added. Full unit suite result is above.

## 17. 70-PDF READINESS

The rules use question/evidence text rather than file names or question IDs. Actual 70-PDF behavior remains untested and needs a fresh ingestion/index and independent validation.

## 18. 6000-QUESTION INDEPENDENCE

No production rule reads dataset answers or IDs. This 300-variant development rerun is not evidence of accuracy or independence on the future 6000-question evaluation; a separate held-out protocol remains necessary.

## 19. GIT DIFF SUMMARY

No commit made. This workspace already contained uncommitted Step 7/8 work and generated artifacts. Current tracked diff:

```text
app.py                          |  26 ++++-
 results/index_build_report.json |  24 ++--
 results/ingestion_report.json   |  12 +-
 src/answer_policy.py            |  12 ++
 src/chunker.py                  |  76 +++++++++++++
 src/config.py                   |   2 +-
 src/evidence.py                 |  62 +++++++++-
 src/generator.py                |  15 +++
 src/grounding_validator.py      |   6 +-
 src/pipeline.py                 | 243 ++++++++++++++++++++++++++++++++++++++--
 src/query_normalization.py      |   6 +
 src/semantic_contract.py        |  65 ++++++++++-
 tests/test_step7_answering.py   |  14 ++-
 vector_db/index.faiss           | Bin 2007085 -> 2011181 bytes
 vector_db/index_manifest.json   |  37 +++---
 vector_db/metadata.pkl          | Bin 407331 -> 407696 bytes
 vector_db/sparse_index.pkl      | Bin 208195 -> 208190 bytes
 17 files changed, 534 insertions(+), 66 deletions(-)
```

Step 8B new files are untracked; see repository status for the full list. No existing result artifact was overwritten.

## 20. REMAINING RISKS

Automatic checks cannot fully resolve paraphrased set completeness or every policy scope. Q016 may safely abstain despite relevant duration evidence; some Q018/Q031 variants may abstain for evidence/validation reasons. The legacy Step 7H Q020 Bangla relation gate remains failed, although it is a pre-existing safe abstention. Minimum sampled available RAM was very low, so operational headroom on this 8-GB machine is not established. Manual answer review remains necessary.

## 21. STEP 8 ENGINEERING-COMPLETE?

NO. The Step 8B safety targets passed, but the frozen Step 7H compatibility gate still has a pre-existing safe-abstention failure. Do not freeze Step 8 as fully regression-clean.

## 22. READY FOR STEP 9?

NO. The remaining compatibility exception and low-memory headroom need an explicit engineering decision; Step 9 was not started.
