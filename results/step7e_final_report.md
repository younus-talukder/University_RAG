# Step 7E Final Report — Two-Stage Multilingual Realization

**Development/regression evaluation only — not final thesis accuracy.** Step 8 was not started. No commit was created.

## 1. ROOT CAUSE

The Step 7D user-facing fallback merged two different conditions: missing evidence and rejected generation. The trace found verified supporting evidence before generation in **7/10 Bangla** review rows and **10/10 Banglish** review rows. Of the multilingual fallbacks, **16** had verified evidence: **6 Bangla + 10 Banglish**. Those 16 were generation/validation failures, not evidence-missing cases.

Bangla failure labels were: entity validation 1, field validation 1, no evidence 1, generation semantic 1, retry failed 5. Banglish labels were: generation semantic 6 and retry failed 4.

## 2. FILES CHANGED

- Runtime: `src/pipeline.py`, `src/generator.py`, `src/language_validator.py`, `src/semantic_contract.py`.
- New runtime modules: `src/realization_validator.py`, `src/semistructured_realizer.py`.
- Evaluation: `scripts/evaluate_step7e.py`.
- Tests: `tests/test_step7e_generation.py`, `tests/test_step7_answering.py`, `tests/test_evidence_contract.py`.
- Documentation: `README.md` and this report.
- Results: failure trace, 30-row review run, before/after review, balanced manual-review sample, 300-row results, and summary JSON.

The working tree also contains accumulated, uncommitted Step 7/7B/7C/7D files from earlier work. They were not committed or discarded.

## 3. FAILURE TRACE

`results/step7e_failure_trace.csv` contains 30 unique review rows and records language, question, requested entity/field, top-three chunks, evidence, pre-generation status, requested strategy, first output and validations, retry details, final answer/status, fallback reason, classification, latency, and error.

The trace was captured before changing behavior. Its classification was later corrected without rerunning the model: Q001 Bangla is an entity-context failure; Q003 Bangla is a field-validation failure; Q005 Bangla is genuine no-evidence.

## 4. STATUS-SEMANTICS FIX

- `INSUFFICIENT_EVIDENCE`: retrieval/evidence genuinely cannot support the answer.
- `GENERATION_REJECTED`: verified evidence exists, but deterministic extraction, canonical generation, or target-language realization cannot produce a safe answer.

Generation rejection no longer mutates `support_status=supported` into `insufficient`, erases citations, or tells the user that documents lacked information. Bangla and Banglish use distinct natural fallback messages.

## 5. CANONICAL ANSWER ARCHITECTURE

Every generation-required question first creates a short English canonical answer from the original question, requested entity/field, and verified evidence. It must pass English-language, evidence grounding, entity, relation/value, count/condition, and polarity validation before any transformation. Unsafe canonical content is never translated.

English queries return the validated canonical answer directly. The architecture does not depend on question IDs, filenames, course lists, reference answers, or ground truth.

## 6. BANGLA REALIZATION

Bangla receives only the validated canonical answer. The constrained prompt requires natural Bengali, exact preservation of protected Latin identifiers/names/numbers, unchanged polarity, and no extra explanation. Reliable parsed relations can use deterministic formatting. The final review gate returned **3/6** generation-required Bangla answers and safely rejected 3.

## 7. BANGLISH REALIZATION

Banglish receives only the validated canonical answer. It requires Latin-script Bangla grammar, rejects pure English and Bengali Unicode, preserves protected facts, and uses a targeted repair at most once. Reliable parsed relations can use deterministic Banglish templates. The final review gate returned **5/10** generation-required Banglish answers and safely rejected 5.

## 8. FACT/POLARITY VALIDATION

The realization validator preserves emails, course codes, numbers, percentages, dates through numeric preservation, official acronyms, relation category, and polarity. It rejects missing facts, added facts, script/language mismatch, wrong negation, and relation changes such as converting accreditation into publication.

The seven mandated cases are covered, plus tests for protected-fact-heavy Bangla, relation templates, call budget, canonical-only realization input, and safe structured-extraction failure.

## 9. STRUCTURED/SEMI-STRUCTURED COVERAGE

Existing structured exact/list behavior remains outside two-stage generation. The 300-row run produced structured answers by language: English **67**, Bangla **71**, Banglish **69**.

Generic semi-structured realization is limited to reliably recognized English relation shapes: `published_by`, `accredited_by`, `established`, `operates_under`, `located_at`, fee, and named role. Unrecognized or unsafe shapes remain model-constrained or are rejected. No question ID is hardcoded in runtime behavior.

## 10. CALL BUDGET / RETRY POLICY

Maximum model calls for a multilingual question: **3**.

- Path A: canonical attempt + canonical repair + one realization.
- Path B: canonical attempt + realization + one targeted realization repair.
- Deterministic semi-structured realization consumes no additional model call.
- No repeated loop is permitted.

The same cached Qwen2.5-7B-Instruct Q4_K_M instance is reused for all model calls.

## 11. TEST RESULTS

- Previous baseline: **159**.
- New net tests: **14**.
- Total run: **173**.
- Failures: **0**.
- Skipped: **1**.

Required command: `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test*.py"`.

## 12. 30-ROW BEFORE VS AFTER

Generation-required accepted answers:

- English: **7/10 before → 7/10 after**.
- Bangla: **0/6 before → 3/6 after**.
- Banglish: **0/10 before → 5/10 after**.

The final balanced gate contains 10 English, 10 Bangla, and 10 Banglish rows. Rejection is not counted as success. Returned multilingual answers were inspected against reference and evidence; unsafe publisher and changed-relation examples remain rejected.

## 13. BANGLA RESULTS

- Evidence available for generation-required rows: **6/6**.
- Canonical successful: **4/6**.
- Realization successful: **3/6**.
- Final generated answer returned: **3/6**.
- Generation rejected: **3/6**.

There is also one deterministic structured Bangla answer in the review sample. Two other Bangla rows are genuine insufficient evidence and one is ambiguous.

## 14. BANGLISH RESULTS

- Evidence available for generation-required rows: **10/10**.
- Canonical successful: **6/10**.
- Realization validator successful: **6/10**.
- Final generated answer returned after grounding: **5/10**.
- Generation rejected: **5/10**.

One incomplete program-count answer reflects a truncated retrieved excerpt; this remains a retrieval/evidence-completeness risk and is not presented as final thesis correctness.

## 15. 300-QUERY RESULTS

The success gate passed, so the checkpointed run was performed. Final integrity: **300 rows, 300 unique `(question_id, language)` keys, 0 errors**. Answer bank OFF; reranker OFF.

Overall final statuses: ANSWER_RETURNED **224**, GENERATION_REJECTED **49**, INSUFFICIENT_EVIDENCE **19**, AMBIGUOUS **5**, CONFLICTING **3**.

| Language | Structured | Generation required | Canonical successful | Realization successful | Final generated answers | Generation rejected | Insufficient evidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| English | 67 | 11 | 7 | N/A | 7 | 19 | 5 |
| Bangla | 71 | 6 | 4 | 3 | 3 | 11 | 11 |
| Banglish | 69 | 14 | 10 | 8 | 7 | 19 | 3 |

`GENERATION_REJECTED` includes supported structured fields whose deterministic extractor could not safely produce a value; these rows do not call Qwen. Counts therefore exceed generation-required model rows.

## 16. LANGUAGE QUALITY

Bangla requires meaningful Bengali grammar, with a narrow allowance for short sentences dominated by protected official Latin names, addresses, codes, or numbers. Banglish requires Latin script plus Bangla-function structure and rejects pure English and Bengali-script leakage.

Technical identifiers and official names intentionally remain in their source form. Some accepted outputs are code-mixed by design; naturalness still requires human scoring in the blank manual-review sheet.

## 17. GROUNDING / SEMANTIC VALIDATION

Canonical and realization stages are validated independently. A realization may pass canonical-fact preservation yet still fail evidence grounding. Subject safeguards prevent a prospectus-publisher question from accepting an unrelated IJCIT publishing passage. Strict relation comparison rejects accreditation/publication swaps. Exact counts use the highest-ranked parsed relation fact rather than incidental numbers elsewhere in a long passage.

## 18. LATENCY

Median seconds in the 300-row development run:

- Canonical generation: **25.908 s**.
- Bangla realization: **0.001 s** (median dominated by deterministic relation templates).
- Banglish realization: **16.327 s**.
- Total generation: **27.754 s**.

Model-based realizations remain substantially slower than deterministic formatting. Quality/safety was prioritized over latency.

## 19. MEMORY

Only the existing Qwen 7B model was used; no translator, 3B, 1.5B, reranker, or other neural model was loaded. Peak observed process RSS was **5,878,292,480 bytes (~5.47 GiB)**. Minimum observed available RAM was **52,772,864 bytes (~50.3 MiB)** and peak swap use was **2,087,407,615 bytes (~1.94 GiB)**, confirming that the present machine remains memory-tight.

## 20. MANUAL REVIEW FILE

`results/step7e_manual_review_sample.csv` contains exactly 30 rows balanced 10/10/10 across English/Bangla/Banglish. Columns include reference, supporting evidence, canonical answer, final answer, and status. All eight human-review fields are blank: correctness, relevance, groundedness, completeness, naturalness, semantic consistency, overall acceptability, and review notes.

## 21. STEP 1–7D COMPATIBILITY

BGE-M3, FAISS, BM25, metadata retrieval, RRF, query normalization, structure-aware chunks, evidence contract, citations, answer bank OFF, reranker OFF, Qwen 7B, structured answers, checkpoint/resume, and prior one-retry safety behavior remain in place. Existing tests plus new regressions pass.

## 22. 70-PDF READINESS

The generation/realization design consumes only verified evidence packages and does not depend on a particular filename or PDF count. It is architecturally compatible with 70 PDFs. Actual 70-PDF retrieval quality, index size, ambiguity, and memory behavior remain future empirical work and are not claimed by this one-PDF run.

## 23. 6000-QUESTION INDEPENDENCE

Runtime behavior is independent of evaluation question count and ground truth. The strategy depends on verified answer content, semantic contract, and target language. Checkpoint/resume uses `(question_id, language)` only for evaluation bookkeeping. The future 6,000-question benchmark remains unrun.

## 24. GIT DIFF SUMMARY

No commit was created. The working tree contains accumulated Step 7 changes and generated artifacts. `git diff --check` reports no whitespace errors; only existing Windows LF→CRLF warnings appear. Step 7E adds the realization validator, semi-structured realizer, evaluator, tests, documentation, and result artifacts, and updates the pipeline/generator/language/semantic modules.

## 25. REMAINING RISKS

- Human correctness/naturalness scores are intentionally blank; accepted is not synonymous with thesis-level correctness.
- Retrieval can return incomplete evidence, as seen in a postgraduate-discipline count.
- Some safe Bangla/Banglish outputs remain technical and code-mixed.
- Current memory headroom is extremely low during 7B evaluation.
- Deterministic date/field extractors do not cover every supported structured relation; safe rejection is used instead.
- The results cover one indexed curriculum PDF, not the future 70-PDF corpus.

## 26. READY FOR STEP 8?

**YES — engineering gate passed**, with the stated human-review, retrieval-completeness, and memory risks. This does not claim final thesis accuracy.

## 27. NEXT STEP

Do not start automatically. The next step is the user-authorized Step 8 plan, preceded by human scoring of `results/step7e_manual_review_sample.csv` and review of remaining retrieval-completeness failures.
