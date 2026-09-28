# Step 7G — verified relations and conservative document reconstruction

Development diagnostic, 2026-09-24. One-PDF corpus only. Step 8 was not started; no commit was made. The Step 7F CSVs and report were not edited. The pre-existing `requirements.txt` modification was left untouched. This is a targeted implementation/evaluation, not a new 300-answer benchmark.

## 1. Root cause after Step 7F

Step 7F often retrieved the right page, but free-form canonical generation or multilingual realization rejected usable facts. Front-matter fields could fail the course-oriented evidence gate; a 420-character excerpt could omit a fact later in the same verified passage; a repeated PDF table header split the HSS 111(B) code from its title/credit/prerequisite cells. Some candidates were genuinely wrong or ambiguous, so extraction must abstain in those cases.

## 2. Files changed

Step 7G code: `src/document_metadata.py`, `src/evidence_spans.py`, `src/relations.py`, `src/evidence.py`, `src/chunker.py`, `src/config.py`, and `src/pipeline.py`. Evaluation/tests: `scripts/evaluate_step7g.py`, `scripts/audit_step7g.py`, `tests/test_step7g_relations.py`. Outputs: this report, `results/step7g_targeted_results.csv`, `results/step7g_targeted_review.csv`, `results/step7g_banglish_audit.csv`, `results/step7g_english_regression.csv`, updated ingestion/index reports and rebuilt `vector_db` artifacts. Other dirty files in Git status belong to earlier work and were not changed for Step 7G.

## 3. Generic relation model

`EvidenceRelation` records subject, relation type, extracted values and labels, source/page/chunk, exact evidence span, target language, and continuation chunk IDs. Extraction runs only on `SUPPORTED` evidence, never the question dataset or an unverified retrieval hit. Deterministic English, Bangla and Banglish templates realize the relation. The final relation validator re-extracts it from its cited verified span and checks relation type, values, labels and provenance; language validation is also mandatory. A mismatch rejects the answer.

## 4. Relation extractors added

Edition/date, publisher, disclaimer, operation-under-Act, initially offered programs, undergraduate/postgraduate discipline counts, Category eligibility, full mark distribution, attendance-to-final-exam requirement, honors and normal-progress CGPA, regular semester count, re-examination deadline, and structurally owned course credits/prerequisites. These are document-language patterns, not question-ID, PDF-name or expected-answer rules. No second model, reranker or external API was added.

## 5. Document metadata handling

Edition/date, publisher and disclaimer requests use explicit labels. `Published by` must begin the metadata passage; an incidental later sentence saying a paper was published by someone is rejected. When the top three omit metadata, only the already retrieved bounded pool (top 30) is checked; the normal conflict gate still applies. The English edition/date and Banglish publisher cases thereby reach the labeled page-2 passage. No arbitrary corpus-wide answer lookup is used.

## 6. Evidence-span expansion

A truncated excerpt expands first from its own verified chunk. Same-parent/same-page siblings or one explicitly linked continuation may be used within a 3,600-character cap. Merely neighboring pages or retrieval ranks are never joined. Expanded text retains its original citation and any continuation IDs; unsupported retrieval text is not promoted to evidence.

## 7. Table continuation reconstruction

The page-51 repeated-header split is reconstructed only when there is one unambiguous course-code suffix and one row's title, credit and prerequisite cells. The indexed rows are distinct: `HSS 111(A) ... 2.00 Nil` and `HSS 111(B) ... 2.00 Nil`. The course relation extractor requires the query's normalized code to equal the single row's code. Synthetic ambiguous-suffix and neighboring-course tests abstain.

## 8. Index rebuild status

Chunk schema changed from `structure-aware-v1.1` to `structure-aware-v1.2`. A safe full replacement rebuilt one successful PDF, 95 text pages and 491 chunks (previously 490). A second normal build reported the FAISS index fresh and reused it. Historical Step 7F artifacts were not rebuilt or overwritten.

## 9. Six Bangla recovered cases

All six use `semi_structured_relation`, cite the named PDF page, pass language and exact-relation grounding validation, and make no answer-generation call. The evidence column states the decisive verified span; full excerpts and chunk IDs are in the targeted CSV.

| Case | Relation and evidence | Final answer |
|---|---|---|
| Q005 | Disclaimer, p.3: information subject to change at the department's discretion | ডিসক্লেইমার অনুযায়ী, the Department of Computer Science & Engineering, UAP-এর বিবেচনায় প্রসপেক্টাসের তথ্য পরিবর্তন হতে পারে। |
| Q007 | Initially offered, p.8: Computer Science & Technology and Business Administration only | UAP শুরুতে Computer Science & Technology এবং Business Administration—এই দুইটি ব্যাচেলর ডিগ্রি প্রোগ্রাম চালু করেছিল। |
| Q010 | Started under, p.8: Private University Act 1992 | UAP Private University Act 1992-এর অধীনে কার্যক্রম শুরু করেছিল। |
| Q011 | Discipline counts, p.8: nine undergraduate and eight postgraduate | UAP 9টি আন্ডারগ্র্যাজুয়েট এবং 8টি পোস্টগ্র্যাজুয়েট ডিসিপ্লিনে প্রোগ্রাম অফার করে। |
| Q019 | Category 1, p.18: prescribed courses passed, no backlog, next-semester registration | Category 1-এ ওই সেমিস্টারের নির্ধারিত সব কোর্স পাস করা এবং কোনো কোর্স বাকি না থাকা শিক্ষার্থীরা পড়ে। তারা পরবর্তী সেমিস্টারের নির্ধারিত সব কোর্সে নিবন্ধন করতে পারে। |
| Q020 | Marks distribution, p.18: assessment 30%, mid-semester 20%, final 50% | নম্বর বণ্টন হলো: মূল্যায়ন 30%, মধ্য সেমিস্টার 20%, চূড়ান্ত পরীক্ষা 50%। |

Q019 had already returned an answer in Step 7F, but its earlier presentation was unreliable; the other five were generation rejected. These new outputs require human correctness/language scoring, not just automatic validation.

## 10. Three Banglish retrieval cases

All three previously returned `INSUFFICIENT_EVIDENCE`; all now return grounded, language-validated deterministic answers without answer generation.

| Case | Previous cause | New safe result |
|---|---|---|
| Q003 | Page-2 edition/date label was present among candidates but the course-style/metadata gate failed | `Prospectus-e 13th Edition ar July 2017 date deya ache.` (p.2) |
| Q036 | HSS 111(B) code and value were split by the repeated table header | `HSS 111(B) course-er credit 2.00.` (p.51) |
| Q085 | Same split obscured ownership of the Nil prerequisite | `HSS 111(B) course-er prerequisite none listed.` (p.51) |

## 11. Broader Banglish canonical-failure analysis

Of 16 frozen Step 7F `CANONICAL_GENERATION_FAILURE` cases, 10 now return automatically validated relations (Q002, Q007, Q010, Q015, Q023, Q024, Q025, Q027, Q035, Q084). Six safely reject (Q001, Q008, Q014, Q016, Q017, Q022). Q016's former structured output failed grounding and is now rejected rather than silently returned. Wrong-evidence questions were not forced into answers. See `step7g_banglish_audit.csv`.

## 12. Qwen calls avoided

All nine targeted answers and all 10 broader Banglish recovered answers have `generation_used=False` and grounded relation output. In the six Bangla subset, five prior generation rejections were resolved without answer generation. The existing Qwen model remains available for unmatched supported cases, but no new model was downloaded.

## 13. Test results

Before Step 7G: 184 pass, one skip. Final: 195 tests run, 194 pass, one skip, zero failures. Eleven new synthetic Step 7G tests cover metadata labels/false positives, relation extraction, excerpt expansion, split-row reconstruction, ambiguous no-merge, and exact course ownership. The full local suite passed after the final safety changes.

## 14. Targeted human-review file

`results/step7g_targeted_review.csv` contains nine question/answer/citation/validation rows. Human correctness, evidence and language scores, plus notes, are deliberately blank (0 of 9 scored). `step7g_targeted_results.csv` is the machine-readable automatic trace.

## 15. English regression check

The same nine English facts were rerun without the answer bank. All nine now return validated, grounded relation answers. The frozen Step 7E baseline had four returned, three generation-rejected and two insufficient. This is a nine-case regression sample, not proof of 300-question English parity. See `step7g_english_regression.csv`.

## 16. Memory / latency impact

No new model is loaded. The relation and metadata checks are in-process text operations; the index grew by one chunk. The targeted nine-case run averaged 4.58 seconds per question (41.18 seconds total, maximum 26.50 seconds including retrieval/rewrite warm-up), with zero answer-generation calls. A controlled peak-memory comparison was not performed, so no memory improvement is claimed.

## 17. Step 1–7F compatibility

The established evidence gate, answer bank default, fallback rules, Qwen path and historical outputs remain. Metadata widening is restricted to explicit document-metadata requests and the preexisting bounded retrieval pool. The new non-generated validation guard applies to multilingual outputs; English exact-fact behavior remains compatible with prior tests. Existing Step 7F files were frozen.

## 18. 70-PDF readiness

Provenance includes document ID/source/page/chunk. The metadata check uses a bounded retrieval pool rather than scanning or loading all PDFs. Ambiguous/conflicting candidates abstain. Actual 70-PDF ingestion, duplicate metadata labels, long tables and cross-document ambiguity have **not** been validated; readiness is architectural, not demonstrated at scale.

## 19. 6000-question independence

Production code contains no question-ID lookup, expected-page lookup, reference answer, or paired-language mapping. The frozen evaluation sets are read only by offline scripts. Pattern coverage is deliberately incomplete: new formulations may still need generation or abstain. Scale to 6,000 questions is not benchmarked.

## 20. Git diff summary

Step 7G added three production modules, two evaluation scripts and one test module; changed the chunker, schema version, evidence gate and pipeline; rebuilt four vector-store artifacts and index/ingestion reports; produced four Step 7G CSVs and this report. Git status also includes pre-existing Step 7F work and the user's `requirements.txt` change; those were not committed or reset. No commit was made.

## 21. Remaining risks

Six of 16 broader Banglish failures still reject. Templates and regexes cover only clearly expressed relations; PDF extraction could distort other table layouts. The English regression sample is only nine cases. The Bangla Q005 answer retains an English department name inside Bangla grammar; human language scoring is especially valuable. Publisher anchoring is conservative and may abstain on differently arranged front matter. No 70-PDF conflict/latency or peak-memory test was done.

## 22. Full-run gate

**CLOSED.** Automatic targeted safety is strong (9/9 targeted grounded/language-valid; 10/16 broader Banglish recovered with 10/10 grounded; 9/9 paired English validated), but six broader Banglish failures remain and human review is 0/9. The expensive full 300-answer evaluation was not run merely because local tests passed. Reconsider after human review and additional representative regression checks.

## 23. Human input needed

Please score the nine rows in `step7g_targeted_review.csv`, especially semantic correctness of Q019/Q020 and Bangla/Banglish phrasing. Decide whether the six remaining safe Banglish rejections warrant another bounded Step 7G iteration before authorizing a full evaluation. No human scores were invented.

## 24. Ready for Step 8?

**NO.** Step 7G implementation and targeted automatic checks are complete, but the full-run gate is closed pending human review and broader validation. Step 8 was not started.
