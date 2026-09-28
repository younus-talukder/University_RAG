# Step 7F — multilingual retrieval/evidence parity

Development diagnostic, 2026-09-24. One-PDF corpus, 100 base questions, 300 variants. No commit; Step 8 not started. The full 300-answer run was **not** authorized by the success gate below. The 300-row retrieval-only diagnostic is not a full answer evaluation.

## 1. Root cause analysis

The 50 Bangla/Banglish safe fallbacks comprise 14 initial insufficient-evidence cases, 30 generation rejections, four ambiguous queries, and two conflicting-evidence queries. Generation rejection is not a retrieval miss.

Primary funnel stages (one per failure; conservative heuristic classification):

| Stage | Bangla | Banglish |
|---|---:|---:|
| Field detection / cross-script evidence mismatch | 8 | 0 |
| Correct page, wrong/unsafe chunk relation | 3 | 3 |
| Canonical generation failure | 10 | 16 |
| Language realization failure | 1 | 2 |
| Semantic validation rejection | 0 | 1 |
| Ambiguous query | 3 | 1 |
| Conflicting evidence / other | 1 | 1 |
| True information proven absent | 0 | 0 |

The English-supported chunk appears among original candidates at top 1/3/10/30 for 10/13/16/17 of 26 Bangla fallbacks, and 15/17/18/19 of 24 Banglish fallbacks. For the eligible insufficient-evidence subset alone, the counts are 1/4/7/8 of 11 Bangla and 0/0/1/1 of three Banglish cases. These are offline comparisons to paired English evidence, **not** runtime inputs. A page hit does not establish a safe entity–field relation: HSS 111(B) reaches the right page, but the chunk/layout still does not prove the requested course-specific credit or prerequisite; paired English also fails those cases.

## 2. Files changed

`src/crosslingual.py`, `src/query_normalization.py`, `src/generator.py`, `src/pipeline.py`, `scripts/evaluate_step7f.py`, `tests/test_step7f_crosslingual.py`, and the six Step 7F CSVs listed below. The pre-existing modification to `requirements.txt` was left untouched.

## 3. Cross-lingual retrieval architecture

Normal multilingual BGE-M3 + FAISS/BM25/metadata/RRF retrieval runs first. Only an initially **insufficient** Bangla/Banglish case may construct an auxiliary English query and use the same retriever a second time. English, supported, conflicting, ambiguous, and generation-rejected paths do not retry. No model, reranker, candidate count, RRF weight, chunking, or index change was introduced. The original question remains the final-answer language source.

## 4. Field-concept normalization

A small centralized search-concept table in `crosslingual.py` maps high-confidence English/Bangla/Banglish cues for credits, prerequisites, title/code, attendance, exam/assessment, percentages, semester, program, publication/publisher, accreditation, topics, objective/CLO/weekly content, requirement, date, disclaimer, registration, establishment, graduation, law, discipline, category, edition, and prospectus. These are generic search labels, not course-answer mappings.

## 5. Deterministic retrieval query

Course entity plus detected field/concepts is preferred. Without a course entity, at least two independently detected concepts are required. Protected numbers, acronyms, dates, percentages and emails are retained. Example synthetic query: `ABC 123 prerequisite`. Uncertain cases abstain from deterministic construction.

## 6. Qwen retrieval rewrite

Only when deterministic construction abstains, the existing cached Qwen2.5-7B is asked for a concise English **search query**, not an answer. In the 14-case subset, 11 deterministic and three Qwen rewrites were used. No additional neural model was loaded.

## 7. Rewrite validation

The rewritten query must be concise English, preserve protected course codes, entities, numbers, dates, percentages, emails and official acronyms, avoid adding protected facts, retain a compatible requested field, and overlap detected academic concepts. Rejections do not enter retrieval. The 128-entry, version-keyed in-memory cache stores only validated queries; no evaluation mappings are persisted. Residual risk: a semantically subtle shift in a general, entity-free Qwen rewrite is not fully detectable by lexical checks.

## 8. Multi-query fusion

Merge on `chunk_id`; retain per-query original/canonical ranks and dense/sparse/metadata provenance. Equal-weight formula: `score = 1/(60 + original_rank) + 1/(60 + canonical_rank)`, omitting a missing term. Stable tie-break favors original rank, then canonical rank, then chunk ID. The same Step-1 evidence gate is run again; canonical validation is accepted only for a validated rewrite with lexical anchors present in its supporting excerpt. The course–field relation gate was not weakened.

## 9. Document-language handling

Retrieved chunks receive a conservative `english_dominant`, `bangla_dominant`, `mixed`, or `unknown` script label during fusion. This changes no chunk semantics or index data. `CROSSLINGUAL_RETRIEVAL_TARGET` allows the auxiliary English rewrite to be disabled for a different configured corpus target. English is an auxiliary representation for this corpus, not a claim that future documents are English.

## 10. Dataset leakage protection

Production cross-lingual code never reads `english_question`, paired translations, references, ground truth, expected page, or question-ID maps. Those fields are used only by `scripts/evaluate_step7f.py` for offline comparison. Answer-bank mode remains off in the subset.

## 11–12. Failure-subset results and recovery rates

| Language | Eligible | Deterministic evidence recoveries | Qwen evidence recoveries | Total evidence recoveries | Answers returned, automatic | Remaining insufficient | Generation rejected after recovery | Evidence recovery | End-to-end automatic recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bangla | 11 | 3 | 3 | 6 | 1 | 5 | 5 | 54.5% | 9.1% |
| Banglish | 3 | 0 | 0 | 0 | 0 | 3 | 0 | 0% | 0% |
| Combined | 14 | 3 | 3 | 6 | 1 | 8 | 5 | 42.9% | 7.1% |

One initial answer for a multi-percentage distribution was incomplete. A narrow post-recovery completeness guard now changes it to `GENERATION_REJECTED`; it is not counted as an answer. The remaining returned Bangla answer is **not yet human-accepted**.

## 13. Language parity before vs after subset

`ALL_THREE_SUPPORTED`: 83 → 89. `ENGLISH_AND_BANGLISH`: 9 → 3. `ENGLISH_ONLY`: 1 → 1. Other mismatch: three cases after; none supported: four unchanged. These are evidence labels, not answer-quality labels.

## 14. Full 300-query results

Not run. Gate decision: evidence recovery was meaningful in Bangla, but zero Banglish recovery, only one unreviewed returned answer, and five new generation rejections after evidence recovery do not justify a costly full answer run yet. A full-run status table would be a projection, not a measured result, and is intentionally omitted.

## 15. Retrieval metrics before vs after

Offline 300-query retrieval-only pass. Percentages below; MRR@3 is shown on a 0–100 scale. Entity/field denominators exclude questions without a detected entity/field. `Supportable@k` uses the production-accepted validated query representation after fallback.

| Language | Page@1 | Page@3 | MRR@3 | Entity@1 | Entity@3 | Field@1 | Field@3 | Supportable@1 | Supportable@3 |
|---|---|---|---|---|---|---|---|---|---|
| English | 67→67 | 99→99 | 82.8→82.8 | 97.1→97.1 | 100→100 | 93.5→93.5 | 97.8→97.8 | 89→89 | 93→93 |
| Bangla | 51→54 | 94→97 | 71.5→74.5 | 97.1→97.1 | 98.6→98.6 | 94.4→94.4 | 97.8→97.8 | 84→89 | 85→91 |
| Banglish | 53→53 | 95→95 | 73.5→73.5 | 97.1→97.1 | 100→100 | 94.4→94.4 | 100→100 | 91→91 | 95→95 |

Overall Page@1: 57.0→58.0%; Page@3: 96.0→97.0%; Supportable@3: 91.0→93.0%. Page labels use the single expected page in the development dataset and can undercount a semantically equivalent repeated passage on another page. Field-hit is lexical/metadata, not proof of a relation.

## 16–18. Status counts before vs after

Measured subset transitions: six Bangla `INSUFFICIENT_EVIDENCE` cases changed to five `GENERATION_REJECTED` and one `ANSWER_RETURNED`; five Bangla and three Banglish remained insufficient. The untouched English and other multilingual cases were **not** rerun through answer generation, so no full-300 after totals are claimed. Step 7E baseline was 224 answers, 49 generation rejected, 19 insufficient, five ambiguous, three conflicting.

## 19. Qwen rewrite usage

All 300 Step 7E baseline variants used normal retrieval only. Under Step 7F, 286 unchanged-status variants would remain on that path; the 14 eligible subset cases used fallback. Of those 14, 11 used deterministic and three Qwen rewrites. The 286-path figure is a status-based projection, not a full Step 7F answer run.

## 20. Latency

Measured on this 14-case subset; median / nearest-rank P95. Step 7E normal retrieval over 300: 0.205 / 1.459 seconds. Deterministic cases (n=11): rewrite 0.004 / 0.013 s; second retrieval 0.227 / 0.429 s; fusion plus reassessment 0.008 / 0.019 s; total fallback overhead 0.233 / 0.451 s. Qwen cases (n=3): rewrite 16.604 / 20.083 s; second retrieval 3.349 / 3.630 s; fusion plus reassessment 0.023 / 0.041 s; total fallback overhead 19.981 / 23.756 s. Small-n P95 is the maximum; timings include Windows paging variability. The full answer path can be much slower than retrieval (the longest subset answer took about 171 s).

## 21. Memory

Observed subset process peak working set was 5.242 GiB; Step 7E's reported maximum RSS was 5.475 GiB. No clear RSS increase is shown, but these are different runs and not a controlled comparison. System-wide pagefile use during the subset reached 3.218 GiB versus 1.944 GiB reported in Step 7E; this is not attributable solely to Step 7F because it includes other processes and machine state. The existing Qwen cache is reused; no second model is loaded.

## 22. Multilingual human-review file

`results/step7f_multilingual_review.csv` contains all six recovered-evidence cases (all Bangla, none Banglish), with correctness, relevance, groundedness, naturalness, semantic consistency, acceptability and notes left blank for human ratings.

## 23. Tests

Previous: 173. New: 11. Total: 184; failures: 0; skipped: 1. The suite covers synthetic `ABC 123` fallback, protected entities/numbers/fields, English no-rewrite, generation-rejected no-retry, cache and fusion behavior. Command: `.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test*.py"`.

## 24–26. Compatibility and scale independence

Steps 1–7E behavior remains the first path, including original multilingual BGE-M3 retrieval, evidence checks and language realization. No current PDF name, course-answer pair, question ID or expected page is in production fallback logic. The second pass uses the same index and Step-5 stack, so it is structurally independent of the future 70-PDF / 6,000-question evaluation set. Generalizing to mixed-language documents still needs corpus-level target-language selection and real multi-PDF evaluation; the present script label is informational only.

## 27. Git diff summary

Changed tracked code: generator, pipeline, shared query object. New: cross-lingual module, diagnostic script, tests and six CSV artifacts plus this report. `requirements.txt` was already modified before Step 7F and was not edited. No commit made.

## 28. Remaining risks

Banglish's three initial retrieval failures are not recovered. The HSS 111(B) failures are about table/chunk ownership, not a simple query translation; relaxing the gate would be unsafe. Several Bangla cases now reach evidence but fail canonical generation or realization. An entity-free Qwen rewrite can still shift nuance despite protected-fact checks. One returned answer needs human acceptance. Pagefile pressure and very slow 7B generation remain important on the 8 GB machine.

## 29. Human input needed

Yes: manually review the six rows in `step7f_multilingual_review.csv`, especially the one returned Q019 answer and the supporting evidence of the five rejected cases. Do not infer that an automatic `ANSWER_RETURNED` equals a thesis-quality answer.

## 30. Ready for Step 8?

**No.** Step 7F improved Bangla evidence parity but not Banglish and has only one unreviewed end-to-end recovery. The success gate intentionally prevented the full 300-answer run.

## 31. Next step (not implemented)

Review the six recovered cases, then decide whether to work on course-table ownership/chunk relation and the Step-7E Bangla realization failures before re-running the full 300-answer development evaluation. Do not weaken evidence validation to raise recovery counts.
