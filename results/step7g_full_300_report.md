# Step 7G full 300-query development evaluation

2026-09-25. Development measurement only; **not final thesis accuracy**. One-PDF, 491-chunk fresh index. Answer bank and neural reranker were off; Qwen2.5-7B was called only by the existing pipeline. Step 8 was not started and no commit was made.

## 1. Completion

**300/300** variants completed, 100 per language. Checkpoint fingerprint: `de9e71eeb05b4056d322`. Runtime exceptions: 0.

## 2. Answers and statuses by language

| Language | Returned | Insufficient | Generation rejected | Ambiguous | Conflicting | Runtime errors |
|---|---|---|---|---|---|---|
| english | 87 | 3 | 8 | 1 | 1 | 0 |
| bangla | 69 | 3 | 24 | 3 | 1 | 0 |
| banglish | 67 | 0 | 31 | 1 | 1 | 0 |

## 3. Answer strategy distribution

| Strategy | english | bangla | banglish | Overall |
|---|---|---|---|---|
| structured_exact | 22 | 5 | 3 | 30 |
| structured_list | 0 | 0 | 0 | 0 |
| semi_structured_relation | 62 | 61 | 59 | 182 |
| gguf_generation | 3 | 3 | 5 | 11 |
| generation_rejected | 8 | 24 | 31 | 63 |
| unsupported | 3 | 3 | 0 | 6 |
| ambiguous | 1 | 3 | 1 | 5 |
| conflicting | 1 | 1 | 1 | 3 |

`semi_structured_relation` is deterministic. `gguf_generation` is an accepted generated answer; rejected generation is counted under `generation_rejected`. `fast_extractive` may appear only if selected by the current pipeline.

## 4. Retrieval metrics

| Language | Page H@1 | Page H@3 | MRR@3 | Entity H@1/H@3 | Field H@1/H@3 | Supportable@1/@3 |
|---|---|---|---|---|---|---|
| english | 67.0% | 99.0% | 0.828 | 100.0%/100.0% | 93.5%/97.8% | 91.0%/94.0% |
| bangla | 53.0% | 97.0% | 0.740 | 100.0%/100.0% | 94.4%/97.8% | 85.0%/88.0% |
| banglish | 53.0% | 95.0% | 0.735 | 100.0%/100.0% | 94.4%/100.0% | 93.0%/97.0% |
| overall | 57.7% | 97.0% | 0.768 | 100.0%/100.0% | 94.1%/98.5% | 89.7%/93.0% |

Entity and field rates use applicable-question denominators. Page labels and language pairings are offline evaluation inputs. These are top-1/top-3 metrics on the final candidate order; a verified metadata hit below rank 3 can still support an answer.

### Frozen baseline comparison

| Language | Step 5 Page H@3 | Step 7F Page H@3 | Step 7G Page H@3 | Step 5 Supportable@3 | Step 7F Supportable@3 | Step 7G Supportable@3 |
|---|---|---|---|---|---|---|
| english | 99.0% | 99.0% | 99.0% | 94.0% | 93.0% | 94.0% |
| bangla | 94.0% | 97.0% | 97.0% | 86.0% | 91.0% | 88.0% |
| banglish | 95.0% | 95.0% | 95.0% | 96.0% | 95.0% | 97.0% |
| overall | 96.0% | 97.0% | 97.0% | 92.0% | 93.0% | 93.0% |

Comparison is descriptive, not a controlled same-index delta: Step 5/7F used 490 chunks; Step 7G uses 491 and a changed evidence gate. Step 7F's post-fallback metrics used its accepted-rewrite gate; Step 7G metrics use current final candidate order and current evidence assessment. No retrieval weights were changed for this run.

## 5. Multilingual support parity

| Category | Base questions |
|---|---|
| ALL_THREE_SUPPORTED | 91 |
| ENGLISH_ONLY | 0 |
| ENGLISH_AND_BANGLA | 0 |
| ENGLISH_AND_BANGLISH | 4 |
| OTHER_MISMATCH | 3 |
| NONE_SUPPORTED | 2 |

Total paired base questions: 100. Pairing is evaluation-only; it never enters runtime answering.

## 6. Generation and validation

| Language | Language passed / 100 | Grounding passed / returned | Generated rows | Attempts | Accepted | Rejected | Retries | Successful | Failed |
|---|---|---|---|---|---|---|---|---|---|
| english | 100/100 | 72/87 | 5 | 5 | 3 | 2 | 0 | 0 | 0 |
| bangla | 100/100 | 69/69 | 5 | 7 | 3 | 2 | 1 | 0 | 1 |
| banglish | 100/100 | 67/67 | 8 | 11 | 5 | 3 | 1 | 0 | 1 |

Grounding pass rate is reported among returned answers; rejected/insufficient responses are not treated as grounded factual answers. Returned answers failing language validation: 0; failing grounding validation: 15.

## 7. Latency

Wall-clock per question: mean 3.56s, median 0.24s, P95 29.90s, maximum 195.20s; summed question time 1069.4s. These include model warm-up and any generation retries; they are not throughput measurements for a 70-PDF corpus.

## 8. Memory and pagefile

Observed peak process RSS 5062.8 MiB; peak process working set 5510.6 MiB; minimum available system RAM 16.4 MiB; maximum system pagefile usage 4761.0 MiB. The pagefile measure is system-wide, not attributable solely to this process. The very low minimum available RAM indicates substantial memory pressure during this one-PDF run; capacity for a 70-PDF workload is not established.

## 9. Errors

Runtime exceptions: 0. Rows recording a generation error: 0. Error details remain in the results CSV; safe rejection is counted separately from an exception.

## 10. Test suite

After the template-only edits: 196 tests run, 195 passed, one skipped, zero failed. The fresh 491-chunk index was reused without rebuild.

## 11. Manual review sample

`results/step7g_full_manual_review.csv` contains 30 rows: 10 English, 10 Bangla and 10 Banglish. It prioritizes relation answers, GGUF outputs, prior failures and cross-lingual recoveries. All eight requested human fields are blank; no human score was invented.

## 12. Remaining failure categories

| Category | Rows |
|---|---|
| AMBIGUOUS | 5 |
| CONFLICTING | 3 |
| GENERATION_REJECTED | 63 |
| INSUFFICIENT_EVIDENCE | 6 |

Generation-rejection reasons (diagnostic labels): MISSING_REQUIRED_ENTITY=31, SEMANTIC_CONTRACT_UNCERTAIN=15, NO_SAFE_ANSWER_FROM_SUPPORTED_EVIDENCE=13, WRONG_RELATION_VALUE=2, UNSUPPORTED_REALIZATION_FACTS=1, UNSUPPORTED_FACTUAL_TOKENS=1.

Safety issue outside the rejection counts: 15 returned English `structured_exact` answers failed automatic grounding: MISSING_REQUIRED_ENTITY=13, SEMANTIC_CONTRACT_UNCERTAIN=1, UNSUPPORTED_FACTUAL_TOKENS=1. Their question IDs and excerpts are retained in the results CSV; no benchmark-time fix was made.

The six previously identified Banglish safe rejections were allowed to remain rejections. No relation type or retrieval/model setting was changed in response to this run.

## 13. Is Step 7 engineering-complete?

**No:** 15 English `structured_exact` answers were returned despite failed automatic grounding. There were no runtime exceptions or returned-answer language failures, but the grounding safety gap prevents an engineering-complete declaration. Inspect the recorded rows before declaring completion.

## 14. Should Step 8 begin?

**Not automatically.** Review the 30 sampled answers and decide whether the observed failures and any validation issues are acceptable for the thesis workflow. Step 8 was not implemented here.
