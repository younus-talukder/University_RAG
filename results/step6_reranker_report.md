# Step 6 Neural Reranker Development Report

**DEVELOPMENT / REGRESSION EVALUATION — NOT FINAL THESIS ACCURACY**

Pinned model: `BAAI/bge-reranker-v2-m3@b5160aeac3c6c8fe7beaaaf04c9e0142826b58d1` (offline/local-only)

| K | Language | Page H@1/H@3 | MRR@3 | Entity H@1/H@3 | Field H@1/H@3 | Supportable@1/@3 | Reranker median/P95 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | English | 29.00%/80.00% | 0.5067 | 95.71%/100.00% | 85.87%/93.48% | 87.00%/94.00% | 4.642s/6.797s |
| 10 | Bangla | 29.00%/75.00% | 0.4950 | 97.10%/100.00% | 88.89%/92.22% | 82.00%/85.00% | 4.642s/6.797s |
| 10 | Banglish | 28.00%/78.00% | 0.4917 | 97.10%/100.00% | 88.89%/95.56% | 90.00%/96.00% | 4.642s/6.797s |
| 10 | Overall | 28.67%/77.67% | 0.4978 | 96.63%/100.00% | 87.87%/93.75% | 86.33%/91.67% | 4.642s/6.797s |
| 15 | English | 27.00%/78.00% | 0.4900 | 95.71%/100.00% | 85.87%/93.48% | 87.00%/94.00% | 6.776s/9.796s |
| 15 | Bangla | 29.00%/75.00% | 0.4933 | 97.10%/100.00% | 88.89%/92.22% | 82.00%/85.00% | 6.776s/9.796s |
| 15 | Banglish | 28.00%/77.00% | 0.4883 | 97.10%/100.00% | 88.89%/95.56% | 90.00%/96.00% | 6.776s/9.796s |
| 15 | Overall | 28.00%/76.67% | 0.4906 | 96.63%/100.00% | 87.87%/93.75% | 86.33%/91.67% | 6.776s/9.796s |
| 20 | English | 27.00%/77.00% | 0.4867 | 95.71%/100.00% | 85.87%/93.48% | 87.00%/94.00% | 8.956s/12.656s |
| 20 | Bangla | 29.00%/75.00% | 0.4917 | 97.10%/100.00% | 88.89%/92.22% | 82.00%/85.00% | 8.956s/12.656s |
| 20 | Banglish | 27.00%/76.00% | 0.4800 | 97.10%/100.00% | 88.89%/95.56% | 90.00%/96.00% | 8.956s/12.656s |
| 20 | Overall | 27.67%/76.00% | 0.4861 | 96.63%/100.00% | 87.87%/93.75% | 86.33%/91.67% | 8.956s/12.656s |

Selected candidate K: **10**

RRF creates the bounded pool; raw neural logits only determine its final order. No raw-score addition is used.

## Decision

K=10 is the least harmful evaluated reranked configuration, but it is rejected for default runtime use. Compared with the frozen normalized hybrid baseline, it lowers overall Supportable@1 by 1.67 percentage points, Field Hit@1 by 2.21 points, Page Hit@1 by 28.33 points, Page Hit@3 by 18.33 points, and MRR@3 by 0.2617. Its median/P95 full retrieval latency is 4.719/6.875 seconds. `RERANKER_ENABLED` therefore remains `false`.

## Frozen baseline

| Page H@1/H@3 | MRR@3 | Entity H@1/H@3 | Field H@1/H@3 | Supportable@1/@3 |
|---:|---:|---:|---:|---:|
| 57.00%/96.00% | 0.7594 | 97.12%/99.52% | 90.07%/94.12% | 88.00%/92.00% |

The frozen Step-5 and Step-6 baseline JSON files were not overwritten.

## Candidate-pool oracle

These are candidate-availability diagnostics, not model accuracy.

| K | Correct page available | Supportable evidence available |
|---:|---:|---:|
| 10 | 98.67% | 92.67% |
| 15 | 99.00% | 93.00% |
| 20 | 99.33% | 93.00% |

At K=10, correct-page availability is 100.00% English, 98.00% Bangla, and 98.00% Banglish. Supportable availability is 95.00%, 86.00%, and 97.00%, respectively.

## K=10 win/loss analysis

| Language | Promoted correct | Demoted correct | Unchanged correct | Unchanged incorrect |
|---|---:|---:|---:|---:|
| English | 2 | 62 | 35 | 1 |
| Bangla | 11 | 53 | 33 | 3 |
| Banglish | 4 | 52 | 40 | 4 |
| Overall | 17 | 167 | 108 | 8 |

## K=10 remaining page-top-3 failures

| Language | First-stage miss | Reranker ordering | Entity | Field | Language normalization | Label ambiguity |
|---|---:|---:|---:|---:|---:|---:|
| English | 0 | 19 | 0 | 1 | 0 | 0 |
| Bangla | 2 | 22 | 0 | 1 | 0 | 0 |
| Banglish | 0 | 19 | 0 | 1 | 2 | 0 |
| Overall | 2 | 60 | 0 | 3 | 2 | 0 |

## Load, memory, and latency

An isolated offline CPU smoke process loaded the model in 8.642 seconds. Its measured process RSS increased by 742,019,072 bytes (about 708 MiB); this is a resident-memory observation and does not include all file-backed model pages. The safetensors weight file is 2,271,071,852 bytes and the complete snapshot is 2,293,259,735 bytes.

The baseline first-stage retrieval measured 0.077/0.078 seconds median/P95 when batched embedding time was amortized per query. Full retrieval median/P95 is 4.719/6.875 seconds for K=10, 6.853/9.873 seconds for K=15, and 9.033/12.734 seconds for K=20. GGUF generation is excluded.

## Scale and independence

Reranking cost is bounded by candidate K and does not directly scale with total corpus chunks. Growing to about 70 PDFs requires rebuilding normal retrieval artifacts but no reranker code changes; first-stage search may grow while cross-encoder work remains K-bounded. The measured CPU latency is nevertheless too high for default local-demo use.

Runtime reranking has no dependency on evaluation dataset size, question IDs, expected pages, or ground truth. A 6,000-question evaluation increases evaluation duration only; labels are used after retrieval solely for metric calculation.
