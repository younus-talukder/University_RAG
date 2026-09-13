# Step 5 Hybrid Retrieval Development Report

**DEVELOPMENT / REGRESSION EVALUATION — NOT FINAL THESIS ACCURACY**

Selected configuration: `A_equal`

| Language | Mode | Page H@1 | Page H@3 | MRR@3 | Entity H@1/H@3 | Field H@1/H@3 | Supportable@1/@3 |
|---|---|---:|---:|---:|---:|---:|---:|
| English | Dense | 23.00% | 80.00% | 0.4733 | 97.14%/100.00% | 75.00%/92.39% | 77.00%/93.00% |
| English | Hybrid | 67.00% | 99.00% | 0.8283 | 97.14%/100.00% | 88.04%/93.48% | 89.00%/94.00% |
| Bangla | Dense | 23.00% | 64.00% | 0.4117 | 95.65%/98.55% | 77.78%/91.11% | 72.00%/84.00% |
| Bangla | Hybrid | 51.00% | 94.00% | 0.7150 | 97.10%/98.55% | 91.11%/93.33% | 84.00%/86.00% |
| Banglish | Dense | 21.00% | 59.00% | 0.3833 | 97.10%/100.00% | 77.78%/93.33% | 79.00%/94.00% |
| Banglish | Hybrid | 53.00% | 95.00% | 0.7350 | 97.10%/100.00% | 91.11%/95.56% | 91.00%/96.00% |
| Overall | Dense | 22.33% | 67.67% | 0.4228 | 96.63%/99.52% | 76.84%/92.28% | 76.00%/90.33% |
| Overall | Hybrid | 57.00% | 96.00% | 0.7594 | 97.12%/99.52% | 90.07%/94.12% | 88.00%/92.00% |

## Dense to hybrid delta

| Language | Page H@1 | Page H@3 | MRR@3 | Entity H@3 | Field H@3 | Supportable@3 |
|---|---:|---:|---:|---:|---:|---:|
| English | 44.00% | 19.00% | +0.3550 | 0.00% | 1.09% | 1.00% |
| Bangla | 28.00% | 30.00% | +0.3033 | 0.00% | 2.22% | 2.00% |
| Banglish | 32.00% | 36.00% | +0.3517 | 0.00% | 2.22% | 2.00% |
| Overall | 34.67% | 28.33% | +0.3367 | 0.00% | 1.84% | 1.67% |

## RRF configurations

- `A_equal`: dense=1.0, sparse=1.0, metadata=0.8, k=60; page H@1/H@3=57.00%/96.00%, entity H@3=99.52%, field H@3=94.12%.
- `B_dense_1_2`: dense=1.2, sparse=1.0, metadata=0.8, k=60; page H@1/H@3=56.33%/95.00%, entity H@3=99.52%, field H@3=93.75%.
- `C_sparse_0_8`: dense=1.0, sparse=0.8, metadata=0.8, k=60; page H@1/H@3=56.33%/95.67%, entity H@3=99.52%, field H@3=93.75%.

## Failure analysis

- DENSE_MISSED: 3
- SPARSE_MISSED: 3
- BOTH_MISSED: 1
- FUSION_RANK_ERROR: 5
- ENTITY_METADATA_MISSED: 0
- FIELD_METADATA_MISSED: 0
- LABEL_AMBIGUITY: 0

## Retrieval latency

- Dense baseline search/post-processing (embedding excluded): median 0.001713s; P95 0.002843s.
- Hybrid search/fusion (embedding excluded): median 0.002228s; P95 0.003244s.
- Shared batched BGE-M3 embedding: 22.644s for 300 queries (0.075480s/query average).
- Channel medians: FAISS 0.000271s; BM25 + metadata 0.000546s; RRF 0.001384s.

## Artifact and sparse-memory cost

- FAISS: 1.91 MiB
- Metadata: 0.39 MiB
- Sparse index: 0.20 MiB
- Approximate loaded sparse Python object graph: 2.46 MiB
- Sparse load + validation: 0.180710s
- Sparse corpus: 490 chunks; 5563 terms.
