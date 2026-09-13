# Step 6 Multilingual Normalization Development Report

**DEVELOPMENT / REGRESSION EVALUATION — NOT FINAL THESIS ACCURACY**

Selected dense representation: `original`

| Language | Stage | Page H@1 | Page H@3 | MRR@3 | Entity H@1/H@3 | Field H@1/H@3 | Supportable@1/@3 |
|---|---|---:|---:|---:|---:|---:|---:|
| English | Step 5 | 67.00% | 99.00% | 0.8283 | 97.14%/100.00% | 88.04%/93.48% | 89.00%/94.00% |
| English | Normalized | 67.00% | 99.00% | 0.8283 | 97.14%/100.00% | 88.04%/93.48% | 89.00%/94.00% |
| Bangla | Step 5 | 51.00% | 94.00% | 0.7150 | 97.10%/98.55% | 91.11%/93.33% | 84.00%/86.00% |
| Bangla | Normalized | 51.00% | 94.00% | 0.7150 | 97.10%/98.55% | 91.11%/93.33% | 84.00%/86.00% |
| Banglish | Step 5 | 53.00% | 95.00% | 0.7350 | 97.10%/100.00% | 91.11%/95.56% | 91.00%/96.00% |
| Banglish | Normalized | 53.00% | 95.00% | 0.7350 | 97.10%/100.00% | 91.11%/95.56% | 91.00%/96.00% |
| Overall | Step 5 | 57.00% | 96.00% | 0.7594 | 97.12%/99.52% | 90.07%/94.12% | 88.00%/92.00% |
| Overall | Normalized | 57.00% | 96.00% | 0.7594 | 97.12%/99.52% | 90.07%/94.12% | 88.00%/92.00% |

## Dense representation experiments

- `original`: supportable@1/@3=88.00%/92.00%; field H@1/H@3=90.07%/94.12%; page H@1/H@3=57.00%/96.00%; MRR@3=0.7594.
- `normalized`: supportable@1/@3=88.00%/91.67%; field H@1/H@3=89.71%/93.75%; page H@1/H@3=60.33%/95.67%; MRR@3=0.7733.
- `original_plus_normalized`: supportable@1/@3=88.33%/92.00%; field H@1/H@3=90.07%/94.12%; page H@1/H@3=59.67%/94.33%; MRR@3=0.7661.

## Language detection

- English: 100/100 (100.00%)
- Bangla: 100/100 (100.00%)
- Banglish: 98/100 (98.00%)

## Failure analysis

- QUERY_NORMALIZATION_ERROR: 0
- LANGUAGE_DETECTION_ERROR: 2
- DENSE_CANDIDATE_MISS: 3
- SPARSE_CANDIDATE_MISS: 2
- FUSION_ERROR: 5
- RERANKER_ERROR: 0
- ENTITY_RESOLUTION_ERROR: 0
- FIELD_RESOLUTION_ERROR: 0
- GROUND_TRUTH_LABEL_AMBIGUITY: 0

## Reranker status

- Model: `BAAI/bge-reranker-v2-m3`
- Proposed pinned revision: `b5160aeac3c6c8fe7beaaaf04c9e0142826b58d1`
- Available locally: False
- Download required: True
- Repository storage: approximately 2.29 GB (2.27 GB weights plus tokenizer/config files)
- Estimated CPU RAM: approximately 3-5 GB for weights and inference overhead; longer batches may require more
- CPU suitability: CPU inference is supported, but top-15 latency must be benchmarked and may be several seconds per query
- License: Apache-2.0 as declared by the model repository
- Result: NOT RUN - USER PERMISSION REQUIRED BEFORE DOWNLOADING A RERANKER
