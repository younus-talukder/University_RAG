# Benchmark run step10-full-generation-pilot-20260929T192719893950Z-367c33

Run classification: **PILOT**

Run mode: **full**

Completion: 300/300; run_complete=true

This run measures retrieval, safe answer coverage, abstentions, latency, and automatic answer proxies. It does not establish human correctness or final thesis accuracy.

Safe answer coverage: 0.867 (Wilson 95% CI 0.824–0.901; n=300)

Abstention rate: 0.13333333333333333

Known unsafe returned answers: 0

## By language

| Language | Rows | Safe answer coverage | Hit@1 | Hit@3 | MRR@3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bangla | 100 | 0.840 | 0.530 | 0.970 | 0.740 |
| banglish | 100 | 0.880 | 0.530 | 0.950 | 0.735 |
| english | 100 | 0.880 | 0.670 | 0.990 | 0.828 |

## Retrieval

- hit_1: 0.5766666666666667
- hit_3: 0.97
- mrr_3: 0.7677777777777778
- source_accuracy: 1.0
- page_accuracy: 0.5766666666666667
- entity_hit_1: 1.0
- entity_hit_3: 1.0
- field_hit_1: 0.9377289377289377
- field_hit_3: 0.9853479853479854
- supportable_1: 0.61
- supportable_3: 0.8933333333333333

## Status distribution

- GENERATION_REJECTED: 25
- INSUFFICIENT_EVIDENCE: 7
- SUPPORTED: 260
- RETRIEVAL_UNCERTAIN: 2
- CONFLICTING_EVIDENCE: 6

## Multilingual parity

- NONE_SUPPORTED: 10
- ENGLISH_AND_BANGLISH: 3
- ALL_THREE_SUPPORTED: 83
- BANGLA_AND_BANGLISH: 1
- ENGLISH_ONLY: 2
- BANGLISH_ONLY: 1

## Latency and generation

Mean / median / P95 total latency: 4.137 / 0.214 / 30.643 seconds.

Generation attempts: 31; retries: 5.

## Post-inference error categories

- UNKNOWN: 281
- RETRIEVAL_MISS: 2
- GROUNDING_REJECTION: 9
- RANKING_MISS: 2
- SOURCE_CONFLICT: 6

## Reproducibility

Dataset SHA-256: 8b2e6c06ffba8d4dee58fd979706f244e5891591c4e8e25d9beffea801e149b2

Corpus fingerprint: 8d25c720865665759d79c3f333f9ba549fc2408e9d8d37c6206fbf2099d83c8e

Git commit: 6c23ad85b2d9f5c0b18a898e72e568106a383989; working_tree_dirty=true

Human sample rows have blank rating fields. No human scores are inferred from automatic metrics.
