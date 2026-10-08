# Benchmark run step9-development-framework-validated-20260929T184727419616Z-f7d9f3

Run classification: **DEVELOPMENT**

Run mode: **skip_generation**

Completion: 300/300; run_complete=true

This run measures retrieval, safe answer coverage, abstentions, latency, and automatic answer proxies. It does not establish human correctness or final thesis accuracy.

Safe answer coverage: 0.79 (n=300)

Abstention rate: 0.21

Known unsafe returned answers: 0

## Retrieval

- hit_1: 0.41333333333333333
- hit_3: 0.8966666666666666
- mrr_3: 0.6338888888888888
- source_accuracy: 0.97
- page_accuracy: 0.41333333333333333
- entity_hit_1: 1.0
- entity_hit_3: 1.0
- field_hit_1: 0.9304029304029304
- field_hit_3: 0.9597069597069597
- supportable_1: 0.6933333333333334
- supportable_3: 0.8666666666666667

## Status distribution

- GENERATION_REJECTED: 35
- INSUFFICIENT_EVIDENCE: 19
- SUPPORTED: 237
- CONFLICTING_EVIDENCE: 8
- RETRIEVAL_UNCERTAIN: 1

## Multilingual parity

- NONE_SUPPORTED: 16
- ENGLISH_ONLY: 3
- ENGLISH_AND_BANGLISH: 9
- ALL_THREE_SUPPORTED: 72

## Reproducibility

Dataset SHA-256: 8b2e6c06ffba8d4dee58fd979706f244e5891591c4e8e25d9beffea801e149b2

Corpus fingerprint: 8d25c720865665759d79c3f333f9ba549fc2408e9d8d37c6206fbf2099d83c8e

Git commit: 6c23ad85b2d9f5c0b18a898e72e568106a383989; working_tree_dirty=true

Human sample rows have blank rating fields. No human scores are inferred from automatic metrics.
