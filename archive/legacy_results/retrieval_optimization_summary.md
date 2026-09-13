# Retrieval Optimization Summary

Baseline experiment files were read only. Optimized retrieval outputs are saved separately.

## Overall

| experiment | n | hit_at_1_pct | hit_at_3_pct | mrr | source_accuracy_pct | page_accuracy_pct | avg_latency_seconds | median_latency_seconds | p95_latency_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline GGUF retrieval | 180 | 35.00 | 71.11 | 0.500 | 87.78 | 35.00 | 7.653 | 7.304 | 19.162 |
| Optimized retrieval | 180 | 47.22 | 72.22 | 0.579 | 88.33 | 47.22 | 0.197 | 0.187 | 0.237 |

## Delta

| metric | baseline | optimized | delta |
| --- | --- | --- | --- |
| hit_at_1_pct | 35.0 | 47.22 | 12.220 |
| hit_at_3_pct | 71.11 | 72.22 | 1.110 |
| mrr | 0.5 | 0.579 | 0.079 |
| source_accuracy_pct | 87.78 | 88.33 | 0.550 |
| page_accuracy_pct | 35.0 | 47.22 | 12.220 |

## Reranking Weights

```text
candidate_pool_min = 15
candidate_pool_multiplier = 5
course_match_boost = 0.8
course_text_boost = 0.2
course_mismatch_penalty = -1.0
intent_match_boost = 0.8
requested_field_boost = 3.2
topic_subject_boost = 0.8
topic_plan_boost = 2.2
topic_assessment_penalty = -1.8
lexical_overlap_weight = 0.12
lexical_overlap_cap = 0.9
section_heading_boost = 0.45
metadata_mismatch_penalty = -1.6
```

Full optimized per-query output: `results\retrieval_optimized_per_query.csv`
Full breakdown output: `results\retrieval_optimized_breakdown.csv`
Baseline failure analysis: `results\retrieval_baseline_failure_analysis.csv`
