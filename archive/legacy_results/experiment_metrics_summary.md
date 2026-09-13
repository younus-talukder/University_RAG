# Experiment Metrics Summary

Answer-quality values are automatic proxy metrics based on language-specific reference token overlap and retrieval grounding. Use them for screening; use manual adjudication for final thesis claims about correctness, relevance, groundedness, and completeness.

## Overall

| experiment | n | hit_at_1_pct | hit_at_3_pct | mrr | source_accuracy_pct | page_accuracy_pct | language_detection_accuracy_pct | response_language_consistency_pct | correctness_proxy_pct | relevance_proxy_pct | groundedness_proxy_pct | completeness_proxy_pct | avg_latency_seconds | median_latency_seconds | p95_latency_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fast | 180 | 20.56 | 61.67 | 0.371 | 85.00 | 20.56 | 100.00 | 93.89 | 52.78 | 45.00 | 61.67 | 40.00 | 0.004 | 0.004 | 0.007 |
| GGUF | 180 | 35.00 | 71.11 | 0.500 | 87.78 | 35.00 | 100.00 | 96.67 | 45.56 | 36.11 | 69.44 | 37.78 | 7.653 | 7.304 | 19.162 |

## Breakdown Tables

Full breakdown written to `results\experiment_metrics_breakdown.csv`.
