# Step 7C Qwen 7B Multilingual Development Report

**DEVELOPMENT / REGRESSION EVALUATION — NOT FINAL THESIS ACCURACY**

Queries: 300; answer bank: off; reranker: off.

## Answer strategies

{"unsupported": 68, "ambiguous": 5, "gguf_generation": 17, "structured_exact": 205, "structured_list": 2, "conflicting": 3}

## Language consistency

- English: 100/100 (100.00%)
- Bangla: 100/100 (100.00%)
- Banglish: 98/100 (98.00%)
- Overall: 298/300 (99.33%)

## Grounding

{"generated_answers": 31, "accepted_generation": 17, "rejected_generation": 14, "retries": 9, "successful_retries": 2, "failed_retries": 7}

## Fact-preservation diagnostic

{"applicable": 276, "preserved": 184, "rate": 0.6666666666666666}

## Automatic diagnostic proxies (not human correctness)

{"precision": 0.3734821032768401, "recall": 0.25207556283872073, "f1": 0.29387167616081183}

## Latency seconds

{"structured_median": 0.00023570000030304072, "structured_p95": 0.0008601999998063548, "gguf_median": 29.103425100000095, "gguf_p95": 81.75733160000073, "retry_total": 232.99255320000202, "total_median": 0.2252859499999431, "total_p95": 31.333402000000206}

## Prompt tokens

{"median": 520, "p95": 663, "maximum": 680}

## Memory

{"peak_rss_bytes": 5767397376, "minimum_available_ram_bytes": 47996928, "peak_swap_used_bytes": 3938754560}
