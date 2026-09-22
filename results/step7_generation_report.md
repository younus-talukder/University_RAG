# Step 7 — Grounded Multilingual Answer Generation Report

**Development/regression evaluation only — not final thesis accuracy.**

## Scope and compatibility

Step 7 adds a controlled answering layer after the approved Step 6 retrieval and evidence-validation pipeline. It does not change BGE-M3, FAISS, BM25, metadata retrieval, RRF, candidate counts, query normalization, chunking, evidence semantics, or the optional reranker. The curated answer bank and reranker were both disabled for this evaluation.

The evaluated corpus remains the single active curriculum PDF (92 evidence-bearing pages and 490 structured chunks). The canonical evaluation remains 100 base questions with English, Bangla, and Banglish variants: 300 queries total.

## Problems found before Step 7

- Supported exact facts were unnecessarily sent to the language model.
- Bangla and Banglish could be produced by attaching a translated prefix to an English answer.
- Language validation could accept a Bangla prefix followed by an English body.
- Context length was estimated from word count instead of the model tokenizer.
- Retrieved passages were not explicitly isolated as untrusted document data.
- A failed language retry could still be returned to the user.

## Implemented architecture

The runtime now selects one of five safe routes after Step 1 evidence assessment:

1. `structured_exact` for credits, prerequisites, percentages, email, dates, and other reliably extractable facts.
2. `structured_list` for reliably extractable lists such as mark distributions.
3. `gguf_generation` only for supported explanatory or summarization questions.
4. Safe unsupported or clarification responses when evidence is absent or ambiguous.
5. A safe conflict response when verified evidence conflicts.

Exact questions never ask the model to guess. If a reliable exact value cannot be extracted, the system abstains. Unsupported, ambiguous, and conflicting evidence never invokes Qwen.

The generation context is a ranked, deduplicated package of verified evidence containing source, page, heading, entity, excerpt, chunk identifier, and score. Complete high-ranked evidence blocks are retained within a tokenizer-measured budget; excerpts are not cut mid-block. The prompt separates the user question from numbered document evidence, labels evidence as data only, and instructs the model to ignore commands embedded in it.

Post-generation checks now validate the requested language and evidence-grounded factual tokens. Unsupported numbers, course codes, emails, acronyms, and salient names cause rejection. One controlled retry is permitted only for a language failure when the facts remain grounded. A retry must pass both checks; otherwise the system returns a safe response. Citations always come from verified evidence objects, never model text.

## Generator configuration

- Model: Qwen2.5-1.5B-Instruct GGUF, existing local `Q4_K_M` file
- Context window: 8192 tokens
- Temperature: 0.0
- Top-p: 1.0
- Maximum new tokens: 180
- Stop sequences: none explicitly configured
- Runtime: CPU, using the machine's available logical thread count
- Loading: cached once per model/context/thread configuration

No model was replaced or downloaded.

## Test results

The existing suite contained 110 tests. Step 7 adds 20 tests, including a separate 30-case synthetic multilingual regression set (10 per language). The final result is **130 tests run, 0 failures, 1 skipped opt-in real-reranker test**.

Coverage includes structured routing, natural language templates, body-level language validation, ranked evidence packaging and deduplication, tokenizer budgeting, prompt-injection isolation, factual grounding, unsupported names and numbers, exact-answer model bypass, citation provenance, unsupported/conflicting no-generation behavior, one-retry enforcement, failed-retry abstention, and model caching.

## Full 300-query evaluation

### Strategy distribution

| Strategy | Count |
|---|---:|
| Structured exact | 205 |
| Structured list | 2 |
| Accepted GGUF generation | 15 |
| Unsupported | 70 |
| Ambiguous | 5 |
| Conflicting | 3 |

Qwen was invoked 31 times. Fifteen final generations were accepted and 16 were rejected. There were 15 controlled retries: 4 succeeded and 11 failed safely.

### Language consistency

| Expected language | Passed | Rate |
|---|---:|---:|
| English | 100/100 | 100.00% |
| Bangla | 100/100 | 100.00% |
| Banglish | 98/100 | 98.00% |
| Overall | 298/300 | 99.33% |

The two Banglish failures were pure-English bodies. This reflects two known Step 6 language-detection outcomes; Step 7 recorded them rather than changing the approved detector.

### Grounding and factual preservation

The validator rejected 16 generated attempts that did not meet the grounded-answer contract. This is a safety result, not a claim that all accepted prose is semantically good.

The automatic fact-preservation diagnostic was applicable to 276 rows and preserved all extracted reference fact tokens in 183 rows (**66.30%**). By final strategy: structured exact 178/202 (88.12%), GGUF generation 5/10 (50.00%), while safe abstentions and conflict responses intentionally do not reproduce reference facts. This diagnostic therefore penalizes many correct safety abstentions.

Automatic token-overlap diagnostics were precision 0.3636, recall 0.2460, and F1 0.2856. These are development proxies only; multilingual paraphrases, translations, and safe abstentions make them unsuitable as final correctness scores.

### Latency

| Route | Median | P95 |
|---|---:|---:|
| Structured answering | 0.00024 s | 0.00050 s |
| GGUF generation | 6.652 s | 22.104 s |
| End-to-end | 0.200 s | 6.853 s |

Retries consumed 89.215 seconds in total across the evaluation.

## Real-model smoke test and capacity decision

Five manually inspected examples per language were run through the final local model path.

- English: **acceptable**. Supported examples were generally factual, though sometimes verbose.
- Bangla: **poor**. Accepted outputs included repetitive or semantically weak prose.
- Banglish: **poor**. Some attempts were pure English or were rejected by the safeguards.

**Decision: the current 1.5B model is not sufficient for the required equal-quality English, Bangla, and Banglish target.** The structured layer is strong for exact facts, but the model-generated Bangla and Banglish prose is not thesis-ready. Script/factual-token validators cannot reliably identify fluent-looking nonsense or poor semantic translation, so manual review remains mandatory.

Potential next-model experiments, requiring explicit user permission, are [Qwen2.5-3B-Instruct `Q4_K_M`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/tree/main) (official file about 2.1 GB) and [Qwen2.5-7B-Instruct `Q4_K_M`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/tree/main) (official split files total about 4.68 GB). Expected RAM and latency impacts must be measured on this machine; rough planning estimates are 3–5 GB RAM and 1.5–2.5× current generation latency for 3B, or 6–9 GB RAM and 3–5× current latency for 7B. No upgrade has been downloaded.

## Remaining risks

- The current model's Bangla and Banglish explanatory generation is below the target quality.
- Language and grounding validators are conservative heuristics, not semantic judges.
- Retrieval can still yield absent, ambiguous, conflicting, or irrelevant evidence; Step 7 safely routes these cases but does not tune retrieval.
- Automatic overlap and fact-token diagnostics are not substitutes for blinded human evaluation.
- The current corpus contains one active PDF; scaling and final thesis evaluation are later steps.

## Artifacts

- `results/step7_generation_results.csv`: all 300 row-level results and timings
- `results/step7_generation_summary.json`: machine-readable aggregate metrics and settings
- `results/step7_manual_review_sample.csv`: score-free 30-row sample, 10 per language
- `results/step7_real_model_smoke.json`: final 15-query real-model smoke output
- `tests/data/step7_generation_regression.csv`: separate 30-case synthetic regression set

Step 8 has not been started. No commit was created.
