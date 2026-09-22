# Step 7B — Controlled Qwen 1.5B vs 3B Comparison

**Development/regression experiment — not final thesis correctness.**

## 1. Readiness / preparation changes

Generator settings were centralized without changing the Step 7 defaults. The real lowercase 1.5B filename is now the portable default. Both models use the same generator implementation. Retrieval, prompts, language policy, validators, evidence construction, retry behavior, and structured-answer policy were not modified.

## 2. Downloaded model

- Repository: `Qwen/Qwen2.5-3B-Instruct-GGUF`
- Immutable revision: `7dabda4d13d513e3e842b20f0d435c732f172cbe`
- Filename: `qwen2.5-3b-instruct-q4_k_m.gguf`
- Quantization: `Q4_K_M`
- Size: 2,104,932,768 bytes
- SHA-256: `626B4A6678B86442240E33DF819E00132D3BA7DDDFE1CDC4FBB18E0A9615C62D`
- License: Qwen Research License Agreement; non-commercial research/evaluation use

Only this authorized model was downloaded. The original 1.5B file remains intact. No 7B model was downloaded.

## 3. Generator configuration refactor

Central configuration now exposes `GENERATOR_MODEL_PATH`, `GENERATOR_CONTEXT_SIZE`, `GENERATOR_THREADS`, `GENERATOR_BATCH_SIZE`, `GENERATOR_GPU_LAYERS`, `GENERATOR_TEMPERATURE`, `GENERATOR_TOP_P`, and `GENERATOR_MAX_TOKENS`. Defaults reproduce Step 7: 8192 context, 12 threads on this machine, batch 512, zero GPU layers, temperature 0, top-p 1, and 180 maximum tokens.

## 4. Portability fix

The default path now uses the exact existing filename `qwen2.5-1.5b-instruct-q4_k_m.gguf`, not a differently cased Windows-only spelling. Missing-model errors include the resolved path and the configuration variable to set.

## 5. Fair-comparison controls

The 45 focused cases contain 15 English, 15 Bangla, and 15 Banglish rows. Retrieval was run once and stored as frozen evidence. Both models received the same question, verified evidence package, prompt, context allowance, generation settings, language validator, grounding validator, and single-retry policy. A/B identities were randomized per row using seed 46.

## 6. Load performance

| Model | Cold load | RSS increase | RSS after load | Available system RAM after load |
|---|---:|---:|---:|---:|
| 1.5B | 1.953 s | 1.45 GiB | 1.86 GiB | 1.05 GiB |
| 3B | 4.609 s | 2.82 GiB | 3.23 GiB | 0.04 GiB |

Both files loaded through `llama-cpp-python` 0.3.35 with 8192 context. English, Bangla, and Banglish UTF-8 tokenization worked.

## 7. Full-pipeline memory

The existing 1.5B full pipeline was measured on one representative generated question: RSS reached 2.91 GiB, available RAM fell to 0.59 GiB, and swap use increased by approximately 0.29 GiB. It completed stably, but memory headroom is already narrow.

The 3B full pipeline was **not started**. Isolated 3B loading left only about 41 MiB available before BGE-M3, FAISS, BM25, and retrieval were added. This triggered the required memory-safety stop. No full-pipeline or 300-query 3B performance figures are fabricated.

## 8. Focused 45-case results

| Language | 1.5B generated / accepted | 3B generated / accepted | Assessment |
|---|---:|---:|---|
| English | 8 / 7 | 8 / 8 | 3B improved factual English reliability |
| Bangla | 4 / 3 | 4 / 3 | No meaningful semantic improvement |
| Banglish | 10 / 2 | 10 / 2 | No accepted-rate improvement; quality remains poor |

Final strategy counts were 1.5B: 12 generated, 28 unsupported, 5 ambiguous; 3B: 13 generated, 27 unsupported, 5 ambiguous. Both models attempted generation on the same 22 cases. The one final-strategy difference came from post-generation acceptance, not pre-generation routing.

## 9. Language validation

The 3B model reduced obvious language-screen flags: Bangla English-dominated first passes fell from four to one, and Banglish pure-English flags fell from nine to six. However, script-level improvement did not imply semantic correctness. Final safe responses make overall final-language rates unsuitable as a generation-quality score.

## 10. Semantic failure analysis

Manual developer inspection found that 3B English was concise and usually correct. Bangla still produced serious meaning failures that passed both script and factual-token checks, including a publisher answer claiming the prospectus was not published, an accreditation answer repeatedly denying accreditation, and a repeat-examination answer containing unrelated malformed concepts. Two Bangla outputs were truncated at the 180-token/output boundary.

Banglish frequently remained pure English, became malformed mixed language, or was safely rejected. For example, an initial-program answer used malformed wording and an incorrect count, while several retries introduced Bengali script. Automatic validation cannot treat these as human-quality answers.

## 11. Grounding results

| Metric | 1.5B | 3B |
|---|---:|---:|
| Generation-attempted cases | 22 | 22 |
| First-pass accepted | 7 | 13 |
| First-pass rejected | 15 | 9 |
| Final accepted generation | 12 | 13 |
| Final safe abstention | 10 | 9 |

The higher 3B first-pass acceptance is partly misleading: several semantically wrong Bangla answers contained no unsupported named token or number, so the unchanged grounding heuristic accepted them.

## 12. Retry results

| Metric | 1.5B | 3B |
|---|---:|---:|
| Retry attempts | 11 | 4 |
| Successful retries | 4 | 0 |
| Failed retries | 7 | 4 |
| Total retry time | 60.876 s | 69.418 s |

3B needed fewer retries because more first passes passed surface checks, but none of its retries succeeded and semantic quality did not meet the adoption target.

## 13. Fact preservation

The diagnostic was applicable to 44 rows. 1.5B preserved all extracted reference facts in 5/44 cases (11.36%); 3B preserved them in 8/44 (18.18%). Safe abstentions count as non-preservation, and named-token preservation does not establish semantic correctness.

## 14. Generation latency

| Metric | 1.5B | 3B |
|---|---:|---:|
| Median | 7.520 s | 11.020 s |
| P95 | 15.247 s | 35.991 s |
| Median tokens/second | 7.623 | 3.621 |

Cold model-loading time is excluded from these warm generation latencies.

## 15. Full 300-query results

- 1.5B baseline: existing completed Step 7 run, 300/300 queries.
- 3B: **not run — memory safety stop before full-pipeline loading**.

The stopped run is explicitly recorded in `step7b_3b_results.json` with zero fabricated completed queries.

## 16. Strategy counts

The frozen 45-case experiment confirmed identical generation-attempt routing: 22 cases per model. Final accepted generation changed from 12 to 13 solely because 3B passed one additional post-generation validation outcome.

## 17. Automatic development proxies

| Proxy | 1.5B | 3B |
|---|---:|---:|
| Precision | 0.1407 | 0.1804 |
| Recall | 0.1791 | 0.2066 |
| F1 | 0.1538 | 0.1881 |

These are **automatic development proxies, not final correctness**.

## 18. Blinded review file

- Path: `results/step7b_model_comparison_review.csv`
- Rows: 45
- Composition: 15 English, 15 Bangla, 15 Banglish
- Fixed A/B randomization seed: 46
- Human-review columns: blank
- Separate key: `results/step7b_model_blinding_key.csv`

## 19. Model decision

**KEEP_1_5B** as the configured default and low-resource model. Do not adopt 3B on this machine.

## 20. Decision reason

3B improves English and some automatic diagnostics, but it does not meaningfully solve Bangla or Banglish generation. It also leaves unsafe RAM headroom before the full RAG pipeline is loaded. It therefore fails both multilingual-quality and machine-stability adoption gates.

## 21. Does 7B need testing?

**NO on the current 7.87 GB RAM machine.** A 7B Q4_K_M model would require substantially more memory than the already unsafe 3B configuration. No 7B file was downloaded. Any future Step 7C experiment requires explicit user permission and a higher-memory environment.

## 22. Test results

The suite now contains 135 tests: 130 prior tests plus 5 Step 7B configuration/model-switch tests. Result: 135 run, 0 failures, 1 intentional skipped real-reranker test.

## 23. Step 1–7 compatibility

Evidence semantics, ingestion, chunking, pinned BGE-M3, FAISS, BM25, metadata retrieval, RRF, normalization, structured answer policy, prompts, validators, and retry logic remain unchanged. The reranker remains disabled by default.

## 24. 70-PDF readiness

Model selection is configuration-driven and generation still consumes only verified evidence. It has no dependency on document filename, PDF count, current course identifiers, or page numbers.

## 25. 6000-question independence

Runtime answering does not inspect evaluation question IDs, reference answers, expected pages, or dataset size. The focused IDs are used only by the development evaluation script to select cases.

## 26. Git diff summary

No commit was created. Existing uncommitted Step 7 work is preserved. Step 7B adds centralized generator configuration, five mocked tests, a controlled evaluator, five required result artifacts, and one frozen-evidence control artifact. Model GGUF files remain ignored by Git.

## 27. Remaining risks

- The current 1.5B generator remains poor for Bangla and Banglish explanations.
- The 3B model is not viable with the complete pipeline on this machine.
- Surface language and factual-token validators cannot detect all semantic nonsense.
- The focused set is a developer comparison, not an independent thesis benchmark.
- The blinded human-review columns remain unscored and require genuine review if formal human results are desired.

## 28. Next step

Do not start Step 8 under the assumption that multilingual generation is solved. Keep 1.5B as the operational fallback. Before a larger-model experiment, obtain explicit permission and use a machine with substantially more RAM; prompt or validator changes would need a separate controlled experiment rather than being mixed into this model ablation.
