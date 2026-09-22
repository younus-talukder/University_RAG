# Step 7C Final Migration and Development Validation Report

**Development/regression validation only — not the future 6,000-question thesis benchmark.**

## 1. Migration status

Completed. Qwen2.5-7B-Instruct Q4_K_M is the only active generator. The old 1.5B and 3B binaries were removed after the two 7B shards passed official-metadata, SHA-256, shard-discovery, chat-template, and isolated-load checks. Retrieval, evidence policy, structured answers, validation, and one-retry behavior remain unchanged.

## 2. Qwen 7B model

- Repository: `Qwen/Qwen2.5-7B-Instruct-GGUF`
- Immutable revision: `bb5d59e06d9551d752d08b292a50eb208b07ab1f`
- Quantization: `Q4_K_M`
- License: Apache-2.0
- Shard 1: `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`
  - Size: 3,993,201,344 bytes
  - SHA-256: `DFCE12E3862A5283CCFB88221B48480E58745165DE856439950D0F22590580DB`
- Shard 2: `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf`
  - Size: 689,872,288 bytes
  - SHA-256: `539CF93F78E887EDEA1C04E2D7D8CDACA9D01DAE9C9025BCB8ACCBE29DF3D72A`
- Total visible model size: 4,683,073,632 bytes (4.36 GiB)

Both hashes exactly match the pinned repository LFS metadata. The first shard is the configured path; llama.cpp automatically discovered shard 2. No concatenation was performed.

## 3. Old models removed

- 1.5B: removed; 1,117,320,736 bytes reclaimed.
- 3B: removed; 2,104,932,768 bytes reclaimed.
- Total reclaimed: 3,222,253,504 bytes (3.00 GiB).

Before removal both files were confirmed ignored by `models/*.gguf`, untracked by Git, and isolated model binaries. Historical results, BGE-M3/reranker caches, PDFs, datasets, FAISS, and BM25 artifacts were not removed.

## 4. Files changed

Step 7C changes are in `src/config.py`, `src/generator.py`, `.env.example`, `scripts/evaluate_step7c.py`, `tests/test_step7b_generator_config.py`, `tests/test_step7c_evaluator.py`, and `README.md`. Step 7C result artifacts were added under `results/`. Pre-existing uncommitted Step 7/7B work remains present and was not reverted.

## 5. Final generator architecture

One active model only: Qwen2.5-7B-Instruct Q4_K_M. There is no 1.5B fallback, 3B fallback, translation model, low-resource model switch, or automatic model switching. Structured exact/list answers bypass the model. Supported explanatory answers receive only tokenizer-budgeted verified evidence, then undergo language and grounding validation with at most one controlled retry. The generator is cached and is not reloaded per query.

## 6. DEVELOPMENT_8GB profile

- Context: 4096
- Threads: 8
- Batch: 128
- GPU layers: 0
- Temperature: 0
- Top-p: 1
- Maximum output tokens: 180
- mmap: true
- mlock: false

## 7. Stable profile selected

Initial Profile A (`n_ctx=4096`, `n_batch=128`) completed isolated loading, the 15-question smoke run, resume validation, and all 300 variants without process or allocation errors. No lower fallback profile was needed. This is the highest requested development profile and remains configuration, not a core-logic assumption.

## 8. 7B isolated load

- Cold load: 7.396 s
- RSS after load: 5,340,581,888 bytes (4.97 GiB)
- RSS after the short generation: 5,573,922,816 bytes (5.19 GiB)
- Available RAM after load: 457,887,744 bytes (436.7 MiB)
- Available RAM after generation: 495,497,216 bytes (472.5 MiB)
- Pagefile/swap used before: 1,537,323,008 bytes
- Pagefile/swap used after generation: 1,787,314,176 bytes

The chat template returned correct Unicode for a short three-line English/Bangla/Banglish probe. The first capture attempt completed but Windows CP1252 failed while printing Bangla; the UTF-8 capture rerun succeeded. This was a console-encoding issue, not model failure.

## 9. Full RAG memory

- Peak process RSS: 5,767,397,376 bytes (5.37 GiB)
- Minimum system-available RAM: 47,996,928 bytes (45.8 MiB)
- Peak pagefile/swap used: 3,938,754,560 bytes (3.67 GiB)
- Observed swap range during the saved run: about 1.01–3.67 GiB

Windows virtual memory was essential. The machine stayed usable enough to finish, but physical-memory headroom became extremely small. Pagefile settings were not modified. Keep a system-managed pagefile and adequate free disk; the current roughly 13 GiB pagefile/20.9 GiB total commit limit was sufficient for this run.

## 10. Prompt token statistics

- Median: 520
- P95: 663
- Maximum: 680

All observed verified prompts fit comfortably within 4096. No evidence was reduced to force the smaller context.

## 11. 15-question smoke test

All 15 generation-required cases completed with zero runtime errors. Eight answers were accepted and seven safely rejected.

| Language | ID | Final outcome | Semantic review |
|---|---|---|---|
| English | Q002 | Accepted | Correct publisher. |
| English | Q005 | Accepted | Correct disclaimer. |
| English | Q007 | Accepted | Correct two initial programs. |
| English | Q009 | Accepted | Correct accreditation. |
| English | Q010 | Accepted | Correct Private University Act 1992. |
| Bangla | Q002 | Accepted automatically | Semantically wrong negation: says publisher information was not given. |
| Bangla | Q009 | Accepted automatically | Correct core fact but grammatically malformed/mixed words. |
| Bangla | Q013 | Rejected after retry | Safe abstention. |
| Bangla | Q026 | Rejected | Safe abstention. |
| Bangla | Q028 | Accepted automatically | Grammatically broken and semantically unreliable. |
| Banglish | Q002 | Rejected | Pure-English failure; safe abstention. |
| Banglish | Q005 | Rejected after retry | Safe Banglish abstention. |
| Banglish | Q007 | Rejected after retry | Safe Banglish abstention. |
| Banglish | Q009 | Rejected after retry | Safe Banglish abstention. |
| Banglish | Q013 | Rejected after retry | Safe Banglish abstention. |

Automatic script checks are not semantic correctness. Complete answers, evidence, validations, latency, token rate, and retry fields are in `step7c_smoke_results.csv` and `step7c_smoke_summary.json`.

## 12. Memory stability

No crash or allocation error occurred. Generation RSS varied from about 4.17 to 5.21 GiB as mmap pages entered/left the working set. Median RSS was about 4.57 GiB in the first half of generation calls and 4.52 GiB in the second half, so there is no monotonic leak signal. BGE-M3 and 7B can remain cached together; measured evidence does not justify repeated unload/reload. Such cycling would save resident/commit pressure temporarily but add large reload latency and complexity.

## 13. 300-variant development run

- Completed: 300/300 (100 English, 100 Bangla, 100 Banglish)
- Errors: 0
- Unique keys: 300/300
- Resume: real run completed 5 rows, stopped, resumed to 10, verified, then resumed to 300
- Answer bank: off
- Reranker: off
- Generator: Qwen2.5-7B-Instruct Q4_K_M

## 14. Answer strategies

- `structured_exact`: 205
- `structured_list`: 2
- `gguf_generation`: 17
- `unsupported`: 68
- `ambiguous`: 5
- `conflicting`: 3

Structured latency remained essentially instantaneous (median 0.000236 s), confirming the LLM bypass.

## 15. Language quality

- English automatic consistency: 100/100; generated acceptance 9/11. Smoke semantics were strong.
- Bangla automatic consistency: 100/100; generated acceptance 4/6. Manual smoke review found wrong negation, malformed grammar, and semantically unreliable text despite validator passes.
- Banglish automatic consistency: 98/100; generated acceptance 4/14. Ten generation attempts were rejected; pure-English leakage remains.

The aggregate automatic rates are dominated by deterministic structured/unsupported responses and must not be reported as human answer correctness.

## 16. Grounding

- Generation attempted: 31
- Accepted: 17
- Rejected/safely abstained: 14
- Retries: 9
- Successful retries: 2
- Failed retries: 7

The grounding boundary remained enforced; failed outputs were not returned as factual answers.

## 17. Generation latency

- Median generation latency: 29.103 s
- P95 generation latency: 81.757 s
- Median completion rate: 1.099 tokens/s
- Median total latency across all 300: 0.225 s
- P95 total latency: 31.333 s

## 18. Manual review sample

`step7c_manual_review_sample.csv` contains 30 rows: 10 English, 10 Bangla, and 10 Banglish. Selection prioritizes generation-attempt cases and includes reference answer, 7B answer, strategy, validation flags, source/page, and supporting evidence. It contains no fabricated human scores.

## 19. Historical 1.5B vs current 7B

On the same 300 variants:

- Accepted generation: 15/31 (1.5B) -> 17/31 (7B)
- English: 9/11 -> 9/11
- Bangla: 3/6 -> 4/6
- Banglish: 3/14 -> 4/14
- Retries: 15 -> 9
- Successful retries: 4 -> 2
- Failed retries: 11 -> 7
- Median generation latency: 6.652 s -> 29.103 s
- P95 generation latency: 22.104 s -> 81.757 s

7B modestly improved automatic acceptance/retry counts and English remained strong, but manual quality shows unresolved Bangla and Banglish failures. Human review remains necessary.

## 20. Checkpoint/resume status

Passed. Results are written to a temporary CSV, flushed and fsynced, then atomically replaced every five rows. Resume uses `(question_id, language)`, not positional slicing. The real test produced 10 unique rows after a 5-row restart, preserved the 39-column schema, left no `.tmp` file, and later completed 300 unique rows with no missing rows.

## 21. Test results

- Previous Step 7B suite: 135 tests
- Current: 139 tests
- Result: `OK (skipped=1)`
- Failures/errors: 0

Normal tests mock the large model. New coverage includes 7B defaults, both runtime profiles, mmap/mlock, first-shard portability, missing shard failure, generator caching, atomic checkpointing, and deduplicated resume.

## 22. Model directory final state

Only the two official 7B Q4_K_M shards are present as model files. No `qwen2.5-1.5b...` or `qwen2.5-3b...` binary remains.

## 23. Disk space

- Removed: 3,222,253,504 bytes (3.00 GiB)
- Added: 4,683,073,632 bytes (4.36 GiB)
- Net visible model increase: 1,460,820,128 bytes (1.36 GiB)

## 24. Step 1–7 compatibility

Preserved. Retrieval constants and neural-reranker-disabled behavior were not redesigned. Evidence-bound answering, generic multi-PDF ingestion, structure-aware chunking, pinned BGE-M3/FAISS, BM25/metadata/RRF, multilingual normalization, structured answers, verified context, validation, and one retry remain active. All 139 tests pass.

## 25. Current 1-PDF / 100-question readiness

**READY: YES for functional development/regression use.** The 300 variants complete stably with no missing/error rows. This does not mean multilingual semantic quality is thesis-ready; Bangla and Banglish generation issues remain documented.

## 26. Future 70-PDF readiness

**Architecture ready: YES. Not benchmarked yet.** Multi-PDF discovery, generic IDs/manifests, chunking, and retrieval were retained. Capacity/performance must be rebuilt and measured on the 32-GB target.

## 27. Future 6,000-question readiness

**Evaluation architecture ready: YES. Not benchmarked yet.** It is input-size/ID independent and supports atomic checkpoint/resume, explicit error rows, progress, and ETA. The 6,000-question evaluation must run on the future machine.

## 28. Future 32-GB machine profile

Start with `FULL_32GB`: context 8192, threads 12, batch 512, GPU layers 0 unless compatible GPU offload is measured, temperature 0, top-p 1, max tokens 180, mmap true, mlock false. Tune threads, batch, and GPU layers by measurement without changing RAG source logic.

## 29. Git diff summary

No commit was created. The working tree already contained uncommitted Step 7/7B changes; Step 7C was added on top without reverting them. `git diff --check` passes (only CRLF conversion warnings). Model binaries remain Git-ignored/untracked.

## 30. Remaining issues

1. Bangla semantic validation is too shallow: it missed wrong negation and malformed meaning.
2. Banglish generation acceptance remains low and pure-English leakage persists.
3. Available physical RAM fell below 50 MiB and pagefile use exceeded 3.6 GiB; other concurrent heavy applications can destabilize this 8-GB profile.
4. 7B latency is high on CPU (about 29 s median and 82 s P95 per generation).
5. The 30-row review requires actual human scoring; none was fabricated.
6. The 70-PDF/6,000-question benchmark remains intentionally unrun.

## 31. Next step

Do not start Step 8 automatically. First, human-score the 30-row review sample and decide whether Step 7 needs a focused Bangla/Banglish semantic-validation/prompt improvement while keeping the finalized 7B-only architecture. Then move the unchanged architecture and `FULL_32GB` profile to the 32-GB machine for large-corpus preparation.
