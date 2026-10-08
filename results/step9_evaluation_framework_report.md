# Step 9 — Dataset-agnostic evaluation and benchmark framework

1. **Step 9 goal.** Implement one offline evaluator that scales by dataset/configuration, separates references from runtime, supports reproducible runs, and does not tune the frozen Step 1–8 RAG pipeline. This engineering goal is met for the framework.

2. **Existing evaluation audit.** Reusable: `src/evaluator.py` language/reference conventions, current index manifest, `src/answer_safety.py` trust audit, and frozen evidence diagnostics. Legacy: `scripts/evaluate.py`, `scripts/analyze_experiments.py`, and `scripts/evaluate_retrieval.py` use fixed files and narrower schemas. Step-specific: Step 5/6/7/8 scripts and reports encode their own gates and run layouts. Duplicated: CSV I/O, multilingual expansion, source/page parsing, checkpoint logic, and result summaries. Missing before Step 9: generic XLSX input, manifest-linked run identity, strict resume compatibility, sample/score workflow, and dynamic-size analysis. All historical scripts/results are preserved; `scripts/evaluate_benchmark.py` is the new primary entry point, not a rewrite of historical evidence.

3. **Files changed.** Added `src/evaluation/{schema,manifest,runner,metrics,human_review}.py`, six benchmark/review/comparison/scale scripts, `tests/test_step9_evaluation.py`, `docs/step9_benchmark.md`, this report/summary, and benchmark run folders. Updated `README.md`. No production RAG files were changed.

4. **Evaluation architecture.** Dataset reader/validator → independent language descriptors → question-only runtime call → post-inference label join → atomic checkpoint → CSV/JSON metrics and Markdown report → optional genuine human-score import. One evaluator handles all languages and row counts.

5. **Dataset schema.** Required per base row: at least one selected-language question. `question_id` is recommended and must be unique if supplied; a stable row-position ID is assigned otherwise. Optional: `course_code`, `intent`, `difficulty`, language-specific references, expected source/page, paired `expected_evidence`, annotation status, and notes. A missing reference makes proxy metrics `NOT_AVAILABLE` unless `--require-references` is selected.

6. **CSV / XLSX support.** Both read without manual conversion; CSV accepts UTF-8 BOM with strict decoding, and XLSX uses the active sheet read-only. Headers, duplicate IDs, question text, annotation labels, and page numbers are validated. The current 14-column dataset passed with 0 errors and 0 warnings.

7. **Dataset manifest.** `dataset_validation.json` records filename, absolute input path, SHA-256 `8b2e6c06ffba8d4dee58fd979706f244e5891591c4e8e25d9beffea801e149b2`, 100 base rows, 300 language variants, 14 columns, selected languages, and validation outcome.

8. **Run manifest.** Every run has a unique UTC timestamp/nonce ID and its own folder. The final validation run is `step9-development-final-20260929T185115270858Z-84c03a`; it records expected/completed IDs, classification, completion, leakage boundary, dataset, index, model, and code identity. `run_config.json` snapshots effective non-secret settings.

9. **Code / Git provenance.** Final-run manifest records Git commit `6c23ad85b2d9f5c0b18a898e72e568106a383989`, `working_tree_dirty=true`, 172 modified/untracked files at run start, and a source-code SHA-256 `672cbfdf2598cbaebe4a1ecb061d8f113689daec99b045c5168be10e2e008538`. The run is not represented as coming from a clean committed revision.

10. **Corpus / index provenance.** Read directly from `vector_db/index_manifest.json`: corpus fingerprint `8d25c720865665759d79c3f333f9ba549fc2408e9d8d37c6206fbf2099d83c8e`, index-configuration fingerprint `8207af3ec864b70f106ddbe0197faa2ccba786bb034942ea7027f3fcc85ebc88`, one document, 491 chunks, BGE-M3 pinned revision, `structure-aware-v1.2`, FAISS `IndexFlatIP`, and local BM25 schema.

11. **Model / runtime provenance.** The manifest records both Qwen2.5-7B-Instruct Q4_K_M shard names, development 8 GB profile, context 4096, 8 threads, batch 128, CPU GPU-layers 0, temperature 0, top-p 1, 180 max tokens, reranker OFF, and answer bank OFF. GGUF hashes are `NOT_AVAILABLE` because no previously verified hash manifest was present; Step 9 did not rehash the large model files. Qwen was configured but **not loaded** in the final generation-skipped validation.

12. **Checkpoint / resume.** Numbered JSON batches are staged, flushed, and atomically replaced every 10 rows. Resume verifies dataset hash, effective config, corpus/index, model, code fingerprint, expected IDs, contiguous batches, and duplicate-free IDs. CSV outputs are atomically rebuilt from checkpoints. A crash can replay only an unfinished in-memory batch, never duplicate a completed evaluation ID.

13. **Error isolation.** A row exception is recorded as `SYSTEM_ERROR` with type and short message; later rows continue. A runtime-returned error is likewise preserved. The final 300-row validation had 0 runtime errors.

14. **Retrieval metrics.** Final development run: Hit@1 41.33%, Hit@3 89.67%, MRR@3 0.6339; these require expected evidence labels. Source and page accuracy, entity/field hits, and supportability are separately exported. Multiple valid sources/pages and paired source-page labels are supported; missing labels produce `NOT_AVAILABLE`.

15. **Answerability metrics.** Final generation-skipped statuses: 237 `SUPPORTED`, 19 `INSUFFICIENT_EVIDENCE`, 1 `RETRIEVAL_UNCERTAIN`, 8 `CONFLICTING_EVIDENCE`, and 35 `GENERATION_REJECTED`; no ambiguous, out-of-domain, or system-error rows. The 35 generation rejections in this mode had no generation attempts and must not be described as failed Qwen outputs.

16. **Safe answer coverage.** 237/300 = 79.0%, Wilson 95% interval about 74.0–83.2%, for this `skip_generation` DEVELOPMENT run only. This is **not** factual accuracy and is not directly comparable to the historical 258/300 full-generation Step-8B safe-answer coverage.

17. **Abstentions.** 63/300 = 21.0%, broken down by the statuses above. Abstention messages are not scored against reference answers as hallucinations.

18. **Automatic answer proxies.** Returned answers with eligible references receive normalized exact match, token precision/recall/F1, reference-token overlap, numeric preservation, and technical identifier preservation. These are labeled `AUTOMATIC DEVELOPMENT / BENCHMARK PROXIES`, not human correctness.

19. **Human review framework.** A deterministic, configurable sampler stratifies by language and diversifies status, strategy, intent, and difficulty. Final sample: 30 blank-rated rows, 10 per language, at `human_review_sample.csv`. Optional blinding hides system/strategy/status and saves a separate key; multi-system answers are randomized.

20. **Human score import.** `scripts/import_human_review.py` validates IDs, duplicate reviewer/row pairs, 1–5 rating ranges, `PASS`/`FAIL`, and `ACCEPT`/`PARTIAL`/`REJECT`. It supports optional `reviewer_id`, sample-only grouped metrics, and agreement only when the same rows have at least two genuine reviewers. No human ratings were fabricated or imported for this run.

21. **Multilingual parity.** Evaluation-only pairing gives 72 `ALL_THREE_SUPPORTED`, 9 `ENGLISH_AND_BANGLISH`, 3 `ENGLISH_ONLY`, and 16 `NONE_SUPPORTED` base questions; other categories are zero. The parity file also records primary-evidence and numeric-value equivalence where deterministic comparison is meaningful. No paired translation affects runtime.

22. **Error taxonomy.** Stable categories include retrieval/ranking misses, detection/validation failures, conflict/ambiguity, generation/language, unsafe return, annotation/source conflict, system error, and unknown. Classification is post-inference. This run classified 15 retrieval misses, 3 ranking misses, 8 source conflicts, and 274 unknown/not-provably-diagnosed; it made no production changes.

23. **Latency / resource metrics.** Final run mean/median/P95 per-question latency: 0.0419/0.0405/0.0602 seconds; max observed process RSS about 448 MB and system pagefile-used observation about 599 MB. Optional telemetry yields `NOT_AVAILABLE` if unavailable. The latency excludes cold model startup and all Qwen generation, so it is not a full-system benchmark.

24. **Confidence intervals.** Wilson 95% intervals are available for proportion metrics with explicit applicable `n`, including Hit@K, safe coverage, language consistency, and imported human accept rates. Tiny groups retain their sample size; no precision claim is made from a small group.

25. **Development vs final benchmark.** Run classes are `DEVELOPMENT`, `PILOT`, and `FINAL_BENCHMARK`. The current 100-base-question set is `DEVELOPMENT` in the manifest and prominently in the report. Final thesis claims require a later independent benchmark and human review.

26. **Leakage audit.** The evaluator's runtime call passes only the question text, top-K, generation mode, and explicit `use_answer_bank=False`. Reference answer, expected source/page, question ID, paired translations, and human scores join only after inference. The strict boundary is exercised by tests; Q007 remains the user-approved page-4 Computer Science & Engineering evaluation annotation without runtime logic.

27. **Strict benchmark profile.** Answer bank is unavailable in the primary evaluator, reranker must be OFF, and temperature must be 0. A non-strict diagnostic mode cannot be classified `FINAL_BENCHMARK`. Checkpointing and fixed config/provenance are mandatory.

28. **Current 300-row DEVELOPMENT validation.** Final run `step9-development-final-20260929T185115270858Z-84c03a`: 300/300 unique variants, 0 runtime errors, 30 checkpoint batches, 873 retrieval trace rows, 30 blank-rated human sample rows, and all standard artifacts. Mode: `skip_generation`; this is framework validation, not new final thesis accuracy.

29. **Resume test.** Real run `step9-development-validation-20260929T184029147737Z-eda40f` stopped cleanly at 10/300, then resumed to 300/300 with the same run ID and no duplicates. Unit tests additionally compare resumed and uninterrupted metrics on the same synthetic inputs and test mismatch/duplicate rejection.

30. **6,000-descriptor simulation.** `results/step9_scale_simulation.json` records 6,000 unique descriptors, 600 planned batches at interval 10, and about 444 KB peak Python allocation for that bookkeeping sample. It did not fabricate inference outputs or run Qwen.

31. **Estimated future 6,000-run time.** `ESTIMATE ONLY`: historical Step-8B full-generation 300-row summed latency was 1,229.1 seconds; a straight-line 20× extrapolation is about 6.83 hours before larger-corpus effects, startup, model/hardware changes, and review. The current Step-9 generation-skipped warm-question extrapolation (~0.070 hours) is **not** a Qwen estimate. No 6,000-row runtime was measured.

32. **Multi-PDF evaluation readiness.** Result and trace schema carry `document_id`, `relative_path`, multiple valid sources/pages, and optional paired source-page evidence; no single PDF name is hardcoded in the new evaluator.

33. **Test results.** Frozen baseline: 238 tests. New Step-9 tests: 33. Total: 271, 0 failures, 1 skip. All Step 1–8 tests remain green.

34. **Memory impact.** The framework stores bounded per-row traces/checkpoints and runs sequentially. No extra model or parallel Qwen was loaded; the generation-skipped 300-row run was chosen because free physical RAM was below 1 GB before validation.

35. **Step 1–8 compatibility.** Production `src/pipeline.py`, retrieval weights, model choice, grounding gates, and historical scripts/results were not changed. A benchmark finding is recorded, not silently used to tune answers.

36. **70-PDF readiness.** Evaluation provenance schema: **YES**. Actual 70-PDF scale benchmark: **NOT TESTED**.

37. **6,000-question readiness.** Evaluator schema/checkpoint bookkeeping: **YES** by simulation and dynamic tests. Actual 6,000-question inference: **NOT RUN**.

38. **Git diff summary.** New evaluation package, scripts, tests, docs, simulation, and unique development-run artifacts; only `README.md` is a modified tracked file. No commit was made. The working tree intentionally remains dirty.

39. **Remaining risks.** No new full-generation 300-row run through this entry point; 70-PDF retrieval/memory/throughput not validated; XLSX formula cells require cached values; automatic overlap cannot establish factual correctness; source-page cross-products need the optional paired-label column; system pagefile telemetry is system-wide, not evaluator-only.

40. **Step 9 engineering-complete?** **YES** for the generic framework and DEVELOPMENT validation. This does not certify final thesis quality or future hardware-scale performance.

41. **Ready for Step 10?** **YES**, as a separate engineering task. A controlled full-generation pilot should precede any final benchmark/accuracy claim, and full-scale 70-PDF/6,000-question testing remains future work.

42. **Next step.** Review the final run artifacts and blank human sample, then define Step 10 explicitly. Step 10 was not implemented here.
