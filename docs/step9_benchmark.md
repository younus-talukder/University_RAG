# Step 9 benchmark framework

The benchmark code in `src/evaluation/` is offline evaluation infrastructure. Runtime RAG remains in `src/pipeline.py`; benchmark references never enter its call. Historical `scripts/evaluate_step*` files and reports are preserved as development evidence.

## Dataset contract

Input: UTF-8 CSV (BOM accepted) or XLSX active worksheet. At least one of `english_question`, `bangla_question`, or `banglish_question` must be populated per base row. `question_id` is recommended; absent IDs become stable row-position IDs for that immutable dataset hash. Supplied IDs must be unique. English/Bangla/Banglish reference answers are optional unless `--require-references` is chosen. `course_code`, `intent`, `difficulty`, `expected_source`, and `expected_page` are optional. Supported annotation statuses are `verified`, `needs_review`, `source_conflict`, `ambiguous_reference`, and `unsupported`; missing status is `unlabeled`. Uncertain annotations are excluded from strict reference-overlap proxies by default.

Multiple valid sources/pages can be semicolon/pipe-delimited or JSON arrays. Use `expected_evidence` as a JSON list of `{ "source": "...", "page": 4 }` objects when valid source–page pairings matter. Missing labels produce `NOT_AVAILABLE`, not zero. The approved Q007 development annotation remains the canonical dataset value: page 4, Computer Science & Engineering. This is evaluation metadata only.

## Reproducibility and recovery

Each run has a unique UTC timestamp/nonce ID. The run manifest records SHA-256 of the dataset, row/variant counts and schema, corpus and index fingerprints from the active manifest, generator and retrieval settings, Git commit/dirty state, and a source-code fingerprint. The current index manifest records document count, chunk count, embedding model/revision, chunker version, FAISS type, and BM25 schema. Since Step 10, available local GGUF shards are hashed at run preparation and their SHA-256 values and sizes are recorded; earlier Step 9 run manifests retain `NOT_AVAILABLE`. Effective installed dependency versions are also captured for new runs.

Every `--checkpoint-every N` rows, the runner writes an atomic numbered JSON batch. A clean stop flushes the final short batch. An interrupted process can replay at most the incomplete in-memory batch; completed IDs are never duplicated. Result and trace CSVs are rebuilt atomically from checkpoints. `--stop-after N` is a development/PILOT-only way to test partial completion and resume; it does not change the run's dataset or scoring configuration. The reanalysis script marks a partial run `run_complete=false`. Resume rejects mismatched dataset, config, corpus/index, model, code fingerprint, unexpected IDs, and duplicate checkpoint IDs. A dirty working tree is recorded, not prohibited.

## Metrics

Retrieval `Hit@1`, `Hit@3`, and `MRR@3` use all available expected evidence labels; source and page accuracy are reported separately. Entity, field, and supportability diagnostics use the frozen runtime evidence machinery offline. A candidate can match one of several valid sources or pages; paired labels avoid invalid cross-pairs. Wilson 95% intervals include the applicable sample size.

`SafeAnswerCoverage` counts returned answers that pass grounding, applicable semantic validation, language validation, and the existing independent safety audit, divided by all inference rows. Abstention statuses are counted separately. `UnsafeReturnedAnswerCount` is a known-gate failure count, not a hallucination rate. Exact match, token precision/recall/F1, numeric preservation, and identifier preservation are automatic reference proxies only for returned answers with suitable references. They are not human correctness. Human scores are sample-only and imported separately. Inter-rater agreement is unavailable until at least two reviewers score the same rows.

Multilingual parity joins base IDs only after inference. It reports the eight support combinations plus `OTHER_MISMATCH`, and conservative primary-evidence/numeric-value comparisons where all variants support an answer. No paired answer is used at runtime. The error taxonomy is a post-inference diagnostic, not a production control. Breakdowns are generated from labels present in the dataset; low-count entity categories are suppressed from summary tables. Latency and optional memory/pagefile readings include `NOT_AVAILABLE` when telemetry is missing. Future 6,000-question runtime projections are estimates from measured throughput, not benchmark results.

## Run comparison

`scripts/compare_benchmarks.py --left <run_dir> --right <run_dir>` requires matching dataset hashes and evaluation IDs. `--aligned-subset` explicitly limits a comparison to common IDs when hashes/rows differ. No obsolete baseline experiments are rerun automatically. The comparison is an automatic paired support diagnostic, not a final thesis claim.

## Scale boundary

`scripts/simulate_benchmark_scale.py` exercises descriptor IDs and checkpoint counts for 6,000 synthetic entries without running Qwen or fabricating responses. It is not a 6,000-question inference benchmark. Multi-document provenance is supported by the schema, but an actual 70-PDF benchmark is not tested in Step 9.
