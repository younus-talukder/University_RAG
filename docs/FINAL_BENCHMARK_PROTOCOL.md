# Final benchmark protocol

This protocol is for a future **FINAL_BENCHMARK** on a better machine, not the current one-PDF, 100-base-question development set. The development set and any Step 10 full-generation run are classified DEVELOPMENT/PILOT, never held-out final accuracy. The target of roughly 70 PDFs and 6,000 inference questions is provisional until the official source inventory and dataset are supplied.

## 1. Hardware and environment gate

Use a machine with approximately 32 GB physical RAM as a planning target, then record the actual CPU, GPU/VRAM (if used), physical RAM, pagefile, OS, Python and dependency versions. Thirty-two GB does not guarantee that the final corpus, indexes and Qwen can run efficiently. Before inference, run a small pilot and observe available RAM, peak process RSS, swap/pagefile use, throughput and allocation failures. Never resize the OS pagefile automatically. Use the existing sequential Qwen profile, no parallel inference workers, answer bank OFF and neural reranker OFF.

## 2. Official source inventory and version policy

Create and approve a source-inventory CSV before ingesting. Required fields: `document_id`, `filename`, `relative_path`, `document_type`, `department_or_source`, `official_source_url`, `edition_or_version`, `publication_date`, `effective_date`, `authority_status`, `supersedes_document`, `include_in_benchmark`, `notes`. Leave unknown facts blank, not inferred. Record who approved inclusion and when in a separate audit note. Do not silently prefer one conflicting official version; list every included version, expected conflicts and how reference labels adjudicate them. An unresolved conflict is marked `source_conflict`/`needs_review`, not forced to a single verified answer.

Place approved PDFs under `data/documents/`, preserving meaningful relative paths. The ingestion script discovers recursively. Check duplicate, partial, failed and text-empty PDFs in `results/ingestion_report.json`, and resolve or document each. Inspect suspicious chunks and source/page metadata. Build fresh FAISS/BM25 indexes with `scripts/build_index.py`, inspect `results/index_build_report.json`, then rerun the builder without forcing a rebuild to verify it reports a fresh reusable index. Save the corpus fingerprint, index-configuration fingerprint, index manifest SHA-256 and chunk/document counts. Do not infer correctness solely from a successful build.

From the project root in PowerShell, the existing pipeline entry points are:

```powershell
.\.venv\Scripts\python.exe scripts\ingest.py
.\.venv\Scripts\python.exe scripts\build_index.py --force
.\.venv\Scripts\python.exe scripts\build_index.py
```

Inspect the ingestion/index reports before moving on. The last command must report a fresh reusable index; it is not a substitute for checking content quality.

## 3. Independent final dataset and annotation freeze

Prepare a dataset substantially independent of the 100-question development/tuning set. Document any overlap. Every base question needs a stable unique ID, question text, language, ground-truth status and a reference answer where applicable. Optional columns include intent, difficulty, source/page evidence, entity and course. For parallel English, Bangla and Banglish variants, retain a `base_question_id` for evaluation-only pairing. The evaluator passes only the current question text into production inference; paired translations and labels never enter runtime.

Validate schema, unique IDs, reference availability, evidence source/page pairs, counts by language and ground-truth status. Resolve annotation disagreements before freeze. Save the dataset filename, SHA-256, column schema, row/variant counts, annotation version, adjudication log and approver. Do not train, tune or select production answer rules using final labels. Any annotation correction after freeze needs a new version, hash and written reason; a changed benchmark is a new run.

## 4. Staged corpus and inference gates

1. **Stage A:** 5–10 PDFs and a small question subset. Verify ingestion, citation paths, page labels, conflicts, memory and latency.
2. **Stage B:** 20–30 PDFs and a larger question subset. Verify index freshness, retrieval, scaling and checkpoint recovery.
3. **Stage C:** full approved corpus. Rebuild/verify the final index and lock its fingerprints.
4. **Final-corpus pilot:** approximately 100–300 questions using the full corpus/index. Detect schema, index, memory, latency and evaluator defects. Do not tune production answer rules from these findings.
5. **Final run:** only after all gates pass, classify `FINAL_BENCHMARK`, use full generation, strict profile, fixed dataset/corpus/index/model/code/configuration, and checkpoint/resume. The actual target count is the validated dataset count, not an assumed 6,000.

The final command must omit `--skip-generation`, `--retrieval-only`, `--non-strict`, `--limit`, and `--stop-after`. Replace `<approved-final-dataset.csv>` with the actual frozen dataset path. Resume with the same options plus the existing run directory:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_benchmark.py --dataset <approved-final-dataset.csv> --classification FINAL_BENCHMARK --checkpoint-every 10 --require-references
.\.venv\Scripts\python.exe scripts\evaluate_benchmark.py --dataset <approved-final-dataset.csv> --classification FINAL_BENCHMARK --checkpoint-every 10 --require-references --resume results\benchmarks\<run-id>
```

If some question classes legitimately have no reference answer, document that and omit `--require-references`; their automatic answer proxies remain unavailable. Do not fill missing references merely to satisfy the flag.

Do not change production code, corpus, annotations, model files, retrieval settings or generation settings once the final run begins. A critical change invalidates the run: stop, record why, create a new run ID, refreeze identities and restart. Resume only when the evaluator accepts identical dataset hash, corpus/index fingerprints, model shard hashes, code fingerprint and effective configuration. Keep checkpoint batches, run manifest, result CSV, retrieval trace, error analysis and all derived reports. Check unique evaluation IDs, no missing completed rows, atomically written checkpoints and `run_complete=true` only for the full expected set (a recorded `SYSTEM_ERROR` still counts as a completed result).

## 5. Human review and metric definitions

Select a statistically reasonable review sample after inspecting the final run size, language/intent/status/strategy distribution and available reviewers. Stratify at least by language, answer strategy, answerability status and intent where possible. Sample both returned answers and abstentions. Prefer two independent reviewers on an overlapping subset, then report agreement; never invent ratings or reviewers. Keep review fields blank until genuine scores are imported through `scripts/import_human_review.py`.

Report these separately, with denominators and applicable counts:

- **Retrieval:** Hit@1, Hit@3, MRR@3, source/page accuracy, entity/field/supportability diagnostics where labeled.
- **Answerability:** SafeAnswerCoverage = independently gate-passing returned answers / all inference rows; AbstentionRate and each status distribution. SafeAnswerCoverage is **not answer accuracy**.
- **Human quality:** correctness, relevance, groundedness, completeness, naturalness and semantic consistency from genuine reviewed rows only, with review selection and uncertainty disclosed.
- **Safety:** UnsafeReturnedAnswerCount, conflict handling and relevant rejection categories. An automatic audit is not proof of zero hallucinations.
- **Efficiency:** mean, median and P95 latency; retrieval/generation/answerability time; throughput; peak RSS, minimum available RAM and pagefile/VRAM observations where available.
- **Multilingual parity:** English, Bangla, Banglish support combinations and only deterministic, defensible factual-equivalence checks.
- **Automatic answer proxies:** normalized exact match, token precision/recall/F1, numeric and identifier preservation. Label every such result `AUTOMATIC PROXY`, not human accuracy.

Use “accuracy” only for an explicitly defined metric and denominator, for example labeled top-1 page accuracy or reviewed correctness acceptance. Break down failures into retrieval miss, ranking miss, entity/field/relation detection, evidence validation, source conflict, ambiguity, generation, language realization, grounding, semantic validation, system error and annotation issue. Preserve uncertainty rather than force a diagnosis.

## 6. Reproducibility package and handoff

Archive the source inventory and version policy, dataset and annotation manifests, corpus/index manifests, both Qwen shard SHA-256 hashes, model settings, fixed retrieval configuration, Git commit and dirty status, code fingerprint, Python/dependency versions, OS/hardware, effective run configuration, checkpoint batches, result files, reviewer sampling protocol and imported score files. Prefer a clean committed checkout on the future machine, but record any dirty files exactly. Recompute every hash on that machine; do not assume a copied file is identical. Run the full test suite there before the final-corpus pilot.

No current 70-PDF ingestion or 6,000-question inference has been performed. This protocol specifies the future execution; it does not assert present validation of that scale.
