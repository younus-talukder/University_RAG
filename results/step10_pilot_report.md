# Step 10 full-generation PILOT

## 1. Goal and boundary

Validate the Step 9 evaluator with genuine Qwen generation on the current **DEVELOPMENT** dataset, then prepare a future final-benchmark handoff. This run is classified **PILOT**, not `FINAL_BENCHMARK`. No 70-PDF or 6,000-question inference was attempted. Production RAG remained frozen.

## 2. Q007 evaluation annotation

Before and after: page **4**, **Computer Science & Engineering** in all three reference answers and `expected_page`. The user explicitly reconfirmed this choice. The source PDF also says **Computer Science & Technology** on page 8; the new Step 10 attachment's contrary approval claim is superseded by the direct confirmation. The dataset row was retained unchanged, with an audit note in `docs/Q007_ANNOTATION_ADJUDICATION.md`. Production code changed: **NO**.

A Q007-only full-mode rerun and the PILOT both returned page-8 Technology answers in all three languages (no Qwen invocation on those rows). They disagree with the approved evaluation reference but are sourced to a real conflicting passage. The old Step 9 skip-generation lexical run returned page-4 Engineering for English/Banglish and abstained in Bangla; full mode uses a different dense/hybrid retrieval path. Record this as a source/version and evaluation disagreement, not a reason to tune production.

## 3. Files changed

Step 10 adds `scripts/freeze_development_dataset.py`, `tests/test_step10_evaluation.py`, `docs/Q007_ANNOTATION_ADJUDICATION.md`, `docs/FINAL_BENCHMARK_PROTOCOL.md`, `docs/FINAL_BENCHMARK_CHECKLIST.md`, and Step 10 result files. It extends only `src/evaluation/manifest.py`, `src/evaluation/runner.py`, `src/evaluation/metrics.py`, and `src/evaluation/human_review.py`. No `src/pipeline.py`, retriever, evidence, generator, model, or index file was changed. Historical Step 9 artifacts remain preserved.

## 4. Frozen development dataset

`data/questions/questions.csv`; SHA-256 `8b2e6c06ffba8d4dee58fd979706f244e5891591c4e8e25d9beffea801e149b2`. Version `v1-final`, 100 base rows, 300 variants (100 English, 100 Bangla, 100 Banglish), all 100 ground-truth statuses `verified`. The 14-column schema and Q007 adjudication version `page4-engineering-confirmed-2026-09-30` are in `results/step10_development_dataset_v1_final_manifest.json`. A future annotation correction requires a new version and hash.

## 5. Model hashes

Qwen2.5-7B-Instruct Q4_K_M, local shards:

- `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`: `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db`
- `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf`: `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a`

## 6. Full-generation PILOT

Run ID `step10-full-generation-pilot-20260929T192719893950Z-367c33`; classification `PILOT`; strict full-generation mode, Qwen temperature 0, answer bank OFF, neural reranker OFF, `DEVELOPMENT_8GB`, CPU-only sequential inference. **300/300** rows complete; **0** runtime errors; `run_complete=true`; no memory-safety stop. The normal Step 9 run folder contains run configuration/manifest, dataset validation, results, retrieval trace, all metric tables, parity, error analysis, blank human-review sample and benchmark report. Dataset/corpus/model/code identities and installed dependency versions are recorded there. The manifest's leakage audit passed: labels and paired translations did not enter runtime.

## 7. Retrieval metrics

| Language | Hit@1 | Hit@3 | MRR@3 | Source top-1 | Page top-1 | Supportable@1 | Supportable@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English | 0.670 | 0.990 | 0.828 | 1.000 | 0.670 | 0.550 | 0.900 |
| Bangla | 0.530 | 0.970 | 0.740 | 1.000 | 0.530 | 0.590 | 0.850 |
| Banglish | 0.530 | 0.950 | 0.735 | 1.000 | 0.530 | 0.690 | 0.930 |
| Overall | 0.577 | 0.970 | 0.768 | 1.000 | 0.577 | 0.610 | 0.893 |

Overall Entity Hit@1/@3 = **1.000/1.000**, applicable `n=210`; Field Hit@1/@3 = **0.938/0.985**, applicable `n=273`. Source top-1 is uninformative as a discriminator in the current single-PDF corpus. These are labeled retrieval diagnostics, not answer correctness.

## 8–10. Answerability, SafeAnswerCoverage and abstentions

SafeAnswerCoverage = **260/300 = 86.67%**. This is **not accuracy**. AbstentionRate = **40/300 = 13.33%**. Status counts:

| Status | Overall | English | Bangla | Banglish |
| --- | ---: | ---: | ---: | ---: |
| SUPPORTED | 260 | 88 | 84 | 88 |
| INSUFFICIENT_EVIDENCE | 7 | 2 | 5 | 0 |
| RETRIEVAL_UNCERTAIN | 2 | 1 | 1 | 0 |
| AMBIGUOUS_QUERY | 0 | 0 | 0 | 0 |
| CONFLICTING_EVIDENCE | 6 | 2 | 2 | 2 |
| GENERATION_REJECTED | 25 | 7 | 8 | 10 |
| OUT_OF_DOMAIN | 0 | 0 | 0 | 0 |
| SYSTEM_ERROR | 0 | 0 | 0 | 0 |

Language SafeAnswerCoverage: English **88%**, Bangla **84%**, Banglish **88%**. Strategy distribution: 181 semi-structured relation, 67 structured exact, 12 accepted GGUF generation, 25 generation-rejected status, 8 unsupported, 6 conflicting, 1 supported/other.

## 11. Generation

Forty rows initially selected a generation strategy; later deterministic relation handling resolved 19 without Qwen. **21** rows actually invoked Qwen, across **31** attempts. **12** generated answers were accepted; **9** Qwen-attempted rows were rejected. There were **5** retried rows, **0** successful retries and **5** failed retries. The broader `GENERATION_REJECTED` answerability status has **25** rows, including 16 without a Qwen attempt; it must not be presented as 25 model-generation failures.

| Language | Qwen cases | Attempts | Accepted | Rejected after attempt | Retried |
| --- | ---: | ---: | ---: | ---: | ---: |
| English | 6 | 7 | 4 | 2 | 1 |
| Bangla | 6 | 10 | 3 | 3 | 2 |
| Banglish | 9 | 14 | 5 | 4 | 2 |

## 12. Automatic answer proxies

**AUTOMATIC PROXY, NOT HUMAN ACCURACY:** normalized exact match **0.0154** (`n=260`), token precision **0.7386** (`n=260`), token recall **0.5451** (`n=260`), token F1 **0.6122** (`n=260`), numeric preservation **0.9957** (`n=235`), identifier preservation **1.000** (`n=201`; stored as `entity_preservation`). Surface-form mismatch across languages and Q007's adjudicated source conflict limit their interpretation.

## 13. Returned-answer safety

The frozen Step 8 independent unsafe-answer audit ran on every returned answer. **UnsafeReturnedAnswerCount = 0/260**; a separate raw-CSV check found the same count. This means no known gate failure among returned answers, not proof of zero factual errors or hallucinations.

## 14. Multilingual parity

Evaluation-only support pairings across 100 base questions: ALL_THREE_SUPPORTED **83**; ENGLISH_ONLY **2**; BANGLA_ONLY **0**; BANGLISH_ONLY **1**; ENGLISH_AND_BANGLA **0**; ENGLISH_AND_BANGLISH **3**; BANGLA_AND_BANGLISH **1**; NONE_SUPPORTED **10**; OTHER_MISMATCH **0**. Among 83 all-supported triplets, the same primary source/page appeared in **65**. Numeric values matched in **79/80** applicable triplets; extracted structured values matched in **57/57** applicable triplets. These narrow deterministic checks do not certify full semantic equivalence.

## 15. Latency

Mean/median/P95 per variant: **4.137/0.214/30.643 seconds**. Mean retrieval **0.560 s**, generation **3.568 s across all 300 rows**, answerability assessment **0.00081 s**. Warm inference throughput **870.1 questions/hour**; generated-question throughput **68.4/hour**. End-to-end console elapsed time was about **1,271 s** including startup. Do not extrapolate this one-PDF result as a measured 70-PDF runtime.

## 16. Memory and pagefile

Physical RAM **8,452,767,744 bytes** (~7.87 GiB); run preflight available RAM **5,075,824,640 bytes** (~4.73 GiB); minimum observed available RAM **8,298,496 bytes** (~7.9 MiB); peak process RSS **5,769,871,360 bytes** (~5.37 GiB); peak system pagefile use **3,976,228,864 bytes** (~3.70 GiB). Pagefile total was 16 GiB. GPU layers were zero; GPU-memory telemetry was not applicable. The run finished, but the very low RAM minimum is a material stability risk on this machine.

## 17. Checkpoint and resume

Five-row atomic checkpoint interval; **60** numbered batches; **300 unique IDs**, no missing or duplicate completed IDs. The completed real PILOT accepted a no-op resume under unchanged dataset/corpus/model/code/configuration and performed no additional inference. Step 10 tests additionally interrupted a simulated generated-row run and resumed it to completion without duplicates. The live PILOT was not intentionally interrupted mid-generation.

## 18. Human review sample

`human_review_sample.csv` has **30** rows: 10 English, 10 Bangla, 10 Banglish. It includes 4 structured-exact, 6 semi-structured, 3 accepted GGUF-generated, 7 generation-rejected, 5 unsupported, 4 conflicting and 1 other strategy; five sampled rows used cross-lingual fallback. All correctness, relevance, groundedness, completeness, naturalness, semantic-consistency, overall-acceptability and notes cells are blank. `human_review_completed=false`; no human accuracy claim.

## 19. Historical comparison

Step 8B's different full-generation development evaluator reported **258/300** automatically safe returned answers; this PILOT records **260/300** under the Step 9 evaluator. This is context, not an isolated improvement claim because evaluator details and project state differ. Step 9's skip-generation DEVELOPMENT run recorded **237/300**, but **must not** be compared as equivalent SafeAnswerCoverage. Its `use_generation=False` path uses fast lexical retrieval, whereas full mode uses dense/hybrid retrieval and cross-lingual fallback; even Hit@1/3 and MRR are not a like-for-like mode comparison.

## 20–27. Final benchmark handoff

The future procedure is specified in `docs/FINAL_BENCHMARK_PROTOCOL.md` and its operational checklist. It requires an approved official-source inventory with explicit version/conflict policy, recursive ingestion, duplicate/bad-PDF review, fresh dense/sparse index verification, corpus fingerprint, an independent frozen final dataset with any parallel variants linked by base ID, and source/page annotation review. Stage A uses 5–10 PDFs, Stage B 20–30, Stage C the full approved corpus. A 100–300-question final-corpus pilot must precede the roughly 6,000-question strict `FINAL_BENCHMARK`. The final run locks dataset, corpus/index, Qwen shards, code and settings; a critical change invalidates it and requires a new run ID. Human review must be stratified by language, strategy, status and intent, preferably with overlapping independent reviewers. Final thesis metrics separate retrieval, answerability, human quality, safety, efficiency and multilingual parity. The reproducibility package includes SHA-256 hashes, Git/dirty state, Python/dependencies, OS/hardware, configuration, run manifest and checkpoints.

## 28. Tests

Step 9 baseline **271**; Step 10 adds **8**; total **279**, failures **0**, skipped **1**. The suite passed after the full PILOT. Tests cover full/skip mode separation, PILOT classification, Q007 frozen labels, generated-row resume, model hashes, dataset freeze and final-run invalidation.

## 29–30. Production and Git

Production RAG changes: **NONE**. `git diff --stat` shows only the pre-existing Step 9 README change among tracked files; new Step 9/10 evaluation code, documentation and results are untracked. The working tree is dirty and no commit was made. PILOT code fingerprint `ce8c16f41d74dc069f9fe158503da47dc1e609bee0f5f5e5c0ef06279cbe0e52` still matched after tests. Git HEAD was `6c23ad85b2d9f5c0b18a898e72e568106a383989`.

## 31–35. Readiness decisions

70-PDF architecture: **YES**; actually validated at 70 PDFs: **NO**. Six-thousand-question evaluator framework: **YES**, including synthetic descriptor-scale checks; 6,000 real inferences: **NO**. Step 10 engineering-complete for the current one-PDF 300-variant scope: **YES**. Ready to move the package to a ~32-GB machine for staged validation: **YES**. This does not authorize immediate final inference or assert 32 GB will necessarily be sufficient.

## 36. Human input needed

The official ~70-PDF source inventory, source-version adjudications, independent final dataset/annotation freeze, future hardware details and genuine human-review scores remain to be supplied. Q007's conflicting source passage is documented under the user's page-4 evaluation choice; it is not silently resolved in production.

## 37. Next action

Transfer the frozen package to the better machine, verify copied hashes and dependencies, then perform Stage A of the final-corpus protocol. Do **not** begin the 70-PDF/6,000-question final benchmark automatically. Do not call SafeAnswerCoverage “accuracy.”
