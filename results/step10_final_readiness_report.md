# Step 10 final benchmark readiness

Status: **STEP 10 ENGINEERING-COMPLETE for the current development scope**. The project is ready to transfer to a better machine for staged final-corpus validation, not to start the final benchmark immediately.

## Current verified boundary

- The Step 1–8 production RAG source remains frozen; Step 10 changes are confined to evaluation tooling, tests and documentation. No commit was made.
- The local development corpus still contains one PDF and a 491-chunk index. This is not a 70-PDF validation.
- The local development dataset has 100 base questions and 300 English/Bangla/Banglish variants. This is not a 6,000-question validation.
- The two existing Qwen2.5-7B-Instruct Q4_K_M shard SHA-256 values are `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db` and `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a`.
- The Step 9 skip-generation DEVELOPMENT run was 300/300 with 0 runtime errors; its 237/300 SafeAnswerCoverage is not full-generation coverage or human accuracy.
- The Step 8B historical full-generation development run was 300/300 and reported 258/300 automatically safe returned answers. It used a different evaluator and cannot establish Step 10 PILOT completion.
- The Step 10 strict full-generation PILOT completed **300/300**, 0 runtime errors, 260/300 SafeAnswerCoverage (86.67%), 0 known unsafe returned answers, 21 Qwen-invoked cases and 31 attempts. SafeAnswerCoverage is **not accuracy**.
- The PILOT has 60 atomic five-row checkpoints and accepted a no-op resume with no duplicate inference. A generated-row interruption/resume test passed.
- Peak process RSS was about 5.37 GiB, system pagefile use peaked near 3.70 GiB and minimum free physical RAM reached about 7.9 MiB. This machine completed the pilot but has little memory safety margin.
- The 30-row PILOT review sample is balanced 10 per language and blank in all human-rating fields. No human accuracy is claimed.

## Q007 adjudication gate

The source PDF contains two conflicting statements: physical PDF page 4 says Computer Science & Engineering; page 8 says Computer Science & Technology. The user explicitly reconfirmed **page 4 — Engineering**. The active evaluation dataset was already page-4 Engineering in English, Bangla and Banglish, so its row was retained unchanged and frozen as `v1-final` (SHA-256 `8b2e6c06ffba8d4dee58fd979706f244e5891591c4e8e25d9beffea801e149b2`). The source discrepancy and decision are recorded in `docs/Q007_ANNOTATION_ADJUDICATION.md`.

The full-mode Q007-only rerun and PILOT returned page-8 Technology in all three languages. This is a recorded source/benchmark disagreement, not a production change or permission to tune the RAG. The earlier Step 9 skip-generation path used lexical retrieval; this full-generation path uses dense/hybrid retrieval, so its Q007 behavior can differ without code changes.

## Future-machine package

`docs/FINAL_BENCHMARK_PROTOCOL.md` specifies source inventory, version/conflict policy, staged ingestion, a final-corpus pilot, final run lock, human sampling, metric definitions and reproducibility. `docs/FINAL_BENCHMARK_CHECKLIST.md` records gates to verify after transfer. The run manifest contains dataset/corpus/index/code identity, both Qwen shard hashes, Python/dependency versions, OS and effective settings. The architecture is designed for a larger corpus and dataset, but those scales remain untested here.

## Current gates passed

1. Q007 adjudication is documented; frozen development dataset manifest has 100 verified base rows and 300 variants.
2. Strict sequential full-generation PILOT completed 300/300; provenance, safety audit, metrics, parity, resource telemetry and blank 30-row review sample were produced.
3. Leakage audit passed, 300 evaluation IDs are unique, no runtime errors or known unsafe returned answers occurred.
4. Full regression suite passed **279 tests**, 0 failures, 1 skip. Production code remained unchanged.

## Required work after transfer

Obtain the official source inventory and independent final dataset, resolve official-document version conflicts, record actual future hardware, verify copied hashes/dependencies, then follow the staged 5–10 PDF, 20–30 PDF and full-corpus gates. A 100–300-question final-corpus pilot must pass before the real ~6,000-question run. Human scores require genuine reviewers and the existing importer. Do **not** perform 70-PDF/6,000-question inference on the current 8-GB machine. A ~32-GB machine is a planning target, not a performance guarantee.
