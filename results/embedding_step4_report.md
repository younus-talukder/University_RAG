# Step 4 Embedding and Index Reproducibility Report

This is a system-integrity and reproducibility report, not a retrieval-accuracy evaluation.

## Selected BGE-M3 identity

- Model: `BAAI/bge-m3`
- Immutable revision: `5617a9f61b028005a4858fdac845db406aefb181`
- Loading policy: local-only by default
- Weight format: `pytorch_model.bin`
- Dimension/dtype: 1024 / float32
- Normalization: enabled and validated
- FAISS: `IndexFlatIP`
- Device: CPU
- Batch size: 16

The local cache also contains revision `9a0624b896d81da7492a910ffa53731274b6cf3d`, but it contains only `model.safetensors`. It is not selected or mixed with the pinned complete snapshot. The selected revision contains configuration, tokenizer, SentenceTransformer configuration, pooling configuration, and the complete PyTorch weights. No internet or download was required.

## Validation contracts

- Missing or incomplete snapshots fail with an embedding-specific error.
- Remote model revisions must be immutable 40-character commit hashes.
- Document and query embeddings share one validated loader/configuration.
- Empty, wrong-row-count, wrong-dimension, NaN, Inf, and non-unit normalized output is rejected.
- Chunk ordering is deterministic by relative path, page, block, and chunk ID.
- Every metadata record stores its exact `vector_row` and chunk-text SHA-256.
- Runtime semantic search rejects revision, dimension, normalization, dtype, weight-format, FAISS-type, or configuration-fingerprint mismatch.
- Staged publication still protects the previous index from model, embedding, validation, or publication failure.

## Current full build

| Measure | Result |
|---|---:|
| Documents | 1 |
| Chunks/vectors/metadata rows | 490 / 490 / 490 |
| Dimension | 1024 |
| Ingestion and chunking | 3.215 s |
| Model loading | 2.394 s |
| Embedding | 411.373 s |
| FAISS creation | 0.006 s |
| Total wall time | 417.093 s |

An immediate unchanged build reused the published index in approximately 0.01 seconds without embedding again.

## Determinism and offline checks

- Same text twice: maximum absolute vector difference `0.0`.
- Batch size 1 vs 3: maximum absolute difference `8.20e-08`.
- Observed norms in the batch comparison: 1.0 to 1.0.
- Pinned local snapshot loaded and embedded successfully with local-only configuration.
- English, Bangla, and Banglish semantic queries each returned three results with document, source, page, and chunk provenance.

The Bangla integrity query ranked page 71 first while English and Banglish ranked page 68 first. This is a retrieval-quality observation for Step 5, not a Step 4 compatibility failure.

## Embedding cache decision

A persistent chunk-vector cache was not implemented. Safe concurrent updates, corruption recovery, garbage collection, and transactional invalidation would add significant correctness complexity. The current implementation instead provides bounded configurable batching and deterministic inputs. A future cache should key on chunk text hash plus the complete embedding compatibility fingerprint, never chunk ID alone.

## Scaling estimate

The measured CPU throughput is about 1.19 chunks/second. If 70 PDFs produced roughly 34,300 chunks of similar length, a full CPU rebuild would plan around eight hours, but this is not guaranteed linear because document sizes, token lengths, hardware, and system load vary. BGE-M3 embedding is the clear bottleneck; parsing and FAISS insertion are comparatively small. At 34,300 vectors, raw float32 vector storage would be roughly 140 MB before FAISS/metadata/model overhead.

## Tests

- Previous Step 1-3 tests: 57
- New Step 4 tests: 15
- Total: 72
- Failures: 0
